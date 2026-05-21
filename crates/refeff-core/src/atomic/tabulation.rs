use crate::Real;
use ndarray::Array1;

use super::{
    ATOM_TABRAT_HARTREE_EV, ATOM_TABRAT_LABELS, ATOM_TABRAT_MOMENT_POWERS, AtomMathError,
    AtomicLagrangeParametersInput, AtomicRadialIntegralRequest, AtomicTabulatedMoment,
    AtomicTabulatedOrbital, AtomicTabulatedOverlap, AtomicTabulation, AtomicTabulationInput,
    AtomicTabulationIntegralRequest, abs_kappa_i32, direct_coulomb_coefficient_at,
    doubled_j_from_kappa, doubled_j_usize_from_kappa, exchange_coulomb_coefficient_at,
    orbital_pair_count, packed_orbital_pair_index, validate_coefficient_table,
    validate_finite_scalar, validate_finite_slice, validate_orbital_table_len,
};

/// Port of FEFF `ATOM/lagdat.f90`, non-diagonal Lagrange parameters.
///
/// The returned vector uses FEFF's packed triangular pair order. For zero-based
/// orbitals `i < j`, the packed index is `i + j * (j - 1) / 2`.
pub fn atomic_lagrange_parameters<F>(
    input: AtomicLagrangeParametersInput<'_>,
    radial_integral: F,
) -> Result<Array1<Real>, AtomMathError>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_lagrange_parameters_input(&input)?;
    AtomicLagrangeContext {
        input,
        radial_integral,
    }
    .calculate()
}

/// Port of FEFF `ATOM/tabrat.f90`, excluding text emission.
///
/// The returned data mirrors the orbital moment rows and same-kappa overlap
/// rows that FEFF writes to the ATOM log. Radial integrals are supplied as a
/// callback so callers can plug in either the Rust `dsordf` port or a test
/// oracle while keeping `tabrat`'s bookkeeping explicit.
pub fn atomic_tabulation<F>(
    input: AtomicTabulationInput<'_>,
    radial_integral: F,
) -> Result<AtomicTabulation, AtomMathError>
where
    F: FnMut(AtomicTabulationIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_tabulation_input(&input)?;
    AtomicTabulationContext {
        input,
        radial_integral,
    }
    .calculate()
}

struct AtomicLagrangeContext<'a, F> {
    input: AtomicLagrangeParametersInput<'a>,
    radial_integral: F,
}

struct AtomicTabulationContext<'a, F> {
    input: AtomicTabulationInput<'a>,
    radial_integral: F,
}

impl<F> AtomicLagrangeContext<'_, F>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<Array1<Real>, AtomMathError> {
        let orbital_count = self.input.kappas.len();
        let pair_count = orbital_pair_count(orbital_count)?;
        let mut parameters = Array1::<Real>::zeros(pair_count);

        if let Some(active_orbital_1based) = self.input.active_orbital_1based {
            let active = active_orbital_1based - 1;
            for other in 0..orbital_count {
                self.accumulate_pair(active, other, &mut parameters)?;
            }
        } else {
            for first in 0..orbital_count.saturating_sub(1) {
                for second in (first + 1)..orbital_count {
                    self.accumulate_pair(first, second, &mut parameters)?;
                }
            }
        }

        Ok(parameters)
    }

    fn accumulate_pair(
        &mut self,
        first: usize,
        second: usize,
        parameters: &mut Array1<Real>,
    ) -> Result<(), AtomMathError> {
        if first == second || self.input.kappas[first] != self.input.kappas[second] {
            return Ok(());
        }
        if self.input.shell_markers[first] < 0 && self.input.shell_markers[second] < 0 {
            return Ok(());
        }
        if self.input.occupations[first] == self.input.occupations[second] {
            return Ok(());
        }
        self.validate_pair_occupation(first)?;
        self.validate_pair_occupation(second)?;

        let mut value = self.direct_terms(first, second)?;
        if self.input.include_exchange {
            value += self.exchange_terms(first, second)?;
        }
        let packed = packed_orbital_pair_index(first, second)?;
        let parameter = value / (self.input.occupations[second] - self.input.occupations[first]);
        validate_finite_scalar("lagrange_parameter", parameter)?;
        let Some(slot) = parameters.get_mut(packed) else {
            return Err(AtomMathError::OrbitalPairTableTooLarge {
                orbital_count: self.input.kappas.len(),
            });
        };
        *slot = parameter;
        Ok(())
    }

    fn direct_terms(&mut self, first: usize, second: usize) -> Result<Real, AtomMathError> {
        let first_j2 = self.j2(first)?;
        let mut value = 0.0;
        for orbital in 0..self.input.kappas.len() {
            let orbital_j2 = self.j2(orbital)?;
            let max_rank = first_j2.min(orbital_j2);
            let mut rank = 0;
            while rank <= max_rank {
                let first_coefficient =
                    self.direct_coefficient(orbital, first, rank)? / self.input.occupations[first];
                let difference = first_coefficient
                    - self.direct_coefficient(orbital, second, rank)?
                        / self.input.occupations[second];
                if significant_relative_difference(difference, first_coefficient) {
                    value += difference
                        * self.radial(orbital + 1, orbital + 1, first + 1, second + 1, rank)?;
                }
                rank = rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
        }
        validate_finite_scalar("lagrange_direct_terms", value)?;
        Ok(value)
    }

    fn exchange_terms(&mut self, first: usize, second: usize) -> Result<Real, AtomMathError> {
        let first_j2 = self.j2(first)?;
        let mut value = 0.0;
        for orbital in 0..self.input.kappas.len() {
            let orbital_j2 = self.j2(orbital)?;
            let max_rank = first_j2
                .checked_add(orbital_j2)
                .ok_or(AtomMathError::CoulombRankOutOfRange { rank: first_j2 })?
                / 2;
            let mut rank = orbital_j2.abs_diff(max_rank);
            if self.input.kappas[first].signum() != self.input.kappas[orbital].signum() {
                rank = rank
                    .checked_add(1)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
            while rank <= max_rank {
                let first_coefficient = self.exchange_coefficient(orbital, second, rank)?
                    / self.input.occupations[second];
                let difference = first_coefficient
                    - self.exchange_coefficient(orbital, first, rank)?
                        / self.input.occupations[first];
                if significant_relative_difference(difference, first_coefficient) {
                    value += difference
                        * self.radial(first + 1, orbital + 1, second + 1, orbital + 1, rank)?;
                }
                rank = rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
        }
        validate_finite_scalar("lagrange_exchange_terms", value)?;
        Ok(value)
    }

    fn direct_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        direct_coulomb_coefficient_at(self.input.coulomb_coefficients, left, right, rank)
    }

    fn exchange_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        exchange_coulomb_coefficient_at(self.input.coulomb_coefficients, left, right, rank)
    }

    fn radial(
        &mut self,
        first_left: usize,
        first_right: usize,
        second_left: usize,
        second_right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let value = (self.radial_integral)(AtomicRadialIntegralRequest {
            first_left,
            first_right,
            second_left,
            second_right,
            rank,
        })?;
        validate_finite_scalar("radial_integral", value)?;
        Ok(value)
    }

    fn j2(&self, orbital: usize) -> Result<usize, AtomMathError> {
        doubled_j_usize_from_kappa(self.input.kappas[orbital])
    }

    fn validate_pair_occupation(&self, orbital: usize) -> Result<(), AtomMathError> {
        let occupation = self.input.occupations[orbital];
        if occupation > 0.0 {
            Ok(())
        } else {
            Err(AtomMathError::NonPositiveOccupation {
                context: "lagdat",
                orbital_1based: orbital + 1,
                occupation,
            })
        }
    }
}

impl<F> AtomicTabulationContext<'_, F>
where
    F: FnMut(AtomicTabulationIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicTabulation, AtomMathError> {
        let orbital_count = self.input.kappas.len();
        let mut orbitals = Vec::with_capacity(orbital_count);
        for orbital in 0..orbital_count {
            let orbital_label = atom_tabrat_orbital_label(self.input.kappas[orbital])?;
            let moments = self.orbital_moments(orbital)?;
            let binding_energy_ev = -self.input.orbital_energies[orbital] * ATOM_TABRAT_HARTREE_EV;
            validate_finite_scalar("tabrat_binding_energy_ev", binding_energy_ev)?;
            orbitals.push(AtomicTabulatedOrbital {
                principal_quantum_number: self.input.principal_quantum_numbers[orbital],
                orbital_label,
                occupation: self.input.occupations[orbital],
                binding_energy_ev,
                moments,
            });
        }

        let mut overlaps = Vec::new();
        for left in 0..orbital_count.saturating_sub(1) {
            for right in (left + 1)..orbital_count {
                if self.input.kappas[left] != self.input.kappas[right] {
                    continue;
                }
                let value = self.radial(left, right, 0)?;
                overlaps.push(AtomicTabulatedOverlap {
                    left,
                    right,
                    left_principal_quantum_number: self.input.principal_quantum_numbers[left],
                    left_orbital_label: atom_tabrat_orbital_label(self.input.kappas[left])?,
                    right_principal_quantum_number: self.input.principal_quantum_numbers[right],
                    right_orbital_label: atom_tabrat_orbital_label(self.input.kappas[right])?,
                    value,
                });
            }
        }

        Ok(AtomicTabulation { orbitals, overlaps })
    }

    fn orbital_moments(
        &mut self,
        orbital: usize,
    ) -> Result<Vec<AtomicTabulatedMoment>, AtomMathError> {
        let moment_count = if abs_kappa_i32(self.input.kappas[orbital])? - 1 <= 0 {
            ATOM_TABRAT_MOMENT_POWERS.len() - 1
        } else {
            ATOM_TABRAT_MOMENT_POWERS.len()
        };
        let mut moments = Vec::with_capacity(moment_count);
        for &power in ATOM_TABRAT_MOMENT_POWERS.iter().take(moment_count) {
            moments.push(AtomicTabulatedMoment {
                power,
                value: self.radial(orbital, orbital, power)?,
            });
        }
        Ok(moments)
    }

    fn radial(&mut self, left: usize, right: usize, power: i32) -> Result<Real, AtomMathError> {
        let value = (self.radial_integral)(AtomicTabulationIntegralRequest { left, right, power })?;
        validate_finite_scalar("tabrat_integral", value)?;
        Ok(value)
    }
}

fn significant_relative_difference(difference: Real, reference: Real) -> bool {
    let relative = if reference == 0.0 {
        difference
    } else {
        difference / reference
    };
    relative.abs() >= 1.0e-7
}

fn atom_tabrat_orbital_label(kappa: i32) -> Result<&'static str, AtomMathError> {
    abs_kappa_i32(kappa)?;
    let title_index = if kappa > 0 {
        kappa.checked_mul(2)
    } else {
        kappa.checked_mul(-2).and_then(|value| value.checked_sub(1))
    }
    .ok_or(AtomMathError::OrbitalLabelKappaOutOfRange { kappa })?;
    let label_index = usize::try_from(title_index - 1)
        .map_err(|_| AtomMathError::OrbitalLabelKappaOutOfRange { kappa })?;
    ATOM_TABRAT_LABELS
        .get(label_index)
        .copied()
        .ok_or(AtomMathError::OrbitalLabelKappaOutOfRange { kappa })
}

fn validate_lagrange_parameters_input(
    input: &AtomicLagrangeParametersInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len("shell_markers", orbital_count, input.shell_markers.len())?;
    if let Some(active_orbital_1based) = input.active_orbital_1based
        && !(1..=orbital_count).contains(&active_orbital_1based)
    {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based,
            orbital_count,
        });
    }
    validate_finite_slice("occupation", input.occupations)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_coefficient_table(
        input.coulomb_coefficients,
        orbital_count - 1,
        orbital_count - 1,
        0,
    )?;
    orbital_pair_count(orbital_count)?;
    Ok(())
}

fn validate_tabulation_input(input: &AtomicTabulationInput<'_>) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len(
        "principal_quantum_numbers",
        orbital_count,
        input.principal_quantum_numbers.len(),
    )?;
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    for (orbital, &principal_quantum_number) in input.principal_quantum_numbers.iter().enumerate() {
        if principal_quantum_number == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number,
            });
        }
    }
    for &kappa in input.kappas {
        atom_tabrat_orbital_label(kappa)?;
    }
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    Ok(())
}
