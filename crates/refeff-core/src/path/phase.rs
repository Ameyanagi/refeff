//! FEFF PATH phase-criteria table generation.

use ndarray::{Array3, ShapeBuilder};

use crate::{Complex, Real, angular::legendre_polynomials_into};

use super::{PathError, PathPhaseCriteriaInput, PathPhaseCriteriaTables};

const PATH_BETA_HALF_WIDTH: i32 = 40;
const PATH_BETA_GRID_SIZE: usize = (2 * PATH_BETA_HALF_WIDTH as usize) + 1;
const PATH_CRITICAL_ENERGY_OFFSETS: [usize; 9] = [0, 5, 10, 15, 20, 30, 34, 38, 40];
const BOHR_ANGSTROM: Real = 0.529_177_249;
const MEAN_FREE_PATH_IMAG_EPSILON: Real = 1.0e-16;

/// Port of FEFF `prcrit` computational core.
///
/// The original routine reads `phase.bin`; this pure function receives the
/// already-decoded energy grid, reference energies, phase shifts, and `lmax`
/// table. It preserves FEFF's single-precision PATH handoff by casting
/// `cksp`, `xlam`, `fbeta`, and critical-table values through `f32` before
/// storing them in Rust `f64` containers.
pub fn path_phase_criteria_tables(
    input: PathPhaseCriteriaInput<'_>,
) -> Result<PathPhaseCriteriaTables, PathError> {
    validate_phase_criteria_input(input)?;

    let energy_count = input.energies.len();
    let (_, angular_channels, potential_count) = input.phase_shifts.dim();
    let mut wave_numbers = Vec::with_capacity(energy_count);
    let mut mean_free_paths = Vec::with_capacity(energy_count);

    for energy in 0..energy_count {
        let energy_value = finite_phase_complex("energy", energy, 0, 0, input.energies[energy])?;
        let reference = finite_phase_complex(
            "reference energy",
            energy,
            0,
            0,
            input.reference_energies[energy],
        )?;
        let wave = (Complex::new(2.0, 0.0) * (energy_value - reference)).sqrt();
        let wave_number = single_precision_path_value(wave.re / BOHR_ANGSTROM);
        if !wave_number.is_finite() {
            return Err(PathError::NonFinitePathPhaseScalar {
                quantity: "wave number",
                energy,
                value: wave_number,
            });
        }

        let mut mean_free_path = single_precision_path_value(1.0e10);
        if wave.im.abs() > MEAN_FREE_PATH_IMAG_EPSILON {
            mean_free_path = single_precision_path_value(1.0 / wave.im);
        }
        mean_free_path = single_precision_path_value(mean_free_path * BOHR_ANGSTROM);
        if !mean_free_path.is_finite() {
            return Err(PathError::NonFinitePathPhaseScalar {
                quantity: "mean free path",
                energy,
                value: mean_free_path,
            });
        }

        wave_numbers.push(wave_number);
        mean_free_paths.push(mean_free_path);
    }

    let mut fbeta = Array3::zeros((PATH_BETA_GRID_SIZE, potential_count, energy_count).f());
    let mut legendre = vec![0.0; angular_channels];
    let two_i = Complex::new(0.0, 2.0);
    for beta_row in 0..PATH_BETA_GRID_SIZE {
        let beta_index = beta_row as i32 - PATH_BETA_HALF_WIDTH;
        let mut cosine = 0.025 * Real::from(beta_index);
        if beta_index == -PATH_BETA_HALF_WIDTH {
            cosine = -1.0;
        } else if beta_index == PATH_BETA_HALF_WIDTH {
            cosine = 1.0;
        }
        legendre_polynomials_into(cosine, &mut legendre);

        for potential in 0..potential_count {
            for energy in 0..energy_count {
                let angular_limit = input.angular_limits[(energy, potential)];
                let mut amplitude = Complex::new(0.0, 0.0);
                for (angular, &legendre_value) in
                    legendre.iter().enumerate().take(angular_limit + 1)
                {
                    let phase = finite_phase_complex(
                        "phase shift",
                        energy,
                        angular,
                        potential,
                        input.phase_shifts[(energy, angular, potential)],
                    )?;
                    let t_matrix = ((two_i * phase).exp() - Complex::new(1.0, 0.0)) / two_i;
                    amplitude += t_matrix * legendre_value * (2 * angular + 1) as Real;
                }
                let value = single_precision_path_value(amplitude.norm());
                if !value.is_finite() {
                    return Err(PathError::NonFinitePathPhaseScalar {
                        quantity: "fbeta",
                        energy,
                        value,
                    });
                }
                fbeta[(beta_row, potential, energy)] = value;
            }
        }
    }

    let critical_energy_indices =
        critical_phase_energy_indices(input.zero_wave_energy_index, energy_count);
    if critical_energy_indices.is_empty() {
        return Err(PathError::NoPathPhaseCriticalEnergies {
            index: input.zero_wave_energy_index,
        });
    }

    let mut critical_wave_numbers = Vec::with_capacity(critical_energy_indices.len());
    let mut critical_mean_free_paths = Vec::with_capacity(critical_energy_indices.len());
    let mut fbeta_critical = Array3::zeros(
        (
            PATH_BETA_GRID_SIZE,
            potential_count,
            critical_energy_indices.len(),
        )
            .f(),
    );
    for (criterion, &energy) in critical_energy_indices.iter().enumerate() {
        critical_wave_numbers.push(wave_numbers[energy]);
        critical_mean_free_paths.push(mean_free_paths[energy]);
        for beta_row in 0..PATH_BETA_GRID_SIZE {
            for potential in 0..potential_count {
                fbeta_critical[(beta_row, potential, criterion)] =
                    fbeta[(beta_row, potential, energy)];
            }
        }
    }

    Ok(PathPhaseCriteriaTables {
        output_energy_count: input.output_energy_count,
        zero_wave_energy_index: input.zero_wave_energy_index,
        wave_numbers,
        mean_free_paths,
        fbeta,
        critical_energy_indices,
        critical_wave_numbers,
        critical_mean_free_paths,
        fbeta_critical,
    })
}

fn validate_phase_criteria_input(input: PathPhaseCriteriaInput<'_>) -> Result<(), PathError> {
    let energy_count = input.energies.len();
    let (phase_energies, phase_angular, phase_potentials) = input.phase_shifts.dim();
    let (limit_energies, limit_potentials) = input.angular_limits.dim();
    if energy_count == 0
        || input.reference_energies.len() != energy_count
        || phase_energies != energy_count
        || phase_angular == 0
        || phase_potentials == 0
        || limit_energies != energy_count
        || limit_potentials != phase_potentials
    {
        return Err(PathError::InvalidPathPhaseCriteriaShape {
            energies: energy_count,
            references: input.reference_energies.len(),
            phase_energies,
            phase_angular,
            phase_potentials,
            limit_energies,
            limit_potentials,
        });
    }

    if input.output_energy_count == 0 || input.output_energy_count > energy_count {
        return Err(PathError::PathPhaseOutputEnergyOutOfRange {
            output_energy_count: input.output_energy_count,
            energies: energy_count,
        });
    }
    if input.zero_wave_energy_index >= energy_count {
        return Err(PathError::PathPhaseZeroWaveIndexOutOfRange {
            index: input.zero_wave_energy_index,
            energies: energy_count,
        });
    }

    for energy in 0..energy_count {
        finite_phase_complex("energy", energy, 0, 0, input.energies[energy])?;
        finite_phase_complex(
            "reference energy",
            energy,
            0,
            0,
            input.reference_energies[energy],
        )?;
        for potential in 0..phase_potentials {
            let angular_limit = input.angular_limits[(energy, potential)];
            if angular_limit >= phase_angular {
                return Err(PathError::PathPhaseAngularLimitOutOfRange {
                    energy,
                    potential,
                    angular_limit,
                    angular_channels: phase_angular,
                });
            }
        }
    }

    Ok(())
}

fn finite_phase_complex(
    quantity: &'static str,
    energy: usize,
    angular: usize,
    potential: usize,
    value: Complex,
) -> Result<Complex, PathError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(PathError::NonFinitePathPhaseComplex {
            quantity,
            energy,
            angular,
            potential,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(crate) fn single_precision_path_value(value: Real) -> Real {
    Real::from(value as f32)
}

fn critical_phase_energy_indices(start: usize, energy_count: usize) -> Vec<usize> {
    PATH_CRITICAL_ENERGY_OFFSETS
        .iter()
        .filter_map(|&offset| start.checked_add(offset))
        .take_while(|&index| index < energy_count)
        .collect()
}
