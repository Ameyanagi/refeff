use crate::Real;
use ndarray::{Array2, ArrayView2, ShapeBuilder};

use super::{
    AtomMathError, AtomicOverlapAmplitudeReductionInput, abs_kappa_i32, validate_finite_scalar,
    validate_finite_slice, validate_orbital_table_len,
};

/// Port of FEFF `ATOM/s02at.f90`, the relaxed-overlap amplitude reduction.
///
/// The overlap matrix is consumed in FEFF group order by kappa, using only the
/// upper triangular entries that FEFF copies into its symmetric work matrix.
pub fn atomic_overlap_amplitude_reduction(
    input: AtomicOverlapAmplitudeReductionInput<'_>,
) -> Result<Real, AtomMathError> {
    const KAPPA_MIN: i32 = -5;
    const KAPPA_MAX: i32 = 4;
    const GROUP_LIMIT: usize = 8;

    validate_overlap_amplitude_input(&input)?;
    let hole_zero_based = input.hole_orbital_1based.map(|index| index - 1);
    let mut amplitude = 1.0;

    for kappa in KAPPA_MIN..=KAPPA_MAX {
        if kappa == 0 {
            continue;
        }
        let group = input
            .kappas
            .iter()
            .enumerate()
            .filter_map(|(orbital, &orbital_kappa)| (orbital_kappa == kappa).then_some(orbital))
            .collect::<Vec<_>>();
        if group.is_empty() {
            continue;
        }
        if group.len() > GROUP_LIMIT {
            return Err(AtomMathError::KappaGroupTooLarge {
                kappa,
                count: group.len(),
                limit: GROUP_LIMIT,
            });
        }

        let mut matrix = s02at_overlap_matrix(&group, input.overlap_integrals);
        let determinant_all = s02at_squared_determinant_in_place(&mut matrix, group.len());
        let determinant_without_last =
            s02at_squared_determinant_in_place(&mut matrix, group.len() - 1);
        let last_orbital = *group.last().ok_or(AtomMathError::KappaGroupTooLarge {
            kappa,
            count: 0,
            limit: GROUP_LIMIT,
        })?;
        let occupation = input.occupations[last_orbital];
        let max_occupation = 2.0 * Real::from(kappa.unsigned_abs());
        let hole_vacancy = max_occupation - occupation;
        let hole_position = hole_zero_based.and_then(|hole| {
            group
                .iter()
                .position(|&orbital| orbital == hole)
                .map(|position| position + 1)
        });

        amplitude *= match hole_position {
            None => determinant_all.powf(occupation) * determinant_without_last.powf(hole_vacancy),
            Some(position) if position == group.len() => {
                determinant_all.powf(occupation - 1.0)
                    * determinant_without_last.powf(hole_vacancy + 1.0)
            }
            Some(position) => {
                let mut eliminated = s02at_eliminate_hole(matrix.view(), position - 1);
                let determinant_eliminated_all =
                    s02at_squared_determinant_in_place(&mut eliminated, group.len());
                let determinant_eliminated_without_last =
                    s02at_squared_determinant_in_place(&mut eliminated, group.len() - 1);
                let mixed = (determinant_eliminated_without_last * determinant_all * hole_vacancy
                    + determinant_eliminated_all * determinant_without_last * occupation)
                    / max_occupation;
                mixed
                    * determinant_all.powf(occupation - 1.0)
                    * determinant_without_last.powf(hole_vacancy - 1.0)
            }
        };
        validate_finite_scalar("s02", amplitude)?;
    }

    Ok(amplitude)
}

fn s02at_overlap_matrix(group: &[usize], overlaps: ArrayView2<'_, Real>) -> Array2<Real> {
    let order = group.len();
    let mut matrix = Array2::zeros((order, order).f());
    for column in 0..order {
        for row in 0..=column {
            let value = overlaps[(group[row], group[column])];
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
    matrix
}

fn s02at_eliminate_hole(matrix: ArrayView2<'_, Real>, hole: usize) -> Array2<Real> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()).f(), |(row, column)| {
        if row == hole && column == hole {
            1.0
        } else if row == hole || column == hole {
            0.0
        } else {
            matrix[(row, column)]
        }
    })
}

fn s02at_squared_determinant_in_place(matrix: &mut Array2<Real>, order: usize) -> Real {
    let determinant = s02at_determinant_in_place(matrix, order);
    determinant * determinant
}

fn s02at_determinant_in_place(matrix: &mut Array2<Real>, order: usize) -> Real {
    let mut determinant = 1.0;
    for pivot in 0..order {
        if matrix[(pivot, pivot)] == 0.0 {
            let Some(swap_column) = (pivot..order).find(|&column| matrix[(pivot, column)] != 0.0)
            else {
                return 0.0;
            };
            for row in pivot..order {
                let saved = matrix[(row, swap_column)];
                matrix[(row, swap_column)] = matrix[(row, pivot)];
                matrix[(row, pivot)] = saved;
            }
            determinant = -determinant;
        }

        determinant *= matrix[(pivot, pivot)];
        if pivot + 1 < order {
            for row in (pivot + 1)..order {
                for column in (pivot + 1)..order {
                    matrix[(row, column)] -=
                        matrix[(row, pivot)] * matrix[(pivot, column)] / matrix[(pivot, pivot)];
                }
            }
        }
    }
    determinant
}

fn validate_overlap_amplitude_input(
    input: &AtomicOverlapAmplitudeReductionInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    if let Some(hole_orbital_1based) = input.hole_orbital_1based
        && !(1..=orbital_count).contains(&hole_orbital_1based)
    {
        return Err(AtomMathError::HoleOrbitalOutOfRange {
            hole_orbital_1based,
            orbital_count,
        });
    }
    let [rows, columns] = input.overlap_integrals.shape().try_into().map_err(|_| {
        AtomMathError::OverlapMatrixShape {
            expected: orbital_count,
            rows: input.overlap_integrals.nrows(),
            columns: input.overlap_integrals.ncols(),
        }
    })?;
    if rows != orbital_count || columns != orbital_count {
        return Err(AtomMathError::OverlapMatrixShape {
            expected: orbital_count,
            rows,
            columns,
        });
    }
    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_slice("occupation", input.occupations)?;
    for &value in &input.overlap_integrals {
        validate_finite_scalar("overlap_integral", value)?;
    }
    Ok(())
}
