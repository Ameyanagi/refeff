use super::*;

/// Port of FEFF `KSPACE/bastrans.f90` `BASTRMAT`.
///
/// The matrices convert matrix representations between real spherical
/// harmonics (`RLM`), complex spherical harmonics (`CLM`), and relativistic
/// `(kappa, mue)` states (`REL`). Storage is `ndarray` Fortran order to match
/// FEFF's column-major matrix layout.
pub fn basis_transform_matrices(lmax: usize) -> Result<BasisTransformMatrices, AngularError> {
    let (orbital_count, order) = basis_transform_dimensions(lmax)?;
    let branch_count = lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(1))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let cgc = relativistic_clebsch_gordan_coefficients(lmax)?;

    let mut complex_to_relativistic = Array2::zeros((order, order).f());
    for lnr in 0..=lmax {
        let lnr_isize =
            isize::try_from(lnr).map_err(|_| AngularError::IndexTooLarge { value: lnr })?;
        for magnetic in -lnr_isize..=lnr_isize {
            let lm = spherical_spin_index(lnr, magnetic)?;
            let mut state = 0usize;
            for branch in 1..=branch_count {
                let orbital = branch / 2;
                let jp05 = if 2 * orbital == branch {
                    orbital
                } else {
                    orbital
                        .checked_add(1)
                        .ok_or(AngularError::IndexTooLarge { value: orbital })?
                };
                let jp05_isize = isize::try_from(jp05)
                    .map_err(|_| AngularError::IndexTooLarge { value: jp05 })?;
                for muem05 in -jp05_isize..jp05_isize {
                    let muep05 = muem05 + 1;
                    if orbital == lnr {
                        if muep05 == magnetic {
                            complex_to_relativistic[(lm, state)] =
                                Complex::new(cgc.coefficients[(state, 0)], 0.0);
                        }
                        if muem05 == magnetic {
                            complex_to_relativistic[(lm + orbital_count, state)] =
                                Complex::new(cgc.coefficients[(state, 1)], 0.0);
                        }
                    }
                    state = state
                        .checked_add(1)
                        .ok_or(AngularError::IndexTooLarge { value: branch })?;
                }
            }
        }
    }

    let mut real_to_complex = Array2::zeros((order, order).f());
    let weight = 1.0 / 2.0_f64.sqrt();
    for orbital in 0..=lmax {
        let orbital_isize =
            isize::try_from(orbital).map_err(|_| AngularError::IndexTooLarge { value: orbital })?;
        for magnetic in -orbital_isize..=orbital_isize {
            let index = spherical_spin_index(orbital, magnetic)?;
            let mirror = spherical_spin_index(orbital, -magnetic)?;
            if magnetic < 0 {
                real_to_complex[(index, index)] = Complex::new(0.0, -weight);
                real_to_complex[(mirror, index)] = Complex::new(weight, 0.0);
                real_to_complex[(index + orbital_count, index + orbital_count)] =
                    Complex::new(0.0, -weight);
                real_to_complex[(mirror + orbital_count, index + orbital_count)] =
                    Complex::new(weight, 0.0);
            } else if magnetic == 0 {
                real_to_complex[(index, index)] = Complex::new(1.0, 0.0);
                real_to_complex[(index + orbital_count, index + orbital_count)] =
                    Complex::new(1.0, 0.0);
            } else {
                let sign = if magnetic % 2 == 0 { 1.0 } else { -1.0 };
                real_to_complex[(index, index)] = Complex::new(weight * sign, 0.0);
                real_to_complex[(mirror, index)] = Complex::new(0.0, weight * sign);
                real_to_complex[(index + orbital_count, index + orbital_count)] =
                    Complex::new(weight * sign, 0.0);
                real_to_complex[(mirror + orbital_count, index + orbital_count)] =
                    Complex::new(0.0, weight * sign);
            }
        }
    }

    let real_to_relativistic = fortran_complex_matrix(
        complex_matmul(real_to_complex.view(), complex_to_relativistic.view()).view(),
    );

    Ok(BasisTransformMatrices {
        lmax,
        order,
        real_to_complex,
        complex_to_relativistic,
        real_to_relativistic,
    })
}

/// Port of FEFF `KSPACE/bastrans.f90` `CHANGEREP`.
///
/// Converts a square matrix between FEFF's `RLM`, `CLM`, and `REL`
/// representations using the transform bundle returned by
/// [`basis_transform_matrices`]. The output uses Fortran-order `ndarray`
/// storage like FEFF matrix work arrays.
pub fn change_basis_representation(
    input: ArrayView2<'_, Complex>,
    mode: BasisTransformMode,
    transforms: &BasisTransformMatrices,
) -> Result<ComplexMat, AngularError> {
    validate_basis_matrix_shape("input", input, transforms.order)?;
    validate_basis_matrix_shape(
        "real_to_complex",
        transforms.real_to_complex.view(),
        transforms.order,
    )?;
    validate_basis_matrix_shape(
        "complex_to_relativistic",
        transforms.complex_to_relativistic.view(),
        transforms.order,
    )?;
    validate_basis_matrix_shape(
        "real_to_relativistic",
        transforms.real_to_relativistic.view(),
        transforms.order,
    )?;

    let (left, right, left_conjugate, right_conjugate) = match mode {
        BasisTransformMode::RelativisticToReal => (
            transforms.real_to_relativistic.view(),
            transforms.real_to_relativistic.view(),
            false,
            true,
        ),
        BasisTransformMode::RealToRelativistic => (
            transforms.real_to_relativistic.view(),
            transforms.real_to_relativistic.view(),
            true,
            false,
        ),
        BasisTransformMode::RelativisticToComplex => (
            transforms.complex_to_relativistic.view(),
            transforms.complex_to_relativistic.view(),
            false,
            true,
        ),
        BasisTransformMode::ComplexToRelativistic => (
            transforms.complex_to_relativistic.view(),
            transforms.complex_to_relativistic.view(),
            true,
            false,
        ),
        BasisTransformMode::ComplexToReal => (
            transforms.real_to_complex.view(),
            transforms.real_to_complex.view(),
            false,
            true,
        ),
        BasisTransformMode::RealToComplex => (
            transforms.real_to_complex.view(),
            transforms.real_to_complex.view(),
            true,
            false,
        ),
    };

    let left_matrix = if left_conjugate {
        conjugate_transpose(left)
    } else {
        left.to_owned()
    };
    let right_matrix = if right_conjugate {
        conjugate_transpose(right)
    } else {
        right.to_owned()
    };
    let work = complex_matmul(left_matrix.view(), input);
    Ok(fortran_complex_matrix(
        complex_matmul(work.view(), right_matrix.view()).view(),
    ))
}

fn basis_transform_dimensions(lmax: usize) -> Result<(usize, usize), AngularError> {
    let orbital_count = lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let order = orbital_count
        .checked_mul(2)
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    Ok((orbital_count, order))
}

fn spherical_spin_index(orbital: usize, magnetic: isize) -> Result<usize, AngularError> {
    let orbital_isize =
        isize::try_from(orbital).map_err(|_| AngularError::IndexTooLarge { value: orbital })?;
    if magnetic < -orbital_isize || magnetic > orbital_isize {
        return Err(AngularError::MagneticIndexOutOfRange {
            magnetic,
            lmax: orbital,
        });
    }
    let base = orbital
        .checked_mul(
            orbital
                .checked_add(1)
                .ok_or(AngularError::IndexTooLarge { value: orbital })?,
        )
        .ok_or(AngularError::IndexTooLarge { value: orbital })?;
    let shifted = isize::try_from(base)
        .map_err(|_| AngularError::IndexTooLarge { value: orbital })?
        + magnetic;
    usize::try_from(shifted).map_err(|_| AngularError::MagneticIndexOutOfRange {
        magnetic,
        lmax: orbital,
    })
}

fn validate_basis_matrix_shape(
    name: &'static str,
    matrix: ArrayView2<'_, Complex>,
    expected: usize,
) -> Result<(), AngularError> {
    if matrix.shape() != [expected, expected] {
        return Err(AngularError::InvalidBasisTransformShape {
            name,
            rows: matrix.nrows(),
            columns: matrix.ncols(),
            expected,
        });
    }
    Ok(())
}

fn conjugate_transpose(matrix: ArrayView2<'_, Complex>) -> ComplexMat {
    Array2::from_shape_fn((matrix.ncols(), matrix.nrows()).f(), |(row, column)| {
        matrix[(column, row)].conj()
    })
}

fn fortran_complex_matrix(matrix: ArrayView2<'_, Complex>) -> ComplexMat {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()).f(), |(row, column)| {
        matrix[(row, column)]
    })
}
