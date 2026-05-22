use super::*;

pub(super) fn potential_count_from_highest(index: usize) -> Result<usize, DensityError> {
    index.checked_add(1).ok_or(DensityError::InvalidIndex {
        name: "highest_potential_index",
        index,
    })
}
pub(super) fn table_column_prefix(values: ArrayView2<'_, Real>, column: usize) -> Array1<Real> {
    (0..OVRLP_DENSITY_POINTS)
        .map(|row| values[(row, column)])
        .collect::<Array1<_>>()
}

pub(super) fn includes_angular_channel(angular: usize, include_high_l: bool) -> bool {
    angular <= 2 || include_high_l
}

pub(super) fn widen_complex32(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}
pub(super) fn ensure_len(
    name: &'static str,
    actual: usize,
    required: usize,
) -> Result<(), DensityError> {
    if actual >= required {
        Ok(())
    } else {
        Err(DensityError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}

pub(super) fn ensure_length_match(
    left_name: &'static str,
    left_len: usize,
    right_name: &'static str,
    right_len: usize,
) -> Result<(), DensityError> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(DensityError::LengthMismatch {
            left_name,
            left_len,
            right_name,
            right_len,
        })
    }
}

pub(super) fn ensure_shape(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
) -> Result<(), DensityError> {
    let rows = shape[0];
    let columns = shape[1];
    if rows >= required_rows && columns >= required_columns {
        Ok(())
    } else {
        Err(DensityError::ShapeTooSmall {
            name,
            rows,
            columns,
            required_rows,
            required_columns,
        })
    }
}

pub(super) fn ensure_shape3(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
    required_depth: usize,
) -> Result<(), DensityError> {
    let rows = shape[0];
    let columns = shape[1];
    let depth = shape[2];
    if rows >= required_rows && columns >= required_columns && depth >= required_depth {
        Ok(())
    } else {
        Err(DensityError::CubeShapeTooSmall {
            name,
            rows,
            columns,
            depth,
            required_rows,
            required_columns,
            required_depth,
        })
    }
}

pub(super) fn validate_real_scalar(name: &'static str, value: Real) -> Result<(), DensityError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DensityError::NonFiniteScalar { name, value })
    }
}

pub(super) fn validate_positive_real_scalar(
    name: &'static str,
    value: Real,
) -> Result<(), DensityError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(DensityError::NonPositiveScalar { name, value })
    }
}

pub(super) fn validate_nonzero_real_scalar(
    name: &'static str,
    value: Real,
) -> Result<(), DensityError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(DensityError::ZeroScalar { name, value })
    }
}

pub(super) fn validate_complex_scalar(
    name: &'static str,
    value: Complex,
) -> Result<(), DensityError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(DensityError::NonFiniteComplex {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn validate_real_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DensityError::NonFiniteValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_positive_real_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), DensityError> {
    for &value in values {
        validate_positive_real_scalar(name, value)?;
    }
    Ok(())
}

pub(super) fn validate_real_table_values(
    name: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DensityError::NonFiniteValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_real_cube_values(
    name: &'static str,
    values: &Array3<Real>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DensityError::NonFiniteValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_position_table(positions: ArrayView2<'_, Real>) -> Result<(), DensityError> {
    if positions.ncols() != 3 {
        return Err(DensityError::InvalidPositionShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    validate_real_table_values("atom_positions", positions)
}

pub(super) fn validate_usize_potential_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
    available: usize,
) -> Result<(), DensityError> {
    for &index in values {
        if index >= available {
            return Err(DensityError::InvalidPotentialIndex {
                name,
                index,
                available,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_usize_positive_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
) -> Result<(), DensityError> {
    for &index in values {
        if index == 0 {
            return Err(DensityError::InvalidIndex { name, index });
        }
    }
    Ok(())
}

pub(super) fn validate_usize_radial_indices(
    name: &'static str,
    values: ArrayView1<'_, usize>,
) -> Result<(), DensityError> {
    for &index in values {
        if index == 0 || index > OVRLP_DENSITY_POINTS {
            return Err(DensityError::InvalidIndex { name, index });
        }
    }
    Ok(())
}

pub(super) fn validate_complex32_values(
    name: &'static str,
    values: ArrayView1<'_, Complex32>,
) -> Result<(), DensityError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re as Real,
                imaginary: value.im as Real,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_complex_values(
    name: &'static str,
    values: impl Iterator<Item = Complex>,
) -> Result<(), DensityError> {
    for (index, value) in values.enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(DensityError::NonFiniteComplexValue {
                name,
                index,
                real: value.re,
                imaginary: value.im,
            });
        }
    }
    Ok(())
}
