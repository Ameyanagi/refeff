use super::radial::loucks_radius;
use super::*;

pub(super) fn validate_delta(delta: Real) -> Result<(), GridError> {
    if delta.is_finite() && delta > 0.0 {
        Ok(())
    } else {
        Err(GridError::InvalidDelta { delta })
    }
}

pub(super) fn validate_positive_grid_length(
    name: &'static str,
    len: usize,
) -> Result<(), GridError> {
    if len > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridLength { name })
    }
}

pub(super) fn ensure_len(
    name: &'static str,
    actual: usize,
    required: usize,
) -> Result<(), GridError> {
    if actual >= required {
        Ok(())
    } else {
        Err(GridError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}

pub(super) fn ensure_shape(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
) -> Result<(), GridError> {
    let rows = shape[0];
    let columns = shape[1];
    if rows >= required_rows && columns >= required_columns {
        Ok(())
    } else {
        Err(GridError::ShapeTooSmall {
            name,
            rows,
            columns,
            required_rows,
            required_columns,
        })
    }
}

pub(super) fn validate_grid_index(name: &'static str, index: usize) -> Result<(), GridError> {
    if index > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridIndex { name, index })
    }
}

pub(super) fn validate_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
}

pub(super) fn validate_positive_finite_scalar(
    name: &'static str,
    value: Real,
) -> Result<(), GridError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(GridError::NonPositiveScalar { name, value })
    }
}

pub(super) fn validate_nonzero_finite_scalar(
    name: &'static str,
    value: Real,
) -> Result<(), GridError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(GridError::ZeroScalar { name, value })
    }
}

pub(super) fn validate_real_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_real_table(
    name: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), GridError> {
    let columns = values.ncols();
    for ((row, column), &value) in values.indexed_iter() {
        if !value.is_finite() {
            let index = row
                .checked_mul(columns)
                .and_then(|value| value.checked_add(column))
                .ok_or(GridError::GridLengthOverflow { name })?;
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_component_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_position_table(positions: ArrayView2<'_, Real>) -> Result<(), GridError> {
    if positions.ncols() != 3 {
        return Err(GridError::InvalidPositionShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    ensure_shape("atom_positions", positions.shape(), positions.nrows(), 3)?;
    for ((atom_index, axis), &value) in positions.indexed_iter() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue {
                name: "atom_positions",
                index: atom_index * 3 + axis,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_usize_potential_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
    available: usize,
) -> Result<(), GridError> {
    for &index in values {
        if index >= available {
            return Err(GridError::InvalidPotentialIndex {
                name,
                index,
                available,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_component_prefix_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().take(active_len).enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_slice_values(name: &'static str, values: &[Real]) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

pub(super) fn validate_positive_radii(
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), GridError> {
    for &radius in values.iter().take(active_len) {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(GridError::InvalidRadius { radius });
        }
    }
    Ok(())
}

pub(super) fn validate_source_len_at_least(
    name: &'static str,
    available: usize,
    required: usize,
) -> Result<(), GridError> {
    if available >= required {
        Ok(())
    } else {
        Err(GridError::SourceGridTooShort {
            name,
            required,
            available,
        })
    }
}

pub(super) fn ensure_source_length(
    name: &'static str,
    required: usize,
    available: usize,
) -> Result<(), GridError> {
    if available >= required {
        Ok(())
    } else {
        Err(GridError::SourceGridTooShort {
            name,
            required,
            available,
        })
    }
}

pub(super) fn checked_index_offset(
    name: &'static str,
    index: usize,
    offset: usize,
) -> Result<usize, GridError> {
    index
        .checked_add(offset)
        .ok_or(GridError::GridLengthOverflow { name })
}

pub(super) fn square_index_as_real(name: &'static str, index: usize) -> Result<Real, GridError> {
    index
        .checked_mul(index)
        .map(|value| value as Real)
        .ok_or(GridError::GridLengthOverflow { name })
}

pub(super) fn radius_cubed_grid_value(
    values: ArrayView1<'_, Real>,
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    Ok(loucks_radius(index_1based).powi(3) * view_value(values, index_1based, name)?)
}

pub(super) fn view_value(
    values: ArrayView1<'_, Real>,
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    if index_1based == 0 || index_1based > values.len() {
        Err(GridError::SourceGridTooShort {
            name,
            required: index_1based.max(1),
            available: values.len(),
        })
    } else {
        Ok(values[index_1based - 1])
    }
}

pub(super) fn source_value(
    values: &[Real],
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    if index_1based == 0 || index_1based > values.len() {
        Err(GridError::SourceGridTooShort {
            name,
            required: index_1based.max(1),
            available: values.len(),
        })
    } else {
        Ok(values[index_1based - 1])
    }
}
