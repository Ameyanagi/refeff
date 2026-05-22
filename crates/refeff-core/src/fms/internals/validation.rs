use super::*;

pub(in crate::fms) fn ensure_spin_channels(spin_channels: usize) -> Result<(), FmsError> {
    if (1..=2).contains(&spin_channels) {
        Ok(())
    } else {
        Err(FmsError::InvalidSpinChannelCount {
            value: spin_channels,
        })
    }
}

pub(in crate::fms) fn ensure_state_spin(spin: usize, spin_channels: usize) -> Result<(), FmsError> {
    if (1..=spin_channels).contains(&spin) {
        Ok(())
    } else {
        Err(FmsError::InvalidStateSpin {
            spin,
            spin_channels,
        })
    }
}

pub(in crate::fms) fn ensure_square_table(
    table: &'static str,
    matrix: ArrayView2<'_, Complex32>,
    expected_order: usize,
) -> Result<(), FmsError> {
    if matrix.shape() == [expected_order, expected_order] {
        Ok(())
    } else {
        Err(FmsError::TableIndexOutOfRange {
            table,
            axis: "shape",
            index: expected_order,
        })
    }
}

pub(in crate::fms) fn potential_lmax_for(
    potential_lmax: &[usize],
    potential: usize,
) -> Result<usize, FmsError> {
    potential_lmax
        .get(potential)
        .copied()
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "lipotx",
            axis: "potential",
            index: potential,
        })
}

pub(in crate::fms) fn representative_offset(
    representative_offsets: &[Option<usize>],
    potential: usize,
) -> Result<usize, FmsError> {
    representative_offsets
        .get(potential)
        .copied()
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "i0",
            axis: "potential",
            index: potential,
        })?
        .ok_or(FmsError::MissingRepresentativePotential { potential })
}

pub(in crate::fms) fn clamp_fms_lipotx(value: i32, global_lmax: usize) -> usize {
    if value < 0 {
        global_lmax
    } else {
        usize::try_from(value).map_or(global_lmax, |lmax| lmax.min(global_lmax))
    }
}

pub(in crate::fms) fn fms_state_ket_error(error: StateKetError) -> FmsError {
    match error {
        StateKetError::InvalidSpinCount => FmsError::InvalidSpinChannelCount { value: 0 },
        StateKetError::PotentialOutOfRange {
            atom,
            potential,
            potential_count,
        } => FmsError::StateKetPotentialOutOfRange {
            atom,
            potential,
            potential_count,
        },
        StateKetError::CapacityExceeded { capacity } => {
            FmsError::StateCapacityExceeded { capacity }
        }
        StateKetError::IntegerOverflow { field, value } => {
            FmsError::IntegerOverflow { field, value }
        }
    }
}

pub(in crate::fms) fn checked_potential(
    potential: i32,
    max_potential: usize,
) -> Result<usize, FmsError> {
    let Ok(potential_index) = usize::try_from(potential) else {
        return Err(FmsError::PotentialOutOfRange {
            potential,
            max_potential,
        });
    };
    if potential_index <= max_potential {
        Ok(potential_index)
    } else {
        Err(FmsError::PotentialOutOfRange {
            potential,
            max_potential,
        })
    }
}

pub(in crate::fms) fn checked_phase_potential(
    potential: i32,
    phase_shifts: ArrayView3<'_, Complex32>,
) -> Result<usize, FmsError> {
    let potential_count = phase_shifts.shape()[2];
    if potential_count == 0 {
        return Err(FmsError::TableIndexOutOfRange {
            table: "xphase",
            axis: "potential",
            index: 0,
        });
    }
    checked_potential(potential, potential_count - 1)
}

pub(in crate::fms) fn ensure_axis_len(
    table: &'static str,
    axis: &'static str,
    len: usize,
    index: usize,
) -> Result<(), FmsError> {
    if index < len {
        Ok(())
    } else {
        Err(FmsError::TableIndexOutOfRange { table, axis, index })
    }
}

pub(in crate::fms) fn normalization_value(
    xnlm: ArrayView2<'_, Real>,
    mu: usize,
    angular_momentum: usize,
) -> Result<f32, FmsError> {
    let value = xnlm[(mu, angular_momentum)] as f32;
    if value.is_finite() && value != 0.0 {
        Ok(value)
    } else {
        Err(FmsError::InvalidNormalization {
            mu,
            angular_momentum,
        })
    }
}

pub(in crate::fms) fn angular_weight(angular_momentum: usize) -> Result<Complex32, FmsError> {
    let value = angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "angular_momentum",
            value: angular_momentum,
            lx: angular_momentum,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}

pub(in crate::fms) fn odd_factor(index: usize, lx: usize) -> Result<Complex32, FmsError> {
    let value = index
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}
