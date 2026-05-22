use super::*;

pub(in crate::fms) fn fms_vector_within_tolerance(vector: &[Complex32], tolerance: f32) -> bool {
    vector
        .iter()
        .all(|value| value.re.abs() <= tolerance && value.im.abs() <= tolerance)
}

pub(in crate::fms) fn fms_scaled_tolerance(
    tolerance: f32,
    scale: f32,
    solver: &'static str,
    step: &'static str,
) -> Result<f32, FmsError> {
    let scaled = tolerance * scale;
    if scaled.is_finite() && scaled >= 0.0 {
        Ok(scaled)
    } else {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    }
}

pub(in crate::fms) fn fms_cdot(left: &[Complex32], right: &[Complex32]) -> Complex32 {
    left.iter()
        .zip(right.iter())
        .map(|(&bra, &ket)| bra.conj() * ket)
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

pub(in crate::fms) fn fms_matvec(
    matrix: ArrayView2<'_, Complex32>,
    vector: &[Complex32],
) -> Vec<Complex32> {
    let mut output = vec![Complex32::new(0.0, 0.0); vector.len()];
    for column in 0..vector.len() {
        for row in 0..vector.len() {
            output[row] += matrix[(row, column)] * vector[column];
        }
    }
    output
}

pub(in crate::fms) fn fms_adjoint_matvec(
    matrix: ArrayView2<'_, Complex32>,
    vector: &[Complex32],
) -> Vec<Complex32> {
    let mut output = vec![Complex32::new(0.0, 0.0); vector.len()];
    for column in 0..vector.len() {
        for row in 0..vector.len() {
            output[column] += matrix[(row, column)].conj() * vector[row];
        }
    }
    output
}

pub(in crate::fms) fn fms_checked_divide(
    numerator: Complex32,
    denominator: Complex32,
    solver: &'static str,
    step: &'static str,
) -> Result<Complex32, FmsError> {
    fms_checked_nonzero(denominator, solver, step)?;
    Ok(numerator / denominator)
}

pub(in crate::fms) fn fms_checked_nonzero(
    value: Complex32,
    solver: &'static str,
    step: &'static str,
) -> Result<(), FmsError> {
    if value == Complex32::new(0.0, 0.0) {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    } else {
        Ok(())
    }
}

pub(in crate::fms) fn fms_checked_positive_real(
    value: f32,
    solver: &'static str,
    step: &'static str,
) -> Result<f32, FmsError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    }
}

pub(in crate::fms) fn fms_checked_nonnegative_real(
    value: f32,
    solver: &'static str,
    step: &'static str,
) -> Result<f32, FmsError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(FmsError::IterativeSolverBreakdown { solver, step })
    }
}

pub(in crate::fms) fn fms_lu_system_matrix(
    states: &[StateKet],
    spin_channels: usize,
    free_propagator: ArrayView2<'_, Complex32>,
    t_matrix: ArrayView2<'_, Complex32>,
) -> Result<Array2<Complex32>, FmsError> {
    if states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }

    let mut system_matrix = Array2::zeros((states.len(), states.len()).f());
    for (column, &state) in states.iter().enumerate() {
        ensure_state_spin(state.spin, spin_channels)?;
        for row in 0..states.len() {
            system_matrix[(row, column)] = -free_propagator[(row, column)] * t_matrix[(0, column)];
        }

        if spin_channels == 2
            && let Some(partner) = fms_spin_partner_index(state, column, states.len())?
        {
            for row in 0..states.len() {
                system_matrix[(row, column)] -=
                    free_propagator[(row, partner)] * t_matrix[(1, column)];
            }
        }
        system_matrix[(column, column)] += Complex32::new(1.0, 0.0);
    }

    Ok(system_matrix)
}

pub(in crate::fms) fn fms_full_potential_lu_system_matrix(
    states: &[StateKet],
    free_propagator: ArrayView2<'_, Complex32>,
    t_matrix: ArrayView2<'_, Complex32>,
) -> Result<Array2<Complex32>, FmsError> {
    if states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    let mut system_matrix = Array2::zeros((states.len(), states.len()).f());
    for column in 0..states.len() {
        for row in 0..states.len() {
            system_matrix[(row, column)] = (0..states.len())
                .map(|inner| -free_propagator[(row, inner)] * t_matrix[(inner, column)])
                .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value);
        }
        system_matrix[(column, column)] =
            free_propagator[(column, column)] + Complex32::new(1.0, 0.0);
    }

    Ok(system_matrix)
}

pub(in crate::fms) fn fms_spin_partner_index(
    state: StateKet,
    column: usize,
    state_count: usize,
) -> Result<Option<usize>, FmsError> {
    let angular_momentum =
        isize::try_from(state.angular_momentum).map_err(|_| FmsError::InvalidAngularLimit {
            name: "l",
            value: state.angular_momentum,
            lx: state.angular_momentum,
        })?;
    let projection = state.magnetic + state.spin as isize;
    if projection <= -angular_momentum + 1 || projection >= angular_momentum + 2 {
        return Ok(None);
    }

    let column = isize::try_from(column).map_err(|_| FmsError::TableIndexOutOfRange {
        table: "states",
        axis: "state",
        index: column,
    })?;
    let partner = match state.spin {
        1 => column - 1,
        2 => column + 1,
        spin => {
            return Err(FmsError::InvalidStateSpin {
                spin,
                spin_channels: 2,
            });
        }
    };
    let partner = usize::try_from(partner).map_err(|_| FmsError::TableIndexOutOfRange {
        table: "states",
        axis: "spin_partner",
        index: 0,
    })?;
    ensure_axis_len("states", "spin_partner", state_count, partner)?;
    Ok(Some(partner))
}
