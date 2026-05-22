use super::*;

pub(in crate::fms) fn phase_shift_value(
    phase_shifts: ArrayView3<'_, Complex32>,
    spin: usize,
    angular_momentum: isize,
    potential: usize,
) -> Result<Complex32, FmsError> {
    let spin_index = spin.checked_sub(1).ok_or(FmsError::InvalidStateSpin {
        spin,
        spin_channels: phase_shifts.shape()[0],
    })?;
    ensure_axis_len("xphase", "spin", phase_shifts.shape()[0], spin_index)?;
    ensure_axis_len("xphase", "potential", phase_shifts.shape()[2], potential)?;
    let angular_len = phase_shifts.shape()[1];
    if angular_len == 0 || angular_len.is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: "xphase",
            value: angular_len,
            lx: angular_len,
        });
    }
    let lmax = (angular_len - 1) / 2;
    let angular_index = signed_magnetic_index(angular_momentum, lmax)?;
    ensure_axis_len("xphase", "l", angular_len, angular_index)?;
    let value = phase_shifts[(spin_index, angular_index, potential)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(FmsError::NonFinitePhaseShift {
            spin,
            angular_momentum,
            potential,
        })
    }
}

pub(in crate::fms) fn t_matrix_phase(phase: Complex32) -> Complex32 {
    let two_i = Complex32::new(0.0, 2.0);
    ((two_i * phase).exp() - Complex32::new(1.0, 0.0)) / two_i
}

pub(in crate::fms) fn spin_orbit_coefficient(
    tables: &SpinOrbitCouplingTables,
    plus: bool,
    angular_momentum: usize,
    magnetic: isize,
    spin: usize,
) -> Result<f32, FmsError> {
    ensure_state_spin(spin, 2)?;
    let table = if plus { &tables.plus } else { &tables.minus };
    let table_name = if plus { "t3jp" } else { "t3jm" };
    ensure_axis_len(table_name, "l", table.shape()[0], angular_momentum)?;
    let offset = isize::try_from(tables.m_offset).map_err(|_| FmsError::InvalidAngularLimit {
        name: table_name,
        value: tables.m_offset,
        lx: tables.m_offset,
    })?;
    let magnetic_index =
        usize::try_from(magnetic + offset).map_err(|_| FmsError::InvalidAngularLimit {
            name: table_name,
            value: magnetic.unsigned_abs(),
            lx: tables.m_offset,
        })?;
    ensure_axis_len(table_name, "m", table.shape()[1], magnetic_index)?;
    let spin_index = spin - 1;
    ensure_axis_len(table_name, "spin", table.shape()[2], spin_index)?;
    Ok(table[(angular_momentum, magnetic_index, spin_index)] as f32)
}
