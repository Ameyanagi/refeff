use super::*;

/// Port of the FEFF FMS single-site T-matrix branch.
///
/// This evaluates the same-atom portion of `fmspack`'s state-pair loop. The
/// scalar non-spin branch uses the diagonal phase-shift expression directly;
/// the spin-orbit branch combines `j=l-1/2` and `j=l+1/2` phase shifts with
/// FEFF's `t3jm` and `t3jp` Clebsch-Gordon tables. Non-single-site pairs and
/// disallowed spin-mixing pairs return zero.
pub fn fms_t_matrix_element(input: FmsTMatrixInput<'_>) -> Result<Complex32, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    ensure_state_spin(input.first.spin, input.spin_channels)?;
    ensure_state_spin(input.second.spin, input.spin_channels)?;
    if input.first.atom != input.second.atom {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let l1 = input.first.angular_momentum;
    let l2 = input.second.angular_momentum;
    let l1_signed = isize::try_from(l1).map_err(|_| FmsError::InvalidAngularLimit {
        name: "l",
        value: l1,
        lx: l1,
    })?;

    if input.spin_channels == 1 && input.spin_selector == 0 {
        return if input.first == input.second {
            let phase = phase_shift_value(
                input.phase_shifts,
                input.first.spin,
                l1_signed,
                input.potential,
            )?;
            Ok(t_matrix_phase(phase))
        } else {
            Ok(Complex32::new(0.0, 0.0))
        };
    }

    if input.first == input.second {
        let coupling_spin = if input.spin_channels == 1 {
            if input.spin_selector > 0 { 2 } else { 1 }
        } else {
            input.first.spin
        };
        let minus = spin_orbit_coefficient(
            input.spin_orbit,
            false,
            l1,
            input.first.magnetic,
            coupling_spin,
        )?;
        let plus = spin_orbit_coefficient(
            input.spin_orbit,
            true,
            l1,
            input.first.magnetic,
            coupling_spin,
        )?;
        let phase_minus = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            l1_signed,
            input.potential,
        )?;
        let phase_plus = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            -l1_signed,
            input.potential,
        )?;
        return Ok(t_matrix_phase(phase_minus) * (minus * minus)
            + t_matrix_phase(phase_plus) * (plus * plus));
    }

    if input.spin_channels == 2
        && l1 == l2
        && input.first.magnetic + input.first.spin as isize
            == input.second.magnetic + input.second.spin as isize
    {
        let minus_first = spin_orbit_coefficient(
            input.spin_orbit,
            false,
            l1,
            input.first.magnetic,
            input.first.spin,
        )?;
        let minus_second = spin_orbit_coefficient(
            input.spin_orbit,
            false,
            l1,
            input.second.magnetic,
            input.second.spin,
        )?;
        let plus_first = spin_orbit_coefficient(
            input.spin_orbit,
            true,
            l1,
            input.first.magnetic,
            input.first.spin,
        )?;
        let plus_second = spin_orbit_coefficient(
            input.spin_orbit,
            true,
            l1,
            input.second.magnetic,
            input.second.spin,
        )?;
        let phase_minus_first = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            l1_signed,
            input.potential,
        )?;
        let phase_minus_second = phase_shift_value(
            input.phase_shifts,
            input.second.spin,
            l1_signed,
            input.potential,
        )?;
        let phase_plus_first = phase_shift_value(
            input.phase_shifts,
            input.first.spin,
            -l1_signed,
            input.potential,
        )?;
        let phase_plus_second = phase_shift_value(
            input.phase_shifts,
            input.second.spin,
            -l1_signed,
            input.potential,
        )?;
        let minus_phase =
            (t_matrix_phase(phase_minus_first) + t_matrix_phase(phase_minus_second)) * 0.5;
        let plus_phase =
            (t_matrix_phase(phase_plus_first) + t_matrix_phase(phase_plus_second)) * 0.5;
        return Ok(minus_phase * minus_first * minus_second + plus_phase * plus_first * plus_second);
    }

    Ok(Complex32::new(0.0, 0.0))
}

/// Build FEFF's compact FMS T-matrix table `tmatrx`.
///
/// The first row contains the same-site diagonal T element for each state. When
/// `spin_channels == 2`, the second row contains the one allowed spin-mixing
/// partner for that state, matching FEFF's compact storage used by `gglu`.
/// The returned table is Fortran-order with shape `(spin_channels, states)`.
pub fn fms_t_matrix_table(input: FmsTMatrixTableInput<'_>) -> Result<Array2<Complex32>, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    let mut table = Array2::zeros((input.spin_channels, input.states.len()).f());

    for (column, &first) in input.states.iter().enumerate() {
        ensure_state_spin(first.spin, input.spin_channels)?;
        let atom = checked_atom_index(first.atom)?;
        ensure_atom_table_index(atom, input.atoms.len())?;
        let potential = checked_phase_potential(input.atoms[atom].potential, input.phase_shifts)?;

        table[(0, column)] = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: input.spin_channels,
            spin_selector: input.spin_selector,
            potential,
            phase_shifts: input.phase_shifts,
            spin_orbit: input.spin_orbit,
        })?;

        if input.spin_channels == 2 {
            for &second in input.states {
                if second == first {
                    continue;
                }
                let value = fms_t_matrix_element(FmsTMatrixInput {
                    first,
                    second,
                    spin_channels: input.spin_channels,
                    spin_selector: input.spin_selector,
                    potential,
                    phase_shifts: input.phase_shifts,
                    spin_orbit: input.spin_orbit,
                })?;
                if value != Complex32::new(0.0, 0.0) {
                    table[(1, column)] = value;
                    break;
                }
            }
        }
    }

    Ok(table)
}
