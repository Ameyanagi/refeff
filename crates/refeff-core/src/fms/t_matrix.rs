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

/// Port of the FEFF `fms_h` magnetic T-matrix branch.
///
/// This is the same single-site formula as [`fms_t_matrix_element`], except
/// `xphase_m` is addressed by both signed `l` and the FEFF magnetic slot
/// `imm = l**2 + l + m + 1`.
pub fn fms_hubbard_t_matrix_element(
    input: FmsHubbardTMatrixInput<'_>,
) -> Result<Complex32, FmsError> {
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
            let phase = magnetic_phase_shift_value(
                input.magnetic_phase_shifts,
                input.first.spin,
                l1_signed,
                input.first.magnetic,
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
        let phase_minus = magnetic_phase_shift_value(
            input.magnetic_phase_shifts,
            input.first.spin,
            l1_signed,
            input.first.magnetic,
            input.potential,
        )?;
        let phase_plus = magnetic_phase_shift_value(
            input.magnetic_phase_shifts,
            input.first.spin,
            -l1_signed,
            input.first.magnetic,
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
        let phase_minus_first = magnetic_phase_shift_value(
            input.magnetic_phase_shifts,
            input.first.spin,
            l1_signed,
            input.first.magnetic,
            input.potential,
        )?;
        let phase_minus_second = magnetic_phase_shift_value(
            input.magnetic_phase_shifts,
            input.second.spin,
            l1_signed,
            input.first.magnetic,
            input.potential,
        )?;
        let phase_plus_first = magnetic_phase_shift_value(
            input.magnetic_phase_shifts,
            input.first.spin,
            -l1_signed,
            input.first.magnetic,
            input.potential,
        )?;
        let phase_plus_second = magnetic_phase_shift_value(
            input.magnetic_phase_shifts,
            input.second.spin,
            -l1_signed,
            input.first.magnetic,
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

/// Build FEFF `fms_h` full Hubbard T-matrix table.
pub fn fms_hubbard_t_matrix_table(
    input: FmsHubbardTMatrixTableInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    let mut table = Array2::zeros((input.states.len(), input.states.len()).f());

    for (column, &second) in input.states.iter().enumerate() {
        ensure_state_spin(second.spin, input.spin_channels)?;
        let atom = checked_atom_index(second.atom)?;
        ensure_atom_table_index(atom, input.atoms.len())?;
        let potential_count = input.magnetic_phase_shifts.shape()[3];
        if potential_count == 0 {
            return Err(FmsError::TableIndexOutOfRange {
                table: "xphase_m",
                axis: "potential",
                index: 0,
            });
        }
        let potential = checked_potential(input.atoms[atom].potential, potential_count - 1)?;

        for (row, &first) in input.states.iter().enumerate() {
            table[(row, column)] = fms_hubbard_t_matrix_element(FmsHubbardTMatrixInput {
                first,
                second,
                spin_channels: input.spin_channels,
                spin_selector: input.spin_selector,
                potential,
                magnetic_phase_shifts: input.magnetic_phase_shifts,
                spin_orbit: input.spin_orbit,
            })?;
        }
    }

    Ok(table)
}

/// Apply FEFF `fms_h` `TFrm * tmatrxfull * TFrmInv` selected block transform.
pub fn fms_hubbard_transform_t_matrix(
    input: FmsHubbardTMatrixTransformInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    ensure_hubbard_transform_spin(input.spin_channels)?;
    ensure_square_table("tmatrxfull", input.t_matrix, input.states.len())?;
    validate_hubbard_transform_tables(input.use_transform, input.transform, input.inverse)?;

    let mut transformed = input.t_matrix.to_owned();
    for (atom_index, atom) in input.atoms.iter().enumerate() {
        let potential = checked_potential_index_for_transform(atom.potential, input.use_transform)?;
        for angular in 0..input.use_transform.shape()[0] {
            if !input.use_transform[(angular, potential)] {
                continue;
            }
            let start = hubbard_transform_block_start(
                input.states,
                atom_index + 1,
                angular,
                input.spin_channels,
            )?;
            let block = hubbard_block_dimension(angular)?;
            transform_square_block(
                &mut transformed,
                start,
                block,
                input.transform,
                input.inverse,
                angular,
                potential,
                false,
            )?;
        }
    }

    Ok(transformed)
}

/// Apply FEFF `fms_h` `TFrmInv * gg * TFrm` selected packed-block transform.
pub fn fms_hubbard_back_transform_scattering(
    input: FmsHubbardScatteringTransformInput<'_>,
) -> Result<Array3<Complex32>, FmsError> {
    ensure_hubbard_transform_spin(input.spin_channels)?;
    validate_hubbard_transform_tables(input.use_transform, input.transform, input.inverse)?;
    let mut transformed = input.scattering.to_owned();
    ensure_axis_len(
        "gg",
        "potential",
        transformed.shape()[2],
        input.potential_lmax.len().saturating_sub(1),
    )?;

    for potential in 0..input.potential_lmax.len() {
        ensure_axis_len(
            "UseTFrm",
            "potential",
            input.use_transform.shape()[1],
            potential,
        )?;
        let lmax = input.potential_lmax[potential];
        for angular in 0..=lmax {
            ensure_axis_len("UseTFrm", "l", input.use_transform.shape()[0], angular)?;
            if !input.use_transform[(angular, potential)] {
                continue;
            }
            let start = hubbard_packed_transform_block_start(input.spin_channels, angular)?;
            let block = hubbard_block_dimension(angular)?;
            let end = start + block - 1;
            ensure_axis_len("gg", "row", transformed.shape()[0], end)?;
            ensure_axis_len("gg", "column", transformed.shape()[1], end)?;
            let mut tmp = Array2::zeros((block, block).f());
            for column in 0..block {
                for row in 0..block {
                    let mut sum = Complex32::new(0.0, 0.0);
                    for k1 in 0..block {
                        for k2 in 0..block {
                            sum += input.inverse[(row, k1, angular, potential)]
                                * transformed[(start + k1, start + k2, potential)]
                                * input.transform[(k2, column, angular, potential)];
                        }
                    }
                    tmp[(row, column)] = sum;
                }
            }
            for column in 0..block {
                for row in 0..block {
                    transformed[(start + row, start + column, potential)] = tmp[(row, column)];
                }
            }
        }
    }

    Ok(transformed)
}

/// Apply FEFF `fms_h` `TFrmInv * gg_full * TFrm` to selected full-matrix blocks.
pub fn fms_hubbard_back_transform_full_scattering(
    input: FmsHubbardFullScatteringTransformInput<'_>,
) -> Result<Array2<Complex32>, FmsError> {
    ensure_hubbard_transform_spin(input.spin_channels)?;
    validate_hubbard_transform_tables(input.use_transform, input.transform, input.inverse)?;
    ensure_square_table("gg_full", input.full_scattering, input.states.len())?;

    let mut transformed = input.full_scattering.to_owned();
    for (atom_index, atom) in input.atoms.iter().enumerate() {
        let potential = checked_potential_index_for_transform(atom.potential, input.use_transform)?;
        ensure_axis_len("lipotx", "potential", input.potential_lmax.len(), potential)?;
        let lmax = input.potential_lmax[potential];
        for angular in 0..=lmax {
            ensure_axis_len("UseTFrm", "l", input.use_transform.shape()[0], angular)?;
            if !input.use_transform[(angular, potential)] {
                continue;
            }
            let start = hubbard_transform_block_start(
                input.states,
                atom_index + 1,
                angular,
                input.spin_channels,
            )?;
            let block = hubbard_block_dimension(angular)?;
            transform_square_block(
                &mut transformed,
                start,
                block,
                input.transform,
                input.inverse,
                angular,
                potential,
                true,
            )?;
        }
    }

    Ok(transformed)
}

fn magnetic_phase_shift_value(
    phase_shifts: ArrayView4<'_, Complex32>,
    spin: usize,
    angular_momentum: isize,
    magnetic: isize,
    potential: usize,
) -> Result<Complex32, FmsError> {
    let spin_index = spin.checked_sub(1).ok_or(FmsError::InvalidStateSpin {
        spin,
        spin_channels: phase_shifts.shape()[0],
    })?;
    ensure_axis_len("xphase_m", "spin", phase_shifts.shape()[0], spin_index)?;
    ensure_axis_len("xphase_m", "potential", phase_shifts.shape()[3], potential)?;
    let angular_len = phase_shifts.shape()[1];
    if angular_len == 0 || angular_len.is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: "xphase_m",
            value: angular_len,
            lx: angular_len,
        });
    }
    let lmax = (angular_len - 1) / 2;
    let angular_index = signed_magnetic_index(angular_momentum, lmax)?;
    ensure_axis_len("xphase_m", "l", angular_len, angular_index)?;
    let unsigned_l = angular_momentum.unsigned_abs();
    let magnetic_index = feff_magnetic_slot(unsigned_l, magnetic)?;
    ensure_axis_len("xphase_m", "imm", phase_shifts.shape()[2], magnetic_index)?;
    let value = phase_shifts[(spin_index, angular_index, magnetic_index, potential)];
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

fn feff_magnetic_slot(angular_momentum: usize, magnetic: isize) -> Result<usize, FmsError> {
    let angular_momentum_isize =
        isize::try_from(angular_momentum).map_err(|_| FmsError::InvalidAngularLimit {
            name: "l",
            value: angular_momentum,
            lx: angular_momentum,
        })?;
    if magnetic < -angular_momentum_isize || magnetic > angular_momentum_isize {
        return Err(FmsError::InvalidAngularLimit {
            name: "m",
            value: magnetic.unsigned_abs(),
            lx: angular_momentum,
        });
    }
    let base =
        angular_momentum
            .checked_mul(angular_momentum)
            .ok_or(FmsError::InvalidAngularLimit {
                name: "l",
                value: angular_momentum,
                lx: angular_momentum,
            })?;
    let offset = usize::try_from(magnetic + angular_momentum_isize).map_err(|_| {
        FmsError::InvalidAngularLimit {
            name: "m",
            value: magnetic.unsigned_abs(),
            lx: angular_momentum,
        }
    })?;
    base.checked_add(offset)
        .ok_or(FmsError::InvalidAngularLimit {
            name: "imm",
            value: base,
            lx: angular_momentum,
        })
}

fn ensure_hubbard_transform_spin(spin_channels: usize) -> Result<(), FmsError> {
    ensure_spin_channels(spin_channels)
}

fn validate_hubbard_transform_tables(
    use_transform: ArrayView2<'_, bool>,
    transform: ArrayView4<'_, Complex32>,
    inverse: ArrayView4<'_, Complex32>,
) -> Result<(), FmsError> {
    if transform.shape() != inverse.shape() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "TFrmInv",
            axis: "shape",
            index: inverse.len(),
        });
    }
    ensure_axis_len(
        "TFrm",
        "l",
        transform.shape()[2],
        use_transform.shape()[0].saturating_sub(1),
    )?;
    ensure_axis_len(
        "TFrm",
        "potential",
        transform.shape()[3],
        use_transform.shape()[1].saturating_sub(1),
    )?;
    for (index, value) in transform.iter().chain(inverse.iter()).enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return Err(FmsError::NonFiniteComplexValue {
                table: "TFrm",
                index,
            });
        }
    }
    Ok(())
}

fn checked_potential_index_for_transform(
    potential: i32,
    use_transform: ArrayView2<'_, bool>,
) -> Result<usize, FmsError> {
    let max_potential =
        use_transform.shape()[1]
            .checked_sub(1)
            .ok_or(FmsError::TableIndexOutOfRange {
                table: "UseTFrm",
                axis: "potential",
                index: 0,
            })?;
    checked_potential(potential, max_potential)
}

fn hubbard_transform_block_start(
    states: &[StateKet],
    atom: usize,
    angular_momentum: usize,
    spin_channels: usize,
) -> Result<usize, FmsError> {
    ensure_spin_channels(spin_channels)?;
    let first_magnetic =
        -isize::try_from(angular_momentum).map_err(|_| FmsError::InvalidAngularLimit {
            name: "l",
            value: angular_momentum,
            lx: angular_momentum,
        })?;
    // FEFF `fms_h` stores the transform start while scanning `lrstat`; for
    // nsp=2 that value is overwritten by the second spin at m=-l, then a
    // contiguous 2*l+1 block is transformed.
    let start = states
        .iter()
        .position(|state| {
            state.atom == atom
                && state.angular_momentum == angular_momentum
                && state.magnetic == first_magnetic
                && state.spin == spin_channels
        })
        .ok_or(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "hubbard_transform_block",
            index: angular_momentum,
        })?;
    let block = hubbard_block_dimension(angular_momentum)?;
    ensure_axis_len(
        "states",
        "hubbard_transform_block",
        states.len(),
        start + block - 1,
    )?;
    for offset in 0..block {
        let state = states[start + offset];
        if state.atom != atom || state.angular_momentum != angular_momentum {
            return Err(FmsError::TableIndexOutOfRange {
                table: "states",
                axis: "hubbard_transform_block",
                index: start + offset,
            });
        }
    }
    Ok(start)
}

fn hubbard_packed_transform_block_start(
    spin_channels: usize,
    angular_momentum: usize,
) -> Result<usize, FmsError> {
    ensure_spin_channels(spin_channels)?;
    let angular_start =
        angular_momentum
            .checked_mul(angular_momentum)
            .ok_or(FmsError::InvalidAngularLimit {
                name: "l",
                value: angular_momentum,
                lx: angular_momentum,
            })?;
    angular_start
        .checked_mul(spin_channels)
        .and_then(|start| start.checked_add(spin_channels - 1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "l",
            value: angular_momentum,
            lx: angular_momentum,
        })
}

fn hubbard_block_dimension(angular_momentum: usize) -> Result<usize, FmsError> {
    angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "l",
            value: angular_momentum,
            lx: angular_momentum,
        })
}

#[allow(clippy::too_many_arguments)]
fn transform_square_block(
    matrix: &mut Array2<Complex32>,
    start: usize,
    block: usize,
    transform: ArrayView4<'_, Complex32>,
    inverse: ArrayView4<'_, Complex32>,
    angular: usize,
    potential: usize,
    inverse_first: bool,
) -> Result<(), FmsError> {
    ensure_axis_len("TFrm", "row", transform.shape()[0], block - 1)?;
    ensure_axis_len("TFrm", "column", transform.shape()[1], block - 1)?;
    ensure_axis_len("TFrm", "l", transform.shape()[2], angular)?;
    ensure_axis_len("TFrm", "potential", transform.shape()[3], potential)?;
    let mut tmp = Array2::zeros((block, block).f());
    for column in 0..block {
        for row in 0..block {
            let mut sum = Complex32::new(0.0, 0.0);
            for k1 in 0..block {
                for k2 in 0..block {
                    let left = if inverse_first {
                        inverse[(row, k1, angular, potential)]
                    } else {
                        transform[(row, k1, angular, potential)]
                    };
                    let right = if inverse_first {
                        transform[(k2, column, angular, potential)]
                    } else {
                        inverse[(k2, column, angular, potential)]
                    };
                    sum += left * matrix[(start + k1, start + k2)] * right;
                }
            }
            tmp[(row, column)] = sum;
        }
    }
    for column in 0..block {
        for row in 0..block {
            matrix[(start + row, start + column)] = tmp[(row, column)];
        }
    }
    Ok(())
}
