use super::*;

pub(super) fn fill_rotxan_small_d(lmax: usize, mmax: usize, beta: f32, dri0: &mut Array3<f32>) {
    let lxp1 = lmax + 1;
    let mxp1 = mmax + 1;
    let ndm = lxp1 + mxp1 - 1;
    let xc = (beta / 2.0).cos();
    let xs = (beta / 2.0).sin();
    let s = beta.sin();

    dri0[(1, 1, 1)] = 1.0;
    if lxp1 < 2 {
        return;
    }
    dri0[(2, 1, 1)] = xc * xc;
    dri0[(2, 1, 2)] = s / 2.0_f32.sqrt();
    dri0[(2, 1, 3)] = xs * xs;
    dri0[(2, 2, 1)] = -dri0[(2, 1, 2)];
    dri0[(2, 2, 2)] = beta.cos();
    dri0[(2, 2, 3)] = dri0[(2, 1, 2)];
    dri0[(2, 3, 1)] = dri0[(2, 1, 3)];
    dri0[(2, 3, 2)] = -dri0[(2, 2, 3)];
    dri0[(2, 3, 3)] = dri0[(2, 1, 1)];

    for l in 3..=lxp1 {
        let mut ln = 2 * l - 1;
        let mut lm = 2 * l - 3;
        if ln > ndm {
            ln = ndm;
        }
        if lm > ndm {
            lm = ndm;
        }
        for n in 1..=ln {
            for m in 1..=lm {
                let l_i = l as i32;
                let n_i = n as i32;
                let m_i = m as i32;
                let t1 = ((2 * l_i - 1 - n_i) * (2 * l_i - 2 - n_i)) as f32;
                let t = ((2 * l_i - 1 - m_i) * (2 * l_i - 2 - m_i)) as f32;
                let f1 = (t1 / t).sqrt();
                let f2 = (((2 * l_i - 1 - n_i) * (n_i - 1)) as f32 / t).sqrt();
                let t3 = ((n_i - 2) * (n_i - 1)) as f32;
                let f3 = (t3 / t).sqrt();
                let mut dlnm = f1 * xc * xc * dri0[(l - 1, n, m)];
                if n > 1 {
                    dlnm -= f2 * s * dri0[(l - 1, n - 1, m)];
                }
                if n > 2 {
                    dlnm += f3 * xs * xs * dri0[(l - 1, n - 2, m)];
                }
                dri0[(l, n, m)] = dlnm;
                if n > (2 * l - 3) {
                    dri0[(l, m, n)] = alternating_f32(n - m) * dlnm;
                }
            }

            if n > (2 * l - 3) {
                dri0[(l, 2 * l - 2, 2 * l - 2)] = dri0[(l, 2, 2)];
                dri0[(l, 2 * l - 1, 2 * l - 2)] = -dri0[(l, 1, 2)];
                dri0[(l, 2 * l - 2, 2 * l - 1)] = -dri0[(l, 2, 1)];
                dri0[(l, 2 * l - 1, 2 * l - 1)] = dri0[(l, 1, 1)];
            }
        }
    }
}

pub(super) fn copy_rotxan_small_d(
    lmax: usize,
    mmax: usize,
    dri0: &ArrayView3<'_, f32>,
    drix: &mut Array3<Complex32>,
) -> Result<(), FmsError> {
    for il in 1..=lmax + 1 {
        let mmx = (il - 1).min(mmax);
        for m1 in -(mmx as isize)..=(mmx as isize) {
            for m2 in -(mmx as isize)..=(mmx as isize) {
                let row = signed_magnetic_index(m2, lmax)?;
                let column = signed_magnetic_index(m1, lmax)?;
                drix[(row, column, il - 1)] = Complex32::new(
                    dri0[(il, (m1 + il as isize) as usize, (m2 + il as isize) as usize)],
                    0.0,
                );
            }
        }
    }
    Ok(())
}

pub(super) fn apply_rotxan_phase(
    lmax: usize,
    phi: f32,
    direction: FmsRotationDirection,
    drix: &mut Array3<Complex32>,
) -> Result<(), FmsError> {
    for il in 0..=lmax {
        for m1 in -(il as isize)..=(il as isize) {
            let angle = match direction {
                FmsRotationDirection::Forward => m1 as f32 * (phi - std::f32::consts::PI),
                FmsRotationDirection::Backward => -m1 as f32 * (phi - std::f32::consts::PI),
            };
            let phase = Complex32::new(0.0, angle).exp();
            for m2 in -(il as isize)..=(il as isize) {
                match direction {
                    FmsRotationDirection::Forward => {
                        let row = signed_magnetic_index(m1, lmax)?;
                        let column = signed_magnetic_index(m2, lmax)?;
                        drix[(row, column, il)] *= phase;
                    }
                    FmsRotationDirection::Backward => {
                        let row = signed_magnetic_index(m2, lmax)?;
                        let column = signed_magnetic_index(m1, lmax)?;
                        drix[(row, column, il)] *= phase;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn signed_magnetic_index(magnetic: isize, lmax: usize) -> Result<usize, FmsError> {
    let lmax_isize = isize::try_from(lmax).map_err(|_| FmsError::InvalidAngularLimit {
        name: "lmax",
        value: lmax,
        lx: lmax,
    })?;
    let index = magnetic + lmax_isize;
    usize::try_from(index).map_err(|_| FmsError::InvalidAngularLimit {
        name: "magnetic",
        value: magnetic.unsigned_abs(),
        lx: lmax,
    })
}

pub(super) fn alternating_f32(value: usize) -> f32 {
    if value.is_multiple_of(2) { 1.0 } else { -1.0 }
}

pub(super) fn fms_atom_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    fms_atom_distance_squared(left, right).sqrt()
}

pub(super) fn fms_atom_distance_squared(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

pub(super) fn fms_free_propagator_prefactor(
    rho: Complex32,
    wave_number: Complex32,
    mean_square_displacement: f32,
) -> Complex32 {
    const BOHR: f32 = 0.529_177_25;
    let phase = (Complex32::new(0.0, 1.0) * rho).exp() / rho;
    let damping_factor = Complex32::new(-mean_square_displacement / (BOHR * BOHR), 0.0);
    let damping = (damping_factor * wave_number * wave_number).exp();
    phase * damping
}

pub(super) fn rotation_table_value(
    table: ArrayView3<'_, Complex32>,
    m2: isize,
    m1: isize,
    angular_momentum: usize,
    table_name: &'static str,
) -> Result<Complex32, FmsError> {
    let shape = table.shape();
    if shape[0] == 0 || shape[0] != shape[1] || shape[0].is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: table_name,
            value: shape[0],
            lx: shape[0],
        });
    }
    ensure_axis_len(table_name, "l", shape[2], angular_momentum)?;
    let lmax = (shape[0] - 1) / 2;
    let row = signed_magnetic_index(m2, lmax)?;
    let column = signed_magnetic_index(m1, lmax)?;
    ensure_axis_len(table_name, "m2", shape[0], row)?;
    ensure_axis_len(table_name, "m1", shape[1], column)?;
    Ok(table[(row, column, angular_momentum)])
}

pub(super) fn rotation_pair_view<'a>(
    rotations: ArrayView6<'a, Complex32>,
    direction: FmsRotationDirection,
    atom2: usize,
    atom1: usize,
) -> Result<ArrayView3<'a, Complex32>, FmsError> {
    let shape = rotations.shape();
    if shape[0] == 0 || shape[0] != shape[1] || shape[0].is_multiple_of(2) {
        return Err(FmsError::InvalidAngularLimit {
            name: "rotations",
            value: shape[0],
            lx: shape[0],
        });
    }
    ensure_axis_len("rotations", "k", shape[3], 1)?;
    ensure_axis_len("rotations", "atom2", shape[4], atom2)?;
    ensure_axis_len("rotations", "atom1", shape[5], atom1)?;

    let branch = match direction {
        FmsRotationDirection::Forward => 0,
        FmsRotationDirection::Backward => 1,
    };
    Ok(rotations
        .index_axis_move(Axis(5), atom1)
        .index_axis_move(Axis(4), atom2)
        .index_axis_move(Axis(3), branch))
}

pub(super) fn ensure_spin_channels(spin_channels: usize) -> Result<(), FmsError> {
    if (1..=2).contains(&spin_channels) {
        Ok(())
    } else {
        Err(FmsError::InvalidSpinChannelCount {
            value: spin_channels,
        })
    }
}

pub(super) fn ensure_state_spin(spin: usize, spin_channels: usize) -> Result<(), FmsError> {
    if (1..=spin_channels).contains(&spin) {
        Ok(())
    } else {
        Err(FmsError::InvalidStateSpin {
            spin,
            spin_channels,
        })
    }
}

pub(super) fn phase_shift_value(
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

pub(super) fn t_matrix_phase(phase: Complex32) -> Complex32 {
    let two_i = Complex32::new(0.0, 2.0);
    ((two_i * phase).exp() - Complex32::new(1.0, 0.0)) / two_i
}

pub(super) fn spin_orbit_coefficient(
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

pub(super) struct FmsIterativeScatteringInput<'a> {
    pub(super) states: &'a [StateKet],
    pub(super) spin_channels: usize,
    pub(super) global_lmax: usize,
    pub(super) potential_lmax: &'a [usize],
    pub(super) representative_offsets: &'a [Option<usize>],
    pub(super) potential_start: usize,
    pub(super) potential_end: usize,
    pub(super) free_propagator: ArrayView2<'a, Complex32>,
    pub(super) t_matrix: ArrayView2<'a, Complex32>,
    pub(super) calculated_l: &'a [bool],
    pub(super) convergence_tolerance: f32,
    pub(super) zero_tolerance: f32,
}

pub(super) struct FmsIterativeScatteringResult {
    pub(super) system_matrix: Array2<Complex32>,
    pub(super) scattering: Array3<Complex32>,
    pub(super) multiple_scattering_order: usize,
}

pub(super) fn fms_iterative_scattering(
    input: FmsIterativeScatteringInput<'_>,
    solve: impl Fn(ArrayView2<'_, Complex32>, usize, f32) -> Result<(Vec<Complex32>, usize), FmsError>,
) -> Result<FmsIterativeScatteringResult, FmsError> {
    let system_matrix = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: input.states,
        spin_channels: input.spin_channels,
        free_propagator: input.free_propagator,
        t_matrix: input.t_matrix,
        zero_tolerance: input.zero_tolerance,
    })?;
    fms_iterative_scattering_with_system(input, system_matrix, solve)
}

pub(super) fn fms_iterative_scattering_with_system(
    input: FmsIterativeScatteringInput<'_>,
    system_matrix: Array2<Complex32>,
    solve: impl Fn(ArrayView2<'_, Complex32>, usize, f32) -> Result<(Vec<Complex32>, usize), FmsError>,
) -> Result<FmsIterativeScatteringResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.states.is_empty() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "states",
            axis: "state",
            index: 0,
        });
    }
    ensure_axis_len(
        "states",
        "potential_start",
        input.representative_offsets.len(),
        input.potential_start,
    )?;
    ensure_axis_len(
        "states",
        "potential_end",
        input.representative_offsets.len(),
        input.potential_end,
    )?;
    if input.potential_start > input.potential_end {
        return Err(FmsError::TableIndexOutOfRange {
            table: "potential_range",
            axis: "potential",
            index: input.potential_start,
        });
    }
    if !input.convergence_tolerance.is_finite() || input.convergence_tolerance < 0.0 {
        return Err(FmsError::InvalidTolerance {
            name: "toler1",
            value: input.convergence_tolerance,
        });
    }
    ensure_square_table("g0t", system_matrix.view(), input.states.len())?;

    let channel_count = input
        .global_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(input.spin_channels))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "global_lmax",
            value: input.global_lmax,
            lx: input.global_lmax,
        })?;
    let mut scattering = Array3::zeros(
        (
            channel_count,
            channel_count,
            input.representative_offsets.len(),
        )
            .f(),
    );
    let mut multiple_scattering_order = 0;

    for potential in input.potential_start..=input.potential_end {
        let lmax = potential_lmax_for(input.potential_lmax, potential)?.min(input.global_lmax);
        let ipart = lmax
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .and_then(|value| value.checked_mul(input.spin_channels))
            .ok_or(FmsError::InvalidAngularLimit {
                name: "lipotx",
                value: lmax,
                lx: input.global_lmax,
            })?;
        let offset = representative_offset(input.representative_offsets, potential)?;
        ensure_axis_len(
            "g0",
            "representative_state",
            input.free_propagator.shape()[0],
            offset,
        )?;
        ensure_axis_len(
            "g0",
            "representative_block",
            input.free_propagator.shape()[0],
            offset
                .checked_add(ipart - 1)
                .ok_or(FmsError::TableIndexOutOfRange {
                    table: "g0",
                    axis: "representative_block",
                    index: ipart,
                })?,
        )?;

        for source_column in 0..ipart {
            let source_state =
                offset
                    .checked_add(source_column)
                    .ok_or(FmsError::TableIndexOutOfRange {
                        table: "states",
                        axis: "source_state",
                        index: source_column,
                    })?;
            ensure_axis_len("states", "source_state", input.states.len(), source_state)?;
            let angular_momentum = input.states[source_state].angular_momentum;
            ensure_axis_len("lcalc", "l", input.calculated_l.len(), angular_momentum)?;
            if !input.calculated_l[angular_momentum] {
                continue;
            }

            let (solution, msord) = solve(
                system_matrix.view(),
                source_state,
                input.convergence_tolerance,
            )?;
            multiple_scattering_order = msord;
            for row in 0..ipart {
                let target_state =
                    offset
                        .checked_add(row)
                        .ok_or(FmsError::TableIndexOutOfRange {
                            table: "g0",
                            axis: "row_state",
                            index: row,
                        })?;
                ensure_axis_len(
                    "g0",
                    "row_state",
                    input.free_propagator.shape()[0],
                    target_state,
                )?;
                let value = (0..input.states.len())
                    .map(|state| input.free_propagator[(target_state, state)] * solution[state])
                    .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value);
                scattering[(row, source_column, potential)] = value;
            }
        }
    }

    Ok(FmsIterativeScatteringResult {
        system_matrix,
        scattering,
        multiple_scattering_order,
    })
}

pub(super) fn fms_bicgstab_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];
    let mut rvec = vec![zero; state_count];
    rvec[source_state] = Complex32::new(1.0, 0.0);

    if fms_vector_within_tolerance(&rvec, tolerance) {
        return Ok((xvec, multiple_scattering_order));
    }

    let pvec = rvec.clone();
    let avec = fms_matvec(system_matrix, &pvec);
    multiple_scattering_order += 1;

    let mut aa = fms_cdot(&avec, &avec);
    let wa = fms_cdot(&rvec, &avec);
    let aw = wa.conj();
    let mut ww = fms_cdot(&rvec, &rvec);
    fms_checked_nonzero(aa, "ggbi", "avec dot avec")?;
    fms_checked_nonzero(ww, "ggbi", "rvec dot rvec")?;
    let dd = aa * ww - aw * wa;
    let scaled_dd = fms_checked_divide(
        fms_checked_divide(dd, aa, "ggbi", "dd/aa")?,
        ww,
        "ggbi",
        "dd/ww",
    )?;
    let yvec = if scaled_dd.norm() < 1.0e-8 {
        rvec.iter().map(|&value| value / ww).collect::<Vec<_>>()
    } else {
        fms_checked_nonzero(dd, "ggbi", "Gram determinant")?;
        ww = (ww - aw) / dd;
        aa = (wa - aa) / dd;
        rvec.iter()
            .zip(avec.iter())
            .map(|(&residual, &matrix_residual)| residual * aa + matrix_residual * ww)
            .collect::<Vec<_>>()
    };
    let del = fms_cdot(&yvec, &rvec);
    let delp = fms_cdot(&yvec, &avec);
    let omega = fms_checked_divide(del, delp, "ggbi", "omega")?;
    let svec = rvec
        .iter()
        .zip(avec.iter())
        .map(|(&residual, &matrix_residual)| residual - omega * matrix_residual)
        .collect::<Vec<_>>();

    if fms_vector_within_tolerance(&svec, tolerance) {
        for (solution, &direction) in xvec.iter_mut().zip(pvec.iter()) {
            *solution += omega * direction;
        }
        return Ok((xvec, multiple_scattering_order));
    }

    let asve = fms_matvec(system_matrix, &svec);
    multiple_scattering_order += 1;
    aa = fms_cdot(&asve, &asve);
    let wa = fms_cdot(&asve, &svec);
    let chi = fms_checked_divide(wa, aa, "ggbi", "chi")?;
    for ((solution, &direction), &shadow) in xvec.iter_mut().zip(pvec.iter()).zip(svec.iter()) {
        *solution += omega * direction + chi * shadow;
    }

    // FEFF `ggbi` resets `ipass` before label 380, so this branch exits after
    // the first residual update even when the residual is still above tolerance.
    Ok((xvec, multiple_scattering_order))
}

pub(super) fn fms_recursion_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    const MAX_RESTARTS: usize = 128;
    const MAX_ITERATIONS: usize = 100;

    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let one = Complex32::new(1.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];

    for restart in 0..MAX_RESTARTS {
        let mut rvec = if restart > 0 {
            fms_matvec(system_matrix, &xvec)
        } else {
            vec![zero; state_count]
        };
        rvec[source_state] -= one;

        let mut xket = rvec.iter().map(|&value| -value).collect::<Vec<_>>();
        let residual_norm = fms_cdot(&xket, &xket);
        if residual_norm == zero {
            return Ok((xvec, multiple_scattering_order));
        }

        let xfnorm =
            1.0 / fms_checked_positive_real(residual_norm.re, "ggrm", "initial residual norm")?;
        let mut xbra = xket.iter().map(|&value| value * xfnorm).collect::<Vec<_>>();
        let mut tket = fms_matvec(system_matrix, &xket);
        multiple_scattering_order += 1;

        let mut aa = fms_cdot(&xbra, &tket);
        let mut aac = aa.conj();
        let mut bb = zero;
        let mut bbc = zero;
        let mut betac = aa;
        fms_checked_nonzero(betac, "ggrm", "initial beta")?;

        let mut yy = one;
        let mut xketp = vec![zero; state_count];
        let mut xbrap = vec![zero; state_count];
        let mut zvec = xket.clone();
        for (solution, &basis) in xvec.iter_mut().zip(zvec.iter()) {
            *solution += basis / betac;
        }
        let mut svec = tket.clone();
        for (residual, &matrix_basis) in rvec.iter_mut().zip(svec.iter()) {
            *residual += matrix_basis / betac;
        }

        for _ in 0..MAX_ITERATIONS {
            for ((matrix_basis, &basis), &previous_basis) in
                tket.iter_mut().zip(xket.iter()).zip(xketp.iter())
            {
                *matrix_basis -= aa * basis + bb * previous_basis;
            }

            let mut tbra = fms_adjoint_matvec(system_matrix, &xbra);
            for ((matrix_bra, &bra), &previous_bra) in
                tbra.iter_mut().zip(xbra.iter()).zip(xbrap.iter())
            {
                *matrix_bra -= aac * bra + bbc * previous_bra;
            }

            let recurrence_norm = fms_cdot(&tbra, &tket);
            if recurrence_norm == zero {
                return Ok((xvec, multiple_scattering_order));
            }
            bb = recurrence_norm.sqrt();
            bbc = bb.conj();
            fms_checked_nonzero(bb, "ggrm", "recursion norm")?;
            fms_checked_nonzero(bbc, "ggrm", "adjoint recursion norm")?;

            xketp = xket;
            xbrap = xbra;
            xket = tket.iter().map(|&value| value / bb).collect();
            xbra = tbra.iter().map(|&value| value / bbc).collect();

            tket = fms_matvec(system_matrix, &xket);
            multiple_scattering_order += 1;
            aa = fms_cdot(&xbra, &tket);
            aac = aa.conj();

            let alphac = fms_checked_divide(bb, betac, "ggrm", "alpha")?;
            for ((basis, &current), (matrix_basis, &matrix_current)) in zvec
                .iter_mut()
                .zip(xket.iter())
                .zip(svec.iter_mut().zip(tket.iter()))
            {
                *basis = current - alphac * *basis;
                *matrix_basis = matrix_current - alphac * *matrix_basis;
            }

            betac = aa - alphac * bb;
            fms_checked_nonzero(betac, "ggrm", "beta")?;
            yy = -alphac * yy;
            let gamma = fms_checked_divide(yy, betac, "ggrm", "gamma")?;
            for ((solution, residual), (&basis, &matrix_basis)) in xvec
                .iter_mut()
                .zip(rvec.iter_mut())
                .zip(zvec.iter().zip(svec.iter()))
            {
                *solution += gamma * basis;
                *residual += gamma * matrix_basis;
            }

            if fms_vector_within_tolerance(&rvec, tolerance) {
                return Ok((xvec, multiple_scattering_order));
            }
        }
    }

    Err(FmsError::IterativeSolverNoConvergence {
        solver: "ggrm",
        restarts: MAX_RESTARTS,
    })
}

pub(super) fn fms_graves_morris_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    const MAX_RESTARTS: usize = 128;
    const MAX_ITERATIONS: usize = 10;

    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let one = Complex32::new(1.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];
    let mut bvec = vec![zero; state_count];
    let mut x0 = vec![zero; state_count];
    let mut q0 = one;
    bvec[source_state] = one;

    for restart in 0..MAX_RESTARTS {
        if restart > 0 {
            fms_checked_nonzero(q0, "gggm", "restart q0")?;
            for (solution, &basis) in xvec.iter_mut().zip(x0.iter()) {
                *solution += basis / q0;
            }
            let avec = fms_matvec(system_matrix, &xvec);
            for ((rhs, &matrix_solution), &solution) in
                bvec.iter_mut().zip(avec.iter()).zip(xvec.iter())
            {
                *rhs = matrix_solution - solution;
            }
            bvec[source_state] += one;
        }

        let mut r0 = bvec.clone();
        x0.fill(zero);
        let mut x1 = bvec.clone();
        let mut r1 = fms_matvec(system_matrix, &bvec);
        multiple_scattering_order += 1;

        let mut ww = fms_cdot(&r0, &r0);
        let mut aa = fms_cdot(&r1, &r1);
        let wa = fms_cdot(&r0, &r1);
        let aw = wa.conj();
        fms_checked_nonzero(aa, "gggm", "r1 norm")?;
        fms_checked_nonzero(ww, "gggm", "r0 norm")?;
        let dd = aa * ww - aw * wa;
        let scaled_dd = fms_checked_divide(
            fms_checked_divide(dd, aa, "gggm", "dd/aa")?,
            ww,
            "gggm",
            "dd/ww",
        )?;
        let wvec = if scaled_dd.norm() < 1.0e-8 {
            r0.iter().map(|&value| value / ww).collect::<Vec<_>>()
        } else {
            fms_checked_nonzero(dd, "gggm", "Gram determinant")?;
            ww = (ww - aw) / dd;
            aa = (wa - aa) / dd;
            r0.iter()
                .zip(r1.iter())
                .map(|(&current, &matrix_current)| current * aa + matrix_current * ww)
                .collect::<Vec<_>>()
        };

        let mut e0 = fms_cdot(&wvec, &r0);
        let mut e1 = fms_cdot(&wvec, &r1);
        q0 = one;
        let mut q1 = one;

        for _ in 0..MAX_ITERATIONS {
            let tol = fms_scaled_tolerance(tolerance, q1.norm() / 10.0, "gggm", "r1 tolerance")?;
            if fms_vector_within_tolerance(&r1, tol) {
                fms_checked_nonzero(q1, "gggm", "q1")?;
                for (solution, &basis) in xvec.iter_mut().zip(x1.iter()) {
                    *solution += basis / q1;
                }
                return Ok((xvec, multiple_scattering_order));
            }

            let alpha = fms_checked_divide(e1, e0, "gggm", "alpha")?;
            let mut t0 = r1
                .iter()
                .zip(r0.iter())
                .map(|(&current, &previous)| current - alpha * previous)
                .collect::<Vec<_>>();
            let t1 = fms_matvec(system_matrix, &t0);
            multiple_scattering_order += 1;

            let wa = fms_cdot(&t0, &t1);
            let ww = fms_cdot(&t0, &t0);
            let aa = fms_cdot(&t1, &t1);
            let aw = wa.conj();
            let theta = fms_checked_divide(wa - aa, ww - aw, "gggm", "theta")?;

            for ((residual, &matrix_basis), &basis) in r0.iter_mut().zip(t1.iter()).zip(t0.iter()) {
                *residual = matrix_basis - theta * basis;
            }
            let dd = one - theta;
            for ((basis, &current), &previous) in x0.iter_mut().zip(t0.iter()).zip(x1.iter()) {
                *basis = current + dd * (previous - alpha * *basis);
            }
            q0 = dd * (q1 - alpha * q0);
            let tol = fms_scaled_tolerance(tolerance, q0.norm(), "gggm", "r0 tolerance")?;
            if fms_vector_within_tolerance(&r0, tol) {
                fms_checked_nonzero(q0, "gggm", "q0")?;
                for (solution, &basis) in xvec.iter_mut().zip(x0.iter()) {
                    *solution += basis / q0;
                }
                return Ok((xvec, multiple_scattering_order));
            }

            e0 = fms_cdot(&wvec, &r0);
            let beta = fms_checked_divide(e0, e1, "gggm", "beta")?;
            for ((basis, &current), &previous) in t0.iter_mut().zip(r0.iter()).zip(r1.iter()) {
                *basis = current - beta * previous;
            }
            let avec = fms_matvec(system_matrix, &t0);
            multiple_scattering_order += 1;
            let dd = beta * theta;
            for (residual, &matrix_basis) in r1.iter_mut().zip(avec.iter()) {
                *residual = matrix_basis + dd * *residual;
            }
            e1 = fms_cdot(&wvec, &r1);

            let dd = beta * (one - theta);
            for ((basis, &current), &correction) in x1.iter_mut().zip(x0.iter()).zip(t0.iter()) {
                *basis = current - dd * *basis + correction;
            }
            q1 = q0 - (one - theta) * beta * q1;
        }
    }

    Err(FmsError::IterativeSolverNoConvergence {
        solver: "gggm",
        restarts: MAX_RESTARTS,
    })
}

pub(super) fn fms_tfqmr_solve(
    system_matrix: ArrayView2<'_, Complex32>,
    source_state: usize,
    tolerance: f32,
) -> Result<(Vec<Complex32>, usize), FmsError> {
    const MAX_RESTARTS: usize = 128;
    let state_count = system_matrix.shape()[0];
    ensure_axis_len("g0t", "source_state", state_count, source_state)?;
    let zero = Complex32::new(0.0, 0.0);
    let mut multiple_scattering_order = 0;
    let mut xvec = vec![zero; state_count];
    let mut avec = vec![zero; state_count];

    for restart in 0..MAX_RESTARTS {
        if restart > 0 {
            avec = fms_matvec(system_matrix, &xvec);
        }
        let mut uvec = avec.iter().map(|&value| -value).collect::<Vec<_>>();
        uvec[source_state] += Complex32::new(1.0, 0.0);
        avec = fms_matvec(system_matrix, &uvec);
        multiple_scattering_order += 1;

        let mut wvec = uvec.clone();
        let mut vvec = avec.clone();
        let mut dvec = vec![zero; state_count];
        let aa = fms_cdot(&uvec, &uvec);
        fms_checked_nonzero(aa, "ggtf", "initial residual norm")?;
        let mut tau = fms_checked_positive_real(aa.re, "ggtf", "tau")?.sqrt();
        let mut nu = 0.0;
        let mut eta = zero;
        let rvec = uvec.iter().map(|&value| value / aa).collect::<Vec<_>>();
        let mut rho = Complex32::new(1.0, 0.0);
        let mut alpha = zero;

        for nit in 0..=20 {
            if nit % 2 == 0 {
                let aa = fms_cdot(&rvec, &vvec);
                alpha = fms_checked_divide(rho, aa, "ggtf", "alpha")?;
            } else {
                avec = fms_matvec(system_matrix, &uvec);
                multiple_scattering_order += 1;
            }

            for (w, &matrix_direction) in wvec.iter_mut().zip(avec.iter()) {
                *w -= alpha * matrix_direction;
            }
            let aa = fms_checked_divide((nu * nu) * eta, alpha, "ggtf", "dvec factor")?;
            let previous_dvec = dvec.clone();
            for ((direction, &basis), &previous) in
                dvec.iter_mut().zip(uvec.iter()).zip(previous_dvec.iter())
            {
                *direction = basis + aa * previous;
            }
            let aa = fms_cdot(&wvec, &wvec);
            let norm = fms_checked_nonnegative_real(aa.re, "ggtf", "wvec norm")?.sqrt();
            nu = norm / tau;
            let cm = 1.0 / (1.0 + nu * nu).sqrt();
            tau *= nu * cm;
            eta = (cm * cm) * alpha;
            for (solution, &direction) in xvec.iter_mut().zip(dvec.iter()) {
                *solution += eta * direction;
            }

            let err = tau * (((1.0 + nit as f32) / state_count as f32).sqrt()) * 10.0;
            if err.abs() < tolerance {
                return Ok((xvec, multiple_scattering_order));
            }

            if nit % 2 != 0 {
                let previous_rho = rho;
                rho = fms_cdot(&rvec, &wvec);
                let beta = fms_checked_divide(rho, previous_rho, "ggtf", "beta")?;
                for (basis, &shadow) in uvec.iter_mut().zip(wvec.iter()) {
                    *basis = shadow + beta * *basis;
                }
                for (matrix_direction, &current) in vvec.iter_mut().zip(avec.iter()) {
                    *matrix_direction = beta * (current + beta * *matrix_direction);
                }
                avec = fms_matvec(system_matrix, &uvec);
                multiple_scattering_order += 1;
                for (matrix_direction, &current) in vvec.iter_mut().zip(avec.iter()) {
                    *matrix_direction += current;
                }
            } else {
                for (basis, &matrix_direction) in uvec.iter_mut().zip(vvec.iter()) {
                    *basis -= alpha * matrix_direction;
                }
            }
        }
    }

    Err(FmsError::IterativeSolverNoConvergence {
        solver: "ggtf",
        restarts: MAX_RESTARTS,
    })
}

pub(super) fn fms_vector_within_tolerance(vector: &[Complex32], tolerance: f32) -> bool {
    vector
        .iter()
        .all(|value| value.re.abs() <= tolerance && value.im.abs() <= tolerance)
}

pub(super) fn fms_scaled_tolerance(
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

pub(super) fn fms_cdot(left: &[Complex32], right: &[Complex32]) -> Complex32 {
    left.iter()
        .zip(right.iter())
        .map(|(&bra, &ket)| bra.conj() * ket)
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

pub(super) fn fms_matvec(
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

pub(super) fn fms_adjoint_matvec(
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

pub(super) fn fms_checked_divide(
    numerator: Complex32,
    denominator: Complex32,
    solver: &'static str,
    step: &'static str,
) -> Result<Complex32, FmsError> {
    fms_checked_nonzero(denominator, solver, step)?;
    Ok(numerator / denominator)
}

pub(super) fn fms_checked_nonzero(
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

pub(super) fn fms_checked_positive_real(
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

pub(super) fn fms_checked_nonnegative_real(
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

pub(super) fn fms_lu_system_matrix(
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

pub(super) fn fms_full_potential_lu_system_matrix(
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

pub(super) fn fms_spin_partner_index(
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

pub(super) fn ensure_square_table(
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

pub(super) fn potential_lmax_for(
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

pub(super) fn representative_offset(
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

pub(super) fn clamp_fms_lipotx(value: i32, global_lmax: usize) -> usize {
    if value < 0 {
        global_lmax
    } else {
        usize::try_from(value).map_or(global_lmax, |lmax| lmax.min(global_lmax))
    }
}

pub(super) fn fms_state_ket_error(error: StateKetError) -> FmsError {
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

pub(super) fn sort_radius_key(index: usize, atom: FmsAtom) -> Result<f64, FmsError> {
    ensure_finite_position(index, atom.position)?;
    Ok(f64::from(atom.position[0]) * f64::from(atom.position[0])
        + f64::from(atom.position[1]) * f64::from(atom.position[1])
        + f64::from(atom.position[2]) * f64::from(atom.position[2])
        + (index as f64 + 1.0) * 1.0e-6)
}

pub(super) fn checked_potential(potential: i32, max_potential: usize) -> Result<usize, FmsError> {
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

pub(super) fn checked_phase_potential(
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

pub(super) fn checked_position(positions: &[[f32; 3]], index: usize) -> Result<[f32; 3], FmsError> {
    let position = positions
        .get(index)
        .copied()
        .ok_or(FmsError::AtomIndexOutOfRange {
            index,
            len: positions.len(),
        })?;
    ensure_finite_position(index, position)?;
    Ok(position)
}

pub(super) fn ensure_finite_position(atom: usize, position: [f32; 3]) -> Result<(), FmsError> {
    for (axis, value) in position.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(FmsError::NonFiniteCoordinate { atom, axis });
        }
    }
    Ok(())
}

pub(super) fn validate_rotation_limits(lmax: usize, mmax: usize) -> Result<(), FmsError> {
    if lmax > FMS_ROTATION_LMAX {
        return Err(FmsError::InvalidAngularLimit {
            name: "lmax",
            value: lmax,
            lx: FMS_ROTATION_LMAX,
        });
    }
    if mmax > lmax {
        return Err(FmsError::InvalidAngularLimit {
            name: "mmax",
            value: mmax,
            lx: lmax,
        });
    }
    Ok(())
}

pub(super) fn copy_rotation_table(
    source: &ArrayView3<'_, Complex32>,
    target: &mut Array6<Complex32>,
    atom2: usize,
    atom1: usize,
    direction: FmsRotationDirection,
) {
    let branch = match direction {
        FmsRotationDirection::Forward => 0,
        FmsRotationDirection::Backward => 1,
    };
    for angular_momentum in 0..source.shape()[2] {
        for magnetic_one in 0..source.shape()[1] {
            for magnetic_two in 0..source.shape()[0] {
                target[(
                    magnetic_two,
                    magnetic_one,
                    angular_momentum,
                    branch,
                    atom2,
                    atom1,
                )] = source[(magnetic_two, magnetic_one, angular_momentum)];
            }
        }
    }
}

pub(super) fn checked_atom_index(atom: usize) -> Result<usize, FmsError> {
    atom.checked_sub(1)
        .ok_or(FmsError::InvalidStateAtom { atom })
}

pub(super) fn ensure_atom_table_index(index: usize, len: usize) -> Result<(), FmsError> {
    if index < len {
        Ok(())
    } else {
        Err(FmsError::AtomIndexOutOfRange { index, len })
    }
}

pub(super) fn ensure_axis_len(
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

pub(super) fn normalization_value(
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

pub(super) fn angular_weight(angular_momentum: usize) -> Result<Complex32, FmsError> {
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

pub(super) fn odd_factor(index: usize, lx: usize) -> Result<Complex32, FmsError> {
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
