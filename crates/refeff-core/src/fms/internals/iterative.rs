use super::*;

pub(in crate::fms) struct FmsIterativeScatteringInput<'a> {
    pub(in crate::fms) states: &'a [StateKet],
    pub(in crate::fms) spin_channels: usize,
    pub(in crate::fms) global_lmax: usize,
    pub(in crate::fms) potential_lmax: &'a [usize],
    pub(in crate::fms) representative_offsets: &'a [Option<usize>],
    pub(in crate::fms) potential_start: usize,
    pub(in crate::fms) potential_end: usize,
    pub(in crate::fms) free_propagator: ArrayView2<'a, Complex32>,
    pub(in crate::fms) t_matrix: ArrayView2<'a, Complex32>,
    pub(in crate::fms) calculated_l: &'a [bool],
    pub(in crate::fms) convergence_tolerance: f32,
    pub(in crate::fms) zero_tolerance: f32,
}

pub(in crate::fms) struct FmsIterativeScatteringResult {
    pub(in crate::fms) system_matrix: Array2<Complex32>,
    pub(in crate::fms) scattering: Array3<Complex32>,
    pub(in crate::fms) multiple_scattering_order: usize,
}

pub(in crate::fms) fn fms_iterative_scattering(
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

pub(in crate::fms) fn fms_iterative_scattering_with_system(
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

pub(in crate::fms) fn fms_bicgstab_solve(
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

pub(in crate::fms) fn fms_recursion_solve(
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

pub(in crate::fms) fn fms_graves_morris_solve(
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

pub(in crate::fms) fn fms_tfqmr_solve(
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
