use super::polynomial::alternating_sign;
use super::validation::*;
use super::*;

/// Build FEFF `rot3i` real rotation matrices for a single path leg.
///
/// The recursion is the Edmonds small-`d` rotation used by FEFF before GENFMT
/// matrix assembly. FEFF writes into a globally padded `dri` array; this helper
/// returns only the active magnetic range `-(mxp1-1)..=(mxp1-1)` for each
/// `il`, with zeroes retained where FEFF would not fill entries.
pub fn initial_state_rotation(
    input: InitialStateRotationInput,
) -> Result<InitialStateRotation, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;
    if !input.beta_angle.is_finite() {
        return Err(GenfmtError::NonFiniteRotationAngle);
    }

    let magnetic_offset = input.mmaxp1 - 1;
    let m_dim = checked_double_plus_one("mmaxp1", magnetic_offset)?;
    let mut matrix = Array3::<Real>::zeros((input.lmaxp1, m_dim, m_dim).f());

    let work_l = input.lmaxp1.max(2);
    let ndm = input
        .lmaxp1
        .checked_add(input.mmaxp1)
        .and_then(|value| value.checked_sub(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?;
    let work_m = checked_double_plus_one("lmaxp1", work_l)?
        .checked_sub(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?
        .max(ndm)
        .max(3);
    let mut work = Array3::<Real>::zeros((work_l + 1, work_m + 1, work_m + 1).f());
    fill_initial_state_rotation_work(input.lmaxp1, input.mmaxp1, input.beta_angle, &mut work);
    copy_initial_state_rotation(
        input.lmaxp1,
        input.mmaxp1,
        magnetic_offset,
        &work,
        &mut matrix,
    )?;

    Ok(InitialStateRotation {
        matrix,
        magnetic_offset,
    })
}

fn fill_initial_state_rotation_work(
    lmaxp1: usize,
    mmaxp1: usize,
    beta: Real,
    work: &mut Array3<Real>,
) {
    let ndm = lmaxp1 + mmaxp1 - 1;
    let half_beta = beta / 2.0;
    let xc = half_beta.cos();
    let xs = half_beta.sin();
    let s = beta.sin();

    work[(1, 1, 1)] = 1.0;
    work[(2, 1, 1)] = xc * xc;
    work[(2, 1, 2)] = s / 2.0_f64.sqrt();
    work[(2, 1, 3)] = xs * xs;
    work[(2, 2, 1)] = -work[(2, 1, 2)];
    work[(2, 2, 2)] = beta.cos();
    work[(2, 2, 3)] = work[(2, 1, 2)];
    work[(2, 3, 1)] = work[(2, 1, 3)];
    work[(2, 3, 2)] = -work[(2, 2, 3)];
    work[(2, 3, 3)] = work[(2, 1, 1)];

    for l in 3..=lmaxp1 {
        let ln = (2 * l - 1).min(ndm);
        let lm = (2 * l - 3).min(ndm);
        for n in 1..=ln {
            for m in 1..=lm {
                let l_signed = l as isize;
                let n_signed = n as isize;
                let m_signed = m as isize;
                let t1 = ((2 * l_signed - 1 - n_signed) * (2 * l_signed - 2 - n_signed)) as Real;
                let t = ((2 * l_signed - 1 - m_signed) * (2 * l_signed - 2 - m_signed)) as Real;
                let f1 = (t1 / t).sqrt();
                let f2 = (((2 * l_signed - 1 - n_signed) * (n_signed - 1)) as Real / t).sqrt();
                let f3 = if n > 2 {
                    (((n - 2) * (n - 1)) as Real / t).sqrt()
                } else {
                    0.0
                };

                let mut dlnm = f1 * xc * xc * work[(l - 1, n, m)];
                if n > 1 {
                    dlnm -= f2 * s * work[(l - 1, n - 1, m)];
                }
                if n > 2 {
                    dlnm += f3 * xs * xs * work[(l - 1, n - 2, m)];
                }
                work[(l, n, m)] = dlnm;

                if n > 2 * l - 3 {
                    work[(l, m, n)] = alternating_sign(n - m) * dlnm;
                }
            }

            if n > 2 * l - 3 {
                work[(l, 2 * l - 2, 2 * l - 2)] = work[(l, 2, 2)];
                work[(l, 2 * l - 1, 2 * l - 2)] = -work[(l, 1, 2)];
                work[(l, 2 * l - 2, 2 * l - 1)] = -work[(l, 2, 1)];
                work[(l, 2 * l - 1, 2 * l - 1)] = work[(l, 1, 1)];
            }
        }
    }
}

fn copy_initial_state_rotation(
    lmaxp1: usize,
    mmaxp1: usize,
    magnetic_offset: usize,
    work: &Array3<Real>,
    matrix: &mut Array3<Real>,
) -> Result<(), GenfmtError> {
    let magnetic_offset =
        isize::try_from(magnetic_offset).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
        })?;

    for il in 1..=lmaxp1 {
        let mx = (il - 1).min(mmaxp1 - 1);
        let mx_signed = isize::try_from(mx).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
        })?;
        let il_signed = isize::try_from(il).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: lmaxp1,
        })?;

        for m1_slot in 0..=(2 * mx) {
            let m1 = isize::try_from(m1_slot).map_err(|_| GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: mmaxp1,
            })? - mx_signed;
            for m2_slot in 0..=(2 * mx) {
                let m2 =
                    isize::try_from(m2_slot).map_err(|_| GenfmtError::InvalidAngularLimit {
                        name: "mmaxp1",
                        value: mmaxp1,
                    })? - mx_signed;
                let row = shifted_index(m1, magnetic_offset, "mmaxp1", mmaxp1)?;
                let column = shifted_index(m2, magnetic_offset, "mmaxp1", mmaxp1)?;
                let work_row = shifted_index(m1, il_signed, "lmaxp1", lmaxp1)?;
                let work_column = shifted_index(m2, il_signed, "lmaxp1", lmaxp1)?;
                matrix[(il - 1, row, column)] = work[(il, work_row, work_column)];
            }
        }
    }
    Ok(())
}

fn shifted_index(
    value: isize,
    offset: isize,
    name: &'static str,
    limit: usize,
) -> Result<usize, GenfmtError> {
    let index = value
        .checked_add(offset)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value: limit })?;
    usize::try_from(index).map_err(|_| GenfmtError::InvalidAngularLimit { name, value: limit })
}
