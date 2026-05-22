use super::*;

/// Port of FEFF `bcoef`: build the energy-independent transition B matrix.
///
/// The returned matrix keeps FEFF's axis order
/// `bmat(ml2, ms2, k2, ml1, ms1, k1)`, with signed magnetic quantum numbers
/// shifted by `l_offset`. Transition slots are zero-based in the raw ndarray
/// but correspond to FEFF's slots `1..=8`.
pub fn transition_b_matrix(
    input: TransitionBMatrixInput,
) -> Result<TransitionBMatrix, AngularError> {
    if input.spin_channels == 0 {
        return Err(AngularError::InvalidSpinChannelCount {
            value: input.spin_channels,
        });
    }
    ensure_finite_polarization_tensor(&input.polarization_tensor)?;

    let lmax_i32 = usize_to_i32(input.lmax)?;
    let magnetic_len = input
        .lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AngularError::IndexTooLarge { value: input.lmax })?;
    let extended_magnetic_len = magnetic_len
        .checked_add(1)
        .ok_or(AngularError::IndexTooLarge { value: input.lmax })?;

    let mut jind = [0_i32; 8];
    let mut lind = [0_i32; 8];
    let mut kiind = [0_i32; 8];
    fill_transition_indices(
        input.initial_kappa,
        input.multipole,
        lmax_i32,
        &mut jind,
        &mut lind,
        &mut kiind,
    );

    let mut matrix = Array6::zeros((magnetic_len, 2, 8, magnetic_len, 2, 8).f());
    if input.polarization == 0 {
        fill_averaged_transition_matrix(input.multipole, lmax_i32, &lind, &mut matrix);
    } else {
        fill_polarized_transition_matrix(
            input,
            lmax_i32,
            magnetic_len,
            extended_magnetic_len,
            &jind,
            &lind,
            &mut matrix,
        )?;
    }

    if input.trace_orbital {
        trace_transition_matrix(lmax_i32, &lind, &mut matrix);
    }
    fold_transition_spin(
        input.spin,
        input.spin_channels,
        lmax_i32,
        &lind,
        &mut matrix,
    );

    Ok(TransitionBMatrix {
        kappa_indices: kiind,
        orbital_momenta: lind,
        matrix,
        l_offset: input.lmax,
    })
}

fn ensure_finite_polarization_tensor(tensor: &[[Complex; 3]; 3]) -> Result<(), AngularError> {
    for (row, values) in tensor.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(AngularError::NonFinitePolarizationTensor {
                    row: row as isize - 1,
                    column: column as isize - 1,
                });
            }
        }
    }
    Ok(())
}

fn fill_transition_indices(
    initial_kappa: i32,
    multipole: i32,
    lmax: i32,
    jind: &mut [i32; 8],
    lind: &mut [i32; 8],
    kiind: &mut [i32; 8],
) {
    for k in -1_i32..=1 {
        let mut kappa = initial_kappa + k;
        if k == 0 {
            kappa = -kappa;
        }
        let mut jkap = kappa.abs();
        let mut lkap = if kappa <= 0 { kappa.abs() - 1 } else { kappa };
        if lkap > lmax {
            jkap = 0;
            lkap = -1;
            kappa = 0;
        }
        let index = (k + 1) as usize;
        jind[index] = jkap;
        lind[index] = lkap;
        kiind[index] = kappa;
    }

    for k in -2_i32..=2 {
        let mut jkap = initial_kappa.abs() + k;
        if jkap <= 0 {
            jkap = 0;
        }
        let mut kappa = jkap;
        if initial_kappa < 0 && k.abs() != 1 {
            kappa = -jkap;
        }
        if initial_kappa > 0 && k.abs() == 1 {
            kappa = -jkap;
        }
        let mut lkap = if kappa <= 0 { -kappa - 1 } else { kappa };
        if lkap > lmax || multipole == 0 || (multipole == 1 && k.abs() == 2) {
            jkap = 0;
            lkap = -1;
            kappa = 0;
        }
        let index = (k + 5) as usize;
        jind[index] = jkap;
        lind[index] = lkap;
        kiind[index] = kappa;
    }
}

fn fill_averaged_transition_matrix(
    multipole: i32,
    lmax: i32,
    lind: &[i32; 8],
    matrix: &mut Array6<Complex>,
) {
    for (transition, &l) in lind.iter().enumerate() {
        if l < 0 {
            continue;
        }
        let multipole_degeneracy = if multipole == 2 && transition >= 3 {
            5.0
        } else {
            3.0
        };
        let mut value = 0.5 / (2.0 * f64::from(l) + 1.0) / multipole_degeneracy;
        if transition < 3 {
            value = -value;
        }
        for spin in 0..=1 {
            for magnetic in -l..=l {
                let index = magnetic_index_i32(magnetic, lmax);
                matrix[[index, spin, transition, index, spin, transition]] =
                    Complex::new(value, 0.0);
            }
        }
    }
}

fn fill_polarized_transition_matrix(
    input: TransitionBMatrixInput,
    lmax: i32,
    magnetic_len: usize,
    extended_magnetic_len: usize,
    jind: &[i32; 8],
    lind: &[i32; 8],
    matrix: &mut Array6<Complex>,
) -> Result<(), AngularError> {
    let mut t3j = vec![0.0; 8 * 2 * extended_magnetic_len];
    let mut x3j = vec![0.0; 8 * 3 * extended_magnetic_len];
    let mut qmat = vec![0.0; extended_magnetic_len * magnetic_len * 2 * 8];
    let mut pmat =
        vec![Complex::new(0.0, 0.0); extended_magnetic_len * 8 * extended_magnetic_len * 8];
    let mut tmat = vec![Complex::new(0.0, 0.0); extended_magnetic_len * 8 * magnetic_len * 2 * 8];

    for transition in 0..8 {
        let j = jind[transition];
        if j <= 0 {
            continue;
        }
        for mp in -j + 1..=j {
            for spin in 0..=1 {
                let j1 = 2 * lind[transition];
                let j2 = 1;
                let j3 = 2 * j - 1;
                let m1 = 2 * (mp - spin as i32);
                let m2 = 2 * spin as i32 - 1;
                let mut value = f64::from(j3 + 1).sqrt() * wigner_3j(j1, j2, j3, m1, m2, 2)?;
                if ((j2 - j1 - m1 - m2) / 2) % 2 != 0 {
                    value = -value;
                }
                let index = t3j_index(transition, spin, mp, lmax, extended_magnetic_len);
                t3j[index] = value;
            }

            for polarization in -1..=1 {
                let j1 = 2 * j - 1;
                let j2 = if transition >= 3 && input.multipole == 2 {
                    4
                } else {
                    2
                };
                let j3 = 2 * input.initial_kappa.abs() - 1;
                let m1 = -2 * mp + 1;
                let m2 = 2 * polarization;
                let index = x3j_index(transition, polarization, mp, lmax, extended_magnetic_len);
                x3j[index] = wigner_3j(j1, j2, j3, m1, m2, 2)?;
            }
        }
    }

    for transition in 0..8 {
        let l = lind[transition];
        let j = jind[transition];
        if l < 0 || j <= 0 {
            continue;
        }
        for spin in 0..=1 {
            for ml in -l..=l {
                for mj in -j + 1..=j {
                    let mp = ml + spin as i32;
                    let jj = 2 * j - 1;
                    let mmj = 2 * mj - 1;
                    let mmp = 2 * mp - 1;
                    let rotation = wigner_rotation(input.spin_vector_angle, jj, mmj, mmp, 2)?;
                    let t3j_value =
                        t3j[t3j_index(transition, spin, mp, lmax, extended_magnetic_len)];
                    let index = qmat_index(mj, ml, spin, transition, lmax, magnetic_len);
                    qmat[index] = rotation * t3j_value;
                }
            }
        }
    }

    for transition2 in 0..8 {
        let j2 = jind[transition2];
        if j2 <= 0 {
            continue;
        }
        for m2 in -j2 + 1..=j2 {
            for transition1 in 0..8 {
                let j1 = jind[transition1];
                if j1 <= 0 {
                    continue;
                }
                for m1 in -j1 + 1..=j1 {
                    if (m2 - m1).abs() <= 2 {
                        let mut value = Complex::new(0.0, 0.0);
                        for p_prime in -1..=1 {
                            for p in -1..=1 {
                                if m1 - p == m2 - p_prime {
                                    let mut sign = 1.0;
                                    if input.multipole == 1 && p > 0 && transition1 >= 3 {
                                        sign = -sign;
                                    }
                                    if input.multipole == 1 && p_prime > 0 && transition2 >= 3 {
                                        sign = -sign;
                                    }
                                    value += input.polarization_tensor[(p + 1) as usize]
                                        [(p_prime + 1) as usize]
                                        * (sign
                                            * x3j[x3j_index(
                                                transition1,
                                                p,
                                                m1,
                                                lmax,
                                                extended_magnetic_len,
                                            )]
                                            * x3j[x3j_index(
                                                transition2,
                                                p_prime,
                                                m2,
                                                lmax,
                                                extended_magnetic_len,
                                            )]);
                                }
                            }
                        }

                        let mut sign = 1.0;
                        if (jind[transition1] - jind[transition2]) % 2 != 0 {
                            sign = -sign;
                        }
                        if transition2 < 3 {
                            sign = -sign;
                        }
                        value *= sign * imaginary_unit_power(lind[transition2] - lind[transition1]);
                        let index = pmat_index(
                            m1,
                            transition1,
                            m2,
                            transition2,
                            lmax,
                            extended_magnetic_len,
                        );
                        pmat[index] = value;
                    }
                }
            }
        }
    }

    for transition1 in 0..8 {
        let l1 = lind[transition1];
        let j1 = jind[transition1];
        if l1 < 0 || j1 <= 0 {
            continue;
        }
        for spin in 0..=1 {
            for ml in -l1..=l1 {
                for transition2 in 0..8 {
                    let j2 = jind[transition2];
                    if j2 <= 0 {
                        continue;
                    }
                    for mj in -j2 + 1..=j2 {
                        let mut value = Complex::new(0.0, 0.0);
                        for mp in -j1 + 1..=j1 {
                            value += pmat[pmat_index(
                                mj,
                                transition2,
                                mp,
                                transition1,
                                lmax,
                                extended_magnetic_len,
                            )] * qmat
                                [qmat_index(mp, ml, spin, transition1, lmax, magnetic_len)];
                        }
                        let index =
                            tmat_index(mj, transition2, ml, spin, transition1, lmax, magnetic_len);
                        tmat[index] = value;
                    }
                }
            }
        }
    }

    for transition1 in 0..8 {
        let l1 = lind[transition1];
        if l1 < 0 {
            continue;
        }
        for spin1 in 0..=1 {
            for ml1 in -l1..=l1 {
                for transition2 in 0..8 {
                    let l2 = lind[transition2];
                    let j2 = jind[transition2];
                    if l2 < 0 || j2 <= 0 {
                        continue;
                    }
                    for spin2 in 0..=1 {
                        for ml2 in -l2..=l2 {
                            let mut value = Complex::new(0.0, 0.0);
                            for mj in -j2 + 1..=j2 {
                                value += qmat
                                    [qmat_index(mj, ml2, spin2, transition2, lmax, magnetic_len)]
                                    * tmat[tmat_index(
                                        mj,
                                        transition2,
                                        ml1,
                                        spin1,
                                        transition1,
                                        lmax,
                                        magnetic_len,
                                    )];
                            }
                            let ml2_index = magnetic_index_i32(ml2, lmax);
                            let ml1_index = magnetic_index_i32(ml1, lmax);
                            matrix
                                [[ml2_index, spin2, transition2, ml1_index, spin1, transition1]] =
                                value;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn trace_transition_matrix(lmax: i32, lind: &[i32; 8], matrix: &mut Array6<Complex>) {
    let zero_index = magnetic_index_i32(0, lmax);
    for transition1 in 0..8 {
        for spin1 in 0..=1 {
            for transition2 in 0..8 {
                for spin2 in 0..=1 {
                    if lind[transition1] != lind[transition2] || spin1 != spin2 {
                        matrix[[
                            zero_index,
                            spin2,
                            transition2,
                            zero_index,
                            spin1,
                            transition1,
                        ]] = Complex::new(0.0, 0.0);
                    } else {
                        let l = lind[transition1];
                        for magnetic in 1..=l {
                            let negative = magnetic_index_i32(-magnetic, lmax);
                            let positive = magnetic_index_i32(magnetic, lmax);
                            let addition = matrix
                                [[negative, spin1, transition2, negative, spin1, transition1]]
                                + matrix
                                    [[positive, spin1, transition2, positive, spin1, transition1]];
                            matrix[[
                                zero_index,
                                spin1,
                                transition2,
                                zero_index,
                                spin1,
                                transition1,
                            ]] += addition;
                        }
                    }
                }
            }
        }
    }
}

fn fold_transition_spin(
    spin: i32,
    spin_channels: usize,
    lmax: i32,
    lind: &[i32; 8],
    matrix: &mut Array6<Complex>,
) {
    if spin == 0 {
        for transition1 in 0..8 {
            let l1 = lind[transition1];
            if l1 < 0 {
                continue;
            }
            for transition2 in 0..8 {
                let l2 = lind[transition2];
                if l2 < 0 {
                    continue;
                }
                for ml1 in -l1..=l1 {
                    for ml2 in -l2..=l2 {
                        let ml1_index = magnetic_index_i32(ml1, lmax);
                        let ml2_index = magnetic_index_i32(ml2, lmax);
                        let addition =
                            matrix[[ml2_index, 1, transition2, ml1_index, 1, transition1]];
                        matrix[[ml2_index, 0, transition2, ml1_index, 0, transition1]] += addition;
                    }
                }
            }
        }
    } else if spin == 2 || (spin == 1 && spin_channels == 1) {
        for transition1 in 0..8 {
            let l1 = lind[transition1];
            if l1 < 0 {
                continue;
            }
            for transition2 in 0..8 {
                let l2 = lind[transition2];
                if l2 < 0 {
                    continue;
                }
                for ml1 in -l1..=l1 {
                    for ml2 in -l2..=l2 {
                        let ml1_index = magnetic_index_i32(ml1, lmax);
                        let ml2_index = magnetic_index_i32(ml2, lmax);
                        matrix[[ml2_index, 0, transition2, ml1_index, 0, transition1]] =
                            matrix[[ml2_index, 1, transition2, ml1_index, 1, transition1]];
                    }
                }
            }
        }
    }
}

fn t3j_index(
    transition: usize,
    spin: usize,
    magnetic: i32,
    lmax: i32,
    extended_magnetic_len: usize,
) -> usize {
    (transition * 2 + spin) * extended_magnetic_len + extended_magnetic_index_i32(magnetic, lmax)
}

fn x3j_index(
    transition: usize,
    polarization: i32,
    magnetic: i32,
    lmax: i32,
    extended_magnetic_len: usize,
) -> usize {
    (transition * 3 + (polarization + 1) as usize) * extended_magnetic_len
        + extended_magnetic_index_i32(magnetic, lmax)
}

fn qmat_index(
    mj: i32,
    ml: i32,
    spin: usize,
    transition: usize,
    lmax: i32,
    magnetic_len: usize,
) -> usize {
    (((extended_magnetic_index_i32(mj, lmax) * magnetic_len + magnetic_index_i32(ml, lmax)) * 2
        + spin)
        * 8)
        + transition
}

fn pmat_index(
    m1: i32,
    transition1: usize,
    m2: i32,
    transition2: usize,
    lmax: i32,
    extended_magnetic_len: usize,
) -> usize {
    (((extended_magnetic_index_i32(m1, lmax) * 8 + transition1) * extended_magnetic_len
        + extended_magnetic_index_i32(m2, lmax))
        * 8)
        + transition2
}

fn tmat_index(
    mj: i32,
    transition2: usize,
    ml: i32,
    spin: usize,
    transition1: usize,
    lmax: i32,
    magnetic_len: usize,
) -> usize {
    ((((extended_magnetic_index_i32(mj, lmax) * 8 + transition2) * magnetic_len
        + magnetic_index_i32(ml, lmax))
        * 2
        + spin)
        * 8)
        + transition1
}

fn magnetic_index_i32(magnetic: i32, lmax: i32) -> usize {
    (magnetic + lmax) as usize
}

fn extended_magnetic_index_i32(magnetic: i32, lmax: i32) -> usize {
    (magnetic + lmax) as usize
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}
