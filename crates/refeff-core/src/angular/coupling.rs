use super::*;

/// Build FEFF `t3jp` and `t3jm` spin-orbit coupling tables.
pub fn spin_orbit_coupling_tables(lmax: usize) -> Result<SpinOrbitCouplingTables, AngularError> {
    let mut plus = Array3::zeros((lmax + 1, 2 * lmax + 1, 2).f());
    let mut minus = Array3::zeros((lmax + 1, 2 * lmax + 1, 2).f());
    let lmax_isize =
        isize::try_from(lmax).map_err(|_| AngularError::IndexTooLarge { value: lmax })?;

    for l in 0..=lmax {
        let l_i32 = usize_to_i32(l)?;
        let l_isize = isize::try_from(l).map_err(|_| AngularError::IndexTooLarge { value: l })?;
        for magnetic in -l_isize..=l_isize {
            let magnetic_i32 = isize_to_i32(magnetic)?;
            let magnetic_index = magnetic_table_index(magnetic, lmax)?;
            for spin_index in 0..2 {
                let spin_i32 = usize_to_i32(spin_index + 1)?;
                let j1 = 2 * l_i32;
                let j2 = 1;
                let j3p = j1 + 1;
                let j3m = j1 - 1;
                let m1 = 2 * magnetic_i32;
                let m2 = 2 * spin_i32 - 3;
                let sign = feff_spin_coupling_sign(j2, j1, m1, m2);

                plus[[l, magnetic_index, spin_index]] =
                    sign * f64::from(j3p + 1).sqrt() * wigner_3j(j1, j2, j3p, m1, m2, 2)?;
                minus[[l, magnetic_index, spin_index]] =
                    sign * f64::from(j3m + 1).sqrt() * wigner_3j(j1, j2, j3m, m1, m2, 2)?;
            }
        }
    }

    Ok(SpinOrbitCouplingTables {
        plus,
        minus,
        m_offset: usize::try_from(lmax_isize)
            .map_err(|_| AngularError::IndexTooLarge { value: lmax })?,
    })
}

/// Port of FEFF `KSPACE/calccgc.f90`.
///
/// Builds FEFF's linear `CGC(IKM, IS)` table for relativistic
/// Clebsch-Gordan coefficients. The branch tables mirror FEFF's `LTAB`,
/// `KAPTAB`, and `NMUETAB` initialization used by KSPACE basis transforms.
pub fn relativistic_clebsch_gordan_coefficients(
    lmax: usize,
) -> Result<RelativisticClebschGordanCoefficients, AngularError> {
    let branch_count = lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let coefficient_count = lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(2))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;

    let mut orbital_momentum = Vec::with_capacity(branch_count);
    let mut kappa = Vec::with_capacity(branch_count);
    let mut spin_multiplicity = Vec::with_capacity(branch_count);
    let mut coefficients = Array2::zeros((coefficient_count, 2).f());
    let mut state = 0usize;

    for branch in 1..=branch_count {
        let orbital = branch / 2;
        let branch_kappa = if branch.is_multiple_of(2) {
            usize_to_i32(orbital)?
        } else {
            -usize_to_i32(
                orbital
                    .checked_add(1)
                    .ok_or(AngularError::IndexTooLarge { value: orbital })?,
            )?
        };
        let multiplicity = usize::try_from(branch_kappa.unsigned_abs())
            .map_err(|_| AngularError::IndexTooLarge { value: branch })?
            .checked_mul(2)
            .ok_or(AngularError::IndexTooLarge { value: branch })?;

        orbital_momentum.push(orbital);
        kappa.push(branch_kappa);
        spin_multiplicity.push(multiplicity);

        let orbital_real = usize_to_real(orbital)?;
        let mut mue = -f64::from(branch_kappa.unsigned_abs()) - 0.5;
        let two_l_plus_one = 2.0 * orbital_real + 1.0;
        for _ in 0..multiplicity {
            mue += 1.0;
            if branch_kappa < 0 {
                coefficients[(state, 0)] = ((orbital_real - mue + 0.5) / two_l_plus_one).sqrt();
                coefficients[(state, 1)] = ((orbital_real + mue + 0.5) / two_l_plus_one).sqrt();
            } else {
                coefficients[(state, 0)] = ((orbital_real + mue + 0.5) / two_l_plus_one).sqrt();
                coefficients[(state, 1)] = -((orbital_real - mue + 0.5) / two_l_plus_one).sqrt();
            }
            state += 1;
        }
    }

    Ok(RelativisticClebschGordanCoefficients {
        coefficients,
        orbital_momentum,
        kappa,
        spin_multiplicity,
    })
}

/// Port of FEFF `MKGTR/calclbcoef.f90`.
///
/// The returned `clbcoef(im, ii, is, ll)` table keeps FEFF's axis order as
/// `(mj_lmax, j_lmax, 2, lmax + 1)` and uses Fortran-order storage. Rust
/// indices are zero-based, while `ii` still represents FEFF's one-based
/// half-integer final-state angular momentum slot.
pub fn mkgtr_clebsch_gordan_coefficients(
    lmax: usize,
    j_lmax: usize,
    mj_lmax: usize,
) -> Result<Array4<Real>, AngularError> {
    if j_lmax == 0 {
        return Err(AngularError::InvalidAngularTableDimension {
            name: "j_lmax",
            value: j_lmax,
            minimum: 1,
        });
    }

    let l_count = lmax
        .checked_add(1)
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let active_j_lmax = j_lmax.min(l_count);
    let required_mj_lmax =
        checked_double_usize(active_j_lmax).ok_or(AngularError::IndexTooLarge {
            value: active_j_lmax,
        })?;
    if mj_lmax < required_mj_lmax {
        return Err(AngularError::InvalidAngularTableDimension {
            name: "mj_lmax",
            value: mj_lmax,
            minimum: required_mj_lmax,
        });
    }

    let mut coefficients = Array4::zeros((mj_lmax, j_lmax, 2, l_count).f());
    for ll in 0..l_count {
        let lnow = checked_double_usize_to_i32(ll)?;
        let active_j = j_lmax.min(
            ll.checked_add(1)
                .ok_or(AngularError::IndexTooLarge { value: ll })?,
        );
        for is in 0..=1 {
            let ms = checked_double_usize_to_i32(is)?
                .checked_sub(1)
                .ok_or(AngularError::IndexTooLarge { value: is })?;
            for ii in 1..=active_j {
                let jnow = checked_double_usize_to_i32(ii)?
                    .checked_sub(1)
                    .ok_or(AngularError::IndexTooLarge { value: ii })?;
                for im in
                    1..=checked_double_usize(ii).ok_or(AngularError::IndexTooLarge { value: ii })?
                {
                    let im_i32 = usize_to_i32(im)?;
                    let mj = checked_double_i32(
                        im_i32
                            .checked_sub(1)
                            .ok_or(AngularError::IndexTooLarge { value: im })?,
                    )?
                    .checked_sub(jnow)
                    .ok_or(AngularError::IndexTooLarge { value: im })?;
                    let neg_mj = mj
                        .checked_neg()
                        .ok_or(AngularError::IndexTooLarge { value: im })?;
                    let mut coefficient = wigner_3j(1, jnow, lnow, ms, neg_mj, 2)?;
                    let sign_argument = lnow
                        .checked_add(mj)
                        .and_then(|value| value.checked_sub(1))
                        .ok_or(AngularError::IndexTooLarge { value: ll })?;
                    if (sign_argument / 2) % 2 != 0 {
                        coefficient = -coefficient;
                    }
                    coefficients[(im - 1, ii - 1, is, ll)] = coefficient;
                }
            }
        }
    }
    Ok(coefficients)
}

/// Port of FEFF `BAND/ikapmue.f90`: one-based `(kappa, MUEM05)` state index.
///
/// `mu_minus_half` is FEFF's integer `MUEM05`, the relativistic magnetic
/// quantum number shifted by `-1/2`. FEFF stores states in kappa-branch order
/// and uses `I = 2*L*(J+1/2) + J + MUEM05 + 1`; this helper returns that same
/// one-based index with explicit validation.
pub fn relativistic_state_index_1based(
    kappa: i32,
    mu_minus_half: i32,
) -> Result<usize, AngularError> {
    if kappa == 0 || kappa == i32::MIN {
        return Err(AngularError::InvalidRelativisticKappa { kappa });
    }

    let jp05 = kappa.abs();
    if mu_minus_half < -jp05 || mu_minus_half >= jp05 {
        return Err(AngularError::RelativisticMagneticIndexOutOfRange {
            kappa,
            mu_minus_half,
        });
    }

    let orbital = if kappa < 0 { -kappa - 1 } else { kappa };
    let index = 2_i128 * i128::from(orbital) * i128::from(jp05)
        + i128::from(jp05)
        + i128::from(mu_minus_half)
        + 1;
    usize::try_from(index).map_err(|_| AngularError::IndexTooLarge { value: usize::MAX })
}

fn checked_double_usize(value: usize) -> Option<usize> {
    value.checked_mul(2)
}

fn checked_double_usize_to_i32(value: usize) -> Result<i32, AngularError> {
    usize_to_i32(checked_double_usize(value).ok_or(AngularError::IndexTooLarge { value })?)
}

fn checked_double_i32(value: i32) -> Result<i32, AngularError> {
    value
        .checked_mul(2)
        .ok_or(AngularError::IndexTooLarge { value: usize::MAX })
}

fn isize_to_i32(value: isize) -> Result<i32, AngularError> {
    i32::try_from(value).map_err(|_| AngularError::MagneticIndexOutOfRange {
        magnetic: value,
        lmax: usize::MAX,
    })
}

fn magnetic_table_index(magnetic: isize, lmax: usize) -> Result<usize, AngularError> {
    let lmax_isize =
        isize::try_from(lmax).map_err(|_| AngularError::IndexTooLarge { value: lmax })?;
    let shifted = magnetic + lmax_isize;
    usize::try_from(shifted).map_err(|_| AngularError::MagneticIndexOutOfRange { magnetic, lmax })
}

fn feff_spin_coupling_sign(j2: i32, j1: i32, m1: i32, m2: i32) -> Real {
    let phase = (j2 - j1 - m1 - m2) / 2;
    if phase % 2 == 0 { 1.0 } else { -1.0 }
}
