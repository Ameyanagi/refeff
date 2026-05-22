use super::*;

/// Port of FEFF `XSPH/xmultjas.f90`.
///
/// Returns the longitudinal multipole prefactor for
/// `<k|exp(i*q*z)|k'>`. FEFF declares `xm` as `complex*16`, but this helper is
/// real-valued because `xmultjas` removes the `i**ls` phase and applies it in
/// the caller.
pub fn xsph_longitudinal_multipole_factor(
    kappa: i32,
    kappa_prime: i32,
    multipole_l: i32,
) -> Result<Complex, XsphError> {
    if kappa == 0 || kappa_prime == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if multipole_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            index: 0,
            value: multipole_l,
        });
    }

    let j2 = doubled_j_from_kappa("kappa", kappa)?;
    let parity = if kappa > 0 { -1 } else { 1 };
    let j2_prime = doubled_j_from_kappa("kappa_prime", kappa_prime)?;
    let parity_prime = if kappa_prime > 0 { -1 } else { 1 };
    let doubled_multipole = multipole_l
        .checked_mul(2)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole_l",
            value: multipole_l,
        })?;

    let parity_check =
        i64::from(j2) + i64::from(j2_prime) + i64::from(doubled_multipole) + i64::from(parity)
            - i64::from(parity_prime);
    let doubled_difference = (i64::from(j2) - i64::from(j2_prime)).abs();
    let doubled_sum = i64::from(j2) + i64::from(j2_prime);
    if parity_check.rem_euclid(4) == 0
        || i64::from(doubled_multipole) < doubled_difference
        || i64::from(doubled_multipole) > doubled_sum
    {
        return Ok(Complex::new(0.0, 0.0));
    }

    validate_cwig3j_doubled_argument("kappa", kappa, j2)?;
    validate_cwig3j_doubled_argument("kappa_prime", kappa_prime, j2_prime)?;
    validate_cwig3j_doubled_argument("multipole_l", multipole_l, doubled_multipole)?;
    let angular_weight = ((f64::from(j2) + 1.0) * (f64::from(j2_prime) + 1.0)).sqrt()
        * (f64::from(doubled_multipole) + 1.0);
    let value = angular_weight * wigner_3j(j2, doubled_multipole, j2_prime, 1, 0, 2)?;
    Ok(Complex::new(value, 0.0))
}

/// Port of FEFF `XSPH/xmult.f90`.
///
/// Returns FEFF's `xm1` and `xm2` angular prefactors for Grant equation 6.30,
/// used by `radint.f90` before the relativistic radial integrals. `bessel_l`
/// is FEFF `ls`, the spherical-Bessel order, and `multipole_l` is FEFF `lb`,
/// the vector multipole order.
pub fn xsph_relativistic_multipole_factors(
    kappa: i32,
    kappa_prime: i32,
    bessel_l: i32,
    multipole_l: i32,
) -> Result<XsphRelativisticMultipoleFactors, XsphError> {
    if kappa == 0 || kappa_prime == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_nonnegative_angular_momentum("bessel_l", bessel_l)?;
    validate_nonnegative_angular_momentum("multipole_l", multipole_l)?;

    let Some(ls_lb_factor) = xsph_ls_lb_factor(bessel_l, multipole_l)? else {
        return Ok(XsphRelativisticMultipoleFactors {
            p_q_prime: Complex::new(0.0, 0.0),
            q_p_prime: Complex::new(0.0, 0.0),
        });
    };

    let j2 = doubled_j_from_kappa("kappa", kappa)?;
    validate_cwig3j_doubled_argument("kappa", kappa, j2)?;
    let parity = if kappa > 0 { -1 } else { 1 };
    let j2_prime = doubled_j_from_kappa("kappa_prime", kappa_prime)?;
    validate_cwig3j_doubled_argument("kappa_prime", kappa_prime, j2_prime)?;
    let parity_prime = if kappa_prime > 0 { -1 } else { 1 };

    let p_q_prime = xsph_relativistic_multipole_component(
        ls_lb_factor,
        (j2 - parity) / 2,
        (j2_prime + parity_prime) / 2,
        bessel_l,
        multipole_l,
        j2,
        j2_prime,
        1.0,
    )?;
    let q_p_prime = xsph_relativistic_multipole_component(
        ls_lb_factor,
        (j2 + parity) / 2,
        (j2_prime - parity_prime) / 2,
        bessel_l,
        multipole_l,
        j2,
        j2_prime,
        -1.0,
    )?;

    Ok(XsphRelativisticMultipoleFactors {
        p_q_prime,
        q_p_prime,
    })
}

/// Port of FEFF `XSPH/acoef.f90`.
///
/// Builds the energy-independent angular matrix used by `szlz` for
/// occupation, spin, orbital, and magnetic-dipole density channels. The output
/// is Fortran-order with shape `(2 * lmax + 1, 2, 2, 3, lmax + 1)`. The first
/// axis stores magnetic quantum numbers as `ml + lmax`; the second and third
/// axes are FEFF's two relativistic branches; the fourth axis stores the three
/// operator channels; and the last axis stores `l = 0..=lmax`.
///
/// FEFF declares the coefficient table and intermediate angular factors as
/// default `real`, so this routine rounds the same inner products through
/// `f32` before exposing the values as [`Real`].
pub fn xsph_angular_density_coefficients(
    spin_selector: i32,
    lmax: usize,
) -> Result<Array5<Real>, XsphError> {
    if lmax > XSPH_MAX_LX {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: lmax,
            ljmax: XSPH_MAX_LX,
        });
    }
    let abs_spin = spin_selector
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "spin_selector",
            value: spin_selector,
        })?;
    let lmax_i32 = usize_to_i32("lmax", lmax)?;
    let ml_count = lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: lmax })?;
    let l_count = lmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: lmax })?;

    let mut coefficients = Array5::<Real>::zeros((ml_count, 2, 2, 3, l_count).f());
    let mut t3j = vec![0.0_f32; l_count * l_count * 2];
    let spin_projection = if spin_selector < 0 { 0_i32 } else { 1_i32 };

    for ml in -lmax_i32..=lmax_i32 {
        let mj = 2 * ml + (2 * spin_projection - 1);
        let magnetic_j = 0.5_f32 * mj as f32;
        let cwig_mj = -mj;
        let ml_index =
            usize::try_from(ml + lmax_i32).map_err(|_| XsphError::IntegerOutOfRange {
                name: "magnetic_l",
                value: ml,
            })?;

        for lp in 0..=lmax {
            let lp_i32 = usize_to_i32("lp", lp)?;
            let lp2 = lp_i32.checked_mul(2).ok_or(XsphError::SizeOutOfRange {
                name: "lp",
                value: lp,
            })?;
            for jp in 0..=lmax {
                let jp_i32 = usize_to_i32("jp", jp)?;
                let jp2 = jp_i32
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(1))
                    .ok_or(XsphError::SizeOutOfRange {
                        name: "jp",
                        value: jp,
                    })?;
                for mp in 0..=1 {
                    let mp2 = 2 * usize_to_i32("mp", mp)? - 1;
                    let angular = wigner_3j(1, jp2, lp2, mp2, cwig_mj, 2)? as f32;
                    t3j[t3j_index(lp, jp, mp, l_count)] =
                        alternating_sign(lp_i32) as f32 * ((jp2 as f32) + 1.0).sqrt() * angular;
                }
            }
        }

        for lpp in 0..=lmax {
            let lpp_i32 = usize_to_i32("lpp", lpp)?;
            let mut operator = [[[0.0_f32; 3]; 2]; 2];
            for m1 in 0..=1 {
                for m2 in 0..=1 {
                    for (iop, operator_value) in operator[m1][m2].iter_mut().enumerate() {
                        if m1 == m2 {
                            let spin_m = m1 as f32 - 0.5;
                            let orbital_m = magnetic_j - spin_m;
                            if (ml + spin_projection - usize_to_i32("m1", m1)?).abs() <= lpp_i32 {
                                if spin_selector == 0 {
                                    *operator_value = 2.0;
                                } else if iop == 0 {
                                    *operator_value = spin_m;
                                } else if iop == 1 && abs_spin == 1 {
                                    *operator_value = orbital_m;
                                } else if iop == 1 && abs_spin == 2 {
                                    *operator_value = 1.0;
                                } else if iop == 2 && abs_spin == 1 {
                                    let numerator = spin_m
                                        * 2.0
                                        * (3.0 * orbital_m * orbital_m
                                            - (lpp_i32 * (lpp_i32 + 1)) as f32);
                                    *operator_value = numerator
                                        / (2 * lpp_i32 + 3) as f32
                                        / (2 * lpp_i32 - 1) as f32;
                                } else if iop == 2 && abs_spin == 2 {
                                    let value = t3j[t3j_index(lpp, lpp, m1, l_count)];
                                    *operator_value = value * value;
                                }
                            }
                        } else if iop == 2
                            && abs_spin <= 1
                            && nint(0.5 + f64::from(magnetic_j.abs())) < lpp_i32
                        {
                            let radial = ((lpp_i32 * (lpp_i32 + 1)) as f32
                                - (magnetic_j * magnetic_j - 0.25))
                                .sqrt();
                            *operator_value = 3.0 * magnetic_j * radial
                                / (2 * lpp_i32 + 3) as f32
                                / (2 * lpp_i32 - 1) as f32;
                        } else if iop == 2 && abs_spin > 1 {
                            *operator_value = t3j[t3j_index(lpp, lpp, m1, l_count)]
                                * t3j[t3j_index(lpp, lpp, m2, l_count)];
                        }
                    }
                }
            }

            for branch_1 in 0..=1 {
                let Some(jj) = xsph_acoef_j(branch_1, lpp) else {
                    continue;
                };
                for branch_2 in 0..=1 {
                    let Some(jp) = xsph_acoef_j(branch_2, lpp) else {
                        continue;
                    };
                    for iop in 0..3 {
                        let mut value =
                            coefficients[(ml_index, branch_1, branch_2, iop, lpp)] as f32;
                        for m2 in 0..=1 {
                            for m1 in 0..=1 {
                                value += operator[m1][m2][iop]
                                    * t3j[t3j_index(lpp, jp, spin_projection as usize, l_count)]
                                    * t3j[t3j_index(lpp, jp, m1, l_count)]
                                    * t3j[t3j_index(lpp, jj, m2, l_count)]
                                    * t3j[t3j_index(lpp, jj, spin_projection as usize, l_count)];
                            }
                        }
                        coefficients[(ml_index, branch_1, branch_2, iop, lpp)] = value as Real;
                    }
                }
            }
        }
    }

    Ok(coefficients)
}
fn t3j_index(lp: usize, jp: usize, mp: usize, l_count: usize) -> usize {
    ((lp * l_count) + jp) * 2 + mp
}

fn xsph_acoef_j(branch: usize, lpp: usize) -> Option<usize> {
    if branch == 0 {
        if lpp == 0 { None } else { Some(lpp - 1) }
    } else {
        Some(lpp)
    }
}
fn validate_nonnegative_angular_momentum(name: &'static str, value: i32) -> Result<(), XsphError> {
    if value < 0 {
        Err(XsphError::NegativeAngularMomentum {
            name,
            index: 0,
            value,
        })
    } else {
        Ok(())
    }
}
fn xsph_ls_lb_factor(bessel_l: i32, multipole_l: i32) -> Result<Option<Complex>, XsphError> {
    let real_factor = if multipole_l > 0 && bessel_l == multipole_l - 1 {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        let value = f64::from(2 * multipole_l - 1) * f64::from(multipole_l + 1) / 2.0;
        value.sqrt()
    } else if bessel_l > 0 && bessel_l - 1 == multipole_l {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        let value = f64::from(2 * multipole_l + 3) * f64::from(multipole_l) / 2.0;
        value.sqrt()
    } else if bessel_l == multipole_l {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        f64::from(2 * multipole_l + 1) / 2.0_f64.sqrt()
    } else {
        return Ok(None);
    };
    Ok(Some(imaginary_unit_power(bessel_l) * real_factor))
}

#[allow(clippy::too_many_arguments)]
fn xsph_relativistic_multipole_component(
    ls_lb_factor: Complex,
    lambda: i32,
    lambda_prime: i32,
    bessel_l: i32,
    multipole_l: i32,
    j2: i32,
    j2_prime: i32,
    imaginary_sign: Real,
) -> Result<Complex, XsphError> {
    let nine_j = xsph_nine_j(lambda, lambda_prime, bessel_l, j2, j2_prime, multipole_l);
    let three_j = wigner_3j(lambda, bessel_l, lambda_prime, 0, 0, 1)?;
    let angular_weight = (6.0
        * (f64::from(j2) + 1.0)
        * (f64::from(j2_prime) + 1.0)
        * f64::from(2 * multipole_l + 1)
        * f64::from(2 * lambda + 1)
        * f64::from(2 * lambda_prime + 1))
    .sqrt();
    Ok(ls_lb_factor
        * (nine_j * three_j * alternating_sign(lambda) * angular_weight)
        * Complex::new(0.0, imaginary_sign))
}

fn xsph_nine_j(
    lambda: i32,
    lambda_prime: i32,
    bessel_l: i32,
    j2: i32,
    j2_prime: i32,
    multipole_l: i32,
) -> Real {
    if bessel_l > multipole_l {
        -f64::from(bessel_l + multipole_l + 1)
            * xsph_six_j(1, 2, 2 * multipole_l, bessel_l + multipole_l, 2 * bessel_l)
            * xsph_six_j(
                2 * multipole_l,
                bessel_l + multipole_l,
                2 * lambda_prime,
                j2_prime,
                j2,
            )
            * xsph_six_j(
                bessel_l + multipole_l,
                2 * bessel_l,
                2 * lambda,
                j2,
                2 * lambda_prime,
            )
    } else if bessel_l < multipole_l {
        -f64::from(bessel_l + multipole_l + 1)
            * xsph_six_j(1, 2, 2 * multipole_l, bessel_l + multipole_l, 2 * bessel_l)
            * xsph_six_j(
                bessel_l + multipole_l,
                2 * multipole_l,
                j2_prime,
                2 * lambda_prime,
                j2,
            )
            * xsph_six_j(
                2 * bessel_l,
                bessel_l + multipole_l,
                j2,
                2 * lambda,
                2 * lambda_prime,
            )
    } else {
        let first_term = -f64::from(2 * bessel_l + 2)
            * xsph_six_j(1, 2, 2 * multipole_l, 2 * multipole_l + 1, 2 * multipole_l)
            * xsph_six_j(
                2 * multipole_l,
                2 * multipole_l + 1,
                2 * lambda_prime,
                j2_prime,
                j2,
            )
            * xsph_six_j(
                2 * multipole_l,
                2 * multipole_l + 1,
                j2,
                2 * lambda,
                2 * lambda_prime,
            );
        if bessel_l == 0 {
            first_term
        } else {
            first_term
                - f64::from(2 * bessel_l)
                    * xsph_six_j(1, 2, 2 * multipole_l, 2 * multipole_l - 1, 2 * multipole_l)
                    * xsph_six_j(
                        2 * multipole_l - 1,
                        2 * multipole_l,
                        j2_prime,
                        2 * lambda_prime,
                        j2,
                    )
                    * xsph_six_j(
                        2 * multipole_l - 1,
                        2 * multipole_l,
                        2 * lambda,
                        j2,
                        2 * lambda_prime,
                    )
        }
    }
}

fn xsph_six_j(j1: i32, j2: i32, j3: i32, j4: i32, j5: i32) -> Real {
    if j2 != j1 + 1 {
        return 0.0;
    }
    if j4 == j3 + 1 {
        let g2 = j5 - 1;
        if g2 < (j1 - j3).abs() || g2 > j1 + j3 {
            return 0.0;
        }
        let value = (1.0 + f64::from(g2 + j1 - j3) / 2.0) * (1.0 + f64::from(g2 - j1 + j3) / 2.0)
            / f64::from(j1 + 1)
            / f64::from(j1 + 2)
            / f64::from(j3 + 1)
            / f64::from(j3 + 2);
        value.sqrt() * alternating_sign(nint(1.0 + f64::from(g2 + j1 + j3) / 2.0))
    } else if j3 == j4 + 1 {
        let g2 = j5;
        if g2 < (j1 - j4).abs() || g2 > j1 + j4 {
            return 0.0;
        }
        let value = (1.0 - f64::from(g2 - j1 - j4) / 2.0) * (2.0 + f64::from(g2 + j1 + j4) / 2.0)
            / f64::from(j1 + 1)
            / f64::from(j1 + 2)
            / f64::from(j4 + 1)
            / f64::from(j4 + 2);
        value.sqrt() * alternating_sign(nint(1.0 + f64::from(g2 + j1 + j4) / 2.0))
    } else {
        0.0
    }
}
fn alternating_sign(exponent: i32) -> Real {
    if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}
