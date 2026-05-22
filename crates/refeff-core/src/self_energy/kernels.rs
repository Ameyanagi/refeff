use super::*;

/// Port of FEFF `Omegaq`: plasmon dispersion frequency.
///
/// `plasma_frequency` is FEFF `Wp` and `momentum_transfer` is FEFF `q`.
/// FEFF's source writes `2/7` in the final radicand; Fortran evaluates that as
/// integer zero, so the Rust port preserves the same compatibility behavior.
pub fn omega_q(plasma_frequency: Real, momentum_transfer: Real) -> Result<Real, SelfEnergyError> {
    ensure_positive_real("Wp", plasma_frequency)?;
    ensure_positive_real("q", momentum_transfer)?;

    let q = momentum_transfer;
    let wp = plasma_frequency;
    let log_argument = 2.0 + q;
    if log_argument <= 0.0 || !log_argument.is_finite() {
        return Err(SelfEnergyError::NonPositiveReal {
            name: "Omegaq log argument",
            value: log_argument,
        });
    }

    let denominator = 4.0 * q * (8.0 * q.powi(2) + 3.0 * wp.powi(2))
        - 3.0
            * (q + 2.0)
            * wp.powi(2)
            * ((q - 2.0) * log_argument.ln() - x_log_x((q - 2.0).abs())?.re);
    if denominator == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "Omegaq inverse moment",
        });
    }

    let inverse_moment = (std::f64::consts::PI / 2.0) * (32.0 * q.powi(3) / denominator - 1.0);
    ensure_finite_real("Omegaq inverse moment", inverse_moment)?;
    if inverse_moment == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "Omegaq moment",
        });
    }

    let radicand = -(std::f64::consts::PI * wp.powi(2)) / (2.0 * inverse_moment);
    if radicand < 0.0 || !radicand.is_finite() {
        return Err(SelfEnergyError::NegativeRadicand {
            name: "Omegaq",
            value: radicand,
        });
    }
    Ok(radicand.sqrt())
}

/// Port of FEFF `Gamq`: momentum-dependent broadening.
///
/// FEFF uses a default-real `2.4` literal here; the Rust port keeps that rounded
/// value before evaluating the double-precision square root.
pub fn gamma_q(base_width: Real, momentum_transfer: Real) -> Result<Real, SelfEnergyError> {
    ensure_nonnegative_real("gam0", base_width)?;
    ensure_nonnegative_real("q", momentum_transfer)?;

    let radicand = base_width.powi(2) + (2.4_f32 as Real) * momentum_transfer.powi(2);
    if radicand < 0.0 || !radicand.is_finite() {
        return Err(SelfEnergyError::NegativeRadicand {
            name: "Gamq",
            value: radicand,
        });
    }
    Ok(radicand.sqrt())
}

/// Port of FEFF `Logi`: imaginary logarithm branch correction.
///
/// FEFF's BPR integrands build complex logarithms from `log(abs(z))` plus this
/// branch term. Only the real part of `argument` selects the branch; `sign`
/// is the FEFF integer multiplier applied to `pi`.
pub fn log_i(argument: Complex, sign: i32) -> Result<Complex, SelfEnergyError> {
    ensure_finite_complex("Logi argument", argument)?;
    let imaginary = if argument.re < 0.0 {
        std::f64::consts::PI * Real::from(sign)
    } else {
        0.0
    };
    Ok(Complex::new(0.0, imaginary))
}

/// Port of FEFF `HFExc`: complex Hartree-Fock exchange self-energy.
///
/// `momentum` is FEFF `ckIn`, while `fermi_energy` and `fermi_momentum` are
/// FEFF `EFermi` and `kFermi`. FEFF uses the limiting value when
/// `momentum / kFermi` is within `1e-5` of one; this branch is preserved.
pub fn hartree_fock_exchange(
    momentum: Complex,
    fermi_energy: Real,
    fermi_momentum: Real,
) -> Result<Complex, SelfEnergyError> {
    ensure_finite_complex("ckIn", momentum)?;
    ensure_positive_real("EFermi", fermi_energy)?;
    ensure_positive_real("kFermi", fermi_momentum)?;

    let normalized = momentum / fermi_momentum;
    let coefficient = Complex::new(
        -2.0 * fermi_energy / (std::f64::consts::PI * fermi_momentum),
        0.0,
    );
    if (normalized - Complex::new(1.0, 0.0)).norm() <= 1.0e-5 {
        return Ok(coefficient);
    }
    if normalized.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "HFExc normalized momentum",
        });
    }

    let log_argument =
        (Complex::new(1.0, 0.0) + normalized) / (normalized - Complex::new(1.0, 0.0));
    if log_argument.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "HFExc logarithm argument",
        });
    }
    ensure_finite_complex("HFExc logarithm argument", log_argument)?;

    let value = coefficient
        * (Complex::new(1.0, 0.0)
            + (Complex::new(1.0, 0.0) / normalized - normalized) * log_argument.ln() / 2.0);
    ensure_finite_complex("HFExc", value)?;
    Ok(value)
}

/// Port of FEFF `fq`: electron-gas pole dispersion for self-energy integrands.
pub fn self_energy_pole_dispersion(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    validate_integrand_input(input)?;

    let q2 = input.q * input.q;
    let pole = Complex::new(input.plasmon_over_fermi, -input.width_over_fermi);
    let value = (pole * pole + (4.0 / 3.0) * q2 + q2 * q2).sqrt();
    ensure_finite_complex("fq", value)?;
    Ok(value)
}

/// Port of FEFF `r1`: the middle Hedin-Lundqvist self-energy integrand.
pub fn self_energy_r1_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let terms = r1_terms(input)?;
    Ok(terms.inverse_q_fq
        * (complex_log(terms.a1, "r1 a1")? + complex_log(terms.a2, "r1 a2")?
            - complex_log(terms.a3, "r1 a3")?
            - complex_log(terms.a4, "r1 a4")?))
}

/// Port of FEFF `dr1`: energy derivative of [`self_energy_r1_integrand`].
pub fn self_energy_dr1_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let terms = r1_terms(input)?;
    Ok(-terms.inverse_q_fq
        * (complex_reciprocal(terms.a1, "dr1 a1")? + complex_reciprocal(terms.a2, "dr1 a2")?
            - complex_reciprocal(terms.a3, "dr1 a3")?
            - complex_reciprocal(terms.a4, "dr1 a4")?))
}

/// Port of FEFF `r2`: the upper-branch Hedin-Lundqvist self-energy integrand.
pub fn self_energy_r2_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let terms = r23_terms(input, R23Branch::Upper)?;
    Ok(terms.inverse_q_fq * (complex_log(terms.a1, "r2 a1")? - complex_log(terms.a2, "r2 a2")?))
}

/// Port of FEFF `dr2`: energy derivative of [`self_energy_r2_integrand`].
pub fn self_energy_dr2_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let terms = r23_terms(input, R23Branch::Upper)?;
    Ok(-terms.inverse_q_fq
        * (complex_reciprocal(terms.a1, "dr2 a1")? - complex_reciprocal(terms.a2, "dr2 a2")?))
}

/// Port of FEFF `r3`: the lower-branch Hedin-Lundqvist self-energy integrand.
pub fn self_energy_r3_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let terms = r23_terms(input, R23Branch::Lower)?;
    Ok(terms.inverse_q_fq * (complex_log(terms.a1, "r3 a1")? - complex_log(terms.a2, "r3 a2")?))
}

/// Port of FEFF `dr3`: energy derivative of [`self_energy_r3_integrand`].
pub fn self_energy_dr3_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let terms = r23_terms(input, R23Branch::Lower)?;
    Ok(-terms.inverse_q_fq
        * (complex_reciprocal(terms.a1, "dr3 a1")? - complex_reciprocal(terms.a2, "dr3 a2")?))
}
#[derive(Clone, Copy)]
struct R1Terms {
    inverse_q_fq: Complex,
    a1: Complex,
    a2: Complex,
    a3: Complex,
    a4: Complex,
}

#[derive(Clone, Copy)]
struct R23Terms {
    inverse_q_fq: Complex,
    a1: Complex,
    a2: Complex,
}

#[derive(Clone, Copy)]
enum R23Branch {
    Upper,
    Lower,
}

fn r1_terms(input: SelfEnergyIntegrandInput) -> Result<R1Terms, SelfEnergyError> {
    let fqq = self_energy_pole_dispersion(input)?;
    let inverse_q_fq = inverse_q_fq(input.q, fqq)?;
    let shift = Complex::new(0.0, SELF_ENERGY_LOG_SHIFT);
    let k_plus_q = input.normalized_momentum + input.q;
    let k_minus_q = input.normalized_momentum - input.q;

    Ok(R1Terms {
        inverse_q_fq,
        a1: Complex::new(1.0 - input.gap_energy, 0.0) - input.normalized_energy - fqq + shift,
        a2: k_plus_q * k_plus_q - input.normalized_energy + fqq + shift,
        a3: k_minus_q * k_minus_q - input.normalized_energy - fqq + shift,
        a4: Complex::new(1.0 + input.gap_energy, 0.0) - input.normalized_energy + fqq + shift,
    })
}

fn r23_terms(
    input: SelfEnergyIntegrandInput,
    branch: R23Branch,
) -> Result<R23Terms, SelfEnergyError> {
    let fqq = self_energy_pole_dispersion(input)?;
    let inverse_q_fq = inverse_q_fq(input.q, fqq)?;
    let shift = Complex::new(0.0, SELF_ENERGY_LOG_SHIFT);
    let k_plus_q = input.normalized_momentum + input.q;
    let k_minus_q = input.normalized_momentum - input.q;
    let signed_fq = match branch {
        R23Branch::Upper => fqq,
        R23Branch::Lower => -fqq,
    };

    Ok(R23Terms {
        inverse_q_fq,
        a1: k_plus_q * k_plus_q - input.normalized_energy + signed_fq + shift,
        a2: k_minus_q * k_minus_q - input.normalized_energy + signed_fq + shift,
    })
}

fn validate_integrand_input(input: SelfEnergyIntegrandInput) -> Result<(), SelfEnergyError> {
    ensure_finite_complex("q", input.q)?;
    ensure_finite_complex("CPar(1)", input.normalized_momentum)?;
    ensure_finite_complex("CPar(2)", input.normalized_energy)?;
    ensure_positive_real("DPPar(1)", input.plasmon_over_fermi)?;
    ensure_nonnegative_real("DPPar(2)", input.width_over_fermi)?;
    ensure_finite_real("DPPar(4)", input.gap_energy)
}

fn inverse_q_fq(q: Complex, fqq: Complex) -> Result<Complex, SelfEnergyError> {
    let denominator = q * fqq;
    if denominator.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "self-energy q*fq",
        });
    }
    let value = Complex::new(1.0, 0.0) / denominator;
    ensure_finite_complex("1/(q*fq)", value)?;
    Ok(value)
}

fn complex_log(value: Complex, name: &'static str) -> Result<Complex, SelfEnergyError> {
    if value.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator { name });
    }
    ensure_finite_complex(name, value)?;
    let result = value.ln();
    ensure_finite_complex(name, result)?;
    Ok(result)
}

fn complex_reciprocal(value: Complex, name: &'static str) -> Result<Complex, SelfEnergyError> {
    if value.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator { name });
    }
    ensure_finite_complex(name, value)?;
    let result = Complex::new(1.0, 0.0) / value;
    ensure_finite_complex(name, result)?;
    Ok(result)
}
