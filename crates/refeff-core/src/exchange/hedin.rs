use super::*;

/// Port of FEFF `ffq`, the Hedin-Lundqvist integral primitive.
///
/// `q` is dimensionless and normalized to the Fermi momentum. `fermi_energy`,
/// `momentum`, and `plasma_frequency` are the FEFF `ef`, `xk`, and `wp`
/// arguments. The Rust API validates the positive denominators and logarithm
/// branch instead of allowing infinities or NaNs to propagate.
pub fn hedin_lundqvist_ffq(
    q: Real,
    fermi_energy: Real,
    momentum: Real,
    plasma_frequency: Real,
    alpha: Real,
) -> Result<Real, ExchangeError> {
    ensure_positive("q", q)?;
    ensure_finite("ef", fermi_energy)?;
    ensure_positive("xk", momentum)?;
    ensure_positive("wp", plasma_frequency)?;
    ensure_finite("alph", alpha)?;

    let radicand = plasma_frequency * plasma_frequency + alpha * q * q + q.powi(4);
    if radicand < 0.0 {
        return Err(ExchangeError::NegativeRadicand {
            name: "ffq wq",
            value: radicand,
        });
    }
    let wq = radicand.sqrt();
    let argument = (plasma_frequency + wq) / (q * q) + alpha / (2.0 * plasma_frequency);
    if argument <= 0.0 || !argument.is_finite() {
        return Err(ExchangeError::NonPositiveLogArgument {
            name: "ffq",
            value: argument,
        });
    }

    Ok(((fermi_energy * plasma_frequency) / (4.0 * momentum)) * argument.ln())
}

/// Port of FEFF `quinn`: low-energy Quinn damping correction.
///
/// `x` is momentum normalized to the Fermi momentum, `rs` is the density
/// parameter, `plasma_over_fermi` is FEFF `wp`, and `fermi_energy` is FEFF
/// `ef`. The returned value is the imaginary self-energy contribution in
/// Hartrees.
pub fn quinn_imaginary_self_energy(
    x: Real,
    rs: Real,
    plasma_over_fermi: Real,
    fermi_energy: Real,
) -> Result<Real, ExchangeError> {
    ensure_positive("x", x)?;
    ensure_positive("rs", rs)?;
    ensure_positive("wp", plasma_over_fermi)?;
    ensure_positive("ef", fermi_energy)?;

    let alpha_q = 1.0 / FEFF_FA;
    let scaled_rs = alpha_q * rs;
    let pi_sqrt = FEFF_PI.sqrt();
    let mut prefactor = pi_sqrt / (32.0 * scaled_rs.powf(1.5_f32 as Real));
    let temp1 = (FEFF_PI / scaled_rs).sqrt().atan();
    let temp2 = (scaled_rs / FEFF_PI).sqrt() / (1.0 + scaled_rs / FEFF_PI);
    prefactor *= temp1 + temp2;

    let cutoff_root = 1.0 + plasma_over_fermi;
    if cutoff_root < 0.0 {
        return Err(ExchangeError::NegativeRadicand {
            name: "quinn cutoff",
            value: cutoff_root,
        });
    }
    let mut cutoff = (cutoff_root.sqrt() - 1.0).powi(2);
    cutoff = (1.0 + ((6.0_f32 / 5.0_f32) as Real) * cutoff / plasma_over_fermi.powi(2))
        * plasma_over_fermi
        * fermi_energy;
    let threshold = cutoff + fermi_energy;
    ensure_positive("quinn threshold", threshold)?;

    let gamma = (prefactor / x) * (x * x - 1.0).powi(2);
    let absolute_energy = fermi_energy * x * x;
    let argument = (absolute_energy - threshold) / ((0.3_f32 as Real) * threshold);
    let cutoff_factor = if argument < 80.0 {
        1.0 / (1.0 + argument.exp())
    } else {
        0.0
    };
    Ok(-gamma * cutoff_factor / 2.0)
}

/// Port of FEFF `imhl`: imaginary Hedin-Lundqvist self-energy.
///
/// The returned [`HedinLundqvistImaginary::cusp`] value is FEFF `icusp != 0`.
/// The Quinn approximation is applied as the final FEFF cutoff.
pub fn hedin_lundqvist_imaginary_self_energy(
    rs: Real,
    momentum: Real,
) -> Result<HedinLundqvistImaginary, ExchangeError> {
    const ALPHA: Real = (4.0_f32 / 3.0_f32) as Real;
    const FERMI_THRESHOLD: Real = 1.00001_f32 as Real;

    ensure_positive("rs", rs)?;
    ensure_positive("xk", momentum)?;

    let fermi_momentum = FEFF_FA / rs;
    let fermi_energy = fermi_momentum * fermi_momentum / 2.0;
    let mut normalized_momentum = momentum / fermi_momentum;
    if normalized_momentum < FERMI_THRESHOLD {
        normalized_momentum = FERMI_THRESHOLD;
    }
    let plasma_over_fermi = (3.0 / rs.powi(3)).sqrt() / fermi_energy;
    let xs = plasma_over_fermi.powi(2) - (normalized_momentum.powi(2) - 1.0).powi(2);

    let mut value = 0.0;
    if xs < 0.0 {
        let inner = ALPHA * ALPHA - 4.0 * xs;
        if inner < 0.0 {
            return Err(ExchangeError::NegativeRadicand {
                name: "imhl q2 inner",
                value: inner,
            });
        }
        let q2_radicand = (inner.sqrt() - ALPHA) / 2.0;
        if q2_radicand < 0.0 {
            return Err(ExchangeError::NegativeRadicand {
                name: "imhl q2",
                value: q2_radicand,
            });
        }
        let qu = q2_radicand.sqrt().min(1.0 + normalized_momentum);
        if qu - (normalized_momentum - 1.0) > 0.0 {
            value = hedin_lundqvist_ffq(qu, fermi_energy, momentum, plasma_over_fermi, ALPHA)?
                - hedin_lundqvist_ffq(
                    normalized_momentum - 1.0,
                    fermi_energy,
                    momentum,
                    plasma_over_fermi,
                    ALPHA,
                )?;
        }
    }

    let roots = hedin_lundqvist_cubic(normalized_momentum, plasma_over_fermi, ALPHA);
    let mut cusp = false;
    if roots.radical <= 0.0 {
        if roots.qplus - (normalized_momentum + 1.0) > 0.0 {
            value += hedin_lundqvist_ffq(
                roots.qplus,
                fermi_energy,
                momentum,
                plasma_over_fermi,
                ALPHA,
            )? - hedin_lundqvist_ffq(
                normalized_momentum + 1.0,
                fermi_energy,
                momentum,
                plasma_over_fermi,
                ALPHA,
            )?;
        }
        if (normalized_momentum - 1.0) - roots.qminus > 0.0 {
            value += hedin_lundqvist_ffq(
                normalized_momentum - 1.0,
                fermi_energy,
                momentum,
                plasma_over_fermi,
                ALPHA,
            )? - hedin_lundqvist_ffq(
                roots.qminus,
                fermi_energy,
                momentum,
                plasma_over_fermi,
                ALPHA,
            )?;
            cusp = true;
        }
    }

    let quinn =
        quinn_imaginary_self_energy(normalized_momentum, rs, plasma_over_fermi, fermi_energy)?;
    if value >= quinn {
        value = quinn;
    }

    Ok(HedinLundqvistImaginary { value, cusp })
}

/// Port of FEFF `rhl`: real and imaginary Hedin-Lundqvist self-energy.
///
/// The real part is evaluated from FEFF's branch interpolation tables. The
/// imaginary part is the analytic [`hedin_lundqvist_imaginary_self_energy`]
/// value used by FEFF `rhl`.
pub fn hedin_lundqvist_self_energy(
    rs: Real,
    momentum: Real,
) -> Result<HedinLundqvistSelfEnergy, ExchangeError> {
    const FERMI_THRESHOLD: Real = 1.00001_f32 as Real;
    const RHL_RS_PANEL_0: Real = 0.2_f32 as Real;
    const RHL_RCFR: [Real; 24] = [
        -0.173_963,
        -0.173_678,
        -0.142_040,
        -0.101_030,
        -0.083_884_3,
        -0.080_704_6,
        -0.135_577,
        -0.177_556,
        -0.064_580_3,
        -0.073_117_2,
        -0.049_882_3,
        -0.039_310_8,
        -0.116_431,
        -0.090_930_0,
        -0.088_697_9,
        -0.070_231_9,
        0.079_105_1,
        -0.035_940_1,
        -0.037_958_4,
        -0.041_980_7,
        -0.062_816_2,
        0.066_925_7,
        0.066_711_9,
        0.064_817_5,
    ];
    const RHL_RCFL: [Real; 48] = [
        59.019_5,
        4.788_60,
        0.812_813,
        0.191_145,
        -291.180,
        -9.265_39,
        -0.858_348,
        -0.246_947,
        363.830,
        4.604_33,
        0.173_067,
        0.023_973_8,
        -181.726,
        -16.970_9,
        -4.094_25,
        -1.730_77,
        886.023,
        30.180_8,
        3.058_36,
        0.743_167,
        -1104.86,
        -14.908_6,
        -0.662_794,
        -0.100_106,
        184.417,
        18.020_4,
        4.504_25,
        1.843_49,
        -895.807,
        -31.869_6,
        -3.458_27,
        -0.855_367,
        1115.49,
        15.644_8,
        0.749_582,
        0.117_680,
        -62.041_1,
        -6.164_27,
        -1.538_74,
        -0.609_114,
        300.946,
        10.915_8,
        1.200_28,
        0.290_985,
        -374.494,
        -5.351_27,
        -0.261_260,
        -0.040_533_7,
    ];

    ensure_positive("rs", rs)?;
    ensure_positive("xk", momentum)?;

    let imaginary = hedin_lundqvist_imaginary_self_energy(rs, momentum)?;
    let fermi_momentum = FEFF_FA / rs;
    let fermi_energy = fermi_momentum * fermi_momentum / 2.0;
    let plasma_frequency = (3.0 / rs.powi(3)).sqrt();
    let smoothing_width = plasma_frequency / 3.0;

    let mut normalized_momentum = momentum / fermi_momentum;
    if normalized_momentum < FERMI_THRESHOLD {
        normalized_momentum = FERMI_THRESHOLD;
    }
    let delta_energy =
        ((normalized_momentum.powi(2) - 1.0) * fermi_energy - plasma_frequency - smoothing_width)
            / smoothing_width;
    let rs_panel = if rs < RHL_RS_PANEL_0 {
        0
    } else if rs < 1.0 {
        1
    } else if rs < 5.0 {
        2
    } else {
        3
    };

    let right_coefficients = [0, 1].map(|branch| {
        rhl_coeff(&RHL_RCFR, rs_panel, branch, 0) * rs
            + rhl_coeff(&RHL_RCFR, rs_panel, branch, 1) * rs * rs.sqrt()
            + rhl_coeff(&RHL_RCFR, rs_panel, branch, 2) * rs.powi(2)
    });
    let right_leading = -FEFF_PI * plasma_frequency / (4.0 * fermi_momentum * fermi_energy);

    let smooth_transition = delta_energy.abs() < 1.0;
    let mut real = 0.0;
    if !imaginary.cusp || smooth_transition {
        let left_coefficients = [0, 1, 2, 3].map(|branch| {
            rhl_coeff(&RHL_RCFL, rs_panel, branch, 0) * rs
                + rhl_coeff(&RHL_RCFL, rs_panel, branch, 1) * rs.powf(1.5)
                + rhl_coeff(&RHL_RCFL, rs_panel, branch, 2) * rs.powi(2)
        });
        real = left_coefficients
            .iter()
            .enumerate()
            .map(|(power, coefficient)| coefficient * normalized_momentum.powi(power as i32))
            .sum();
    }
    if imaginary.cusp || smooth_transition {
        let right = right_leading / normalized_momentum
            + right_coefficients[0] / normalized_momentum.powi(2)
            + right_coefficients[1] / normalized_momentum.powi(3);
        if smooth_transition {
            let weight = if delta_energy < 0.0 {
                (1.0 + delta_energy).powi(2) / 2.0
            } else {
                1.0 - (1.0 - delta_energy).powi(2) / 2.0
            };
            real = weight * right + (1.0 - weight) * real;
        } else {
            real = right;
        }
    }

    Ok(HedinLundqvistSelfEnergy {
        real: real * fermi_energy,
        imaginary: imaginary.value,
        cusp: imaginary.cusp,
    })
}

#[derive(Debug, Clone, Copy)]
struct HedinLundqvistCubic {
    radical: Real,
    qplus: Real,
    qminus: Real,
}

fn hedin_lundqvist_cubic(xk0: Real, plasma_over_fermi: Real, alpha: Real) -> HedinLundqvistCubic {
    let a2 = (alpha / (4.0 * xk0 * xk0) - 1.0) * xk0;
    let a0 = plasma_over_fermi * plasma_over_fermi / (4.0 * xk0);
    let a1 = 0.0;
    let q = a1 / 3.0 - a2 * a2 / 9.0;
    let r = (a1 * a2 - 3.0 * a0) / 6.0 - a2.powi(3) / 27.0;
    let radical = q.powi(3) + r * r;
    if radical > 0.0 {
        return HedinLundqvistCubic {
            radical,
            qplus: 0.0,
            qminus: 0.0,
        };
    }

    let s1 = Complex::new(r, (-radical).sqrt()).powf(1.0 / 3.0);
    let qplus = (2.0 * s1 - Complex::new(a2 / 3.0, 0.0)).re;
    let qminus = -(s1.re - 3.0_f64.sqrt() * s1.im + a2 / 3.0);
    HedinLundqvistCubic {
        radical,
        qplus,
        qminus,
    }
}

fn rhl_coeff(coefficients: &[Real], rs_panel: usize, branch: usize, power: usize) -> Real {
    coefficients[(branch * 3 + power) * 4 + rs_panel]
}
