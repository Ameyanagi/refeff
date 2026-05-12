//! Exchange and self-energy scalar helpers from FEFF.
//!
//! This module ports small routines from `EXCH/`: the Dirac-Hara
//! energy-dependent exchange potential (`edp`), the Von Barth-Hedin spin
//! potential (`vbh`), Perdew-Zunger and Perrot-Dharma-Wardana LDA potentials,
//! and the Hedin-Lundqvist helper function `ffq`.

use thiserror::Error;

use crate::{Complex, Real};

const FEFF_FA: Real = 1.919_158_292_677_512_8;
const FEFF_PI: Real = std::f64::consts::PI;

/// Error returned by exchange-potential helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ExchangeError {
    /// Inputs must be finite real values.
    #[error("exchange input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Inputs used as positive physical scales must be strictly positive.
    #[error("exchange input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Inputs used as nonnegative physical factors must be zero or positive.
    #[error("exchange input {name} must be nonnegative, got {value}")]
    NegativeInput { name: &'static str, value: Real },
    /// A square-root radicand fell outside the real branch used by FEFF.
    #[error("exchange radicand {name} must be nonnegative, got {value}")]
    NegativeRadicand { name: &'static str, value: Real },
    /// A logarithm argument fell outside the real branch used by FEFF.
    #[error("exchange logarithm argument {name} must be positive, got {value}")]
    NonPositiveLogArgument { name: &'static str, value: Real },
}

/// Exchange-correlation energy and potential from FEFF LDA helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExchangeCorrelation {
    /// Exchange-correlation energy per particle in Hartrees.
    pub energy_per_particle: Real,
    /// Exchange-correlation potential in Hartrees.
    pub potential: Real,
}

/// Result from FEFF `imhl`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedinLundqvistImaginary {
    /// Imaginary self-energy returned by FEFF `imhl`.
    pub value: Real,
    /// FEFF `icusp` flag, true at the beginning of the imaginary branch cusp.
    pub cusp: bool,
}

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

/// Port of FEFF `edp`: Dirac-Hara energy-dependent exchange potential.
///
/// `rs` is the density parameter in atomic units and `momentum` is FEFF `xk`.
/// FEFF returns zero for `rs > 100`; the Rust port preserves that cutoff.
pub fn dirac_hara_exchange_potential(rs: Real, momentum: Real) -> Result<Real, ExchangeError> {
    ensure_positive("rs", rs)?;
    ensure_finite("xk", momentum)?;
    if rs > 100.0 {
        return Ok(0.0);
    }

    let fermi_momentum = FEFF_FA / rs;
    let mut x = momentum / fermi_momentum + 1.0e-5;
    if x < 1.00001 {
        x = 1.00001;
    }
    let log_argument = ((1.0 + x) / (1.0 - x)).abs();
    if log_argument <= 0.0 || !log_argument.is_finite() {
        return Err(ExchangeError::NonPositiveLogArgument {
            name: "edp",
            value: log_argument,
        });
    }
    let correction = log_argument.ln() * (1.0 - x * x) / (2.0 * x);
    Ok(-(fermi_momentum / FEFF_PI) * (1.0 + correction))
}

/// Port of FEFF `vbh`: Von Barth-Hedin spin exchange-correlation potential.
///
/// `spin_fraction_twice` is FEFF `xmag`, twice the fraction of the requested
/// spin orientation. FEFF returns zero for `rs > 1000`; the Rust port preserves
/// that cutoff and returns Hartrees, matching the final FEFF division by two.
pub fn von_barth_hedin_potential(
    rs: Real,
    spin_fraction_twice: Real,
) -> Result<Real, ExchangeError> {
    const GAMMA: Real = 5.129_762_802_484_097;

    ensure_positive("rs", rs)?;
    ensure_nonnegative("xmag", spin_fraction_twice)?;
    if rs > 1000.0 {
        return Ok(0.0);
    }

    let epc = -0.0504 * vbh_flarge(rs / 30.0)?;
    let efc = -0.0254 * vbh_flarge(rs / 75.0)?;
    let log_argument = 1.0 + 30.0 / rs;
    if log_argument <= 0.0 || !log_argument.is_finite() {
        return Err(ExchangeError::NonPositiveLogArgument {
            name: "vbh xmup",
            value: log_argument,
        });
    }
    let xmup = -0.0504 * log_argument.ln();
    let vu = GAMMA * (efc - epc);
    let alg = -1.22177412 / rs + vu;
    let blg = xmup - vu;
    Ok((alg * spin_fraction_twice.cbrt() + blg) / 2.0)
}

/// Port of FEFF `pz_vxc`: Perdew-Zunger LDA exchange-correlation potential.
///
/// `rs` is the Wigner-Seitz radius in Bohr. This follows the public FEFF
/// `pz_vxc` path, using Slater exchange with `alpha = 2/3` plus the
/// Perdew-Zunger 1981 unpolarized correlation fit.
pub fn perdew_zunger_vxc(rs: Real) -> Result<Real, ExchangeError> {
    Ok(perdew_zunger_exchange_correlation(rs)?.potential)
}

/// Perdew-Zunger LDA exchange-correlation energy and potential from FEFF.
///
/// This exposes the full pair produced internally by FEFF `lda_xc_pz`; callers
/// that only need the potential can use [`perdew_zunger_vxc`].
pub fn perdew_zunger_exchange_correlation(rs: Real) -> Result<ExchangeCorrelation, ExchangeError> {
    ensure_positive("rs", rs)?;

    let exchange = slater_exchange_unpolarized(rs);
    let correlation = perdew_zunger_correlation_unpolarized(rs);
    Ok(ExchangeCorrelation {
        energy_per_particle: exchange.energy_per_particle + correlation.energy_per_particle,
        potential: exchange.potential + correlation.potential,
    })
}

/// Port of FEFF `pdw_vxc`: finite-temperature PDW exchange-correlation potential.
///
/// `temperature` is in Hartrees, matching FEFF's public entry point. The
/// reduced temperature is computed from the homogeneous electron-gas Fermi
/// energy before evaluating the Perrot-Dharma-Wardana fit.
pub fn perrot_dharma_wardana_vxc(rs: Real, temperature: Real) -> Result<Real, ExchangeError> {
    ensure_positive("rs", rs)?;
    ensure_nonnegative("temp", temperature)?;

    let fermi_base = (4.0_f32 / 9.0_f32) as Real / FEFF_PI;
    let fermi_exponent = (2.0_f32 / 3.0_f32) as Real;
    let fermi_energy = 1.0 / (2.0 * fermi_base.powf(fermi_exponent) * rs * rs);
    perrot_dharma_wardana_reduced_vxc(rs, temperature / fermi_energy)
}

/// Port of FEFF `pdw_vxc1`: PDW potential at reduced temperature `t`.
///
/// `t` is the temperature divided by the electron-gas Fermi energy. This helper
/// exposes the second FEFF entry point used by tests and future thermal SCF code.
pub fn perrot_dharma_wardana_reduced_vxc(
    rs: Real,
    reduced_temperature: Real,
) -> Result<Real, ExchangeError> {
    ensure_positive("rs", rs)?;
    ensure_nonnegative("t", reduced_temperature)?;

    Ok(pdw_exchange_potential(rs, reduced_temperature)
        + pdw_correlation_potential(rs, reduced_temperature))
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
    let mut prefactor = pi_sqrt / (32.0 * scaled_rs.powf(1.5));
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
    cutoff =
        (1.0 + (6.0 / 5.0) * cutoff / plasma_over_fermi.powi(2)) * plasma_over_fermi * fermi_energy;
    let threshold = cutoff + fermi_energy;
    ensure_positive("quinn threshold", threshold)?;

    let gamma = (prefactor / x) * (x * x - 1.0).powi(2);
    let absolute_energy = fermi_energy * x * x;
    let argument = (absolute_energy - threshold) / (0.3 * threshold);
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
    const ALPHA: Real = 4.0 / 3.0;

    ensure_positive("rs", rs)?;
    ensure_positive("xk", momentum)?;

    let fermi_momentum = FEFF_FA / rs;
    let fermi_energy = fermi_momentum * fermi_momentum / 2.0;
    let mut normalized_momentum = momentum / fermi_momentum;
    if normalized_momentum < 1.00001 {
        normalized_momentum = 1.00001;
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

fn slater_exchange_unpolarized(rs: Real) -> ExchangeCorrelation {
    const F: Real = -0.687_247_939_924_714;
    const ALPHA: Real = 2.0 / 3.0;

    ExchangeCorrelation {
        energy_per_particle: F * ALPHA / rs,
        potential: (4.0 / 3.0) * F * ALPHA / rs,
    }
}

fn perdew_zunger_correlation_unpolarized(rs: Real) -> ExchangeCorrelation {
    const A: Real = 0.0311;
    const B: Real = -0.048;
    const C: Real = 0.0020;
    const D: Real = -0.0116;
    const GC: Real = -0.1423;
    const B1: Real = 1.0529;
    const B2: Real = 0.3334;

    if rs < 1.0 {
        let lnrs = rs.ln();
        let energy = A * lnrs + B + C * rs * lnrs + D * rs;
        let potential =
            A * lnrs + (B - A / 3.0) + (2.0 / 3.0) * C * rs * lnrs + ((2.0 * D - C) / 3.0) * rs;
        ExchangeCorrelation {
            energy_per_particle: energy,
            potential,
        }
    } else {
        let rs_sqrt = rs.sqrt();
        let ox = 1.0 + B1 * rs_sqrt + B2 * rs;
        let dox = 1.0 + (7.0 / 6.0) * B1 * rs_sqrt + (4.0 / 3.0) * B2 * rs;
        let energy = GC / ox;
        let potential = energy * dox / ox;
        ExchangeCorrelation {
            energy_per_particle: energy,
            potential,
        }
    }
}

fn pdw_exchange_potential(rs: Real, reduced_temperature: Real) -> Real {
    let zero_temperature = -FEFF_FA / (FEFF_PI * rs);
    if reduced_temperature < 1.0e-5 {
        zero_temperature
    } else {
        let t = reduced_temperature;
        let numerator = 1.0 + (2.83431_f32 as Real) * t.powi(2)
            - (0.215_120_f32 as Real) * t.powi(3)
            + (5.27586_f32 as Real) * t.powi(4);
        let denominator =
            1.0 + (3.94309_f32 as Real) * t.powi(2) + (7.91379_f32 as Real) * t.powi(4);
        (numerator / denominator) * (1.0 / t).tanh() * zero_temperature
    }
}

fn pdw_correlation_potential(rs: Real, reduced_temperature: Real) -> Real {
    let zero_temperature = -(0.02545_f32 as Real) * (1.0 + 19.0 / rs).ln();
    if reduced_temperature <= 0.0 {
        zero_temperature
    } else {
        let t = reduced_temperature;
        let rs_quarter = rs.powf(0.25_f32 as Real);
        let rs_three_quarters = rs.powf(0.75_f32 as Real);
        let c1 = (9.55432_f32 as Real) / (1.0 + (0.06666_f32 as Real) * rs);
        let c2 = ((3.57912_f32 as Real) - (5.99065_f32 as Real) * rs_quarter
            + (1.29722_f32 as Real) * rs_three_quarters)
            / (1.0 + (1.61126_f32 as Real) * rs_quarter);
        let c3 = (4.80217_f32 as Real) / (1.0 + (0.423_387_f32 as Real) * rs.sqrt());
        let c4 = (0.29335_f32 as Real) + (0.322_565_f32 as Real) * rs.sqrt();
        let high_temperature = -(0.638_168_f32 as Real) * (t / rs).sqrt() * (1.0 / t).tanh();

        zero_temperature * (1.0 + c1 * t + c2 * t.powf(0.25)) * (-c3 * t).exp()
            + high_temperature * (-c4 / t).exp()
    }
}

fn vbh_flarge(x: Real) -> Result<Real, ExchangeError> {
    ensure_positive("flarge x", x)?;
    let log_argument = 1.0 + 1.0 / x;
    if log_argument <= 0.0 || !log_argument.is_finite() {
        return Err(ExchangeError::NonPositiveLogArgument {
            name: "flarge",
            value: log_argument,
        });
    }
    Ok((1.0 + x.powi(3)) * log_argument.ln() + x / 2.0 - x * x - 1.0 / 3.0)
}

fn ensure_finite(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ExchangeError::NonFiniteInput { name, value })
    }
}

fn ensure_positive(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    ensure_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ExchangeError::NonPositiveInput { name, value })
    }
}

fn ensure_nonnegative(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    ensure_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ExchangeError::NegativeInput { name, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_real_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-14,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn dirac_hara_exchange_potential_matches_feff_reference() -> Result<(), ExchangeError> {
        assert_real_close(
            dirac_hara_exchange_potential(2.0, 1.3)?,
            -0.127_197_685_551_790_5,
        );
        assert_eq!(dirac_hara_exchange_potential(150.0, 0.4)?, 0.0);
        Ok(())
    }

    #[test]
    fn hedin_lundqvist_ffq_matches_feff_reference() -> Result<(), ExchangeError> {
        assert_real_close(
            hedin_lundqvist_ffq(0.8, 0.42, 1.2, 0.7, 4.0 / 3.0)?,
            0.086_644_481_840_666_34,
        );
        assert_real_close(
            hedin_lundqvist_ffq(1.6, 1.8, 2.4, 0.35, 0.9)?,
            0.062_528_814_886_981_41,
        );
        Ok(())
    }

    #[test]
    fn von_barth_hedin_potential_matches_feff_reference() -> Result<(), ExchangeError> {
        assert_real_close(
            von_barth_hedin_potential(2.5, 1.2)?,
            -0.318_654_527_096_978_5,
        );
        assert_eq!(von_barth_hedin_potential(1200.0, 0.8)?, 0.0);
        Ok(())
    }

    #[test]
    fn perdew_zunger_vxc_matches_feff_reference() -> Result<(), ExchangeError> {
        assert_real_close(perdew_zunger_vxc(0.75)?, -0.888_417_338_140_178);
        assert_real_close(perdew_zunger_vxc(2.0)?, -0.357_256_470_778_624_55);
        assert_real_close(perdew_zunger_vxc(10.0)?, -0.083_694_351_354_168_7);

        let full = perdew_zunger_exchange_correlation(2.0)?;
        assert_real_close(full.potential, -0.357_256_470_778_624_55);
        Ok(())
    }

    #[test]
    fn perrot_dharma_wardana_vxc_matches_feff_reference() -> Result<(), ExchangeError> {
        assert_real_close(
            perrot_dharma_wardana_vxc(2.0, 0.0)?,
            -0.365_286_030_418_622_73,
        );
        assert_real_close(
            perrot_dharma_wardana_vxc(2.0, 0.05)?,
            -0.372_727_169_113_195_8,
        );
        assert_real_close(
            perrot_dharma_wardana_vxc(0.75, 0.12)?,
            -0.898_713_301_179_314_3,
        );
        assert_real_close(
            perrot_dharma_wardana_reduced_vxc(2.0, 0.5)?,
            -0.371_754_474_403_717,
        );
        assert_real_close(
            perrot_dharma_wardana_reduced_vxc(8.0, 4.0)?,
            -0.094_263_857_322_040_14,
        );
        Ok(())
    }

    #[test]
    fn quinn_imaginary_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
        assert_real_close(
            quinn_imaginary_self_energy(1.15, 2.0, 0.65, 0.42)?,
            -0.002_466_676_107_350_141,
        );
        assert_real_close(
            quinn_imaginary_self_energy(2.4, 4.0, 0.35, 0.18)?,
            -0.000_005_424_604_055_647_378,
        );
        Ok(())
    }

    #[test]
    fn hedin_lundqvist_imaginary_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
        let first = hedin_lundqvist_imaginary_self_energy(2.0, 1.3)?;
        assert_real_close(first.value, -0.014_367_116_928_351_7);
        assert!(!first.cusp);

        let second = hedin_lundqvist_imaginary_self_energy(5.0, 0.8)?;
        assert_real_close(second.value, -0.062_054_668_115_880_95);
        assert!(second.cusp);
        Ok(())
    }

    #[test]
    fn exchange_helpers_reject_invalid_inputs() {
        assert!(matches!(
            dirac_hara_exchange_potential(0.0, 1.0),
            Err(ExchangeError::NonPositiveInput { name: "rs", .. })
        ));
        assert!(matches!(
            hedin_lundqvist_ffq(0.0, 1.0, 1.0, 1.0, 1.0),
            Err(ExchangeError::NonPositiveInput { name: "q", .. })
        ));
        assert!(matches!(
            von_barth_hedin_potential(1.0, -0.1),
            Err(ExchangeError::NegativeInput { name: "xmag", .. })
        ));
        assert!(matches!(
            perdew_zunger_vxc(0.0),
            Err(ExchangeError::NonPositiveInput { name: "rs", .. })
        ));
        assert!(matches!(
            perrot_dharma_wardana_vxc(1.0, -0.1),
            Err(ExchangeError::NegativeInput { name: "temp", .. })
        ));
        assert!(matches!(
            perrot_dharma_wardana_reduced_vxc(1.0, -0.1),
            Err(ExchangeError::NegativeInput { name: "t", .. })
        ));
        assert!(matches!(
            quinn_imaginary_self_energy(0.0, 1.0, 1.0, 1.0),
            Err(ExchangeError::NonPositiveInput { name: "x", .. })
        ));
        assert!(matches!(
            hedin_lundqvist_imaginary_self_energy(1.0, 0.0),
            Err(ExchangeError::NonPositiveInput { name: "xk", .. })
        ));
    }
}
