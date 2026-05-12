//! Exchange and self-energy scalar helpers from FEFF.
//!
//! This module ports small routines from `EXCH/`: the Dirac-Hara
//! energy-dependent exchange potential (`edp`), the Von Barth-Hedin spin
//! potential (`vbh`), and the Hedin-Lundqvist helper function `ffq`.

use thiserror::Error;

use crate::Real;

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
    }
}
