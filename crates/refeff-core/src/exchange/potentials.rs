use super::*;

/// Port of FEFF `edp`: Dirac-Hara energy-dependent exchange potential.
///
/// `rs` is the density parameter in atomic units and `momentum` is FEFF `xk`.
/// FEFF returns zero for `rs > 100`; the Rust port preserves that cutoff.
pub fn dirac_hara_exchange_potential(rs: Real, momentum: Real) -> Result<Real, ExchangeError> {
    const MOMENTUM_OFFSET: Real = 1.0e-5_f32 as Real;
    const FERMI_THRESHOLD: Real = 1.00001_f32 as Real;

    ensure_positive("rs", rs)?;
    ensure_finite("xk", momentum)?;
    if rs > 100.0 {
        return Ok(0.0);
    }

    let fermi_momentum = FEFF_FA / rs;
    let mut x = momentum / fermi_momentum + MOMENTUM_OFFSET;
    if x < FERMI_THRESHOLD {
        x = FERMI_THRESHOLD;
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
