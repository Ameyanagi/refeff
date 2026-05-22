use super::*;

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

/// Port of FEFF `ksdt_vxc`: KSDT finite-temperature exchange-correlation potential.
///
/// `temperature` is in Hartrees. FEFF's public `ksdt_vxc` entry point always
/// uses the spin-unpolarized branch.
pub fn karasiev_sjostrom_dufty_trickey_vxc(
    rs: Real,
    temperature: Real,
) -> Result<Real, ExchangeError> {
    ensure_positive("rs", rs)?;
    ensure_nonnegative("temp", temperature)?;

    let reduced_temperature = 2.0 * temperature * rs * rs / (9.0 * FEFF_PI / 4.0).powf(2.0 / 3.0);
    Ok(
        karasiev_sjostrom_dufty_trickey_free_energy(
            rs,
            reduced_temperature,
            KsdTSpin::Unpolarized,
        )?
        .potential,
    )
}

/// Port of FEFF `fxc_ksdt_01`: KSDT free energy and potential.
///
/// `reduced_temperature` is temperature divided by the electron-gas Fermi
/// temperature. [`KsdTSpin`] selects FEFF `iz = 0` or `iz = 1`.
pub fn karasiev_sjostrom_dufty_trickey_free_energy(
    rs: Real,
    reduced_temperature: Real,
    spin: KsdTSpin,
) -> Result<KsdTFreeEnergy, ExchangeError> {
    ensure_positive("rs", rs)?;
    ensure_nonnegative("t", reduced_temperature)?;

    Ok(ksdt_free_energy_unchecked(rs, reduced_temperature, spin))
}

/// Port of FEFF `exc_ksdt_01`: KSDT internal energy per particle.
///
/// `reduced_temperature` is temperature divided by the electron-gas Fermi
/// temperature. [`KsdTSpin`] selects FEFF `iz = 0` or `iz = 1`.
pub fn karasiev_sjostrom_dufty_trickey_internal_energy(
    rs: Real,
    reduced_temperature: Real,
    spin: KsdTSpin,
) -> Result<Real, ExchangeError> {
    ensure_positive("rs", rs)?;
    ensure_nonnegative("t", reduced_temperature)?;

    if reduced_temperature <= 0.0 {
        return Ok(ksdt_free_energy_unchecked(rs, 0.0, spin).free_energy_per_particle);
    }

    let (omega, coefficients) = ksdt_spin_coefficients(spin);
    let terms = ksdt_temperature_terms(reduced_temperature, omega, coefficients);
    let sqrt_rs = rs.sqrt();
    let f1 = -1.0 / rs;
    let numerator = omega * terms.aa + terms.bb * sqrt_rs + terms.cc * rs;
    let denominator = 1.0 + terms.dd * sqrt_rs + terms.ee * rs;
    let free_energy = f1 * numerator / denominator;
    let dnum_dt = omega * terms.daa + terms.dbb * sqrt_rs + terms.dcc * rs;
    let dden_dt = terms.ddd * sqrt_rs + terms.dee * rs;

    let density = 3.0 / (4.0 * FEFF_PI * rs.powi(3));
    let fermi_temperature =
        (3.0 * FEFF_PI * FEFF_PI * density).powf(2.0 / 3.0) * omega * omega / 2.0;
    let entropy = -(f1 * dnum_dt / denominator - f1 * numerator * dden_dt / denominator.powi(2))
        / fermi_temperature;
    Ok(free_energy + reduced_temperature * fermi_temperature * entropy)
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

#[derive(Debug, Clone, Copy)]
struct KsdTCoefficients {
    b: [Real; 4],
    c: [Real; 3],
    d: [Real; 5],
    e: [Real; 5],
}

#[derive(Debug, Clone, Copy)]
struct KsdTTemperatureTerms {
    aa: Real,
    daa: Real,
    bb: Real,
    dbb: Real,
    cc: Real,
    dcc: Real,
    dd: Real,
    ddd: Real,
    ee: Real,
    dee: Real,
}

fn ksdt_free_energy_unchecked(
    rs: Real,
    reduced_temperature: Real,
    spin: KsdTSpin,
) -> KsdTFreeEnergy {
    let (omega, coefficients) = ksdt_spin_coefficients(spin);
    let sqrt_rs = rs.sqrt();
    let density = 3.0 / (4.0 * FEFF_PI * rs.powi(3));
    let drs_dn = -(1.0 / 3.0) * rs / density;
    let f1 = -1.0 / rs;

    if reduced_temperature <= 0.0 {
        let lambda = (4.0 / (9.0 * FEFF_PI)).powf(1.0 / 3.0);
        let a0 = 1.0 / (FEFF_PI * lambda);
        let [b1, _, _, _] = coefficients.b;
        let [c1, _, _] = coefficients.c;
        let [d1, _, _, _, _] = coefficients.d;
        let [e1, _, _, _, _] = coefficients.e;

        let numerator = omega * a0 * 0.75 + b1 * sqrt_rs + c1 * e1 * rs;
        let denominator = 1.0 + d1 * sqrt_rs + e1 * rs;
        let free_energy = f1 * numerator / denominator;
        let dnum_drs = b1 / (2.0 * sqrt_rs) + c1 * e1;
        let dden_drs = d1 / (2.0 * sqrt_rs) + e1;
        let potential = free_energy
            + ((1.0 / 3.0) * f1) * numerator / denominator
            + density * f1 * (dnum_drs * drs_dn) / denominator
            - density * f1 * numerator * (dden_drs * drs_dn) / denominator.powi(2);
        return KsdTFreeEnergy {
            free_energy_per_particle: free_energy,
            potential,
        };
    }

    let terms = ksdt_temperature_terms(reduced_temperature, omega, coefficients);
    let numerator = omega * terms.aa + terms.bb * sqrt_rs + terms.cc * rs;
    let denominator = 1.0 + terms.dd * sqrt_rs + terms.ee * rs;
    let free_energy = f1 * numerator / denominator;

    let dnum_dt = omega * terms.daa + terms.dbb * sqrt_rs + terms.dcc * rs;
    let dnum_drs = terms.bb / (2.0 * sqrt_rs) + terms.cc;
    let dden_dt = terms.ddd * sqrt_rs + terms.dee * rs;
    let dden_drs = terms.dd / (2.0 * sqrt_rs) + terms.ee;
    let dt_dn = -(2.0 / 3.0) * reduced_temperature / density;
    let potential = free_energy
        + ((1.0 / 3.0) * f1) * numerator / denominator
        + density * f1 * (dnum_dt * dt_dn + dnum_drs * drs_dn) / denominator
        - density * f1 * numerator * (dden_dt * dt_dn + dden_drs * drs_dn) / denominator.powi(2);

    KsdTFreeEnergy {
        free_energy_per_particle: free_energy,
        potential,
    }
}

fn ksdt_temperature_terms(
    reduced_temperature: Real,
    omega: Real,
    coefficients: KsdTCoefficients,
) -> KsdTTemperatureTerms {
    let t = reduced_temperature;
    let lambda = (4.0 / (9.0 * FEFF_PI)).powf(1.0 / 3.0);
    let a0 = 1.0 / (FEFF_PI * lambda);
    let [b1, b2, b3, b4] = coefficients.b;
    let [c1, c2, c3] = coefficients.c;
    let [d1, d2, d3, d4, d5] = coefficients.d;
    let [e1, e2, e3, e4, e5] = coefficients.e;

    let tanh_t = (1.0 / t).tanh();
    let tanh_sqrt = (1.0 / t.sqrt()).tanh();
    let dtanh_t = (tanh_t.powi(2) - 1.0) / t.powi(2);
    let dtanh_sqrt = (tanh_sqrt.powi(2) - 1.0) / (2.0 * t.powf(1.5));

    let num_a = 0.75 + 3.04363 * t.powi(2) - 0.092_270 * t.powi(3) + 1.70350 * t.powi(4);
    let den_a = 1.0 + 8.31051 * t.powi(2) + 5.1105 * t.powi(4);
    let dnum_a = 2.0 * 3.04363 * t - 3.0 * 0.092_270 * t.powi(2) + 4.0 * 1.70350 * t.powi(3);
    let dden_a = 2.0 * 8.31051 * t + 4.0 * 5.1105 * t.powi(3);
    let aa = a0 * tanh_t * num_a / den_a;
    let daa = a0
        * (dtanh_t * num_a / den_a + tanh_t * dnum_a / den_a
            - tanh_t * num_a * dden_a / den_a.powi(2));

    let num_b = b1 + b2 * t.powi(2) + b3 * t.powi(4);
    let den_b = 1.0
        + b4 * t.powi(2)
        + omega * 3.0_f64.sqrt() * b3 / (2.0 * lambda.powi(2)).sqrt() * t.powi(4);
    let dnum_b = 2.0 * b2 * t + 4.0 * b3 * t.powi(3);
    let dden_b = 2.0 * b4 * t
        + omega * 3.0_f64.sqrt() * b3 / (2.0 * lambda.powi(2)).sqrt() * 4.0 * t.powi(3);
    let bb = tanh_sqrt * num_b / den_b;
    let dbb = dtanh_sqrt * num_b / den_b + tanh_sqrt * dnum_b / den_b
        - tanh_sqrt * num_b * dden_b / den_b.powi(2);

    let num_d = d1 + d2 * t.powi(2) + d3 * t.powi(4);
    let den_d = 1.0 + d4 * t.powi(2) + d5 * t.powi(4);
    let dnum_d = 2.0 * d2 * t + 4.0 * d3 * t.powi(3);
    let dden_d = 2.0 * d4 * t + 4.0 * d5 * t.powi(3);
    let dd = tanh_sqrt * num_d / den_d;
    let ddd = dtanh_sqrt * num_d / den_d + tanh_sqrt * dnum_d / den_d
        - tanh_sqrt * num_d * dden_d / den_d.powi(2);

    let num_e = e1 + e2 * t.powi(2) + e3 * t.powi(4);
    let den_e = 1.0 + e4 * t.powi(2) + e5 * t.powi(4);
    let dnum_e = 2.0 * e2 * t + 4.0 * e3 * t.powi(3);
    let dden_e = 2.0 * e4 * t + 4.0 * e5 * t.powi(3);
    let ee = tanh_t * num_e / den_e;
    let dee =
        dtanh_t * num_e / den_e + tanh_t * dnum_e / den_e - tanh_t * num_e * dden_e / den_e.powi(2);

    let num_c = c1 + c2 * (-c3 / t).exp();
    let dnum_c = c2 * c3 * (-c3 / t).exp() / t.powi(2);
    let cc = num_c * ee;
    let dcc = dnum_c * ee + num_c * dee;

    KsdTTemperatureTerms {
        aa,
        daa,
        bb,
        dbb,
        cc,
        dcc,
        dd,
        ddd,
        ee,
        dee,
    }
}

fn ksdt_spin_coefficients(spin: KsdTSpin) -> (Real, KsdTCoefficients) {
    match spin {
        KsdTSpin::Unpolarized => (
            1.0,
            KsdTCoefficients {
                b: [0.283_997, 48.932_154, 0.370_919, 61.095_357],
                c: [0.870_089, 0.193_077, 2.414_644],
                d: [0.579_824, 94.537_454, 97.839_603, 59.939_999, 24.388_037],
                e: [0.212_036, 16.731_249, 28.485_792, 34.028_876, 17.235_515],
            },
        ),
        KsdTSpin::FullyPolarized => (
            2.0_f64.powf(1.0 / 3.0),
            KsdTCoefficients {
                b: [0.329_001, 111.598_308, 0.537_053, 105.086_663],
                c: [0.848_930, 0.167_952, 0.088_820],
                d: [0.551_330, 180.213_159, 134.486_231, 103.861_695, 17.750_710],
                e: [0.153_124, 19.543_945, 43.400_337, 120.255_145, 15.662_836],
            },
        ),
    }
}
