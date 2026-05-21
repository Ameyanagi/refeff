use super::*;

/// Port FEFF DMDW `Calc_DW` `sig2` accumulation for `RunTyp` 0 and 3.
///
/// `temperatures` are FEFF `Lanc_In%T`, `reduced_mass` is FEFF `mu`, and
/// `angular_frequencies`/`weights` are FEFF `w_pole`/`wil`. Imaginary and
/// zero-frequency poles are ignored exactly as in FEFF's guarded pole loop.
/// The returned values are FEFF `DW_Out%s2` or `DW_Out%u2` in square angstrom.
pub fn dmdw_debye_waller_factors_from_poles(
    temperatures: ArrayView1<'_, Real>,
    reduced_mass: Real,
    angular_frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_pole_thermal_inputs(temperatures, angular_frequencies, weights)?;
    ensure_positive("DMDW reduced mass", reduced_mass)?;
    let scale = DMDW_HBARC_EV_ANGSTROM * DMDW_LIGHT_SPEED_ANGSTROM_PER_PS
        / (2.0 * reduced_mass * DMDW_AMU_EV);
    ensure_finite_output("DMDW Debye-Waller scale", scale)?;

    temperatures
        .iter()
        .copied()
        .map(|temperature| {
            let cotarg = dmdw_coth_argument_scale(temperature)?;
            let sigma = angular_frequencies
                .iter()
                .zip(weights.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(&frequency, &weight)| {
                    let coth = 1.0 / (cotarg * frequency).tanh();
                    weight / frequency * coth
                })
                .sum::<Real>();
            let value = scale * sigma;
            ensure_finite_output("DMDW Debye-Waller factor", value)?;
            Ok(value)
        })
        .collect()
}

/// Port FEFF DMDW `Calc_DW` vibrational free-energy accumulation for `RunTyp` 1.
///
/// The returned values are FEFF `DW_Out%vfe` in J/mol. FEFF prints these values
/// as eV by dividing by `Jpmol2eV` at output time.
pub fn dmdw_vibrational_free_energy_from_poles(
    temperatures: ArrayView1<'_, Real>,
    angular_frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_pole_thermal_inputs(temperatures, angular_frequencies, weights)?;

    temperatures
        .iter()
        .copied()
        .map(|temperature| {
            let cotarg = dmdw_coth_argument_scale(temperature)?;
            let entropy_sum = angular_frequencies
                .iter()
                .zip(weights.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(&frequency, &weight)| {
                    let argument = cotarg * frequency;
                    let logarithm = if argument <= 50.0 {
                        (2.0 * argument.sinh()).ln()
                    } else {
                        argument
                    };
                    weight * logarithm
                })
                .sum::<Real>();
            let value = DMDW_GAS_CONSTANT_J_PER_MOL_K * temperature * entropy_sum;
            ensure_finite_output("DMDW vibrational free energy", value)?;
            Ok(value)
        })
        .collect()
}

/// Build FEFF DMDW's single-pole Einstein diagnostic row.
///
/// `frequency_thz` is FEFF `SPole_Frq`, and `reduced_mass` is FEFF `RedMass`
/// in AMU. The returned temperature and force constant match `Print_DW_Out`.
pub fn dmdw_single_pole_einstein_summary(
    frequency_thz: Real,
    reduced_mass: Real,
) -> Result<DmdwEinsteinSummary, DebyeError> {
    ensure_positive("DMDW Einstein frequency", frequency_thz)?;
    dmdw_einstein_summary(frequency_thz, reduced_mass)
}

/// Build FEFF DMDW `n = -2..=2` projected-DOS moment summaries.
///
/// `frequencies_thz` and `weights` are FEFF `DW_Out%Poles_Frq` and
/// `DW_Out%Poles_Wgt`. Imaginary and zero-frequency poles are excluded from
/// the moment sum, and the remaining weights are renormalized by FEFF's
/// `1 - Corr` correction.
pub fn dmdw_moment_summaries_from_poles(
    reduced_mass: Real,
    frequencies_thz: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<Vec<DmdwMomentSummary>, DebyeError> {
    validate_dmdw_frequency_weight_poles(frequencies_thz, weights)?;
    ensure_positive("DMDW reduced mass", reduced_mass)?;

    let removed_weight = frequencies_thz
        .iter()
        .zip(weights.iter())
        .filter(|(frequency, _)| **frequency <= 0.0)
        .map(|(_, &weight)| weight)
        .sum::<Real>();
    let normalization = 1.0 - removed_weight;
    ensure_positive("DMDW positive pole weight normalization", normalization)?;

    (-2..=2)
        .map(|order| {
            let moment = frequencies_thz
                .iter()
                .zip(weights.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(&frequency, &weight)| frequency.powi(order) * weight)
                .sum::<Real>()
                / normalization;
            ensure_finite_output("DMDW pole moment", moment)?;

            if order == 0 {
                Ok(DmdwMomentSummary {
                    order,
                    moment_thz_power_n: moment,
                    frequency_thz: None,
                    temperature_kelvin: None,
                    effective_force_constant_n_per_m: None,
                })
            } else {
                ensure_positive("DMDW nonzero-order pole moment", moment)?;
                let frequency = moment.powf(1.0 / f64::from(order));
                let summary = dmdw_einstein_summary(frequency, reduced_mass)?;
                Ok(DmdwMomentSummary {
                    order,
                    moment_thz_power_n: moment,
                    frequency_thz: Some(summary.frequency_thz),
                    temperature_kelvin: Some(summary.temperature_kelvin),
                    effective_force_constant_n_per_m: Some(
                        summary.effective_force_constant_n_per_m,
                    ),
                })
            }
        })
        .collect()
}
