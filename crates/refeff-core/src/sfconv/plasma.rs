use ndarray::{Array1, ArrayView1};

use super::support::*;
use super::*;

/// Port of `SFCONV/plset.f90`: select one epsilon-inverse pole.
///
/// `pole_index_1based` follows FEFF's one-based `ipl` convention. The input
/// arrays correspond to `plengy`, `plwt`, and `plbrd`, and must have matching
/// lengths.
pub fn sfconv_select_pole(
    pole_index_1based: usize,
    energy: ArrayView1<'_, Real>,
    weight: ArrayView1<'_, Real>,
    broadening: ArrayView1<'_, Real>,
) -> Result<SfconvPole, SfconvError> {
    validate_count_at_least("poles", energy.len(), 1)?;
    validate_matching_lengths("energy", energy.len(), "weight", weight.len())?;
    validate_matching_lengths("energy", energy.len(), "broadening", broadening.len())?;
    validate_finite_array("energy", energy)?;
    validate_finite_array("weight", weight)?;
    validate_finite_array("broadening", broadening)?;

    if pole_index_1based == 0 || pole_index_1based > energy.len() {
        return Err(SfconvError::IndexOutOfRange {
            field: "pole",
            index: pole_index_1based,
            len: energy.len(),
        });
    }
    let index = pole_index_1based - 1;
    Ok(SfconvPole {
        energy: energy[index],
        weight: weight[index],
        broadening: broadening[index],
    })
}

/// Port of `SFCONV/ppset`: electron-gas parameters for a Wigner-Seitz radius.
pub fn sfconv_plasma_parameters(
    wigner_seitz_radius: Real,
) -> Result<SfconvPlasmaParameters, SfconvError> {
    validate_positive_scalar("wigner_seitz_radius", wigner_seitz_radius)?;

    let pi = std::f64::consts::PI;
    let fermi_momentum = (9.0 * pi / 4.0).powf(1.0 / 3.0) / wigner_seitz_radius;
    let fermi_energy = fermi_momentum * fermi_momentum / 2.0;
    let concentration = 3.0 / (4.0 * pi * wigner_seitz_radius.powi(3));
    let plasma_frequency = (4.0 * pi * concentration).sqrt();
    Ok(SfconvPlasmaParameters {
        fermi_momentum,
        fermi_energy,
        plasma_frequency,
    })
}

/// Port of the `SO2CONV` material-constant setup from FEFF output headers.
///
/// FEFF stores `Gam_ch`, `Vint`, `Mu`, and `kf` in spectrum-file headers and
/// converts them using legacy local constants in `so2conv.f90`. This helper
/// preserves those constants and returns the electron-gas quantities that feed
/// pole loading, threshold selection, momentum refinement, and convolution.
pub fn sfconv_so2conv_material_parameters(
    input: SfconvSo2convMaterialInput,
) -> Result<SfconvSo2convMaterialParameters, SfconvError> {
    validate_so2conv_material_input(input)?;

    let core_hole_lifetime = finite_result(
        "so2conv core_hole_lifetime",
        (input.core_hole_width_ev / 2.0) / SFCONV_SO2CONV_HARTREE_EV,
    )?;
    let interstitial_potential = finite_result(
        "so2conv interstitial_potential",
        input.interstitial_potential_ev / SFCONV_SO2CONV_HARTREE_EV,
    )?;
    let chemical_potential_offset = finite_result(
        "so2conv chemical_potential_offset",
        (input.chemical_potential_ev - input.interstitial_potential_ev) / SFCONV_SO2CONV_HARTREE_EV,
    )?;
    let fermi_wave_number = finite_result(
        "so2conv fermi_wave_number",
        input.fermi_wave_number_inv_angstrom * SFCONV_SO2CONV_BOHR_ANGSTROM,
    )?;
    let pi = std::f64::consts::PI;
    let fermi_momentum = finite_result(
        "so2conv fermi_momentum",
        (9.0 * pi / 4.0).powf(1.0 / 3.0) / input.wigner_seitz_radius,
    )?;
    let fermi_energy = finite_result("so2conv fermi_energy", fermi_momentum.powi(2) / 2.0)?;
    let electron_concentration = finite_result(
        "so2conv electron_concentration",
        3.0 / (4.0 * pi * input.wigner_seitz_radius.powi(3)),
    )?;
    let plasma_frequency = checked_sqrt(
        "so2conv plasma_frequency",
        4.0 * pi * electron_concentration,
    )?;
    let dispersion_parameter =
        finite_result("so2conv dispersion_parameter", 2.0 * fermi_energy / 3.0)?;

    Ok(SfconvSo2convMaterialParameters {
        core_hole_lifetime,
        interstitial_potential,
        chemical_potential_offset,
        fermi_wave_number,
        fermi_momentum,
        fermi_energy,
        electron_concentration,
        plasma_frequency,
        dispersion_parameter,
        initial_photoelectron_energy: fermi_energy,
        initial_photoelectron_momentum: fermi_momentum,
        accuracy: 1.0e-4,
    })
}

/// Port of `SFCONV/ppole.f90` `wdisp`: pole dispersion relation.
pub fn sfconv_pole_dispersion(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    pole_dispersion_value(momentum, pole_energy, dispersion_parameter)
}

/// Port of `SFCONV/ppole.f90` `dwdq`: first dispersion derivative.
pub fn sfconv_pole_dispersion_derivative(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    let dispersion = pole_dispersion_value(momentum, pole_energy, dispersion_parameter)?;
    Ok((momentum.powi(3) + 2.0 * dispersion_parameter * momentum) / (2.0 * dispersion))
}

/// Port of `SFCONV/ppole.f90` `d2wdq2`: second dispersion derivative.
pub fn sfconv_pole_dispersion_second_derivative(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    let dispersion = pole_dispersion_value(momentum, pole_energy, dispersion_parameter)?;
    let derivative =
        (momentum.powi(3) + 2.0 * dispersion_parameter * momentum) / (2.0 * dispersion);
    let numerator = (3.0 * momentum.powi(2) + 2.0 * dispersion_parameter) * dispersion
        - (momentum.powi(3) + 2.0 * dispersion_parameter * momentum) * derivative;
    Ok(numerator / (2.0 * dispersion.powi(2)))
}

/// Port of `SFCONV/ppole.f90` `qdisp`: inverse pole dispersion relation.
pub fn sfconv_inverse_pole_dispersion(
    energy: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("energy", energy)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;

    let discriminant = dispersion_parameter.powi(2) + energy.powi(2) - pole_energy.powi(2);
    if discriminant >= 0.0 {
        let radicand = -2.0 * dispersion_parameter + 2.0 * discriminant.sqrt();
        if radicand >= 0.0 {
            return Ok(radicand.sqrt());
        }
    }
    Ok(0.0)
}

/// Port of `SFCONV/ppole.f90` `vpp2`: squared pole-coupling potential.
pub fn sfconv_coupling_potential_squared(
    momentum: Real,
    plasma_frequency: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum.abs())?;
    validate_positive_scalar("plasma_frequency", plasma_frequency)?;
    let dispersion = sfconv_pole_dispersion(momentum, pole_energy, dispersion_parameter)?;
    Ok(2.0 * std::f64::consts::PI * plasma_frequency.powi(2) / (momentum.powi(2) * dispersion))
}

/// Port of `SFCONV/qlimits.f90`: momentum limits for pole-loss inequalities.
pub fn sfconv_q_limits(
    energy: Real,
    photoelectron_momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
    upper_limit: Real,
) -> Result<SfconvQLimits, SfconvError> {
    validate_finite_scalar("energy", energy)?;
    validate_positive_scalar("photoelectron_momentum", photoelectron_momentum)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;
    validate_positive_scalar("upper_limit", upper_limit)?;

    sfconv_q_limits_with_upper(
        energy,
        photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )
}

pub(crate) fn sfconv_q_limits_with_upper(
    energy: Real,
    photoelectron_momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
    upper_limit: Real,
) -> Result<SfconvQLimits, SfconvError> {
    let a = photoelectron_momentum;
    let b = energy + dispersion_parameter - 3.0 * photoelectron_momentum.powi(2) / 2.0;
    let c = photoelectron_momentum.powi(3) - 2.0 * energy * photoelectron_momentum;
    let d = pole_energy.powi(2) - energy.powi(2) + energy * photoelectron_momentum.powi(2)
        - photoelectron_momentum.powi(4) / 4.0;
    let roots =
        real_polynomial_roots([a, b, c, d]).map_err(|source| SfconvError::RootSolve { source })?;
    let values = roots.into_inner();

    if roots.real_root_count() == 3 {
        let root0 = values[0].re;
        let root1 = values[1].re;
        let root2 = values[2].re;
        let dev0 = (pole_dispersion_value(root0, pole_energy, dispersion_parameter)?
            + (root0 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let dev1 = (pole_dispersion_value(root1, pole_energy, dispersion_parameter)?
            + (root1 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let dev2 = (pole_dispersion_value(root2, pole_energy, dispersion_parameter)?
            + (root2 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let (q1, q2, q3) = if dev0 > dev1 && dev0 > dev2 {
            (
                root1.abs().min(root2.abs()),
                root1.abs().max(root2.abs()),
                root0.abs(),
            )
        } else if dev1 > dev2 {
            (
                root0.abs().min(root2.abs()),
                root0.abs().max(root2.abs()),
                root1.abs(),
            )
        } else {
            (
                root0.abs().min(root1.abs()),
                root0.abs().max(root1.abs()),
                root2.abs(),
            )
        };
        Ok(SfconvQLimits {
            count: 3,
            q1: q1.min(upper_limit),
            q2: q2.min(upper_limit),
            q3,
        })
    } else {
        let imag0 = values[0].im.abs();
        let imag1 = values[1].im.abs();
        let imag2 = values[2].im.abs();
        let q3 = if imag0 < imag1 && imag0 < imag2 {
            values[0].re.abs()
        } else if imag1 < imag2 {
            values[1].re.abs()
        } else {
            values[2].re.abs()
        };
        Ok(SfconvQLimits {
            count: 1,
            q1: 0.0,
            q2: 0.0,
            q3,
        })
    }
}

/// Port of `SFCONV/ppole.f90` `qthresh`: plasmon-loss onset momentum.
pub fn sfconv_plasmon_threshold_momentum(
    pole_energy: Real,
    dispersion_parameter: Real,
    fermi_energy: Real,
    fermi_momentum: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;
    validate_positive_scalar("fermi_energy", fermi_energy)?;
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;

    let roots = real_polynomial_roots([
        1.0,
        -3.0 * dispersion_parameter,
        3.0 * dispersion_parameter.powi(2) - 27.0 * pole_energy.powi(2) / 4.0,
        -dispersion_parameter.powi(3),
    ])
    .map_err(|source| SfconvError::RootSolve { source })?;
    let qthresh1 = if roots.real_root_count() == 1 {
        let sorted = roots_sorted_by_imag_descending(roots.into_inner());
        sorted[1].re
    } else {
        roots
            .roots()
            .iter()
            .map(|root| root.re)
            .fold(f64::NEG_INFINITY, Real::max)
    };
    let qthresh1 = if qthresh1 > 0.0 { qthresh1.sqrt() } else { 0.0 };

    let b = 1.5 * fermi_momentum + dispersion_parameter / fermi_momentum;
    let c = fermi_momentum.powi(2) + 2.0 * dispersion_parameter;
    let d = fermi_momentum.powi(3) / 4.0
        + dispersion_parameter * fermi_momentum
        + pole_energy.powi(2) / fermi_momentum;
    let roots_a = real_polynomial_roots([1.0, b, c, d])
        .map_err(|source| SfconvError::RootSolve { source })?;
    let values_a = roots_a.into_inner();
    let q01 = if roots_a.real_root_count() == 1 {
        roots_sorted_by_imag_descending(values_a)[1].re
    } else {
        let selected = select_threshold_root(values_a, |root| {
            let xfact = threshold_factor(dispersion_parameter, pole_energy, root)?;
            Ok(root - fermi_momentum - checked_sqrt("qthresh test", 2.0 * xfact)?)
        })?;
        selected.re
    };

    let roots_b = real_polynomial_roots([1.0, -b, c, -d])
        .map_err(|source| SfconvError::RootSolve { source })?;
    let values_b = roots_b.into_inner();
    let q02 = if roots_b.real_root_count() == 1 {
        roots_sorted_by_imag_descending(values_b)[1].re
    } else {
        // FEFF selects the index using the second cubic, but returns from the
        // first root array. Preserve that historical behavior.
        let index = select_threshold_root_index(values_b, |root| {
            let xfact = threshold_factor(dispersion_parameter, pole_energy, root)?;
            Ok(root + fermi_momentum - checked_sqrt("qthresh test", 2.0 * xfact)?)
        })?;
        values_a[index].re
    };

    let qthresh2 = q01.abs().min(q02.abs());
    let upper_limit = 1000.0 * fermi_momentum;
    let energy1 = qthresh1.powi(2) / 2.0;
    let limits_a = sfconv_q_limits(
        energy1,
        qthresh1,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let _q0a =
        sfconv_inverse_pole_dispersion(energy1 - fermi_energy, pole_energy, dispersion_parameter)?;

    let energy2 = qthresh2.powi(2) / 2.0;
    let limits_b = sfconv_q_limits(
        energy2,
        qthresh2,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let q0b =
        sfconv_inverse_pole_dispersion(energy2 - fermi_energy, pole_energy, dispersion_parameter)?;

    if limits_a.count == 0 || (limits_a.q1 - limits_a.q2).abs() < (limits_b.q1 - q0b).abs() {
        Ok(qthresh1)
    } else {
        Ok(qthresh2)
    }
}

/// Port of the FEFF `SO2CONV` minimal momentum grid construction.
///
/// FEFF tabulates spectral functions on 66 momentum rows. The first section
/// bridges from the Fermi momentum `qf` to the plasmon threshold `pthresh`;
/// subsequent sections extend that grid to `300 * pthresh`.
pub fn sfconv_so2conv_momentum_grid(
    fermi_momentum: Real,
    threshold_momentum: Real,
) -> Result<RealVec, SfconvError> {
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;
    validate_positive_scalar("threshold_momentum", threshold_momentum)?;
    if threshold_momentum <= fermi_momentum {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: fermi_momentum,
            upper: threshold_momentum,
        });
    }

    let mut grid = Array1::<Real>::zeros(SFCONV_SO2CONV_MOMENTUM_GRID_LEN);

    let first_step = (threshold_momentum - fermi_momentum) / 10.0;
    for (index, value) in grid.iter_mut().take(10).enumerate() {
        *value = fermi_momentum + (index as Real + 1.0) * first_step;
    }

    let second_step = 0.25 * threshold_momentum / 30.0;
    let second_anchor = grid[9];
    for offset in 1..=30 {
        grid[9 + offset] = second_anchor + offset as Real * second_step;
    }

    let third_step = 0.75 * threshold_momentum / 10.0;
    let third_anchor = grid[39];
    for offset in 1..=10 {
        grid[39 + offset] = third_anchor + offset as Real * third_step;
    }

    let fourth_step = 2.0 * threshold_momentum / 10.0;
    let fourth_anchor = grid[49];
    for offset in 1..=10 {
        grid[49 + offset] = fourth_anchor + offset as Real * fourth_step;
    }

    for (index, multiplier) in [5.0, 7.0, 10.0, 30.0, 100.0, 300.0].into_iter().enumerate() {
        grid[60 + index] = multiplier * threshold_momentum;
    }

    validate_strictly_increasing("so2conv_momentum_grid", grid.view())?;
    Ok(grid)
}
