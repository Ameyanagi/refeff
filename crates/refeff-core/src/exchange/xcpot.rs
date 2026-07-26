//! Helpers from FEFF `EXCH/xcpot.f90`.

use super::*;
use crate::interpolation::terp;
use crate::self_energy::{
    ManyPoleSelfEnergy, ManyPoleSelfEnergyInput, SelfEnergyError, many_pole_self_energy,
};
use ndarray::Array1;

/// Port of the MPSE `Rs1` density grid setup in FEFF `EXCH/xcpot.f90`.
///
/// `xcpot` derives a small Wigner-Seitz radius grid from `densty(1:jri+1)`.
/// For `iPl == 2` FEFF builds a radial grid used for many-pole interpolation;
/// all other selectors use the interstitial radius for every sample.
pub fn xcpot_many_pole_density_grid(
    input: XcpotManyPoleDensityGridInput<'_>,
) -> Result<XcpotManyPoleDensityGrid, ExchangeError> {
    let reference_index_1based =
        input
            .radial_match_index_1based
            .checked_add(1)
            .ok_or(ExchangeError::InvalidIndex {
                name: "radial_match_index_1based",
                index: input.radial_match_index_1based,
            })?;
    if input.radial_match_index_1based == 0 {
        return Err(ExchangeError::InvalidIndex {
            name: "radial_match_index_1based",
            index: input.radial_match_index_1based,
        });
    }
    if input.density.len() < reference_index_1based {
        return Err(ExchangeError::LengthTooShort {
            name: "density",
            required: reference_index_1based,
            actual: input.density.len(),
        });
    }
    for index in 0..reference_index_1based {
        ensure_finite("density", input.density[index])?;
    }

    let interstitial_density = input.density[reference_index_1based - 1];
    let interstitial_radius = density_radius_or(interstitial_density, 10.0);
    let core_radius = density_radius_or(input.density[0], 101.0);

    let mut min_density = input.density[0];
    let mut max_density = input.density[0];
    for index in 1..reference_index_1based {
        let density = input.density[index];
        min_density = min_density.min(density);
        max_density = max_density.max(density);
    }

    let min_radius = if max_density <= 0.0 {
        interstitial_radius * 1.0e-3
    } else {
        density_to_wigner_seitz_radius(max_density)
    };
    let max_radius = if min_density <= 0.0 {
        interstitial_radius * 2.0
    } else {
        density_to_wigner_seitz_radius(min_density)
    };
    let radius_step = (max_radius - min_radius) / (XCPOT_MPSE_GRID_POINTS as Real - 2.0);

    let mut radii = [interstitial_radius; XCPOT_MPSE_GRID_POINTS];
    if input.plasmon_selector == 2 {
        radii[0] = min_radius;
        for index in 1..XCPOT_MPSE_GRID_POINTS {
            if max_radius > interstitial_radius {
                if radii[index - 1] < interstitial_radius {
                    let next = min_radius + index as Real * radius_step;
                    radii[index] = next.min(interstitial_radius);
                } else if index > 1 {
                    radii[index] = min_radius + (index - 1) as Real * radius_step;
                }
            } else {
                radii[index] = min_radius + index as Real * radius_step;
            }
        }
    }

    Ok(XcpotManyPoleDensityGrid {
        interstitial_radius,
        core_radius,
        min_radius,
        max_radius,
        radius_step,
        radii,
    })
}

/// Port of FEFF `EXCH/xcpot.f90` MPSE enable and pole-count setup.
///
/// FEFF enables the many-pole branch only for `iPl > 0` and `ixc == 0`, then
/// scans `WpCorr` until the first value strictly less than `-1`. If FEFF never
/// finds that sentinel, `NPoles` is left undefined; safe Rust reports that as a
/// typed error instead.
pub fn xcpot_many_pole_control(
    input: XcpotManyPoleControlInput<'_>,
) -> Result<XcpotManyPoleControl, ExchangeError> {
    let enabled = input.plasmon_selector > 0 && input.exchange_selector % 10 == 0;
    if !enabled {
        return Ok(XcpotManyPoleControl {
            enabled,
            active_pole_count: 0,
        });
    }

    for (index, &frequency) in input.pole_frequencies.iter().enumerate() {
        ensure_finite("pole_frequencies", frequency)?;
        if frequency < -1.0 {
            return Ok(XcpotManyPoleControl {
                enabled,
                active_pole_count: index,
            });
        }
    }

    Err(ExchangeError::MissingManyPoleSentinel {
        name: "pole_frequencies",
        len: input.pole_frequencies.len(),
    })
}

/// Port of FEFF `EXCH/xcpot.f90` MPSE delta-self-energy table shaping.
///
/// This helper applies the `SigF` reuse, `ZRnrm * (deltaHL - SigF)`
/// normalization, and `delrHL`/`deliHL` split immediately around the `CSigZ`
/// calls. The `CSigZ` calculation itself is intentionally not part of this
/// small port slice; callers provide its Fermi-level and current-energy values.
pub fn xcpot_many_pole_delta_table(
    input: XcpotManyPoleDeltaTableInput<'_>,
) -> Result<XcpotManyPoleDeltaTable, ExchangeError> {
    validate_many_pole_delta_table_input(&input)?;

    let radial = input.plasmon_selector == 2;
    let last_index = XCPOT_MPSE_GRID_POINTS - 1;
    let mut fermi_self_energy = [Complex::new(0.0, 0.0); XCPOT_MPSE_GRID_POINTS];
    let mut delta_self_energy = [Complex::new(0.0, 0.0); XCPOT_MPSE_GRID_POINTS];

    for (index, value) in fermi_self_energy.iter_mut().enumerate() {
        if radial || index == last_index {
            *value = input.fermi_self_energy[index];
        }
    }
    if !radial {
        let bulk_fermi_self_energy = fermi_self_energy[last_index];
        fermi_self_energy[..last_index].fill(bulk_fermi_self_energy);
    }

    for (index, value) in delta_self_energy.iter_mut().enumerate() {
        if radial || index == last_index {
            *value = input.renormalization
                * (input.energy_self_energy[index] - fermi_self_energy[index]);
        }
    }
    if !radial {
        let bulk_delta_self_energy = delta_self_energy[last_index];
        delta_self_energy[..last_index].fill(bulk_delta_self_energy);
    }

    Ok(XcpotManyPoleDeltaTable {
        fermi_self_energy,
        delta_self_energy,
        real: delta_self_energy.map(|value| value.re),
        imaginary: delta_self_energy.map(|value| value.im),
    })
}

/// Port of the FEFF `EXCH/xcpot.f90` `CSigZ` MPSE table calculation.
///
/// FEFF scales `WpCorr(1:NPoles)` by `sqrt(3 / Rs1(i)^3)` for each sampled
/// radius, computes the Fermi self energy once at `cmu = xmu * 1.00001`, then
/// computes the current-energy self energy and applies that sample's `ZRnrm`.
/// Selector `iPl == 1` computes only the bulk row and copies it to every
/// interpolation slot, while `iPl == 2` keeps all radial samples.
pub fn xcpot_many_pole_self_energy_delta_table(
    input: XcpotManyPoleSelfEnergyTableInput<'_>,
) -> Result<XcpotManyPoleDeltaTable, ExchangeError> {
    validate_many_pole_self_energy_table_input(&input)?;

    let radial = input.plasmon_selector == 2;
    let last_index = XCPOT_MPSE_GRID_POINTS - 1;
    let mut fermi_self_energy = [Complex::new(0.0, 0.0); XCPOT_MPSE_GRID_POINTS];
    let mut delta_self_energy = [Complex::new(0.0, 0.0); XCPOT_MPSE_GRID_POINTS];
    let fermi_energy = Complex::new(input.fermi_level * 1.00001, 0.0);

    for index in 0..XCPOT_MPSE_GRID_POINTS {
        if radial || index == last_index {
            let radius = input.density_grid.radii[index];
            let fermi = xcpot_many_pole_self_energy_sample(XcpotManyPoleSelfEnergySampleInput {
                pole_frequencies: input.pole_frequencies,
                pole_widths: input.pole_widths,
                amplitudes: input.amplitudes,
                active_pole_count: input.active_pole_count,
                energy: fermi_energy,
                fermi_level: input.fermi_level,
                radius,
                gap_energy: input.gap_energy,
                use_broadened_pole: input.use_broadened_pole,
            })?;
            let current = xcpot_many_pole_self_energy_sample(XcpotManyPoleSelfEnergySampleInput {
                pole_frequencies: input.pole_frequencies,
                pole_widths: input.pole_widths,
                amplitudes: input.amplitudes,
                active_pole_count: input.active_pole_count,
                energy: input.energy,
                fermi_level: input.fermi_level,
                radius,
                gap_energy: input.gap_energy,
                use_broadened_pole: input.use_broadened_pole,
            })?;
            fermi_self_energy[index] = fermi.self_energy;
            delta_self_energy[index] =
                current.renormalization * (current.self_energy - fermi.self_energy);
        }
    }

    if !radial {
        let fermi_bulk = fermi_self_energy[last_index];
        let delta_bulk = delta_self_energy[last_index];
        fermi_self_energy[..last_index].fill(fermi_bulk);
        delta_self_energy[..last_index].fill(delta_bulk);
    }

    Ok(XcpotManyPoleDeltaTable {
        fermi_self_energy,
        delta_self_energy,
        real: delta_self_energy.map(|value| value.re),
        imaginary: delta_self_energy.map(|value| value.im),
    })
}

#[derive(Clone, Copy)]
struct XcpotManyPoleSelfEnergySampleInput<'a> {
    pole_frequencies: ndarray::ArrayView1<'a, Real>,
    pole_widths: ndarray::ArrayView1<'a, Real>,
    amplitudes: ndarray::ArrayView1<'a, Real>,
    active_pole_count: usize,
    energy: Complex,
    fermi_level: Real,
    radius: Real,
    gap_energy: Real,
    use_broadened_pole: bool,
}

fn xcpot_many_pole_self_energy_sample(
    input: XcpotManyPoleSelfEnergySampleInput<'_>,
) -> Result<ManyPoleSelfEnergy, ExchangeError> {
    ensure_positive("Rs1", input.radius)?;
    let scale = (3.0 / input.radius.powi(3)).sqrt();
    ensure_finite("WpCorr scale", scale)?;
    let scaled_frequencies = Array1::from_iter(
        input
            .pole_frequencies
            .iter()
            .take(input.active_pole_count)
            .map(|&frequency| frequency * scale),
    );

    many_pole_self_energy(ManyPoleSelfEnergyInput {
        energy: input.energy,
        fermi_level: input.fermi_level,
        radius: input.radius,
        pole_frequencies: scaled_frequencies.view(),
        pole_widths: input.pole_widths,
        amplitudes: input.amplitudes,
        gap_energy: input.gap_energy,
        active_pole_count: input.active_pole_count,
        use_broadened_pole: input.use_broadened_pole,
    })
    .map_err(map_self_energy_error)
}

/// Port of FEFF `EXCH/xcpot.f90` MPSE row-delta selection.
///
/// In the `csig` branch FEFF skips the ordinary self-consistency loop and jumps
/// to label `15` after selecting `delr/deli` either from the bulk entry
/// `del*HL(1)` or by order-1 interpolation on `Rs1`.
pub fn xcpot_many_pole_row_delta(
    input: XcpotManyPoleRowDeltaInput,
) -> Result<XcpotSigma, ExchangeError> {
    ensure_positive("rs", input.radius)?;

    if input.plasmon_selector == 2 {
        ensure_finite("min_radius", input.density_grid.min_radius)?;
        ensure_finite("max_radius", input.density_grid.max_radius)?;

        if input.radius < input.density_grid.min_radius
            || input.radius > input.density_grid.max_radius
        {
            return Ok(XcpotSigma {
                real: 0.0,
                imaginary: 0.0,
            });
        }

        for &radius in &input.density_grid.radii {
            ensure_finite("density_grid.radii", radius)?;
        }
        for &real in &input.delta_table.real {
            ensure_finite("delta_table.real", real)?;
        }
        for &imaginary in &input.delta_table.imaginary {
            ensure_finite("delta_table.imaginary", imaginary)?;
        }

        let real = terp(
            &input.density_grid.radii,
            &input.delta_table.real,
            1,
            input.radius,
        )
        .map_err(ExchangeError::Interpolation)?
        .value;
        let imaginary = terp(
            &input.density_grid.radii,
            &input.delta_table.imaginary,
            1,
            input.radius,
        )
        .map_err(ExchangeError::Interpolation)?
        .value;
        Ok(XcpotSigma { real, imaginary })
    } else if input.plasmon_selector > 0 {
        ensure_finite("delta_table.real", input.delta_table.real[0])?;
        ensure_finite("delta_table.imaginary", input.delta_table.imaginary[0])?;
        Ok(XcpotSigma {
            real: input.delta_table.real[0],
            imaginary: input.delta_table.imaginary[0],
        })
    } else {
        Err(ExchangeError::InvalidSelector {
            name: "plasmon_selector",
            value: input.plasmon_selector,
        })
    }
}

/// Port of FEFF `EXCH/xcpot.f90` self-energy delta application.
///
/// This covers the block after label `15`: `delta` is added to `vtot` to form
/// complex `v`, while `ixc >= 5` also adds `deltav` to `vvalgs`. FEFF's `ixc ==
/// 5` branch uses `delvi` rather than `deli` for the imaginary part of `delta`.
pub fn xcpot_apply_self_energy_deltas(
    input: XcpotSelfEnergyApplicationInput<'_>,
) -> Result<XcpotSelfEnergyApplication, ExchangeError> {
    validate_self_energy_application_input(&input)?;

    let exchange_branch = input.exchange_selector % 10;
    let total_potential = Array1::from_iter((0..input.active_len).map(|index| {
        let imaginary = if exchange_branch == 5 {
            input.valence_delta_imaginary[index]
        } else {
            input.delta_imaginary[index]
        };
        Complex::new(
            input.total_potential[index] + input.delta_real[index],
            imaginary,
        )
    }));

    let valence_potential = if exchange_branch >= 5 {
        Array1::from_iter((0..input.active_len).map(|index| {
            Complex::new(
                input.valence_potential[index] + input.valence_delta_real[index],
                input.valence_delta_imaginary[index],
            )
        }))
    } else {
        Array1::from_vec(Vec::new())
    };

    validate_complex_slice("delta_total_potential", &total_potential)?;
    validate_complex_slice("delta_valence_potential", &valence_potential)?;

    Ok(XcpotSelfEnergyApplication {
        total_potential,
        valence_potential,
    })
}

/// Port of FEFF `EXCH/xcpot.f90` local density and momentum scales.
///
/// This covers the per-radial-row setup immediately before the `sigma` calls:
/// `rs`, `xf`, spin-magnetized scales, and the `ixc == 5`/`ixc >= 6` valence
/// and core-density branches.
pub fn xcpot_local_scales(input: XcpotLocalScalesInput) -> Result<XcpotLocalScales, ExchangeError> {
    ensure_finite("density", input.density)?;
    ensure_finite("magnetization", input.magnetization)?;
    ensure_finite("valence_density", input.valence_density)?;

    let magnetization_factor = 1.0 + input.magnetization;
    ensure_positive("magnetization_factor", magnetization_factor)?;

    let exchange_branch = input.exchange_selector % 10;
    let radius = density_radius_or(input.density, 10.0);
    let fermi_momentum = FEFF_FA / radius;
    let magnetized_radius = radius / magnetization_factor.powf(1.0 / 3.0);
    let magnetized_fermi_momentum = FEFF_FA / magnetized_radius;

    let (valence_radius, valence_fermi_momentum) = if exchange_branch == 5 {
        let valence_radius = if input.valence_density > 1.0e-5 {
            density_to_wigner_seitz_radius(input.valence_density).min(10.0)
        } else {
            10.0
        };
        (Some(valence_radius), Some(FEFF_FA / valence_radius))
    } else {
        (None, None)
    };

    let core_radius = if exchange_branch >= 6 {
        if input.density <= input.valence_density {
            Some(101.0)
        } else {
            Some(density_to_wigner_seitz_radius(
                input.density - input.valence_density,
            ))
        }
    } else {
        None
    };

    Ok(XcpotLocalScales {
        radius,
        fermi_momentum,
        magnetized_radius,
        magnetized_fermi_momentum,
        valence_radius,
        valence_fermi_momentum,
        core_radius,
    })
}

/// Port of FEFF `EXCH/xcpot.f90` nested `sigma` helper.
///
/// This dispatches the analytic FEFF branches used by `xcpot`: Hedin-Lundqvist
/// (`rhl`), Dirac-Hara (`edp`), Dirac-Hara plus `imhl`, and the `ixc >= 6`
/// core subtraction. Use [`xcpot_sigma_with_broadened_table`] for FEFF's
/// external-table `ibp == 1` branch.
pub fn xcpot_sigma(input: XcpotSigmaInput) -> Result<XcpotSigma, ExchangeError> {
    xcpot_sigma_impl(input, None)
}

/// Evaluate the nested FEFF `sigma` helper with an author-supplied
/// `bphl.dat` table available to the `ibp == 1` branch.
pub fn xcpot_sigma_with_broadened_table(
    input: XcpotSigmaInput,
    table: &BroadenedHedinLundqvistTable,
) -> Result<XcpotSigma, ExchangeError> {
    xcpot_sigma_impl(input, Some(table))
}

fn xcpot_sigma_impl(
    input: XcpotSigmaInput,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<XcpotSigma, ExchangeError> {
    ensure_positive("rs", input.radius)?;
    ensure_finite("rscore", input.core_radius)?;
    ensure_positive("xk", input.momentum)?;

    let exchange_branch = input.exchange_selector % 10;
    let broadened_branch = input.exchange_selector / 10;

    let (mut real, imaginary) =
        if (exchange_branch == 0 || exchange_branch >= 5) && broadened_branch == 0 {
            let sigma = hedin_lundqvist_self_energy(input.radius, input.momentum)?;
            (sigma.real, sigma.imaginary)
        } else if (exchange_branch == 0 || exchange_branch >= 5) && broadened_branch == 1 {
            let table = broadened_table.ok_or(ExchangeError::MissingReferenceData {
                name: "index / 10",
                value: broadened_branch,
                data: "bphl.dat",
            })?;
            let sigma = broadened_hedin_lundqvist_self_energy(table, input.radius, input.momentum)?;
            (sigma.real, sigma.imaginary)
        } else if exchange_branch == 1 {
            (
                dirac_hara_exchange_potential(input.radius, input.momentum)?,
                0.0,
            )
        } else if exchange_branch == 3 {
            (
                dirac_hara_exchange_potential(input.radius, input.momentum)?,
                hedin_lundqvist_imaginary_self_energy(input.radius, input.momentum)?.value,
            )
        } else {
            return Err(ExchangeError::InvalidSelector {
                name: "exchange_selector % 10",
                value: exchange_branch,
            });
        };

    if exchange_branch >= 6 {
        ensure_positive("rscore", input.core_radius)?;
        real -= dirac_hara_exchange_potential(input.core_radius, input.momentum)?;
    }

    ensure_finite("sigma.real", real)?;
    ensure_finite("sigma.imaginary", imaginary)?;

    Ok(XcpotSigma { real, imaginary })
}

/// Port of FEFF `EXCH/xcpot.f90` Fermi-level self-energy cache setup.
///
/// This covers the `if (ifirst .eq. 0)` block inside the radial loop. FEFF
/// caches `sigma(mu)` for the total and valence channels and leaves `gsrel(i)`
/// at `1.0`; the commented Von Barth-Hedin magnetization ratio is intentionally
/// not reintroduced here.
pub fn xcpot_fermi_cache(input: XcpotFermiCacheInput) -> Result<XcpotFermiCache, ExchangeError> {
    xcpot_fermi_cache_impl(input, None)
}

/// Build the FEFF Fermi-level cache with an author-supplied `bphl.dat` table
/// available to the `ibp == 1` branch.
pub fn xcpot_fermi_cache_with_broadened_table(
    input: XcpotFermiCacheInput,
    table: &BroadenedHedinLundqvistTable,
) -> Result<XcpotFermiCache, ExchangeError> {
    xcpot_fermi_cache_impl(input, Some(table))
}

fn xcpot_fermi_cache_impl(
    input: XcpotFermiCacheInput,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<XcpotFermiCache, ExchangeError> {
    ensure_positive("rs", input.radius)?;
    ensure_finite("rscore", input.core_radius)?;

    let exchange_branch = input.exchange_selector % 10;
    let broadened_branch = input.exchange_selector / 10;
    let fermi_momentum = FEFF_FA / input.radius;
    let cache_momentum = fermi_momentum * 1.00001;
    let total_selector = if exchange_branch < 5 {
        input.exchange_selector
    } else {
        broadened_branch * 10
    };

    let total_self_energy = xcpot_sigma_impl(
        XcpotSigmaInput {
            exchange_selector: total_selector,
            radius: input.radius,
            core_radius: input.core_radius,
            momentum: cache_momentum,
        },
        broadened_table,
    )?;

    let valence_self_energy = if exchange_branch == 5 {
        let valence_radius = input
            .valence_radius
            .ok_or(ExchangeError::MissingRequiredInput {
                name: "valence_radius",
                value: exchange_branch,
            })?;
        ensure_positive("valence_radius", valence_radius)?;
        let valence_self_energy = xcpot_sigma_impl(
            XcpotSigmaInput {
                exchange_selector: input.exchange_selector,
                radius: valence_radius,
                core_radius: input.core_radius,
                momentum: (FEFF_FA / valence_radius) * 1.00001,
            },
            broadened_table,
        )?;
        if input.interstitial {
            total_self_energy
        } else {
            valence_self_energy
        }
    } else if exchange_branch >= 6 {
        let valence_self_energy = xcpot_sigma_impl(
            XcpotSigmaInput {
                exchange_selector: input.exchange_selector,
                radius: input.radius,
                core_radius: input.core_radius,
                momentum: cache_momentum,
            },
            broadened_table,
        )?;
        if exchange_branch == 6 && input.interstitial {
            total_self_energy
        } else {
            valence_self_energy
        }
    } else {
        XcpotSigma {
            real: 0.0,
            imaginary: 0.0,
        }
    };

    Ok(XcpotFermiCache {
        total_self_energy,
        valence_self_energy,
        ground_state_ratio: 1.0,
    })
}

/// Port of FEFF `EXCH/xcpot.f90` non-MPSE Dyson self-energy correction.
///
/// This covers the local momentum setup, one FEFF refinement iteration
/// (`nmax = 1`), `delr/deli`, and the valence `delvr/delvi` branches used later
/// when adding the correction to `vtot` and `vvalgs`.
pub fn xcpot_self_energy_correction(
    input: XcpotSelfEnergyCorrectionInput,
) -> Result<XcpotSelfEnergyCorrection, ExchangeError> {
    xcpot_self_energy_correction_impl(input, None)
}

/// Evaluate the non-MPSE Dyson correction with an author-supplied `bphl.dat`
/// table available to all nested `sigma` calls.
pub fn xcpot_self_energy_correction_with_broadened_table(
    input: XcpotSelfEnergyCorrectionInput,
    table: &BroadenedHedinLundqvistTable,
) -> Result<XcpotSelfEnergyCorrection, ExchangeError> {
    xcpot_self_energy_correction_impl(input, Some(table))
}

fn xcpot_self_energy_correction_impl(
    input: XcpotSelfEnergyCorrectionInput,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<XcpotSelfEnergyCorrection, ExchangeError> {
    validate_self_energy_correction_input(input)?;

    let exchange_branch = input.exchange_selector % 10;
    let broadened_branch = input.exchange_selector / 10;
    let initial_momentum_squared =
        2.0 * (input.energy - input.fermi_level) + input.fermi_momentum.powi(2);
    let initial_momentum = sqrt_nonnegative("initial_momentum", initial_momentum_squared)?;

    let magnetized_momentum_squared =
        2.0 * (input.energy - input.fermi_level) + input.magnetized_fermi_momentum.powi(2);
    let magnetized_momentum = if magnetized_momentum_squared < 0.0 {
        initial_momentum
    } else {
        sqrt_nonnegative("magnetized_momentum", magnetized_momentum_squared)?
    };

    let initial_selector = if exchange_branch < 5 {
        input.exchange_selector
    } else {
        broadened_branch * 10
    };
    let initial_sigma = xcpot_sigma_impl(
        XcpotSigmaInput {
            exchange_selector: initial_selector,
            radius: input.radius,
            core_radius: input.core_radius,
            momentum: initial_momentum,
        },
        broadened_table,
    )?;

    let mut delta_real = input.fermi_cache.ground_state_ratio
        * (initial_sigma.real - input.fermi_cache.total_self_energy.real);
    let mut corrected_momentum = initial_momentum;
    let mut total_delta = XcpotSigma {
        real: 0.0,
        imaginary: 0.0,
    };
    let mut valence_delta = None;

    for iteration in 0..=1 {
        let corrected_momentum_squared =
            2.0 * (input.energy - input.fermi_level - delta_real) + input.fermi_momentum.powi(2);
        corrected_momentum = sqrt_nonnegative("corrected_momentum", corrected_momentum_squared)?;

        let sigma = xcpot_sigma_impl(
            XcpotSigmaInput {
                exchange_selector: input.exchange_selector,
                radius: input.radius,
                core_radius: input.core_radius,
                momentum: corrected_momentum,
            },
            broadened_table,
        )?;
        total_delta = XcpotSigma {
            real: input.fermi_cache.ground_state_ratio
                * (sigma.real - input.fermi_cache.total_self_energy.real),
            imaginary: sigma.imaginary - input.fermi_cache.total_self_energy.imaginary,
        };

        if exchange_branch >= 5
            && input.interstitial
            && corrected_momentum > input.fermi_momentum
            && (exchange_branch == 5 || exchange_branch == 6)
        {
            valence_delta = Some(total_delta);
        }

        if iteration == 0 {
            delta_real = total_delta.real;
        }
    }

    if exchange_branch >= 5 && !input.interstitial && corrected_momentum > input.fermi_momentum {
        let valence_sigma = if exchange_branch == 5 {
            let valence_radius =
                input
                    .valence_radius
                    .ok_or(ExchangeError::MissingRequiredInput {
                        name: "valence_radius",
                        value: exchange_branch,
                    })?;
            let valence_fermi_momentum =
                input
                    .valence_fermi_momentum
                    .ok_or(ExchangeError::MissingRequiredInput {
                        name: "valence_fermi_momentum",
                        value: exchange_branch,
                    })?;
            let valence_momentum_squared = corrected_momentum.powi(2)
                - input.fermi_momentum.powi(2)
                + valence_fermi_momentum.powi(2);
            let valence_momentum =
                sqrt_nonnegative("valence_corrected_momentum", valence_momentum_squared)?;
            xcpot_sigma_impl(
                XcpotSigmaInput {
                    exchange_selector: input.exchange_selector,
                    radius: valence_radius,
                    core_radius: input.core_radius,
                    momentum: valence_momentum,
                },
                broadened_table,
            )?
        } else {
            xcpot_sigma_impl(
                XcpotSigmaInput {
                    exchange_selector: input.exchange_selector,
                    radius: input.radius,
                    core_radius: input.core_radius,
                    momentum: corrected_momentum,
                },
                broadened_table,
            )?
        };
        valence_delta = Some(XcpotSigma {
            real: valence_sigma.real - input.fermi_cache.valence_self_energy.real,
            imaginary: valence_sigma.imaginary - input.fermi_cache.valence_self_energy.imaginary,
        });
    }

    Ok(XcpotSelfEnergyCorrection {
        magnetized_momentum,
        corrected_momentum,
        total_delta,
        valence_delta,
    })
}

/// Port of FEFF `EXCH/xcpot.f90` final potential referencing.
///
/// FEFF sets `eref = v(jri1)`, subtracts that reference from `v`, then either
/// subtracts it from `vval` for `ixc >= 5` or copies `v` into `vval` for lower
/// exchange selectors. The later `lreal` branch only forces `vval` real for
/// `ixc > 4`, which is intentionally preserved here.
pub fn xcpot_reference_shift(
    input: XcpotReferenceShiftInput<'_>,
) -> Result<XcpotReferenceShift, ExchangeError> {
    validate_reference_shift_input(&input)?;

    let exchange_branch = input.exchange_selector % 10;
    let reference_energy = input.total_potential[input.active_len - 1];
    let mut total_potential = Array1::from_iter(
        input
            .total_potential
            .iter()
            .take(input.active_len)
            .map(|&value| value - reference_energy),
    );
    let mut valence_potential = if exchange_branch >= 5 {
        Array1::from_iter(
            input
                .valence_potential
                .iter()
                .take(input.active_len)
                .map(|&value| value - reference_energy),
        )
    } else {
        total_potential.clone()
    };
    let mut reference_energy = reference_energy;

    validate_complex_slice("referenced_total_potential", &total_potential)?;
    validate_complex_slice("referenced_valence_potential", &valence_potential)?;

    if input.lreal > 0 {
        for value in &mut total_potential {
            *value = Complex::new(value.re, 0.0);
        }
        if exchange_branch > 4 {
            for value in &mut valence_potential {
                *value = Complex::new(value.re, 0.0);
            }
        }
        reference_energy = Complex::new(reference_energy.re, 0.0);
    }

    Ok(XcpotReferenceShift {
        reference_energy,
        total_potential,
        valence_potential,
    })
}

/// Port of FEFF `EXCH/xcpot.f90` ground-state/static-potential branch.
///
/// FEFF enters this branch when `ixc == 2` or `Re(em) <= xmu`, copies `vtot`
/// and `vvalgs` into complex work arrays, skips the self-energy calculation,
/// and then falls through the common reference-potential finalization.
pub fn xcpot_ground_state_branch(
    input: XcpotGroundStateBranchInput<'_>,
) -> Result<Option<XcpotReferenceShift>, ExchangeError> {
    let exchange_branch = input.exchange_selector % 10;
    if exchange_branch != 2 && input.energy.re > input.fermi_level {
        return Ok(None);
    }

    validate_ground_state_branch_input(&input)?;

    let total_potential = Array1::from_iter(
        input
            .total_potential
            .iter()
            .take(input.active_len)
            .map(|&value| Complex::new(value, 0.0)),
    );
    let valence_potential = if exchange_branch >= 5 {
        Array1::from_iter(
            input
                .valence_potential
                .iter()
                .take(input.active_len)
                .map(|&value| Complex::new(value, 0.0)),
        )
    } else {
        Array1::from_vec(Vec::new())
    };

    xcpot_reference_shift(XcpotReferenceShiftInput {
        exchange_selector: input.exchange_selector,
        lreal: input.lreal,
        total_potential: total_potential.view(),
        valence_potential: valence_potential.view(),
        active_len: input.active_len,
    })
    .map(Some)
}

/// Composed port of FEFF `EXCH/xcpot.f90` for one energy/potential call.
///
/// This stitches together the already ported `xcpot` sub-blocks: the static
/// ground-state branch, density-radius setup, optional MPSE row deltas, non-MPSE
/// Fermi caches and Dyson correction, delta application, and final reference
/// shift. MPSE callers may either provide a shaped [`XcpotManyPoleDeltaTable`]
/// or raw FEFF pole data for the non-BPR `CSigZ` path.
pub fn xcpot(input: XcpotInput<'_>) -> Result<XcpotResult, ExchangeError> {
    xcpot_impl(input, None)
}

/// Composed FEFF `xcpot` evaluation with an author-supplied `bphl.dat` table
/// available to the broadened-plasmon selector family.
pub fn xcpot_with_broadened_table(
    input: XcpotInput<'_>,
    table: &BroadenedHedinLundqvistTable,
) -> Result<XcpotResult, ExchangeError> {
    xcpot_impl(input, Some(table))
}

fn xcpot_impl(
    input: XcpotInput<'_>,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<XcpotResult, ExchangeError> {
    if let Some(reference) = xcpot_ground_state_branch(XcpotGroundStateBranchInput {
        exchange_selector: input.exchange_selector,
        lreal: input.lreal,
        energy: input.energy,
        fermi_level: input.fermi_level,
        total_potential: input.total_potential,
        valence_potential: input.valence_potential,
        active_len: input.active_len,
    })? {
        return Ok(XcpotResult {
            reference_energy: reference.reference_energy,
            total_potential: reference.total_potential,
            valence_potential: reference.valence_potential,
            fermi_cache: Array1::from_vec(Vec::new()),
            density_grid: None,
        });
    }

    validate_xcpot_input(&input)?;

    let exchange_branch = input.exchange_selector % 10;
    let use_many_pole = input.plasmon_selector > 0 && exchange_branch == 0;
    let density_grid = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: input.plasmon_selector,
        density: input.density,
        radial_match_index_1based: input.active_len - 1,
    })?;
    let many_pole_delta_table = if use_many_pole {
        if let Some(delta_table) = input.many_pole_delta_table {
            Some(delta_table)
        } else {
            let many_pole =
                input
                    .many_pole_self_energy
                    .ok_or(ExchangeError::MissingReferenceData {
                        name: "many_pole_self_energy",
                        value: input.plasmon_selector,
                        data: "WpCorr/Gamma/AmpFac",
                    })?;
            let control = xcpot_many_pole_control(XcpotManyPoleControlInput {
                plasmon_selector: input.plasmon_selector,
                exchange_selector: input.exchange_selector,
                pole_frequencies: many_pole.pole_frequencies,
            })?;
            Some(xcpot_many_pole_self_energy_delta_table(
                XcpotManyPoleSelfEnergyTableInput {
                    plasmon_selector: input.plasmon_selector,
                    energy: input.energy,
                    fermi_level: input.fermi_level,
                    density_grid,
                    pole_frequencies: many_pole.pole_frequencies,
                    pole_widths: many_pole.pole_widths,
                    amplitudes: many_pole.amplitudes,
                    gap_energy: many_pole.gap_energy,
                    active_pole_count: control.active_pole_count,
                    use_broadened_pole: many_pole.use_broadened_pole,
                },
            )?)
        }
    } else {
        None
    };

    let mut total_potential =
        Array1::<Complex>::from_elem(input.active_len, Complex::new(0.0, 0.0));
    let mut valence_potential = if exchange_branch >= 5 {
        Array1::<Complex>::from_elem(input.active_len, Complex::new(0.0, 0.0))
    } else {
        Array1::from_vec(Vec::new())
    };
    let mut fermi_cache = if use_many_pole {
        Vec::new()
    } else {
        vec![zero_xcpot_fermi_cache(); input.active_len]
    };

    for index in (0..input.active_len).rev() {
        let scales = xcpot_local_scales(XcpotLocalScalesInput {
            exchange_selector: input.exchange_selector,
            density: input.density[index],
            magnetization: input.magnetization[index],
            valence_density: input.valence_density[index],
        })?;
        let core_radius = scales.core_radius.unwrap_or(density_grid.core_radius);

        let (total_delta, valence_delta) = if let Some(delta_table) = many_pole_delta_table {
            (
                xcpot_many_pole_row_delta(XcpotManyPoleRowDeltaInput {
                    plasmon_selector: input.plasmon_selector,
                    radius: scales.radius,
                    density_grid,
                    delta_table,
                })?,
                None,
            )
        } else {
            let cache = if let Some(cached) = input.fermi_cache {
                cached[index]
            } else {
                xcpot_fermi_cache_impl(
                    XcpotFermiCacheInput {
                        exchange_selector: input.exchange_selector,
                        radius: scales.radius,
                        core_radius,
                        valence_radius: scales.valence_radius,
                        interstitial: index + 1 == input.active_len,
                    },
                    broadened_table,
                )?
            };
            fermi_cache[index] = cache;

            let correction = xcpot_self_energy_correction_impl(
                XcpotSelfEnergyCorrectionInput {
                    exchange_selector: input.exchange_selector,
                    energy: input.energy.re,
                    fermi_level: input.fermi_level,
                    radius: scales.radius,
                    core_radius,
                    fermi_momentum: scales.fermi_momentum,
                    magnetized_fermi_momentum: scales.magnetized_fermi_momentum,
                    valence_radius: scales.valence_radius,
                    valence_fermi_momentum: scales.valence_fermi_momentum,
                    interstitial: index + 1 == input.active_len,
                    fermi_cache: cache,
                },
                broadened_table,
            )?;
            (correction.total_delta, correction.valence_delta)
        };

        let valence_delta = if exchange_branch >= 5 {
            Some(valence_delta.ok_or(ExchangeError::MissingRequiredInput {
                name: "valence_delta",
                value: exchange_branch,
            })?)
        } else {
            None
        };
        let total_imaginary = if exchange_branch == 5 {
            valence_delta
                .ok_or(ExchangeError::MissingRequiredInput {
                    name: "valence_delta",
                    value: exchange_branch,
                })?
                .imaginary
        } else {
            total_delta.imaginary
        };
        total_potential[index] = Complex::new(
            input.total_potential[index] + total_delta.real,
            total_imaginary,
        );
        if exchange_branch >= 5 {
            let valence_delta = valence_delta.ok_or(ExchangeError::MissingRequiredInput {
                name: "valence_delta",
                value: exchange_branch,
            })?;
            valence_potential[index] = Complex::new(
                input.valence_potential[index] + valence_delta.real,
                valence_delta.imaginary,
            );
        }
    }

    let referenced = xcpot_reference_shift(XcpotReferenceShiftInput {
        exchange_selector: input.exchange_selector,
        lreal: input.lreal,
        total_potential: total_potential.view(),
        valence_potential: valence_potential.view(),
        active_len: input.active_len,
    })?;

    Ok(XcpotResult {
        reference_energy: referenced.reference_energy,
        total_potential: referenced.total_potential,
        valence_potential: referenced.valence_potential,
        fermi_cache: Array1::from_vec(fermi_cache),
        density_grid: Some(density_grid),
    })
}

fn validate_reference_shift_input(
    input: &XcpotReferenceShiftInput<'_>,
) -> Result<(), ExchangeError> {
    if input.active_len == 0 {
        return Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: input.active_len,
        });
    }
    if input.total_potential.len() < input.active_len {
        return Err(ExchangeError::LengthTooShort {
            name: "total_potential",
            required: input.active_len,
            actual: input.total_potential.len(),
        });
    }
    for index in 0..input.active_len {
        ensure_finite_complex("total_potential", input.total_potential[index])?;
    }

    let exchange_branch = input.exchange_selector % 10;
    if exchange_branch >= 5 {
        if input.valence_potential.len() < input.active_len {
            return Err(ExchangeError::LengthTooShort {
                name: "valence_potential",
                required: input.active_len,
                actual: input.valence_potential.len(),
            });
        }
        for index in 0..input.active_len {
            ensure_finite_complex("valence_potential", input.valence_potential[index])?;
        }
    }

    Ok(())
}

fn validate_ground_state_branch_input(
    input: &XcpotGroundStateBranchInput<'_>,
) -> Result<(), ExchangeError> {
    if input.active_len == 0 {
        return Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: input.active_len,
        });
    }
    if input.total_potential.len() < input.active_len {
        return Err(ExchangeError::LengthTooShort {
            name: "total_potential",
            required: input.active_len,
            actual: input.total_potential.len(),
        });
    }
    for index in 0..input.active_len {
        ensure_finite("total_potential", input.total_potential[index])?;
    }
    if input.exchange_selector % 10 >= 5 {
        if input.valence_potential.len() < input.active_len {
            return Err(ExchangeError::LengthTooShort {
                name: "valence_potential",
                required: input.active_len,
                actual: input.valence_potential.len(),
            });
        }
        for index in 0..input.active_len {
            ensure_finite("valence_potential", input.valence_potential[index])?;
        }
    }
    Ok(())
}

fn validate_xcpot_input(input: &XcpotInput<'_>) -> Result<(), ExchangeError> {
    ensure_finite_complex("energy", input.energy)?;
    ensure_finite("fermi_level", input.fermi_level)?;
    if input.active_len < 2 {
        return Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: input.active_len,
        });
    }

    validate_real_slice_prefix("total_potential", input.total_potential, input.active_len)?;
    validate_real_slice_prefix("density", input.density, input.active_len)?;
    validate_real_slice_prefix("magnetization", input.magnetization, input.active_len)?;
    validate_real_slice_prefix("valence_density", input.valence_density, input.active_len)?;

    if input.exchange_selector % 10 >= 5 {
        validate_real_slice_prefix(
            "valence_potential",
            input.valence_potential,
            input.active_len,
        )?;
    }

    let use_many_pole = input.plasmon_selector > 0 && input.exchange_selector % 10 == 0;
    if !use_many_pole
        && let Some(cache) = input.fermi_cache
        && cache.len() < input.active_len
    {
        return Err(ExchangeError::LengthTooShort {
            name: "fermi_cache",
            required: input.active_len,
            actual: cache.len(),
        });
    }

    Ok(())
}

fn validate_self_energy_application_input(
    input: &XcpotSelfEnergyApplicationInput<'_>,
) -> Result<(), ExchangeError> {
    if input.active_len == 0 {
        return Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: input.active_len,
        });
    }

    validate_real_slice_prefix("total_potential", input.total_potential, input.active_len)?;
    validate_real_slice_prefix("delta_real", input.delta_real, input.active_len)?;

    let exchange_branch = input.exchange_selector % 10;
    if exchange_branch != 5 {
        validate_real_slice_prefix("delta_imaginary", input.delta_imaginary, input.active_len)?;
    }

    if exchange_branch >= 5 {
        validate_real_slice_prefix(
            "valence_potential",
            input.valence_potential,
            input.active_len,
        )?;
        validate_real_slice_prefix(
            "valence_delta_real",
            input.valence_delta_real,
            input.active_len,
        )?;
        validate_real_slice_prefix(
            "valence_delta_imaginary",
            input.valence_delta_imaginary,
            input.active_len,
        )?;
    }

    Ok(())
}

fn validate_self_energy_correction_input(
    input: XcpotSelfEnergyCorrectionInput,
) -> Result<(), ExchangeError> {
    ensure_finite("energy", input.energy)?;
    ensure_finite("fermi_level", input.fermi_level)?;
    ensure_positive("rs", input.radius)?;
    ensure_finite("rscore", input.core_radius)?;
    ensure_positive("fermi_momentum", input.fermi_momentum)?;
    ensure_positive("magnetized_fermi_momentum", input.magnetized_fermi_momentum)?;
    ensure_finite(
        "total_self_energy.real",
        input.fermi_cache.total_self_energy.real,
    )?;
    ensure_finite(
        "total_self_energy.imaginary",
        input.fermi_cache.total_self_energy.imaginary,
    )?;
    ensure_finite(
        "valence_self_energy.real",
        input.fermi_cache.valence_self_energy.real,
    )?;
    ensure_finite(
        "valence_self_energy.imaginary",
        input.fermi_cache.valence_self_energy.imaginary,
    )?;
    ensure_finite("ground_state_ratio", input.fermi_cache.ground_state_ratio)?;

    if input.exchange_selector % 10 == 5 {
        match input.valence_radius {
            Some(radius) => ensure_positive("valence_radius", radius)?,
            None => {
                return Err(ExchangeError::MissingRequiredInput {
                    name: "valence_radius",
                    value: 5,
                });
            }
        }
        match input.valence_fermi_momentum {
            Some(momentum) => ensure_positive("valence_fermi_momentum", momentum)?,
            None => {
                return Err(ExchangeError::MissingRequiredInput {
                    name: "valence_fermi_momentum",
                    value: 5,
                });
            }
        }
    }

    Ok(())
}

fn validate_many_pole_delta_table_input(
    input: &XcpotManyPoleDeltaTableInput<'_>,
) -> Result<(), ExchangeError> {
    if input.plasmon_selector <= 0 {
        return Err(ExchangeError::InvalidSelector {
            name: "plasmon_selector",
            value: input.plasmon_selector,
        });
    }
    if input.fermi_self_energy.len() < XCPOT_MPSE_GRID_POINTS {
        return Err(ExchangeError::LengthTooShort {
            name: "fermi_self_energy",
            required: XCPOT_MPSE_GRID_POINTS,
            actual: input.fermi_self_energy.len(),
        });
    }
    if input.energy_self_energy.len() < XCPOT_MPSE_GRID_POINTS {
        return Err(ExchangeError::LengthTooShort {
            name: "energy_self_energy",
            required: XCPOT_MPSE_GRID_POINTS,
            actual: input.energy_self_energy.len(),
        });
    }
    ensure_finite_complex("renormalization", input.renormalization)?;

    if input.plasmon_selector == 2 {
        for index in 0..XCPOT_MPSE_GRID_POINTS {
            ensure_finite_complex("fermi_self_energy", input.fermi_self_energy[index])?;
            ensure_finite_complex("energy_self_energy", input.energy_self_energy[index])?;
        }
    } else {
        let last_index = XCPOT_MPSE_GRID_POINTS - 1;
        ensure_finite_complex("fermi_self_energy", input.fermi_self_energy[last_index])?;
        ensure_finite_complex("energy_self_energy", input.energy_self_energy[last_index])?;
    }

    Ok(())
}

fn validate_many_pole_self_energy_table_input(
    input: &XcpotManyPoleSelfEnergyTableInput<'_>,
) -> Result<(), ExchangeError> {
    if input.plasmon_selector <= 0 {
        return Err(ExchangeError::InvalidSelector {
            name: "plasmon_selector",
            value: input.plasmon_selector,
        });
    }
    ensure_finite_complex("energy", input.energy)?;
    ensure_finite("fermi_level", input.fermi_level)?;
    ensure_finite("EGap", input.gap_energy)?;
    if input.active_pole_count == 0 {
        return Err(ExchangeError::InvalidIndex {
            name: "active_pole_count",
            index: input.active_pole_count,
        });
    }
    validate_real_slice_prefix(
        "pole_frequencies",
        input.pole_frequencies,
        input.active_pole_count,
    )?;
    validate_real_slice_prefix("pole_widths", input.pole_widths, input.active_pole_count)?;
    validate_real_slice_prefix("amplitudes", input.amplitudes, input.active_pole_count)?;
    for &radius in &input.density_grid.radii {
        ensure_positive("density_grid.radii", radius)?;
    }
    Ok(())
}

fn map_self_energy_error(error: SelfEnergyError) -> ExchangeError {
    match error {
        SelfEnergyError::NonFiniteComplex { name, value } => {
            ExchangeError::NonFiniteComplex { name, value }
        }
        SelfEnergyError::NonFiniteReal { name, value } => {
            ExchangeError::NonFiniteInput { name, value }
        }
        SelfEnergyError::NonPositiveReal { name, value }
        | SelfEnergyError::NonPositiveTolerance { name, value } => {
            ExchangeError::NonPositiveInput { name, value }
        }
        SelfEnergyError::NegativeReal { name, value } => {
            ExchangeError::NegativeInput { name, value }
        }
        SelfEnergyError::NegativeRadicand { name, value } => {
            ExchangeError::NegativeRadicand { name, value }
        }
        SelfEnergyError::ZeroDenominator { name } => ExchangeError::ZeroDenominator { name },
        SelfEnergyError::LengthTooShort {
            name,
            required,
            actual,
        } => ExchangeError::LengthTooShort {
            name,
            required,
            actual,
        },
        SelfEnergyError::InvalidPoleCount => ExchangeError::InvalidIndex {
            name: "active_pole_count",
            index: 0,
        },
        SelfEnergyError::InvalidFunction { .. } => ExchangeError::SelfEnergyFailure {
            routine: "CSigZ",
            detail: "invalid singularity selector",
        },
        SelfEnergyError::InvalidIntegrationInterval { .. } => ExchangeError::SelfEnergyFailure {
            routine: "CSigZ",
            detail: "invalid integration interval",
        },
        SelfEnergyError::TooManySingularities { .. } => ExchangeError::SelfEnergyFailure {
            routine: "CSigZ",
            detail: "too many singularities",
        },
        SelfEnergyError::InvalidSingularity { .. } => ExchangeError::SelfEnergyFailure {
            routine: "CSigZ",
            detail: "invalid singularity",
        },
        SelfEnergyError::TooManyIntegrationRegions { .. } => ExchangeError::SelfEnergyFailure {
            routine: "CSigZ",
            detail: "too many integration regions",
        },
        SelfEnergyError::LossGridLengthMismatch { .. }
        | SelfEnergyError::InsufficientLossGrid { .. }
        | SelfEnergyError::NonIncreasingLossEnergy { .. }
        | SelfEnergyError::ZeroMoment { .. }
        | SelfEnergyError::SpecialFunction(_)
        | SelfEnergyError::Root(_)
        | SelfEnergyError::Interpolation(_) => ExchangeError::SelfEnergyFailure {
            routine: "CSigZ",
            detail: "internal self-energy calculation",
        },
    }
}

fn sqrt_nonnegative(name: &'static str, value: Real) -> Result<Real, ExchangeError> {
    ensure_finite(name, value)?;
    if value < 0.0 {
        Err(ExchangeError::NegativeRadicand { name, value })
    } else {
        Ok(value.sqrt())
    }
}

fn validate_real_slice_prefix(
    name: &'static str,
    values: ndarray::ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), ExchangeError> {
    if values.len() < active_len {
        return Err(ExchangeError::LengthTooShort {
            name,
            required: active_len,
            actual: values.len(),
        });
    }
    for index in 0..active_len {
        ensure_finite(name, values[index])?;
    }
    Ok(())
}

fn validate_complex_slice(
    name: &'static str,
    values: &Array1<Complex>,
) -> Result<(), ExchangeError> {
    for &value in values {
        ensure_finite_complex(name, value)?;
    }
    Ok(())
}

fn density_radius_or(density: Real, fallback: Real) -> Real {
    if density <= 0.0 {
        fallback
    } else {
        density_to_wigner_seitz_radius(density)
    }
}

fn density_to_wigner_seitz_radius(density: Real) -> Real {
    (3.0 / (4.0 * FEFF_PI * density)).powf(1.0 / 3.0)
}

fn zero_xcpot_fermi_cache() -> XcpotFermiCache {
    XcpotFermiCache {
        total_self_energy: XcpotSigma {
            real: 0.0,
            imaginary: 0.0,
        },
        valence_self_energy: XcpotSigma {
            real: 0.0,
            imaginary: 0.0,
        },
        ground_state_ratio: 1.0,
    }
}
