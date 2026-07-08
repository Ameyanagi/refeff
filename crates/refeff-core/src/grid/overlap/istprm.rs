use super::super::radial::{loucks_index_below, loucks_radius};
use super::super::validation::*;
use super::super::*;
use super::geometry::sphere_overlap_lens_volume;
use crate::exchange::{
    ExchangeError, dirac_hara_exchange_potential, karasiev_sjostrom_dufty_trickey_vxc,
    perdew_zunger_vxc, perrot_dharma_wardana_vxc, von_barth_hedin_potential,
};
use ndarray::{Axis, Slice};

/// Build FEFF `istprm` first-call muffin-tin radii and overlap limits.
///
/// This ports the radius/nearest-neighbor block guarded by `rmt(0) <= 0` in
/// `POT/istprm.f90`. It computes `rmt`, `inrm`, `lnear`, nearest-neighbor
/// bookkeeping, and the `folpx` reductions consumed by later AFOLP/overlap
/// passes. The density, exchange-correlation, `movrlp`, and `ovp2mt` work later
/// in `istprm` remains separate and composes with the existing Rust overlap
/// helpers.
pub fn muffin_tin_radius_parameters(
    input: MuffinTinRadiusParametersInput<'_>,
) -> Result<MuffinTinRadiusParameters, GridError> {
    validate_muffin_tin_radius_parameters_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let mut interstitial_selector = input.interstitial_selector;
    let mut muffin_tin_radii = Array1::<Real>::zeros(potential_count);
    let norman_radii = input.norman_radii.to_owned();
    let mut norman_indices = Array1::<usize>::zeros(potential_count);
    let mut max_overlap_factors = input.max_overlap_factors.to_owned();
    let mut near_neighbor_flags = Array1::<bool>::from_elem(potential_count, false);
    let mut nearest_neighbor_distances = Array1::<Real>::zeros(potential_count);
    let mut nearest_neighbor_potentials = Array1::<usize>::zeros(potential_count);
    let mut norman_radius_fallbacks = Array1::<bool>::from_elem(potential_count, false);

    for potential in 0..potential_count {
        if !input.explicit_overlaps[potential].is_empty() {
            interstitial_selector %= 6;
        }

        let norman_index = loucks_index_below(input.norman_radii[potential])?;
        validate_grid_index("norman_indices", norman_index)?;
        norman_indices[potential] = norman_index;

        let neighbors = istprm_neighbors(input, potential)?;
        if neighbors.is_empty() {
            return Err(GridError::NoMuffinTinNeighbor { potential });
        }

        let mut nearest_distance = Real::INFINITY;
        let mut nearest_potential = 0usize;
        let mut weighted_radius_sum = 0.0;
        let mut overlap_volume_sum = 0.0;
        let mut matching_radius = None;
        let explicit = !input.explicit_overlaps[potential].is_empty();

        for neighbor in neighbors {
            if neighbor.distance <= nearest_distance {
                nearest_distance = neighbor.distance;
                nearest_potential = neighbor.source_potential;
            }

            let norman_sum =
                input.norman_radii[potential] + input.norman_radii[neighbor.source_potential];
            let no_overlap = if explicit {
                norman_sum <= neighbor.distance
            } else {
                norman_sum < neighbor.distance
            };
            if no_overlap {
                continue;
            }

            if interstitial_selector < 6 {
                let lens_volume = sphere_overlap_lens_volume(
                    input.norman_radii[potential],
                    input.norman_radii[neighbor.source_potential],
                    neighbor.distance,
                )?;
                let muffin_radius = neighbor.distance
                    * input.overlap_factors[potential]
                    * input.norman_radii[potential]
                    / norman_sum;
                weighted_radius_sum += muffin_radius * lens_volume * neighbor.multiplicity;
                overlap_volume_sum += lens_volume * neighbor.multiplicity;
            } else if let Some(radius) =
                matching_point_radius(input, potential, norman_index, neighbor)?
            {
                matching_radius = Some(match matching_radius {
                    Some(current) if current <= radius => current,
                    _ => radius,
                });
            }
        }

        nearest_neighbor_distances[potential] = nearest_distance;
        nearest_neighbor_potentials[potential] = nearest_potential;
        near_neighbor_flags[potential] = input.norman_radii[potential] >= nearest_distance;

        let radius = if interstitial_selector < 6 {
            if weighted_radius_sum <= 0.0 {
                norman_radius_fallbacks[potential] = true;
                input.norman_radii[potential]
            } else {
                validate_nonzero_finite_scalar("istprm_overlap_volume", overlap_volume_sum)?;
                let mut radius = weighted_radius_sum / overlap_volume_sum;
                if near_neighbor_flags[potential] {
                    radius = near_neighbor_muffin_tin_radius(input, potential, nearest_distance)?;
                }
                radius
            }
        } else if let Some(radius) = matching_radius {
            radius
        } else {
            norman_radius_fallbacks[potential] = true;
            input.norman_radii[potential]
        };
        validate_positive_finite_scalar("muffin_tin_radii", radius)?;
        muffin_tin_radii[potential] = radius;
    }

    reduce_max_overlap_factors(
        input.afolp_enabled,
        &muffin_tin_radii,
        input.norman_radii,
        &near_neighbor_flags,
        &nearest_neighbor_distances,
        &nearest_neighbor_potentials,
        &mut max_overlap_factors,
    )?;

    Ok(MuffinTinRadiusParameters {
        muffin_tin_radii,
        norman_radii,
        norman_indices,
        max_overlap_factors,
        near_neighbor_flags,
        nearest_neighbor_distances,
        nearest_neighbor_potentials,
        norman_radius_fallbacks,
        interstitial_selector,
    })
}

/// Run FEFF `istprm` density, XC-potential, overlap-projection, and Fermi setup.
///
/// This composes the second half of `POT/istprm.f90`: `sidx` tail adjustment,
/// ground-state `vtot`/`vvalgs` construction, `movrlp` overlap matrix
/// assembly, `ovp2mt` density and potential projection, FEFF's `vint >= xmu`
/// fixed-potential retry, and the final `fermi` calculation.
pub fn muffin_tin_interstitial_parameters(
    input: MuffinTinInterstitialParametersInput<'_>,
) -> Result<MuffinTinInterstitialParameters, GridError> {
    validate_muffin_tin_interstitial_parameters_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let exchange_branch = input.exchange_selector % 10;
    let mut total_potential = Array2::<Real>::zeros((251, potential_count));
    let mut valence_potential = Array2::<Real>::zeros((251, potential_count));
    let mut max_density_indices = Array1::<usize>::zeros(potential_count);
    let mut muffin_tin_indices = Array1::<usize>::zeros(potential_count);
    let mut norman_indices = Array1::<usize>::zeros(potential_count);
    let mut norman_radii = input.norman_radii.to_owned();

    for potential in 0..potential_count {
        let density_prefix_rows = input
            .electron_density
            .slice_axis(Axis(0), Slice::from(0..250));
        let indices = overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: density_prefix_rows.column(potential),
            muffin_tin_radius: input.muffin_tin_radii[potential],
            norman_radius: norman_radii[potential],
        })?;
        max_density_indices[potential] = indices.max_density_index;
        muffin_tin_indices[potential] = indices.muffin_tin_index;
        norman_indices[potential] = indices.norman_index;
        norman_radii[potential] = indices.norman_radius;

        for radial in 0..indices.max_density_index {
            let density = input.electron_density[(radial, potential)];
            let magnetization = input.magnetization[(radial, potential)];
            let (density_radius, spin_fraction_twice) = if density <= 0.0 {
                (100.0, 1.0)
            } else {
                (
                    (density / 3.0).powf(-1.0 / 3.0),
                    1.0 + input.spin_polarization as Real * magnetization,
                )
            };
            let exchange_correlation =
                istprm_exchange_correlation_potential(input, density_radius, spin_fraction_twice)?;
            total_potential[(radial, potential)] =
                input.coulomb_potential[(radial, potential)] + exchange_correlation;

            if exchange_branch == 5 {
                let valence_density = input.valence_density[(radial, potential)];
                let mut valence_radius = 10.0;
                if valence_density > 1.0e-5 {
                    valence_radius = (valence_density / 3.0).powf(-1.0 / 3.0);
                }
                if valence_radius > 10.0 {
                    valence_radius = 10.0;
                }
                let valence_spin_fraction_twice = 1.0
                    + input.spin_polarization as Real * magnetization * density / valence_density;
                let valence_exchange_correlation =
                    von_barth_hedin_potential(valence_radius, valence_spin_fraction_twice)?;
                valence_potential[(radial, potential)] =
                    input.coulomb_potential[(radial, potential)] + valence_exchange_correlation;
            } else if exchange_branch >= 6 {
                let valence_density = input.valence_density[(radial, potential)];
                let core_radius = if density <= valence_density {
                    101.0
                } else {
                    ((density - valence_density) / 3.0).powf(-1.0 / 3.0)
                };
                let magnetized_radius =
                    (density * (1.0 + input.spin_polarization as Real * magnetization) / 3.0)
                        .powf(-1.0 / 3.0);
                let magnetized_fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / magnetized_radius;
                let dirac_hara =
                    dirac_hara_exchange_potential(core_radius, magnetized_fermi_momentum)?;
                valence_potential[(radial, potential)] =
                    input.coulomb_potential[(radial, potential)] + exchange_correlation
                        - dirac_hara;
            }
        }
    }

    let initial_volume = istprm_initial_interstitial_volume(
        input.total_volume,
        norman_radii.view(),
        input.muffin_tin_radii,
        input.potential_multiplicities,
    )?;
    let average_norman_radius =
        istprm_average_norman_radius(norman_radii.view(), input.potential_multiplicities)?;

    let overlap_matrix = muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
        highest_potential_index: input.highest_potential_index,
        atom_potentials: input.atom_potentials,
        atom_positions: input.atom_positions,
        representative_atoms: input.representative_atoms,
        potential_multiplicities: input.potential_multiplicities,
        explicit_overlaps: input.explicit_overlaps,
        muffin_tin_indices: muffin_tin_indices.view(),
        muffin_tin_radii: input.muffin_tin_radii,
        norman_radii: norman_radii.view(),
        near_neighbor_flags: input.near_neighbor_flags,
        interstitial_selector: input.interstitial_selector,
        interstitial_volume: initial_volume,
    })?;
    if overlap_matrix.interstitial_volume <= 0.0 {
        return Err(GridError::NoInterstitialVolume {
            volume: overlap_matrix.interstitial_volume,
        });
    }

    let projected_density = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
        highest_potential_index: input.highest_potential_index,
        values: input.electron_density,
        radii: overlap_matrix.radii.view(),
        potential_multiplicities: input.potential_multiplicities,
        norman_indices: norman_indices.view(),
        muffin_tin_indices: muffin_tin_indices.view(),
        muffin_tin_radii: input.muffin_tin_radii,
        norman_radii: norman_radii.view(),
        near_neighbor_flags: input.near_neighbor_flags,
        overlap_matrix: &overlap_matrix,
        interstitial_selector: input.interstitial_selector,
        interstitial_value: 0.0,
        mode: MuffinTinOverlapProjectionMode::Density {
            total_charge: input.total_charge,
        },
    })?;
    let interstitial_density =
        4.0 * PI * projected_density.interstitial_value / overlap_matrix.interstitial_volume;
    validate_positive_finite_scalar("interstitial_density", interstitial_density)?;

    if input.exchange_selector >= 5 {
        valence_potential = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
            highest_potential_index: input.highest_potential_index,
            values: valence_potential.view(),
            radii: overlap_matrix.radii.view(),
            potential_multiplicities: input.potential_multiplicities,
            norman_indices: norman_indices.view(),
            muffin_tin_indices: muffin_tin_indices.view(),
            muffin_tin_radii: input.muffin_tin_radii,
            norman_radii: norman_radii.view(),
            near_neighbor_flags: input.near_neighbor_flags,
            overlap_matrix: &overlap_matrix,
            interstitial_selector: input.interstitial_selector,
            interstitial_value: 0.0,
            mode: MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial,
        })?
        .values;
    }

    let projected_total = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
        highest_potential_index: input.highest_potential_index,
        values: total_potential.view(),
        radii: overlap_matrix.radii.view(),
        potential_multiplicities: input.potential_multiplicities,
        norman_indices: norman_indices.view(),
        muffin_tin_indices: muffin_tin_indices.view(),
        muffin_tin_radii: input.muffin_tin_radii,
        norman_radii: norman_radii.view(),
        near_neighbor_flags: input.near_neighbor_flags,
        overlap_matrix: &overlap_matrix,
        interstitial_selector: input.interstitial_selector,
        interstitial_value: 0.0,
        mode: MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial,
    })?;
    let mut total_potential = projected_total.values;
    let mut interstitial_potential = projected_total.interstitial_value;
    let mut interstitial_potential_limited = false;

    if interstitial_potential >= input.fermi_level {
        interstitial_potential = input.fermi_level - 0.05;
        total_potential = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
            highest_potential_index: input.highest_potential_index,
            values: total_potential.view(),
            radii: overlap_matrix.radii.view(),
            potential_multiplicities: input.potential_multiplicities,
            norman_indices: norman_indices.view(),
            muffin_tin_indices: muffin_tin_indices.view(),
            muffin_tin_radii: input.muffin_tin_radii,
            norman_radii: norman_radii.view(),
            near_neighbor_flags: input.near_neighbor_flags,
            overlap_matrix: &overlap_matrix,
            interstitial_selector: input.interstitial_selector,
            interstitial_value: interstitial_potential,
            mode: MuffinTinOverlapProjectionMode::PotentialFixedInterstitial,
        })?
        .values;
        interstitial_potential_limited = true;
    }

    let fermi = interstitial_fermi_level(FermiLevelInput {
        interstitial_density,
        interstitial_potential,
    })?;

    Ok(MuffinTinInterstitialParameters {
        total_potential,
        valence_potential,
        max_density_indices,
        muffin_tin_indices,
        muffin_tin_radii: input.muffin_tin_radii.to_owned(),
        norman_indices,
        norman_radii,
        average_norman_radius,
        interstitial_volume: overlap_matrix.interstitial_volume,
        interstitial_potential,
        interstitial_density,
        fermi,
        interstitial_potential_limited,
    })
}

#[derive(Debug, Clone, Copy)]
struct IstprmNeighbor {
    source_potential: usize,
    multiplicity: Real,
    distance: Real,
}

fn istprm_neighbors(
    input: MuffinTinRadiusParametersInput<'_>,
    target: usize,
) -> Result<Vec<IstprmNeighbor>, GridError> {
    let explicit = input.explicit_overlaps[target];
    if !explicit.is_empty() {
        return Ok(explicit
            .iter()
            .map(|neighbor| IstprmNeighbor {
                source_potential: neighbor.source_potential,
                multiplicity: neighbor.multiplicity as Real,
                distance: neighbor.distance,
            })
            .collect());
    }

    let representative = input.representative_atoms[target];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut neighbors = Vec::new();
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        neighbors.push(IstprmNeighbor {
            source_potential: input.atom_potentials[atom],
            multiplicity: 1.0,
            distance: distance_between(center, position),
        });
    }
    Ok(neighbors)
}

fn matching_point_radius(
    input: MuffinTinRadiusParametersInput<'_>,
    target: usize,
    target_norman_index: usize,
    neighbor: IstprmNeighbor,
) -> Result<Option<Real>, GridError> {
    let neighbor_edge = neighbor.distance - input.norman_radii[target];
    let neighbor_index = loucks_index_below(neighbor_edge)?;
    if neighbor_index < 2 {
        return Err(GridError::InvalidGridIndex {
            name: "matching_point_neighbor_index",
            index: neighbor_index,
        });
    }
    ensure_source_length(
        "coulomb_potential",
        target_norman_index + 1,
        input.coulomb_potential.nrows(),
    )?;
    ensure_source_length(
        "coulomb_potential",
        neighbor_index,
        input.coulomb_potential.nrows(),
    )?;

    for radial in (1..=target_norman_index).rev() {
        let target_potential = input.coulomb_potential[(radial - 1, target)];
        let source_potential =
            input.coulomb_potential[(neighbor_index - 1, neighbor.source_potential)];
        if target_potential <= source_potential {
            let left_radius = loucks_radius(radial);
            let right_radius = loucks_radius(radial + 1);
            let neighbor_left_radius = loucks_radius(neighbor_index - 1);
            let neighbor_radius = loucks_radius(neighbor_index);
            let target_slope = (input.coulomb_potential[(radial, target)] - target_potential)
                / (right_radius - left_radius);
            let source_slope = (source_potential
                - input.coulomb_potential[(neighbor_index - 2, neighbor.source_potential)])
                / (neighbor_radius - neighbor_left_radius);
            let denominator = target_slope + source_slope;
            validate_nonzero_finite_scalar("matching_point_slope", denominator)?;
            let radius = left_radius
                + (source_potential
                    + source_slope * (neighbor.distance - left_radius - neighbor_radius)
                    - target_potential)
                    / denominator;
            validate_positive_finite_scalar("matching_point_radius", radius)?;
            return Ok(Some(radius));
        }
    }

    Ok(None)
}

fn near_neighbor_muffin_tin_radius(
    input: MuffinTinRadiusParametersInput<'_>,
    potential: usize,
    nearest_distance: Real,
) -> Result<Real, GridError> {
    let mut maximum_index = loucks_index_below(nearest_distance)?.checked_sub(1).ok_or(
        GridError::InvalidGridIndex {
            name: "nearest_neighbor_index",
            index: 0,
        },
    )?;
    loop {
        ensure_source_length(
            "coulomb_potential",
            maximum_index + 1,
            input.coulomb_potential.nrows(),
        )?;
        if input.coulomb_potential[(maximum_index - 1, potential)]
            < input.coulomb_potential[(maximum_index, potential)]
        {
            let radius = loucks_radius(maximum_index) - 0.0001;
            validate_positive_finite_scalar("near_neighbor_muffin_tin_radius", radius)?;
            return Ok(radius);
        }
        if maximum_index == 1 {
            return Err(GridError::NoMuffinTinMatchingPoint {
                target: potential,
                source_potential: potential,
                distance: nearest_distance,
            });
        }
        maximum_index -= 1;
    }
}

fn reduce_max_overlap_factors(
    afolp_enabled: bool,
    muffin_tin_radii: &Array1<Real>,
    norman_radii: ArrayView1<'_, Real>,
    near_neighbor_flags: &Array1<bool>,
    nearest_neighbor_distances: &Array1<Real>,
    nearest_neighbor_potentials: &Array1<usize>,
    max_overlap_factors: &mut Array1<Real>,
) -> Result<(), GridError> {
    let window_tail = (-(MOVRLP_NOVP as Real - 3.0) * LOUCKS_DELTA).exp();
    for potential in 0..muffin_tin_radii.len() {
        let nearest_potential = nearest_neighbor_potentials[potential];
        validate_positive_finite_scalar("muffin_tin_radii", muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("muffin_tin_radii", muffin_tin_radii[nearest_potential])?;

        let base = if afolp_enabled { 0.2 } else { 0.3 };
        let norman_weight = if afolp_enabled { 0.8 } else { 0.7 };
        let mut limit =
            base + norman_weight * norman_radii[potential] / muffin_tin_radii[potential];
        validate_finite_scalar("folpx_limit", limit)?;
        if limit < max_overlap_factors[potential] {
            max_overlap_factors[potential] = limit;
        }

        limit = nearest_neighbor_distances[potential] / muffin_tin_radii[potential] / 1.06;
        validate_finite_scalar("folpx_limit", limit)?;
        if limit < max_overlap_factors[potential] {
            max_overlap_factors[potential] = limit;
        }

        if near_neighbor_flags[potential] {
            limit = nearest_neighbor_distances[potential]
                / (muffin_tin_radii[potential] * 1.05
                    + window_tail * muffin_tin_radii[nearest_potential]);
            validate_finite_scalar("folpx_limit", limit)?;
            if limit < max_overlap_factors[potential] {
                max_overlap_factors[potential] = limit;
            }
            if limit < max_overlap_factors[nearest_potential] {
                max_overlap_factors[nearest_potential] = limit;
            }
        } else {
            limit = (nearest_neighbor_distances[potential] - norman_radii[potential])
                / (window_tail * muffin_tin_radii[nearest_potential]);
            validate_finite_scalar("folpx_limit", limit)?;
            if limit < max_overlap_factors[nearest_potential] {
                max_overlap_factors[nearest_potential] = limit;
            }
        }
    }
    validate_real_values("max_overlap_factors", max_overlap_factors.view())?;
    Ok(())
}

fn validate_muffin_tin_radius_parameters_input(
    input: MuffinTinRadiusParametersInput<'_>,
) -> Result<(), GridError> {
    let potential_count = input
        .highest_potential_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "potential" })?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        potential_count,
    )?;
    ensure_len(
        "explicit_overlaps",
        input.explicit_overlaps.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len(
        "overlap_factors",
        input.overlap_factors.len(),
        potential_count,
    )?;
    ensure_len(
        "max_overlap_factors",
        input.max_overlap_factors.len(),
        potential_count,
    )?;
    ensure_shape(
        "coulomb_potential",
        input.coulomb_potential.shape(),
        251,
        potential_count,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(GridError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values("atom_potentials", input.atom_potentials, potential_count)?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_real_values("overlap_factors", input.overlap_factors)?;
    validate_real_values("max_overlap_factors", input.max_overlap_factors)?;
    validate_real_table("coulomb_potential", input.coulomb_potential)?;

    for potential in 0..potential_count {
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        validate_positive_finite_scalar("overlap_factors", input.overlap_factors[potential])?;
        validate_positive_finite_scalar(
            "max_overlap_factors",
            input.max_overlap_factors[potential],
        )?;
        for neighbor in input.explicit_overlaps[potential] {
            if neighbor.source_potential >= potential_count {
                return Err(GridError::InvalidPotentialIndex {
                    name: "explicit_overlaps.source_potential",
                    index: neighbor.source_potential,
                    available: potential_count,
                });
            }
            if neighbor.multiplicity == 0 {
                return Err(GridError::InvalidGridIndex {
                    name: "explicit_overlaps.multiplicity",
                    index: 0,
                });
            }
            validate_positive_finite_scalar("explicit_overlaps.distance", neighbor.distance)?;
        }
    }

    Ok(())
}

fn istprm_exchange_correlation_potential(
    input: MuffinTinInterstitialParametersInput<'_>,
    density_radius: Real,
    spin_fraction_twice: Real,
) -> Result<Real, GridError> {
    match input.scf_exchange_selector {
        11 => Ok(von_barth_hedin_potential(
            density_radius,
            spin_fraction_twice,
        )?),
        12 => Ok(perdew_zunger_vxc(density_radius)?),
        21 => Ok(perrot_dharma_wardana_vxc(
            density_radius,
            input.scf_temperature_hartree,
        )?),
        22 => Ok(karasiev_sjostrom_dufty_trickey_vxc(
            density_radius,
            input.scf_temperature_hartree,
        )?),
        selector => Err(ExchangeError::InvalidSelector {
            name: "iscfxc",
            value: selector,
        }
        .into()),
    }
}

fn istprm_initial_interstitial_volume(
    total_volume: Real,
    norman_radii: ArrayView1<'_, Real>,
    muffin_tin_radii: ArrayView1<'_, Real>,
    potential_multiplicities: ArrayView1<'_, Real>,
) -> Result<Real, GridError> {
    let mut norman_volume = 0.0;
    let mut interstitial_volume = 0.0;
    for potential in 0..norman_radii.len() {
        norman_volume += potential_multiplicities[potential] * norman_radii[potential].powi(3);
        interstitial_volume -=
            potential_multiplicities[potential] * muffin_tin_radii[potential].powi(3);
    }
    let volume = if total_volume <= 0.0 {
        4.0 * PI / 3.0 * (interstitial_volume + norman_volume)
    } else {
        4.0 * PI / 3.0 * interstitial_volume + total_volume
    };
    validate_finite_scalar("interstitial_volume", volume)?;
    Ok(volume)
}

fn istprm_average_norman_radius(
    norman_radii: ArrayView1<'_, Real>,
    potential_multiplicities: ArrayView1<'_, Real>,
) -> Result<Real, GridError> {
    let mut norman_volume = 0.0;
    let mut multiplicity_sum = 0.0;
    for potential in 0..norman_radii.len() {
        norman_volume += potential_multiplicities[potential] * norman_radii[potential].powi(3);
        multiplicity_sum += potential_multiplicities[potential];
    }
    validate_nonzero_finite_scalar("potential_multiplicity_sum", multiplicity_sum)?;
    let radius = (norman_volume / multiplicity_sum).powf(1.0 / 3.0);
    validate_positive_finite_scalar("average_norman_radius", radius)?;
    Ok(radius)
}

fn validate_muffin_tin_interstitial_parameters_input(
    input: MuffinTinInterstitialParametersInput<'_>,
) -> Result<(), GridError> {
    let potential_count = input
        .highest_potential_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "potential" })?;
    ensure_len(
        "atom_potentials",
        input.atom_potentials.len(),
        input.atom_positions.nrows(),
    )?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        potential_count,
    )?;
    ensure_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len(
        "explicit_overlaps",
        input.explicit_overlaps.len(),
        potential_count,
    )?;
    ensure_shape(
        "electron_density",
        input.electron_density.shape(),
        251,
        potential_count,
    )?;
    ensure_shape(
        "valence_density",
        input.valence_density.shape(),
        251,
        potential_count,
    )?;
    ensure_shape(
        "magnetization",
        input.magnetization.shape(),
        251,
        potential_count,
    )?;
    ensure_shape(
        "coulomb_potential",
        input.coulomb_potential.shape(),
        251,
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_radii",
        input.muffin_tin_radii.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len(
        "near_neighbor_flags",
        input.near_neighbor_flags.len(),
        potential_count,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(GridError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values("atom_potentials", input.atom_potentials, potential_count)?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;
    validate_real_values("potential_multiplicities", input.potential_multiplicities)?;
    validate_real_table("electron_density", input.electron_density)?;
    validate_real_table("valence_density", input.valence_density)?;
    validate_real_table("magnetization", input.magnetization)?;
    validate_real_table("coulomb_potential", input.coulomb_potential)?;
    validate_real_values("muffin_tin_radii", input.muffin_tin_radii)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_finite_scalar("scf_temperature_hartree", input.scf_temperature_hartree)?;
    validate_finite_scalar("total_charge", input.total_charge)?;
    validate_finite_scalar("fermi_level", input.fermi_level)?;
    validate_finite_scalar("total_volume", input.total_volume)?;

    for potential in 0..potential_count {
        validate_positive_finite_scalar(
            "potential_multiplicities",
            input.potential_multiplicities[potential],
        )?;
        validate_positive_finite_scalar("muffin_tin_radii", input.muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        for neighbor in input.explicit_overlaps[potential] {
            if neighbor.source_potential >= potential_count {
                return Err(GridError::InvalidPotentialIndex {
                    name: "explicit_overlaps.source_potential",
                    index: neighbor.source_potential,
                    available: potential_count,
                });
            }
            if neighbor.multiplicity == 0 {
                return Err(GridError::InvalidGridIndex {
                    name: "explicit_overlaps.multiplicity",
                    index: 0,
                });
            }
            validate_positive_finite_scalar("explicit_overlaps.distance", neighbor.distance)?;
        }
    }

    Ok(())
}
