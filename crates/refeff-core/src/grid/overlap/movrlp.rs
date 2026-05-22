use super::super::radial::{loucks_index_below, loucks_radius};
use super::super::validation::*;
use super::super::*;
use super::geometry::{sphere_overlap_cap_volume, sphere_overlap_lens_volume};

/// Construct FEFF's muffin-tin overlap matrix from `POT/movrlp.f90`.
///
/// FEFF stores only a moving `novp = 50` radial window for each potential and
/// appends one equation for the interstitial potential. This function builds
/// that active matrix, applies FEFF-compatible single-complex LU factorization,
/// and returns the factors for downstream `ovp2mt`-style solves.
pub fn muffin_tin_overlap_matrix(
    input: MuffinTinOverlapMatrixInput<'_>,
) -> Result<MuffinTinOverlapMatrix, GridError> {
    validate_muffin_tin_overlap_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let active_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .and_then(|value| value.checked_add(1))
        .ok_or(GridError::GridLengthOverflow { name: "movrlp" })?;
    let radii = (1..=251).map(loucks_radius).collect::<Array1<_>>();
    let grid_half_step = (LOUCKS_DELTA / 2.0).exp();
    let radius_mode = (input.interstitial_selector - (input.interstitial_selector % 2)) / 2;
    let absorber_only = input.interstitial_selector % 2 == 1;

    let mut matrix = Array2::<Complex32>::zeros((active_order, active_order));
    for row in 0..active_order {
        for column in 0..(active_order - 1) {
            matrix[(row, column)] = Complex32::new(0.0, 0.0);
        }
        matrix[(row, row)] = Complex32::new(1.0, 0.0);
        matrix[(row, active_order - 1)] = Complex32::new(0.01, 0.0);
    }

    let mut bmat = Array2::<f32>::zeros((potential_count, active_order - 1));
    let mut interstitial_volume = input.interstitial_volume;
    validate_finite_scalar("interstitial_volume", interstitial_volume)?;
    let mut atom_count = 0.0;

    for target in 0..potential_count {
        let rav = movrlp_average_radius(input, &radii, target, radius_mode)?;
        let neighbors = movrlp_neighbors(input, target)?;
        for neighbor in neighbors {
            let source = neighbor.source_potential;
            let distance = neighbor.distance;
            let multiplicity = neighbor.multiplicity as Real;
            let pair = MovrlpPair {
                target,
                source,
                distance,
                multiplicity,
            };

            if distance < input.muffin_tin_radii[target] + input.muffin_tin_radii[source] {
                interstitial_volume += input.potential_multiplicities[target]
                    * multiplicity
                    * sphere_overlap_cap_volume(
                        input.muffin_tin_radii[target],
                        input.muffin_tin_radii[source],
                        distance,
                    )?;
            }

            if rav + input.muffin_tin_radii[source] > distance {
                movrlp_fill_boundary_row(input, &radii, &mut bmat, pair, rav, grid_half_step)?;
            }

            if input.muffin_tin_radii[target] + input.muffin_tin_radii[source] > distance {
                movrlp_fill_overlap_matrix(input, &radii, &mut matrix, pair, grid_half_step)?;
            }
        }
        atom_count += input.potential_multiplicities[target];
    }
    validate_nonzero_finite_scalar("atom_count", atom_count)?;

    if absorber_only {
        for column in 0..(active_order - 1) {
            matrix[(active_order - 1, column)] += Complex32::new(bmat[(0, column)], 0.0);
        }
    } else {
        for potential in 0..potential_count {
            let weight = (input.potential_multiplicities[potential] / atom_count) as f32;
            for column in 0..(active_order - 1) {
                matrix[(active_order - 1, column)] +=
                    Complex32::new(weight * bmat[(potential, column)], 0.0);
            }
        }
    }

    let lu = complex32_lu_factor(matrix.view())?;
    let final_pivot =
        lu.pivots()
            .get(active_order - 1)
            .copied()
            .ok_or(GridError::LengthTooShort {
                name: "movrlp_pivots",
                required: active_order,
                actual: lu.pivots().len(),
            })?;
    if final_pivot != active_order {
        return Err(GridError::IllegalFinalPivot {
            expected: active_order,
            actual: final_pivot,
        });
    }

    Ok(MuffinTinOverlapMatrix {
        radii,
        lu,
        interstitial_volume,
        active_order,
    })
}

/// Project overlapped potentials or densities onto FEFF muffin-tin spheres.
///
/// This ports `POT/ovp2mt.f90`. FEFF solves only the active `novp = 50`
/// radial window for each potential; when the interstitial potential is fixed
/// or the input is a density, it intentionally solves a prefix of the LU system
/// produced by `movrlp`. This function preserves that behavior and returns a
/// cloned output table rather than mutating the caller's array.
pub fn project_muffin_tin_overlap(
    input: MuffinTinOverlapProjectionInput<'_>,
) -> Result<MuffinTinOverlapProjection, GridError> {
    validate_muffin_tin_projection_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let window_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;
    let full_order = window_order
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;
    let solve_order = match input.mode {
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial => full_order,
        MuffinTinOverlapProjectionMode::Density { .. }
        | MuffinTinOverlapProjectionMode::PotentialFixedInterstitial => window_order,
    };

    let mut rhs = Array1::<Complex32>::zeros(solve_order);
    for potential in 0..potential_count {
        let first_row = input.muffin_tin_indices[potential] - MOVRLP_NOVP;
        for offset in 0..MOVRLP_NOVP {
            let mut value = input.values[(first_row + offset, potential)];
            if input.mode == MuffinTinOverlapProjectionMode::PotentialFixedInterstitial {
                value -= input.interstitial_value;
            }
            rhs[potential * MOVRLP_NOVP + offset] =
                Complex32::new(movrlp_real32("ovp2mt_rhs", value)?, 0.0);
        }
    }

    let absorber_only = input.interstitial_selector % 2 == 1;
    let radius_mode = input.interstitial_selector / 2;
    if input.mode == MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial {
        let average_values = ovp2mt_average_values(input, potential_count, radius_mode)?;
        let last_potential = if absorber_only {
            0
        } else {
            input.highest_potential_index
        };
        let mut average_sum = 0.0;
        let mut multiplicity_sum = 0.0;
        for potential in 0..=last_potential {
            average_sum += average_values[potential] * input.potential_multiplicities[potential];
            multiplicity_sum += input.potential_multiplicities[potential];
        }
        validate_nonzero_finite_scalar("ovp2mt_multiplicity_sum", multiplicity_sum)?;
        rhs[window_order] = Complex32::new(
            movrlp_real32("ovp2mt_rhs", average_sum / multiplicity_sum)?,
            0.0,
        );
    }

    let solved =
        complex32_lu_solve_prefix_vector(&input.overlap_matrix.lu, rhs.view(), solve_order)?;
    let window_values = solved
        .iter()
        .take(window_order)
        .map(|value| value.re as Real)
        .collect::<Array1<_>>();
    let mut output_values = input.values.to_owned();

    let interstitial_value = match input.mode {
        MuffinTinOverlapProjectionMode::Density { total_charge } => {
            total_charge - ovp2mt_density_muffin_tin_charge(input, &window_values, potential_count)?
        }
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial => {
            solved[window_order].re as Real / 100.0
        }
        MuffinTinOverlapProjectionMode::PotentialFixedInterstitial => input.interstitial_value,
    };

    match input.mode {
        MuffinTinOverlapProjectionMode::Density { .. } => {}
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial
        | MuffinTinOverlapProjectionMode::PotentialFixedInterstitial => {
            ovp2mt_rewrite_potentials(
                input,
                &window_values,
                interstitial_value,
                potential_count,
                &mut output_values,
            )?;
        }
    }

    Ok(MuffinTinOverlapProjection {
        values: output_values,
        interstitial_value,
        window_values,
    })
}

fn ovp2mt_average_values(
    input: MuffinTinOverlapProjectionInput<'_>,
    potential_count: usize,
    radius_mode: usize,
) -> Result<Array1<Real>, GridError> {
    let mut average_values = Array1::<Real>::zeros(potential_count);
    let radii = input.radii.iter().copied().collect::<Vec<_>>();
    for potential in 0..potential_count {
        let active_len =
            checked_index_offset("norman_indices", input.norman_indices[potential], 2)?;
        let values = input
            .values
            .column(potential)
            .iter()
            .take(active_len)
            .copied()
            .collect::<Vec<_>>();
        let average_radius = muffin_tin_average_radius(
            input.radii,
            input.muffin_tin_indices,
            input.muffin_tin_radii,
            input.norman_radii,
            input.near_neighbor_flags,
            potential,
            radius_mode,
        )?;
        average_values[potential] = terp(&radii[..active_len], &values, 3, average_radius)?.value;
    }
    Ok(average_values)
}

fn ovp2mt_density_muffin_tin_charge(
    input: MuffinTinOverlapProjectionInput<'_>,
    window_values: &Array1<Real>,
    potential_count: usize,
) -> Result<Real, GridError> {
    let radii = input.radii.iter().take(251).copied().collect::<Vec<_>>();
    let mut total_charge = 0.0;
    for potential in 0..potential_count {
        let muffin_index = input.muffin_tin_indices[potential];
        let active_len = checked_index_offset("muffin_tin_indices", muffin_index, 2)?;
        let window_start = muffin_index - MOVRLP_NOVP + 1;
        let mut density_moment = Array1::<Real>::zeros(251);
        for radial_index in 1..=muffin_index {
            let density = if radial_index < window_start {
                input.values[(radial_index - 1, potential)]
            } else {
                let window_index = potential * MOVRLP_NOVP + radial_index - window_start;
                window_values[window_index]
            };
            density_moment[radial_index - 1] = density * input.radii[radial_index - 1].powi(2);
        }

        let interpolation_radii = radii[..muffin_index].to_vec();
        let interpolation_values = density_moment
            .iter()
            .take(muffin_index)
            .copied()
            .collect::<Vec<_>>();
        for radial_index in (muffin_index + 1)..=active_len {
            density_moment[radial_index - 1] = terp(
                &interpolation_radii,
                &interpolation_values,
                2,
                input.radii[radial_index - 1],
            )?
            .value;
        }

        let density_values = density_moment
            .iter()
            .take(active_len)
            .copied()
            .collect::<Vec<_>>();
        let charge = somm2(
            &radii[..active_len],
            &density_values,
            LOUCKS_DELTA,
            0.0,
            input.muffin_tin_radii[potential],
            0,
        )?;
        total_charge += input.potential_multiplicities[potential] * charge;
    }
    Ok(total_charge)
}

fn ovp2mt_rewrite_potentials(
    input: MuffinTinOverlapProjectionInput<'_>,
    window_values: &Array1<Real>,
    interstitial_value: Real,
    potential_count: usize,
    output_values: &mut Array2<Real>,
) -> Result<(), GridError> {
    let radii = input.radii.iter().take(251).copied().collect::<Vec<_>>();
    for potential in 0..potential_count {
        let muffin_index = input.muffin_tin_indices[potential];
        let tail_start = checked_index_offset("muffin_tin_indices", muffin_index, 1)?;
        let first_row = muffin_index - MOVRLP_NOVP;
        for offset in 0..MOVRLP_NOVP {
            output_values[(first_row + offset, potential)] =
                window_values[potential * MOVRLP_NOVP + offset] + interstitial_value;
        }

        let interpolation_values = output_values
            .column(potential)
            .iter()
            .take(muffin_index)
            .copied()
            .collect::<Vec<_>>();
        output_values[(muffin_index, potential)] = terp(
            &radii[..muffin_index],
            &interpolation_values,
            2,
            input.radii[muffin_index],
        )?
        .value;
        for radial_index in tail_start..251 {
            output_values[(radial_index, potential)] = interstitial_value;
        }
    }
    Ok(())
}

fn complex32_lu_solve_prefix_vector(
    lu: &Complex32Lu,
    right_hand_side: ArrayView1<'_, Complex32>,
    order: usize,
) -> Result<Array1<Complex32>, GridError> {
    if right_hand_side.len() != order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side",
            left: right_hand_side.len(),
            right_name: "solve order",
            right: order,
        }
        .into());
    }
    ensure_shape("overlap_lu", lu.factors().shape(), order, order)?;
    ensure_len("overlap_pivots", lu.pivots().len(), order)?;

    let factors = lu.factors();
    let mut solution = right_hand_side.to_owned();
    for (pivot, &pivot_row) in lu.pivots().iter().take(order).enumerate() {
        if pivot_row == 0 || pivot_row > order {
            return Err(GridError::InvalidGridIndex {
                name: "overlap_pivot",
                index: pivot_row,
            });
        }
        let swap_row = pivot_row - 1;
        if swap_row != pivot {
            let left = solution[pivot];
            solution[pivot] = solution[swap_row];
            solution[swap_row] = left;
        }
    }

    for pivot in 0..order {
        for row in (pivot + 1)..order {
            let factor = factors[(row, pivot)];
            let pivot_value = solution[pivot];
            solution[row] -= factor * pivot_value;
        }
    }

    for pivot in (0..order).rev() {
        let diagonal = factors[(pivot, pivot)];
        if diagonal == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot }.into());
        }
        solution[pivot] /= diagonal;
        let pivot_value = solution[pivot];
        for row in 0..pivot {
            let factor = factors[(row, pivot)];
            solution[row] -= factor * pivot_value;
        }
    }

    Ok(solution)
}

fn muffin_tin_average_radius(
    radii: ArrayView1<'_, Real>,
    muffin_tin_indices: ArrayView1<'_, usize>,
    muffin_tin_radii: ArrayView1<'_, Real>,
    norman_radii: ArrayView1<'_, Real>,
    near_neighbor_flags: ArrayView1<'_, bool>,
    potential: usize,
    radius_mode: usize,
) -> Result<Real, GridError> {
    let after_muffin = radii[movrlp_radii_index_after_muffin(muffin_tin_indices[potential])?];
    if near_neighbor_flags[potential] {
        return Ok(after_muffin);
    }
    if radius_mode == 1 {
        Ok((muffin_tin_radii[potential] + norman_radii[potential]) / 2.0)
    } else if radius_mode == 0 {
        Ok(norman_radii[potential])
    } else {
        Ok(after_muffin)
    }
}

fn movrlp_average_radius(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    potential: usize,
    radius_mode: usize,
) -> Result<Real, GridError> {
    muffin_tin_average_radius(
        radii.view(),
        input.muffin_tin_indices,
        input.muffin_tin_radii,
        input.norman_radii,
        input.near_neighbor_flags,
        potential,
        radius_mode,
    )
}

fn movrlp_radii_index_after_muffin(muffin_tin_index: usize) -> Result<usize, GridError> {
    muffin_tin_index
        .checked_add(1)
        .filter(|&index| index <= 251)
        .map(|index| index - 1)
        .ok_or(GridError::SourceGridTooShort {
            name: "radii",
            required: muffin_tin_index.saturating_add(1),
            available: 251,
        })
}

fn movrlp_neighbors(
    input: MuffinTinOverlapMatrixInput<'_>,
    target: usize,
) -> Result<Vec<MuffinTinOverlapNeighbor>, GridError> {
    let explicit = input.explicit_overlaps[target];
    if !explicit.is_empty() {
        return Ok(explicit.to_vec());
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
        neighbors.push(MuffinTinOverlapNeighbor {
            source_potential: input.atom_potentials[atom],
            multiplicity: 1,
            distance: distance_between(center, position),
        });
    }
    Ok(neighbors)
}

#[derive(Debug, Clone, Copy)]
struct MovrlpPair {
    target: usize,
    source: usize,
    distance: Real,
    multiplicity: Real,
}

fn movrlp_fill_boundary_row(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    bmat: &mut Array2<f32>,
    pair: MovrlpPair,
    average_radius: Real,
    grid_half_step: Real,
) -> Result<(), GridError> {
    let check_index = loucks_index_below(pair.distance - average_radius)?;
    if input.muffin_tin_indices[pair.source].saturating_sub(check_index) >= MOVRLP_NOVP - 1 {
        return Err(GridError::MuffinTinOverlapTooLarge {
            left: pair.target,
            right: pair.source,
        });
    }
    let start = movrlp_window_start(input.muffin_tin_indices[pair.source], pair.source)?;
    for radial in start..=input.muffin_tin_indices[pair.source] {
        let radius = radii[radial - 1];
        let mut r1 = radius / grid_half_step;
        let mut r2 = radius * grid_half_step;
        if radial == input.muffin_tin_indices[pair.source] {
            r2 = input.muffin_tin_radii[pair.source];
            r1 = (r1 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                - input.muffin_tin_radii[pair.source])
                / 2.0;
        }
        if radial + 1 == input.muffin_tin_indices[pair.source] {
            r2 = (r2 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                - input.muffin_tin_radii[pair.source])
                / 2.0;
        }
        if r2 + average_radius < pair.distance {
            continue;
        }

        if r1 + average_radius < pair.distance {
            let mut fraction = (pair.distance - average_radius - r1) / (r2 - r1);
            r1 = pair.distance - average_radius;
            let contribution = (r2.powi(2) - r1.powi(2)) / (4.0 * pair.distance * average_radius)
                * pair.multiplicity;
            let neighbor_index = if radial == input.muffin_tin_indices[pair.source] {
                radial - 1
            } else {
                radial + 1
            };
            fraction *= (r2 - radius) / (radii[neighbor_index - 1] - radius);
            let column = pair.source * MOVRLP_NOVP + radial - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution * (1.0 - fraction))?;
            let column = pair.source * MOVRLP_NOVP + neighbor_index - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution * fraction)?;
        } else {
            let contribution = (r2.powi(2) - r1.powi(2)) / (4.0 * pair.distance * average_radius)
                * pair.multiplicity;
            let column = pair.source * MOVRLP_NOVP + radial - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution)?;
        }
    }
    Ok(())
}

fn movrlp_fill_overlap_matrix(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    matrix: &mut Array2<Complex32>,
    pair: MovrlpPair,
    grid_half_step: Real,
) -> Result<(), GridError> {
    let check_target = loucks_index_below(pair.distance - input.muffin_tin_radii[pair.source])?;
    let check_source = loucks_index_below(pair.distance - input.muffin_tin_radii[pair.target])?;
    if input.muffin_tin_indices[pair.target].saturating_sub(check_target) >= MOVRLP_NOVP - 1
        || input.muffin_tin_indices[pair.source].saturating_sub(check_source) >= MOVRLP_NOVP - 1
    {
        return Err(GridError::MuffinTinOverlapTooLarge {
            left: pair.target,
            right: pair.source,
        });
    }

    let target_start = movrlp_window_start(input.muffin_tin_indices[pair.target], pair.target)?;
    let source_start = movrlp_window_start(input.muffin_tin_indices[pair.source], pair.source)?;
    for target_radial in target_start..=input.muffin_tin_indices[pair.target] {
        let target_radius = radii[target_radial - 1];
        let mut target_r1 = target_radius / grid_half_step;
        let mut target_r2 = target_radius * grid_half_step;
        if target_radial == input.muffin_tin_indices[pair.target] {
            target_r2 = input.muffin_tin_radii[pair.target];
            target_r1 = (target_r1 + 2.0 * radii[input.muffin_tin_indices[pair.target] - 1]
                - input.muffin_tin_radii[pair.target])
                / 2.0;
        }
        if target_radial + 1 == input.muffin_tin_indices[pair.target] {
            target_r2 = (target_r2 + 2.0 * radii[input.muffin_tin_indices[pair.target] - 1]
                - input.muffin_tin_radii[pair.target])
                / 2.0;
        }
        let target_column = pair.target * MOVRLP_NOVP + target_radial - target_start;

        for source_radial in source_start..=input.muffin_tin_indices[pair.source] {
            let source_radius = radii[source_radial - 1];
            let mut source_r1 = source_radius / grid_half_step;
            let mut source_r2 = source_radius * grid_half_step;
            if source_radial == input.muffin_tin_indices[pair.source] {
                source_r2 = input.muffin_tin_radii[pair.source];
                source_r1 = (source_r1 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                    - input.muffin_tin_radii[pair.source])
                    / 2.0;
            }
            if source_radial + 1 == input.muffin_tin_indices[pair.source] {
                source_r2 = (source_r2 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                    - input.muffin_tin_radii[pair.source])
                    / 2.0;
            }
            if source_r2 + target_r2 < pair.distance {
                continue;
            }

            let mut contribution = sphere_overlap_lens_volume(target_r2, source_r2, pair.distance)?;
            if target_r1 + source_r2 > pair.distance {
                contribution -= sphere_overlap_lens_volume(target_r1, source_r2, pair.distance)?;
            }
            if target_r2 + source_r1 > pair.distance {
                contribution -= sphere_overlap_lens_volume(target_r2, source_r1, pair.distance)?;
            }
            if target_r1 + source_r1 > pair.distance {
                contribution += sphere_overlap_lens_volume(target_r1, source_r1, pair.distance)?;
            }
            contribution = contribution
                / (4.0 / 3.0 * PI * (target_r2.powi(3) - target_r1.powi(3)))
                * pair.multiplicity;

            if source_r1 + target_r2 < pair.distance {
                let mut fraction =
                    (pair.distance - target_radius - source_r1) / (source_r2 - source_r1);
                let neighbor_index = if source_radial == input.muffin_tin_indices[pair.source] {
                    source_radial - 1
                } else {
                    source_radial + 1
                };
                fraction *=
                    (source_r2 - source_radius) / (radii[neighbor_index - 1] - source_radius);
                let column = pair.source * MOVRLP_NOVP + source_radial - source_start;
                matrix[(target_column, column)] += Complex32::new(
                    movrlp_real32("cmovp", contribution * (1.0 - fraction))?,
                    0.0,
                );
                let column = pair.source * MOVRLP_NOVP + neighbor_index - source_start;
                matrix[(target_column, column)] +=
                    Complex32::new(movrlp_real32("cmovp", contribution * fraction)?, 0.0);
            } else {
                let column = pair.source * MOVRLP_NOVP + source_radial - source_start;
                matrix[(target_column, column)] +=
                    Complex32::new(movrlp_real32("cmovp", contribution)?, 0.0);
            }
        }
    }
    Ok(())
}

fn movrlp_window_start(muffin_tin_index: usize, potential: usize) -> Result<usize, GridError> {
    if muffin_tin_index < MOVRLP_NOVP {
        Err(GridError::MuffinTinIndexTooSmall {
            name: "muffin_tin_indices",
            potential,
            minimum: MOVRLP_NOVP,
            index: muffin_tin_index,
        })
    } else {
        Ok(muffin_tin_index - MOVRLP_NOVP + 1)
    }
}

fn movrlp_real32(name: &'static str, value: Real) -> Result<f32, GridError> {
    validate_finite_scalar(name, value)?;
    let narrowed = value as f32;
    if narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
}

fn validate_muffin_tin_overlap_input(
    input: MuffinTinOverlapMatrixInput<'_>,
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
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len(
        "explicit_overlaps",
        input.explicit_overlaps.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_indices",
        input.muffin_tin_indices.len(),
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
    validate_real_values("muffin_tin_radii", input.muffin_tin_radii)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    for potential in 0..potential_count {
        validate_positive_finite_scalar(
            "potential_multiplicities",
            input.potential_multiplicities[potential],
        )?;
        validate_positive_finite_scalar("muffin_tin_radii", input.muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        if input.muffin_tin_indices[potential] < MOVRLP_NOVP {
            return Err(GridError::MuffinTinIndexTooSmall {
                name: "muffin_tin_indices",
                potential,
                minimum: MOVRLP_NOVP,
                index: input.muffin_tin_indices[potential],
            });
        }
        if input.muffin_tin_indices[potential] >= 251 {
            return Err(GridError::SourceGridTooShort {
                name: "radii",
                required: input.muffin_tin_indices[potential] + 1,
                available: 251,
            });
        }
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

fn validate_muffin_tin_projection_input(
    input: MuffinTinOverlapProjectionInput<'_>,
) -> Result<(), GridError> {
    let potential_count = input
        .highest_potential_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "potential" })?;
    let window_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;
    let full_order = window_order
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;

    ensure_shape("values", input.values.shape(), 251, potential_count)?;
    ensure_len("radii", input.radii.len(), 251)?;
    ensure_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len(
        "norman_indices",
        input.norman_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_indices",
        input.muffin_tin_indices.len(),
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
    if input.overlap_matrix.active_order != full_order {
        return Err(GridError::OverlapMatrixOrderMismatch {
            required: full_order,
            actual: input.overlap_matrix.active_order,
        });
    }
    ensure_shape(
        "overlap_lu",
        input.overlap_matrix.lu.factors().shape(),
        full_order,
        full_order,
    )?;
    ensure_len(
        "overlap_pivots",
        input.overlap_matrix.lu.pivots().len(),
        full_order,
    )?;

    validate_positive_radii(input.radii, 251)?;
    validate_real_table("values", input.values)?;
    validate_real_values("potential_multiplicities", input.potential_multiplicities)?;
    validate_real_values("muffin_tin_radii", input.muffin_tin_radii)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_finite_scalar("interstitial_value", input.interstitial_value)?;
    if let MuffinTinOverlapProjectionMode::Density { total_charge } = input.mode {
        validate_finite_scalar("total_charge", total_charge)?;
    }

    for potential in 0..potential_count {
        validate_positive_finite_scalar(
            "potential_multiplicities",
            input.potential_multiplicities[potential],
        )?;
        validate_positive_finite_scalar("muffin_tin_radii", input.muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        if input.muffin_tin_indices[potential] < MOVRLP_NOVP {
            return Err(GridError::MuffinTinIndexTooSmall {
                name: "muffin_tin_indices",
                potential,
                minimum: MOVRLP_NOVP,
                index: input.muffin_tin_indices[potential],
            });
        }
        let muffin_required =
            checked_index_offset("muffin_tin_indices", input.muffin_tin_indices[potential], 2)?;
        let norman_required =
            checked_index_offset("norman_indices", input.norman_indices[potential], 2)?;
        ensure_source_length("values", muffin_required, input.values.nrows())?;
        ensure_source_length("radii", muffin_required, input.radii.len())?;
        ensure_source_length("values", norman_required, input.values.nrows())?;
        ensure_source_length("radii", norman_required, input.radii.len())?;
        validate_grid_index("muffin_tin_indices", input.muffin_tin_indices[potential])?;
        validate_grid_index("norman_indices", input.norman_indices[potential])?;
    }

    Ok(())
}
