use super::*;

mod coupling;
mod lanczos;
mod matrix;
mod rigid_body;
mod spectral;
mod thermal;

pub use coupling::*;
pub use lanczos::*;
pub use matrix::*;
pub use rigid_body::*;
pub use spectral::*;
pub use thermal::*;

pub(super) fn validate_dmdw_atom_positions(
    atom_positions: ArrayView2<'_, Real>,
) -> Result<(), DebyeError> {
    if atom_positions.ncols() != 3 {
        return Err(DebyeError::InvalidDmdwAtomShape {
            rows: atom_positions.nrows(),
            columns: atom_positions.ncols(),
        });
    }
    if atom_positions.nrows() == 0 {
        return Err(DebyeError::EmptyDmdwAtomTable);
    }
    for value in atom_positions.iter().copied() {
        ensure_finite("DMDW atom coordinate", value)?;
    }
    Ok(())
}

pub(super) fn validate_dmdw_atoms(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    validate_dmdw_atom_positions(atom_positions)?;
    if atom_positions.nrows() != atom_masses.len() {
        return Err(DebyeError::InvalidDmdwMassCount {
            positions: atom_positions.nrows(),
            masses: atom_masses.len(),
        });
    }
    for mass in atom_masses.iter().copied() {
        ensure_positive("DMDW atom mass", mass)?;
    }
    Ok(())
}

fn validate_dmdw_seed_projection(
    seed: ArrayView1<'_, Real>,
    projection_modes: ArrayView2<'_, Real>,
) -> Result<(), DebyeError> {
    validate_dmdw_seed(seed)?;
    if projection_modes.nrows() != seed.len() {
        return Err(DebyeError::InvalidDmdwProjectionShape {
            seed_len: seed.len(),
            rows: projection_modes.nrows(),
            columns: projection_modes.ncols(),
        });
    }
    for value in projection_modes.iter().copied() {
        ensure_finite("DMDW projection mode", value)?;
    }
    Ok(())
}

fn normalize_dmdw_projection_modes(modes: &mut Array2<Real>) -> Result<(), DebyeError> {
    for mode_index in 0..modes.ncols() {
        let norm = modes
            .column(mode_index)
            .iter()
            .map(|value| value * value)
            .sum::<Real>()
            .sqrt();
        ensure_finite_output("DMDW projection mode norm", norm)?;
        if norm == 0.0 {
            return Err(DebyeError::ZeroDmdwProjectionModeNorm { mode: mode_index });
        }
        for value in modes.column_mut(mode_index) {
            *value /= norm;
        }
    }
    Ok(())
}

fn validate_dmdw_force_blocks(
    force_blocks: ArrayView4<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    let shape = force_blocks.shape();
    if atom_masses.is_empty() {
        return Err(DebyeError::EmptyDmdwAtomTable);
    }
    if shape[0] != atom_masses.len()
        || shape[1] != atom_masses.len()
        || shape[2] != 3
        || shape[3] != 3
    {
        return Err(DebyeError::InvalidDmdwBlockShape {
            atoms_i: shape[0],
            atoms_j: shape[1],
            components_i: shape[2],
            components_j: shape[3],
            masses: atom_masses.len(),
        });
    }
    for mass in atom_masses.iter().copied() {
        ensure_positive("DMDW atom mass", mass)?;
    }
    for value in force_blocks.iter().copied() {
        ensure_finite("DMDW force block", value)?;
    }
    Ok(())
}

fn validate_dmdw_ir_dipoles(
    atom_masses: ArrayView1<'_, Real>,
    dipole_derivatives: ArrayView3<'_, Real>,
) -> Result<(), DebyeError> {
    if atom_masses.is_empty() {
        return Err(DebyeError::EmptyDmdwAtomTable);
    }
    if dipole_derivatives.shape() != [atom_masses.len(), 3, 3] {
        let shape = dipole_derivatives.shape();
        return Err(DebyeError::InvalidDmdwDipoleDerivativeShape {
            atoms: shape[0],
            displacements: shape[1],
            dipoles: shape[2],
            masses: atom_masses.len(),
        });
    }
    for mass in atom_masses.iter().copied() {
        ensure_positive("DMDW atom mass", mass)?;
    }
    for value in dipole_derivatives.iter().copied() {
        ensure_finite("DMDW IR dipole derivative", value)?;
    }
    Ok(())
}

fn validate_dmdw_lanczos_inputs(
    dynamical_matrix: ArrayView2<'_, Real>,
    seed: ArrayView1<'_, Real>,
    pole_count: usize,
) -> Result<(), DebyeError> {
    if pole_count == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole count",
            value: pole_count as Real,
        });
    }
    if dynamical_matrix.nrows() == 0
        || dynamical_matrix.nrows() != dynamical_matrix.ncols()
        || dynamical_matrix.nrows() != seed.len()
    {
        return Err(DebyeError::InvalidDmdwLanczosShape {
            rows: dynamical_matrix.nrows(),
            columns: dynamical_matrix.ncols(),
            seed_len: seed.len(),
        });
    }
    for value in dynamical_matrix.iter().copied() {
        ensure_finite("DMDW Lanczos matrix", value)?;
    }
    validate_dmdw_seed(seed)?;
    Ok(())
}

fn validate_dmdw_lanczos_polynomial_inputs(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    if order == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW Lanczos polynomial order",
            value: order as Real,
        });
    }
    if alpha.len() < order || beta.len() < order {
        return Err(DebyeError::InvalidDmdwLanczosPolynomialShape {
            order,
            alpha_len: alpha.len(),
            beta_len: beta.len(),
        });
    }
    ensure_finite("DMDW Lanczos polynomial x", x)?;
    for value in alpha.iter().take(order).copied() {
        ensure_finite("DMDW Lanczos alpha", value)?;
    }
    for value in beta.iter().take(order).copied() {
        ensure_finite("DMDW Lanczos beta", value)?;
    }
    Ok(())
}

fn validate_dmdw_lanczos_pole_search_inputs(
    order: usize,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
    search_limit: Real,
    samples_per_pole: usize,
) -> Result<(), DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, 0.0, alpha, beta)?;
    ensure_positive("DMDW Lanczos pole search limit", search_limit)?;
    if samples_per_pole == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole samples per pole",
            value: samples_per_pole as Real,
        });
    }
    Ok(())
}

fn validate_dmdw_pole_thermal_inputs(
    temperatures: ArrayView1<'_, Real>,
    angular_frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    if temperatures.is_empty() {
        return Err(DebyeError::EmptyDmdwTemperatureTable);
    }
    validate_dmdw_frequency_weight_poles(angular_frequencies, weights)?;
    for temperature in temperatures.iter().copied() {
        ensure_positive("DMDW temperature", temperature)?;
    }
    Ok(())
}

fn validate_dmdw_frequency_weight_poles(
    frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    if frequencies.len() != weights.len() {
        return Err(DebyeError::InvalidDmdwPoleTableShape {
            frequencies: frequencies.len(),
            weights: weights.len(),
        });
    }
    if frequencies.is_empty() {
        return Err(DebyeError::EmptyDmdwPoleTable);
    }
    for frequency in frequencies.iter().copied() {
        ensure_finite("DMDW pole frequency", frequency)?;
    }
    for weight in weights.iter().copied() {
        ensure_finite("DMDW pole weight", weight)?;
    }
    Ok(())
}

fn validate_dmdw_coupling_tables(
    pds_energy_hartree: ArrayView1<'_, Real>,
    phonon_dos: ArrayView1<'_, Real>,
    a2f_energy_hartree: ArrayView1<'_, Real>,
    eliashberg: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    let point_count = pds_energy_hartree.len();
    if phonon_dos.len() != point_count
        || a2f_energy_hartree.len() != point_count
        || eliashberg.len() != point_count
    {
        return Err(DebyeError::InvalidDmdwCouplingTableShape {
            pds_energy: point_count,
            phonon_dos: phonon_dos.len(),
            a2f_energy: a2f_energy_hartree.len(),
            eliashberg: eliashberg.len(),
        });
    }
    if point_count == 0 {
        return Err(DebyeError::EmptyDmdwCouplingTable);
    }

    for row in 0..point_count {
        let pds_energy = pds_energy_hartree[row];
        let a2f_energy = a2f_energy_hartree[row];
        let density = phonon_dos[row];
        ensure_finite("DMDW PDS energy", pds_energy)?;
        ensure_finite("DMDW a2f energy", a2f_energy)?;
        ensure_finite("DMDW Eliashberg coupling", eliashberg[row])?;

        let tolerance =
            DMDW_COUPLING_GRID_TOLERANCE * pds_energy.abs().max(a2f_energy.abs()).max(1.0);
        if (pds_energy - a2f_energy).abs() > tolerance {
            return Err(DebyeError::MismatchedDmdwCouplingEnergyGrid {
                row: row + 1,
                pds_energy,
                a2f_energy,
            });
        }
        if !density.is_finite() {
            return Err(DebyeError::NonFinite {
                name: "DMDW phonon density",
                value: density,
            });
        }
        if density <= 0.0 {
            return Err(DebyeError::NonPositiveDmdwPhononDensity {
                row: row + 1,
                value: density,
            });
        }
    }
    Ok(())
}

fn validate_dmdw_phonon_coupling_result(coupling: &DmdwPhononCoupling) -> Result<(), DebyeError> {
    let point_count = coupling.point_count();
    if coupling.energy_ev.len() != point_count
        || coupling.eliashberg.len() != point_count
        || coupling.matrix_element.len() != point_count
    {
        return Err(DebyeError::InvalidDmdwCouplingTableShape {
            pds_energy: point_count,
            phonon_dos: coupling.energy_ev.len(),
            a2f_energy: coupling.eliashberg.len(),
            eliashberg: coupling.matrix_element.len(),
        });
    }
    if point_count == 0 {
        return Err(DebyeError::EmptyDmdwCouplingTable);
    }
    for row in 0..point_count {
        ensure_finite("DMDW coupling energy Hartree", coupling.energy_hartree[row])?;
        ensure_finite("DMDW coupling energy eV", coupling.energy_ev[row])?;
        ensure_finite("DMDW Eliashberg coupling", coupling.eliashberg[row])?;
        ensure_finite("DMDW coupling matrix element", coupling.matrix_element[row])?;
    }
    ensure_positive("DMDW coupling normalization", coupling.normalization)
}

fn validate_dmdw_type2_a2f_inputs(
    atom_count: usize,
    unique_atom_groups: &[DmdwType2AtomGroup],
    displacement_option: i32,
    pole_count: usize,
) -> Result<(), DebyeError> {
    if pole_count == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW type 2 Lanczos pole count",
            value: pole_count as Real,
        });
    }
    if unique_atom_groups.is_empty() {
        return Err(DebyeError::EmptyDmdwType2UniqueAtomTable);
    }
    if !(0..=3).contains(&displacement_option) {
        return Err(DebyeError::InvalidDmdwType2DisplacementOption {
            option: displacement_option,
        });
    }
    for (group, unique_atom) in unique_atom_groups.iter().enumerate() {
        if unique_atom.center_atom_indices.is_empty() {
            return Err(DebyeError::EmptyDmdwType2CenterAtomGroup { group });
        }
        for &index in &unique_atom.center_atom_indices {
            if index >= atom_count {
                return Err(DebyeError::InvalidDmdwType2CenterAtomIndex {
                    group,
                    index,
                    atom_count,
                });
            }
        }
    }
    Ok(())
}

fn validate_dmdw_self_energy_inputs(
    temperature_kelvin: Real,
    energy_ev: Complex,
    pole_energy_ev: ArrayView1<'_, Real>,
    pole_weight: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    ensure_positive("DMDW self-energy temperature", temperature_kelvin)?;
    ensure_finite_complex("DMDW self-energy energy", energy_ev)?;
    if pole_energy_ev.len() != pole_weight.len() {
        return Err(DebyeError::InvalidDmdwSelfEnergyPoleTableShape {
            energies: pole_energy_ev.len(),
            weights: pole_weight.len(),
        });
    }
    if pole_energy_ev.is_empty() {
        return Err(DebyeError::EmptyDmdwPoleTable);
    }
    Ok(())
}

fn dmdw_einstein_summary(
    frequency_thz: Real,
    reduced_mass: Real,
) -> Result<DmdwEinsteinSummary, DebyeError> {
    ensure_positive("DMDW reduced mass", reduced_mass)?;
    ensure_positive("DMDW Einstein frequency", frequency_thz)?;
    let temperature_kelvin = frequency_thz * DMDW_THZ_TO_KELVIN;
    let effective_force_constant_n_per_m = reduced_mass
        * (2.0 * std::f64::consts::PI * frequency_thz).powi(2)
        * DMDW_AMU_THZ2_TO_NEWTON_PER_METER;
    ensure_finite_output("DMDW Einstein temperature", temperature_kelvin)?;
    ensure_finite_output(
        "DMDW Einstein effective force constant",
        effective_force_constant_n_per_m,
    )?;
    Ok(DmdwEinsteinSummary {
        frequency_thz,
        temperature_kelvin,
        effective_force_constant_n_per_m,
    })
}

fn dmdw_coth_argument_scale(temperature: Real) -> Result<Real, DebyeError> {
    ensure_positive("DMDW temperature", temperature)?;
    let beta = 1.0 / (DMDW_BOLTZMANN_EV_PER_K * temperature);
    let scale = 0.5 * DMDW_HBAR_EV_PS * beta;
    ensure_finite_output("DMDW coth argument scale", scale)?;
    Ok(scale)
}

fn validate_dmdw_seed(seed: ArrayView1<'_, Real>) -> Result<(), DebyeError> {
    if seed.is_empty() {
        return Err(DebyeError::EmptyDmdwSeed);
    }
    for value in seed.iter().copied() {
        ensure_finite("DMDW seed", value)?;
    }
    Ok(())
}

fn dmdw_apply_dynamical_matrix(
    matrix: ArrayView2<'_, Real>,
    vector: ArrayView1<'_, Real>,
) -> Array1<Real> {
    Array1::from_iter(matrix.columns().into_iter().map(|column| {
        column
            .iter()
            .zip(vector.iter())
            .map(|(&matrix_value, &vector_value)| matrix_value * vector_value)
            .sum()
    }))
}

fn lanczos_residual(
    mut applied: Array1<Real>,
    current: ArrayView1<'_, Real>,
    alpha: Real,
    previous: Option<(Real, ArrayView1<'_, Real>)>,
) -> Array1<Real> {
    for (value, &current_value) in applied.iter_mut().zip(current.iter()) {
        *value -= alpha * current_value;
    }
    if let Some((beta, previous_vector)) = previous {
        for (value, &previous_value) in applied.iter_mut().zip(previous_vector.iter()) {
            *value -= beta * previous_value;
        }
    }
    applied
}

fn normalize_lanczos_vector(
    mut vector: Array1<Real>,
    norm: Real,
    iteration: usize,
) -> Result<Array1<Real>, DebyeError> {
    ensure_finite_output("DMDW Lanczos beta", norm)?;
    if norm == 0.0 {
        return Err(DebyeError::DmdwLanczosBreakdown { iteration });
    }
    vector.mapv_inplace(|value| value / norm);
    Ok(vector)
}

fn dot_array_views(lhs: ArrayView1<'_, Real>, rhs: ArrayView1<'_, Real>) -> Real {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&lhs_value, &rhs_value)| lhs_value * rhs_value)
        .sum()
}

fn array_vector_norm(vector: ArrayView1<'_, Real>) -> Real {
    vector
        .iter()
        .map(|value| value * value)
        .sum::<Real>()
        .sqrt()
}

fn percent_or_zero(numerator: Real, denominator: Real) -> Real {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator * 100.0
    }
}

pub(super) fn dmdw_director_sum(
    atom_positions: ArrayView2<'_, Real>,
    atom: usize,
    previous: usize,
    next: usize,
) -> Result<[Real; 3], DebyeError> {
    let previous_vector = dmdw_director_cosine(atom_positions, atom, previous)?;
    let next_vector = dmdw_director_cosine(atom_positions, atom, next)?;
    Ok([
        previous_vector[0] + next_vector[0],
        previous_vector[1] + next_vector[1],
        previous_vector[2] + next_vector[2],
    ])
}

fn dmdw_director_cosine(
    atom_positions: ArrayView2<'_, Real>,
    atom: usize,
    neighbor: usize,
) -> Result<[Real; 3], DebyeError> {
    if atom == neighbor {
        return Ok([0.0; 3]);
    }
    let vector = [
        atom_positions[(atom, 0)] - atom_positions[(neighbor, 0)],
        atom_positions[(atom, 1)] - atom_positions[(neighbor, 1)],
        atom_positions[(atom, 2)] - atom_positions[(neighbor, 2)],
    ];
    let norm = vector_norm(vector);
    if norm == 0.0 {
        return Err(DebyeError::ZeroLengthDmdwAtomPair {
            first: atom,
            second: neighbor,
        });
    }
    Ok([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}
