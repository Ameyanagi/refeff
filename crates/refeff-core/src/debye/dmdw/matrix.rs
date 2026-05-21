use super::*;

/// Port FEFF DMDW run-type 4's IR Lanczos seed construction.
///
/// `atom_masses` is FEFF `dym_In%am`, and `dipole_derivatives` is the type 3
/// `.dym` payload arranged as `(atom, displacement_component, dipole_component)`.
/// FEFF's active branch uses the second dipole component (`jq = 2`) squared,
/// scaled by `sqrt(mass)`, in component-major seed order before normalizing.
pub fn dmdw_ir_dipole_seed_vector(
    atom_masses: ArrayView1<'_, Real>,
    dipole_derivatives: ArrayView3<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_ir_dipoles(atom_masses, dipole_derivatives)?;

    let atom_count = atom_masses.len();
    let mut seed = Array1::<Real>::zeros(atom_count * 3);
    for atom in 0..atom_count {
        let mass_scale = atom_masses[atom].sqrt();
        for displacement_component in 0..3 {
            let derivative = dipole_derivatives[(atom, displacement_component, 1)];
            seed[displacement_component * atom_count + atom] = mass_scale * derivative.powi(2);
        }
    }
    dmdw_normalize_seed_vector(seed.view())
}

/// Port FEFF DMDW `Make_DM`: mass-weight a block force-constant matrix.
///
/// `force_blocks` is FEFF `dym_In%dm_block(iAt,jAt,ip,jq)`, shaped as
/// `(atom_i, atom_j, component_i, component_j)`. The returned matrix uses
/// FEFF's component-major coordinate order: all x atom coordinates, then all
/// y, then all z. Values are scaled by FEFF's `auf2npm * npm2amups2` and
/// divided by `sqrt(m_i * m_j)`.
pub fn dmdw_mass_weighted_dynamical_matrix(
    force_blocks: ArrayView4<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<DmdwDynamicalMatrix, DebyeError> {
    validate_dmdw_force_blocks(force_blocks, atom_masses)?;
    let atom_count = atom_masses.len();
    let coordinate_count = atom_count * 3;
    let mut matrix = Array2::<Real>::zeros((coordinate_count, coordinate_count));

    for atom_i in 0..atom_count {
        for atom_j in 0..atom_count {
            let mass_scale =
                DMDW_DYNAMICAL_MATRIX_SCALE / (atom_masses[atom_i] * atom_masses[atom_j]).sqrt();
            for component_i in 0..3 {
                for component_j in 0..3 {
                    matrix[(
                        component_i * atom_count + atom_i,
                        component_j * atom_count + atom_j,
                    )] = mass_scale * force_blocks[(atom_i, atom_j, component_i, component_j)];
                }
            }
        }
    }

    let element_count = (coordinate_count * coordinate_count) as Real;
    let average_value = matrix.iter().map(|value| value.abs()).sum::<Real>() / element_count;
    let max_value = matrix.iter().map(|value| value.abs()).fold(0.0, Real::max);
    let average_asymmetry = matrix
        .indexed_iter()
        .map(|((row, column), value)| (value - matrix[(column, row)]).abs())
        .sum::<Real>()
        / element_count;
    let asymmetry_percent_average = percent_or_zero(average_asymmetry, average_value);
    let asymmetry_percent_max = percent_or_zero(average_asymmetry, max_value);

    for value in matrix.iter().copied() {
        ensure_finite_output("DMDW dynamical matrix", value)?;
    }
    ensure_finite_output("DMDW matrix average value", average_value)?;
    ensure_finite_output("DMDW matrix max value", max_value)?;
    ensure_finite_output("DMDW matrix average asymmetry", average_asymmetry)?;
    ensure_finite_output(
        "DMDW matrix average asymmetry percent",
        asymmetry_percent_average,
    )?;
    ensure_finite_output("DMDW matrix max asymmetry percent", asymmetry_percent_max)?;

    Ok(DmdwDynamicalMatrix {
        matrix,
        average_value,
        max_value,
        average_asymmetry,
        asymmetry_percent_average,
        asymmetry_percent_max,
    })
}
