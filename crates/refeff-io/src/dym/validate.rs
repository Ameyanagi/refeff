use ndarray::{Array1, Array2};
use refeff_core::atomic::atomic_weight;

use crate::error::{IoError, Result};

use super::{DymCoordinates, DymData, DymType2Metadata};

pub(super) fn validate_dym(data: &DymData) -> Result<()> {
    if !matches!(data.dym_type, 1..=4) {
        return Err(invalid_dym(
            "type",
            format!(
                "type {} is not supported; expected 1, 2, 3, or 4",
                data.dym_type
            ),
        ));
    }

    let atom_count = data.atom_count();
    if atom_count == 0 {
        return Err(invalid_dym("atom count", "value must be positive"));
    }
    if data.atomic_masses.len() != atom_count {
        return Err(shape_error(
            "atomic masses",
            vec![data.atomic_masses.len()],
            vec![atom_count],
        ));
    }
    for atomic_number in &data.atomic_numbers {
        if *atomic_number <= 0 {
            return Err(invalid_dym("atomic number", "all values must be positive"));
        }
    }
    for atomic_mass in &data.atomic_masses {
        if !atomic_mass.is_finite() || *atomic_mass <= 0.0 {
            return Err(invalid_dym(
                "atomic mass",
                "all values must be finite and positive",
            ));
        }
    }

    match &data.coordinates {
        DymCoordinates::Cartesian(positions) => {
            if data.dym_type == 4 {
                return Err(invalid_dym(
                    "coordinates",
                    "type 4 .dym data requires reduced coordinates",
                ));
            }
            validate_matrix_shape("coordinates", positions, atom_count, 3)?;
            validate_finite_array2("coordinates", positions)?;
        }
        DymCoordinates::Reduced { reduced, cell } => {
            if data.dym_type != 4 {
                return Err(invalid_dym(
                    "coordinates",
                    "reduced coordinates are only valid for type 4 .dym data",
                ));
            }
            validate_matrix_shape("reduced coordinates", reduced, atom_count, 3)?;
            validate_matrix_shape("cell vectors", cell, 3, 3)?;
            validate_finite_array2("reduced coordinates", reduced)?;
            validate_finite_array2("cell vectors", cell)?;
        }
    }

    if data.force_constants.shape() != [atom_count, atom_count, 3, 3] {
        return Err(shape_error(
            "force constants",
            data.force_constants.shape().to_vec(),
            vec![atom_count, atom_count, 3, 3],
        ));
    }
    for value in &data.force_constants {
        if !value.is_finite() {
            return Err(invalid_dym("force constants", "all values must be finite"));
        }
    }

    match (&data.type2_metadata, data.dym_type) {
        (Some(metadata), 2) => validate_type2_metadata(metadata, atom_count)?,
        (Some(_), _) => {
            return Err(invalid_dym(
                "type 2 metadata",
                "unique-atom metadata is only valid for type 2 .dym data",
            ));
        }
        (None, 2) => {
            return Err(invalid_dym(
                "type 2 metadata",
                "type 2 .dym data requires unique-atom metadata",
            ));
        }
        (None, _) => {}
    }

    match (&data.dipole_derivatives, data.dym_type) {
        (Some(dipole_derivatives), 3) => {
            if dipole_derivatives.shape() != [atom_count, 3, 3] {
                return Err(IoError::DymShape {
                    field: "dipole derivatives",
                    actual: dipole_derivatives.shape().to_vec(),
                    expected: vec![atom_count, 3, 3],
                });
            }
            for value in dipole_derivatives {
                if !value.is_finite() {
                    return Err(invalid_dym(
                        "dipole derivatives",
                        "all values must be finite",
                    ));
                }
            }
        }
        (Some(_), _) => {
            return Err(invalid_dym(
                "dipole derivatives",
                "dipole derivatives are only valid for type 3 .dym data",
            ));
        }
        (None, 3) => {
            return Err(invalid_dym(
                "dipole derivatives",
                "type 3 .dym data requires dipole derivatives",
            ));
        }
        (None, _) => {}
    }

    Ok(())
}

fn validate_type2_metadata(metadata: &DymType2Metadata, atom_count: usize) -> Result<()> {
    if metadata.cell_atom_count == 0 {
        return Err(invalid_dym(
            "type 2 cell atom count",
            "value must be positive",
        ));
    }
    if metadata.unique_atoms.is_empty() {
        return Err(invalid_dym(
            "type 2 unique atoms",
            "at least one unique-atom group is required",
        ));
    }
    for unique_atom in &metadata.unique_atoms {
        if unique_atom.atom_type <= 0 {
            return Err(invalid_dym("type 2 atom type", "value must be positive"));
        }
        let degeneracy = unique_atom.center_atom_indices.len();
        if degeneracy == 0 {
            return Err(invalid_dym("type 2 degeneracy", "value must be positive"));
        }
        if unique_atom.weights.len() != degeneracy {
            return Err(shape_error(
                "type 2 weights",
                vec![unique_atom.weights.len()],
                vec![degeneracy],
            ));
        }
        if unique_atom.coordinates.shape() != [degeneracy, 3] {
            return Err(shape_error(
                "type 2 coordinates",
                unique_atom.coordinates.shape().to_vec(),
                vec![degeneracy, 3],
            ));
        }
        for &index in &unique_atom.center_atom_indices {
            if index >= atom_count {
                return Err(invalid_dym(
                    "type 2 center atom index",
                    format!("index {} is outside 1..={atom_count}", index + 1),
                ));
            }
        }
        for &weight in &unique_atom.weights {
            if !weight.is_finite() {
                return Err(invalid_dym("type 2 weight", "all values must be finite"));
            }
        }
        validate_finite_array2("type 2 coordinates", &unique_atom.coordinates)?;
    }
    Ok(())
}

pub(super) fn fix_atomic_numbers_and_masses(
    atomic_numbers: &mut Array1<i32>,
    atomic_masses: &mut Array1<f64>,
) -> Result<()> {
    for (atomic_number, atomic_mass) in atomic_numbers.iter_mut().zip(atomic_masses.iter_mut()) {
        if *atomic_number <= 0 && *atomic_mass < 0.2 {
            return Err(invalid_dym(
                "atomic metadata",
                "atomic number and atomic mass cannot both be missing",
            ));
        }
        if *atomic_number <= 0 {
            *atomic_number = infer_atomic_number(*atomic_mass)?;
        }
        if *atomic_mass < 0.2 {
            let atomic_number = usize::try_from(*atomic_number).map_err(|_| {
                invalid_dym("atomic number", "value must fit a positive atomic number")
            })?;
            *atomic_mass = atomic_weight(atomic_number)
                .map_err(|error| invalid_dym("atomic mass", error.to_string()))?;
        }
    }
    Ok(())
}

fn infer_atomic_number(atomic_mass: f64) -> Result<i32> {
    for atomic_number in 1..=139_usize {
        let weight = atomic_weight(atomic_number)
            .map_err(|error| invalid_dym("atomic number", error.to_string()))?;
        if atomic_mass > weight - 0.2 && atomic_mass < weight + 0.2 {
            return i32::try_from(atomic_number)
                .map_err(|_| invalid_dym("atomic number", "value must fit i32"));
        }
    }
    Err(invalid_dym(
        "atomic number",
        format!("could not infer atomic number from mass {atomic_mass}"),
    ))
}

fn validate_matrix_shape(
    field: &'static str,
    matrix: &Array2<f64>,
    rows: usize,
    columns: usize,
) -> Result<()> {
    if matrix.shape() == [rows, columns] {
        Ok(())
    } else {
        Err(shape_error(
            field,
            matrix.shape().to_vec(),
            vec![rows, columns],
        ))
    }
}

fn validate_finite_array2(field: &'static str, matrix: &Array2<f64>) -> Result<()> {
    for value in matrix {
        if !value.is_finite() {
            return Err(invalid_dym(field, "all values must be finite"));
        }
    }
    Ok(())
}

pub(super) fn shape_error(
    field: &'static str,
    actual: Vec<usize>,
    expected: Vec<usize>,
) -> IoError {
    IoError::DymShape {
        field,
        actual,
        expected,
    }
}

pub(super) fn invalid_dym(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidDym {
        field,
        message: message.into(),
    }
}
