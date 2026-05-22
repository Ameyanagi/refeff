use std::path::Path;
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3, Array4};

use crate::error::{IoError, Result};

use super::common::DymTokenCursor;
use super::validate::{fix_atomic_numbers_and_masses, invalid_dym, validate_dym};
use super::{DymCoordinates, DymData, DymType2Metadata, DymUniqueAtom};

/// Parse FEFF `.dym` text.
pub fn parse_dym(text: &str) -> Result<DymData> {
    let mut cursor = DymTokenCursor::new(text);
    let dym_type = cursor.parse::<i32>("type")?;
    if !matches!(dym_type, 1..=4) {
        return Err(invalid_dym(
            "type",
            format!("type {dym_type} is not supported; expected 1, 2, 3, or 4"),
        ));
    }

    let atom_count = cursor.parse::<usize>("atom count")?;
    if atom_count == 0 {
        return Err(invalid_dym("atom count", "value must be positive"));
    }

    let mut atomic_numbers = parse_array1(&mut cursor, atom_count, "atomic number")?;
    let mut atomic_masses = parse_array1(&mut cursor, atom_count, "atomic mass")?;
    fix_atomic_numbers_and_masses(&mut atomic_numbers, &mut atomic_masses)?;
    let position_rows = parse_array2(&mut cursor, atom_count, 3, "coordinate")?;
    let force_constants = parse_force_constants(&mut cursor, atom_count)?;
    let type2_metadata = if dym_type == 2 {
        Some(parse_type2_metadata(&mut cursor, atom_count)?)
    } else {
        None
    };
    let dipole_derivatives = if dym_type == 3 {
        Some(parse_array3(
            &mut cursor,
            atom_count,
            3,
            3,
            "dipole derivative",
        )?)
    } else {
        None
    };
    let coordinates = if dym_type == 4 {
        let cell = parse_array2(&mut cursor, 3, 3, "cell vector")?;
        DymCoordinates::Reduced {
            reduced: position_rows,
            cell,
        }
    } else {
        DymCoordinates::Cartesian(position_rows)
    };

    if cursor.remaining_count() != 0 {
        return Err(IoError::DymTrailingTokens {
            count: cursor.remaining_count(),
        });
    }

    let data = DymData {
        dym_type,
        atomic_numbers,
        atomic_masses,
        coordinates,
        force_constants,
        type2_metadata,
        dipole_derivatives,
    };
    validate_dym(&data)?;
    Ok(data)
}

/// Read FEFF `.dym` text from a file.
pub fn read_dym(path: impl AsRef<Path>) -> Result<DymData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dym(&text)
}

fn parse_array1<T>(
    cursor: &mut DymTokenCursor<'_>,
    len: usize,
    field: &'static str,
) -> Result<Array1<T>>
where
    T: FromStr,
{
    let values = (0..len)
        .map(|_| cursor.parse(field))
        .collect::<Result<Vec<T>>>()?;
    Ok(Array1::from_vec(values))
}

fn parse_array2(
    cursor: &mut DymTokenCursor<'_>,
    rows: usize,
    columns: usize,
    field: &'static str,
) -> Result<Array2<f64>> {
    let values = (0..rows * columns)
        .map(|_| cursor.parse(field))
        .collect::<Result<Vec<f64>>>()?;
    Array2::from_shape_vec((rows, columns), values).map_err(|_| IoError::DymShape {
        field,
        actual: vec![rows * columns],
        expected: vec![rows, columns],
    })
}

fn parse_array3(
    cursor: &mut DymTokenCursor<'_>,
    rows: usize,
    columns: usize,
    depth: usize,
    field: &'static str,
) -> Result<Array3<f64>> {
    let values = (0..rows * columns * depth)
        .map(|_| cursor.parse(field))
        .collect::<Result<Vec<f64>>>()?;
    Array3::from_shape_vec((rows, columns, depth), values).map_err(|_| IoError::DymShape {
        field,
        actual: vec![rows * columns * depth],
        expected: vec![rows, columns, depth],
    })
}

fn parse_force_constants(
    cursor: &mut DymTokenCursor<'_>,
    atom_count: usize,
) -> Result<Array4<f64>> {
    let mut force_constants = Array4::zeros((atom_count, atom_count, 3, 3));
    let mut seen = vec![false; atom_count * atom_count];

    for _ in 0..atom_count * atom_count {
        let i_atom = parse_atom_index(cursor, atom_count, "force-constant i atom")?;
        let j_atom = parse_atom_index(cursor, atom_count, "force-constant j atom")?;
        let seen_index = i_atom * atom_count + j_atom;
        if seen[seen_index] {
            return Err(invalid_dym(
                "force constants",
                format!(
                    "duplicate block for atom pair ({}, {})",
                    i_atom + 1,
                    j_atom + 1
                ),
            ));
        }
        seen[seen_index] = true;

        for row in 0..3 {
            for column in 0..3 {
                force_constants[[i_atom, j_atom, row, column]] = cursor.parse("force constant")?;
            }
        }
    }

    Ok(force_constants)
}

fn parse_type2_metadata(
    cursor: &mut DymTokenCursor<'_>,
    atom_count: usize,
) -> Result<DymType2Metadata> {
    let unique_atom_count = cursor.parse::<usize>("type 2 unique atom count")?;
    let cell_atom_count = cursor.parse::<usize>("type 2 cell atom count")?;
    if unique_atom_count == 0 {
        return Err(invalid_dym(
            "type 2 unique atom count",
            "value must be positive",
        ));
    }
    if cell_atom_count == 0 {
        return Err(invalid_dym(
            "type 2 cell atom count",
            "value must be positive",
        ));
    }

    let mut unique_atoms = Vec::with_capacity(unique_atom_count);
    for _ in 0..unique_atom_count {
        let atom_type = cursor.parse::<i32>("type 2 atom type")?;
        let degeneracy = cursor.parse::<usize>("type 2 degeneracy")?;
        if degeneracy == 0 {
            return Err(invalid_dym("type 2 degeneracy", "value must be positive"));
        }
        let mut center_atom_indices = Vec::with_capacity(degeneracy);
        let mut weights = Vec::with_capacity(degeneracy);
        let mut coordinates = Vec::with_capacity(degeneracy * 3);
        for _ in 0..degeneracy {
            center_atom_indices.push(parse_atom_index(
                cursor,
                atom_count,
                "type 2 center atom index",
            )?);
            weights.push(cursor.parse::<f64>("type 2 weight")?);
            for _ in 0..3 {
                coordinates.push(cursor.parse::<f64>("type 2 coordinate")?);
            }
        }
        let coordinates = Array2::from_shape_vec((degeneracy, 3), coordinates).map_err(|_| {
            IoError::DymShape {
                field: "type 2 coordinates",
                actual: vec![degeneracy * 3],
                expected: vec![degeneracy, 3],
            }
        })?;
        unique_atoms.push(DymUniqueAtom {
            atom_type,
            center_atom_indices: Array1::from_vec(center_atom_indices),
            weights: Array1::from_vec(weights),
            coordinates,
        });
    }

    Ok(DymType2Metadata {
        cell_atom_count,
        unique_atoms,
    })
}

fn parse_atom_index(
    cursor: &mut DymTokenCursor<'_>,
    atom_count: usize,
    field: &'static str,
) -> Result<usize> {
    let token = cursor.parse::<usize>(field)?;
    if token == 0 || token > atom_count {
        return Err(invalid_dym(
            field,
            format!("index {token} is outside 1..={atom_count}"),
        ));
    }
    Ok(token - 1)
}
