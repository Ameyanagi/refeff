//! FEFF DMDW `.dym` dynamical-matrix codec.
//!
//! `DMDW/m_dmdw.f90` reads `.dym` files as a dynamical-matrix type flag,
//! atom metadata, coordinates, and one 3x3 force-constant block for every
//! atom pair. Type 1 files store Cartesian coordinates directly. Type 2 files
//! add unique-atom metadata for self-energy runs. Type 3 files add
//! Gaussian-style dipole derivatives for DMDW IR runs. Type 4 files store
//! reduced coordinates followed by three cell vectors.

use std::fmt::Write as _;
use std::path::Path;
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3, Array4};
use refeff_core::atomic::atomic_weight;

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

/// Coordinate section from a FEFF `.dym` file.
#[derive(Debug, Clone, PartialEq)]
pub enum DymCoordinates {
    /// Cartesian atom positions in atomic units.
    Cartesian(Array2<f64>),
    /// Reduced atom positions plus cell vectors in atomic units.
    Reduced {
        /// Fractional/reduced positions, one atom per row.
        reduced: Array2<f64>,
        /// Cell vectors, one vector per row.
        cell: Array2<f64>,
    },
}

/// One FEFF type-2 unique-atom group from a `.dym` file.
#[derive(Debug, Clone, PartialEq)]
pub struct DymUniqueAtom {
    /// FEFF `utype` value for this unique-atom group.
    pub atom_type: i32,
    /// Zero-based FEFF `centeratomindex` entries.
    pub center_atom_indices: Array1<usize>,
    /// The auxiliary scalar read between `centeratomindex` and coordinates.
    pub weights: Array1<f64>,
    /// FEFF `u_xyz` rows for each degenerate atom in the group.
    pub coordinates: Array2<f64>,
}

/// FEFF type-2 unique-atom metadata stored after the force constants.
#[derive(Debug, Clone, PartialEq)]
pub struct DymType2Metadata {
    /// FEFF `natom` value from the type-2 metadata block.
    pub cell_atom_count: usize,
    /// Unique-atom groups in file order.
    pub unique_atoms: Vec<DymUniqueAtom>,
}

impl DymCoordinates {
    /// Return Cartesian atom positions in atomic units.
    #[must_use]
    pub fn cartesian_positions(&self) -> Array2<f64> {
        match self {
            Self::Cartesian(positions) => positions.clone(),
            Self::Reduced { reduced, cell } => reduced.dot(cell),
        }
    }
}

/// Parsed FEFF `.dym` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct DymData {
    /// FEFF dynamical-matrix type flag.
    pub dym_type: i32,
    /// Atomic numbers, one per atom.
    pub atomic_numbers: Array1<i32>,
    /// Atomic masses, one per atom.
    pub atomic_masses: Array1<f64>,
    /// Coordinate block.
    pub coordinates: DymCoordinates,
    /// Force-constant blocks indexed as `[iatom, jatom, row, column]`.
    pub force_constants: Array4<f64>,
    /// Optional unique-atom metadata for type 2 DMDW self-energy files.
    pub type2_metadata: Option<DymType2Metadata>,
    /// Optional dipole derivatives for type 3 DMDW IR files.
    ///
    /// The shape is `(atom, displacement_component, dipole_component)`, matching
    /// FEFF's nested `iAt`, `ip`, `jq` read order while keeping atom rows first.
    pub dipole_derivatives: Option<Array3<f64>>,
}

impl DymData {
    /// Number of atoms in the `.dym` file.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atomic_numbers.len()
    }

    /// Build FEFF's mass-weighted dynamical matrix in coordinate-major layout.
    ///
    /// FEFF stores coordinate blocks as `dm_block(iAt,jAt,ip,jq)` and then
    /// maps them to `dm((iAt-1)+nAt*(ip-1),(jAt-1)+nAt*(jq-1))`, divided by
    /// `sqrt(am(iAt)*am(jAt))`.
    pub fn mass_weighted_dynamical_matrix(&self) -> Result<Array2<f64>> {
        validate_dym(self)?;

        let atom_count = self.atom_count();
        let mut matrix = Array2::zeros((3 * atom_count, 3 * atom_count));
        for i_atom in 0..atom_count {
            for j_atom in 0..atom_count {
                let scale = (self.atomic_masses[i_atom] * self.atomic_masses[j_atom]).sqrt();
                for row in 0..3 {
                    for column in 0..3 {
                        matrix[[i_atom + atom_count * row, j_atom + atom_count * column]] =
                            self.force_constants[[i_atom, j_atom, row, column]] / scale;
                    }
                }
            }
        }
        Ok(matrix)
    }
}

/// Render a FEFF-compatible `.dym` text file.
pub fn dym_string(data: &DymData) -> Result<String> {
    validate_dym(data)?;

    let mut out = String::new();
    let extended_coordinates = coordinates_need_extended_fields(&data.coordinates);
    writeln!(out, "{:5}", data.dym_type)?;
    writeln!(out, "{:5}", data.atom_count())?;
    for atomic_number in &data.atomic_numbers {
        writeln!(out, "{atomic_number:5}")?;
    }
    for atomic_mass in &data.atomic_masses {
        writeln!(out, "{atomic_mass:12.6}")?;
    }

    match &data.coordinates {
        DymCoordinates::Cartesian(positions) => {
            for row in positions.rows() {
                write_coordinate_row(&mut out, row[0], row[1], row[2], extended_coordinates)?;
            }
        }
        DymCoordinates::Reduced { reduced, .. } => {
            for row in reduced.rows() {
                write_coordinate_row(&mut out, row[0], row[1], row[2], extended_coordinates)?;
            }
        }
    }

    let atom_count = data.atom_count();
    for i_atom in 0..atom_count {
        for j_atom in 0..atom_count {
            writeln!(out, "{:5}{:5}", i_atom + 1, j_atom + 1)?;
            for row in 0..3 {
                for column in 0..3 {
                    write_fortran_exp(
                        &mut out,
                        data.force_constants[[i_atom, j_atom, row, column]],
                        14,
                        6,
                    )?;
                }
                out.push('\n');
            }
        }
    }

    if let Some(metadata) = &data.type2_metadata {
        writeln!(
            out,
            "{:5}{:5}",
            metadata.unique_atoms.len(),
            metadata.cell_atom_count
        )?;
        for unique_atom in &metadata.unique_atoms {
            writeln!(
                out,
                "{:5}{:5}",
                unique_atom.atom_type,
                unique_atom.center_atom_indices.len()
            )?;
            for row in 0..unique_atom.center_atom_indices.len() {
                write!(out, "{:5}", unique_atom.center_atom_indices[row] + 1)?;
                write_fortran_exp(&mut out, unique_atom.weights[row], 14, 6)?;
                for column in 0..3 {
                    write_fortran_exp(&mut out, unique_atom.coordinates[[row, column]], 14, 6)?;
                }
                out.push('\n');
            }
        }
    }

    if let DymCoordinates::Reduced { cell, .. } = &data.coordinates {
        writeln!(out)?;
        for row in cell.rows() {
            write_coordinate_row(&mut out, row[0], row[1], row[2], extended_coordinates)?;
        }
    }

    if let Some(dipole_derivatives) = &data.dipole_derivatives {
        writeln!(out)?;
        for atom in 0..atom_count {
            for displacement_component in 0..3 {
                for dipole_component in 0..3 {
                    write_fortran_exp(
                        &mut out,
                        dipole_derivatives[[atom, displacement_component, dipole_component]],
                        14,
                        6,
                    )?;
                }
                out.push('\n');
            }
        }
    }

    Ok(out)
}

fn write_coordinate_row(
    out: &mut String,
    x: f64,
    y: f64,
    z: f64,
    extended: bool,
) -> std::fmt::Result {
    if extended {
        writeln!(out, "{x:18.10}{y:16.10}{z:16.10}")
    } else {
        writeln!(out, "{x:14.8}{y:14.8}{z:14.8}")
    }
}

fn coordinates_need_extended_fields(coordinates: &DymCoordinates) -> bool {
    match coordinates {
        DymCoordinates::Cartesian(positions) => matrix_needs_extended_fields(positions),
        DymCoordinates::Reduced { reduced, cell } => {
            matrix_needs_extended_fields(reduced) || matrix_needs_extended_fields(cell)
        }
    }
}

fn matrix_needs_extended_fields(values: &Array2<f64>) -> bool {
    values
        .iter()
        .any(|value| coordinate_needs_extended_field(*value))
}

fn coordinate_needs_extended_field(value: f64) -> bool {
    let scaled = value * 1.0e8;
    (scaled - scaled.round()).abs() > 1.0e-4
}

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

/// Write FEFF `.dym` text to a file.
pub fn write_dym(path: impl AsRef<Path>, data: &DymData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dym_string(data)?).map_err(|source| IoError::io(path, source))
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

fn validate_dym(data: &DymData) -> Result<()> {
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

fn fix_atomic_numbers_and_masses(
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

fn shape_error(field: &'static str, actual: Vec<usize>, expected: Vec<usize>) -> IoError {
    IoError::DymShape {
        field,
        actual,
        expected,
    }
}

fn invalid_dym(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidDym {
        field,
        message: message.into(),
    }
}

#[derive(Debug, Clone, Copy)]
struct DymToken<'a> {
    line: usize,
    text: &'a str,
}

#[derive(Debug)]
struct DymTokenCursor<'a> {
    tokens: Vec<DymToken<'a>>,
    index: usize,
}

impl<'a> DymTokenCursor<'a> {
    fn new(text: &'a str) -> Self {
        let tokens = text
            .lines()
            .enumerate()
            .flat_map(|(line, text)| {
                text.split_whitespace().map(move |token| DymToken {
                    line: line + 1,
                    text: token,
                })
            })
            .collect();
        Self { tokens, index: 0 }
    }

    fn parse<T>(&mut self, field: &'static str) -> Result<T>
    where
        T: FromStr,
    {
        let token = self.next_token(field)?;
        token.text.parse::<T>().map_err(|_| IoError::DymParse {
            field,
            line: token.line,
            token: token.text.to_string(),
        })
    }

    fn next_token(&mut self, field: &'static str) -> Result<DymToken<'a>> {
        let Some(token) = self.tokens.get(self.index).copied() else {
            return Err(IoError::DymMissing { field });
        };
        self.index += 1;
        Ok(token)
    }

    fn remaining_count(&self) -> usize {
        self.tokens.len().saturating_sub(self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_type1_dym_and_builds_mass_weighted_matrix() -> Result<()> {
        let parsed = parse_dym(TYPE1_DYM)?;
        assert_eq!(parsed.dym_type, 1);
        assert_eq!(parsed.atom_count(), 2);
        assert_eq!(parsed.atomic_numbers.to_vec(), vec![29, 8]);
        let positions = parsed.coordinates.cartesian_positions();
        assert_eq!(positions[[1, 0]], 1.0);
        assert_eq!(parsed.force_constants[[0, 1, 0, 0]], -1.0);

        let matrix = parsed.mass_weighted_dynamical_matrix()?;
        assert_eq!(matrix.shape(), &[6, 6]);
        assert_eq!(matrix[[0, 0]], 2.0 / 64.0);
        assert!(matrix[[0, 1]] < 0.0);
        Ok(())
    }

    #[test]
    fn roundtrips_type1_dym_text() -> Result<()> {
        let parsed = parse_dym(TYPE1_DYM)?;
        let rendered = dym_string(&parsed)?;
        assert!(rendered.contains("    1.00000000    0.00000000    0.00000000"));
        assert!(rendered.contains("  2.000000E+00  0.000000E+00  0.000000E+00"));
        let reparsed = parse_dym(&rendered)?;
        assert_eq!(reparsed.dym_type, parsed.dym_type);
        assert_eq!(reparsed.atomic_numbers, parsed.atomic_numbers);
        assert_eq!(reparsed.atomic_masses, parsed.atomic_masses);
        assert_eq!(reparsed.force_constants, parsed.force_constants);
        Ok(())
    }

    #[test]
    fn renders_extended_coordinate_fields_when_needed() -> Result<()> {
        let mut parsed = parse_dym(TYPE1_DYM)?;
        let DymCoordinates::Cartesian(positions) = &mut parsed.coordinates else {
            return Err(invalid_dym("coordinates", "expected Cartesian coordinates"));
        };
        positions[[1, 0]] = 1.000000001;

        let rendered = dym_string(&parsed)?;
        assert!(rendered.contains("      1.0000000010    0.0000000000    0.0000000000"));
        assert_eq!(parse_dym(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_type4_reduced_coordinates_and_cell() -> Result<()> {
        let parsed = parse_dym(TYPE4_DYM)?;
        assert_eq!(parsed.dym_type, 4);
        let DymCoordinates::Reduced { reduced, cell } = &parsed.coordinates else {
            return Err(invalid_dym("coordinates", "expected reduced coordinates"));
        };
        assert_eq!(reduced[[1, 0]], 0.5);
        assert_eq!(cell[[0, 0]], 4.0);
        let cartesian = parsed.coordinates.cartesian_positions();
        assert_eq!(cartesian[[1, 0]], 2.0);
        assert_eq!(cartesian[[1, 1]], 0.0);
        Ok(())
    }

    #[test]
    fn parses_type2_unique_atom_metadata() -> Result<()> {
        let (_, type1_body) = TYPE1_DYM
            .split_once('\n')
            .ok_or_else(|| invalid_dym("type", "test fixture missing type header"))?;
        let type2_text = String::from("    2\n")
            + type1_body
            + "\
    2    2
   29    1
    1  1.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
    8    1
    2  2.000000E+00  1.000000E+00  0.000000E+00  0.000000E+00
";
        let parsed = parse_dym(&type2_text)?;
        assert_eq!(parsed.dym_type, 2);
        let metadata = parsed
            .type2_metadata
            .as_ref()
            .ok_or_else(|| invalid_dym("type 2 metadata", "missing test metadata"))?;
        assert_eq!(metadata.cell_atom_count, 2);
        assert_eq!(metadata.unique_atoms.len(), 2);
        assert_eq!(metadata.unique_atoms[0].atom_type, 29);
        assert_eq!(
            metadata.unique_atoms[0].center_atom_indices.to_vec(),
            vec![0]
        );
        assert_eq!(metadata.unique_atoms[0].weights.to_vec(), vec![1.0]);
        assert_eq!(metadata.unique_atoms[1].atom_type, 8);
        assert_eq!(
            metadata.unique_atoms[1].center_atom_indices.to_vec(),
            vec![1]
        );
        assert_eq!(metadata.unique_atoms[1].coordinates[[0, 0]], 1.0);

        let rendered = dym_string(&parsed)?;
        let reparsed = parse_dym(&rendered)?;
        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn parses_type3_dipole_derivatives_for_ir_runs() -> Result<()> {
        let parsed = parse_dym(TYPE3_DYM)?;
        assert_eq!(parsed.dym_type, 3);
        let dipoles = parsed
            .dipole_derivatives
            .as_ref()
            .ok_or_else(|| invalid_dym("dipole derivatives", "missing test dipoles"))?;
        assert_eq!(dipoles.shape(), &[2, 3, 3]);
        assert_eq!(dipoles[[0, 1, 1]], 0.5);
        assert_eq!(dipoles[[1, 2, 2]], 1.8);

        let rendered = dym_string(&parsed)?;
        let reparsed = parse_dym(&rendered)?;
        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn fills_missing_atomic_metadata_like_feff() -> Result<()> {
        let parsed = parse_dym(TYPE1_MISSING_ATOMIC_METADATA_DYM)?;
        assert_eq!(parsed.atomic_numbers.to_vec(), vec![29, 8]);
        assert!((parsed.atomic_masses[1] - 15.999).abs() < 1.0e-6);
        Ok(())
    }

    #[test]
    fn rejects_bad_dym_inputs() -> Result<()> {
        assert!(matches!(
            parse_dym("1\n"),
            Err(IoError::DymMissing {
                field: "atom count"
            })
        ));
        assert!(matches!(
            parse_dym(TYPE1_BAD_PAIR_DYM),
            Err(IoError::InvalidDym {
                field: "force-constant i atom",
                ..
            })
        ));
        let mut bad_mass = parse_dym(TYPE1_DYM)?;
        bad_mass.atomic_masses[0] = 0.0;
        assert!(matches!(
            dym_string(&bad_mass),
            Err(IoError::InvalidDym {
                field: "atomic mass",
                ..
            })
        ));
        Ok(())
    }

    const TYPE1_DYM: &str = "\
    1
    2
   29
    8
   64.000000
   16.000000
    0.00000000    0.00000000    0.00000000
    1.00000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
";

    const TYPE4_DYM: &str = "\
    4
    2
   29
    8
   64.000000
   16.000000
    0.00000000    0.00000000    0.00000000
    0.50000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00

    4.00000000    0.00000000    0.00000000
    0.00000000    5.00000000    0.00000000
    0.00000000    0.00000000    6.00000000
";

    const TYPE3_DYM: &str = "\
    3
    2
   29
    8
   64.000000
   16.000000
    0.00000000    0.00000000    0.00000000
    1.00000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00

  1.000000E-01  2.000000E-01  3.000000E-01
  4.000000E-01  5.000000E-01  6.000000E-01
  7.000000E-01  8.000000E-01  9.000000E-01
  1.000000E+00  1.100000E+00  1.200000E+00
  1.300000E+00  1.400000E+00  1.500000E+00
  1.600000E+00  1.700000E+00  1.800000E+00
";

    const TYPE1_BAD_PAIR_DYM: &str = "\
    1
    1
   29
   64.000000
    0.00000000    0.00000000    0.00000000
    2    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
";

    const TYPE1_MISSING_ATOMIC_METADATA_DYM: &str = "\
    1
    2
    0
    8
   63.546000
    0.000000
    0.00000000    0.00000000    0.00000000
    1.00000000    0.00000000    0.00000000
    1    1
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
    1    2
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    1
 -1.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00 -1.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00 -1.000000E+00
    2    2
  2.000000E+00  0.000000E+00  0.000000E+00
  0.000000E+00  2.000000E+00  0.000000E+00
  0.000000E+00  0.000000E+00  2.000000E+00
";
}
