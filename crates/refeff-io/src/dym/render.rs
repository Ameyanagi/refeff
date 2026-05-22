use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array2;

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

use super::validate::validate_dym;
use super::{DymCoordinates, DymData};

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

/// Write FEFF `.dym` text to a file.
pub fn write_dym(path: impl AsRef<Path>, data: &DymData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dym_string(data)?).map_err(|source| IoError::io(path, source))
}
