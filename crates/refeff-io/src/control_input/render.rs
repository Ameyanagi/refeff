use std::fmt::Write as _;

use crate::{IoError, Result};

use super::common::{
    DENSITY_FILENAME_WIDTH, control_bool, fixed_left, fortran_fixed_string, write_f13_5_line,
};
use super::types::{
    BandInput, DensityGrid, DensityInput, FullSpectrumInput, OpconsInput, ReciprocalInput,
};

/// Render FEFF-compatible `band.inp` text.
pub fn band_input_string(input: &BandInput) -> Result<String> {
    validate_band_input(input)?;

    let mut out = String::new();
    writeln!(out, "mband : calculate bands if = 1")?;
    writeln!(out, "{:4}", input.mband)?;
    writeln!(out, "emin, emax, estep : energy mesh")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.energy_mesh.emin, input.energy_mesh.emax, input.energy_mesh.estep
    )?;
    writeln!(out, "nkp : # points in k-path")?;
    writeln!(out, "{:4}", input.nkp)?;
    writeln!(out, "ikpath : type of k-path")?;
    writeln!(out, "{:4}", input.ikpath)?;
    writeln!(out, "freeprop :  empty lattice if = T")?;
    writeln!(out, " {}", control_bool(input.freeprop))?;
    Ok(out)
}

/// Render FEFF-compatible `density.inp` text.
pub fn density_input_string(input: &DensityInput) -> Result<String> {
    let mut out = String::new();
    for grid in &input.grids {
        validate_density_grid(grid)?;
        let filename = density_output_filename(&grid.filename)?;
        writeln!(
            out,
            "{} {} {:.15} {:.15} {:.15}{}",
            grid.kind.as_command(),
            filename,
            grid.origin[0],
            grid.origin[1],
            grid.origin[2],
            if grid.core { " core" } else { "" }
        )?;
        for axis in &grid.axes {
            writeln!(
                out,
                "{:.15} {:.15} {:.15} {}",
                axis.vector[0], axis.vector[1], axis.vector[2], axis.points
            )?;
        }
    }
    Ok(out)
}

/// Render FEFF-compatible `fullspectrum.inp` text.
pub fn fullspectrum_input_string(input: &FullSpectrumInput) -> Result<String> {
    let mut out = String::new();
    writeln!(out, " mFullSpectrum")?;
    writeln!(out, "{:12}", input.m_full_spectrum)?;
    Ok(out)
}

/// Render FEFF-compatible `opcons.inp` text.
pub fn opcons_input_string(input: &OpconsInput) -> Result<String> {
    validate_opcons_input(input)?;

    let mut out = String::new();
    writeln!(out, "run_opcons")?;
    writeln!(out, " {}", control_bool(input.run_opcons))?;
    writeln!(out, "print_eps")?;
    writeln!(out, " {}", control_bool(input.print_eps))?;
    writeln!(out, "NumDens(0:nphx)")?;
    write!(out, "  ")?;
    for (index, density) in input.number_densities.iter().copied().enumerate() {
        if index > 0 {
            write!(out, "       ")?;
        }
        write!(out, "{density:.16}")?;
    }
    writeln!(out, "     ")?;
    Ok(out)
}

/// Render FEFF-compatible `reciprocal.inp` text.
pub fn reciprocal_input_string(input: &ReciprocalInput) -> Result<String> {
    validate_reciprocal_input(input)?;

    let mut out = String::new();
    writeln!(out, "ispace")?;
    writeln!(out, "{:4}", input.ispace)?;
    if let Some(cell) = &input.cell {
        writeln!(out, "lattice vectors  (in A, in Carthesian coordinates)")?;
        for row in cell.lattice_vectors {
            write_f13_5_line(&mut out, row)?;
        }
        writeln!(out, "Volume scaling factor (A^3); eimag; core hole")?;
        write_f13_5_line(
            &mut out,
            [
                cell.volume_scale,
                cell.imaginary_energy,
                cell.core_hole_strength,
            ],
        )?;
        writeln!(out, "lattice type  (P,I,F,R,B,CXY,CYZ,CXZ)")?;
        writeln!(
            out,
            "{}{}{:>3}",
            fixed_left(&cell.lattice_name, 7),
            fixed_left(&cell.space_group_hm, 13),
            cell.space_group
        )?;
        writeln!(out, "#atoms in unit cell ; position absorber ; corehole?")?;
        writeln!(
            out,
            "{:4}{:4}{:4}",
            cell.atom_count, cell.absorber, cell.core_hole
        )?;
        writeln!(out, "# k-points total/x/y/z ; ktype; use symmetry?")?;
        writeln!(
            out,
            "{:12}{:12}{:12}{:12}{:12}{:12}",
            cell.k_mesh.total,
            cell.k_mesh.x,
            cell.k_mesh.y,
            cell.k_mesh.z,
            cell.k_mesh.kind,
            i32::from(cell.k_mesh.use_symmetry)
        )?;
        writeln!(out, "ppos")?;
        for position in &cell.positions {
            write_f13_5_line(&mut out, *position)?;
        }
        writeln!(out, "ppot")?;
        for potential in &cell.potentials {
            write!(out, "{potential:12}")?;
        }
        writeln!(out)?;
        writeln!(out, "label")?;
        writeln!(out, "{}", reciprocal_label_line(&cell.labels))?;
        writeln!(out, "streta,strgmax,strrmax")?;
        write_f13_5_line(&mut out, cell.stretch)?;
    }
    Ok(out)
}

fn density_output_filename(filename: &str) -> Result<String> {
    if filename.is_empty() {
        return Err(IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: "density grid filename must not be empty".to_string(),
        });
    }
    if filename.chars().any(|character| character.is_whitespace()) {
        return Err(IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: "density grid filename must not contain whitespace".to_string(),
        });
    }
    Ok(fortran_fixed_string(filename, DENSITY_FILENAME_WIDTH))
}

fn validate_density_grid(grid: &DensityGrid) -> Result<()> {
    let dimensions = grid.kind.dimensions();
    if grid.axes.len() != dimensions {
        return Err(IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: format!(
                "density grid {:?} requires {dimensions} axis row(s), got {}",
                grid.kind,
                grid.axes.len()
            ),
        });
    }
    for (index, value) in grid.origin.iter().copied().enumerate() {
        validate_finite_density_value("origin", index, value)?;
    }
    for (axis_index, axis) in grid.axes.iter().enumerate() {
        if axis.points == 0 {
            return Err(IoError::Parse {
                path: "density.inp".into(),
                line: 0,
                message: format!("density axis {axis_index} point count must be positive"),
            });
        }
        for (coordinate, value) in axis.vector.iter().copied().enumerate() {
            validate_finite_density_value("axis", axis_index * 3 + coordinate, value)?;
        }
    }
    Ok(())
}

fn validate_band_input(input: &BandInput) -> Result<()> {
    validate_finite_control_value("band.inp", "emin", input.energy_mesh.emin)?;
    validate_finite_control_value("band.inp", "emax", input.energy_mesh.emax)?;
    validate_finite_control_value("band.inp", "estep", input.energy_mesh.estep)
}

fn validate_opcons_input(input: &OpconsInput) -> Result<()> {
    for (index, density) in input.number_densities.iter().copied().enumerate() {
        if !density.is_finite() {
            return Err(IoError::Parse {
                path: "opcons.inp".into(),
                line: 0,
                message: format!("opcons number density {index} must be finite"),
            });
        }
    }
    Ok(())
}

fn validate_finite_density_value(field: &'static str, index: usize, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "density.inp".into(),
            line: 0,
            message: format!("density {field}[{index}] must be finite"),
        })
    }
}

fn validate_finite_control_value(
    path: &'static str,
    field: &'static str,
    value: f64,
) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: path.into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn validate_reciprocal_input(input: &ReciprocalInput) -> Result<()> {
    match (input.ispace, input.cell.as_ref()) {
        (0, Some(cell)) => {
            if cell.positions.len() != cell.atom_count {
                return Err(IoError::Parse {
                    path: "reciprocal.inp".into(),
                    line: 0,
                    message: format!(
                        "reciprocal.inp atom_count is {} but has {} positions",
                        cell.atom_count,
                        cell.positions.len()
                    ),
                });
            }
            if cell.potentials.len() != cell.atom_count {
                return Err(IoError::Parse {
                    path: "reciprocal.inp".into(),
                    line: 0,
                    message: format!(
                        "reciprocal.inp atom_count is {} but has {} potentials",
                        cell.atom_count,
                        cell.potentials.len()
                    ),
                });
            }
            Ok(())
        }
        (0, None) => Err(IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: "reciprocal-space input requires a cell block".to_string(),
        }),
        (1, None) => Ok(()),
        (1, Some(_)) => Err(IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: "real-space reciprocal.inp must not include a cell block".to_string(),
        }),
        (ispace, _) => Err(IoError::Parse {
            path: "reciprocal.inp".into(),
            line: 0,
            message: format!("unsupported reciprocal ispace {ispace}"),
        }),
    }
}

fn reciprocal_label_line(labels: &[String]) -> String {
    let mut out = String::new();
    for label in labels {
        out.push_str(&fixed_left(label, 3));
    }
    for _ in 0..14 {
        out.push(' ');
    }
    out
}
