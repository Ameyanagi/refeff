//! Renderer and file I/O for FEFF `dmdw.out` reports.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

use super::parse::parse_dmdw_out;
use super::types::{
    DmdwOutData, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature, DmdwOutTemperatureValue,
};
use super::validate::validate_dmdw_out;

/// Render FEFF-compatible `dmdw.out` text.
pub fn dmdw_out_string(data: &DmdwOutData) -> Result<String> {
    validate_dmdw_out(data)?;
    let Some(header) = &data.header else {
        return Ok(String::new());
    };

    let mut out = String::new();
    writeln!(
        out,
        "# Lanczos recursion order: {:4}",
        header.lanczos_recursion_order
    )?;
    match header.temperature {
        DmdwOutTemperature::Single(temperature) => {
            writeln!(out, "# Temperature: {:7.2}", temperature)?;
        }
        DmdwOutTemperature::ListedBelow => {
            writeln!(out, "# Temperature: (See list Below)")?;
        }
    }
    writeln!(
        out,
        "# Dynamical matrix file: {}",
        header.dynamical_matrix_file
    )?;
    if data.mass_enhancement_header {
        writeln!(out, "Mass Enchancement Factor  ")?;
        return Ok(out);
    }
    writeln!(out)?;

    for section in &data.sections {
        write_separator(&mut out)?;
        write_subject(&mut out, &section.subject)?;
        write_pdos_poles(&mut out, section)?;
        write_einstein(&mut out, section)?;
        write_moments(&mut out, section)?;
        write_path_result(&mut out, section)?;
        write_vfe_result(&mut out, section)?;
        write_u2_result(&mut out, section)?;
        if section.projected_dos_component_computed {
            writeln!(out, " Projected DOS component computed.")?;
            writeln!(out)?;
        }
    }
    write_separator(&mut out)?;
    Ok(out)
}

/// Read FEFF `dmdw.out` text from a file.
pub fn read_dmdw_out(path: impl AsRef<Path>) -> Result<DmdwOutData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_out(&text)
}

/// Write FEFF `dmdw.out` text to a file.
pub fn write_dmdw_out(path: impl AsRef<Path>, data: &DmdwOutData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_out_string(data)?).map_err(|source| IoError::io(path, source))
}

fn write_subject(out: &mut String, subject: &DmdwOutSubject) -> Result<()> {
    match subject {
        DmdwOutSubject::PathIndices(indices) => {
            write!(out, " Path Indices:")?;
            write_index_list(out, indices)?;
            writeln!(out)?;
        }
        DmdwOutSubject::AtomIndex { indices, direction } => {
            write!(out, " Atom Index:")?;
            write_index_list(out, indices)?;
            writeln!(out)?;
            if let Some(direction) = direction {
                writeln!(
                    out,
                    "--------- Direction {direction} ---------------------------"
                )?;
            }
        }
        DmdwOutSubject::TotalPdos => {
            writeln!(out, " Total PDOS results:")?;
        }
        DmdwOutSubject::TotalVfe => {
            writeln!(out, " Total VFE for the paths requested:")?;
        }
    }
    Ok(())
}

fn write_separator(out: &mut String) -> Result<()> {
    writeln!(
        out,
        "--------------------------------------------------------------"
    )?;
    Ok(())
}

fn write_index_list(out: &mut String, indices: &[usize]) -> Result<()> {
    if let Some((first, rest)) = indices.split_first() {
        write!(out, "{first:5}")?;
        for index in rest {
            write!(out, "{index:4}")?;
        }
    }
    Ok(())
}

fn write_pdos_poles(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if section.pdos_poles.is_empty() {
        return Ok(());
    }
    writeln!(out, " PDOS Poles:")?;
    writeln!(out, "     Freq. (THz)    Weight")?;
    for pole in &section.pdos_poles {
        writeln!(out, "     {:8.3}{:18.9}", pole.frequency_thz, pole.weight)?;
    }
    writeln!(out)?;
    Ok(())
}

fn write_einstein(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    let Some(einstein) = &section.einstein else {
        return Ok(());
    };
    match section.subject {
        DmdwOutSubject::TotalPdos | DmdwOutSubject::TotalVfe => writeln!(
            out,
            " PDOS Einstein freq (avg. of single poles), associated temp and eff. force constant: "
        )?,
        _ => writeln!(
            out,
            " PDOS Einstein freq (single pole), associated temp and eff. force constant: "
        )?,
    }
    writeln!(out, " Freq (THz)   Temp (K)   Eff. FC (N/m)")?;
    writeln!(
        out,
        "{:8.3}     {:8.2} {:12.4}",
        einstein.frequency_thz,
        einstein.temperature_kelvin,
        einstein.effective_force_constant_n_per_m
    )?;
    writeln!(out)?;
    Ok(())
}

fn write_moments(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if section.moments.is_empty() {
        return Ok(());
    }
    writeln!(
        out,
        " pDOS n Moments, associated Einstein freqs, temps and eff. force constants:"
    )?;
    writeln!(
        out,
        "  n     Mom (THz^n)   Freq (THz)     Temp (K)    Eff. FC (N/m)"
    )?;
    for moment in &section.moments {
        match (
            moment.frequency_thz,
            moment.temperature_kelvin,
            moment.effective_force_constant_n_per_m,
        ) {
            (Some(frequency), Some(temperature), Some(force_constant)) => writeln!(
                out,
                "{:3}{:14.5}{:14.5}     {:8.2} {:12.4}",
                moment.order, moment.moment_thz_power_n, frequency, temperature, force_constant
            )?,
            _ => writeln!(
                out,
                "{:3}{:14.5}     ---------     --------",
                moment.order, moment.moment_thz_power_n
            )?,
        }
    }
    writeln!(out)?;
    Ok(())
}

fn write_path_result(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if let Some(reduced_mass) = section.reduced_mass_amu {
        writeln!(out, " Path Red. Mass (AMU):{:12.6}", reduced_mass)?;
    }
    if let (Some(length), Some(sigma2)) = (
        section.path_length_angstrom,
        section.sigma2_1e_minus_3_angstrom2,
    ) {
        writeln!(
            out,
            " Path Length (Ang), s^2 (1e-3 Ang^2):{:8.4}{:9.4}",
            length, sigma2
        )?;
    } else if let Some(length) = section.path_length_angstrom
        && !section.sigma2_by_temperature.is_empty()
    {
        writeln!(out, " Path Length (Ang):{:8.4}", length)?;
        writeln!(out, " Temp (K)   s^2 (1e-3 Ang^2)")?;
        write_temperature_values(out, &section.sigma2_by_temperature, 4)?;
        writeln!(out)?;
    }
    Ok(())
}

fn write_vfe_result(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if let Some(value) = section.vibrational_free_energy_ev {
        writeln!(out, " VFE (eV):{:16.6}", value)?;
    }
    if !section.vibrational_free_energy_by_temperature.is_empty() {
        writeln!(out, " Temp (K)        VFE (eV)")?;
        write_temperature_values(out, &section.vibrational_free_energy_by_temperature, 6)?;
    }
    Ok(())
}

fn write_u2_result(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if let Some(value) = section.u2_1e_minus_3_angstrom2 {
        writeln!(out, " u^2 (1e-3 Ang^2):{:8.3}", value)?;
    }
    if !section.u2_by_temperature.is_empty() {
        writeln!(out, " Temp (K)   u^2 (1e-3 Ang^2)")?;
        write_temperature_values(out, &section.u2_by_temperature, 3)?;
    }
    Ok(())
}

fn write_temperature_values(
    out: &mut String,
    rows: &[DmdwOutTemperatureValue],
    decimals: usize,
) -> Result<()> {
    let width = decimals + 4;
    for row in rows {
        writeln!(
            out,
            "{:8.2}     {:width$.*}",
            row.temperature_kelvin, decimals, row.value
        )?;
    }
    Ok(())
}
