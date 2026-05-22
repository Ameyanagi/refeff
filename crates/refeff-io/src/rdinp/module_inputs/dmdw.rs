//! `dmdw.inp` writer for FEFF `rdinp` handoff data.

use crate::Result;
use crate::model::FeffDocument;

use super::super::max_interatomic_distance;

/// Render FEFF-compatible `dmdw.inp` content from a `DEBYE` card.
pub fn dmdw_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_dmdw_inp(document, &mut out)?;
    Ok(out)
}

fn write_dmdw_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let Some(debye) = document.debye.as_ref().filter(|debye| debye.idwopt == 5) else {
        writeln!(out, "-999")?;
        return Ok(());
    };

    writeln!(out, "{:4}", 1)?;
    writeln!(out, "{:4}", debye.dmdw_order)?;
    writeln!(out, "{:4}{:11.3}", 1, debye.temperature)?;
    writeln!(out, "{:4}", debye.dmdw_type)?;
    writeln!(out, "{}", debye.dym_file.as_deref().unwrap_or("feff.dym"))?;

    let max_distance = max_interatomic_distance(document);
    match debye.dmdw_route {
        0 => writeln!(out, "{:4}", 0)?,
        1 => {
            writeln!(out, "{:4}", 1)?;
            write_dmdw_single_scattering(out, 1, max_distance)?;
        }
        2 => {
            writeln!(out, "{:4}", 2)?;
            write_dmdw_single_scattering(out, 1, max_distance)?;
            write_dmdw_double_scattering(out, 1, max_distance)?;
        }
        3 => {
            writeln!(out, "{:4}", 3)?;
            write_dmdw_single_scattering(out, 1, max_distance)?;
            write_dmdw_double_scattering(out, 1, max_distance)?;
            write_dmdw_triple_scattering(out, 1, max_distance)?;
        }
        11 => {
            writeln!(out, "{:4}", 1)?;
            write_dmdw_single_scattering(out, 0, max_distance)?;
        }
        12 => {
            writeln!(out, "{:4}", 2)?;
            write_dmdw_single_scattering(out, 0, max_distance)?;
            write_dmdw_double_scattering(out, 0, max_distance)?;
        }
        13 => {
            writeln!(out, "{:4}", 3)?;
            write_dmdw_single_scattering(out, 0, max_distance)?;
            write_dmdw_double_scattering(out, 0, max_distance)?;
            write_dmdw_triple_scattering(out, 0, max_distance)?;
        }
        _ => {}
    }
    Ok(())
}

fn write_dmdw_single_scattering(
    out: &mut impl std::fmt::Write,
    absorber_selector: i32,
    max_distance: f64,
) -> Result<()> {
    writeln!(
        out,
        "{:4}{:4}{:4}        {:7.2}",
        2,
        absorber_selector,
        0,
        1.1 * max_distance * 1.8897
    )?;
    Ok(())
}

fn write_dmdw_double_scattering(
    out: &mut impl std::fmt::Write,
    absorber_selector: i32,
    max_distance: f64,
) -> Result<()> {
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}    {:7.2}",
        3,
        absorber_selector,
        0,
        0,
        1.1 * max_distance * 2.0 * 1.8897
    )?;
    Ok(())
}

fn write_dmdw_triple_scattering(
    out: &mut impl std::fmt::Write,
    absorber_selector: i32,
    max_distance: f64,
) -> Result<()> {
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:7.2}",
        4,
        absorber_selector,
        0,
        0,
        0,
        1.1 * max_distance * 3.0 * 1.8897
    )?;
    Ok(())
}
