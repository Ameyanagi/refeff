//! `rixs.inp` writer for FEFF `rdinp` handoff data.

use crate::Result;
use crate::model::FeffDocument;

use super::super::fortran_bool;

/// Render FEFF-compatible default `rixs.inp` content.
pub fn rixs_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_rixs_inp(document, &mut out)?;
    Ok(out)
}

fn write_rixs_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    const HARTREE_EV: f64 = 27.211_396;
    let gamma_default = 0.1 / (HARTREE_EV * HARTREE_EV);
    let gamma_exp = document
        .rixs
        .gamma_exp
        .map(|gamma| gamma.map_or(gamma_default, |gamma| gamma / HARTREE_EV));
    let xmu = document
        .rixs
        .xmu
        .map_or(-1.0e10 / HARTREE_EV, |xmu| xmu / HARTREE_EV);
    let edge_count = if document.rixs.run {
        document.rixs.edges.len()
    } else {
        1
    };

    writeln!(out, " m_run")?;
    writeln!(out, "{:12}", i32::from(document.rixs.run))?;
    writeln!(out, " gam_ch, gam_exp(1), gam_exp(2)")?;
    writeln!(
        out,
        "{gamma_default:20.10}{:20.10}{:20.10}",
        gamma_exp[0], gamma_exp[1]
    )?;
    writeln!(out, " EMinI, EMaxI, EMinF, EMaxF")?;
    writeln!(out, "{:20.10}{:20.10}{:20.10}{:20.10}", 0.0, 0.0, 0.0, 0.0)?;
    writeln!(out, " xmu")?;
    writeln!(out, " {xmu:20.8}     ")?;
    writeln!(out, " Readpoles, SkipCalc, MBConv, ReadSigma")?;
    writeln!(out, " T F {} F", fortran_bool(document.rixs.mbconv))?;
    writeln!(out, " nEdges")?;
    writeln!(out, "{edge_count:12}")?;
    for (idx, edge) in document.rixs.edges.iter().take(edge_count).enumerate() {
        writeln!(out, " Edge{:12}", idx + 1)?;
        writeln!(out, " {edge}")?;
    }
    Ok(())
}
