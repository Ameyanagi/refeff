//! `ldos.inp` writer for FEFF `rdinp` handoff data.

use crate::Result;
use crate::ldos_input::write_ldos_mesh_row;
use crate::model::FeffDocument;

use super::super::{lmaxph, nph, potential_for_ipot, write_i4_list};

/// Render FEFF-compatible `ldos.inp` content from an [`FeffDocument`].
pub fn ldos_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_ldos_inp(document, &mut out)?;
    Ok(out)
}

fn write_ldos_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let ldos = document.ldos.as_ref();
    let fms = document.fms.as_ref();
    let ixc = document
        .exchange
        .as_ref()
        .map(|exchange| exchange.ixc)
        .unwrap_or(0);

    writeln!(out, "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc")?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4} {:7} {:4}",
        if ldos.is_some() { 1 } else { 0 },
        fms.map(|fms| fms.lfms).unwrap_or(0),
        ixc,
        document.spin,
        fms.map(|fms| fms.minv).unwrap_or(0),
        ldos.map(|ldos| ldos.neldos).unwrap_or(101),
        document.iscfxc
    )?;
    writeln!(out, "rfms2, emin, emax, eimag, rgrd")?;
    write_ldos_mesh_row(
        out,
        fms.map(|fms| fms.radius).unwrap_or(-1.0),
        ldos.map(|ldos| ldos.emin).unwrap_or(1000.0),
        ldos.map(|ldos| ldos.emax).unwrap_or(0.0),
        ldos.map(|ldos| ldos.eimag).unwrap_or(-1.0),
        document.rgrid,
    )?;
    writeln!(out, "rdirec, toler1, toler2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        fms.map(|fms| fms.rdirec).unwrap_or(-1.0),
        fms.map(|fms| fms.toler1).unwrap_or(0.001),
        fms.map(|fms| fms.toler2).unwrap_or(0.001)
    )?;
    writeln!(out, " lmaxph(0:nph)")?;
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    )?;
    writeln!(out, "ldostype")?;
    write_i4_list(out, [ldos.map(|ldos| ldos.ldostype).unwrap_or(0)])?;
    Ok(())
}
