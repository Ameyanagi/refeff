//! `fms.inp` writer for FEFF `rdinp` handoff data.

use crate::Result;
use crate::model::FeffDocument;

use super::super::{
    debye_values, do_fms_flag, fms_flag, lmaxph, nph, potential_for_ipot, write_i4_list,
};

/// Render FEFF-compatible `fms.inp` content from an [`FeffDocument`].
pub fn fms_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_fms_inp(document, &mut out)?;
    Ok(out)
}

fn write_fms_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let (tk, thetad, idwopt) = debye_values(document);
    let fms = document.fms.as_ref();

    writeln!(out, "mfms, idwopt, minv")?;
    write_i4_list(
        out,
        [
            fms_flag(document),
            idwopt,
            fms.map(|fms| fms.minv).unwrap_or(0),
        ],
    )?;
    writeln!(out, "rfms2, rdirec, toler1, toler2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        fms.map(|fms| fms.radius).unwrap_or(-1.0),
        fms.map(|fms| fms.rdirec).unwrap_or(-1.0),
        fms.map(|fms| fms.toler1).unwrap_or(0.001),
        fms.map(|fms| fms.toler2).unwrap_or(0.001)
    )?;
    writeln!(out, "tk, thetad, sig2g")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        tk, thetad, document.fine_structure_damping.sig2g
    )?;
    writeln!(out, " lmaxph(0:nph)")?;
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    )?;
    writeln!(out, " the number of decomposi")?;
    writeln!(
        out,
        "{:5}",
        document
            .nrixs
            .as_ref()
            .map(|nrixs| nrixs.ldecmx)
            .unwrap_or(-1)
    )?;
    writeln!(out, " save_gg_slice")?;
    writeln!(out, "{}", if document.ispec == 5 { "T" } else { "F" })?;
    writeln!(out, "do_fms")?;
    write_i4_list(out, [do_fms_flag(document)])?;
    Ok(())
}
