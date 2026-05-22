//! `xsph.inp` writer for FEFF `rdinp` handoff data.

use crate::model::FeffDocument;
use crate::{IoError, Result};

use super::super::{
    control_flag, core_hole_width_for_handoff, document_ihole, fixed_a6, fortran_bool, lmaxph, nph,
    output_ispec, potential_for_ipot, spinph, write_i4_list,
};

/// Render FEFF-compatible `xsph.inp` content from an [`FeffDocument`].
pub fn xsph_inp_string(document: &FeffDocument) -> Result<String> {
    if document.potentials.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write xsph.inp without POTENTIALS rows".to_string(),
        });
    }

    let mut out = String::new();
    write_xsph_inp(document, &mut out)?;
    Ok(out)
}

fn write_xsph_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let ihole = document_ihole(document)?;
    let central_z = potential_for_ipot(document, 0)?
        .z
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "xsph.inp requires numeric Z for absorbing potential".to_string(),
        })?;
    let ixc = document
        .exchange
        .as_ref()
        .map(|exchange| exchange.ixc)
        .unwrap_or(0);
    let (vr0, vi0) = document
        .exchange
        .as_ref()
        .map(|exchange| (exchange.vr0, exchange.vi0))
        .unwrap_or((0.0, 0.0));
    let ipr2 = document.print.map(|print| print[1]).unwrap_or(0);
    let spectrum_grid = document.spectrum_grid;

    writeln!(
        out,
        "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc"
    )?;
    write_i4_list(
        out,
        [
            control_flag(document, 1, 1),
            ipr2,
            ixc,
            spectrum_grid.ixc0,
            output_ispec(document),
            document.lreal,
            document.fms.as_ref().map(|fms| fms.lfms).unwrap_or(0),
            nph,
            document.nrixs.as_ref().map(|_| 30).unwrap_or(0),
            document.i_plsmn,
            document.n_poles,
            document.xsph_handoff.core_hole_broadening,
            document.i_grid,
            document.xsph_handoff.core_state,
            document.iscfxc,
        ],
    )?;
    writeln!(out, "vr0, vi0")?;
    writeln!(out, "{:13.5}{:13.5}", vr0, vi0)?;
    writeln!(out, " lmaxph(0:nph)")?;
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    )?;
    writeln!(out, " potlbl(iph)")?;
    for ipot in 0..=nph {
        let potential = potential_for_ipot(document, ipot)?;
        write!(out, "{}", fixed_a6(potential.tag.as_deref().unwrap_or("")))?;
    }
    writeln!(out)?;
    writeln!(out, "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        document.rgrid,
        document.fms.as_ref().map(|fms| fms.radius).unwrap_or(-1.0),
        core_hole_width_for_handoff(document, central_z, ihole)?,
        spectrum_grid.xkstep,
        spectrum_grid.xkmax,
        spectrum_grid.vixan,
        document.xsph_handoff.eps0,
        document.xsph_handoff.egap
    )?;
    writeln!(out, "spinph(0:nph)")?;
    for ipot in 0..=nph {
        write!(out, "{:13.5}", spinph(document, ipot))?;
    }
    writeln!(out)?;
    writeln!(out, "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis")?;
    write_i4_list(
        out,
        [
            document.xsph_advanced.izstd,
            document.xsph_advanced.ifxc,
            document.xsph_advanced.ipmbse,
            document.xsph_advanced.itdlda,
            document.xsph_advanced.nonlocal,
            document.xsph_advanced.ibasis,
        ],
    )?;
    writeln!(out, "electronic temperature")?;
    writeln!(out, "{:13.5}", document.electronic_temperature)?;
    writeln!(out, "ChSh_Type:")?;
    writeln!(out, "{:4}", document.chsh_type)?;
    writeln!(
        out,
        " the number of decomposition channels ; only used for nrixs"
    )?;
    writeln!(
        out,
        "{:5}",
        document
            .nrixs
            .as_ref()
            .map(|nrixs| nrixs.ldecmx)
            .unwrap_or(-1)
    )?;
    writeln!(out, "lopt")?;
    writeln!(out, " {}", fortran_bool(document.xsph_handoff.set_edge))?;
    writeln!(out, "PrintRL")?;
    writeln!(
        out,
        " {}",
        fortran_bool(document.xsph_handoff.print_radial_wavefunctions)
    )?;
    Ok(())
}
