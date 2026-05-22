//! `pot.inp` writer for FEFF `rdinp` handoff data.

use crate::model::FeffDocument;
use crate::{IoError, Result};

use super::super::{
    automatic_folp_flag, control_flag, core_hole_width_for_handoff, document_ihole, fixed_title,
    fortran_bool, interstitial_volume, lmaxsc, nearest_nonabsorber_distance, nph, output_ispec,
    overlap_shell_count, potential_for_ipot, potential_ionization, potential_overlap_factor,
    write_overlap_shells, xnatph,
};

/// Render FEFF-compatible `pot.inp` content from an [`FeffDocument`].
pub fn pot_inp_string(document: &FeffDocument) -> Result<String> {
    if document.potentials.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write pot.inp without POTENTIALS rows".to_string(),
        });
    }

    let mut out = String::new();
    write_pot_inp(document, &mut out)?;
    Ok(out)
}

fn write_pot_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let ihole = document_ihole(document)?;
    let ipr1 = document.print.map(|print| print[0]).unwrap_or(0);
    let ixc = document
        .exchange
        .as_ref()
        .map(|exchange| exchange.ixc)
        .unwrap_or(0);
    let scf = document.scf.as_ref();
    let ntitle = document.titles.len().max(1);
    let nscmt = scf.map(|scf| scf.iterations).unwrap_or(0);
    let ca1 = scf.map(|scf| scf.ca).unwrap_or(0.0);
    let rfms1 = scf
        .map(|scf| scf.radius)
        .map(|radius| match nearest_nonabsorber_distance(document) {
            Some(ratmin) if radius < ratmin => -1.0,
            _ => radius,
        })
        .unwrap_or(-1.0);
    let lfms1 = scf.map(|scf| scf.lfms).unwrap_or(0);
    let nmix = scf.map(|scf| scf.nmix).unwrap_or(1);
    let ecv = scf.map(|scf| scf.ecv).unwrap_or(-40.0);
    let icoul = scf.map(|scf| scf.icoul).unwrap_or(0);
    let inters = document
        .interstitial
        .map(|interstitial| interstitial.mode)
        .unwrap_or(0);
    let totvol = interstitial_volume(document);
    let central_z = potential_for_ipot(document, 0)?
        .z
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "pot.inp requires numeric Z for absorbing potential".to_string(),
        })?;
    let gamach = core_hole_width_for_handoff(document, central_z, ihole)?;

    writeln!(
        out,
        "mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc"
    )?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}",
        control_flag(document, 0, 1),
        nph,
        ntitle,
        ihole,
        ipr1,
        automatic_folp_flag(document),
        ixc,
        output_ispec(document),
        document.iscfxc
    )?;
    writeln!(
        out,
        "nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf"
    )?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}",
        nmix,
        document.nohole,
        i32::from(document.jump_removal),
        inters,
        nscmt,
        icoul,
        lfms1,
        i32::from(document.unfreezef)
    )?;
    if document.titles.is_empty() {
        writeln!(out, "{}", fixed_title("Once upon a time ..."))?;
    } else {
        for title in &document.titles {
            writeln!(out, "{}", fixed_title(title))?;
        }
    }
    writeln!(out, "gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        gamach, document.rgrid, ca1, ecv, totvol, rfms1, document.corval_emin
    )?;
    writeln!(out, " iz, lmaxsc, xnatph, xion, folp")?;
    for ipot in 0..=nph {
        let potential = potential_for_ipot(document, ipot)?;
        let z = potential.z.ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("pot.inp requires numeric Z for potential {ipot}"),
        })?;
        let lmaxsc = lmaxsc(document, potential);
        let xnatph = xnatph(document, potential);
        writeln!(
            out,
            "{:5}{:5}{:20.10}{:20.10}{:20.10}",
            z,
            lmaxsc,
            xnatph,
            potential_ionization(document, ipot),
            potential_overlap_factor(document, ipot)
        )?;
    }
    writeln!(out, "ExternalPot switch, StartFromFile switch")?;
    writeln!(
        out,
        " {} {}",
        fortran_bool(document.external_pot),
        fortran_bool(document.restart_from_pot_bin)
    )?;
    writeln!(out, "OVERLAP option: novr(iph)")?;
    for ipot in 0..=nph {
        write!(out, "{:4}", overlap_shell_count(document, ipot))?;
        if ipot == nph {
            writeln!(out)?;
        }
    }
    writeln!(out, " iphovr  nnovr rovr ")?;
    write_overlap_shells(document, out, nph)?;
    writeln!(out, "ChSh_Type:")?;
    writeln!(out, "{:4}", document.chsh_type)?;
    writeln!(out, "ConfigType:")?;
    writeln!(out, "{:4}", document.config_type)?;
    writeln!(out, "Temperature (in eV):")?;
    write_pot_temperature(out, document.electronic_temperature)?;
    writeln!(out, "scf_th,  xntol,  nmu")?;
    write_pot_thermal_scf(
        out,
        document.scf_thermal.iscfth,
        document.scf_thermal.xntol,
        document.scf_thermal.nmu,
    )?;
    writeln!(out, "negrid,  emaxscf")?;
    writeln!(
        out,
        "{:12}{:21.16}     ",
        document.scf_thermal.negrid, document.scf_thermal.emaxscf
    )?;
    writeln!(out, "FiniteNucleus, WarnIon")?;
    writeln!(
        out,
        " {} {}",
        fortran_bool(document.finite_nucleus),
        fortran_bool(document.warn_ion)
    )?;
    writeln!(out, "ramp_scf  rfms_start  nramp")?;
    writeln!(
        out,
        " {}{:13.8}{:16}",
        fortran_bool(document.scf_ramp.enabled),
        document.scf_ramp.rfms_start,
        document.scf_ramp.nramp
    )?;
    writeln!(out, "tolmu, tolq, tolqp")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        document.scf_tolerances.tolmu, document.scf_tolerances.tolq, document.scf_tolerances.tolqp
    )?;
    Ok(())
}

fn write_pot_temperature(out: &mut impl std::fmt::Write, temperature: f64) -> Result<()> {
    if temperature == 0.0 {
        writeln!(out, "{temperature:21.16}{:17}", 1)?;
    } else if temperature.abs() < 0.1 {
        let exponential = pad_exponent(format!("{temperature:24.16E}"));
        writeln!(out, "{exponential}{:12}", 1)?;
    } else if temperature.abs() < 1.0 {
        writeln!(out, "{temperature:21.17}{:17}", 1)?;
    } else {
        writeln!(out, "{temperature:21.16}{:17}", 1)?;
    }
    Ok(())
}

fn write_pot_thermal_scf(
    out: &mut impl std::fmt::Write,
    iscfth: i32,
    xntol: f64,
    nmu: i32,
) -> Result<()> {
    if iscfth == 2 && xntol == 1.0e-4 && nmu == 100 {
        writeln!(out, "           2   1.0000000000000000E-004         100")?;
    } else {
        let exponential = pad_exponent(format!("{xntol:24.16E}"));
        writeln!(out, "{iscfth:12}{exponential}{nmu:12}")?;
    }
    Ok(())
}

fn pad_exponent(value: String) -> String {
    let Some(index) = value.rfind('E') else {
        return value;
    };
    let (mantissa, exponent) = value.split_at(index + 1);
    let (sign, digits) = exponent.split_at(1);
    format!("{mantissa}{sign}{digits:0>3}")
}
