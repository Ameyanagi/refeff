//! Writers for the first `rdinp`-level compatibility outputs.
//!
//! The full FEFF `rdinp` module emits many module input files. This module
//! starts with `atoms.dat`, which is the structural bridge consumed by later
//! FEFF modules, and will grow as the port advances.

use std::collections::BTreeMap;

use crate::model::{Atom, FeffDocument, Potential};
use crate::{IoError, Result};

/// Render all currently supported text outputs from FEFF's `rdinp` stage.
pub fn text_outputs(document: &FeffDocument) -> Result<BTreeMap<&'static str, String>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(".dimensions.dat", dimensions_dat_string(document)?);
    outputs.insert("atoms.dat", atoms_dat_string(document)?);
    outputs.insert("band.inp", band_inp_string());
    outputs.insert("compton.inp", compton_inp_string());
    outputs.insert("crpa.inp", crpa_inp_string());
    outputs.insert("density.inp", density_inp_string());
    outputs.insert("dmdw.inp", dmdw_inp_string(document));
    outputs.insert("eels.inp", eels_inp_string());
    outputs.insert("ff2x.inp", ff2x_inp_string(document));
    outputs.insert("fms.inp", fms_inp_string(document)?);
    outputs.insert("fullspectrum.inp", fullspectrum_inp_string());
    outputs.insert("genfmt.inp", genfmt_inp_string(document));
    outputs.insert("geom.dat", geom_dat_string(document)?);
    outputs.insert("global.inp", global_inp_string(document));
    outputs.insert("hubbard.inp", hubbard_inp_string());
    outputs.insert("ldos.inp", ldos_inp_string(document)?);
    outputs.insert("opcons.inp", opcons_inp_string());
    outputs.insert("paths.inp", paths_inp_string(document));
    outputs.insert("pot.inp", pot_inp_string(document)?);
    outputs.insert("reciprocal.inp", reciprocal_inp_string());
    outputs.insert("rixs.inp", rixs_inp_string(document));
    outputs.insert("screen.inp", screen_inp_string());
    outputs.insert("sfconv.inp", sfconv_inp_string(document));
    outputs.insert("xsph.inp", xsph_inp_string(document)?);
    Ok(outputs)
}

/// Render FEFF-compatible `atoms.dat` content from an [`FeffDocument`].
pub fn atoms_dat_string(document: &FeffDocument) -> Result<String> {
    if document.atoms.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write atoms.dat without ATOMS rows".to_string(),
        });
    }

    let mut out = String::new();
    write_atoms_dat(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `.dimensions.dat` content from an [`FeffDocument`].
pub fn dimensions_dat_string(document: &FeffDocument) -> Result<String> {
    let (nclusx, lx, nphu, nspu) = dimensions_values(document)?;
    Ok(format!("{nclusx:12}{lx:12}{nphu:12}{nspu:12}\n"))
}

/// Render FEFF-compatible `geom.dat` content from an [`FeffDocument`].
pub fn geom_dat_string(document: &FeffDocument) -> Result<String> {
    if document.atoms.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write geom.dat without ATOMS rows".to_string(),
        });
    }

    let mut out = String::new();
    write_geom_dat(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `global.inp` content from an [`FeffDocument`].
pub fn global_inp_string(_document: &FeffDocument) -> String {
    let mut out = String::new();
    write_global_inp(&mut out);
    out
}

/// Render FEFF-compatible `reciprocal.inp` content for real-space inputs.
#[must_use]
pub fn reciprocal_inp_string() -> String {
    "ispace\n   1\n".to_string()
}

/// Render FEFF-compatible `crpa.inp` content with current default controls.
#[must_use]
pub fn crpa_inp_string() -> String {
    " do_CRPA           0\n rcut   1.6000000238418579     \n l_crpa           3\n".to_string()
}

/// Render FEFF-compatible `fullspectrum.inp` content with current defaults.
#[must_use]
pub fn fullspectrum_inp_string() -> String {
    " mFullSpectrum\n           0\n".to_string()
}

/// Render FEFF-compatible default `eels.inp` content.
#[must_use]
pub fn eels_inp_string() -> String {
    concat!(
        "calculate ELNES?\n",
        "   0\n",
        "average? relativistic? cross-terms? Which input?\n",
        "   0   1   1   1   4\n",
        "polarizations to be used ; min step max\n",
        "   1   1   1\n",
        "beam energy in eV\n",
        "      0.00000\n",
        "beam direction in arbitrary units\n",
        "      0.00000      0.00000      0.00000\n",
        "collection and convergence semiangle in rad\n",
        "      0.00000      0.00000\n",
        "qmesh - radial and angular grid size\n",
        "   0   0\n",
        "detector positions - two angles in rad\n",
        "      0.00000      0.00000\n",
        "calculate magic angle if magic=1\n",
        "   0\n",
        "energy for magic angle - eV above threshold\n",
        "      0.00000\n",
    )
    .to_string()
}

/// Render FEFF-compatible default `compton.inp` content.
#[must_use]
pub fn compton_inp_string() -> String {
    concat!(
        "run compton module?\n",
        "           0\n",
        "pqmax, npq\n",
        "   5.00000000            1000\n",
        "ns, nphi, nz, nzp\n",
        "  32  32  32 144\n",
        "smax, phimax, zmax, zpmax\n",
        "      0.00000      6.28319      0.00000     10.00000\n",
        "jpq? rhozzp? force_recalc_jzzp?\n",
        " F F F\n",
        "window_type (0=Step, 1=Hann), window_cutoff\n",
        "           1   0.00000000    \n",
        "temperature (in eV)\n",
        "      0.00000\n",
        "set_chemical_potential? chemical_potential(eV)\n",
        " F   0.00000000    \n",
        "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?\n",
        " F F F F F\n",
        "qhat_x qhat_y qhat_z\n",
        "   0.0000000000000000        0.0000000000000000        1.0000000000000000     \n",
    )
    .to_string()
}

/// Render FEFF-compatible default `band.inp` content.
#[must_use]
pub fn band_inp_string() -> String {
    concat!(
        "mband : calculate bands if = 1\n",
        "   0\n",
        "emin, emax, estep : energy mesh\n",
        "      0.00000      0.00000      0.00000\n",
        "nkp : # points in k-path\n",
        "   0\n",
        "ikpath : type of k-path\n",
        "  -1\n",
        "freeprop :  empty lattice if = T\n",
        " F\n",
    )
    .to_string()
}

/// Render FEFF-compatible default `hubbard.inp` content.
#[must_use]
pub fn hubbard_inp_string() -> String {
    concat!(
        "i_hubbard mldos_hubb U_hubbard J_hubbard fermi_shift l_hubbard\n",
        "           1           1   0.0000000000000000        0.0000000000000000        0.0000000000000000                0\n",
    )
    .to_string()
}

/// Render FEFF-compatible default `opcons.inp` content for current tests.
#[must_use]
pub fn opcons_inp_string() -> String {
    concat!(
        "run_opcons\n",
        " F\n",
        "print_eps\n",
        " F\n",
        "NumDens(0:nphx)\n",
        "  -1.0000000000000000       -1.0000000000000000     \n",
    )
    .to_string()
}

/// Render FEFF-compatible default `screen.inp` content.
#[must_use]
pub fn screen_inp_string() -> String {
    concat!(
        " ner          40\n",
        " nei          20\n",
        " maxl           4\n",
        " irrh           1\n",
        " iend           0\n",
        " lfxc           0\n",
        " emin  -40.000000000000000     \n",
        " emax   0.0000000000000000     \n",
        " eimax   2.0000000000000000     \n",
        " ermin   1.0000000000000000E-003\n",
        " rfms   4.0000000000000000     \n",
        " nrptx0         251\n",
        " icore          -1\n",
    )
    .to_string()
}

/// Render FEFF-compatible empty `density.inp` content.
#[must_use]
pub fn density_inp_string() -> String {
    String::new()
}

/// Render FEFF-compatible `dmdw.inp` content from a `DEBYE` card.
#[must_use]
pub fn dmdw_inp_string(document: &FeffDocument) -> String {
    let mut out = String::new();
    write_dmdw_inp(document, &mut out);
    out
}

/// Render FEFF-compatible default `rixs.inp` content.
#[must_use]
pub fn rixs_inp_string(document: &FeffDocument) -> String {
    let mut out = String::new();
    write_rixs_inp(document, &mut out);
    out
}

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

/// Render FEFF-compatible `ldos.inp` content from an [`FeffDocument`].
pub fn ldos_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_ldos_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `fms.inp` content from an [`FeffDocument`].
pub fn fms_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_fms_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `paths.inp` content from an [`FeffDocument`].
#[must_use]
pub fn paths_inp_string(document: &FeffDocument) -> String {
    let mut out = String::new();
    write_paths_inp(document, &mut out);
    out
}

/// Render FEFF-compatible `genfmt.inp` content from an [`FeffDocument`].
#[must_use]
pub fn genfmt_inp_string(document: &FeffDocument) -> String {
    let mut out = String::new();
    write_genfmt_inp(document, &mut out);
    out
}

/// Render FEFF-compatible `ff2x.inp` content from an [`FeffDocument`].
#[must_use]
pub fn ff2x_inp_string(document: &FeffDocument) -> String {
    let mut out = String::new();
    write_ff2x_inp(document, &mut out);
    out
}

/// Render FEFF-compatible `sfconv.inp` content from an [`FeffDocument`].
#[must_use]
pub fn sfconv_inp_string(document: &FeffDocument) -> String {
    let mut out = String::new();
    write_sfconv_inp(document, &mut out);
    out
}

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

/// Write FEFF-compatible `atoms.dat` content into an arbitrary formatter.
pub fn write_atoms_dat(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "natx =  {:>7}", document.atoms.len()).expect("write to string");
    writeln!(out, "    x       y        z       iph  ").expect("write to string");

    let origin = &document.atoms[0];
    for atom in &document.atoms {
        let distance = atom.distance.unwrap_or_else(|| distance_from(origin, atom));
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:4}{:13.5}",
            atom.x, atom.y, atom.z, atom.ipot, distance
        )
        .expect("write to string");
    }

    Ok(())
}

fn write_geom_dat(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let rows = geometry_rows(document)?;
    let iatph = geometry_model_atoms(document, &rows);
    let nph = iatph.len().saturating_sub(1);

    writeln!(out, "nat, nph = {:5}{:5}", rows.len(), nph).expect("write to string");
    for model_atom in &iatph {
        write!(out, "{model_atom:5}").expect("write to string");
    }
    writeln!(out).expect("write to string");
    writeln!(out, " iat     x       y        z       iph  ").expect("write to string");
    writeln!(out, " {}", "-".repeat(71)).expect("write to string");
    for (idx, row) in rows.iter().enumerate() {
        writeln!(
            out,
            "{:4}{:13.5}{:13.5}{:13.5}{:4}{:4}",
            idx + 1,
            row.x,
            row.y,
            row.z,
            row.ipot,
            1
        )
        .expect("write to string");
    }
    Ok(())
}

fn write_global_inp(out: &mut impl std::fmt::Write) {
    writeln!(out, " nabs, iphabs - CFAVERAGE data").expect("write to string");
    writeln!(out, "{:8}{:8}{:13.5}", 1, 0, 100000.0).expect("write to string");
    writeln!(
        out,
        " ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj"
    )
    .expect("write to string");
    writeln!(
        out,
        "{:5}{:5}{:5}{:12.4}{:12.4}{:5}{:5}{:5}{:5}",
        0, 0, 0, 0.0, 0.0, 0, 0, -1, -1
    )
    .expect("write to string");
    writeln!(out, "evec\t\t  xivec \t   spvec").expect("write to string");
    for _ in 0..3 {
        writeln!(out, "{:13.5}{:13.5}{:13.5}", 0.0, 0.0, 0.0).expect("write to string");
    }
    writeln!(out, " polarization tensor ").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        1.0 / 3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    )
    .expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        0.0,
        0.0,
        1.0 / 3.0,
        0.0,
        0.0,
        0.0
    )
    .expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / 3.0,
        0.0
    )
    .expect("write to string");
    writeln!(out, "evnorm, xivnorm, spvnorm - only used for nrixs").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}{:13.5}", 0.0, 0.0, 0.0).expect("write to string");
    writeln!(out, "nq,    imdff,   qaverage,   mixdff").expect("write to string");
    writeln!(out, "{:12}{:12} T F", 0, 0).expect("write to string");
    writeln!(
        out,
        " q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi"
    )
    .expect("write to string");
}

fn write_pot_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let ihole = document
        .edge
        .as_ref()
        .map(|edge| edge_hole(&edge.label))
        .transpose()?
        .unwrap_or(1);
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
    let central_z = potential_for_ipot(document, 0)?
        .z
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "pot.inp requires numeric Z for absorbing potential".to_string(),
        })?;
    let gamach = core_hole_width(central_z, ihole);

    writeln!(
        out,
        "mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc"
    )
    .expect("write to string");
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}",
        1, nph, ntitle, ihole, ipr1, 0, ixc, 0, 11
    )
    .expect("write to string");
    writeln!(
        out,
        "nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf"
    )
    .expect("write to string");
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}",
        nmix, -1, 0, 0, nscmt, icoul, lfms1, 0
    )
    .expect("write to string");
    if document.titles.is_empty() {
        writeln!(out, "{}", fixed_title("Once upon a time ...")).expect("write to string");
    } else {
        for title in &document.titles {
            writeln!(out, "{}", fixed_title(title)).expect("write to string");
        }
    }
    writeln!(out, "gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        gamach, 0.05, ca1, ecv, 0.0, rfms1, -70.0
    )
    .expect("write to string");
    writeln!(out, " iz, lmaxsc, xnatph, xion, folp").expect("write to string");
    for ipot in 0..=nph {
        let potential = potential_for_ipot(document, ipot)?;
        let z = potential.z.ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("pot.inp requires numeric Z for potential {ipot}"),
        })?;
        let lmaxsc = lmaxsc(potential);
        let xnatph = xnatph(document, potential);
        writeln!(
            out,
            "{:5}{:5}{:20.10}{:20.10}{:20.10}",
            z, lmaxsc, xnatph, 0.0, 1.15
        )
        .expect("write to string");
    }
    writeln!(out, "ExternalPot switch, StartFromFile switch").expect("write to string");
    writeln!(out, " F F").expect("write to string");
    writeln!(out, "OVERLAP option: novr(iph)").expect("write to string");
    for ipot in 0..=nph {
        write!(out, "{:4}", 0).expect("write to string");
        if ipot == nph {
            writeln!(out).expect("write to string");
        }
    }
    writeln!(out, " iphovr  nnovr rovr ").expect("write to string");
    writeln!(out, "ChSh_Type:").expect("write to string");
    writeln!(out, "{:4}", 0).expect("write to string");
    writeln!(out, "ConfigType:").expect("write to string");
    writeln!(out, "{:4}", 1).expect("write to string");
    writeln!(out, "Temperature (in eV):").expect("write to string");
    writeln!(out, "{:21.16}{:17}", 0.0, 1).expect("write to string");
    writeln!(out, "scf_th,  xntol,  nmu").expect("write to string");
    writeln!(out, "           2   1.0000000000000000E-004         100").expect("write to string");
    writeln!(out, "negrid,  emaxscf").expect("write to string");
    writeln!(out, "{:12}{:21.16}     ", 400, 5.0).expect("write to string");
    writeln!(out, "FiniteNucleus, WarnIon").expect("write to string");
    writeln!(out, " F F").expect("write to string");
    writeln!(out, "ramp_scf  rfms_start  nramp").expect("write to string");
    writeln!(out, " F   0.00000000               1").expect("write to string");
    writeln!(out, "tolmu, tolq, tolqp").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}{:13.5}", 0.001, 0.001, 0.0002).expect("write to string");
    Ok(())
}

fn write_ldos_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let ldos = document.ldos.as_ref();
    let ixc = document
        .exchange
        .as_ref()
        .map(|exchange| exchange.ixc)
        .unwrap_or(0);

    writeln!(out, "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc").expect("write to string");
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4} {:7} {:4}",
        if ldos.is_some() { 1 } else { 0 },
        0,
        ixc,
        0,
        0,
        ldos.map(|ldos| ldos.neldos).unwrap_or(101),
        11
    )
    .expect("write to string");
    writeln!(out, "rfms2, emin, emax, eimag, rgrd").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        -1.0,
        ldos.map(|ldos| ldos.emin).unwrap_or(1000.0),
        ldos.map(|ldos| ldos.emax).unwrap_or(0.0),
        ldos.map(|ldos| ldos.eimag).unwrap_or(-1.0),
        0.05
    )
    .expect("write to string");
    writeln!(out, "rdirec, toler1, toler2").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}{:13.5}", -1.0, 0.001, 0.001).expect("write to string");
    writeln!(out, " lmaxph(0:nph)").expect("write to string");
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    );
    writeln!(out, "ldostype").expect("write to string");
    write_i4_list(out, [ldos.map(|ldos| ldos.ldostype).unwrap_or(0)]);
    Ok(())
}

fn write_fms_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let (tk, thetad, idwopt) = debye_values(document);
    let fms = document.fms.as_ref();

    writeln!(out, "mfms, idwopt, minv").expect("write to string");
    write_i4_list(
        out,
        [
            control_flag(document, 2, 1),
            idwopt,
            fms.map(|fms| fms.minv).unwrap_or(0),
        ],
    );
    writeln!(out, "rfms2, rdirec, toler1, toler2").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        fms.map(|fms| fms.radius).unwrap_or(-1.0),
        fms.map(|fms| fms.rdirec).unwrap_or(-1.0),
        fms.map(|fms| fms.toler1).unwrap_or(0.001),
        fms.map(|fms| fms.toler2).unwrap_or(0.001)
    )
    .expect("write to string");
    writeln!(out, "tk, thetad, sig2g").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}{:13.5}", tk, thetad, 0.0).expect("write to string");
    writeln!(out, " lmaxph(0:nph)").expect("write to string");
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    );
    writeln!(out, " the number of decomposi").expect("write to string");
    writeln!(out, "{:5}", -1).expect("write to string");
    writeln!(out, " save_gg_slice").expect("write to string");
    writeln!(out, "F").expect("write to string");
    writeln!(out, "do_fms").expect("write to string");
    write_i4_list(out, [0]);
    Ok(())
}

fn write_paths_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) {
    writeln!(out, "mpath, ms, nncrit, nlegxx, ipr4").expect("write to string");
    write_i4_list(
        out,
        [
            control_flag(document, 3, 1),
            1,
            0,
            7,
            print_flag(document, 3, 0),
        ],
    );
    writeln!(out, "critpw, pcritk, pcrith,  rmax, rfms2").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        2.5,
        0.0,
        0.0,
        document.rpath.unwrap_or(-1.0),
        -1.0
    )
    .expect("write to string");
    writeln!(out, "ica").expect("write to string");
    write_i4_list(out, [-1]);
}

fn write_genfmt_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) {
    writeln!(out, "mfeff, ipr5, iorder, critcw, wnstar").expect("write to string");
    writeln!(
        out,
        "{:4}{:4}{:8}{:13.5}{:>5}",
        control_flag(document, 4, 1),
        print_flag(document, 4, 0),
        2,
        4.0,
        "F"
    )
    .expect("write to string");
    writeln!(out, " the number of decomposi").expect("write to string");
    writeln!(out, "{:5}", -1).expect("write to string");
}

fn write_ff2x_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) {
    let (tk, thetad, idwopt) = debye_values(document);

    writeln!(out, "mchi, ispec, idwopt, ipr6, mbconv, absolu, iGammaCH").expect("write to string");
    write_i4_list(
        out,
        [
            control_flag(document, 5, 1),
            0,
            idwopt,
            print_flag(document, 5, 0),
            0,
            0,
            0,
        ],
    );
    writeln!(out, "vrcorr, vicorr, s02, critcw").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        0.0,
        0.0,
        document.s02.unwrap_or(1.0),
        4.0
    )
    .expect("write to string");
    writeln!(out, "tk, thetad, alphat, thetae, sig2g, sig_gk").expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        tk, thetad, 0.0, 0.0, 0.0, 0.0
    )
    .expect("write to string");
    writeln!(out, "momentum transfer").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}{:13.5}", 0.0, 0.0, 0.0).expect("write to string");
    writeln!(out, " the number of decomposi").expect("write to string");
    writeln!(out, "{:5}", -1).expect("write to string");
    writeln!(out, "electronic temperature").expect("write to string");
    writeln!(out, "{:13.5}", 0.0).expect("write to string");
}

fn write_sfconv_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) {
    writeln!(out, "msfconv, ipse, ipsk").expect("write to string");
    write_i4_list(out, [0, 0, 0]);
    writeln!(out, "wsigk, cen").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}", 0.0, 0.0).expect("write to string");
    writeln!(out, "ispec, ipr6").expect("write to string");
    write_i4_list(out, [0, print_flag(document, 5, 0)]);
    writeln!(out, "cfname").expect("write to string");
    writeln!(out, "NULL        ").expect("write to string");
}

fn write_dmdw_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) {
    let Some(debye) = document.debye.as_ref().filter(|debye| debye.idwopt == 5) else {
        writeln!(out, "-999").expect("write to string");
        return;
    };

    writeln!(out, "{:4}", 1).expect("write to string");
    writeln!(out, "{:4}", debye.dmdw_order).expect("write to string");
    writeln!(out, "{:4}{:11.3}", 1, debye.temperature).expect("write to string");
    writeln!(out, "{:4}", debye.dmdw_type).expect("write to string");
    writeln!(out, "{}", debye.dym_file.as_deref().unwrap_or("feff.dym")).expect("write to string");

    let max_distance = max_interatomic_distance(document);
    match debye.dmdw_route {
        0 => writeln!(out, "{:4}", 0).expect("write to string"),
        1 => {
            writeln!(out, "{:4}", 1).expect("write to string");
            write_dmdw_single_scattering(out, 1, max_distance);
        }
        2 => {
            writeln!(out, "{:4}", 2).expect("write to string");
            write_dmdw_single_scattering(out, 1, max_distance);
            write_dmdw_double_scattering(out, 1, max_distance);
        }
        3 => {
            writeln!(out, "{:4}", 3).expect("write to string");
            write_dmdw_single_scattering(out, 1, max_distance);
            write_dmdw_double_scattering(out, 1, max_distance);
            write_dmdw_triple_scattering(out, 1, max_distance);
        }
        11 => {
            writeln!(out, "{:4}", 1).expect("write to string");
            write_dmdw_single_scattering(out, 0, max_distance);
        }
        12 => {
            writeln!(out, "{:4}", 2).expect("write to string");
            write_dmdw_single_scattering(out, 0, max_distance);
            write_dmdw_double_scattering(out, 0, max_distance);
        }
        13 => {
            writeln!(out, "{:4}", 3).expect("write to string");
            write_dmdw_single_scattering(out, 0, max_distance);
            write_dmdw_double_scattering(out, 0, max_distance);
            write_dmdw_triple_scattering(out, 0, max_distance);
        }
        _ => {}
    }
}

fn write_dmdw_single_scattering(
    out: &mut impl std::fmt::Write,
    absorber_selector: i32,
    max_distance: f64,
) {
    writeln!(
        out,
        "{:4}{:4}{:4}        {:7.2}",
        2,
        absorber_selector,
        0,
        1.1 * max_distance * 1.8897
    )
    .expect("write to string");
}

fn write_dmdw_double_scattering(
    out: &mut impl std::fmt::Write,
    absorber_selector: i32,
    max_distance: f64,
) {
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}    {:7.2}",
        3,
        absorber_selector,
        0,
        0,
        1.1 * max_distance * 2.0 * 1.8897
    )
    .expect("write to string");
}

fn write_dmdw_triple_scattering(
    out: &mut impl std::fmt::Write,
    absorber_selector: i32,
    max_distance: f64,
) {
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:7.2}",
        4,
        absorber_selector,
        0,
        0,
        0,
        1.1 * max_distance * 3.0 * 1.8897
    )
    .expect("write to string");
}

fn write_rixs_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) {
    const HARTREE_EV: f64 = 27.211_396;
    let gamma = 0.1 / (HARTREE_EV * HARTREE_EV);
    let edge = document
        .edge
        .as_ref()
        .map_or("K", |edge| edge.label.as_str());

    writeln!(out, " m_run").expect("write to string");
    writeln!(out, "{:12}", 0).expect("write to string");
    writeln!(out, " gam_ch, gam_exp(1), gam_exp(2)").expect("write to string");
    writeln!(out, "{gamma:20.10}{gamma:20.10}{gamma:20.10}").expect("write to string");
    writeln!(out, " EMinI, EMaxI, EMinF, EMaxF").expect("write to string");
    writeln!(out, "{:20.10}{:20.10}{:20.10}{:20.10}", 0.0, 0.0, 0.0, 0.0).expect("write to string");
    writeln!(out, " xmu").expect("write to string");
    writeln!(out, "  -367493090.02742821     ").expect("write to string");
    writeln!(out, " Readpoles, SkipCalc, MBConv, ReadSigma").expect("write to string");
    writeln!(out, " T F F F").expect("write to string");
    writeln!(out, " nEdges").expect("write to string");
    writeln!(out, "{:12}", 1).expect("write to string");
    writeln!(out, " Edge{:12}", 1).expect("write to string");
    writeln!(out, " {edge}").expect("write to string");
}

fn write_xsph_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let ihole = document
        .edge
        .as_ref()
        .map(|edge| edge_hole(&edge.label))
        .transpose()?
        .unwrap_or(1);
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
    let xkmax = document
        .exafs
        .as_ref()
        .map(|exafs| exafs.xkmax)
        .unwrap_or(20.0);

    writeln!(
        out,
        "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc"
    )
    .expect("write to string");
    write_i4_list(
        out,
        [1, ipr2, ixc, 0, 0, 0, 0, nph, 0, 0, 100, 0, 0, -1, 11],
    );
    writeln!(out, "vr0, vi0").expect("write to string");
    writeln!(out, "{:13.5}{:13.5}", vr0, vi0).expect("write to string");
    writeln!(out, " lmaxph(0:nph)").expect("write to string");
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    );
    writeln!(out, " potlbl(iph)").expect("write to string");
    for ipot in 0..=nph {
        let potential = potential_for_ipot(document, ipot)?;
        write!(out, "{}", fixed_a6(potential.tag.as_deref().unwrap_or("")))
            .expect("write to string");
    }
    writeln!(out).expect("write to string");
    writeln!(out, "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap")
        .expect("write to string");
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        0.05,
        -1.0,
        core_hole_width(central_z, ihole),
        0.07,
        xkmax,
        0.0,
        0.0,
        0.0
    )
    .expect("write to string");
    writeln!(out, "spinph(0:nph)").expect("write to string");
    for ipot in 0..=nph {
        write!(out, "{:13.5}", spinph(document, ipot)).expect("write to string");
    }
    writeln!(out).expect("write to string");
    writeln!(out, "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis").expect("write to string");
    write_i4_list(out, [0, 0, 0, 0, 0, 0]);
    writeln!(out, "electronic temperature").expect("write to string");
    writeln!(out, "{:13.5}", 0.0).expect("write to string");
    writeln!(out, "ChSh_Type:").expect("write to string");
    writeln!(out, "{:4}", 0).expect("write to string");
    writeln!(
        out,
        " the number of decomposition channels ; only used for nrixs"
    )
    .expect("write to string");
    writeln!(out, "{:5}", -1).expect("write to string");
    writeln!(out, "lopt").expect("write to string");
    writeln!(out, " F").expect("write to string");
    writeln!(out, "PrintRL").expect("write to string");
    writeln!(out, " F").expect("write to string");
    Ok(())
}

#[derive(Debug, Clone)]
struct GeometryRow {
    x: f64,
    y: f64,
    z: f64,
    ipot: i32,
    input_index: usize,
    distance2: f64,
}

fn geometry_rows(document: &FeffDocument) -> Result<Vec<GeometryRow>> {
    let absorber_index = absorber_index(document)?;
    let absorber = &document.atoms[absorber_index];
    let mut rows = Vec::with_capacity(document.atoms.len());
    rows.push(GeometryRow {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        ipot: 0,
        input_index: absorber_index,
        distance2: 0.0,
    });

    for (input_index, atom) in document.atoms.iter().enumerate() {
        if input_index == absorber_index {
            continue;
        }
        let x = atom.x - absorber.x;
        let y = atom.y - absorber.y;
        let z = atom.z - absorber.z;
        let distance2 = x * x + y * y + z * z;
        if distance2 > 0.0 {
            rows.push(GeometryRow {
                x,
                y,
                z,
                ipot: atom.ipot,
                input_index,
                distance2,
            });
        }
    }

    rows.sort_by(|lhs, rhs| {
        lhs.distance2
            .total_cmp(&rhs.distance2)
            .then_with(|| lhs.input_index.cmp(&rhs.input_index))
    });
    Ok(rows)
}

fn geometry_model_atoms(document: &FeffDocument, rows: &[GeometryRow]) -> Vec<usize> {
    let max_potential = document
        .potentials
        .iter()
        .map(|potential| potential.ipot.max(0) as usize)
        .max()
        .unwrap_or_else(|| {
            rows.iter()
                .map(|row| row.ipot.max(0) as usize)
                .max()
                .unwrap_or(0)
        });
    let mut iatph = vec![0_usize; max_potential + 1];
    if !iatph.is_empty() {
        iatph[0] = 1;
    }
    for (iph, model_atom) in iatph.iter_mut().enumerate().take(max_potential + 1).skip(1) {
        if let Some((idx, _)) = rows
            .iter()
            .enumerate()
            .find(|(_, row)| row.ipot == iph as i32)
        {
            *model_atom = idx + 1;
        }
    }
    while iatph.len() > 1 && iatph.last() == Some(&0) {
        iatph.pop();
    }
    iatph
}

fn write_i4_list(out: &mut impl std::fmt::Write, values: impl IntoIterator<Item = i32>) {
    for value in values {
        write!(out, "{value:4}").expect("write to string");
    }
    writeln!(out).expect("write to string");
}

fn control_flag(document: &FeffDocument, index: usize, default: i32) -> i32 {
    document
        .control
        .and_then(|control| control.get(index).copied())
        .unwrap_or(default)
}

fn print_flag(document: &FeffDocument, index: usize, default: i32) -> i32 {
    document
        .print
        .and_then(|print| print.get(index).copied())
        .unwrap_or(default)
}

fn debye_values(document: &FeffDocument) -> (f64, f64, i32) {
    document
        .debye
        .as_ref()
        .map(|debye| (debye.temperature, debye.debye_temperature, debye.idwopt))
        .unwrap_or((0.0, 0.0, -1))
}

fn dimensions_values(document: &FeffDocument) -> Result<(usize, i32, i32, i32)> {
    const NCLUSX_HARD_LIMIT: usize = 3000;
    const LX_HARD_LIMIT: i32 = 20;
    const NPHX_HARD_LIMIT: i32 = 31;

    let mut nclusx = dimension_cluster_size(document)?;
    if let Some(limit) = document.dims.map(|dims| dims.nclusx) {
        if limit > 0 {
            nclusx = nclusx.min(limit as usize);
        } else {
            nclusx = nclusx.min(NCLUSX_HARD_LIMIT);
        }
    } else {
        nclusx = nclusx.min(NCLUSX_HARD_LIMIT);
    }

    let mut lx = document
        .potentials
        .iter()
        .map(|potential| lmaxsc(potential).max(lmaxph(potential)))
        .max()
        .unwrap_or(0);
    if let Some(limit) = document.dims.map(|dims| dims.lx) {
        if limit >= 0 {
            lx = lx.min(limit);
        } else {
            lx = lx.min(LX_HARD_LIMIT);
        }
    } else {
        lx = lx.min(LX_HARD_LIMIT);
    }

    let nphu = nph(document)?.clamp(1, NPHX_HARD_LIMIT);
    Ok((nclusx, lx, nphu, 1))
}

fn dimension_cluster_size(document: &FeffDocument) -> Result<usize> {
    if document.atoms.is_empty() {
        return Ok(0);
    }

    let absorber = &document.atoms[absorber_index(document)?];
    let ratmin = nearest_nonabsorber_distance(document).unwrap_or(0.0);
    let rfms1 = document
        .scf
        .as_ref()
        .map(|scf| radius_or_disabled(scf.radius, ratmin))
        .unwrap_or(-1.0);
    let rfms2 = document
        .fms
        .as_ref()
        .map(|fms| radius_or_disabled(fms.radius, ratmin))
        .unwrap_or(-1.0);
    let rmax = document.rpath.unwrap_or(-1.0);
    let rdims = rfms1.max(rfms2).max(rmax);

    if rdims <= 0.0 {
        return Ok(document.atoms.len());
    }

    Ok(document
        .atoms
        .iter()
        .filter(|atom| distance_from(absorber, atom) <= rdims)
        .count())
}

fn radius_or_disabled(radius: f64, nearest_neighbor: f64) -> f64 {
    if radius < nearest_neighbor {
        -1.0
    } else {
        radius
    }
}

fn nph(document: &FeffDocument) -> Result<i32> {
    document
        .potentials
        .iter()
        .map(|potential| potential.ipot)
        .max()
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "POTENTIALS is empty".to_string(),
        })
}

fn potential_for_ipot(document: &FeffDocument, ipot: i32) -> Result<&Potential> {
    document
        .potentials
        .iter()
        .find(|potential| potential.ipot == ipot)
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("missing potential {ipot}"),
        })
}

fn absorber_index(document: &FeffDocument) -> Result<usize> {
    if document.atoms.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "ATOMS is empty".to_string(),
        });
    }

    Ok(document
        .atoms
        .iter()
        .position(|atom| atom.ipot == 0)
        .unwrap_or(0))
}

fn edge_hole(label: &str) -> Result<i32> {
    const EDGES: [&str; 41] = [
        "NO", "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5",
        "N6", "N7", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "P1", "P2", "P3", "P4",
        "P5", "P6", "P7", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3",
    ];
    let normalized = label.trim().to_ascii_uppercase();
    if let Some(idx) = EDGES.iter().position(|edge| *edge == normalized) {
        return Ok(idx as i32);
    }
    normalized.parse::<i32>().map_err(|_| IoError::Parse {
        path: Default::default(),
        line: 0,
        message: format!("unknown EDGE {label:?}"),
    })
}

fn fixed_title(title: &str) -> String {
    let mut out: String = title.chars().take(80).collect();
    while out.len() < 80 {
        out.push(' ');
    }
    out
}

fn fixed_a6(value: &str) -> String {
    let mut out: String = value.chars().take(6).collect();
    while out.len() < 6 {
        out.push(' ');
    }
    out
}

fn lmaxsc(potential: &Potential) -> i32 {
    let default = match potential.z {
        Some(z) if z < 6 => 1,
        Some(z) if z < 55 => 2,
        Some(_) => 3,
        None => 0,
    };
    let requested = potential.lmax1.unwrap_or(default);
    requested.min(2)
}

fn lmaxph(potential: &Potential) -> i32 {
    let default = match potential.z {
        Some(z) if z < 6 => 2,
        Some(_) => 3,
        None => 0,
    };
    potential.lmax2.unwrap_or(default)
}

fn xnatph(document: &FeffDocument, potential: &Potential) -> f64 {
    if let Some(xnatph) = potential.xnatph {
        return xnatph;
    }
    let count = document
        .atoms
        .iter()
        .filter(|atom| atom.ipot == potential.ipot)
        .count();
    if count == 0 { 1.0 } else { count as f64 }
}

fn spinph(document: &FeffDocument, ipot: i32) -> f64 {
    potential_for_ipot(document, ipot)
        .ok()
        .and_then(|potential| potential.spinph)
        .unwrap_or(0.0)
}

fn nearest_nonabsorber_distance(document: &FeffDocument) -> Option<f64> {
    let origin = document.atoms.first()?;
    document
        .atoms
        .iter()
        .skip(1)
        .map(|atom| atom.distance.unwrap_or_else(|| distance_from(origin, atom)))
        .filter(|distance| *distance > 0.0)
        .min_by(f64::total_cmp)
}

fn max_interatomic_distance(document: &FeffDocument) -> f64 {
    let mut max_distance = 0.0_f64;
    for (idx, atom) in document.atoms.iter().enumerate() {
        for other in document.atoms.iter().skip(idx + 1) {
            max_distance = max_distance.max(distance_from(atom, other));
        }
    }
    max_distance
}

fn core_hole_width(z: i32, ihole: i32) -> f64 {
    if ihole <= 0 {
        return 0.0;
    }
    if ihole > 16 {
        return 0.1;
    }

    // FEFF's SETGAM stores Rahkonen-Krause widths by edge. This first slice
    // ports the K-shell row, which covers the generated Cu reference tests.
    let (grid_z, grid_gamma) = match ihole {
        1 => (
            [0.99, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 95.1],
            [0.02, 0.28, 0.75, 4.8, 10.5, 21.0, 60.0, 105.0],
        ),
        _ => return 0.1,
    };
    let z = f64::from(z);
    let idx = grid_z
        .windows(2)
        .position(|window| z < window[1])
        .unwrap_or(grid_z.len() - 2);
    let x0 = grid_z[idx];
    let x1 = grid_z[idx + 1];
    let y0 = f64::log10(grid_gamma[idx]);
    let y1 = f64::log10(grid_gamma[idx + 1]);
    10.0_f64.powf(y0 + (z - x0) * (y1 - y0) / (x1 - x0))
}

fn distance_from(origin: &Atom, atom: &Atom) -> f64 {
    let dx = atom.x - origin.x;
    let dy = atom.y - origin.y;
    let dz = atom.z - origin.z;
    dx.hypot(dy).hypot(dz)
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput};

    use super::{
        atoms_dat_string, dimensions_dat_string, dmdw_inp_string, geom_dat_string,
        global_inp_string, pot_inp_string, rixs_inp_string, xsph_inp_string,
    };

    #[test]
    fn writes_atoms_dat_with_feff_widths() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 2.0 2.0 1 Cu1
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");
        let atoms = atoms_dat_string(&doc).expect("atoms.dat");

        assert_eq!(
            atoms,
            "natx =        2\n    x       y        z       iph  \n      0.00000      0.00000      0.00000   0      0.00000\n      1.00000      2.00000      2.00000   1      3.00000\n"
        );
    }

    #[test]
    fn writes_global_inp_defaults() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");
        let global = global_inp_string(&doc);

        assert!(global.contains(" nabs, iphabs - CFAVERAGE data\n       1       0 100000.00000\n"));
        assert!(global.contains(" polarization tensor \n      0.33333"));
    }

    #[test]
    fn writes_dimensions_dat_from_cluster_radius() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RPATH 2.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
3.0 0.0 0.0 1 Cu2
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");

        assert_eq!(
            dimensions_dat_string(&doc).expect("dimensions.dat"),
            "           2           3           1           1\n"
        );
    }

    #[test]
    fn writes_geom_dat_with_sorted_relative_cluster() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
2.0 0.0 0.0 1 Cu2
1.0 0.0 0.0 1 Cu1
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");

        assert_eq!(
            geom_dat_string(&doc).expect("geom.dat"),
            concat!(
                "nat, nph =     3    1\n",
                "    1    2\n",
                " iat     x       y        z       iph  \n",
                " -----------------------------------------------------------------------\n",
                "   1      0.00000      0.00000      0.00000   0   1\n",
                "   2      1.00000      0.00000      0.00000   1   1\n",
                "   3      2.00000      0.00000      0.00000   1   1\n",
            )
        );
    }

    #[test]
    fn writes_pot_inp_for_copper_defaults() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");
        let pot = pot_inp_string(&doc).expect("pot.inp");

        assert!(pot.contains("   1   1   1   1   0   0   0   0  11\n"));
        assert!(pot.contains("      1.72919      0.05000      0.00000    -40.00000"));
        assert!(pot.contains("   29    2        1.0000000000"));
    }

    #[test]
    fn writes_xsph_inp_for_copper_exafs() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
EXAFS 20.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");
        let xsph = xsph_inp_string(&doc).expect("xsph.inp");

        assert!(xsph.contains("   1   0   0   0   0   0   0   1   0   0 100   0   0  -1  11\n"));
        assert!(xsph.contains("Cu    Cu    \n"));
        assert!(xsph.contains("      0.05000     -1.00000      1.72919      0.07000     20.00000"));
    }

    #[test]
    fn writes_dmdw_inp_for_dynamical_matrix_debye() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DEBYE 450 315 5 feff.dym 6 0 1
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");

        assert_eq!(
            dmdw_inp_string(&doc),
            "   1\n   6\n   1    450.000\n   0\nfeff.dym\n   1\n   2   1   0           2.08\n"
        );
    }

    #[test]
    fn writes_rixs_inp_defaults() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EDGE K
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");

        assert_eq!(
            rixs_inp_string(&doc),
            concat!(
                " m_run\n",
                "           0\n",
                " gam_ch, gam_exp(1), gam_exp(2)\n",
                "        0.0001350512        0.0001350512        0.0001350512\n",
                " EMinI, EMaxI, EMinF, EMaxF\n",
                "        0.0000000000        0.0000000000        0.0000000000        0.0000000000\n",
                " xmu\n",
                "  -367493090.02742821     \n",
                " Readpoles, SkipCalc, MBConv, ReadSigma\n",
                " T F F F\n",
                " nEdges\n",
                "           1\n",
                " Edge           1\n",
                " K\n",
            )
        );
    }
}
