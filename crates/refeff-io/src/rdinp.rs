//! Writers for the first `rdinp`-level compatibility outputs.
//!
//! The full FEFF `rdinp` module emits many module input files. This module
//! starts with `atoms.dat`, which is the structural bridge consumed by later
//! FEFF modules, and will grow as the port advances.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::model::{Atom, FeffDocument, Potential};
use crate::{IoError, Result};
use refeff_core::core_hole_width_ev;

/// Render all currently supported text outputs from FEFF's `rdinp` stage.
pub fn text_outputs(document: &FeffDocument) -> Result<BTreeMap<&'static str, String>> {
    let mut outputs = BTreeMap::new();
    if !document.reciprocal && !document.potentials.is_empty() && !document.atoms.is_empty() {
        outputs.insert(".dimensions.dat", dimensions_dat_string(document)?);
    }
    if !document.reciprocal && !document.atoms.is_empty() {
        outputs.insert("atoms.dat", atoms_dat_string(document)?);
    }
    outputs.insert("band.inp", band_inp_string());
    outputs.insert("compton.inp", compton_inp_string(document)?);
    outputs.insert("crpa.inp", crpa_inp_string(document));
    outputs.insert("density.inp", density_inp_string());
    outputs.insert("dmdw.inp", dmdw_inp_string(document)?);
    outputs.insert("eels.inp", eels_inp_string(document)?);
    outputs.insert("ff2x.inp", ff2x_inp_string(document)?);
    if !document.potentials.is_empty() {
        outputs.insert("fms.inp", fms_inp_string(document)?);
    }
    outputs.insert("fullspectrum.inp", fullspectrum_inp_string());
    outputs.insert("genfmt.inp", genfmt_inp_string(document)?);
    if !document.reciprocal && !document.atoms.is_empty() {
        outputs.insert("geom.dat", geom_dat_string(document)?);
    }
    outputs.insert("global.inp", global_inp_string(document)?);
    outputs.insert("hubbard.inp", hubbard_inp_string(document));
    if !document.potentials.is_empty() {
        outputs.insert("ldos.inp", ldos_inp_string(document)?);
    }
    if !document.potentials.is_empty() {
        outputs.insert("opcons.inp", opcons_inp_string(document));
    }
    outputs.insert("paths.inp", paths_inp_string(document)?);
    if !document.potentials.is_empty() {
        outputs.insert("pot.inp", pot_inp_string(document)?);
    }
    if !document.reciprocal {
        outputs.insert("reciprocal.inp", reciprocal_inp_string());
    }
    outputs.insert("rixs.inp", rixs_inp_string(document)?);
    outputs.insert("screen.inp", screen_inp_string());
    outputs.insert("sfconv.inp", sfconv_inp_string(document)?);
    if !document.potentials.is_empty() {
        outputs.insert("xsph.inp", xsph_inp_string(document)?);
    }
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
pub fn global_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_global_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `reciprocal.inp` content for real-space inputs.
#[must_use]
pub fn reciprocal_inp_string() -> String {
    "ispace\n   1\n".to_string()
}

/// Render FEFF-compatible `crpa.inp` content with current CRPA controls.
#[must_use]
pub fn crpa_inp_string(document: &FeffDocument) -> String {
    let crpa = document.crpa;
    format!(
        " do_CRPA{:12}\n rcut{:21.16}     \n l_crpa{:12}\n",
        i32::from(crpa.enabled),
        crpa.rcut,
        crpa.l
    )
}

/// Render FEFF-compatible `fullspectrum.inp` content with current defaults.
#[must_use]
pub fn fullspectrum_inp_string() -> String {
    " mFullSpectrum\n           0\n".to_string()
}

/// Render FEFF-compatible default `eels.inp` content.
pub fn eels_inp_string(document: &FeffDocument) -> Result<String> {
    let eels = document.eels;
    let beam_direction = if eels.enabled {
        eels.beam_direction
    } else {
        document
            .nrixs
            .map(|nrixs| nrixs.qvec)
            .filter(|vector| vector_norm(*vector) > 0.0)
            .unwrap_or(document.incidence_vector)
    };
    let mut out = String::new();
    writeln!(out, "calculate ELNES?")?;
    write_i4_list(&mut out, [i32::from(eels.enabled)])?;
    writeln!(out, "average? relativistic? cross-terms? Which input?")?;
    write_i4_list(
        &mut out,
        [
            eels.average,
            eels.relativistic,
            eels.cross_terms,
            eels.input,
            eels.spectrum_column,
        ],
    )?;
    writeln!(out, "polarizations to be used ; min step max")?;
    write_i4_list(
        &mut out,
        [
            eels.polarization_min,
            eels.polarization_step,
            eels.polarization_max,
        ],
    )?;
    writeln!(out, "beam energy in eV")?;
    writeln!(out, "{:13.5}", eels.beam_energy)?;
    writeln!(out, "beam direction in arbitrary units")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        beam_direction[0], beam_direction[1], beam_direction[2]
    )?;
    writeln!(out, "collection and convergence semiangle in rad")?;
    writeln!(
        out,
        "{:13.5}{:13.5}",
        eels.collection_angle, eels.convergence_angle
    )?;
    writeln!(out, "qmesh - radial and angular grid size")?;
    write_i4_list(&mut out, [eels.qmesh_radial, eels.qmesh_angular])?;
    writeln!(out, "detector positions - two angles in rad")?;
    writeln!(out, "{:13.5}{:13.5}", eels.detector[0], eels.detector[1])?;
    writeln!(out, "calculate magic angle if magic=1")?;
    write_i4_list(&mut out, [eels.magic])?;
    writeln!(out, "energy for magic angle - eV above threshold")?;
    writeln!(out, "{:13.5}", eels.magic_energy)?;
    Ok(out)
}

/// Render FEFF-compatible `compton.inp` content from the COMPTON card family.
pub fn compton_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_compton_inp(document, &mut out)?;
    Ok(out)
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

fn write_compton_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let compton = &document.compton;
    let run_compton = i32::from(compton.do_compton || compton.do_rhozzp);

    writeln!(out, "run compton module?")?;
    writeln!(out, "{run_compton:12}")?;
    writeln!(out, "pqmax, npq")?;
    writeln!(out, "{:13.8}{:16}", compton.pqmax, compton.npq)?;
    writeln!(out, "ns, nphi, nz, nzp")?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}",
        compton.ns, compton.nphi, compton.nz, compton.nzp
    )?;
    writeln!(out, "smax, phimax, zmax, zpmax")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        0.0,
        std::f64::consts::TAU,
        0.0,
        compton.zpmax
    )?;
    writeln!(out, "jpq? rhozzp? force_recalc_jzzp?")?;
    writeln!(
        out,
        " {} {} {}",
        fortran_bool(compton.do_compton),
        fortran_bool(compton.do_rhozzp),
        fortran_bool(compton.force_jzzp)
    )?;
    writeln!(out, "window_type (0=Step, 1=Hann), window_cutoff")?;
    writeln!(out, "           1   0.00000000    ")?;
    writeln!(out, "temperature (in eV)")?;
    writeln!(out, "{:13.5}", 0.0)?;
    writeln!(out, "set_chemical_potential? chemical_potential(eV)")?;
    writeln!(out, " F   0.00000000    ")?;
    writeln!(out, "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?")?;
    writeln!(out, " F F F F F")?;
    writeln!(out, "qhat_x qhat_y qhat_z")?;
    writeln!(
        out,
        "   0.0000000000000000        0.0000000000000000        1.0000000000000000     "
    )?;
    Ok(())
}

/// Render FEFF-compatible `hubbard.inp` content.
#[must_use]
pub fn hubbard_inp_string(document: &FeffDocument) -> String {
    let hubbard = document.hubbard;
    let j_field = if hubbard.j == 0.0 {
        format!("{:26.16}", hubbard.j)
    } else {
        format!("{:26.17}", hubbard.j)
    };
    format!(
        "i_hubbard mldos_hubb U_hubbard J_hubbard fermi_shift l_hubbard\n{:12}{:12}{:21.16}{j_field}{:26.16}{:17}\n",
        hubbard.i_hubbard, hubbard.mldos_hubb, hubbard.u, hubbard.fermi_shift, hubbard.l
    )
}

/// Render FEFF-compatible default `opcons.inp` content for current tests.
#[must_use]
pub fn opcons_inp_string(document: &FeffDocument) -> String {
    let nph = nph(document).unwrap_or(1).max(1);
    let mut out = format!(
        "run_opcons\n {}\nprint_eps\n F\nNumDens(0:nphx)\n  ",
        fortran_bool(document.opcons)
    );
    for ipot in 0..=nph {
        if ipot > 0 {
            out.push_str("       ");
        }
        out.push_str("-1.0000000000000000");
    }
    out.push_str("     \n");
    out
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
pub fn dmdw_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_dmdw_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible default `rixs.inp` content.
pub fn rixs_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_rixs_inp(document, &mut out)?;
    Ok(out)
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
pub fn paths_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_paths_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `genfmt.inp` content from an [`FeffDocument`].
pub fn genfmt_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_genfmt_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `ff2x.inp` content from an [`FeffDocument`].
pub fn ff2x_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_ff2x_inp(document, &mut out)?;
    Ok(out)
}

/// Render FEFF-compatible `sfconv.inp` content from an [`FeffDocument`].
pub fn sfconv_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_sfconv_inp(document, &mut out)?;
    Ok(out)
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
    writeln!(out, "natx =  {:>7}", document.atoms.len())?;
    writeln!(out, "    x       y        z       iph  ")?;

    let origin = &document.atoms[0];
    for atom in &document.atoms {
        let distance = distance_from(origin, atom);
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:4}{:13.5}",
            atom.x, atom.y, atom.z, atom.ipot, distance
        )?;
    }

    Ok(())
}

fn write_geom_dat(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let rows = geometry_rows(document)?;
    let iatph = geometry_model_atoms(document, &rows);
    let nph = iatph.len().saturating_sub(1);

    writeln!(out, "nat, nph = {:5}{:5}", rows.len(), nph)?;
    for model_atom in &iatph {
        write!(out, "{model_atom:5}")?;
    }
    writeln!(out)?;
    writeln!(out, " iat     x       y        z       iph  ")?;
    writeln!(out, " {}", "-".repeat(71))?;
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
        )?;
    }
    Ok(())
}

fn write_global_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let ipol = if document.eels.enabled || document.nrixs.is_some() {
        1
    } else {
        document.ipol
    };
    let nrixs = document.nrixs;
    let xivec = if let Some(nrixs) = nrixs {
        normalize_vector(rotate_into_reference_frame(nrixs.qvec, nrixs.qvec))
    } else if document.eels.enabled {
        document.eels.beam_direction
    } else if vector_norm(document.incidence_vector) > 0.0 {
        normalize_vector(document.incidence_vector)
    } else if document.spin != 0 {
        normalize_vector(document.spin_vector)
    } else {
        document.polarization_vector
    };
    let evec = if nrixs.is_some() {
        [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ]
    } else {
        [0.0; 3]
    };
    let evnorm = if let Some(nrixs) = nrixs {
        [0.0, nrixs.qnorm, 0.0]
    } else if document.eels.enabled {
        [0.0, 1.0, 0.0]
    } else if vector_norm(document.incidence_vector) > 0.0 {
        [0.0, vector_norm(document.incidence_vector), 0.0]
    } else if document.spin != 0 {
        [0.0, 0.0, vector_norm(document.spin_vector)]
    } else {
        [0.0; 3]
    };
    let polarization_tensor = if nrixs.is_some() {
        [
            [0.5, -0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        ]
    } else if document.ipol == 2 {
        [
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    } else if document.eels.enabled {
        [[0.0; 6]; 3]
    } else {
        [
            [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0],
        ]
    };

    writeln!(out, " nabs, iphabs - CFAVERAGE data")?;
    writeln!(out, "{:8}{:8}{:13.5}", 1, 0, 100000.0)?;
    writeln!(
        out,
        " ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj"
    )?;
    writeln!(
        out,
        "{:5}{:5}{:5}{:12.4}{:12.4}{:5}{:5}{:5}{:5}",
        ipol,
        document.spin,
        nrixs.map(|nrixs| nrixs.lj).unwrap_or(document.le2),
        nrixs
            .map(|nrixs| if nrixs.qaverage { -nrixs.nq } else { nrixs.nq } as f64)
            .unwrap_or(document.ellipticity),
        0.0,
        nrixs.map(|_| 30).unwrap_or(document.l2lp),
        i32::from(nrixs.is_some()),
        nrixs.map(|nrixs| nrixs.ldecmx).unwrap_or(-1),
        nrixs.map(|nrixs| nrixs.lj).unwrap_or(-1)
    )?;
    writeln!(out, "evec\t\t  xivec \t   spvec")?;
    for idx in 0..3 {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}",
            evec[idx], xivec[idx], document.spin_vector[idx]
        )?;
    }
    writeln!(out, " polarization tensor ")?;
    for row in polarization_tensor {
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
            row[0], row[1], row[2], row[3], row[4], row[5]
        )?;
    }
    writeln!(out, "evnorm, xivnorm, spvnorm - only used for nrixs")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        evnorm[0], evnorm[1], evnorm[2]
    )?;
    writeln!(out, "nq,    imdff,   qaverage,   mixdff")?;
    writeln!(
        out,
        "{:12}{:12} {} F",
        nrixs.map(|nrixs| nrixs.nq).unwrap_or(0),
        0,
        fortran_bool(nrixs.map(|nrixs| nrixs.qaverage).unwrap_or(true))
    )?;
    writeln!(
        out,
        " q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi"
    )?;
    if let Some(nrixs) = nrixs {
        let qtrig = nrixs_qtrig(nrixs.qvec, nrixs.qnorm);
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
            nrixs.qvec[0],
            nrixs.qvec[1],
            nrixs.qvec[2],
            nrixs.qnorm,
            1.0,
            0.0,
            qtrig[0],
            qtrig[1],
            qtrig[2],
            qtrig[3]
        )?;
    }
    Ok(())
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
    let gamach = core_hole_width_for_document(document, central_z, ihole)?;

    writeln!(
        out,
        "mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc"
    )?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}",
        1,
        nph,
        ntitle,
        ihole,
        ipr1,
        0,
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
        0,
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
        gamach, document.rgrid, ca1, ecv, totvol, rfms1, -70.0
    )?;
    writeln!(out, " iz, lmaxsc, xnatph, xion, folp")?;
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
            z, lmaxsc, xnatph, 0.0, document.afolp
        )?;
    }
    writeln!(out, "ExternalPot switch, StartFromFile switch")?;
    writeln!(out, " F F")?;
    writeln!(out, "OVERLAP option: novr(iph)")?;
    for ipot in 0..=nph {
        write!(out, "{:4}", 0)?;
        if ipot == nph {
            writeln!(out)?;
        }
    }
    writeln!(out, " iphovr  nnovr rovr ")?;
    writeln!(out, "ChSh_Type:")?;
    writeln!(out, "{:4}", 0)?;
    writeln!(out, "ConfigType:")?;
    writeln!(out, "{:4}", document.config_type)?;
    writeln!(out, "Temperature (in eV):")?;
    write_pot_temperature(out, document.electronic_temperature)?;
    writeln!(out, "scf_th,  xntol,  nmu")?;
    writeln!(out, "           2   1.0000000000000000E-004         100")?;
    writeln!(out, "negrid,  emaxscf")?;
    writeln!(out, "{:12}{:21.16}     ", 400, 5.0)?;
    writeln!(out, "FiniteNucleus, WarnIon")?;
    writeln!(out, " F {}", fortran_bool(document.warn_ion))?;
    writeln!(out, "ramp_scf  rfms_start  nramp")?;
    writeln!(out, " F   0.00000000               1")?;
    writeln!(out, "tolmu, tolq, tolqp")?;
    writeln!(out, "{:13.5}{:13.5}{:13.5}", 0.001, 0.001, 0.0002)?;
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

fn pad_exponent(value: String) -> String {
    let Some(index) = value.rfind('E') else {
        return value;
    };
    let (mantissa, exponent) = value.split_at(index + 1);
    let (sign, digits) = exponent.split_at(1);
    format!("{mantissa}{sign}{digits:0>3}")
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
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        fms.map(|fms| fms.radius).unwrap_or(-1.0),
        ldos.map(|ldos| ldos.emin).unwrap_or(1000.0),
        ldos.map(|ldos| ldos.emax).unwrap_or(0.0),
        ldos.map(|ldos| ldos.eimag).unwrap_or(-1.0),
        document.rgrid
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

fn write_fms_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let nph = nph(document)?;
    let (tk, thetad, idwopt) = debye_values(document);
    let fms = document.fms.as_ref();

    writeln!(out, "mfms, idwopt, minv")?;
    write_i4_list(
        out,
        [
            control_flag(document, 2, 1),
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
    writeln!(out, "{:13.5}{:13.5}{:13.5}", tk, thetad, 0.0)?;
    writeln!(out, " lmaxph(0:nph)")?;
    write_i4_list(
        out,
        (0..=nph).map(|ipot| potential_for_ipot(document, ipot).map(lmaxph).unwrap_or(0)),
    )?;
    writeln!(out, " the number of decomposi")?;
    writeln!(
        out,
        "{:5}",
        document.nrixs.map(|nrixs| nrixs.ldecmx).unwrap_or(-1)
    )?;
    writeln!(out, " save_gg_slice")?;
    writeln!(out, "{}", if document.ispec == 5 { "T" } else { "F" })?;
    writeln!(out, "do_fms")?;
    write_i4_list(out, [i32::from(document.fms.is_some())])?;
    Ok(())
}

fn write_paths_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let rfms2 = document.fms.as_ref().map(|fms| fms.radius).unwrap_or(-1.0);
    let rmax = path_rmax(document);

    writeln!(out, "mpath, ms, nncrit, nlegxx, ipr4")?;
    write_i4_list(
        out,
        [
            control_flag(document, 3, 1),
            1,
            0,
            document.nleg.unwrap_or(7),
            print_flag(document, 3, 0),
        ],
    )?;
    writeln!(out, "critpw, pcritk, pcrith,  rmax, rfms2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        document.critpw, document.pcritk, document.pcrith, rmax, rfms2
    )?;
    writeln!(out, "ica")?;
    let ica = document
        .nrixs
        .map(|nrixs| if nrixs.qaverage { 5 } else { 7 })
        .unwrap_or(-1);
    write_i4_list(out, [ica])?;
    Ok(())
}

fn write_genfmt_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "mfeff, ipr5, iorder, critcw, wnstar")?;
    writeln!(
        out,
        "{:4}{:4}{:8}{:13.5}{:>5}",
        control_flag(document, 4, 1),
        print_flag(document, 4, 0),
        2,
        document.critcw,
        "F"
    )?;
    writeln!(out, " the number of decomposi")?;
    writeln!(
        out,
        "{:5}",
        document.nrixs.map(|nrixs| nrixs.ldecmx).unwrap_or(-1)
    )?;
    Ok(())
}

fn write_ff2x_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let (tk, thetad, idwopt) = debye_values(document);
    let absolu = i32::from(document.absolute || document.eels.enabled);
    let momentum_transfer = if document.eels.enabled {
        document.eels.beam_direction
    } else if let Some(nrixs) = document.nrixs {
        nrixs.qvec
    } else if vector_norm(document.incidence_vector) > 0.0 {
        normalize_vector(document.incidence_vector)
    } else {
        [0.0; 3]
    };

    writeln!(out, "mchi, ispec, idwopt, ipr6, mbconv, absolu, iGammaCH")?;
    write_i4_list(
        out,
        [
            control_flag(document, 5, 1),
            output_ispec(document),
            idwopt,
            print_flag(document, 5, 0),
            0,
            absolu,
            0,
        ],
    )?;
    writeln!(out, "vrcorr, vicorr, s02, critcw")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        document.corrections[0],
        document.corrections[1],
        document.s02.unwrap_or(1.0),
        document.critcw
    )?;
    writeln!(out, "tk, thetad, alphat, thetae, sig2g, sig_gk")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        tk, thetad, 0.0, 0.0, 0.0, 0.0
    )?;
    writeln!(out, "momentum transfer")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        momentum_transfer[0], momentum_transfer[1], momentum_transfer[2]
    )?;
    writeln!(out, " the number of decomposi")?;
    writeln!(
        out,
        "{:5}",
        document.nrixs.map(|nrixs| nrixs.ldecmx).unwrap_or(-1)
    )?;
    writeln!(out, "electronic temperature")?;
    writeln!(out, "{:13.5}", document.electronic_temperature)?;
    Ok(())
}

fn write_sfconv_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "msfconv, ipse, ipsk")?;
    write_i4_list(out, [i32::from(document.sfconv), 0, 0])?;
    writeln!(out, "wsigk, cen")?;
    writeln!(out, "{:13.5}{:13.5}", 0.0, 0.0)?;
    writeln!(out, "ispec, ipr6")?;
    write_i4_list(out, [output_ispec(document), print_flag(document, 5, 0)])?;
    writeln!(out, "cfname")?;
    writeln!(out, "NULL        ")?;
    Ok(())
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
            1,
            ipr2,
            ixc,
            spectrum_grid.ixc0,
            output_ispec(document),
            document.lreal,
            document.fms.as_ref().map(|fms| fms.lfms).unwrap_or(0),
            nph,
            document.nrixs.map(|_| 30).unwrap_or(0),
            document.i_plsmn,
            document.n_poles,
            0,
            document.i_grid,
            -1,
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
        core_hole_width_for_document(document, central_z, ihole)?,
        spectrum_grid.xkstep,
        spectrum_grid.xkmax,
        spectrum_grid.vixan,
        0.0,
        0.0
    )?;
    writeln!(out, "spinph(0:nph)")?;
    for ipot in 0..=nph {
        write!(out, "{:13.5}", spinph(document, ipot))?;
    }
    writeln!(out)?;
    writeln!(out, "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis")?;
    write_i4_list(out, [0, 0, 0, 0, 0, 0])?;
    writeln!(out, "electronic temperature")?;
    writeln!(out, "{:13.5}", document.electronic_temperature)?;
    writeln!(out, "ChSh_Type:")?;
    writeln!(out, "{:4}", 0)?;
    writeln!(
        out,
        " the number of decomposition channels ; only used for nrixs"
    )?;
    writeln!(
        out,
        "{:5}",
        document.nrixs.map(|nrixs| nrixs.ldecmx).unwrap_or(-1)
    )?;
    writeln!(out, "lopt")?;
    writeln!(out, " F")?;
    writeln!(out, "PrintRL")?;
    writeln!(out, " F")?;
    Ok(())
}

#[derive(Debug, Clone)]
struct GeometryRow {
    x: f64,
    y: f64,
    z: f64,
    ipot: i32,
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
        distance2: 0.0,
    });

    for (input_index, atom) in document.atoms.iter().enumerate() {
        if input_index == absorber_index {
            continue;
        }
        let x = atom.x - absorber.x;
        let y = atom.y - absorber.y;
        let z = atom.z - absorber.z;
        let distance2 = feff_geometry_distance2(x, y, z);
        if distance2 > 0.0 {
            rows.push(GeometryRow {
                x,
                y,
                z,
                ipot: atom.ipot,
                distance2,
            });
        }
    }

    feff_selection_sort_by_distance(&mut rows);
    if let Some(nrixs) = document.nrixs {
        for row in &mut rows {
            let [x, y, z] = rotate_into_reference_frame([row.x, row.y, row.z], nrixs.qvec);
            row.x = x;
            row.y = y;
            row.z = z;
        }
    }
    Ok(rows)
}

fn rotate_into_reference_frame(vector: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let Some(rotation) = reference_frame_rotation(axis) else {
        return vector;
    };
    rotation.rotate(vector)
}

fn reference_frame_rotation(axis: [f64; 3]) -> Option<ReferenceFrameRotation> {
    let norm = axis[0].hypot(axis[1]).hypot(axis[2]);
    if norm == 0.0 {
        return None;
    }

    let xy_norm2 = axis[0] * axis[0] + axis[1] * axis[1];
    if xy_norm2 == 0.0 && axis[2] >= 0.0 {
        return None;
    }

    let (cst, snt, csf, snf) = if xy_norm2 == 0.0 {
        (-1.0, 0.0, 1.0, 0.0)
    } else {
        let xy_norm = xy_norm2.sqrt();
        (
            axis[2] / norm,
            xy_norm / norm,
            axis[0] / xy_norm,
            axis[1] / xy_norm,
        )
    };

    Some(ReferenceFrameRotation { cst, snt, csf, snf })
}

#[derive(Debug, Clone, Copy)]
struct ReferenceFrameRotation {
    cst: f64,
    snt: f64,
    csf: f64,
    snf: f64,
}

impl ReferenceFrameRotation {
    fn rotate(self, vector: [f64; 3]) -> [f64; 3] {
        [
            vector[0] * self.cst * self.csf + vector[1] * self.cst * self.snf
                - vector[2] * self.snt,
            -vector[0] * self.snf + vector[1] * self.csf,
            vector[0] * self.csf * self.snt
                + vector[1] * self.snt * self.snf
                + vector[2] * self.cst,
        ]
    }
}

fn nrixs_qtrig(qvec: [f64; 3], qnorm: f64) -> [f64; 4] {
    let xy_norm2 = qvec[0] * qvec[0] + qvec[1] * qvec[1];
    if qnorm <= 0.0 {
        return [1.0, 0.0, 1.0, 0.0];
    }
    if xy_norm2 == 0.0 {
        return [-1.0, 0.0, 1.0, 0.0];
    }
    if qvec[2] < 0.0 {
        let xy_norm = xy_norm2.sqrt();
        return [
            qvec[2] / qnorm,
            xy_norm / qnorm,
            qvec[0] / xy_norm,
            qvec[1] / xy_norm,
        ];
    }
    [1.0, 0.0, 1.0, 0.0]
}

fn feff_selection_sort_by_distance(rows: &mut [GeometryRow]) {
    // FEFF's ffsort uses a strict less-than selection sort. That matters for
    // equal-distance shells because swaps can move tied atoms differently than
    // a stable distance sort.
    let mut i = 0;
    while i + 1 < rows.len() {
        let mut min_idx = i;
        let mut min_distance = rows[i].distance2;
        let mut j = i + 1;
        while j < rows.len() {
            if rows[j].distance2 < min_distance {
                min_distance = rows[j].distance2;
                min_idx = j;
            }
            j += 1;
        }
        if min_idx != i {
            rows.swap(i, min_idx);
        }
        i += 1;
    }
}

fn feff_geometry_distance2(x: f64, y: f64, z: f64) -> f64 {
    // The reference FEFF10 binaries are optimized, so equal-shell ordering
    // follows the fused arithmetic generated for the ffsort distance sum.
    z.mul_add(z, x.mul_add(x, y * y))
}

fn normalize_vector(vector: [f64; 3]) -> [f64; 3] {
    let norm = vector_norm(vector);
    if norm == 0.0 {
        [0.0; 3]
    } else {
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    }
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
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

fn write_i4_list(
    out: &mut impl std::fmt::Write,
    values: impl IntoIterator<Item = i32>,
) -> Result<()> {
    for value in values {
        write!(out, "{value:4}")?;
    }
    writeln!(out)?;
    Ok(())
}

fn fortran_bool(value: bool) -> &'static str {
    if value { "T" } else { "F" }
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

fn output_ispec(document: &FeffDocument) -> i32 {
    if document.ipol == 2 && document.spin != 0 {
        -1
    } else {
        document.ispec
    }
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
    let nspu = if document.hubbard.i_hubbard >= 2
        || document.spin != 0
        || document
            .potentials
            .iter()
            .any(|potential| potential.spinph.unwrap_or(0.0) != 0.0)
    {
        2
    } else {
        1
    };
    Ok((nclusx, lx, nphu, nspu))
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

fn document_ihole(document: &FeffDocument) -> Result<i32> {
    if let Some(hole) = document.hole {
        return Ok(hole);
    }
    document
        .edge
        .as_ref()
        .map(|edge| edge_hole(&edge.label))
        .transpose()
        .map(|hole| hole.unwrap_or(1))
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
        Some(z) if z < 3 => 1,
        Some(_) => 2,
        None => 0,
    };
    let cap = match potential.z {
        Some(z) if z >= 57 => 3,
        _ => default,
    };
    let requested = potential
        .lmax1
        .filter(|requested| *requested >= 0)
        .unwrap_or(default);
    requested.min(cap)
}

fn lmaxph(potential: &Potential) -> i32 {
    let default = match potential.z {
        Some(z) if z < 6 => 2,
        Some(_) => 3,
        None => 0,
    };
    potential
        .lmax2
        .filter(|requested| *requested >= 0)
        .unwrap_or(default)
}

fn xnatph(document: &FeffDocument, potential: &Potential) -> f64 {
    if document
        .potentials
        .iter()
        .any(|potential| potential.xnatph.is_some())
    {
        if potential.ipot == 0 {
            return 1.0;
        }

        let absorber_count = document
            .potentials
            .iter()
            .find(|potential| potential.ipot == 0)
            .and_then(|potential| potential.xnatph)
            .filter(|value| *value > 0.01)
            .unwrap_or(0.01);
        let raw = potential.xnatph.unwrap_or(0.0);
        let adjusted = if raw <= 0.0 { 1.0 } else { raw };
        return adjusted / absorber_count;
    }

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

fn interstitial_volume(document: &FeffDocument) -> f64 {
    let Some(interstitial) = document.interstitial else {
        return 0.0;
    };
    let Some(ratmin) = nearest_nonabsorber_distance(document) else {
        return 0.0;
    };
    let xnat = document
        .potentials
        .iter()
        .map(|potential| xnatph(document, potential))
        .sum::<f64>();
    interstitial.volume_scale * ratmin.powi(3) * xnat
}

fn path_rmax(document: &FeffDocument) -> f64 {
    if let Some(rpath) = document.rpath {
        return rpath;
    }
    if document.ispec > 0 {
        return -1.0;
    }
    let Some(ratmin) = nearest_nonabsorber_distance(document) else {
        return -1.0;
    };
    let Some(absorber) = document.atoms.first() else {
        return -1.0;
    };
    let ratmax = document
        .atoms
        .iter()
        .map(|atom| distance_from(absorber, atom))
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    (2.2 * ratmin).min(1.01 * ratmax)
}

fn spinph(document: &FeffDocument, ipot: i32) -> f64 {
    if document.hubbard.i_hubbard >= 2 {
        return 0.0;
    }
    potential_for_ipot(document, ipot)
        .ok()
        .and_then(|potential| {
            potential.spinph.or_else(|| {
                (document.spin != 0)
                    .then(|| potential.z.and_then(default_atomic_spinph))
                    .flatten()
            })
        })
        .unwrap_or(0.0)
}

fn default_atomic_spinph(z: i32) -> Option<f64> {
    match z {
        64 => Some(7.0),
        _ => None,
    }
}

fn nearest_nonabsorber_distance(document: &FeffDocument) -> Option<f64> {
    let origin = document.atoms.first()?;
    document
        .atoms
        .iter()
        .skip(1)
        .map(|atom| distance_from(origin, atom))
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

fn core_hole_width_for_document(document: &FeffDocument, z: i32, ihole: i32) -> Result<f64> {
    core_hole_width_ev(z, ihole).map_err(|err| IoError::Parse {
        path: document.source.clone(),
        line: 0,
        message: err.to_string(),
    })
}

fn distance_from(origin: &Atom, atom: &Atom) -> f64 {
    let dx = atom.x - origin.x;
    let dy = atom.y - origin.y;
    let dz = atom.z - origin.z;
    dx.hypot(dy).hypot(dz)
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, Result};

    use super::{
        atoms_dat_string, compton_inp_string, dimensions_dat_string, dmdw_inp_string,
        geom_dat_string, global_inp_string, pot_inp_string, rixs_inp_string, xsph_inp_string,
    };

    #[test]
    fn writes_atoms_dat_with_feff_widths() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 2.0 2.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let atoms = atoms_dat_string(&doc)?;

        assert_eq!(
            atoms,
            "natx =        2\n    x       y        z       iph  \n      0.00000      0.00000      0.00000   0      0.00000\n      1.00000      2.00000      2.00000   1      3.00000\n"
        );
        Ok(())
    }

    #[test]
    fn writes_global_inp_defaults() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let global = global_inp_string(&doc)?;

        assert!(global.contains(" nabs, iphabs - CFAVERAGE data\n       1       0 100000.00000\n"));
        assert!(global.contains(" polarization tensor \n      0.33333"));
        Ok(())
    }

    #[test]
    fn writes_dimensions_dat_from_cluster_radius() -> Result<()> {
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
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(
            dimensions_dat_string(&doc)?,
            "           2           3           1           1\n"
        );
        Ok(())
    }

    #[test]
    fn writes_geom_dat_with_sorted_relative_cluster() -> Result<()> {
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
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(
            geom_dat_string(&doc)?,
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
        Ok(())
    }

    #[test]
    fn writes_compton_inp_from_cards() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
COMPTON
RHOZZP
CGRID 10 32 32 32 120
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(
            compton_inp_string(&doc)?,
            concat!(
                "run compton module?\n",
                "           1\n",
                "pqmax, npq\n",
                "   5.00000000            1000\n",
                "ns, nphi, nz, nzp\n",
                "  32  32  32 120\n",
                "smax, phimax, zmax, zpmax\n",
                "      0.00000      6.28319      0.00000     10.00000\n",
                "jpq? rhozzp? force_recalc_jzzp?\n",
                " T T F\n",
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
        );
        Ok(())
    }

    #[test]
    fn writes_pot_inp_for_copper_defaults() -> Result<()> {
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
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.contains("   1   1   1   1   0   0   0   0  11\n"));
        assert!(pot.contains("      1.72919      0.05000      0.00000    -40.00000"));
        assert!(pot.contains("   29    2        1.0000000000"));
        Ok(())
    }

    #[test]
    fn writes_xsph_inp_for_copper_exafs() -> Result<()> {
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
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let xsph = xsph_inp_string(&doc)?;

        assert!(xsph.contains("   1   0   0   0   0   0   0   1   0   0 100   0   0  -1  11\n"));
        assert!(xsph.contains("Cu    Cu    \n"));
        assert!(xsph.contains("      0.05000     -1.00000      1.72919      0.07000     20.00000"));
        Ok(())
    }

    #[test]
    fn writes_dmdw_inp_for_dynamical_matrix_debye() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DEBYE 450 315 5 feff.dym 6 0 1
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(
            dmdw_inp_string(&doc)?,
            "   1\n   6\n   1    450.000\n   0\nfeff.dym\n   1\n   2   1   0           2.08\n"
        );
        Ok(())
    }

    #[test]
    fn writes_rixs_inp_defaults() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EDGE K
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(
            rixs_inp_string(&doc)?,
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
        Ok(())
    }
}
