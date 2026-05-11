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
    outputs.insert("atoms.dat", atoms_dat_string(document)?);
    outputs.insert("global.inp", global_inp_string(document));
    outputs.insert("pot.inp", pot_inp_string(document)?);
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

/// Render FEFF-compatible `global.inp` content from an [`FeffDocument`].
pub fn global_inp_string(_document: &FeffDocument) -> String {
    let mut out = String::new();
    write_global_inp(&mut out);
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

fn write_i4_list(out: &mut impl std::fmt::Write, values: impl IntoIterator<Item = i32>) {
    for value in values {
        write!(out, "{value:4}").expect("write to string");
    }
    writeln!(out).expect("write to string");
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

    use super::{atoms_dat_string, global_inp_string, pot_inp_string, xsph_inp_string};

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
}
