//! Render a [`FeffDocument`] back into canonical `feff.inp` text.
//!
//! This is the inverse of [`FeffDocument::from_input`]: for each field the
//! typed model currently understands it emits the FEFF card that produces
//! that value, and otherwise stays silent so re-parsing the rendered input
//! reproduces an equivalent value (RDINP's own default for an absent card).
//!
//! A few families of fields are intentionally left unwritten because they
//! are module handoffs assembled from several cooperating cards or from
//! auxiliary files rather than a single round-trippable card:
//! `reciprocal`/`cif_equivalence`/`coordinate_mode`/`reciprocal_input`
//! (LATTICE/CIF/RECIPROCAL/KMESH/TARGET/EQUIVALENCE/COORDINATES),
//! `band_input` (BAND), `full_spectrum_input` (FULLSPECTRUM), `screen_input`
//! (SCREEN), `eels`/`nrixs`/`mdff` (ELNES/EXELFS/NRIXS/MDFF), and `rixs`
//! except for `rixs.edges` (reconstructed from the extra tokens on the
//! `EDGE` card, but the `RIXS` card itself is not emitted),
//! `opcons_input` (OPCONS/NUMDENS/PREPS), `sfconv_input`
//! (SFCONV/SELF/SFSE/RCONV), `config_records`/`egrid_records`/
//! `density_records` (raw payload blocks copied verbatim by RDINP), and
//! `spring_input_text`/`dym_input`/`DEBYE` modes that require an auxiliary
//! `spring.inp` or `.dym` file (`idwopt` 1, 2, or 5). `r_multiplier` is also
//! left unwritten: [`FeffDocument::from_input`] already applies it to
//! coordinates and radii, so re-emitting `RMULT` would double-scale them.
//!
//! Callers that need one of these should keep templating the corresponding
//! card by hand; every other field documented on [`FeffDocument`] round-trips
//! through [`feff_inp_string`].

use std::fmt::Write as _;

use super::*;

/// Render `document` into canonical FEFF input text.
///
/// The output is not guaranteed to be byte-identical to whatever text
/// originally produced `document`, but re-parsing it with
/// [`crate::input::FeffInput::parse_str`] and [`FeffDocument::from_input`]
/// yields an equivalent document for every field this module writes.
pub fn feff_inp_string(document: &FeffDocument) -> Result<String> {
    let mut out = String::new();
    write_titles(&mut out, document)?;
    write_edge_hole_s02(&mut out, document)?;
    write_corrections_and_chshift(&mut out, document)?;
    write_control_print(&mut out, document)?;
    write_exchange(&mut out, document)?;
    write_spectroscopy(&mut out, document)?;
    write_scf(&mut out, document)?;
    write_fms(&mut out, document)?;
    write_path_controls(&mut out, document)?;
    write_grid_and_criteria(&mut out, document)?;
    write_hole_treatment(&mut out, document)?;
    write_polarization(&mut out, document)?;
    write_damping(&mut out, document)?;
    write_flags(&mut out, document)?;
    write_xsph_handoff(&mut out, document)?;
    write_xsph_advanced(&mut out, document)?;
    write_thermal(&mut out, document)?;
    write_debye(&mut out, document)?;
    write_mpse(&mut out, document)?;
    write_crpa(&mut out, document)?;
    write_hubbard(&mut out, document)?;
    write_compton(&mut out, document)?;
    write_folp_ion(&mut out, document)?;
    write_overlap_and_ss(&mut out, document)?;
    write_potentials(&mut out, document)?;
    write_atoms(&mut out, document)?;
    writeln!(out, "END")?;
    Ok(out)
}

/// Render `document` and write it to `path`.
pub fn write_feff_inp(path: impl AsRef<Path>, document: &FeffDocument) -> Result<()> {
    let text = feff_inp_string(document)?;
    std::fs::write(path.as_ref(), text).map_err(|source| IoError::io(path.as_ref(), source))
}

fn write_titles(out: &mut String, document: &FeffDocument) -> Result<()> {
    for title in &document.titles {
        writeln!(out, "TITLE {title}")?;
    }
    Ok(())
}

fn write_edge_hole_s02(out: &mut String, document: &FeffDocument) -> Result<()> {
    if let Some(edge) = &document.edge {
        // Extra `EDGE` tokens beyond the primary edge label select additional
        // RIXS edges (or `VAL` for many-body convolution); reproduce them so
        // `rixs.edges` round-trips too.
        write!(out, "EDGE {}", edge.label)?;
        for extra_edge in document.rixs.edges.iter().skip(1) {
            write!(out, " {extra_edge}")?;
        }
        writeln!(out)?;
    } else if let Some(hole) = document.hole {
        writeln!(out, "HOLE {hole}")?;
    }
    if let Some(s02) = document.s02 {
        writeln!(out, "S02 {s02}")?;
    }
    Ok(())
}

fn write_corrections_and_chshift(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.corrections != [0.0, 0.0] {
        writeln!(
            out,
            "CORRECTIONS {} {}",
            document.corrections[0], document.corrections[1]
        )?;
    }
    if document.chsh_type != 0 {
        writeln!(out, "CHSHIFT {}", document.chsh_type)?;
    }
    Ok(())
}

fn write_control_print(out: &mut String, document: &FeffDocument) -> Result<()> {
    if let Some(control) = document.control {
        writeln!(out, "CONTROL {}", join_i32(&control))?;
    }
    if let Some(print) = document.print {
        writeln!(out, "PRINT {}", join_i32(&print))?;
    }
    Ok(())
}

fn write_exchange(out: &mut String, document: &FeffDocument) -> Result<()> {
    let Some(exchange) = &document.exchange else {
        return Ok(());
    };
    write!(
        out,
        "EXCHANGE {} {} {}",
        exchange.ixc, exchange.vr0, exchange.vi0
    )?;
    if let Some(ixc0) = exchange.ixc0 {
        write!(out, " {ixc0}")?;
    }
    writeln!(out)?;
    Ok(())
}

/// Emit the single spectroscopy card this writer supports reconstructing:
/// `EXAFS` from [`FeffDocument::exafs`], or `XANES` from
/// [`FeffDocument::spectrum_grid`] when `ispec == 1` and neither EELS nor
/// NRIXS produced it. `XES`/`DANES`/`FPRIME`/`ELNES`/`NRIXS` spectroscopy
/// selectors are not reconstructed here.
fn write_spectroscopy(out: &mut String, document: &FeffDocument) -> Result<()> {
    if let Some(exafs) = &document.exafs {
        writeln!(out, "EXAFS {}", exafs.xkmax)?;
    } else if document.ispec == 1 && !document.eels.enabled && document.nrixs.is_none() {
        let grid = document.spectrum_grid;
        writeln!(out, "XANES {} {} {}", grid.xkmax, grid.xkstep, grid.vixan)?;
    }
    Ok(())
}

fn write_scf(out: &mut String, document: &FeffDocument) -> Result<()> {
    let Some(scf) = &document.scf else {
        return Ok(());
    };
    writeln!(
        out,
        "SCF {} {} {} {} {} {} {}",
        scf.radius, scf.lfms, scf.iterations, scf.ca, scf.nmix, scf.ecv, scf.icoul
    )?;
    Ok(())
}

fn write_fms(out: &mut String, document: &FeffDocument) -> Result<()> {
    let Some(fms) = &document.fms else {
        return Ok(());
    };
    writeln!(
        out,
        "FMS {} {} {} {} {} {}",
        fms.radius, fms.lfms, fms.minv, fms.toler1, fms.toler2, fms.rdirec
    )?;
    Ok(())
}

fn write_path_controls(out: &mut String, document: &FeffDocument) -> Result<()> {
    if let Some(rpath) = document.rpath {
        writeln!(out, "RPATH {rpath}")?;
    }
    match document.nleg {
        Some(7) => writeln!(out, "NLEG")?,
        Some(nleg) => writeln!(out, "NLEG {nleg}")?,
        None => {}
    }
    if (1..=7).contains(&document.path_symmetry) {
        writeln!(out, "SYMMETRY {}", document.path_symmetry)?;
    }
    if document.no_geom {
        writeln!(out, "NOGEOM")?;
    }
    Ok(())
}

fn write_grid_and_criteria(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.rgrid != 0.05 {
        writeln!(out, "RGRID {}", document.rgrid)?;
    }
    if (document.critcw, document.critpw) != (4.0, 2.5) {
        writeln!(out, "CRITERIA {} {}", document.critcw, document.critpw)?;
    }
    if (document.pcritk, document.pcrith) != (0.0, 0.0) {
        writeln!(out, "PCRITERIA {} {}", document.pcritk, document.pcrith)?;
    }
    match document.lreal {
        2 => writeln!(out, "RPHASES")?,
        1 => writeln!(out, "RSIGMA")?,
        _ => {}
    }
    if document.iorder != 2 {
        writeln!(out, "IORD {}", document.iorder)?;
    }
    Ok(())
}

fn write_hole_treatment(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.nohole != -1 {
        writeln!(out, "NOHOLE {}", document.nohole)?;
    }
    if document.cfaverage != default_cfaverage() {
        writeln!(
            out,
            "CFAVERAGE {} {} {}",
            document.cfaverage.iphabs, document.cfaverage.nabs, document.cfaverage.rclabs
        )?;
    }
    if document.corval_emin != -70.0 {
        writeln!(out, "CORVAL {}", document.corval_emin)?;
    }
    Ok(())
}

fn write_polarization(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.le2 != 0 || document.l2lp != 0 {
        writeln!(out, "MULT {} {}", document.le2, document.l2lp)?;
    }
    if document.polarization_vector != [0.0, 0.0, 0.0] {
        let [x, y, z] = document.polarization_vector;
        writeln!(out, "POLARIZATION {x} {y} {z}")?;
    }
    if document.ellipticity != 0.0 || document.incidence_vector != [0.0, 0.0, 0.0] {
        let [x, y, z] = document.incidence_vector;
        writeln!(out, "ELLIPTICITY {} {x} {y} {z}", document.ellipticity)?;
    }
    if document.spin != 0 || document.spin_vector != [0.0, 0.0, 0.0] {
        let [x, y, z] = document.spin_vector;
        writeln!(out, "SPIN {} {x} {y} {z}", document.spin)?;
    }
    if document.nstar {
        writeln!(out, "NSTAR")?;
    }
    Ok(())
}

fn write_damping(out: &mut String, document: &FeffDocument) -> Result<()> {
    let damping = document.fine_structure_damping;
    if damping.alphat != 0.0 || damping.thetae != 0.0 {
        writeln!(out, "SIG3 {} {}", damping.alphat, damping.thetae)?;
    }
    if damping.sig2g != 0.0 {
        writeln!(out, "SIG2 {}", damping.sig2g)?;
    }
    if damping.sig_gk != 0.0 {
        writeln!(out, "SIGGK {}", damping.sig_gk)?;
    }
    Ok(())
}

fn write_flags(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.jump_removal {
        writeln!(out, "JUMPRM")?;
    }
    if document.many_body_convolution {
        writeln!(out, "MBCONV")?;
    }
    if document.absolute {
        writeln!(out, "ABSOLUTE")?;
    }
    if document.warn_ion {
        writeln!(out, "WARN")?;
    }
    if document.finite_nucleus {
        writeln!(out, "HIGHZ")?;
    }
    if document.unfreezef {
        writeln!(out, "UNFREEZEF")?;
    }
    if document.external_pot {
        writeln!(out, "EXTPOT")?;
    }
    if document.restart_from_pot_bin {
        writeln!(out, "RESTART")?;
    }
    if let Some(dims) = document.dims {
        writeln!(out, "DIMS {} {}", dims.nclusx, dims.lx)?;
    }
    if let Some(ldos) = &document.ldos {
        writeln!(
            out,
            "LDOS {} {} {} {} {}",
            ldos.emin, ldos.emax, ldos.eimag, ldos.neldos, ldos.ldostype
        )?;
    }
    if let Some(interstitial) = document.interstitial {
        writeln!(
            out,
            "INTERSTITIAL {} {}",
            interstitial.mode, interstitial.volume_scale
        )?;
    }
    if document.afolp != 1.15 {
        writeln!(out, "AFOLP {}", document.afolp)?;
    }
    Ok(())
}

fn write_xsph_handoff(out: &mut String, document: &FeffDocument) -> Result<()> {
    let controls = document.xsph_handoff;
    if controls.core_hole_broadening != 0 {
        writeln!(out, "CHBROADENING {}", controls.core_hole_broadening)?;
    }
    if controls.core_state != -1 {
        writeln!(out, "ICOR {}", controls.core_state)?;
    }
    if controls.eps0 != 0.0 {
        writeln!(out, "EPS0 {}", controls.eps0)?;
    }
    if controls.egap != 0.0 {
        writeln!(out, "EGAP {}", controls.egap)?;
    }
    if let Some(width) = controls.core_hole_width {
        writeln!(out, "CHWIDTH {width}")?;
    }
    if controls.set_edge {
        writeln!(out, "SETEDGE")?;
    }
    if controls.print_radial_wavefunctions {
        writeln!(out, "RLPRINT")?;
    }
    Ok(())
}

fn write_xsph_advanced(out: &mut String, document: &FeffDocument) -> Result<()> {
    let advanced = &document.xsph_advanced;
    if advanced.izstd == 1 {
        writeln!(out, "TDLDA {}", advanced.ifxc)?;
    }
    if advanced.itdlda == 2 {
        writeln!(
            out,
            "PMBSE {} {} {} {}",
            advanced.ipmbse, advanced.nonlocal, advanced.ifxc, advanced.ibasis
        )?;
    }
    Ok(())
}

fn write_thermal(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.electronic_temperature != 0.0 || document.iscfxc != 11 {
        writeln!(
            out,
            "TEMP {} {}",
            document.electronic_temperature, document.iscfxc
        )?;
    }
    // `parse_scf_thermal`'s default (returned when `SCFTH` is absent) is not
    // `ScfThermal::default()`-equivalent, so compare against it explicitly.
    let scf_thermal_default = ScfThermal {
        iscfth: 2,
        xntol: 1.0e-4,
        nmu: 100,
        negrid: 400,
        emaxscf: 5.0,
    };
    if document.scf_thermal != scf_thermal_default {
        let thermal = document.scf_thermal;
        writeln!(
            out,
            "SCFTH {} {} {} {} {}",
            thermal.iscfth, thermal.emaxscf, thermal.negrid, thermal.nmu, thermal.xntol
        )?;
    }
    if document.scf_ramp.enabled {
        writeln!(
            out,
            "SCFR {} {}",
            document.scf_ramp.rfms_start, document.scf_ramp.nramp
        )?;
    }
    let tolerances = document.scf_tolerances;
    if tolerances.tolmu == 0.001 && (tolerances.tolq, tolerances.tolqp) != (0.001, 0.0002) {
        writeln!(out, "TOLS 1 {} {}", tolerances.tolq, tolerances.tolqp)?;
    }
    Ok(())
}

fn write_debye(out: &mut String, document: &FeffDocument) -> Result<()> {
    let Some(debye) = &document.debye else {
        return Ok(());
    };
    // Modes 1, 2, and 5 require an auxiliary `spring.inp`/`.dym` file that
    // this text-only writer does not emit alongside `feff.inp`.
    if matches!(debye.idwopt, 1 | 2 | 5) {
        return Ok(());
    }
    if debye.dmdw_order != 2 || debye.dmdw_type != 0 || debye.dmdw_route != 0 {
        return Ok(());
    }
    writeln!(
        out,
        "DEBYE {} {} {}",
        debye.temperature, debye.debye_temperature, debye.idwopt
    )?;
    Ok(())
}

fn write_mpse(out: &mut String, document: &FeffDocument) -> Result<()> {
    if (document.i_plsmn, document.n_poles) != (0, 100) {
        writeln!(out, "MPSE {} {}", document.i_plsmn, document.n_poles)?;
    }
    Ok(())
}

fn write_crpa(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.crpa.enabled {
        writeln!(out, "CRPA {} {}", document.crpa.l, document.crpa.rcut)?;
    }
    Ok(())
}

fn write_hubbard(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.hubbard != Hubbard::default() {
        let hubbard = document.hubbard;
        writeln!(
            out,
            "HUBBARD {} {} {} {}",
            hubbard.u, hubbard.j, hubbard.fermi_shift, hubbard.l
        )?;
    }
    Ok(())
}

fn write_compton(out: &mut String, document: &FeffDocument) -> Result<()> {
    let compton = &document.compton;
    if compton.do_compton {
        write!(out, "COMPTON {} {}", compton.pqmax, compton.npq)?;
        if compton.force_jzzp {
            write!(out, " 1")?;
        }
        writeln!(out)?;
    }
    if compton.do_rhozzp {
        writeln!(out, "RHOZZP")?;
    }
    let grid_is_default = compton.zpmax == 10.0
        && compton.ns == 32
        && compton.nphi == 32
        && compton.nz == 32
        && compton.nzp == 144;
    if (compton.do_compton || compton.do_rhozzp) && !grid_is_default {
        writeln!(
            out,
            "CGRID {} {} {} {} {}",
            compton.zpmax, compton.ns, compton.nphi, compton.nz, compton.nzp
        )?;
    }
    Ok(())
}

fn write_folp_ion(out: &mut String, document: &FeffDocument) -> Result<()> {
    for factor in &document.overlap_factors {
        writeln!(out, "FOLP {} {}", factor.potential_index, factor.factor)?;
    }
    for ionization in &document.ionizations {
        writeln!(
            out,
            "ION {} {}",
            ionization.potential_index, ionization.value
        )?;
    }
    Ok(())
}

fn write_overlap_and_ss(out: &mut String, document: &FeffDocument) -> Result<()> {
    let mut seen_potentials: Vec<i32> = Vec::new();
    for shell in &document.overlap_shells {
        if !seen_potentials.contains(&shell.potential_index) {
            seen_potentials.push(shell.potential_index);
            writeln!(out, "OVERLAP {}", shell.potential_index)?;
        }
        writeln!(
            out,
            "{} {} {}",
            shell.neighbor_potential_index, shell.count, shell.distance
        )?;
    }
    for path in &document.single_scattering_paths {
        writeln!(
            out,
            "SS {} {} {} {}",
            path.index, path.potential_index, path.degeneracy, path.distance
        )?;
    }
    Ok(())
}

fn write_potentials(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.potentials.is_empty() {
        return Ok(());
    }
    writeln!(out, "POTENTIALS")?;
    for potential in &document.potentials {
        write!(out, "{} {}", potential.ipot, potential.z_token)?;
        // `tag`, `lmax1`, `lmax2`, `xnatph`, and `spinph` are positional
        // trailing fields on the FEFF `POTENTIALS` row: writing a later
        // field without an earlier one would shift every field after it, so
        // stop at the first absent field in the chain.
        let trailing: [Option<String>; 5] = [
            potential.tag.clone(),
            potential.lmax1.map(|value| value.to_string()),
            potential.lmax2.map(|value| value.to_string()),
            potential.xnatph.map(|value| value.to_string()),
            potential.spinph.map(|value| value.to_string()),
        ];
        for field in trailing {
            let Some(field) = field else { break };
            write!(out, " {field}")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

fn write_atoms(out: &mut String, document: &FeffDocument) -> Result<()> {
    if document.atoms.is_empty() {
        return Ok(());
    }
    writeln!(out, "ATOMS")?;
    for atom in &document.atoms {
        write!(out, "{} {} {} {}", atom.x, atom.y, atom.z, atom.ipot)?;
        if let Some(tag) = &atom.tag {
            write!(out, " {tag}")?;
            if let Some(distance) = atom.distance {
                write!(out, " {distance}")?;
                if let Some(index) = atom.index {
                    write!(out, " {index}")?;
                }
            }
        }
        writeln!(out)?;
    }
    Ok(())
}

fn join_i32(values: &[i32; 6]) -> String {
    values
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}
