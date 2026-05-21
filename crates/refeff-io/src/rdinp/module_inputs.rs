use super::*;

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

/// Render FEFF-compatible `paths.dat` content for explicit `SS` cards.
///
/// FEFF handles `SS` cards inside `rdinp` instead of running the pathfinder:
/// each card becomes a two-leg path with the requested scatterer followed by
/// the absorber at the origin.
pub fn single_scattering_paths_dat_string(document: &FeffDocument) -> Result<String> {
    if document.single_scattering_paths.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write paths.dat without SS cards".to_string(),
        });
    }

    let mut out = String::new();
    write_single_scattering_paths_dat(document, &mut out)?;
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
    sfconv_input_string(&document.sfconv_input)
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

fn write_paths_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let rfms2 = document.fms.as_ref().map(|fms| fms.radius).unwrap_or(-1.0);
    let rmax = path_rmax(document);

    writeln!(out, "mpath, ms, nncrit, nlegxx, ipr4")?;
    write_i4_list(
        out,
        [
            path_flag(document),
            path_ms_flag(document),
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
        .as_ref()
        .map(|nrixs| if nrixs.qaverage { 5 } else { 7 })
        .unwrap_or(document.path_symmetry);
    write_i4_list(out, [ica])?;
    Ok(())
}

fn write_single_scattering_paths_dat(
    document: &FeffDocument,
    out: &mut impl std::fmt::Write,
) -> Result<()> {
    for title in &document.titles {
        writeln!(out, " {title}")?;
    }
    writeln!(
        out,
        " Single scattering paths from ss lines cards in feff input"
    )?;
    writeln!(out, " {}", "-".repeat(71))?;

    let rmax = path_rmax(document);
    for path in document
        .single_scattering_paths
        .iter()
        .filter(|path| rmax <= 0.0 || path.distance <= rmax)
    {
        write_single_scattering_path(document, path, out)?;
    }
    Ok(())
}

fn write_single_scattering_path(
    document: &FeffDocument,
    path: &SingleScatteringPath,
    out: &mut impl std::fmt::Write,
) -> Result<()> {
    validate_single_scattering_path(document, path)?;

    writeln!(
        out,
        "{:4}{:4}{:8.3}  index,nleg,degeneracy,r={:8.4}",
        path.index, 2, path.degeneracy, path.distance
    )?;
    writeln!(out, " single scattering")?;

    let scatterer_label = fixed_a6(potential_label(document, path.potential_index)?);
    writeln!(
        out,
        "{:12.6}{:12.6}{:12.6}{:4} '{}'",
        path.distance, 0.0, 0.0, path.potential_index, scatterer_label
    )?;

    let absorber_label = fixed_a6(potential_label(document, 0)?);
    writeln!(
        out,
        "{:12.6}{:12.6}{:12.6}{:4} '{}'  x,y,z,ipot",
        0.0, 0.0, 0.0, 0, absorber_label
    )?;
    Ok(())
}

fn write_genfmt_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "mfeff, ipr5, iorder, critcw, wnstar")?;
    writeln!(
        out,
        "{:4}{:4}{:8}{:13.5}{:>5}",
        control_flag(document, 4, 1),
        print_flag(document, 4, 0),
        document.iorder,
        document.critcw,
        fortran_bool(document.nstar)
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
    Ok(())
}

fn write_ff2x_inp(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let (tk, thetad, idwopt) = debye_values(document);
    let absolu = i32::from(document.absolute || document.eels.enabled);
    let momentum_transfer = if document.eels.enabled {
        document.eels.beam_direction
    } else if let Some(nrixs) = document.nrixs.as_ref() {
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
            i32::from(document.many_body_convolution),
            absolu,
            document.xsph_handoff.core_hole_broadening,
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
        tk,
        thetad,
        document.fine_structure_damping.alphat,
        document.fine_structure_damping.thetae,
        document.fine_structure_damping.sig2g,
        document.fine_structure_damping.sig_gk
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
        document
            .nrixs
            .as_ref()
            .map(|nrixs| nrixs.ldecmx)
            .unwrap_or(-1)
    )?;
    writeln!(out, "electronic temperature")?;
    writeln!(out, "{:13.5}", document.electronic_temperature)?;
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
