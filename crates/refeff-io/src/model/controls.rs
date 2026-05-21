use super::*;

pub(super) fn parse_scf(input: &FeffInput) -> Result<Option<Scf>> {
    let Some(line) = card_by_feff_name(input, "SCF") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "SCF requires a radius"));
    };
    let mut iterations = parse_optional_i32(line, args.get(2))?.unwrap_or(100);
    if iterations <= 0 || iterations > 100 {
        iterations = 100;
    }
    let mut ca = parse_optional_f64(line, args.get(3))?.unwrap_or(0.2);
    if ca < 0.0 {
        ca = 0.0;
    }
    let mut nmix = parse_optional_i32(line, args.get(4))?.unwrap_or(1);
    if nmix <= 0 {
        nmix = 1;
    } else if nmix > 30 {
        nmix = 30;
    }
    let mut ecv = parse_optional_f64(line, args.get(5))?.unwrap_or(-40.0);
    if ecv >= 0.0 {
        ecv = -40.0;
    }

    Ok(Some(Scf {
        radius: parse_f64(line, radius)?,
        lfms: parse_optional_i32(line, args.get(1))?.unwrap_or(0).min(1),
        iterations,
        ca,
        nmix,
        ecv,
        icoul: parse_optional_i32(line, args.get(6))?.unwrap_or(0),
    }))
}

pub(super) fn parse_exchange(input: &FeffInput) -> Result<Option<Exchange>> {
    let Some(line) = card_by_feff_name(input, "EXCHANGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "EXCHANGE requires ixc, vr0, and vi0"));
    }

    Ok(Some(Exchange {
        ixc: parse_i32(line, &args[0])?,
        vr0: parse_f64(line, &args[1])?,
        vi0: parse_f64(line, &args[2])?,
        ixc0: parse_optional_i32(line, args.get(3))?,
    }))
}

pub(super) fn parse_exafs(input: &FeffInput) -> Result<Option<Exafs>> {
    let Some(line) = card_by_feff_name(input, "EXAFS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(xkmax) = args.first() else {
        return Err(parse_error(line, "EXAFS requires an xkmax value"));
    };

    Ok(Some(Exafs {
        xkmax: parse_f64(line, xkmax)?,
    }))
}

pub(super) fn parse_spectrum_grid(
    input: &FeffInput,
    exchange: Option<&Exchange>,
    ispec: i32,
) -> Result<SpectrumGrid> {
    let mut grid = SpectrumGrid {
        ixc0: exchange
            .and_then(|exchange| exchange.ixc0)
            .filter(|ixc0| *ixc0 >= 0)
            .unwrap_or_else(|| if (1..=4).contains(&ispec) { 2 } else { 0 }),
        ..SpectrumGrid::default()
    };

    if let Some((name, line)) = ["XANES", "DANES", "ELNES"]
        .into_iter()
        .find_map(|name| card_by_feff_name(input, name).map(|line| (name, line)))
    {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            grid.xkstep = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep < 0.01 {
            grid.xkstep = 0.01;
        }
        if matches!(name, "XANES" | "ELNES") {
            if grid.xkstep > 2.0 {
                grid.xkstep = 0.5;
            }
            if grid.xkmax.abs() < 2.0 {
                grid.xkmax = 2.0;
            }
            if grid.xkmax.abs() > 200.0 {
                grid.xkmax = 200.0;
            }
        } else if grid.xkmax < 2.0 {
            grid.xkmax = 2.0;
        }
    } else if let Some(line) = card_by_feff_name(input, "XES") {
        let args = card_args(line)?;
        grid.xkstep = 0.01;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            grid.xkstep = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep <= grid.xkmax {
            grid.xkstep = 0.01;
        }
        if grid.xkmax >= 0.0 {
            grid.xkmax = -40.0;
        }
    } else if let Some(line) = card_by_feff_name(input, "FPRIME") {
        let args = card_args(line)?;
        if args.len() < 2 {
            return Err(parse_error(line, "FPRIME requires emin and emax"));
        }
        grid.xkmax = parse_f64(line, &args[0])?;
        grid.xkstep = parse_f64(line, &args[1])?;
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep < grid.xkmax {
            grid.xkstep = grid.xkmax;
        }
    } else if let Some(line) =
        card_by_feff_name(input, "EXAFS").or_else(|| card_by_feff_name(input, "EXELFS"))
    {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
    }

    Ok(grid)
}

pub(super) fn parse_temp(input: &FeffInput) -> Result<(f64, i32)> {
    let mut temperature = 0.0;
    let mut iscfxc = 11;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("TEMP") => {
                let args = card_args(line)?;
                if let Some(value) = args.first() {
                    temperature = parse_f64(line, value)?;
                    if !temperature.is_finite() {
                        return Err(parse_error(line, "TEMP value must be finite"));
                    }
                }
                if let Some(value) = args.get(1) {
                    iscfxc = parse_i32(line, value)?;
                }
            }
            Some("SCXC") => {
                let args = card_args(line)?;
                let Some(value) = args.first() else {
                    return Err(parse_error(line, "SCXC requires iscfxc"));
                };
                iscfxc = parse_i32(line, value)?;
                validate_scxc(line, iscfxc)?;
            }
            _ => {}
        }
    }

    Ok((temperature, iscfxc))
}

fn validate_scxc(line: &FeffLine, iscfxc: i32) -> Result<()> {
    if matches!(iscfxc, 11 | 12 | 21 | 22) {
        Ok(())
    } else {
        Err(parse_error(
            line,
            "SCXC iscfxc must be one of 11, 12, 21, or 22",
        ))
    }
}

pub(super) fn parse_criteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = card_by_feff_name(input, "CRITERIA") else {
        return Ok((4.0, 2.5));
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "CRITERIA requires critcw and critpw"));
    }
    Ok((parse_f64(line, &args[0])?, parse_f64(line, &args[1])?))
}

pub(super) fn parse_pcriteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = card_by_feff_name(input, "PCRITERIA") else {
        return Ok((0.0, 0.0));
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "PCRITERIA requires pcritk and pcrith"));
    }
    Ok((parse_f64(line, &args[0])?, parse_f64(line, &args[1])?))
}

pub(super) fn parse_lreal(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "RPHASES").is_some() {
        2
    } else {
        i32::from(card_by_feff_name(input, "RSIGMA").is_some())
    }
}

pub(super) fn parse_iorder(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "IORD") else {
        return Ok(2);
    };
    let args = card_args(line)?;
    parse_optional_i32(line, args.first()).map(|value| value.unwrap_or(0))
}

pub(super) fn parse_mpse(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = card_by_feff_name(input, "MPSE") else {
        return Ok((0, 100));
    };
    let args = card_args(line)?;
    let mut i_plsmn = parse_optional_i32(line, args.first())?.unwrap_or(1);
    if i_plsmn == 4 {
        i_plsmn = 1;
    }
    let n_poles = parse_optional_i32(line, args.get(1))?.unwrap_or(100);
    Ok((i_plsmn, n_poles))
}

pub(super) fn parse_ispec(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "COMPTON").is_some() || card_by_feff_name(input, "DENS").is_some() {
        5
    } else if card_by_feff_name(input, "FPRIME").is_some() {
        4
    } else if card_by_feff_name(input, "DANES").is_some() {
        3
    } else if card_by_feff_name(input, "XES").is_some() {
        2
    } else if card_by_feff_name(input, "XANES").is_some()
        || card_by_feff_name(input, "ELNES").is_some()
        || card_by_feff_name(input, "NRIXS").is_some()
    {
        1
    } else {
        0
    }
}

pub(super) fn parse_ipol(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "XMCD").is_some() {
        2
    } else if card_by_feff_name(input, "POLARIZATION").is_some() {
        1
    } else {
        0
    }
}

pub(super) fn parse_multipole(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = card_by_feff_name(input, "MULT") else {
        return Ok((0, 0));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_i32(line, args.first())?.unwrap_or(0),
        parse_optional_i32(line, args.get(1))?.unwrap_or(0),
    ))
}

pub(super) fn parse_polarization_vector(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = card_by_feff_name(input, "POLARIZATION") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "POLARIZATION requires x, y, and z"));
    }
    Ok([
        parse_f64(line, &args[0])?,
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
    ])
}

pub(super) fn parse_ellipticity(input: &FeffInput) -> Result<(f64, [f64; 3])> {
    let Some(line) = card_by_feff_name(input, "ELLIPTICITY") else {
        return Ok((0.0, [0.0; 3]));
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "ELLIPTICITY requires ellipticity and incident direction",
        ));
    }
    Ok((
        parse_f64(line, &args[0])?,
        [
            parse_f64(line, &args[1])?,
            parse_f64(line, &args[2])?,
            parse_f64(line, &args[3])?,
        ],
    ))
}

pub(super) fn parse_spin(input: &FeffInput) -> Result<(i32, [f64; 3])> {
    let Some(line) = card_by_feff_name(input, "SPIN") else {
        return Ok((0, [0.0; 3]));
    };
    let args = card_args(line)?;
    let spin = parse_optional_i32(line, args.first())?.unwrap_or(0);
    let default_vector = if spin == 0 { [0.0; 3] } else { [0.0, 0.0, 1.0] };
    Ok((
        spin,
        [
            parse_optional_f64(line, args.get(1))?.unwrap_or(default_vector[0]),
            parse_optional_f64(line, args.get(2))?.unwrap_or(default_vector[1]),
            parse_optional_f64(line, args.get(3))?.unwrap_or(default_vector[2]),
        ],
    ))
}

pub(super) fn parse_nohole(input: &FeffInput) -> Result<i32> {
    if let (Some(corehole), Some(_)) = (
        card_by_feff_name(input, "COREHOLE"),
        card_by_feff_name(input, "NOHOLE"),
    ) {
        return Err(parse_error(
            corehole,
            "NOHOLE and COREHOLE cards are mutually exclusive",
        ));
    }

    if let Some(line) = card_by_feff_name(input, "COREHOLE") {
        let args = card_args(line)?;
        let Some(mode) = args.first() else {
            return Ok(-1);
        };
        return match mode.to_ascii_uppercase().as_str() {
            "NONE" => Ok(0),
            "RPA" => Ok(2),
            "FSR" | "REGULAR" => Ok(-1),
            _ => Err(parse_error(
                line,
                "COREHOLE must be NONE, RPA, FSR, or REGULAR",
            )),
        };
    }

    if let Some(line) = card_by_feff_name(input, "NOHOLE") {
        let args = card_args(line)?;
        return parse_optional_i32(line, args.first()).map(|value| value.unwrap_or(0));
    }

    Ok(-1)
}

pub(super) fn parse_fms(input: &FeffInput) -> Result<Option<Fms>> {
    let Some(line) = card_by_feff_name(input, "FMS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "FMS requires a radius"));
    };

    let radius = parse_f64(line, radius)?;
    let lfms = parse_optional_i32(line, args.get(1))?.unwrap_or(0).min(1);
    let rdirec = parse_optional_f64(line, args.get(5))?.unwrap_or(2.0 * radius);

    Ok(Some(Fms {
        radius,
        lfms,
        minv: parse_optional_i32(line, args.get(2))?.unwrap_or(0),
        toler1: parse_optional_f64(line, args.get(3))?.unwrap_or(0.001),
        toler2: parse_optional_f64(line, args.get(4))?.unwrap_or(0.001),
        rdirec: if rdirec < 0.0 || rdirec > 2.0 * radius {
            2.0 * radius
        } else {
            rdirec
        },
    }))
}

pub(super) fn parse_crpa(input: &FeffInput) -> Result<Crpa> {
    let Some(line) = card_by_feff_name(input, "CRPA") else {
        return Ok(Crpa::default());
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "CRPA requires l and rcut values"));
    }
    Ok(Crpa {
        enabled: true,
        l: parse_i32(line, &args[0])?,
        rcut: parse_f64(line, &args[1])?,
    })
}

pub(super) fn parse_compton(input: &FeffInput) -> Result<Compton> {
    let mut compton = Compton::default();

    if let Some(line) = card_by_feff_name(input, "COMPTON") {
        let args = card_args(line)?;
        compton.do_compton = true;
        if let Some(value) = args.first() {
            compton.pqmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            compton.npq = parse_i32(line, value)?;
        }
        if parse_optional_i32(line, args.get(2))?.unwrap_or(0) > 0 {
            compton.force_jzzp = true;
        }
    }

    compton.do_rhozzp = card_by_feff_name(input, "RHOZZP").is_some();

    if let Some(line) = card_by_feff_name(input, "CGRID") {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            compton.zpmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            compton.ns = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(2) {
            compton.nphi = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(3) {
            compton.nz = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(4) {
            compton.nzp = parse_i32(line, value)?;
        }
    }

    Ok(compton)
}

pub(super) fn parse_hubbard(input: &FeffInput) -> Result<Hubbard> {
    let Some(line) = card_by_feff_name(input, "HUBBARD") else {
        return Ok(Hubbard::default());
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "HUBBARD requires U, J, fermi_shift, and l values",
        ));
    }
    Ok(Hubbard {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: parse_f64(line, &args[0])?,
        j: parse_f64(line, &args[1])?,
        fermi_shift: parse_f64(line, &args[2])?,
        l: parse_i32(line, &args[3])?,
    })
}
