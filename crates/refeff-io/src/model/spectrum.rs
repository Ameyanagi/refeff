use super::*;

pub(super) fn parse_eels(input: &FeffInput) -> Result<Eels> {
    let magic_energy = parse_magic_energy(input)?;
    let (section, section_line) = if let Some(line) = card_by_feff_name(input, "ELNES") {
        ("ELNES", line)
    } else if let Some(line) = card_by_feff_name(input, "EXELFS") {
        ("EXELFS", line)
    } else {
        return Ok(Eels::default());
    };

    let rows = input.section_rows(section).collect::<Vec<_>>();
    let mut eels = Eels {
        enabled: true,
        ..Eels::default()
    };

    let line = required_eels_row(
        section_line,
        &rows,
        0,
        format!("{section} requires beam-energy row"),
    )?;
    let fields = section_fields_before_star(line)?;
    let Some(beam_energy) = fields.first() else {
        return Err(parse_error(
            line,
            format!("{section} beam row requires beam energy"),
        ));
    };
    eels.beam_energy = parse_f64(line, beam_energy)? * 1000.0;
    if let Some(value) = fields.get(1) {
        eels.average = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(2) {
        eels.cross_terms = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(3) {
        eels.relativistic = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(4) {
        eels.input = parse_i32(line, value)?;
    }
    if let Some(value) = fields.get(5) {
        eels.spectrum_column = parse_i32(line, value)?;
    }

    let mut row_index = 1;
    if eels.average != 1 {
        let line = required_eels_row(
            section_line,
            &rows,
            row_index,
            format!("{section} requires beam-direction row"),
        )?;
        let fields = section_fields_before_star(line)?;
        if fields.len() < 3 {
            return Err(parse_error(
                line,
                format!("{section} beam-direction row requires x, y, and z"),
            ));
        }
        let mut vector = [
            parse_f64(line, fields[0])?,
            parse_f64(line, fields[1])?,
            parse_f64(line, fields[2])?,
        ];
        let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        eels.beam_direction = vector;
        row_index += 1;
    }

    let line = required_eels_row(
        section_line,
        &rows,
        row_index,
        format!("{section} requires collection-angle row"),
    )?;
    let fields = section_fields_before_star(line)?;
    if fields.len() < 2 {
        return Err(parse_error(
            line,
            format!("{section} collection row requires collection and convergence angles"),
        ));
    }
    eels.collection_angle = parse_f64(line, fields[0])? / 1000.0;
    eels.convergence_angle = parse_f64(line, fields[1])? / 1000.0;
    row_index += 1;

    let line = required_eels_row(
        section_line,
        &rows,
        row_index,
        format!("{section} requires q-mesh row"),
    )?;
    let fields = section_fields_before_star(line)?;
    if fields.len() < 2 {
        return Err(parse_error(
            line,
            format!("{section} q-mesh row requires radial and angular values"),
        ));
    }
    eels.qmesh_radial = parse_i32(line, fields[0])?;
    eels.qmesh_angular = parse_i32(line, fields[1])?;
    row_index += 1;

    let line = required_eels_row(
        section_line,
        &rows,
        row_index,
        format!("{section} requires detector row"),
    )?;
    let fields = section_fields_before_star(line)?;
    if fields.len() < 2 {
        return Err(parse_error(
            line,
            format!("{section} detector row requires detector angles"),
        ));
    }
    eels.detector = [
        parse_f64(line, fields[0])? / 1000.0,
        parse_f64(line, fields[1])? / 1000.0,
    ];

    if let Some(magic_energy) = magic_energy {
        eels.magic = 1;
        eels.magic_energy = magic_energy;
    }

    if eels.average == 1 {
        eels.polarization_min = 10;
        eels.polarization_step = 1;
        eels.polarization_max = 10;
    } else {
        eels.polarization_min = 1;
        eels.polarization_step = if eels.cross_terms == 1 { 1 } else { 4 };
        eels.polarization_max = 9;
    }

    Ok(eels)
}

fn required_eels_row<'a>(
    section_line: &FeffLine,
    rows: &'a [&'a FeffLine],
    index: usize,
    message: String,
) -> Result<&'a FeffLine> {
    rows.get(index)
        .copied()
        .ok_or_else(|| parse_error(section_line, message))
}

fn parse_magic_energy(input: &FeffInput) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, "MAGIC") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "MAGIC requires emagic"));
    };
    Ok(Some(parse_f64(line, value)?))
}

pub(super) fn parse_nrixs(input: &FeffInput) -> Result<Option<Nrixs>> {
    let Some(line) = card_by_feff_name(input, "NRIXS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(raw_nq) = args.first() else {
        return Err(parse_error(line, "NRIXS requires nq"));
    };
    let raw_nq = parse_i32(line, raw_nq)?;
    if raw_nq == i32::MIN {
        return Err(parse_error(line, "NRIXS q-vector count is too negative"));
    }
    let qaverage = raw_nq < 0;
    let nq = raw_nq.abs().max(1);
    let q_count = nq as usize;
    let mut q_vectors = Vec::with_capacity(q_count);
    q_vectors.push(parse_nrixs_card_vector(line, args, qaverage, nq > 1)?);
    for row in input.section_rows("NRIXS").take(q_count.saturating_sub(1)) {
        q_vectors.push(parse_nrixs_section_vector(row, qaverage)?);
    }
    if q_vectors.len() != q_count {
        return Err(parse_error(
            line,
            format!(
                "NRIXS nq={nq} requires {} q-vector rows after the card",
                q_count.saturating_sub(1)
            ),
        ));
    }
    let first = q_vectors[0];
    let ldecmx = match parse_scalar_card(input, "LDEC")? {
        Some(value) => value,
        None => parse_scalar_card(input, "LDECMX")?.unwrap_or(-1.0),
    };
    Ok(Some(Nrixs {
        nq,
        qaverage,
        qvec: first.vector,
        qnorm: first.norm,
        q_vectors,
        ldecmx: ldecmx as i32,
        lj: parse_scalar_card(input, "LJMAX")?.unwrap_or(0.0) as i32,
    }))
}

fn parse_nrixs_card_vector(
    line: &FeffLine,
    args: &[String],
    qaverage: bool,
    require_weight: bool,
) -> Result<NrixsQVector> {
    if qaverage {
        let Some(qz) = args.get(1) else {
            return Err(parse_error(line, "NRIXS q-average card requires nq and q"));
        };
        let qz = parse_nrixs_qaverage_magnitude(line, qz)?;
        let weight = if require_weight {
            parse_nrixs_weight(line, args.get(2), args.get(3), true)?
        } else {
            [1.0, 0.0]
        };
        return Ok(NrixsQVector {
            vector: [0.0, 0.0, qz],
            norm: qz,
            weight,
        });
    }

    if args.len() < 4 {
        return Err(parse_error(line, "NRIXS card requires nq qx qy qz"));
    }
    let vector = [
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
        parse_f64(line, &args[3])?,
    ];
    let weight = if require_weight {
        parse_nrixs_weight(line, args.get(4), args.get(5), true)?
    } else {
        [1.0, 0.0]
    };
    Ok(NrixsQVector {
        vector,
        norm: vector[0].hypot(vector[1]).hypot(vector[2]),
        weight,
    })
}

fn parse_nrixs_section_vector(line: &FeffLine, qaverage: bool) -> Result<NrixsQVector> {
    let fields = section_fields_before_star(line)?;
    if qaverage {
        if fields.len() < 2 {
            return Err(parse_error(
                line,
                "NRIXS q-average row requires q and weight",
            ));
        }
        let qz = parse_nrixs_qaverage_magnitude(line, fields[0])?;
        let weight =
            parse_nrixs_weight(line, fields.get(1).copied(), fields.get(2).copied(), true)?;
        return Ok(NrixsQVector {
            vector: [0.0, 0.0, qz],
            norm: qz,
            weight,
        });
    }

    if fields.len() < 4 {
        return Err(parse_error(
            line,
            "NRIXS q-vector row requires qx qy qz and weight",
        ));
    }
    let vector = [
        parse_f64(line, fields[0])?,
        parse_f64(line, fields[1])?,
        parse_f64(line, fields[2])?,
    ];
    let weight = parse_nrixs_weight(line, fields.get(3).copied(), fields.get(4).copied(), true)?;
    Ok(NrixsQVector {
        vector,
        norm: vector[0].hypot(vector[1]).hypot(vector[2]),
        weight,
    })
}

fn parse_nrixs_qaverage_magnitude(line: &FeffLine, value: &str) -> Result<f64> {
    let qz = parse_f64(line, value)?;
    if qz <= 0.0 {
        return Err(parse_error(
            line,
            "NRIXS q-average magnitude must be positive",
        ));
    }
    Ok(qz)
}

fn parse_nrixs_weight(
    line: &FeffLine,
    real: Option<&String>,
    imaginary: Option<&String>,
    require_real: bool,
) -> Result<[f64; 2]> {
    if require_real {
        let Some(real) = real else {
            return Err(parse_error(line, "NRIXS q weight is required when nq > 1"));
        };
        return Ok([
            parse_f64(line, real)?,
            parse_optional_f64(line, imaginary)?.unwrap_or(0.0),
        ]);
    }
    Ok([
        parse_optional_f64(line, real)?.unwrap_or(1.0),
        parse_optional_f64(line, imaginary)?.unwrap_or(0.0),
    ])
}

pub(super) fn parse_mdff(
    input: &FeffInput,
    nrixs: &mut Option<Nrixs>,
    eels: &Eels,
) -> Result<Mdff> {
    let Some(line) = card_by_feff_name(input, "MDFF") else {
        return Ok(default_mdff());
    };
    let args = card_args(line)?;
    let imdff = parse_optional_i32(line, args.first())?.unwrap_or(1);
    let mut mdff = Mdff {
        imdff,
        qqmdff: -1.0,
        cosmdff_angle: 0.0,
    };

    if imdff == 2 {
        match args.len() {
            1 => {}
            3 => {
                mdff.qqmdff = parse_f64(line, &args[1])?;
                mdff.cosmdff_angle = parse_f64(line, &args[2])?;
            }
            _ => {
                return Err(parse_error(
                    line,
                    "MDFF 2 requires either no q-prime parameters or qqmdff cosmdff",
                ));
            }
        }
    }

    match imdff {
        value if value <= 0 => Ok(default_mdff()),
        1 => {
            if nrixs.is_some() {
                Ok(mdff)
            } else {
                Err(parse_error(line, "MDFF 1 requires NRIXS"))
            }
        }
        2 => {
            let Some(nrixs) = nrixs.as_mut() else {
                return Err(parse_error(line, "MDFF 2 requires NRIXS"));
            };
            if nrixs.nq != 2 {
                return Err(parse_error(line, "MDFF 2 requires NRIXS nq=2"));
            }
            if mdff.qqmdff >= 0.0 {
                apply_generated_mdff_qprime(line, &mdff, nrixs)?;
            }
            Ok(mdff)
        }
        3 => {
            if eels.enabled {
                Ok(mdff)
            } else {
                Err(parse_error(line, "MDFF 3 requires ELNES or EXELFS"))
            }
        }
        _ => Err(parse_error(line, "MDFF selector must be 0, 1, 2, or 3")),
    }
}

fn apply_generated_mdff_qprime(line: &FeffLine, mdff: &Mdff, nrixs: &mut Nrixs) -> Result<()> {
    if nrixs.q_vectors.len() < 2 {
        return Err(parse_error(line, "MDFF 2 requires two NRIXS q-vector rows"));
    }
    let first = nrixs.q_vectors[0];
    if first.norm == 0.0 {
        return Err(parse_error(
            line,
            "MDFF 2 generated q-prime requires a nonzero first q-vector",
        ));
    }
    let scale = mdff.qqmdff / first.norm;
    let angle = mdff.cosmdff_angle.to_radians();
    let (sin_angle, cos_angle) = angle.sin_cos();
    nrixs.q_vectors[1].vector = [
        first.vector[0] * scale,
        scale * (first.vector[1] * cos_angle + first.vector[2] * sin_angle),
        scale * (-first.vector[1] * sin_angle + first.vector[2] * cos_angle),
    ];
    Ok(())
}

fn default_mdff() -> Mdff {
    Mdff {
        imdff: 0,
        qqmdff: -1.0,
        cosmdff_angle: 0.0,
    }
}

pub(super) fn parse_rixs(input: &FeffInput) -> Result<Rixs> {
    let mut rixs = Rixs::default();

    if let Some(line) = card_by_feff_name(input, "EDGE") {
        let args = card_args(line)?;
        if let Some(edge) = args.first() {
            rixs.edges.clear();
            rixs.edges.push(edge.to_ascii_uppercase());
            for edge in args.iter().skip(1) {
                let edge = edge.to_ascii_uppercase();
                let is_valence = edge == "VAL";
                rixs.edges.push(edge);
                if is_valence {
                    rixs.mbconv = true;
                    break;
                }
            }
        }
    }

    if let Some(line) = card_by_feff_name(input, "RIXS") {
        let args = card_args(line)?;
        rixs.run = true;
        rixs.gamma_exp[0] = parse_optional_f64(line, args.first())?;
        rixs.gamma_exp[1] = parse_optional_f64(line, args.get(1))?;
        rixs.xmu = parse_optional_f64(line, args.get(2))?;
        rixs.read_poles = parse_optional_logical(line, args.get(3))?.unwrap_or(rixs.read_poles);
        rixs.skip_calc = parse_optional_logical(line, args.get(4))?.unwrap_or(rixs.skip_calc);
        rixs.mbconv |= parse_optional_logical(line, args.get(5))?.unwrap_or(false);
        rixs.read_sigma = parse_optional_logical(line, args.get(6))?.unwrap_or(rixs.read_sigma);
    }

    Ok(rixs)
}

fn parse_optional_logical(line: &FeffLine, value: Option<&String>) -> Result<Option<bool>> {
    value.map(|value| parse_logical(line, value)).transpose()
}
