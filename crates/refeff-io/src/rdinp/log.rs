use super::*;

/// Build the FEFF `rdinp` run summary that is written to `log.dat`.
///
/// This mirrors the summary block emitted by FEFF's `RDINP` stage: launch
/// banner, core-hole lifetime, title lines, spectroscopy/core-hole summary,
/// feature descriptions, active card list, and the final blank log line.
pub fn rdinp_log_dat(document: &FeffDocument) -> Result<LogDatData> {
    if document.active_cards.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "log.dat requires at least one active FEFF card".to_string(),
        });
    }

    let absorber = absorber_label(document)?;
    let absorber_z = absorber_potential(document)?
        .z
        .ok_or_else(|| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "log.dat requires numeric Z for absorbing potential".to_string(),
        })?;
    let ihole = document_ihole(document)?;
    let edge_label = summary_edge_label(document)?;
    let core_hole_lifetime_ev = core_hole_width_for_handoff(document, absorber_z, ihole)?;
    let spectroscopy = rdinp_spectroscopy_name(document);
    let corehole = rdinp_corehole_name(document.nohole);

    let titles = if document.titles.is_empty() {
        vec![" Once upon a time ...".to_string()]
    } else {
        document
            .titles
            .iter()
            .map(|title| format!(" {}", title.trim_end()))
            .collect()
    };

    Ok(LogDatData {
        version: FEFF_VERSION.to_string(),
        preamble_lines: rdinp_preamble_lines(document),
        core_hole_lifetime_ev: Some(core_hole_lifetime_ev),
        post_core_lines: rdinp_post_core_lines(document),
        titles,
        calculation_summary: Some(format!(
            "{absorber} {edge_label} edge {spectroscopy} using {corehole} corehole."
        )),
        features: rdinp_feature_descriptions(document),
        cards: document.active_cards.clone(),
        trailing_lines: vec![String::new()],
    })
}

/// Render the FEFF `rdinp` `log.dat` text for a parsed document.
pub fn rdinp_log_dat_string(document: &FeffDocument) -> Result<String> {
    render_log_dat_string(&rdinp_log_dat(document)?)
}

/// Build the FEFF `rdinp` error log for failures after line tokenization.
///
/// FEFF records input-scan messages before writing the fatal input line. This
/// keeps invalid templates, such as the HIGHZ example with `XXX` atomic-number
/// placeholders, comparable to the Fortran reference even when extraction
/// stops before any module handoff files are available.
pub fn rdinp_error_log(input: &FeffInput, error: &IoError) -> LogDatData {
    let failing_line = rdinp_error_line(input, error);
    if let Some(log) = rdinp_legacy_error_log(input, error, failing_line) {
        return log;
    }
    let mut lines = rdinp_error_preamble_lines(input, failing_line);
    lines.push(" Error reading input, bad line follows:".to_string());
    lines.push(format!(" {}", rdinp_error_raw_line(failing_line, error)));
    lines.push("RDINP fatal error.".to_string());

    LogDatData {
        version: FEFF_VERSION.to_string(),
        preamble_lines: lines,
        core_hole_lifetime_ev: None,
        post_core_lines: Vec::new(),
        titles: Vec::new(),
        calculation_summary: None,
        features: Vec::new(),
        cards: Vec::new(),
        trailing_lines: Vec::new(),
    }
}

/// Render FEFF-compatible `log.dat` text for an `rdinp` input failure.
pub fn rdinp_error_log_string(input: &FeffInput, error: &IoError) -> Result<String> {
    render_log_dat_string(&rdinp_error_log(input, error))
}

/// Render FEFF's `.feff.error` crash sentinel for an active `rdinp` run.
pub fn rdinp_error_sentinel_string() -> String {
    const SENTINEL: &str = " Starting FEFF9 module rdinp.  If this message is still here after the module finishes running, it must have crashed. The content of this file is wiped on successful termination.";
    format!("{SENTINEL:<501}\n")
}

fn rdinp_legacy_error_log(
    input: &FeffInput,
    error: &IoError,
    failing_line: Option<&FeffLine>,
) -> Option<LogDatData> {
    let message = parse_error_message(error)?;
    let keyword = failing_line.and_then(card_keyword);

    if message.starts_with("HOLE requires") || message.starts_with("HOLE ihole") {
        return Some(simple_rdinp_error_log(
            vec![
                " Use NOHOLE to calculate without core hole.  Only ihole greater than zero are allowed."
                    .to_string(),
                "RDINP".to_string(),
            ],
            None,
        ));
    }

    if message.contains("BANDSTRUCTURE requires") {
        return Some(simple_rdinp_error_log(
            vec![
                "BANDSTRUCTURE card is experimental.".to_string(),
                "BANDSTRUCTURE requires at least: emin  emax  estep  ikpath".to_string(),
                String::new(),
            ],
            None,
        ));
    }

    if message.starts_with("SCXC requires") || message.starts_with("SCXC iscfxc") {
        return Some(simple_rdinp_error_log(
            vec![
                "Error: iscfxc should take one of the values 11 for vBH, 12 for PZ, 21 for PDW, or 22 for KSDT ... stopping"
                    .to_string(),
            ],
            None,
        ));
    }

    if matches!(keyword, Some("OVERLAP")) && input_has_card(input, "ATOMS") {
        return Some(simple_rdinp_error_log(
            vec![
                " Cannot use ATOMS and OVERLAP in the same feff.inp.".to_string(),
                "RDINP".to_string(),
            ],
            None,
        ));
    }

    if matches!(keyword, Some("RCONV")) {
        return Some(simple_rdinp_error_log(
            vec![
                " RCONV".to_string(),
                " RCONV".to_string(),
                " Token        0".to_string(),
                " Keyword unrecognized.".to_string(),
                " See FEFF document -- some old features are no longer available.".to_string(),
                "RDINP-2".to_string(),
            ],
            None,
        ));
    }

    if message.starts_with("COORDINATES requires") || message.starts_with("COORDINATES must") {
        return Some(simple_rdinp_error_log(
            vec![
                "Attempt to enter funky lattice coordinates.".to_string(),
                "Please stick to one of the formats described in the manual.".to_string(),
                "Exiting now.".to_string(),
            ],
            None,
        ));
    }

    if message.starts_with("MDFF") && message.contains("requires NRIXS") {
        return Some(simple_rdinp_error_log(
            vec![
                "NRIXS type MDFF calculation selected - summed over all q,q' pairs.".to_string(),
                "ERROR - the selected MDFF option is only available with the NRIXS card."
                    .to_string(),
                "RDINP".to_string(),
            ],
            None,
        ));
    }

    if matches!(keyword, Some("SCREEN")) && message.starts_with("SCREEN requires") {
        return Some(simple_rdinp_error_log(Vec::new(), None));
    }

    if (message.starts_with("LDEC and LJMAX cards only allowed with NRIXS")
        || message.starts_with("Cannot use CGRID without"))
        && let Some(core_hole) = partial_core_hole_lifetime(input)
    {
        return Some(simple_rdinp_error_log(Vec::new(), Some(core_hole)));
    }

    if message.starts_with("ERROR more than one type of spectroscopy selected")
        && (input_has_card(input, "ELNES") || input_has_card(input, "EXELFS"))
        && input_has_card(input, "POTENTIALS")
    {
        return Some(simple_rdinp_error_log(
            vec![
                " Error reading input, bad line follows:".to_string(),
                " POTENTIALS".to_string(),
                "RDINP fatal error.".to_string(),
            ],
            None,
        ));
    }

    None
}

fn simple_rdinp_error_log(
    preamble_lines: Vec<String>,
    core_hole_lifetime_ev: Option<f64>,
) -> LogDatData {
    LogDatData {
        version: FEFF_VERSION.to_string(),
        preamble_lines,
        core_hole_lifetime_ev,
        post_core_lines: Vec::new(),
        titles: Vec::new(),
        calculation_summary: None,
        features: Vec::new(),
        cards: Vec::new(),
        trailing_lines: Vec::new(),
    }
}

fn parse_error_message(error: &IoError) -> Option<&str> {
    match error {
        IoError::Parse { message, .. } => Some(message.as_str()),
        _ => None,
    }
}

fn card_keyword(line: &FeffLine) -> Option<&str> {
    match &line.kind {
        LineKind::Card { keyword, .. } => Some(keyword.as_str()),
        LineKind::SectionData { .. } => None,
    }
}

fn input_has_card(input: &FeffInput, canonical: &str) -> bool {
    input.cards().any(|line| {
        card_keyword(line).is_some_and(|keyword| match canonical {
            "ATOMS" => matches!(keyword, "ATOMS" | "ATOM"),
            "POTENTIALS" => matches!(keyword, "POTENTIALS" | "POTENTIAL" | "POT"),
            "ELNES" => matches!(keyword, "ELNES" | "ELNE"),
            "EXELFS" => matches!(keyword, "EXELFS" | "EXEL"),
            _ => keyword == canonical,
        })
    })
}

fn partial_core_hole_lifetime(input: &FeffInput) -> Option<f64> {
    let z = input
        .section_rows("POTENTIALS")
        .find_map(absorber_atomic_number)?;
    let ihole = partial_ihole(input)?;
    core_hole_width_ev(z, ihole).ok()
}

fn absorber_atomic_number(line: &FeffLine) -> Option<i32> {
    let LineKind::SectionData { fields, .. } = &line.kind else {
        return None;
    };
    let ipot = fields.first()?.parse::<i32>().ok()?;
    (ipot == 0).then(|| fields.get(1)?.parse::<i32>().ok())?
}

fn partial_ihole(input: &FeffInput) -> Option<i32> {
    input
        .cards()
        .find(|line| card_keyword(line) == Some("EDGE"))
        .map(|line| match &line.kind {
            LineKind::Card { args, .. } => args.first().map_or("K", String::as_str),
            LineKind::SectionData { .. } => "K",
        })
        .map_or(Some(1), edge_index)
}

/// Render the FEFF `rdinp` stdout text for a parsed document.
///
/// FEFF normally mirrors the `rdinp` log to stdout. One legacy diagnostic is
/// stdout-only: when spin is enabled and the absorber potential omits
/// `spinph`, the Fortran code writes a list-directed integer before logging the
/// default spin table.
pub fn rdinp_stdout_string(document: &FeffDocument) -> Result<String> {
    let mut data = rdinp_log_dat(document)?;
    let mut post_core_lines = rdinp_stdout_only_post_core_lines(document);
    post_core_lines.extend(data.post_core_lines);
    data.post_core_lines = post_core_lines;
    render_log_dat_string(&data)
}
