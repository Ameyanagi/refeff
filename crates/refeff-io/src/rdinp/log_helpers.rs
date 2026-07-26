//! Run-summary and input-scan log helpers for FEFF `rdinp`.

use crate::format::fortran_exp;
use crate::input::{FeffInput, FeffLine, LineKind};
use crate::model::FeffDocument;
use crate::{IoError, Result};
use refeff_core::standard_edge_label;

use super::helpers::{default_atomic_spinph, document_ihole, raw_lmaxsc};

pub(super) fn summary_edge_label(document: &FeffDocument) -> Result<&'static str> {
    let ihole = document_ihole(document)?;
    standard_edge_label(&ihole.to_string()).ok_or_else(|| IoError::Parse {
        path: document.source.clone(),
        line: 0,
        message: format!("unknown FEFF hole index {ihole}"),
    })
}

pub(super) fn rdinp_spectroscopy_name(document: &FeffDocument) -> &'static str {
    const SPECTROSCOPY_ORDER: [(&str, &str); 11] = [
        ("XANES", "XANES"),
        ("EXAFS", "EXAFS"),
        ("ELNES", "ELNES"),
        ("EXELFS", "EXELFS"),
        ("COMPTON", "COMPTON"),
        ("NRIXS", "NRIXS"),
        ("RIXS", "RIXS"),
        ("XES", "XES"),
        ("FPRIME", "FPRIME"),
        ("XMCD", "XMCD"),
        ("DANES", "DANES"),
    ];

    SPECTROSCOPY_ORDER
        .iter()
        .rev()
        .find(|(card, _)| active_card(document, card))
        .map(|(_, name)| *name)
        .unwrap_or("EXAFS")
}

pub(super) fn rdinp_corehole_name(nohole: i32) -> &'static str {
    match nohole {
        0 => "no",
        2 => "RPA",
        _ => "FSR",
    }
}

pub(super) fn rdinp_feature_descriptions(document: &FeffDocument) -> Vec<String> {
    const FEATURES: [(&str, &str); 11] = [
        ("DEBYE", "Debye-Waller factors"),
        ("MPSE", "Many-Pole Self-Energy"),
        ("TDLDA", "Time-Dependent Density-Functional-Theory"),
        ("SPIN", "Spin Polarization"),
        ("SCF", "Self-Consistent Field potentials"),
        ("UNFREEZEF", "SCF-converged f-states"),
        ("PMBSE", "approximated Bethe-Salpeter cross-section"),
        ("S02CONV", "Satellite Spectral Function"),
        ("EXTPOT", "External Potentials"),
        ("CONFIG", "Custom electron configuration"),
        ("TEMP", "Finite Temperature Fermi Distribution"),
    ];

    FEATURES
        .iter()
        .filter(|(card, _)| active_feature_card(document, card))
        .map(|(_, prose)| (*prose).to_string())
        .collect()
}

fn active_feature_card(document: &FeffDocument, feature_card: &str) -> bool {
    match feature_card {
        "CONFIG" => active_card(document, "CONFIGURATION"),
        card => active_card(document, card),
    }
}

fn active_card(document: &FeffDocument, card: &str) -> bool {
    document.active_cards.iter().any(|active| active == card)
}

pub(super) fn rdinp_preamble_lines(document: &FeffDocument) -> Vec<String> {
    let mut lines = Vec::new();

    for card in &document.input_cards {
        match card.as_str() {
            "RGRID" => lines.push(format!(
                " RGRID, rgrd; {}",
                fortran_exp(document.rgrid, 13, 5)
            )),
            "XES" => lines.push("  XES:".to_string()),
            "DANES" => lines.push("  DANES:".to_string()),
            "FPRIME" => lines.push(" FPRIME:".to_string()),
            "RSIGMA" => lines.push(
                " Real self energy only will be used.  FEFF results will be unreliable."
                    .to_string(),
            ),
            "RPHASES" => lines.push(
                " Real phase shifts only will be used.  FEFF results will be unreliable."
                    .to_string(),
            ),
            "DEBYE" => {
                if let Some(debye) = document
                    .debye
                    .as_ref()
                    .filter(|debye| debye.requested_idwopt > 5)
                {
                    lines.push(format!(
                        " Option idwopt={:5}  is not available.",
                        debye.requested_idwopt
                    ));
                    lines.push("...setting idwopt=2 to use RM.".to_string());
                }
            }
            "SYMMETRY" => lines.push(symmetry_log_line(document.path_symmetry)),
            "BAND" => lines.push("BANDSTRUCTURE card is experimental.".to_string()),
            "SCREEN" => lines.push(screen_log_line()),
            "REAL" => lines.push("Working in real space.".to_string()),
            "RECIPROCAL" => lines.push("Working in reciprocal space.".to_string()),
            "LATTICE" if document.reciprocal && document.reciprocal_input.is_some() => lines.push(
                "Taking crystal structure from feff.inp.  Note: .cif input is now recommended."
                    .to_string(),
            ),
            "CIF" => lines.push("Taking crystal structure from .cif file.".to_string()),
            _ => {}
        }
    }

    lines.extend(
        document
            .potentials
            .iter()
            .filter(|potential| !document.unfreezef && raw_lmaxsc(potential) > 2)
            .map(|potential| {
                format!(
                    "Resetting lmaxsc to 2 for iph = {:4}.  Use  UNFREEZE to prevent this.",
                    potential.ipot
                )
            }),
    );
    lines
}

pub(super) fn rdinp_post_core_lines(document: &FeffDocument) -> Vec<String> {
    if document.spin == 0 {
        return Vec::new();
    }

    document
        .potentials
        .iter()
        .filter(|potential| potential.spinph.is_none())
        .filter_map(|potential| {
            let spin = if potential.ipot == 0 {
                potential
                    .z
                    .and_then(|z| default_atomic_spinph(z).or(Some(0.0)))
            } else {
                potential.z.and_then(default_atomic_spinph)
            }?;
            Some(vec![
                "No spin set in POTENTIALS card. Using default spins:".to_string(),
                "iph   spinph".to_string(),
                format!("{:3} {spin:.1}", potential.ipot),
            ])
        })
        .flatten()
        .collect()
}

pub(super) fn rdinp_error_line<'a>(input: &'a FeffInput, error: &IoError) -> Option<&'a FeffLine> {
    let IoError::Parse { path, line, .. } = error else {
        return None;
    };

    input
        .lines
        .iter()
        .find(|input_line| input_line.location.line == *line && input_line.location.path == *path)
}

pub(super) fn rdinp_error_preamble_lines(
    input: &FeffInput,
    failing_line: Option<&FeffLine>,
) -> Vec<String> {
    input
        .lines
        .iter()
        .take_while(|line| failing_line != Some(*line))
        .filter_map(rdinp_input_scan_log_line)
        .collect()
}

fn rdinp_input_scan_log_line(line: &FeffLine) -> Option<String> {
    let LineKind::Card { keyword, args, .. } = &line.kind else {
        return None;
    };

    match keyword.as_str() {
        "HIGHZ" => Some("Using finite nucleus.".to_string()),
        "RGRID" => args
            .first()
            .and_then(|value| value.parse::<f64>().ok())
            .map(|rgrid| format!(" RGRID, rgrd; {}", fortran_exp(rgrid, 13, 5))),
        "XES" => Some("  XES:".to_string()),
        "DANES" => Some("  DANES:".to_string()),
        "FPRIME" => Some(" FPRIME:".to_string()),
        "RSIGMA" => Some(
            " Real self energy only will be used.  FEFF results will be unreliable.".to_string(),
        ),
        "RPHASES" => Some(
            " Real phase shifts only will be used.  FEFF results will be unreliable.".to_string(),
        ),
        "SYMMETRY" => args
            .first()
            .and_then(|value| value.parse::<i32>().ok())
            .map(|ica| symmetry_log_line(if (1..=7).contains(&ica) { ica } else { -1 })),
        "BANDSTRUCTURE" | "BAND" => Some("BANDSTRUCTURE card is experimental.".to_string()),
        "SCREEN" => Some(screen_log_line()),
        "REAL" => Some("Working in real space.".to_string()),
        "RECIPROCAL" => Some("Working in reciprocal space.".to_string()),
        "CIF" => Some("Taking crystal structure from .cif file.".to_string()),
        _ => None,
    }
}

fn symmetry_log_line(ica: i32) -> String {
    format!(" SYMMETRY CARD - fixing icase to {ica:4} in module PATH.")
}

fn screen_log_line() -> String {
    ":INFO  User provides options for screen.inp".to_string()
}

pub(super) fn rdinp_error_raw_line(failing_line: Option<&FeffLine>, error: &IoError) -> String {
    match failing_line {
        Some(line) => line.raw.chars().take(71).collect(),
        None => error.to_string().chars().take(71).collect(),
    }
}

pub(super) fn rdinp_stdout_only_post_core_lines(document: &FeffDocument) -> Vec<String> {
    if document.spin == 0 {
        return Vec::new();
    }

    let absorber_spin_defaults = document
        .potentials
        .iter()
        .any(|potential| potential.ipot == 0 && potential.spinph.is_none());
    if absorber_spin_defaults {
        vec!["           1".to_string()]
    } else {
        Vec::new()
    }
}
