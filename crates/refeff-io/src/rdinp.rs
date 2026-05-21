//! Writers for the first `rdinp`-level compatibility outputs.
//!
//! The full FEFF `rdinp` module emits many module input files. This module
//! starts with `atoms.dat`, which is the structural bridge consumed by later
//! FEFF modules, and will grow as the port advances.

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::config_input::config_inp_lines_string;
use crate::control_input::{
    band_input_string, fullspectrum_input_string, opcons_input_string, reciprocal_input_string,
};
use crate::format::fortran_exp;
use crate::input::{FeffInput, FeffLine, LineKind};
use crate::log_dat::{LogDatData, log_dat_string as render_log_dat_string};
use crate::model::{Atom, FeffDocument, Potential, SingleScatteringPath};
use crate::screen_input::screen_input_string;
use crate::sfconv_input::sfconv_input_string;
use crate::{IoError, Result};
use num_complex::Complex64;
use refeff_core::{
    atomic_symbol, core_hole_width_ev, edge_index, normalize_vector, nrixs_qtrig,
    rotate_into_reference_frame, standard_edge_label, vector_norm,
};

const FEFF_VERSION: &str = "FEFF 10.0.0";

/// File name key for a FEFF `rdinp` text output.
pub type TextOutputName = Cow<'static, str>;

/// Ordered text outputs rendered by the FEFF `rdinp` compatibility stage.
pub type TextOutputs = BTreeMap<TextOutputName, String>;

mod control_inputs;
mod log;
mod module_inputs;
mod structure;

pub use control_inputs::{
    band_inp_string, compton_inp_string, config_inp_string, crpa_inp_string, density_inp_string,
    eels_inp_string, fullspectrum_inp_string, fullspectrum_inp_string_for_document,
    global_inp_string, grid_inp_string, hubbard_inp_string, mdff_inp_string, opcons_inp_string,
    reciprocal_inp_string, screen_inp_string, screen_inp_string_for_document,
};
pub use log::{
    rdinp_error_log, rdinp_error_log_string, rdinp_error_sentinel_string, rdinp_log_dat,
    rdinp_log_dat_string, rdinp_stdout_string,
};
pub use module_inputs::{
    dmdw_inp_string, ff2x_inp_string, fms_inp_string, genfmt_inp_string, ldos_inp_string,
    paths_inp_string, pot_inp_string, rixs_inp_string, sfconv_inp_string,
    single_scattering_paths_dat_string, xsph_inp_string,
};
pub use structure::{atoms_dat_string, dimensions_dat_string, geom_dat_string, write_atoms_dat};

/// Render all currently supported text outputs from FEFF's `rdinp` stage.
pub fn text_outputs(document: &FeffDocument) -> Result<TextOutputs> {
    let mut outputs = TextOutputs::new();
    if !document.potentials.is_empty() && !document.atoms.is_empty() {
        insert_output(
            &mut outputs,
            ".dimensions.dat",
            dimensions_dat_string(document)?,
        );
    }
    if !document.atoms.is_empty() {
        insert_output(&mut outputs, "atoms.dat", atoms_dat_string(document)?);
    }
    insert_output(
        &mut outputs,
        "band.inp",
        band_input_string(&document.band_input)?,
    );
    insert_output(&mut outputs, "compton.inp", compton_inp_string(document)?);
    if !document.config_records.is_empty() {
        insert_output(&mut outputs, "config.inp", config_inp_string(document)?);
    }
    insert_output(&mut outputs, "crpa.inp", crpa_inp_string(document));
    if !document.density_records.is_empty() {
        insert_output(&mut outputs, "density.inp", density_inp_string(document)?);
    }
    insert_output(&mut outputs, "dmdw.inp", dmdw_inp_string(document)?);
    insert_output(&mut outputs, "eels.inp", eels_inp_string(document)?);
    if document.mdff.imdff == 3 {
        insert_output(&mut outputs, "mdff.inp", mdff_inp_string()?);
    }
    insert_output(&mut outputs, "ff2x.inp", ff2x_inp_string(document)?);
    if !document.potentials.is_empty() {
        insert_output(&mut outputs, "fms.inp", fms_inp_string(document)?);
    }
    insert_output(
        &mut outputs,
        "fullspectrum.inp",
        fullspectrum_inp_string_for_document(document)?,
    );
    insert_output(&mut outputs, "genfmt.inp", genfmt_inp_string(document)?);
    if !document.no_geom && !document.atoms.is_empty() {
        insert_output(&mut outputs, "geom.dat", geom_dat_string(document)?);
    }
    insert_output(&mut outputs, "global.inp", global_inp_string(document)?);
    if !document.egrid_records.is_empty()
        || document.active_cards.iter().any(|card| card == "EGRID")
    {
        insert_output(&mut outputs, "grid.inp", grid_inp_string(document)?);
    }
    insert_output(&mut outputs, "hubbard.inp", hubbard_inp_string(document));
    if !document.potentials.is_empty() {
        insert_output(&mut outputs, "ldos.inp", ldos_inp_string(document)?);
    }
    if !document.potentials.is_empty() {
        insert_output(&mut outputs, "opcons.inp", opcons_inp_string(document)?);
    }
    insert_output(&mut outputs, "paths.inp", paths_inp_string(document)?);
    if should_write_single_scattering_paths(document) {
        insert_output(
            &mut outputs,
            "paths.dat",
            single_scattering_paths_dat_string(document)?,
        );
    }
    if !document.potentials.is_empty() {
        insert_output(&mut outputs, "pot.inp", pot_inp_string(document)?);
    }
    if let Some(input) = &document.reciprocal_input {
        insert_output(
            &mut outputs,
            "reciprocal.inp",
            reciprocal_input_string(input)?,
        );
    } else if !document.reciprocal {
        insert_output(&mut outputs, "reciprocal.inp", reciprocal_inp_string());
    }
    insert_output(&mut outputs, "rixs.inp", rixs_inp_string(document)?);
    insert_output(
        &mut outputs,
        "screen.inp",
        screen_inp_string_for_document(document)?,
    );
    insert_output(&mut outputs, "sfconv.inp", sfconv_inp_string(document)?);
    if let Some(dym_input) = &document.dym_input {
        insert_output(
            &mut outputs,
            dym_input.output_name.clone(),
            dym_input.text.clone(),
        );
    }
    if let Some(spring_input_text) = &document.spring_input_text {
        insert_output(&mut outputs, "spring.inp", spring_input_text.clone());
    }
    if !document.potentials.is_empty() {
        insert_output(&mut outputs, "xsph.inp", xsph_inp_string(document)?);
    }
    Ok(outputs)
}

fn insert_output(outputs: &mut TextOutputs, name: impl Into<TextOutputName>, text: String) {
    outputs.insert(name.into(), text);
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
    if let Some(nrixs) = document.nrixs.as_ref() {
        for row in &mut rows {
            let [x, y, z] = rotate_into_reference_frame([row.x, row.y, row.z], nrixs.qvec);
            row.x = x;
            row.y = y;
            row.z = z;
        }
    }
    Ok(rows)
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

fn should_write_single_scattering_paths(document: &FeffDocument) -> bool {
    !document.single_scattering_paths.is_empty() && control_flag(document, 3, 1) != 0
}

fn fms_flag(document: &FeffDocument) -> i32 {
    if uses_overlap_geometry(document) {
        0
    } else {
        control_flag(document, 2, 1)
    }
}

fn do_fms_flag(document: &FeffDocument) -> i32 {
    if uses_overlap_geometry(document) {
        0
    } else {
        i32::from(document.fms.is_some())
    }
}

fn path_flag(document: &FeffDocument) -> i32 {
    if uses_overlap_geometry(document) {
        0
    } else {
        control_flag(document, 3, 1)
    }
}

fn path_ms_flag(document: &FeffDocument) -> i32 {
    i32::from(!uses_overlap_geometry(document))
}

fn uses_overlap_geometry(document: &FeffDocument) -> bool {
    !document.overlap_shells.is_empty() && document.atoms.is_empty()
}

fn automatic_folp_flag(document: &FeffDocument) -> i32 {
    if document.overlap_factors.is_empty() {
        0
    } else {
        -1
    }
}

fn potential_overlap_factor(document: &FeffDocument, ipot: i32) -> f64 {
    document
        .overlap_factors
        .iter()
        .rev()
        .find(|factor| factor.potential_index == ipot)
        .map(|factor| factor.factor)
        .unwrap_or_else(|| {
            if document.overlap_factors.is_empty() {
                document.afolp
            } else {
                1.0
            }
        })
}

fn potential_ionization(document: &FeffDocument, ipot: i32) -> f64 {
    document
        .ionizations
        .iter()
        .rev()
        .find(|ionization| ionization.potential_index == ipot)
        .map(|ionization| ionization.value)
        .unwrap_or(0.0)
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
        .map(|potential| lmaxsc(document, potential).max(lmaxph(potential)))
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

fn potential_label(document: &FeffDocument, ipot: i32) -> Result<&str> {
    Ok(potential_for_ipot(document, ipot)?
        .tag
        .as_deref()
        .unwrap_or(""))
}

fn overlap_shell_count(document: &FeffDocument, ipot: i32) -> usize {
    document
        .overlap_shells
        .iter()
        .filter(|shell| shell.potential_index == ipot)
        .count()
}

fn write_overlap_shells(
    document: &FeffDocument,
    out: &mut impl std::fmt::Write,
    nph: i32,
) -> Result<()> {
    for ipot in 0..=nph {
        for shell in document
            .overlap_shells
            .iter()
            .filter(|shell| shell.potential_index == ipot)
        {
            writeln!(
                out,
                "{:5}{:5}{:13.5}",
                shell.neighbor_potential_index, shell.count, shell.distance
            )?;
        }
    }
    Ok(())
}

fn validate_single_scattering_path(
    document: &FeffDocument,
    path: &SingleScatteringPath,
) -> Result<()> {
    if path.index < 0 {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("SS path index must be non-negative: {}", path.index),
        });
    }
    if !path.degeneracy.is_finite() || !path.distance.is_finite() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "SS degeneracy and distance must be finite".to_string(),
        });
    }
    if path.distance < 0.0 {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("SS distance must be non-negative: {}", path.distance),
        });
    }
    Ok(())
}

fn absorber_potential(document: &FeffDocument) -> Result<&Potential> {
    potential_for_ipot(document, 0)
}

fn absorber_label(document: &FeffDocument) -> Result<String> {
    let potential = absorber_potential(document)?;
    if let Some(tag) = potential
        .tag
        .as_deref()
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
    {
        return Ok(tag.to_string());
    }

    if let Some(z) = potential.z {
        let atomic_number = usize::try_from(z).map_err(|_| IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: format!("invalid absorbing atomic number {z}"),
        })?;
        return atomic_symbol(atomic_number)
            .map(str::to_string)
            .map_err(|err| IoError::Parse {
                path: document.source.clone(),
                line: 0,
                message: err.to_string(),
            });
    }

    let label = potential.z_token.trim();
    if label.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "absorbing potential has no label".to_string(),
        });
    }
    Ok(label.to_string())
}

fn summary_edge_label(document: &FeffDocument) -> Result<&'static str> {
    let ihole = document_ihole(document)?;
    standard_edge_label(&ihole.to_string()).ok_or_else(|| IoError::Parse {
        path: document.source.clone(),
        line: 0,
        message: format!("unknown FEFF hole index {ihole}"),
    })
}

fn rdinp_spectroscopy_name(document: &FeffDocument) -> &'static str {
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

fn rdinp_corehole_name(nohole: i32) -> &'static str {
    match nohole {
        0 => "no",
        2 => "RPA",
        _ => "FSR",
    }
}

fn rdinp_feature_descriptions(document: &FeffDocument) -> Vec<String> {
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

fn rdinp_preamble_lines(document: &FeffDocument) -> Vec<String> {
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

fn rdinp_post_core_lines(document: &FeffDocument) -> Vec<String> {
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

fn rdinp_error_line<'a>(input: &'a FeffInput, error: &IoError) -> Option<&'a FeffLine> {
    let IoError::Parse { path, line, .. } = error else {
        return None;
    };

    input
        .lines
        .iter()
        .find(|input_line| input_line.location.line == *line && input_line.location.path == *path)
}

fn rdinp_error_preamble_lines(input: &FeffInput, failing_line: Option<&FeffLine>) -> Vec<String> {
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

fn rdinp_error_raw_line(failing_line: Option<&FeffLine>, error: &IoError) -> String {
    match failing_line {
        Some(line) => line.raw.chars().take(71).collect(),
        None => error.to_string().chars().take(71).collect(),
    }
}

fn rdinp_stdout_only_post_core_lines(document: &FeffDocument) -> Vec<String> {
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
    edge_index(label).ok_or_else(|| IoError::Parse {
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

fn lmaxsc(document: &FeffDocument, potential: &Potential) -> i32 {
    let raw = raw_lmaxsc(potential);
    if !document.unfreezef && raw > 2 {
        2
    } else {
        raw
    }
}

fn raw_lmaxsc(potential: &Potential) -> i32 {
    let default = match potential.z {
        Some(z) if z < 6 => 1,
        Some(z) if z < 55 => 2,
        Some(_) => 3,
        None => 0,
    };

    potential
        .lmax1
        .filter(|requested| *requested >= 0)
        .unwrap_or(default)
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
    if document.cfaverage.iphabs > 0 && potential.ipot == document.cfaverage.iphabs {
        return count.saturating_sub(1) as f64;
    }
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

fn core_hole_width_for_handoff(document: &FeffDocument, z: i32, ihole: i32) -> Result<f64> {
    let default_width = core_hole_width_for_document(document, z, ihole)?;
    Ok(match document.xsph_handoff.core_hole_width {
        Some(width) if width > 0.0 => width,
        Some(width) => default_width.min(width.abs()),
        None => default_width,
    })
}

fn distance_from(origin: &Atom, atom: &Atom) -> f64 {
    let dx = atom.x - origin.x;
    let dy = atom.y - origin.y;
    let dz = atom.z - origin.z;
    dx.hypot(dy).hypot(dz)
}

#[cfg(test)]
mod tests;
