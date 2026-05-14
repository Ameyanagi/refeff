//! Writers for the first `rdinp`-level compatibility outputs.
//!
//! The full FEFF `rdinp` module emits many module input files. This module
//! starts with `atoms.dat`, which is the structural bridge consumed by later
//! FEFF modules, and will grow as the port advances.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;

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
    if !document.egrid_records.is_empty() {
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

/// Render FEFF-compatible `config.inp` content from `CONFIG card` payload rows.
pub fn config_inp_string(document: &FeffDocument) -> Result<String> {
    config_inp_lines_string(&document.config_records)
}

/// Render FEFF-compatible `fullspectrum.inp` content with current defaults.
#[must_use]
pub fn fullspectrum_inp_string() -> String {
    " mFullSpectrum\n           0\n".to_string()
}

/// Render FEFF-compatible `fullspectrum.inp` content from `FULLSPECTRUM`.
pub fn fullspectrum_inp_string_for_document(document: &FeffDocument) -> Result<String> {
    fullspectrum_input_string(&document.full_spectrum_input)
}

/// Render FEFF-compatible default `eels.inp` content.
pub fn eels_inp_string(document: &FeffDocument) -> Result<String> {
    let eels = document.eels;
    let beam_direction = if eels.enabled {
        eels.beam_direction
    } else {
        document
            .nrixs
            .as_ref()
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

/// Render FEFF-compatible `opcons.inp` content.
pub fn opcons_inp_string(document: &FeffDocument) -> Result<String> {
    opcons_input_string(&document.opcons_input)
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

/// Render FEFF-compatible `screen.inp` content from parsed `SCREEN` cards.
pub fn screen_inp_string_for_document(document: &FeffDocument) -> Result<String> {
    screen_input_string(&document.screen_input)
}

/// Render FEFF-compatible `density.inp` content from a `DENSITY` block.
pub fn density_inp_string(document: &FeffDocument) -> Result<String> {
    if document.density_records.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write density.inp without DENSITY payload rows".to_string(),
        });
    }

    let mut out = String::new();
    for record in &document.density_records {
        writeln!(out, "{record}")?;
    }
    Ok(out)
}

/// Render FEFF-compatible `grid.inp` content from an `EGRID` block.
pub fn grid_inp_string(document: &FeffDocument) -> Result<String> {
    if document.egrid_records.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write grid.inp without EGRID payload rows".to_string(),
        });
    }

    let mut out = String::new();
    for record in &document.egrid_records {
        writeln!(out, " {record} ")?;
    }
    Ok(out)
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
    let nrixs = document.nrixs.as_ref();
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
    let polarization_tensor = global_polarization_tensor(document, nrixs.is_some())?;
    let le2 = if let Some(nrixs) = nrixs {
        nrixs.lj
    } else if !document.eels.enabled
        && document.ipol == 1
        && vector_norm(document.incidence_vector) == 0.0
    {
        0
    } else {
        document.le2
    };

    writeln!(out, " nabs, iphabs - CFAVERAGE data")?;
    writeln!(
        out,
        "{:8}{:8}{:13.5}",
        document.cfaverage.nabs, document.cfaverage.iphabs, document.cfaverage.rclabs
    )?;
    writeln!(
        out,
        " ipol, ispin, le2, elpty, angks, l2lp, do_nrixs, ldecmx, lj"
    )?;
    writeln!(
        out,
        "{:5}{:5}{:5}{:12.4}{:12.4}{:5}{:5}{:5}{:5}",
        ipol,
        document.spin,
        le2,
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
    let mixdff = global_mixdff(document);
    writeln!(
        out,
        "{:12}{:12} {} {}",
        nrixs.map(|nrixs| nrixs.nq).unwrap_or(0),
        document.mdff.imdff,
        fortran_bool(nrixs.map(|nrixs| nrixs.qaverage).unwrap_or(true)),
        fortran_bool(mixdff)
    )?;
    writeln!(
        out,
        " q-vectors : qx, qy, qz, q(norm), weight, qcosth, qsinth, qcosfi, qsinfi"
    )?;
    if let Some(nrixs) = nrixs {
        for q_vector in &nrixs.q_vectors {
            let qtrig = nrixs_qtrig(q_vector.vector, q_vector.norm);
            writeln!(
                out,
                "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
                q_vector.vector[0],
                q_vector.vector[1],
                q_vector.vector[2],
                q_vector.norm,
                q_vector.weight[0],
                q_vector.weight[1],
                qtrig[0],
                qtrig[1],
                qtrig[2],
                qtrig[3]
            )?;
        }
    }
    if mixdff {
        let Some(nrixs) = nrixs else {
            return Err(IoError::Parse {
                path: document.source.clone(),
                line: 0,
                message: "MDFF mixdff requires NRIXS".to_string(),
            });
        };
        writeln!(out, "    qqmdff,   cos<q,q'>")?;
        write!(out, "{:22.16}", document.mdff.qqmdff)?;
        for value in global_mdff_cosines(nrixs) {
            write!(out, "{value:22.16}")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

fn global_mixdff(document: &FeffDocument) -> bool {
    document.nrixs.is_some() && matches!(document.mdff.imdff, 1 | 2)
}

fn global_mdff_cosines(nrixs: &crate::model::Nrixs) -> Vec<f64> {
    nrixs
        .q_vectors
        .iter()
        .flat_map(|left| {
            nrixs.q_vectors.iter().map(move |right| {
                let norm_product = left.norm * right.norm;
                let normalized_dot = if norm_product > 0.0 {
                    left.vector
                        .iter()
                        .zip(right.vector)
                        .map(|(left, right)| *left * right)
                        .sum::<f64>()
                        / norm_product
                } else {
                    0.0
                };
                (std::f64::consts::PI / 180.0 * normalized_dot).cos()
            })
        })
        .collect()
}

fn global_polarization_tensor(document: &FeffDocument, has_nrixs: bool) -> Result<[[f64; 6]; 3]> {
    if has_nrixs {
        return Ok([
            [0.5, -0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        ]);
    }
    if document.ipol == 2 {
        return Ok([
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
    }
    if document.eels.enabled {
        return Ok([[0.0; 6]; 3]);
    }
    if document.ipol == 1 {
        return linear_polarization_tensor(
            document.polarization_vector,
            document.incidence_vector,
            document.ellipticity,
        );
    }
    Ok(averaged_polarization_tensor())
}

fn averaged_polarization_tensor() -> [[f64; 6]; 3] {
    [
        [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0],
    ]
}

fn linear_polarization_tensor(
    polarization: [f64; 3],
    incidence: [f64; 3],
    ellipticity: f64,
) -> Result<[[f64; 6]; 3]> {
    let incidence_norm = vector_norm(incidence);
    let rotated_polarization = if incidence_norm > 0.0 {
        rotate_into_reference_frame(polarization, incidence)
    } else {
        polarization
    };
    let rotated_incidence = if incidence_norm > 0.0 {
        normalize_vector(rotate_into_reference_frame(incidence, incidence))
    } else {
        [0.0; 3]
    };

    let mut evec = normalize_checked_polarization(rotated_polarization)?;
    let mut effective_ellipticity = ellipticity;
    if incidence_norm > 0.0 {
        let dot = dot_product(evec, rotated_incidence);
        if dot.abs() > 0.9 {
            return Err(IoError::InvalidPolarizationGeometry { dot });
        }
        if dot.abs() > 0.00001 {
            evec = normalize_checked_polarization([
                evec[0] - dot * rotated_incidence[0],
                evec[1] - dot * rotated_incidence[1],
                evec[2] - dot * rotated_incidence[2],
            ])?;
        }
    } else {
        effective_ellipticity = 0.0;
    }

    let e2 = cross_product(rotated_incidence, evec);
    let positive = [
        Complex64::new(evec[0], effective_ellipticity * e2[0]),
        Complex64::new(evec[1], effective_ellipticity * e2[1]),
        Complex64::new(evec[2], effective_ellipticity * e2[2]),
    ];
    let negative = [
        Complex64::new(evec[0], -effective_ellipticity * e2[0]),
        Complex64::new(evec[1], -effective_ellipticity * e2[1]),
        Complex64::new(evec[2], -effective_ellipticity * e2[2]),
    ];
    let eps = spherical_components(positive);
    let epc = spherical_components(negative);
    let scale = 1.0 / (1.0 + effective_ellipticity * effective_ellipticity) / 2.0;
    let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    for row_magnetic in -1..=1 {
        for column_magnetic in -1..=1 {
            let sign = if column_magnetic % 2 == 0 { 1.0 } else { -1.0 };
            let row = tensor_index(row_magnetic);
            let column = tensor_index(column_magnetic);
            tensor[row][column] = sign
                * (epc[column] * eps[tensor_index(-row_magnetic)]
                    + eps[column] * epc[tensor_index(-row_magnetic)])
                * scale;
        }
    }
    Ok(polarization_rows(tensor))
}

fn normalize_checked_polarization(vector: [f64; 3]) -> Result<[f64; 3]> {
    let norm = vector_norm(vector);
    if norm <= 0.000001 {
        return Err(IoError::InvalidPolarizationVector { norm });
    }
    Ok([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

fn spherical_components(vector: [Complex64; 3]) -> [Complex64; 3] {
    let root_half = 1.0 / 2.0_f64.sqrt();
    let imaginary = Complex64::new(0.0, 1.0);
    [
        (vector[0] - imaginary * vector[1]) * root_half,
        vector[2],
        -(vector[0] + imaginary * vector[1]) * root_half,
    ]
}

fn polarization_rows(tensor: [[Complex64; 3]; 3]) -> [[f64; 6]; 3] {
    let mut rows = [[0.0; 6]; 3];
    for row in 0..3 {
        for column in 0..3 {
            rows[row][2 * column] = tensor[row][column].re;
            rows[row][2 * column + 1] = tensor[row][column].im;
        }
    }
    rows
}

fn tensor_index(magnetic: i32) -> usize {
    match magnetic {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 0,
    }
}

fn dot_product(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross_product(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
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
mod tests {
    use crate::global_input::GlobalInput;
    use crate::{FeffDocument, FeffInput, IoError, Result, parse_log_dat, parse_paths_dat};

    use super::{
        atoms_dat_string, compton_inp_string, config_inp_string, density_inp_string,
        dimensions_dat_string, dmdw_inp_string, eels_inp_string, ff2x_inp_string, fms_inp_string,
        fullspectrum_inp_string_for_document, genfmt_inp_string, geom_dat_string,
        global_inp_string, grid_inp_string, hubbard_inp_string, ldos_inp_string, opcons_inp_string,
        paths_inp_string, pot_inp_string, rdinp_error_log_string, rdinp_log_dat,
        rdinp_log_dat_string, rdinp_stdout_string, reciprocal_inp_string, rixs_inp_string,
        screen_inp_string_for_document, sfconv_inp_string, single_scattering_paths_dat_string,
        text_outputs, xsph_inp_string,
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
    fn writes_config_inp_from_config_card_payload() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONFIG card 1
0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let config = config_inp_string(&doc)?;

        assert_eq!(config.lines().next().map(str::len), Some(150));
        assert!(config.starts_with("0 Cu 1s -2 2s -2 2p -2 -4"));
        Ok(())
    }

    #[test]
    fn writes_configuration_alias_payload_into_config_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONF card 1
2 1 0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;
        let config = outputs.get("config.inp").ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "missing config.inp".to_string(),
        })?;

        assert_eq!(config.lines().next().map(str::len), Some(150));
        assert!(config.starts_with("2 1 0"));
        Ok(())
    }

    #[test]
    fn writes_grid_inp_from_egrid_payload_tokens() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EGRID
e_grid    -15  -1.0  1.0
e_grid  last 10.0 0.1
k_grid  last 5.0 0.05
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(
            doc.egrid_records,
            [
                "e_grid -15 -1.0 1.0",
                "e_grid last 10.0 0.1",
                "k_grid last 5.0 0.05"
            ]
        );

        assert_eq!(
            grid_inp_string(&doc)?,
            " e_grid -15 -1.0 1.0 \n e_grid last 10.0 0.1 \n k_grid last 5.0 0.05 \n"
        );
        Ok(())
    }

    #[test]
    fn writes_density_inp_only_from_density_block() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DENSITY
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(
            density_inp_string(&doc)?,
            "line line.dat 0.0 0.0 0.0 core\n1.0 0.0 0.0 101\n"
        );
        assert!(text_outputs(&doc)?.contains_key("density.inp"));

        let input_without_density = FeffInput::parse_str("feff.inp", "END\n")?;
        let doc_without_density = FeffDocument::from_input(&input_without_density)?;
        assert!(!text_outputs(&doc_without_density)?.contains_key("density.inp"));
        Ok(())
    }

    #[test]
    fn writes_density_alias_payload_into_density_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DENS
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;

        assert_eq!(
            outputs.get("density.inp").map(String::as_str),
            Some("line line.dat 0.0 0.0 0.0 core\n1.0 0.0 0.0 101\n")
        );
        Ok(())
    }

    #[test]
    fn writes_single_scattering_paths_dat_from_ss_cards() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE SS smoke
CONTROL 1 1 1 1 1 1
RPATH 6.0
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 29 1 48 5.98
SS 30 1 2 8.0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let paths = single_scattering_paths_dat_string(&doc)?;

        assert_eq!(
            paths,
            concat!(
                " SS smoke\n",
                " Single scattering paths from ss lines cards in feff input\n",
                " -----------------------------------------------------------------------\n",
                "  29   2  48.000  index,nleg,degeneracy,r=  5.9800\n",
                " single scattering\n",
                "    5.980000    0.000000    0.000000   1 'Cu1   '\n",
                "    0.000000    0.000000    0.000000   0 'Cu0   '  x,y,z,ipot\n",
            )
        );

        let parsed = parse_paths_dat(&paths)?;
        assert_eq!(parsed.paths.len(), 1);
        assert_eq!(parsed.paths[0].index, 29);
        assert_eq!(parsed.paths[0].atoms[0].label, "Cu1");

        let outputs = text_outputs(&doc)?;
        assert_eq!(
            outputs.get("paths.dat").map(String::as_str),
            Some(paths.as_str())
        );
        Ok(())
    }

    #[test]
    fn omits_single_scattering_paths_dat_when_path_module_is_disabled() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONTROL 1 1 1 0 1 1
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 1 1 1 2.0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert!(!text_outputs(&doc)?.contains_key("paths.dat"));
        Ok(())
    }

    #[test]
    fn writes_overlap_geometry_into_module_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE SS smoke
CONTROL 1 1 1 1 1 1
RPATH 6.0
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 29 1 48 5.98
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.contains(concat!(
            "OVERLAP option: novr(iph)\n",
            "   1   1\n",
            " iphovr  nnovr rovr \n",
            "    1   12      2.55266\n",
            "    0   12      2.55266\n",
            "ChSh_Type:\n",
        )));
        assert!(paths_inp_string(&doc)?.starts_with(concat!(
            "mpath, ms, nncrit, nlegxx, ipr4\n",
            "   0   0   0   7   0\n",
        )));
        assert!(
            fms_inp_string(&doc)?.starts_with(concat!("mfms, idwopt, minv\n", "   0  -1   0\n",))
        );
        Ok(())
    }

    #[test]
    fn writes_manual_folp_factors_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
AFOLP 1.30
FOLP 1 1.2
FOLP 2 0.8
POTENTIALS
0 29 Cu0
1 29 Cu1
2 1 H
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.starts_with(concat!(
            "mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc\n",
            "   1   2   1   1   0  -1   0   0  11\n",
        )));
        assert!(pot.contains(concat!(
            " iz, lmaxsc, xnatph, xion, folp\n",
            "   29    2        1.0000000000        0.0000000000        1.0000000000\n",
            "   29    2        1.0000000000        0.0000000000        1.2000000000\n",
            "    1    1        1.0000000000        0.0000000000        0.8000000000\n",
        )));

        let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
        assert_eq!(parsed.control.iafolp, -1);
        assert_eq!(parsed.potentials[0].folp, 1.0);
        assert_eq!(parsed.potentials[1].folp, 1.2);
        assert_eq!(parsed.potentials[2].folp, 0.8);
        assert_eq!(crate::pot_input_string(&parsed)?, pot);
        Ok(())
    }

    #[test]
    fn writes_ionization_values_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ION 1 0.2
ION 2 -0.1
POTENTIALS
0 29 Cu0
1 29 Cu1
2 8 O
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.contains(concat!(
            " iz, lmaxsc, xnatph, xion, folp\n",
            "   29    2        1.0000000000        0.0000000000        1.1500000000\n",
            "   29    2        1.0000000000        0.2000000000        1.1500000000\n",
            "    8    2        1.0000000000       -0.1000000000        1.1500000000\n",
        )));

        let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
        assert_eq!(parsed.potentials[0].xion, 0.0);
        assert_eq!(parsed.potentials[1].xion, 0.2);
        assert_eq!(parsed.potentials[2].xion, -0.1);
        assert_eq!(crate::pot_input_string(&parsed)?, pot);
        Ok(())
    }

    #[test]
    fn writes_interstitial_alias_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
INTE 1 1.25
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 0 Cu0
2.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot_text = pot_inp_string(&doc)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_text)?;

        assert_eq!(pot.run.inters, 1);
        assert_eq!(pot.scattering.totvol, 20.0);
        assert_eq!(crate::pot_input_string(&pot)?, pot_text);
        Ok(())
    }

    #[test]
    fn writes_hubbard_alias_into_hubbard_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
HUBB 3.0 0.5 -0.1 2
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let hubbard_text = hubbard_inp_string(&doc);
        let hubbard = crate::HubbardInput::parse_str("hubbard.inp", &hubbard_text)?;

        assert_eq!(hubbard.i_hubbard, 2);
        assert_eq!(hubbard.mldos_hubb, 2);
        assert_eq!(hubbard.u, 3.0);
        assert_eq!(hubbard.j, 0.5);
        assert_eq!(hubbard.fermi_shift, -0.1);
        assert_eq!(hubbard.l, 2);
        assert_eq!(crate::hubbard_input_string(&hubbard)?, hubbard_text);
        Ok(())
    }

    #[test]
    fn writes_jump_removal_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
JUMPRM
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.contains(concat!(
            "nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf\n",
            "   1  -1   1   0   0   0   0   0\n",
        )));

        let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
        assert_eq!(parsed.run.jumprm, 1);
        assert_eq!(crate::pot_input_string(&parsed)?, pot);
        Ok(())
    }

    #[test]
    fn writes_external_potential_restart_switches_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EXTPOT
RESTART
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.contains(concat!(
            "ExternalPot switch, StartFromFile switch\n",
            " T T\n",
        )));

        let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
        assert!(parsed.external_pot);
        assert!(parsed.start_from_file);
        assert_eq!(crate::pot_input_string(&parsed)?, pot);
        Ok(())
    }

    #[test]
    fn writes_chemical_shift_type_into_pot_and_xsph_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CHSHIFT 3
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;
        let xsph = xsph_inp_string(&doc)?;

        assert!(pot.contains(concat!("ChSh_Type:\n", "   3\n")));
        assert!(xsph.contains(concat!("ChSh_Type:\n", "   3\n")));

        let parsed_pot = crate::PotInput::parse_str("pot.inp", &pot)?;
        let parsed_xsph = crate::XsphInput::parse_str("xsph.inp", &xsph)?;
        assert_eq!(parsed_pot.chsh_type, 3);
        assert_eq!(parsed_xsph.chsh_type, 3);
        assert_eq!(crate::pot_input_string(&parsed_pot)?, pot);
        assert_eq!(crate::xsph_input_string(&parsed_xsph)?, xsph);
        Ok(())
    }

    #[test]
    fn writes_corval_and_highz_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CORVAL -120
HIGHZ
WARNION
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;

        assert!(pot.contains("gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin\n"));
        assert!(pot.contains("   -120.00000\n"));
        assert!(pot.contains(concat!("FiniteNucleus, WarnIon\n", " T T\n",)));

        let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
        assert_eq!(parsed.scattering.corval_emin, -120.0);
        assert!(parsed.finite_nucleus);
        assert!(parsed.warn_ion);
        assert_eq!(crate::pot_input_string(&parsed)?, pot);
        Ok(())
    }

    #[test]
    fn writes_warn_alias_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
WARN
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot_text = pot_inp_string(&doc)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_text)?;

        assert!(pot.warn_ion);
        assert_eq!(crate::pot_input_string(&pot)?, pot_text);
        Ok(())
    }

    #[test]
    fn writes_scf_tail_controls_into_pot_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCFTH 1 6.5 640 80 5e-5
SCFR 2.0 4
TOLS 0.1 0.002 0.0003
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = pot_inp_string(&doc)?;
        let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;

        assert_eq!(parsed.thermal.iscfth, 1);
        assert_eq!(parsed.thermal.emaxscf, 6.5);
        assert_eq!(parsed.thermal.negrid, 640);
        assert_eq!(parsed.thermal.nmu, 80);
        assert_eq!(parsed.thermal.xntol, 5.0e-5);
        assert!(parsed.ramp.ramp_scf);
        assert_eq!(parsed.ramp.rfms_start, 2.0);
        assert_eq!(parsed.ramp.nramp, 4);
        assert_eq!(parsed.tolerances.tolmu, 0.001);
        assert_eq!(parsed.tolerances.tolq, 0.002);
        assert_eq!(parsed.tolerances.tolqp, 0.0003);
        assert_eq!(crate::pot_input_string(&parsed)?, pot);
        Ok(())
    }

    #[test]
    fn writes_scxc_into_module_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TEMP 0.25 12
SCXC 22
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot_text = pot_inp_string(&doc)?;
        let xsph_text = xsph_inp_string(&doc)?;
        let ldos_text = ldos_inp_string(&doc)?;
        let ff2x_text = ff2x_inp_string(&doc)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_text)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_text)?;
        let ldos = crate::LdosInput::parse_str("ldos.inp", &ldos_text)?;
        let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_text)?;

        assert_eq!(pot.control.iscfxc, 22);
        assert_eq!(xsph.control.iscfxc, 22);
        assert_eq!(ldos.control.iscfxc, 22);
        assert_eq!(xsph.electronic_temperature, 0.25);
        assert_eq!(ff2x.electronic_temperature, 0.25);
        assert_eq!(crate::pot_input_string(&pot)?, pot_text);
        assert_eq!(crate::xsph_input_string(&xsph)?, xsph_text);
        assert_eq!(crate::ldos_input_string(&ldos)?, ldos_text);
        assert_eq!(crate::ff2x_input_string(&ff2x)?, ff2x_text);
        Ok(())
    }

    #[test]
    fn writes_ff2x_convolution_and_damping_controls() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EXAFS 20.0
MBCONV
SIG2 0.012
SIG3 0.034 250
SIGGK 0.056
FMS 3.0
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let ff2x_text = ff2x_inp_string(&doc)?;
        let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_text)?;

        assert_eq!(ff2x.control.mbconv, 1);
        assert_eq!(ff2x.debye.sig2g, 0.012);
        assert_eq!(ff2x.debye.alphat, 0.034);
        assert_eq!(ff2x.debye.thetae, 250.0);
        assert_eq!(ff2x.debye.sig_gk, 0.056);
        assert_eq!(crate::ff2x_input_string(&ff2x)?, ff2x_text);
        assert!(fms_inp_string(&doc)?.contains(concat!(
            "tk, thetad, sig2g\n",
            "      0.00000      0.00000      0.01200\n",
        )));
        Ok(())
    }

    #[test]
    fn writes_genfmt_and_real_phase_switches() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
IORDER 4
POLARIZATION 1 0 0
RPHASES
NSTAR
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let genfmt_text = genfmt_inp_string(&doc)?;
        let genfmt = crate::GenfmtInput::parse_str("genfmt.inp", &genfmt_text)?;
        let xsph_text = xsph_inp_string(&doc)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_text)?;

        assert_eq!(genfmt.control.iorder, 4);
        assert!(genfmt.control.wnstar);
        assert_eq!(crate::genfmt_input_string(&genfmt)?, genfmt_text);
        assert_eq!(xsph.control.lreal, 2);
        assert_eq!(crate::xsph_input_string(&xsph)?, xsph_text);
        assert!(
            rdinp_stdout_string(&doc)?.contains(
                " Real phase shifts only will be used.  FEFF results will be unreliable.\n"
            )
        );
        Ok(())
    }

    #[test]
    fn writes_path_symmetry_into_paths_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SYMMETRY 3
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let paths_text = paths_inp_string(&doc)?;
        let paths = crate::PathsInput::parse_str("paths.inp", &paths_text)?;

        assert_eq!(paths.ica, 3);
        assert_eq!(crate::paths_input_string(&paths)?, paths_text);
        assert!(
            rdinp_stdout_string(&doc)?
                .contains(" SYMMETRY CARD - fixing icase to    3 in module PATH.\n")
        );
        Ok(())
    }

    #[test]
    fn nrixs_overrides_path_symmetry_like_feff() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SYMMETRY 3
XANES
NRIXS 1 0.0 0.0 1.0
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let paths = crate::PathsInput::parse_str("paths.inp", &paths_inp_string(&doc)?)?;

        assert_eq!(doc.path_symmetry, 3);
        assert_eq!(paths.ica, 7);
        Ok(())
    }

    #[test]
    fn writes_nrixs_alias_into_global_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIX 1 0.0 0.0 2.0
LDEC 4
LJMAX 2
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let global_text = global_inp_string(&doc)?;
        let global = GlobalInput::parse_str("global.inp", &global_text)?;

        assert_eq!(global.control.do_nrixs, 1);
        assert_eq!(global.control.ldecmx, 4);
        assert_eq!(global.control.lj, 2);
        assert_eq!(global.control.l2lp, 30);
        assert_eq!(global.q_control.nq, 1);
        assert_eq!(global.q_vectors[0].q, [0.0, 0.0, 2.0]);
        assert_eq!(crate::global_input_string(&global)?, global_text);
        Ok(())
    }

    #[test]
    fn writes_bandstructure_into_band_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
BANDSTRUCTURE -5.0 10.0 0.25 2 64 T
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;
        let band_text = outputs.get("band.inp").ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "missing band.inp output".to_string(),
        })?;
        let band = crate::BandInput::parse_str("band.inp", band_text)?;

        assert_eq!(band.mband, 1);
        assert_eq!(band.energy_mesh.emin, -5.0);
        assert_eq!(band.energy_mesh.emax, 10.0);
        assert_eq!(band.energy_mesh.estep, 0.25);
        assert_eq!(band.ikpath, 2);
        assert_eq!(band.nkp, 64);
        assert!(band.freeprop);
        assert_eq!(crate::band_input_string(&band)?, *band_text);
        assert!(rdinp_stdout_string(&doc)?.contains("BANDSTRUCTURE card is experimental.\n"));
        Ok(())
    }

    #[test]
    fn writes_fullspectrum_switch_into_fullspectrum_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FULLSPECTRUM
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;
        let fullspectrum_text = outputs
            .get("fullspectrum.inp")
            .ok_or_else(|| IoError::Parse {
                path: "feff.inp".into(),
                line: 0,
                message: "missing fullspectrum.inp output".to_string(),
            })?;
        let fullspectrum =
            crate::FullSpectrumInput::parse_str("fullspectrum.inp", fullspectrum_text)?;

        assert_eq!(fullspectrum.m_full_spectrum, 1);
        assert_eq!(
            crate::fullspectrum_input_string(&fullspectrum)?,
            *fullspectrum_text
        );
        assert_eq!(
            fullspectrum_inp_string_for_document(&doc)?,
            *fullspectrum_text
        );
        Ok(())
    }

    #[test]
    fn writes_sfconv_alias_controls_into_sfconv_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
SO2C
SELF
SFSE 2.5
RCONV 10.25 longfilename.dat
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let sfconv_text = sfconv_inp_string(&doc)?;
        let sfconv = crate::SfconvInput::parse_str("sfconv.inp", &sfconv_text)?;

        assert_eq!(sfconv.control.msfconv, 1);
        assert_eq!(sfconv.control.ipse, 1);
        assert_eq!(sfconv.control.ipsk, 1);
        assert_eq!(sfconv.window.wsigk, 2.5);
        assert_eq!(sfconv.window.cen, 10.25);
        assert_eq!(sfconv.spectrum.ispec, 1);
        assert_eq!(sfconv.spectrum.ipr6, 0);
        assert_eq!(sfconv.cfname, "longfilename");
        assert_eq!(crate::sfconv_input_string(&sfconv)?, sfconv_text);
        Ok(())
    }

    #[test]
    fn writes_opcons_number_density_controls() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
OPCONS
NUMDENS 0 8.5
NUMDENS 2 4.25
PREPS
POTENTIALS
0 29 Cu0
1 29 Cu1
2 8 O
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let opcons_text = opcons_inp_string(&doc)?;
        let opcons = crate::OpconsInput::parse_str("opcons.inp", &opcons_text)?;

        assert!(opcons.run_opcons);
        assert!(opcons.print_eps);
        assert_eq!(opcons.number_densities, vec![8.5, -1.0, 4.25]);
        assert_eq!(crate::opcons_input_string(&opcons)?, opcons_text);
        Ok(())
    }

    #[test]
    fn writes_opcons_alias_controls_into_opcons_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
OPCO
NUMD 0 8.5
PREP
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let opcons_text = opcons_inp_string(&doc)?;
        let opcons = crate::OpconsInput::parse_str("opcons.inp", &opcons_text)?;

        assert!(opcons.run_opcons);
        assert!(opcons.print_eps);
        assert_eq!(opcons.number_densities, vec![8.5, -1.0]);
        assert_eq!(crate::opcons_input_string(&opcons)?, opcons_text);
        Ok(())
    }

    #[test]
    fn writes_screen_controls_into_screen_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EDGE K
SCREEN rfms 5.5
SCREEN ner 64.4
SCREEN eimax 3.25
SCREEN icore 2
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;
        let screen_text = outputs.get("screen.inp").ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "missing screen.inp output".to_string(),
        })?;
        let screen = crate::ScreenInput::parse_str("screen.inp", screen_text)?;

        assert_eq!(screen.rfms, 5.5);
        assert_eq!(screen.ner, 64);
        assert_eq!(screen.eimax, 3.25);
        assert_eq!(screen.icore, 2);
        assert_eq!(crate::screen_input_string(&screen)?, *screen_text);
        assert_eq!(screen_inp_string_for_document(&doc)?, *screen_text);
        assert!(
            rdinp_stdout_string(&doc)?.contains(":INFO  User provides options for screen.inp\n")
        );
        Ok(())
    }

    #[test]
    fn writes_reciprocal_inp_from_lattice_atoms() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 1000 0
TARGET 1
LATTICE P 2.456
0.86603 -0.5000 0.00000
0.00000 1.00000 0.00000
0.00000 0.00000 2.72638
ATOMS
0.00000 0.00000 0.68160 1 C1
0.00000 0.00000 2.04479 1 C1
0.57735 0.00000 0.68160 2 C2
0.28868 0.50000 2.04479 2 C2
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;
        let reciprocal = outputs
            .get("reciprocal.inp")
            .ok_or_else(|| IoError::Parse {
                path: "test".into(),
                line: 0,
                message: "missing reciprocal.inp".to_string(),
            })?;

        assert!(reciprocal.starts_with("ispace\n   0\n"));
        assert!(reciprocal.contains("      2.12697     -1.22800      0.00000\n"));
        assert!(reciprocal.contains(
            "        1000        1000           0           0           1           0\n"
        ));
        assert!(reciprocal.contains("           1           1           2           2\n"));
        Ok(())
    }

    #[test]
    fn writes_reciprocal_coordinates_one_like_feff() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE COORDINATES smoke
EDGE K
RECIPROCAL
KMESH 10 0
TARGET 1
FMS 2.0
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
COORDINATES 1
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;
        let geom = crate::GeomDat::parse_str("geom.dat", &geom_dat_string(&doc)?)?;
        let reciprocal_input = doc
            .reciprocal_input
            .as_ref()
            .ok_or_else(|| IoError::Parse {
                path: "feff.inp".into(),
                line: 0,
                message: "missing reciprocal input".to_string(),
            })?;
        let reciprocal = crate::reciprocal_input_string(reciprocal_input)?;

        assert_eq!(atoms.atoms[0].iph, 0);
        assert_eq!(atoms.atoms[1].distance, 1.0);
        assert_eq!(atoms.atoms[2].distance, 1.0);
        assert_eq!(geom.atoms[1].x, -1.0);
        assert_eq!(geom.atoms[2].x, 1.0);
        assert!(reciprocal.contains("      0.50000      0.00000      0.00000\n"));
        Ok(())
    }

    #[test]
    fn logs_real_and_reciprocal_cards_in_input_order() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
REAL
EDGE K
POTENTIALS
0 29 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;
        let expected_reciprocal = reciprocal_inp_string();

        assert!(!doc.reciprocal);
        assert!(doc.reciprocal_input.is_none());
        assert!(
            rdinp_stdout_string(&doc)?
                .contains("Working in reciprocal space.\nWorking in real space.\n")
        );
        assert_eq!(
            outputs.get("reciprocal.inp").map(String::as_str),
            Some(expected_reciprocal.as_str())
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
    fn writes_cfaverage_into_global_pot_and_geom_outputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE CFAVERAGE smoke
EDGE K
CFAVERAGE 1 0 0
POTENTIALS
1 29 Cu
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
2.0 0.0 0.0 1 Cu2
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let global = GlobalInput::parse_str("global.inp", &global_inp_string(&doc)?)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;
        let geom = crate::GeomDat::parse_str("geom.dat", &geom_dat_string(&doc)?)?;

        assert_eq!(global.cfaverage.nabs, 3);
        assert_eq!(global.cfaverage.iphabs, 1);
        assert_eq!(global.cfaverage.rclabs, 100000.0);
        assert_eq!(pot.control.nph, 1);
        assert_eq!(pot.potentials[0].z, 29);
        assert_eq!(pot.potentials[0].xnatph, 1.0);
        assert_eq!(pot.potentials[1].z, 29);
        assert_eq!(pot.potentials[1].xnatph, 2.0);
        assert_eq!(atoms.atoms[0].iph, 1);
        assert_eq!(geom.model_atoms, [1, 2]);
        assert_eq!(geom.atoms[0].iph, 0);
        assert_eq!(geom.atoms[1].iph, 1);
        Ok(())
    }

    #[test]
    fn writes_global_inp_linear_polarization_tensor() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POLARIZATION 1 0 0
MULTIPOLE 2
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let text = global_inp_string(&doc)?;
        let global = GlobalInput::parse_str("global.inp", &text)?;

        assert_eq!(global.control.ipol, 1);
        assert_eq!(global.control.le2, 0);
        assert_eq!(
            global.polarization_tensor,
            [
                [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
                [0.0; 6],
                [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn writes_spectroscopy_aliases_into_global_and_xsph_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EDGE K
XANE 20.0 1.25 0.2
POLA 1.0 0.0 0.0
ELLI 0.25 0.0 1.0 0.0
MULT 2 1
POTENTIALS
0 29 Cu
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let global = GlobalInput::parse_str("global.inp", &global_inp_string(&doc)?)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;

        assert_eq!(global.control.ipol, 1);
        assert_eq!(global.control.le2, 2);
        assert_eq!(global.control.l2lp, 1);
        assert_eq!(global.control.elpty, 0.25);
        assert_eq!(global.xivec, [0.0, 1.0, 0.0]);
        assert_eq!(global.norms.xivnorm, 1.0);
        assert_eq!(xsph.control.ispec, 1);
        assert_eq!(xsph.grid.xkstep, 1.25);
        assert_eq!(xsph.grid.xkmax, 20.0);
        assert_eq!(xsph.grid.vixan, 0.2);
        Ok(())
    }

    #[test]
    fn rejects_zero_length_global_polarization_vector() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POLARIZATION 0 0 0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        assert!(global_inp_string(&doc).is_err());
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
    fn nogeom_suppresses_geom_dat_output_like_feff() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
NOGEOM
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let outputs = text_outputs(&doc)?;

        assert!(doc.no_geom);
        assert!(outputs.contains_key("atoms.dat"));
        assert!(!outputs.contains_key("geom.dat"));
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
    fn writes_compton_aliases_into_compton_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
COMP 7.0 300 1
RHOZ
CGRI 12.0 20 21 22 23
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let compton_text = compton_inp_string(&doc)?;
        let compton = crate::ComptonInput::parse_str("compton.inp", &compton_text)?;

        assert!(compton.run);
        assert_eq!(compton.momentum.pqmax, 7.0);
        assert_eq!(compton.momentum.npq, 300);
        assert_eq!(compton.grid.ns, 20);
        assert_eq!(compton.grid.nphi, 21);
        assert_eq!(compton.grid.nz, 22);
        assert_eq!(compton.grid.nzp, 23);
        assert_eq!(compton.limits.zpmax, 12.0);
        assert!(compton.switches.jpq);
        assert!(compton.switches.rhozzp);
        assert!(compton.switches.force_recalc_jzzp);
        assert_eq!(crate::compton_input_string(&compton)?, compton_text);
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
    fn writes_block_alias_rows_into_structure_outputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;

        assert_eq!(pot.control.nph, 1);
        assert_eq!(pot.potentials.len(), 2);
        assert_eq!(pot.potentials[0].z, 29);
        assert_eq!(atoms.atoms.len(), 2);
        assert_eq!(atoms.atoms[1].iph, 1);
        Ok(())
    }

    #[test]
    fn writes_cif_equivalence_two_into_structure_outputs() -> Result<()> {
        let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
        let cif_path = temp.path().join("three-site.cif");
        std::fs::write(
            &cif_path,
            r#"
data_three_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1 0.0 0.0 0.0
H2 0.25 0.0 0.0
O1 0.5 0.5 0.5
"#,
        )
        .map_err(|source| IoError::io(&cif_path, source))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(
            &input_path,
            r#"
CIF three-site.cif
TARGET 3
EQUIVALENCE 2
FMS 4.0
EDGE K
XANES
END
"#,
        )
        .map_err(|source| IoError::io(&input_path, source))?;

        let input = FeffInput::parse_file(&input_path)?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;

        assert_eq!(pot.control.nph, 2);
        assert_eq!(pot.potentials.len(), 3);
        assert_eq!(pot.potentials[0].z, 8);
        assert_eq!(pot.potentials[1].z, 1);
        assert_eq!(pot.potentials[1].xnatph, 200.0);
        assert_eq!(pot.potentials[2].z, 8);
        assert!(atoms.atoms.iter().any(|atom| atom.iph == 1));
        assert!(atoms.atoms.iter().all(|atom| atom.iph <= 2));
        Ok(())
    }

    #[test]
    fn writes_common_control_aliases_into_module_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITL Alias controls
EDGE K
CONT 1 0 1 0 1 0
PRIN 3 4 5 6 7 8
EXCH 2 1.25 0.5 9
CORR -1.5 0.75
RGRI 0.03
CORE NONE
UNFR
ABSO
POTENTIALS
0 29 Cu
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;
        let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_inp_string(&doc)?)?;

        assert_eq!(pot.titles, ["Alias controls"]);
        assert_eq!(pot.control.ipr1, 3);
        assert_eq!(pot.control.ixc, 2);
        assert_eq!(pot.run.nohole, 0);
        assert_eq!(pot.run.iunf, 1);
        assert_eq!(pot.scattering.rgrd, 0.03);
        assert_eq!(xsph.control.ipr2, 4);
        assert_eq!(xsph.control.ixc, 2);
        assert_eq!(xsph.control.ixc0, 9);
        assert_eq!(xsph.vr0, 1.25);
        assert_eq!(xsph.vi0, 0.5);
        assert_eq!(xsph.grid.rgrd, 0.03);
        assert_eq!(ff2x.control.ipr6, 8);
        assert_eq!(ff2x.control.absolu, 1);
        assert_eq!(ff2x.corrections.vrcorr, -1.5);
        assert_eq!(ff2x.corrections.vicorr, 0.75);
        Ok(())
    }

    #[test]
    fn expands_feff7_control_and_print_into_module_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE FEFF7 control print compatibility
EDGE K
XANES
CONTROL 0 1 0 1
PRINT 5 2 1 4
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;
        let fms = crate::FmsInput::parse_str("fms.inp", &fms_inp_string(&doc)?)?;
        let paths = crate::PathsInput::parse_str("paths.inp", &paths_inp_string(&doc)?)?;
        let genfmt = crate::GenfmtInput::parse_str("genfmt.inp", &genfmt_inp_string(&doc)?)?;
        let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_inp_string(&doc)?)?;

        assert_eq!(pot.control.mpot, 0);
        assert_eq!(pot.control.ipr1, 5);
        assert_eq!(xsph.control.mphase, 0);
        assert_eq!(xsph.control.ipr2, 5);
        assert_eq!(fms.control.mfms, 0);
        assert_eq!(paths.control.mpath, 1);
        assert_eq!(paths.control.ipr4, 2);
        assert_eq!(genfmt.control.mfeff, 0);
        assert_eq!(genfmt.control.ipr5, 1);
        assert_eq!(ff2x.control.mchi, 1);
        assert_eq!(ff2x.control.ipr6, 4);
        Ok(())
    }

    #[test]
    fn writes_four_character_control_aliases_into_module_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITL Prefix aliases
EDGE K
HOLE 1 0.77
SCF 5.0 1 40 0.3
FMS 4.0 1 0 0.002 0.003 7.0
DEBY 190 315 0
RPAT 4.5
CRIT 3.1 2.2
PCRI 0.7 0.8
RMUL 2.0
AFOL 1.25
FOLP 1 1.35
PLAS 3 50
ELNE
100.0 1 1 1 1 1
MAGI 7112.0
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let fms = crate::FmsInput::parse_str("fms.inp", &fms_inp_string(&doc)?)?;
        let paths = crate::PathsInput::parse_str("paths.inp", &paths_inp_string(&doc)?)?;
        let eels = crate::EelsInput::parse_str("eels.inp", &eels_inp_string(&doc)?)?;

        assert_eq!(pot.titles, ["Prefix aliases"]);
        assert_eq!(pot.control.ihole, 1);
        assert_eq!(pot.scattering.rfms1, 10.0);
        assert_eq!(pot.potentials[1].folp, 1.35);
        assert_eq!(fms.cluster.rfms2, 8.0);
        assert_eq!(fms.cluster.rdirec, 7.0);
        assert_eq!(fms.debye.tk, 190.0);
        assert_eq!(fms.debye.thetad, 315.0);
        assert_eq!(paths.criteria.critpw, 2.2);
        assert_eq!(paths.criteria.pcritk, 0.7);
        assert_eq!(paths.criteria.pcrith, 0.8);
        assert_eq!(paths.criteria.rmax, 9.0);
        assert_eq!(paths.criteria.rfms2, 8.0);
        assert!(eels.calculate_elnes);
        assert_eq!(eels.magic, 1);
        assert_eq!(eels.magic_energy, 7112.0);
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
    fn writes_xsph_core_hole_controls_into_module_inputs() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
CHBROADENING 1
CHWIDTH 0.75
EPS0 -2.0
EGAP 1.25
SETEDGE
RLPRINT
ICORE 3
POTENTIALS
0 29 Cu
1 29 Cu
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;
        let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_inp_string(&doc)?)?;
        let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
        let log = rdinp_log_dat(&doc)?;

        assert_eq!(xsph.control.i_gamma_ch, 1);
        assert_eq!(xsph.control.i_core_state, 3);
        assert_eq!(xsph.grid.gamach, 0.75);
        assert_eq!(xsph.grid.eps0, -2.0);
        assert_eq!(xsph.grid.egap, 1.25);
        assert!(xsph.lopt);
        assert!(xsph.print_rl);
        assert_eq!(ff2x.control.i_gamma_ch, 1);
        assert_eq!(pot.scattering.gamach, 0.75);
        assert_eq!(log.core_hole_lifetime_ev, Some(0.75));
        Ok(())
    }

    #[test]
    fn writes_tdl_and_pmbse_controls_into_xsph_inp() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE TDLDA PMBSE smoke
EDGE K
TDLDA 7
PMBSE 3 4 5 6
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let text = xsph_inp_string(&doc)?;
        let xsph = crate::XsphInput::parse_str("xsph.inp", &text)?;

        assert_eq!(xsph.advanced.izstd, 1);
        assert_eq!(xsph.advanced.ifxc, 7);
        assert_eq!(xsph.advanced.ipmbse, 3);
        assert_eq!(xsph.advanced.itdlda, 2);
        assert_eq!(xsph.advanced.nonlocal, 4);
        assert_eq!(xsph.advanced.ibasis, 6);
        assert_eq!(crate::xsph_input_string(&xsph)?, text);
        Ok(())
    }

    #[test]
    fn writes_rdinp_log_dat_for_copper_exafs_summary() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
DEBYE 190 315 0
EDGE K
S02 1.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
LDOS -30 20 0.1
EXAFS 20.0
RPATH 5.5
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
        let log = rdinp_log_dat_string(&doc)?;

        assert_eq!(
            log,
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Core hole lifetime is   1.729 eV.\n",
                "Your calculation:\n",
                " Cu crystal\n",
                "Cu K edge EXAFS using FSR corehole.\n",
                "Using:     * Debye-Waller factors\n",
                "Using cards:   ATOMS CONTROL TITLE RPATH DEBYE PRINT POTENTIALS EXAFS EDGE LDOS S02\n",
                "\n",
            )
        );
        assert_eq!(parse_log_dat(&log)?.features, vec!["Debye-Waller factors"]);
        Ok(())
    }

    #[test]
    fn writes_rdinp_stdout_only_absorber_spin_default_diagnostic() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Gd_L1 hcp
XMCD
EDGE L1
SPIN 1
POTENTIALS
0 64 Gd
1 64 Gd
ATOMS
0.0 0.0 0.0 0 Gd0
1.0 0.0 0.0 1 Gd1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let log = rdinp_log_dat_string(&doc)?;
        let stdout = rdinp_stdout_string(&doc)?;

        assert!(!log.contains("\n           1\n"));
        assert!(stdout.contains("Core hole lifetime is   5.533 eV.\n           1\n"));
        assert!(stdout.contains("No spin set in POTENTIALS card. Using default spins:\n"));
        Ok(())
    }

    #[test]
    fn writes_rdinp_error_log_for_highz_template_failure() -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE test_element
HIGHZ
POTENTIALS
       0    XXX   Te
END
"#,
        )?;
        let error = FeffDocument::from_input(&input)
            .err()
            .ok_or_else(|| IoError::Parse {
                path: "feff.inp".into(),
                line: 0,
                message: "HIGHZ template should fail".to_string(),
            })?;

        assert_eq!(
            rdinp_error_log_string(&input, &error)?,
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Using finite nucleus.\n",
                " Error reading input, bad line follows:\n",
                " 0    XXX   Te\n",
                "RDINP fatal error.\n",
            )
        );
        Ok(())
    }

    #[test]
    fn writes_dmdw_inp_for_dynamical_matrix_debye() -> Result<()> {
        let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())
            .map_err(|source| IoError::io("feff.dym", source))?;
        let input = FeffInput::parse_str(
            &input_path,
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
    fn copies_dym_file_for_dynamical_matrix_debye() -> Result<()> {
        let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
        let input_path = temp.path().join("feff.inp");
        let dym_text = minimal_dym_text();
        std::fs::write(temp.path().join("custom.dym"), dym_text)
            .map_err(|source| IoError::io("custom.dym", source))?;
        let input = FeffInput::parse_str(
            &input_path,
            r#"
DEBYE 450 315 5 custom.dym 6 0 1
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        let dym_input = doc.dym_input.as_ref().ok_or_else(|| IoError::Parse {
            path: input_path.clone(),
            line: 0,
            message: "missing DMDW auxiliary".to_string(),
        })?;
        assert_eq!(dym_input.output_name, "custom.dym");
        assert_eq!(dym_input.text, dym_text);
        assert_eq!(
            text_outputs(&doc)?.get("custom.dym").map(String::as_str),
            Some(dym_text)
        );
        Ok(())
    }

    #[test]
    fn copies_spring_inp_for_emm_and_recursion_debye() -> Result<()> {
        let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
        let input_path = temp.path().join("feff.inp");
        let spring_text = concat!(
            "* res wmax dosfit acut\n",
            " VDOS 0.03 0.5 1\n",
            "\n",
            " STRETCHES\n",
            " 0 1 27.9 2.\n",
        );
        std::fs::write(temp.path().join("spring.inp"), spring_text)
            .map_err(|source| IoError::io("spring.inp", source))?;
        for idwopt in [1, 2] {
            let input =
                FeffInput::parse_str(&input_path, &format!("DEBYE 450 315 {idwopt}\nEND\n"))?;
            let doc = FeffDocument::from_input(&input)?;

            assert_eq!(doc.spring_input_text.as_deref(), Some(spring_text));
            assert_eq!(
                text_outputs(&doc)?.get("spring.inp").map(String::as_str),
                Some(spring_text)
            );
        }
        Ok(())
    }

    fn minimal_dym_text() -> &'static str {
        concat!(
            "    1\n",
            "    1\n",
            "   29\n",
            "   63.546000\n",
            "    0.00000000    0.00000000    0.00000000\n",
            "    1    1\n",
            "  1.000000E+00  0.000000E+00  0.000000E+00\n",
            "  0.000000E+00  1.000000E+00  0.000000E+00\n",
            "  0.000000E+00  0.000000E+00  1.000000E+00\n",
        )
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
