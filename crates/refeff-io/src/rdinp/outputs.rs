//! Aggregate writer for FEFF `rdinp` text outputs.

use crate::Result;
use crate::control_input::{band_input_string, reciprocal_input_string};
use crate::model::FeffDocument;

use super::helpers::should_write_single_scattering_paths;
use super::{
    TextOutputName, TextOutputs, atoms_dat_string, compton_inp_string, config_inp_string,
    crpa_inp_string, density_inp_string, dimensions_dat_string, dmdw_inp_string, eels_inp_string,
    ff2x_inp_string, fms_inp_string, fullspectrum_inp_string_for_document, genfmt_inp_string,
    geom_dat_string, global_inp_string, grid_inp_string, hubbard_inp_string, ldos_inp_string,
    mdff_inp_string, opcons_inp_string, paths_inp_string, pot_inp_string, reciprocal_inp_string,
    rixs_inp_string, screen_inp_string_for_document, sfconv_inp_string,
    single_scattering_paths_dat_string, xsph_inp_string,
};

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
