//! Writers for the first `rdinp`-level compatibility outputs.
//!
//! The full FEFF `rdinp` module emits many module input files. This module
//! starts with `atoms.dat`, which is the structural bridge consumed by later
//! FEFF modules, and will grow as the port advances.

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::config_input::config_inp_lines_string;
use crate::control_input::{fullspectrum_input_string, opcons_input_string};
use crate::input::{FeffInput, FeffLine, LineKind};
use crate::log_dat::{LogDatData, log_dat_string as render_log_dat_string};
use crate::model::FeffDocument;
use crate::screen_input::screen_input_string;
use crate::{IoError, Result};
use num_complex::Complex64;
use refeff_core::{
    core_hole_width_ev, edge_index, normalize_vector, nrixs_qtrig, rotate_into_reference_frame,
    vector_norm,
};

const FEFF_VERSION: &str = "FEFF 10.0.0";

/// File name key for a FEFF `rdinp` text output.
pub type TextOutputName = Cow<'static, str>;

/// Ordered text outputs rendered by the FEFF `rdinp` compatibility stage.
pub type TextOutputs = BTreeMap<TextOutputName, String>;

mod control_inputs;
mod geometry;
mod helpers;
mod log;
mod log_helpers;
mod module_inputs;
mod outputs;
mod structure;

use geometry::{geometry_model_atoms, geometry_rows};
use helpers::{
    absorber_label, absorber_potential, automatic_folp_flag, control_flag,
    core_hole_width_for_handoff, debye_values, dimensions_values, distance_from, do_fms_flag,
    document_ihole, fixed_a6, fixed_title, fms_flag, fortran_bool, interstitial_volume, lmaxph,
    lmaxsc, max_interatomic_distance, nearest_nonabsorber_distance, nph, output_ispec,
    overlap_shell_count, path_flag, path_ms_flag, path_rmax, potential_for_ipot,
    potential_ionization, potential_label, potential_overlap_factor, print_flag, spinph,
    validate_single_scattering_path, write_i4_list, write_overlap_shells, xnatph,
};
use log_helpers::{
    rdinp_corehole_name, rdinp_error_line, rdinp_error_preamble_lines, rdinp_error_raw_line,
    rdinp_feature_descriptions, rdinp_post_core_lines, rdinp_preamble_lines,
    rdinp_spectroscopy_name, rdinp_stdout_only_post_core_lines, summary_edge_label,
};

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
pub use outputs::text_outputs;
pub use structure::{atoms_dat_string, dimensions_dat_string, geom_dat_string, write_atoms_dat};

#[cfg(test)]
mod tests;
