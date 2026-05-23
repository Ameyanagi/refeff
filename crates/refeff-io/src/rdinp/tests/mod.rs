use super::{
    atoms_dat_string, compton_inp_string, config_inp_string, density_inp_string,
    dimensions_dat_string, dmdw_inp_string, eels_inp_string, ff2x_inp_string, fms_inp_string,
    fullspectrum_inp_string_for_document, genfmt_inp_string, geom_dat_string, global_inp_string,
    grid_inp_string, hubbard_inp_string, ldos_inp_string, opcons_inp_string, paths_inp_string,
    pot_inp_string, rdinp_error_log_string, rdinp_log_dat, rdinp_log_dat_string,
    rdinp_stdout_string, reciprocal_inp_string, rixs_inp_string, screen_inp_string_for_document,
    sfconv_inp_string, single_scattering_paths_dat_string, text_outputs, xsph_inp_string,
};
use crate::global_input::GlobalInput;
use crate::{FeffDocument, FeffInput, IoError, Result, parse_log_dat, parse_paths_dat};

mod aliases_xsph;
mod basic_outputs;
mod dmdw_rixs;
mod global_geometry;
mod logs_errors;
mod module_controls;
mod potential_path;
mod structure_compton;
