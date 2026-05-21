use super::*;
use anyhow::{Context as _, ensure};
mod aliases;
mod cif;
mod core_cards;
mod dmdw_potentials;
mod generic_controls;
mod handoff_controls;
mod module_handoffs;
mod nrixs_mdff;
mod reciprocal;
mod single_scattering;
mod spectroscopy_validation;

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
