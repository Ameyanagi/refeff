use std::path::Path;

use anyhow::{Context, Result};
use refeff_io::{potential_dat_outputs_from_bins, read_apot_bin, read_pot_bin};

use crate::work_dir_for_input;

/// Run FEFF `wpot`-compatible potential output generation beside an input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Write `potNN.dat` files from `pot.bin` and `apot.bin` in a work directory.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let pot_path = work_dir.join("pot.bin");
    let apot_path = work_dir.join("apot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let apot = read_apot_bin(&apot_path)
        .with_context(|| format!("failed to read {}", apot_path.display()))?;
    let outputs = potential_dat_outputs_from_bins(&pot, &apot)
        .context("failed to render FEFF wpot potential outputs")?;
    let count = outputs.len();
    for (name, content) in outputs {
        let output_path = work_dir.join(&name);
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    Ok(count)
}
