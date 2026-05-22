use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

use super::common::{DMDW_EGRID_INFO_PATH, numeric_values, parse_error};
use super::types::DmdwEnergyGridInfo;
use super::validate::validate_dmdw_egrid_info;

/// Parse FEFF `dmdw_Egrid.info` text.
pub fn parse_dmdw_egrid_info(text: &str) -> Result<DmdwEnergyGridInfo> {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.len() < 4 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            lines.len() + 1,
            format!(
                "dmdw_Egrid.info has {} line(s), expected at least 4",
                lines.len()
            ),
        );
    }

    let low_high = numeric_values(DMDW_EGRID_INFO_PATH, 2, lines[1])?;
    let step_w0 = numeric_values(DMDW_EGRID_INFO_PATH, 3, lines[2])?;
    let electron_selected = numeric_values(DMDW_EGRID_INFO_PATH, 4, lines[3])?;
    if low_high.len() != 2 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            2,
            format!(
                "low/high line has {} numeric value(s), expected 2",
                low_high.len()
            ),
        );
    }
    if step_w0.len() != 2 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            3,
            format!(
                "step/w0 line has {} numeric value(s), expected 2",
                step_w0.len()
            ),
        );
    }
    if electron_selected.len() != 2 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            4,
            format!(
                "electron/grid line has {} numeric value(s), expected 2",
                electron_selected.len()
            ),
        );
    }

    let data = DmdwEnergyGridInfo {
        low_energy_mev: low_high[0],
        high_energy_mev: low_high[1],
        step_mev: step_w0[0],
        characteristic_energy_mev: step_w0[1],
        electron_energy_mev: electron_selected[0],
        selected_energy_mev: electron_selected[1],
    };
    validate_dmdw_egrid_info(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_Egrid.info` text.
pub fn dmdw_egrid_info_string(data: &DmdwEnergyGridInfo) -> Result<String> {
    validate_dmdw_egrid_info(data)?;

    let mut out = String::new();
    writeln!(out, "#  Energies printed in meV")?;
    writeln!(
        out,
        "#  lowE {:10.3} highE {:10.3}",
        data.low_energy_mev, data.high_energy_mev
    )?;
    writeln!(
        out,
        "#  dE  = {:10.3} w0 = {:10.3}",
        data.step_mev, data.characteristic_energy_mev
    )?;
    writeln!(
        out,
        "#  Ek  = {:10.3} --> E = {:10.3}",
        data.electron_energy_mev, data.selected_energy_mev
    )?;
    Ok(out)
}

/// Read FEFF `dmdw_Egrid.info` from disk.
pub fn read_dmdw_egrid_info(path: impl AsRef<Path>) -> Result<DmdwEnergyGridInfo> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_egrid_info(&text)
}

/// Write FEFF `dmdw_Egrid.info` text to disk.
pub fn write_dmdw_egrid_info(path: impl AsRef<Path>, data: &DmdwEnergyGridInfo) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_egrid_info_string(data)?).map_err(|source| IoError::io(path, source))
}
