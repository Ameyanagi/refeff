use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::{IoError, Result};

use super::common::{
    array2_from_fortran, array3_from_fortran, checked_count2, checked_count3, decode_pad_line,
    i32_from_i64, invalid_pot_bin, usize_from_i64,
};
use super::types::{
    POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_MISC_SCALARS, POT_BIN_ORBITALS,
    POT_BIN_RADIAL_POINTS, PotBinData, PotBinScalars,
};
use super::validate::validate_pot_bin;

/// Parse FEFF `pot.bin` text.
pub fn parse_pot_bin(text: &str) -> Result<PotBinData> {
    let mut lines = PotBinLines::new(text);
    let header = lines.int_values("header", 9)?;
    let title_count = usize_from_i64(header[0], "ntitle")?;
    let potential_count = usize_from_i64(header[1], "nph")?
        .checked_add(1)
        .ok_or_else(|| invalid_pot_bin("nph", "potential count overflowed"))?;
    let pad_width = usize_from_i64(header[2], "npadx")?;
    let nohole = i32_from_i64(header[3], "nohole")?;
    let ihole = i32_from_i64(header[4], "ihole")?;
    let interstitial_selector = i32_from_i64(header[5], "inters")?;
    let automatic_folp = i32_from_i64(header[6], "iafolp")?;
    let jump_mode = i32_from_i64(header[7], "jumprm")?;
    let unfreeze_f = i32_from_i64(header[8], "iunf")?;

    let mut titles = Vec::with_capacity(title_count);
    for _ in 0..title_count {
        titles.push(lines.title()?);
    }

    let misc = lines.pad_reals("dum", pad_width, POT_BIN_MISC_SCALARS)?;
    let scalars = PotBinScalars::from_slice(&misc)?;
    let muffin_tin_indices = lines.usize_array("imt", potential_count)?;
    let muffin_tin_radii = lines.real_array("rmt", pad_width, potential_count)?;
    let norman_indices = lines.usize_array("inrm", potential_count)?;
    let atomic_numbers = lines.usize_array("iz", potential_count)?;
    let kappa = lines.i32_array("kappa", POT_BIN_ORBITALS)?;
    let norman_radii = lines.real_array("rnrm", pad_width, potential_count)?;
    let overlap_factors = lines.real_array("folp", pad_width, potential_count)?;
    let max_overlap_factors = lines.real_array("folpx", pad_width, potential_count)?;
    let potential_multiplicities = lines.real_array("xnatph", pad_width, potential_count)?;
    let ionization = lines.real_array("xion", pad_width, potential_count)?;
    let initial_large_component = lines.real_array("dgc0", pad_width, POT_BIN_RADIAL_POINTS)?;
    let initial_small_component = lines.real_array("dpc0", pad_width, POT_BIN_RADIAL_POINTS)?;

    let radial_orbital_potential = checked_count3(
        "dgc",
        POT_BIN_RADIAL_POINTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let coefficient_orbital_potential = checked_count3(
        "adgc",
        POT_BIN_COEFFICIENTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let radial_potential = checked_count2("edens", POT_BIN_RADIAL_POINTS, potential_count)?;
    let orbital_potential = checked_count2("xnval", POT_BIN_ORBITALS, potential_count)?;

    let large_components = array3_from_fortran(
        "dgc",
        lines.pad_reals("dgc", pad_width, radial_orbital_potential)?,
        POT_BIN_RADIAL_POINTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let small_components = array3_from_fortran(
        "dpc",
        lines.pad_reals("dpc", pad_width, radial_orbital_potential)?,
        POT_BIN_RADIAL_POINTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let large_coefficients = array3_from_fortran(
        "adgc",
        lines.pad_reals("adgc", pad_width, coefficient_orbital_potential)?,
        POT_BIN_COEFFICIENTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let small_coefficients = array3_from_fortran(
        "adpc",
        lines.pad_reals("adpc", pad_width, coefficient_orbital_potential)?,
        POT_BIN_COEFFICIENTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let electron_density = array2_from_fortran(
        "edens",
        lines.pad_reals("edens", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let coulomb_potential = array2_from_fortran(
        "vclap",
        lines.pad_reals("vclap", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let total_potential = array2_from_fortran(
        "vtot",
        lines.pad_reals("vtot", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let valence_density = array2_from_fortran(
        "edenvl",
        lines.pad_reals("edenvl", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let valence_potential = array2_from_fortran(
        "vvalgs",
        lines.pad_reals("vvalgs", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let magnetization_density = array2_from_fortran(
        "dmag",
        lines.pad_reals("dmag", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let orbital_occupancy = array2_from_fortran(
        "xnval",
        lines.pad_reals("xnval", pad_width, orbital_potential)?,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let orbital_energies = lines.real_array("eorb", pad_width, POT_BIN_ORBITALS)?;

    let mut occupied_orbital_indices = Array2::<i32>::zeros((POT_BIN_IORB_SLOTS, potential_count));
    for potential in 0..potential_count {
        let values = lines.i32_values("iorb", POT_BIN_IORB_SLOTS)?;
        for slot in 0..POT_BIN_IORB_SLOTS {
            occupied_orbital_indices[(slot, potential)] = values[slot];
        }
    }

    let norman_charges = lines.real_array("qnrm", pad_width, potential_count)?;
    let xnmues = lines.pad_reals_to_eof("xnmues", pad_width)?;
    if xnmues.is_empty() || xnmues.len() % potential_count != 0 {
        return Err(IoError::PotBinShape {
            field: "xnmues",
            actual: vec![xnmues.len()],
            expected: vec![potential_count],
        });
    }
    let angular_count = xnmues.len() / potential_count;
    let valence_occupancy = array2_from_fortran("xnmues", xnmues, angular_count, potential_count)?;
    lines.finish()?;

    let data = PotBinData {
        titles,
        pad_width,
        nohole,
        ihole,
        interstitial_selector,
        automatic_folp,
        jump_mode,
        unfreeze_f,
        scalars,
        muffin_tin_indices,
        muffin_tin_radii,
        norman_indices,
        atomic_numbers,
        kappa,
        norman_radii,
        overlap_factors,
        max_overlap_factors,
        potential_multiplicities,
        ionization,
        initial_large_component,
        initial_small_component,
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        electron_density,
        coulomb_potential,
        total_potential,
        valence_density,
        valence_potential,
        magnetization_density,
        orbital_occupancy,
        orbital_energies,
        occupied_orbital_indices,
        norman_charges,
        valence_occupancy,
        raw_text: Some(text.to_string()),
    };
    validate_pot_bin(&data)?;
    Ok(data)
}

/// Read FEFF `pot.bin` text from a file.
pub fn read_pot_bin(path: impl AsRef<Path>) -> Result<PotBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_pot_bin(&text)
}

struct PotBinLines<'a> {
    lines: Vec<&'a str>,
    position: usize,
}

impl<'a> PotBinLines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines().collect(),
            position: 0,
        }
    }

    fn finish(self) -> Result<()> {
        let count = self.lines[self.position..]
            .iter()
            .filter(|line| !line.trim().is_empty())
            .count();
        if count == 0 {
            Ok(())
        } else {
            Err(IoError::PotBinTrailingLines { count })
        }
    }

    fn title(&mut self) -> Result<String> {
        let line = self.next_line("title")?;
        Ok(line.to_string())
    }

    fn int_values(&mut self, field: &'static str, expected: usize) -> Result<Vec<i64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            for token in line.split_whitespace() {
                if values.len() == expected {
                    break;
                }
                values.push(token.parse::<i64>().map_err(|_| IoError::PotBinParse {
                    field,
                    token: token.to_string(),
                })?);
            }
        }
        Ok(values)
    }

    fn i32_values(&mut self, field: &'static str, expected: usize) -> Result<Vec<i32>> {
        self.int_values(field, expected)?
            .into_iter()
            .map(|value| i32_from_i64(value, field))
            .collect()
    }

    fn i32_array(&mut self, field: &'static str, len: usize) -> Result<Array1<i32>> {
        Ok(Array1::from_vec(self.i32_values(field, len)?))
    }

    fn usize_array(&mut self, field: &'static str, len: usize) -> Result<Array1<usize>> {
        let values = self
            .int_values(field, len)?
            .into_iter()
            .map(|value| usize_from_i64(value, field))
            .collect::<Result<Vec<_>>>()?;
        Ok(Array1::from_vec(values))
    }

    fn real_array(
        &mut self,
        field: &'static str,
        pad_width: usize,
        len: usize,
    ) -> Result<Array1<f64>> {
        Ok(Array1::from_vec(self.pad_reals(field, pad_width, len)?))
    }

    fn pad_reals(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            let decoded = decode_pad_line(field, line, pad_width)?;
            if decoded.is_empty() {
                return Err(IoError::PadPayload {
                    payload_len: 0,
                    unit_len: pad_width,
                });
            }
            for value in decoded {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(values)
    }

    fn pad_reals_to_eof(&mut self, field: &'static str, pad_width: usize) -> Result<Vec<f64>> {
        let mut values = Vec::new();
        while self.position < self.lines.len() {
            let line = self.next_line(field)?;
            if line.trim().is_empty() {
                continue;
            }
            values.extend(decode_pad_line(field, line, pad_width)?);
        }
        Ok(values)
    }

    fn next_line(&mut self, field: &'static str) -> Result<&'a str> {
        let line = self
            .lines
            .get(self.position)
            .copied()
            .ok_or(IoError::PotBinMissing { field })?;
        self.position += 1;
        Ok(line)
    }
}
