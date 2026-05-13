//! FEFF `dmdw.out` Debye-Waller diagnostic output support.
//!
//! The FEFF DMDW stage writes a human-readable report containing Lanczos
//! settings, projected-density-of-states poles, Einstein-frequency summaries,
//! moment-derived summaries, and the run-type-specific result block. This
//! parser keeps those pieces structured so generated FEFF10 DMDW reports can
//! be validated without string matching.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

const DMDW_OUT_PATH: &str = "dmdw.out";

/// Parsed contents of FEFF `dmdw.out`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutData {
    /// File header, or `None` for FEFF runs that produced an empty report.
    pub header: Option<DmdwOutHeader>,
    /// DMDW report sections in file order.
    pub sections: Vec<DmdwOutSection>,
}

impl DmdwOutData {
    /// Number of report sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
}

/// Header values written before the DMDW report sections.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutHeader {
    /// Lanczos recursion order used to build the pole expansion.
    pub lanczos_recursion_order: usize,
    /// Temperature declaration from the file header.
    pub temperature: DmdwOutTemperature,
    /// Dynamical-matrix file named in the DMDW run.
    pub dynamical_matrix_file: String,
}

/// Temperature declaration from the `dmdw.out` header.
#[derive(Debug, Clone, PartialEq)]
pub enum DmdwOutTemperature {
    /// Single-temperature DMDW run.
    Single(f64),
    /// Multi-temperature run whose temperatures are printed in result tables.
    ListedBelow,
}

/// One path, atom, or total-PDOS block from `dmdw.out`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutSection {
    /// Subject of the report block.
    pub subject: DmdwOutSubject,
    /// Projected DOS pole positions and weights.
    pub pdos_poles: Vec<DmdwOutPole>,
    /// Single-pole Einstein summary.
    pub einstein: Option<DmdwOutEinstein>,
    /// Moment-derived Einstein summaries.
    pub moments: Vec<DmdwOutMoment>,
    /// Reduced mass for path Debye-Waller output.
    pub reduced_mass_amu: Option<f64>,
    /// Path length for path Debye-Waller output.
    pub path_length_angstrom: Option<f64>,
    /// Single-temperature sigma2 value in `1e-3 Ang^2`.
    pub sigma2_1e_minus_3_angstrom2: Option<f64>,
    /// Multi-temperature sigma2 values in `1e-3 Ang^2`.
    pub sigma2_by_temperature: Vec<DmdwOutTemperatureValue>,
    /// Single-temperature vibrational free energy in eV.
    pub vibrational_free_energy_ev: Option<f64>,
    /// Multi-temperature vibrational free energy values in eV.
    pub vibrational_free_energy_by_temperature: Vec<DmdwOutTemperatureValue>,
    /// Single-temperature mean-square displacement in `1e-3 Ang^2`.
    pub u2_1e_minus_3_angstrom2: Option<f64>,
    /// Multi-temperature mean-square displacement values in `1e-3 Ang^2`.
    pub u2_by_temperature: Vec<DmdwOutTemperatureValue>,
    /// Whether the projected-DOS component completion line was written.
    pub projected_dos_component_computed: bool,
}

impl DmdwOutSection {
    /// Build an empty report section for the given subject.
    #[must_use]
    pub fn new(subject: DmdwOutSubject) -> Self {
        Self {
            subject,
            pdos_poles: Vec::new(),
            einstein: None,
            moments: Vec::new(),
            reduced_mass_amu: None,
            path_length_angstrom: None,
            sigma2_1e_minus_3_angstrom2: None,
            sigma2_by_temperature: Vec::new(),
            vibrational_free_energy_ev: None,
            vibrational_free_energy_by_temperature: Vec::new(),
            u2_1e_minus_3_angstrom2: None,
            u2_by_temperature: Vec::new(),
            projected_dos_component_computed: false,
        }
    }
}

/// Subject of a DMDW output block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmdwOutSubject {
    /// Run type 0 path output.
    PathIndices(Vec<usize>),
    /// Atom-index output for atom-local run types, with optional direction.
    AtomIndex {
        /// FEFF atom indices printed in the block header.
        indices: Vec<usize>,
        /// Optional perturbation direction label.
        direction: Option<String>,
    },
    /// Total projected-density-of-states output.
    TotalPdos,
}

/// One projected-density-of-states pole.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutPole {
    /// Pole frequency in THz.
    pub frequency_thz: f64,
    /// Pole weight.
    pub weight: f64,
}

/// Single-pole Einstein-frequency summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutEinstein {
    /// Einstein frequency in THz.
    pub frequency_thz: f64,
    /// Associated Einstein temperature in K.
    pub temperature_kelvin: f64,
    /// Effective force constant in N/m.
    pub effective_force_constant_n_per_m: f64,
}

/// Moment-derived Einstein-frequency summary row.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutMoment {
    /// Moment order `n`.
    pub order: i32,
    /// Moment value in `THz^n`.
    pub moment_thz_power_n: f64,
    /// Derived frequency in THz, absent for the `n = 0` placeholder row.
    pub frequency_thz: Option<f64>,
    /// Derived temperature in K, absent for the `n = 0` placeholder row.
    pub temperature_kelvin: Option<f64>,
    /// Derived effective force constant in N/m.
    pub effective_force_constant_n_per_m: Option<f64>,
}

/// A temperature-dependent scalar result row.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutTemperatureValue {
    /// Temperature in K.
    pub temperature_kelvin: f64,
    /// Scalar result value in the units of the table that owns the row.
    pub value: f64,
}

/// Parse FEFF `dmdw.out` text.
pub fn parse_dmdw_out(text: &str) -> Result<DmdwOutData> {
    if text.trim().is_empty() {
        return Ok(DmdwOutData {
            header: None,
            sections: Vec::new(),
        });
    }

    let lines = text
        .lines()
        .enumerate()
        .map(|(index, raw)| (index + 1, raw.trim()))
        .collect::<Vec<_>>();
    let mut position = 0;
    skip_blank_lines(&lines, &mut position);
    let header = parse_header(&lines, &mut position)?;

    let mut sections = Vec::new();
    while position < lines.len() {
        skip_section_gap(&lines, &mut position);
        if position >= lines.len() {
            break;
        }
        sections.push(parse_section(&lines, &mut position)?);
    }

    let data = DmdwOutData {
        header: Some(header),
        sections,
    };
    validate_dmdw_out(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw.out` text.
pub fn dmdw_out_string(data: &DmdwOutData) -> Result<String> {
    validate_dmdw_out(data)?;
    let Some(header) = &data.header else {
        return Ok(String::new());
    };

    let mut out = String::new();
    writeln!(
        out,
        "# Lanczos recursion order: {:4}",
        header.lanczos_recursion_order
    )?;
    match header.temperature {
        DmdwOutTemperature::Single(temperature) => {
            writeln!(out, "# Temperature: {:7.2}", temperature)?;
        }
        DmdwOutTemperature::ListedBelow => {
            writeln!(out, "# Temperature: (See list Below)")?;
        }
    }
    writeln!(
        out,
        "# Dynamical matrix file: {}",
        header.dynamical_matrix_file
    )?;
    writeln!(out)?;

    for section in &data.sections {
        write_separator(&mut out)?;
        write_subject(&mut out, &section.subject)?;
        write_pdos_poles(&mut out, section)?;
        write_einstein(&mut out, section)?;
        write_moments(&mut out, section)?;
        write_path_result(&mut out, section)?;
        write_vfe_result(&mut out, section)?;
        write_u2_result(&mut out, section)?;
        if section.projected_dos_component_computed {
            writeln!(out, " Projected DOS component computed.")?;
            writeln!(out)?;
        }
    }
    write_separator(&mut out)?;
    Ok(out)
}

/// Read FEFF `dmdw.out` text from a file.
pub fn read_dmdw_out(path: impl AsRef<Path>) -> Result<DmdwOutData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_out(&text)
}

/// Write FEFF `dmdw.out` text to a file.
pub fn write_dmdw_out(path: impl AsRef<Path>, data: &DmdwOutData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_out_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_header(lines: &[(usize, &str)], position: &mut usize) -> Result<DmdwOutHeader> {
    let (order_line, order_text) = next_line(lines, position, "Lanczos recursion order")?;
    let order = parse_prefixed_usize(order_line, order_text, "# Lanczos recursion order:")?;

    let (temperature_line, temperature_text) = next_line(lines, position, "temperature")?;
    let temperature_value =
        parse_prefixed_tail(temperature_line, temperature_text, "# Temperature:")?;
    let temperature = if temperature_value.eq_ignore_ascii_case("(See list Below)") {
        DmdwOutTemperature::ListedBelow
    } else {
        DmdwOutTemperature::Single(parse_f64(
            temperature_line,
            "temperature",
            temperature_value,
        )?)
    };

    let (dym_line, dym_text) = next_line(lines, position, "dynamical matrix file")?;
    let dynamical_matrix_file =
        parse_prefixed_tail(dym_line, dym_text, "# Dynamical matrix file:")?.to_owned();

    Ok(DmdwOutHeader {
        lanczos_recursion_order: order,
        temperature,
        dynamical_matrix_file,
    })
}

fn parse_section(lines: &[(usize, &str)], position: &mut usize) -> Result<DmdwOutSection> {
    let (line_number, line) = lines[*position];
    let subject = parse_subject(line_number, line)?;
    *position += 1;

    let mut section = DmdwOutSection::new(subject);
    if matches!(section.subject, DmdwOutSubject::AtomIndex { .. }) {
        parse_optional_direction(lines, position, &mut section)?;
    }

    while *position < lines.len() {
        let (current_line, current_text) = lines[*position];
        if current_text.is_empty() || is_separator_line(current_text) {
            *position += 1;
            continue;
        }
        if is_section_start(current_text) {
            break;
        }

        if current_text.starts_with("PDOS Poles:") {
            *position += 1;
            section.pdos_poles = parse_pdos_poles(lines, position)?;
        } else if current_text.starts_with("PDOS Einstein freq") {
            *position += 1;
            section.einstein = Some(parse_einstein(lines, position)?);
        } else if current_text.starts_with("pDOS n Moments") {
            *position += 1;
            section.moments = parse_moments(lines, position)?;
        } else if current_text.starts_with("Path Red. Mass (AMU):") {
            section.reduced_mass_amu = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "reduced_mass_amu",
            )?);
            *position += 1;
        } else if current_text.starts_with("Path Length (Ang), s^2") {
            let values = parse_values_after_colon(current_line, current_text)?;
            if values.len() != 2 {
                return parse_error(
                    current_line,
                    format!(
                        "path length/sigma2 row has {} value(s), expected 2",
                        values.len()
                    ),
                );
            }
            section.path_length_angstrom = Some(values[0]);
            section.sigma2_1e_minus_3_angstrom2 = Some(values[1]);
            *position += 1;
        } else if current_text.starts_with("Path Length (Ang):") {
            section.path_length_angstrom = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "path_length_angstrom",
            )?);
            *position += 1;
            skip_blank_lines(lines, position);
            if *position < lines.len() && lines[*position].1.starts_with("Temp (K)") {
                *position += 1;
                section.sigma2_by_temperature =
                    parse_temperature_values(lines, position, "sigma2")?;
            }
        } else if current_text.starts_with("VFE (eV):") {
            section.vibrational_free_energy_ev = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "vibrational_free_energy_ev",
            )?);
            *position += 1;
        } else if current_text.starts_with("Temp (K)") && current_text.contains("VFE") {
            *position += 1;
            section.vibrational_free_energy_by_temperature =
                parse_temperature_values(lines, position, "vibrational_free_energy_ev")?;
        } else if current_text.starts_with("u^2 (1e-3 Ang^2):") {
            section.u2_1e_minus_3_angstrom2 = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "u2_1e_minus_3_angstrom2",
            )?);
            *position += 1;
        } else if current_text.starts_with("Temp (K)") && current_text.contains("u^2") {
            *position += 1;
            section.u2_by_temperature = parse_temperature_values(lines, position, "u2")?;
        } else if current_text.starts_with("Projected DOS component computed.") {
            section.projected_dos_component_computed = true;
            *position += 1;
        } else {
            return parse_error(
                current_line,
                format!("unexpected DMDW output line {current_text:?}"),
            );
        }
    }

    Ok(section)
}

fn parse_subject(line_number: usize, line: &str) -> Result<DmdwOutSubject> {
    if let Some(tail) = line.strip_prefix("Path Indices:") {
        return Ok(DmdwOutSubject::PathIndices(parse_index_list(
            line_number,
            tail,
        )?));
    }
    if let Some(tail) = line.strip_prefix("Atom Index:") {
        return Ok(DmdwOutSubject::AtomIndex {
            indices: parse_index_list(line_number, tail)?,
            direction: None,
        });
    }
    if line == "Total PDOS results:" {
        return Ok(DmdwOutSubject::TotalPdos);
    }
    parse_error(
        line_number,
        format!("expected DMDW section header, found {line:?}"),
    )
}

fn parse_optional_direction(
    lines: &[(usize, &str)],
    position: &mut usize,
    section: &mut DmdwOutSection,
) -> Result<()> {
    if *position >= lines.len() {
        return Ok(());
    }
    let (line_number, line) = lines[*position];
    if !line.contains("Direction") {
        return Ok(());
    }
    let mut tokens = line.split_whitespace();
    let direction = tokens
        .find(|token| token.eq_ignore_ascii_case("Direction"))
        .and_then(|_| tokens.next())
        .filter(|token| token.chars().any(|ch| ch != '-'))
        .map(ToOwned::to_owned)
        .ok_or_else(|| parse_error_value(line_number, "missing atom perturbation direction"))?;
    if let DmdwOutSubject::AtomIndex {
        direction: slot, ..
    } = &mut section.subject
    {
        *slot = Some(direction);
    }
    *position += 1;
    Ok(())
}

fn parse_pdos_poles(lines: &[(usize, &str)], position: &mut usize) -> Result<Vec<DmdwOutPole>> {
    skip_blank_lines(lines, position);
    if *position < lines.len() && lines[*position].1.starts_with("Freq.") {
        *position += 1;
    }

    let mut poles = Vec::new();
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        if line.is_empty() {
            *position += 1;
            break;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            break;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return parse_error(
                line_number,
                format!("PDOS pole row has {} token(s), expected 2", tokens.len()),
            );
        }
        poles.push(DmdwOutPole {
            frequency_thz: parse_f64(line_number, "pole_frequency_thz", tokens[0])?,
            weight: parse_f64(line_number, "pole_weight", tokens[1])?,
        });
        *position += 1;
    }
    Ok(poles)
}

fn parse_einstein(lines: &[(usize, &str)], position: &mut usize) -> Result<DmdwOutEinstein> {
    skip_blank_lines(lines, position);
    if *position < lines.len() && lines[*position].1.starts_with("Freq (THz)") {
        *position += 1;
    }
    let (line_number, line) = next_data_line(lines, position, "Einstein row")?;
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 3 {
        return parse_error(
            line_number,
            format!("Einstein row has {} token(s), expected 3", tokens.len()),
        );
    }
    Ok(DmdwOutEinstein {
        frequency_thz: parse_f64(line_number, "einstein_frequency_thz", tokens[0])?,
        temperature_kelvin: parse_f64(line_number, "einstein_temperature_kelvin", tokens[1])?,
        effective_force_constant_n_per_m: parse_f64(
            line_number,
            "einstein_effective_force_constant_n_per_m",
            tokens[2],
        )?,
    })
}

fn parse_moments(lines: &[(usize, &str)], position: &mut usize) -> Result<Vec<DmdwOutMoment>> {
    skip_blank_lines(lines, position);
    if *position < lines.len() && lines[*position].1.starts_with('n') {
        *position += 1;
    }

    let mut moments = Vec::new();
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        if line.is_empty() {
            *position += 1;
            break;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            break;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 2 || tokens.len() > 5 {
            return parse_error(
                line_number,
                format!("moment row has {} token(s), expected 2 to 5", tokens.len()),
            );
        }
        moments.push(DmdwOutMoment {
            order: parse_i32(line_number, "moment_order", tokens[0])?,
            moment_thz_power_n: parse_f64(line_number, "moment_thz_power_n", tokens[1])?,
            frequency_thz: parse_optional_f64(
                line_number,
                "moment_frequency_thz",
                tokens.get(2).copied(),
            )?,
            temperature_kelvin: parse_optional_f64(
                line_number,
                "moment_temperature_kelvin",
                tokens.get(3).copied(),
            )?,
            effective_force_constant_n_per_m: parse_optional_f64(
                line_number,
                "moment_effective_force_constant_n_per_m",
                tokens.get(4).copied(),
            )?,
        });
        *position += 1;
    }
    Ok(moments)
}

fn parse_temperature_values(
    lines: &[(usize, &str)],
    position: &mut usize,
    value_field: &'static str,
) -> Result<Vec<DmdwOutTemperatureValue>> {
    let mut rows = Vec::new();
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        if line.is_empty() {
            *position += 1;
            break;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            break;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return parse_error(
                line_number,
                format!(
                    "temperature result row has {} token(s), expected 2",
                    tokens.len()
                ),
            );
        }
        rows.push(DmdwOutTemperatureValue {
            temperature_kelvin: parse_f64(line_number, "temperature_kelvin", tokens[0])?,
            value: parse_f64(line_number, value_field, tokens[1])?,
        });
        *position += 1;
    }
    Ok(rows)
}

fn next_data_line<'a>(
    lines: &'a [(usize, &'a str)],
    position: &mut usize,
    description: &'static str,
) -> Result<(usize, &'a str)> {
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        *position += 1;
        if line.is_empty() {
            continue;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            return parse_error(line_number, format!("missing {description}"));
        }
        return Ok((line_number, line));
    }
    parse_error(0, format!("missing {description}"))
}

fn write_subject(out: &mut String, subject: &DmdwOutSubject) -> Result<()> {
    match subject {
        DmdwOutSubject::PathIndices(indices) => {
            write!(out, " Path Indices:")?;
            write_index_list(out, indices)?;
            writeln!(out)?;
        }
        DmdwOutSubject::AtomIndex { indices, direction } => {
            write!(out, " Atom Index:")?;
            write_index_list(out, indices)?;
            writeln!(out)?;
            if let Some(direction) = direction {
                writeln!(
                    out,
                    "--------- Direction {direction} ---------------------------"
                )?;
            }
        }
        DmdwOutSubject::TotalPdos => {
            writeln!(out, " Total PDOS results:")?;
        }
    }
    Ok(())
}

fn write_separator(out: &mut String) -> Result<()> {
    writeln!(
        out,
        "--------------------------------------------------------------"
    )?;
    Ok(())
}

fn write_index_list(out: &mut String, indices: &[usize]) -> Result<()> {
    if let Some((first, rest)) = indices.split_first() {
        write!(out, "{first:5}")?;
        for index in rest {
            write!(out, "{index:4}")?;
        }
    }
    Ok(())
}

fn write_pdos_poles(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if section.pdos_poles.is_empty() {
        return Ok(());
    }
    writeln!(out, " PDOS Poles:")?;
    writeln!(out, "     Freq. (THz)    Weight")?;
    for pole in &section.pdos_poles {
        writeln!(out, "     {:8.3}{:18.9}", pole.frequency_thz, pole.weight)?;
    }
    writeln!(out)?;
    Ok(())
}

fn write_einstein(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    let Some(einstein) = &section.einstein else {
        return Ok(());
    };
    match section.subject {
        DmdwOutSubject::TotalPdos => writeln!(
            out,
            " PDOS Einstein freq (avg. of single poles), associated temp and eff. force constant: "
        )?,
        _ => writeln!(
            out,
            " PDOS Einstein freq (single pole), associated temp and eff. force constant: "
        )?,
    }
    writeln!(out, " Freq (THz)   Temp (K)   Eff. FC (N/m)")?;
    writeln!(
        out,
        "{:8.3}     {:8.2} {:12.4}",
        einstein.frequency_thz,
        einstein.temperature_kelvin,
        einstein.effective_force_constant_n_per_m
    )?;
    writeln!(out)?;
    Ok(())
}

fn write_moments(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if section.moments.is_empty() {
        return Ok(());
    }
    writeln!(
        out,
        " pDOS n Moments, associated Einstein freqs, temps and eff. force constants:"
    )?;
    writeln!(
        out,
        "  n     Mom (THz^n)   Freq (THz)     Temp (K)    Eff. FC (N/m)"
    )?;
    for moment in &section.moments {
        match (
            moment.frequency_thz,
            moment.temperature_kelvin,
            moment.effective_force_constant_n_per_m,
        ) {
            (Some(frequency), Some(temperature), Some(force_constant)) => writeln!(
                out,
                "{:3}{:14.5}{:14.5}     {:8.2} {:12.4}",
                moment.order, moment.moment_thz_power_n, frequency, temperature, force_constant
            )?,
            _ => writeln!(
                out,
                "{:3}{:14.5}     ---------     --------",
                moment.order, moment.moment_thz_power_n
            )?,
        }
    }
    writeln!(out)?;
    Ok(())
}

fn write_path_result(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if let Some(reduced_mass) = section.reduced_mass_amu {
        writeln!(out, " Path Red. Mass (AMU):{:12.6}", reduced_mass)?;
    }
    if let (Some(length), Some(sigma2)) = (
        section.path_length_angstrom,
        section.sigma2_1e_minus_3_angstrom2,
    ) {
        writeln!(
            out,
            " Path Length (Ang), s^2 (1e-3 Ang^2):{:8.4}{:9.4}",
            length, sigma2
        )?;
    } else if let Some(length) = section.path_length_angstrom
        && !section.sigma2_by_temperature.is_empty()
    {
        writeln!(out, " Path Length (Ang):{:8.4}", length)?;
        writeln!(out, " Temp (K)   s^2 (1e-3 Ang^2)")?;
        write_temperature_values(out, &section.sigma2_by_temperature, 4)?;
        writeln!(out)?;
    }
    Ok(())
}

fn write_vfe_result(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if let Some(value) = section.vibrational_free_energy_ev {
        writeln!(out, " VFE (eV):{:16.6}", value)?;
    }
    if !section.vibrational_free_energy_by_temperature.is_empty() {
        writeln!(out, " Temp (K)        VFE (eV)")?;
        write_temperature_values(out, &section.vibrational_free_energy_by_temperature, 6)?;
    }
    Ok(())
}

fn write_u2_result(out: &mut String, section: &DmdwOutSection) -> Result<()> {
    if let Some(value) = section.u2_1e_minus_3_angstrom2 {
        writeln!(out, " u^2 (1e-3 Ang^2):{:8.3}", value)?;
    }
    if !section.u2_by_temperature.is_empty() {
        writeln!(out, " Temp (K)   u^2 (1e-3 Ang^2)")?;
        write_temperature_values(out, &section.u2_by_temperature, 3)?;
    }
    Ok(())
}

fn write_temperature_values(
    out: &mut String,
    rows: &[DmdwOutTemperatureValue],
    decimals: usize,
) -> Result<()> {
    let width = decimals + 4;
    for row in rows {
        writeln!(
            out,
            "{:8.2}     {:width$.*}",
            row.temperature_kelvin, decimals, row.value
        )?;
    }
    Ok(())
}

fn validate_dmdw_out(data: &DmdwOutData) -> Result<()> {
    match &data.header {
        Some(header) => {
            if header.lanczos_recursion_order == 0 {
                return parse_error(1, "Lanczos recursion order must be positive");
            }
            validate_temperature(&header.temperature)?;
            if header.dynamical_matrix_file.trim().is_empty() {
                return parse_error(3, "dynamical matrix file must not be empty");
            }
            if data.sections.is_empty() {
                return parse_error(0, "non-empty dmdw.out must contain at least one section");
            }
            for (index, section) in data.sections.iter().enumerate() {
                validate_section(index + 1, header, section)?;
            }
        }
        None if !data.sections.is_empty() => {
            return parse_error(0, "sections require a dmdw.out header");
        }
        None => {}
    }
    Ok(())
}

fn validate_section(index: usize, header: &DmdwOutHeader, section: &DmdwOutSection) -> Result<()> {
    validate_subject(index, &section.subject)?;
    if !section.pdos_poles.is_empty() && section.pdos_poles.len() != header.lanczos_recursion_order
    {
        return parse_error(
            index,
            format!(
                "section has {} PDOS pole(s), expected {} from the header",
                section.pdos_poles.len(),
                header.lanczos_recursion_order
            ),
        );
    }
    for pole in &section.pdos_poles {
        validate_finite("pole_frequency_thz", pole.frequency_thz, index)?;
        validate_finite("pole_weight", pole.weight, index)?;
    }
    if let Some(einstein) = &section.einstein {
        validate_finite("einstein_frequency_thz", einstein.frequency_thz, index)?;
        validate_finite(
            "einstein_temperature_kelvin",
            einstein.temperature_kelvin,
            index,
        )?;
        validate_finite(
            "einstein_effective_force_constant_n_per_m",
            einstein.effective_force_constant_n_per_m,
            index,
        )?;
    }
    for moment in &section.moments {
        validate_finite("moment_thz_power_n", moment.moment_thz_power_n, index)?;
        validate_optional_finite("moment_frequency_thz", moment.frequency_thz, index)?;
        validate_optional_finite(
            "moment_temperature_kelvin",
            moment.temperature_kelvin,
            index,
        )?;
        validate_optional_finite(
            "moment_effective_force_constant_n_per_m",
            moment.effective_force_constant_n_per_m,
            index,
        )?;
    }
    validate_optional_finite("reduced_mass_amu", section.reduced_mass_amu, index)?;
    validate_optional_finite("path_length_angstrom", section.path_length_angstrom, index)?;
    validate_optional_finite(
        "sigma2_1e_minus_3_angstrom2",
        section.sigma2_1e_minus_3_angstrom2,
        index,
    )?;
    validate_temperature_values("sigma2", &section.sigma2_by_temperature, index)?;
    validate_optional_finite(
        "vibrational_free_energy_ev",
        section.vibrational_free_energy_ev,
        index,
    )?;
    validate_temperature_values(
        "vibrational_free_energy_ev",
        &section.vibrational_free_energy_by_temperature,
        index,
    )?;
    validate_optional_finite(
        "u2_1e_minus_3_angstrom2",
        section.u2_1e_minus_3_angstrom2,
        index,
    )?;
    validate_temperature_values("u2", &section.u2_by_temperature, index)?;
    Ok(())
}

fn validate_subject(index: usize, subject: &DmdwOutSubject) -> Result<()> {
    match subject {
        DmdwOutSubject::PathIndices(indices) | DmdwOutSubject::AtomIndex { indices, .. } => {
            if indices.is_empty() {
                return parse_error(index, "section index list must not be empty");
            }
            if indices.contains(&0) {
                return parse_error(index, "section indices must be one-based positive values");
            }
        }
        DmdwOutSubject::TotalPdos => {}
    }
    Ok(())
}

fn validate_temperature(temperature: &DmdwOutTemperature) -> Result<()> {
    match temperature {
        DmdwOutTemperature::Single(value) => validate_finite("temperature", *value, 2),
        DmdwOutTemperature::ListedBelow => Ok(()),
    }
}

fn validate_temperature_values(
    field: &'static str,
    rows: &[DmdwOutTemperatureValue],
    section_index: usize,
) -> Result<()> {
    for row in rows {
        validate_finite("temperature_kelvin", row.temperature_kelvin, section_index)?;
        validate_finite(field, row.value, section_index)?;
    }
    Ok(())
}

fn parse_prefixed_usize(line: usize, text: &str, prefix: &'static str) -> Result<usize> {
    let token = parse_prefixed_tail(line, text, prefix)?;
    parse_usize(line, "lanczos_recursion_order", token)
}

fn parse_prefixed_tail<'a>(line: usize, text: &'a str, prefix: &'static str) -> Result<&'a str> {
    text.strip_prefix(prefix)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| parse_error_value(line, format!("expected {prefix:?} line")))
}

fn parse_index_list(line: usize, tail: &str) -> Result<Vec<usize>> {
    let indices = tail
        .split_whitespace()
        .map(|token| parse_usize(line, "index", token))
        .collect::<Result<Vec<_>>>()?;
    if indices.is_empty() {
        return parse_error(line, "section index list must not be empty");
    }
    if indices.contains(&0) {
        return parse_error(line, "section indices must be positive");
    }
    Ok(indices)
}

fn parse_single_value_after_colon(line: usize, text: &str, field: &'static str) -> Result<f64> {
    let values = parse_values_after_colon(line, text)?;
    if values.len() != 1 {
        return parse_error(
            line,
            format!("{field} line has {} value(s), expected 1", values.len()),
        );
    }
    Ok(values[0])
}

fn parse_values_after_colon(line: usize, text: &str) -> Result<Vec<f64>> {
    let (_, tail) = text
        .split_once(':')
        .ok_or_else(|| parse_error_value(line, "expected ':' separator"))?;
    let values = tail
        .split_whitespace()
        .map(|token| parse_f64(line, "value", token))
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return parse_error(line, "expected numeric value after ':'");
    }
    Ok(values)
}

fn next_line<'a>(
    lines: &'a [(usize, &'a str)],
    position: &mut usize,
    description: &'static str,
) -> Result<(usize, &'a str)> {
    let Some(line) = lines.get(*position).copied() else {
        return parse_error(0, format!("missing {description} line"));
    };
    *position += 1;
    Ok(line)
}

fn skip_blank_lines(lines: &[(usize, &str)], position: &mut usize) {
    while *position < lines.len() && lines[*position].1.is_empty() {
        *position += 1;
    }
}

fn skip_section_gap(lines: &[(usize, &str)], position: &mut usize) {
    while *position < lines.len() {
        let line = lines[*position].1;
        if line.is_empty() || is_separator_line(line) {
            *position += 1;
        } else {
            break;
        }
    }
}

fn is_section_start(line: &str) -> bool {
    line.starts_with("Path Indices:")
        || line.starts_with("Atom Index:")
        || line == "Total PDOS results:"
}

fn is_section_gap_or_label(line: &str) -> bool {
    is_separator_line(line)
        || line.starts_with("PDOS Poles:")
        || line.starts_with("PDOS Einstein freq")
        || line.starts_with("pDOS n Moments")
        || line.starts_with("Path Red. Mass (AMU):")
        || line.starts_with("Path Length (Ang)")
        || line.starts_with("VFE (eV):")
        || line.starts_with("Temp (K)")
        || line.starts_with("u^2 (1e-3 Ang^2):")
        || line.starts_with("Projected DOS component computed.")
}

fn is_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 3 && trimmed.chars().all(|ch| ch == '-')
}

fn parse_optional_f64(
    line: usize,
    field: &'static str,
    token: Option<&str>,
) -> Result<Option<f64>> {
    match token {
        Some(value) if is_placeholder(value) => Ok(None),
        Some(value) => parse_f64(line, field, value).map(Some),
        None => Ok(None),
    }
}

fn is_placeholder(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|ch| ch == '-')
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token
        .parse::<i32>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn validate_optional_finite(field: &'static str, value: Option<f64>, line: usize) -> Result<()> {
    match value {
        Some(value) => validate_finite(field, value, line),
        None => Ok(()),
    }
}

fn validate_finite(field: &'static str, value: f64, line: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(line, format!("{field} must be finite"))
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: DMDW_OUT_PATH.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_empty_dmdw_out() -> Result<()> {
        let parsed = parse_dmdw_out("")?;
        assert_eq!(parsed.header, None);
        assert_eq!(parsed.section_count(), 0);
        assert_eq!(dmdw_out_string(&parsed)?, "");
        Ok(())
    }

    #[test]
    fn parses_path_dmdw_out() -> Result<()> {
        let parsed = parse_dmdw_out(DMDW_OUT)?;
        let header = parsed
            .header
            .as_ref()
            .ok_or_else(|| parse_error_value(0, "test fixture should contain a dmdw.out header"))?;
        assert_eq!(header.lanczos_recursion_order, 6);
        assert_eq!(header.temperature, DmdwOutTemperature::Single(450.0));
        assert_eq!(header.dynamical_matrix_file, "feff.dym");
        assert_eq!(parsed.section_count(), 1);

        let section = &parsed.sections[0];
        assert_eq!(section.subject, DmdwOutSubject::PathIndices(vec![1, 2]));
        assert_eq!(section.pdos_poles.len(), 6);
        assert_eq!(section.pdos_poles[0].frequency_thz, 2.860);
        assert_eq!(section.pdos_poles[0].weight, 0.039_469_598);
        assert_eq!(
            section.einstein,
            Some(DmdwOutEinstein {
                frequency_thz: 5.784,
                temperature_kelvin: 277.60,
                effective_force_constant_n_per_m: 69.6914,
            })
        );
        assert_eq!(section.moments.len(), 5);
        assert_eq!(section.moments[2].order, 0);
        assert_eq!(section.moments[2].frequency_thz, None);
        assert_eq!(section.reduced_mass_amu, Some(31.773));
        assert_eq!(section.path_length_angstrom, Some(2.5323));
        assert_eq!(section.sigma2_1e_minus_3_angstrom2, Some(11.8576));

        let rendered = dmdw_out_string(&parsed)?;
        assert_eq!(rendered, DMDW_OUT);
        assert_eq!(parse_dmdw_out(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_multi_temperature_and_atom_variants() -> Result<()> {
        let parsed = parse_dmdw_out(DMDW_OUT_VARIANTS)?;
        assert_eq!(
            parsed.header.as_ref().map(|header| &header.temperature),
            Some(&DmdwOutTemperature::ListedBelow)
        );
        assert_eq!(parsed.section_count(), 3);

        let path_section = &parsed.sections[0];
        assert_eq!(path_section.sigma2_by_temperature.len(), 2);
        assert_eq!(
            path_section.sigma2_by_temperature[1].temperature_kelvin,
            300.0
        );
        assert_eq!(path_section.sigma2_by_temperature[1].value, 2.5);

        let atom_section = &parsed.sections[1];
        assert_eq!(
            atom_section.subject,
            DmdwOutSubject::AtomIndex {
                indices: vec![3],
                direction: Some("x".to_owned()),
            }
        );
        assert_eq!(atom_section.u2_by_temperature.len(), 2);
        assert!(atom_section.projected_dos_component_computed);

        let total_section = &parsed.sections[2];
        assert_eq!(total_section.subject, DmdwOutSubject::TotalPdos);
        assert_eq!(total_section.vibrational_free_energy_ev, Some(-0.125));

        let rendered = dmdw_out_string(&parsed)?;
        assert_eq!(rendered, DMDW_OUT_VARIANTS);
        assert_eq!(parse_dmdw_out(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_dmdw_out(DMDW_OUT.replace("450.00", "4.5D+02").as_str())?;
        assert_eq!(
            parsed.header.as_ref().map(|header| &header.temperature),
            Some(&DmdwOutTemperature::Single(450.0))
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_dmdw_out_inputs() {
        assert!(parse_dmdw_out("# Temperature: 300.00\n").is_err());
        assert!(
            parse_dmdw_out(
                "# Lanczos recursion order: 0\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\n"
            )
            .is_err()
        );
        assert!(
            parse_dmdw_out(
                "# Lanczos recursion order: 1\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\nPath Indices: 0\n"
            )
            .is_err()
        );
        assert!(
            parse_dmdw_out(
                "# Lanczos recursion order: 1\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\nPath Indices: 1\nPDOS Poles:\nFreq. (THz) Weight\nNaN 1\n"
            )
            .is_err()
        );
        assert!(
            parse_dmdw_out(
                "# Lanczos recursion order: 2\n# Temperature: 300.00\n# Dynamical matrix file: feff.dym\nPath Indices: 1\nPDOS Poles:\nFreq. (THz) Weight\n1 1\n"
            )
            .is_err()
        );
    }

    const DMDW_OUT: &str = r#"# Lanczos recursion order:    6
# Temperature:  450.00
# Dynamical matrix file: feff.dym

--------------------------------------------------------------
 Path Indices:    1   2
 PDOS Poles:
     Freq. (THz)    Weight
        2.860       0.039469598
        3.854       0.182890396
        4.940       0.220041663
        6.026       0.159715119
        6.812       0.284980130
        7.306       0.112876736

 PDOS Einstein freq (single pole), associated temp and eff. force constant: 
 Freq (THz)   Temp (K)   Eff. FC (N/m)
   5.784       277.60      69.6914

 pDOS n Moments, associated Einstein freqs, temps and eff. force constants:
  n     Mom (THz^n)   Freq (THz)     Temp (K)    Eff. FC (N/m)
 -2       0.03881       5.07607       243.60      53.6688
 -1       0.18959       5.27461       253.13      57.9492
  0       0.99997     ---------     --------
  1       5.63317       5.63317       270.34      66.0957
  2      33.45823       5.78431       277.59      69.6899

 Path Red. Mass (AMU):   31.773000
 Path Length (Ang), s^2 (1e-3 Ang^2):  2.5323  11.8576
--------------------------------------------------------------
"#;

    const DMDW_OUT_VARIANTS: &str = r#"# Lanczos recursion order:    2
# Temperature: (See list Below)
# Dynamical matrix file: feff.dym

--------------------------------------------------------------
 Path Indices:    1   2
 PDOS Poles:
     Freq. (THz)    Weight
        2.000       0.500000000
        3.000       0.500000000

 Path Red. Mass (AMU):   31.773000
 Path Length (Ang):  2.5323
 Temp (K)   s^2 (1e-3 Ang^2)
  100.00       1.2500
  300.00       2.5000

--------------------------------------------------------------
 Atom Index:    3
--------- Direction x ---------------------------
 PDOS Poles:
     Freq. (THz)    Weight
        2.000       0.500000000
        3.000       0.500000000

 Temp (K)   u^2 (1e-3 Ang^2)
  100.00       0.500
  300.00       0.750
 Projected DOS component computed.

--------------------------------------------------------------
 Total PDOS results:
 PDOS Poles:
     Freq. (THz)    Weight
        2.000       0.500000000
        3.000       0.500000000

 VFE (eV):       -0.125000
--------------------------------------------------------------
"#;
}
