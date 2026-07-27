//! FEFF `paths.dat` text codec.
//!
//! `PATH/pathsd.f90` writes this file for GENFMT. The header is free text
//! terminated by FEFF's dashed separator, followed by path records. Each path
//! starts with `index`, `nleg`, degeneracy, and `r=...`; the next line is a row
//! label, then `nleg` atom rows. Full PATH output includes `rleg`, `beta`, and
//! `eta`; RDINP-generated single-scattering rows omit those trailing values.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView2};
use refeff_core::{
    GenfmtNStarPathInput, PathDegeneracyGroup, PathDegeneracyReduction, PathRotationInput,
    path_output_parameters,
};

use crate::control_input::FEFF_BOHR_ANGSTROM;
use crate::error::{IoError, Result};

/// One atom row from FEFF `paths.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatAtom {
    /// Cartesian position in Angstroms as written by FEFF.
    pub position_angstrom: [f64; 3],
    /// Potential index, `ipot`.
    pub potential_index: i32,
    /// FEFF potential label without surrounding quotes.
    pub label: String,
    /// Leg length in Angstroms when present in full PATH output.
    pub leg_distance_angstrom: Option<f64>,
    /// Euler beta angle in degrees when present in full PATH output.
    pub beta_degrees: Option<f64>,
    /// Euler eta angle in degrees when present in full PATH output.
    pub eta_degrees: Option<f64>,
}

/// One FEFF `paths.dat` path record.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatPath {
    /// Path index, `ipath`.
    pub index: usize,
    /// Path degeneracy.
    pub degeneracy: f64,
    /// Effective half path length in Angstroms, from the `r=` header field.
    pub effective_half_path_length_angstrom: f64,
    /// The label line between the path header and atom rows.
    pub row_header: String,
    /// Atom rows, including the final absorber row.
    pub atoms: Vec<PathsDatAtom>,
}

impl PathsDatPath {
    /// Number of legs, `nleg`.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.atoms.len()
    }

    /// Potential indices as an ndarray vector in path-row order.
    #[must_use]
    pub fn potential_indices(&self) -> Array1<i32> {
        Array1::from_iter(self.atoms.iter().map(|atom| atom.potential_index))
    }

    /// Path atom coordinates converted to FEFF GENFMT bohr units.
    ///
    /// This mirrors `GENFMT/rdpath.f90`, which reads `paths.dat` coordinates in
    /// Angstroms and divides each component by `bohr` before filling `rat`.
    pub fn genfmt_positions_bohr(&self) -> Result<Array2<f64>> {
        ensure_nonempty_path(self)?;
        let mut positions = Array2::zeros((self.leg_count(), 3));
        for (leg, atom) in self.atoms.iter().enumerate() {
            for component in 0..3 {
                positions[(leg, component)] =
                    angstrom_to_bohr("position", atom.position_angstrom[component])?;
            }
        }
        Ok(positions)
    }

    /// Potential indices converted to the unsigned GENFMT representation.
    ///
    /// FEFF `rdpath` rejects `ipot(ileg) > npot`; Rust also rejects negative
    /// indices before they can be used to index potential tables.
    pub fn genfmt_potential_indices(&self, max_potential_index: usize) -> Result<Array1<usize>> {
        ensure_nonempty_path(self)?;
        let mut indices = Vec::with_capacity(self.leg_count());
        for (leg, atom) in self.atoms.iter().enumerate() {
            let potential_index = usize::try_from(atom.potential_index).map_err(|_| {
                invalid_paths_dat(
                    "ipot",
                    format!(
                        "leg {} potential index {} must be nonnegative",
                        leg + 1,
                        atom.potential_index
                    ),
                )
            })?;
            if potential_index > max_potential_index {
                return Err(invalid_paths_dat(
                    "ipot",
                    format!(
                        "leg {} potential index {} exceeds npot {}",
                        leg + 1,
                        potential_index,
                        max_potential_index
                    ),
                ));
            }
            indices.push(potential_index);
        }
        Ok(Array1::from_vec(indices))
    }

    /// Build owned GENFMT-ready path data from this `paths.dat` record.
    pub fn to_genfmt_path(&self, max_potential_index: usize) -> Result<PathsDatGenfmtPath> {
        if !self.degeneracy.is_finite() {
            return Err(invalid_paths_dat("degeneracy", "value must be finite"));
        }
        Ok(PathsDatGenfmtPath {
            index: self.index,
            degeneracy: self.degeneracy,
            effective_half_path_length_bohr: angstrom_to_bohr(
                "r",
                self.effective_half_path_length_angstrom,
            )?,
            potential_indices: self.genfmt_potential_indices(max_potential_index)?,
            positions_bohr: self.genfmt_positions_bohr()?,
        })
    }
}

/// Owned `paths.dat` path data normalized for GENFMT core helpers.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatGenfmtPath {
    /// FEFF path index, `ipath`.
    pub index: usize,
    /// Path degeneracy.
    pub degeneracy: f64,
    /// Header effective half path length converted from Angstroms to bohr.
    pub effective_half_path_length_bohr: f64,
    /// FEFF `ipot(1:nleg)` as unsigned indices.
    pub potential_indices: Array1<usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions_bohr: Array2<f64>,
}

impl PathsDatGenfmtPath {
    /// Number of legs, `nleg`.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.positions_bohr.shape()[0]
    }

    /// Number of scattering events, `nsc=nleg-1`.
    #[must_use]
    pub fn scattering_count(&self) -> Option<usize> {
        self.leg_count().checked_sub(1)
    }

    /// Borrow this path as FEFF `rdpath` geometry input for GENFMT core.
    #[must_use]
    pub fn path_rotation_input(&self, polarized: bool) -> PathRotationInput<'_> {
        PathRotationInput {
            positions: self.positions_bohr.view(),
            polarized,
        }
    }

    /// Borrow this path as one FEFF GENFMT `nstar.dat` path row input.
    #[must_use]
    pub fn nstar_path_input(&self) -> GenfmtNStarPathInput<'_> {
        GenfmtNStarPathInput {
            positions: self.positions_bohr.view(),
            degeneracy: self.degeneracy,
        }
    }
}

/// FEFF `paths.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatData {
    /// Header/title lines before FEFF's dashed separator.
    pub titles: Vec<String>,
    /// Path records.
    pub paths: Vec<PathsDatPath>,
}

/// Inputs for rendering one FEFF `pathsd` retained path to `paths.dat`.
#[derive(Debug, Clone, Copy)]
pub struct PathsdPathsDatPathInput<'a> {
    /// FEFF `ipath` output index.
    pub path_index: usize,
    /// Retained unique path group from the `pathsd` degeneracy reducer.
    pub group: &'a PathDegeneracyGroup,
    /// Atom-indexed Cartesian coordinates in Angstroms, with row `0` as absorber.
    pub atom_positions: ArrayView2<'a, f64>,
    /// Atom-indexed FEFF potential IDs, equivalent to `ipot(atom)`.
    pub atom_potentials: &'a [usize],
    /// Potential labels indexed by potential ID, equivalent to `potlbl`.
    pub potential_labels: &'a [&'a str],
}

/// Inputs for rendering a complete FEFF `pathsd` reduction to `paths.dat`.
#[derive(Debug, Clone, Copy)]
pub struct PathsdPathsDatDataInput<'a> {
    /// Header lines copied from FEFF `paths.bin`, before the dashed separator.
    pub titles: &'a [String],
    /// Completed `pathsd` degeneracy reduction.
    pub reduction: &'a PathDegeneracyReduction,
    /// Atom-indexed Cartesian coordinates in Angstroms, with row `0` as absorber.
    pub atom_positions: ArrayView2<'a, f64>,
    /// Atom-indexed FEFF potential IDs, equivalent to `ipot(atom)`.
    pub atom_potentials: &'a [usize],
    /// Potential labels indexed by potential ID, equivalent to `potlbl`.
    pub potential_labels: &'a [&'a str],
}

impl PathsDatData {
    /// Convert all `paths.dat` records to owned GENFMT-ready path data.
    ///
    /// Output order matches FEFF's `paths.dat` traversal order.
    pub fn to_genfmt_paths(&self, max_potential_index: usize) -> Result<Vec<PathsDatGenfmtPath>> {
        self.paths
            .iter()
            .map(|path| path.to_genfmt_path(max_potential_index))
            .collect()
    }
}

/// Build FEFF `paths.dat` contents from a full retained `pathsd` reduction.
///
/// This ports the retained-path output ordering in `PATH/pathsd.f90`: ranges are
/// traversed in `paths.bin` order, unique groups use the hash-order produced by
/// the reducer, and only groups whose `critpw` decision retained them receive a
/// sequential one-based path index.
pub fn pathsd_paths_dat_data(input: PathsdPathsDatDataInput<'_>) -> Result<PathsDatData> {
    let PathsdPathsDatDataInput {
        titles,
        reduction,
        atom_positions,
        atom_potentials,
        potential_labels,
    } = input;

    let mut paths = Vec::with_capacity(reduction.retained_unique_count);
    let mut retained_total_degeneracy = 0_usize;
    for processed_range in &reduction.ranges {
        let range = &processed_range.range;
        for decision in &range.retention.decisions {
            if !decision.retained {
                continue;
            }
            let group = range.groups.get(decision.group_index).ok_or_else(|| {
                invalid_paths_dat(
                    "pathsd reduction",
                    format!(
                        "retained decision references group {} but range has {} groups",
                        decision.group_index,
                        range.groups.len()
                    ),
                )
            })?;
            retained_total_degeneracy = retained_total_degeneracy
                .checked_add(group.degeneracy)
                .ok_or_else(|| {
                    invalid_paths_dat("pathsd reduction", "retained degeneracy total is too large")
                })?;
            let mut path = pathsd_paths_dat_path(PathsdPathsDatPathInput {
                path_index: paths.len() + 1,
                group,
                atom_positions,
                atom_potentials,
                potential_labels,
            })?;
            // FEFF writes `rcurr/2` from the first paths.bin record in this
            // total-length range, rather than recomputing the selected
            // degeneracy representative's length.
            path.effective_half_path_length_angstrom =
                processed_range.representative_total_path_length / 2.0;
            paths.push(path);
        }
    }

    if paths.len() != reduction.retained_unique_count
        || retained_total_degeneracy != reduction.retained_total_degeneracy
    {
        return Err(invalid_paths_dat(
            "pathsd reduction",
            format!(
                "retained decisions produced {} unique paths and total degeneracy {}, but reduction summary reports {} and {}",
                paths.len(),
                retained_total_degeneracy,
                reduction.retained_unique_count,
                reduction.retained_total_degeneracy
            ),
        ));
    }

    Ok(PathsDatData {
        titles: titles.to_vec(),
        paths,
    })
}

/// Build one FEFF `paths.dat` record from a retained `pathsd` degeneracy group.
///
/// This ports the output block after `pathsd` applies `critpw`: it calls the
/// Rust `mpprmd` port for `rleg`, `beta`, and `eta`, writes each scattering atom
/// in canonical path order, and appends the absorber row.
pub fn pathsd_paths_dat_path(input: PathsdPathsDatPathInput<'_>) -> Result<PathsDatPath> {
    let PathsdPathsDatPathInput {
        path_index,
        group,
        atom_positions,
        atom_potentials,
        potential_labels,
    } = input;
    if atom_positions.ncols() != 3 || atom_positions.nrows() == 0 {
        return Err(invalid_paths_dat(
            "position",
            format!(
                "atom positions must be natoms x 3, got {}x{}",
                atom_positions.nrows(),
                atom_positions.ncols()
            ),
        ));
    }

    let output = path_output_parameters(atom_positions, &group.path_indices)
        .map_err(|source| invalid_paths_dat("path geometry", source.to_string()))?;
    let total_path_length = output.leg_distances.iter().copied().sum::<f64>();
    let mut atoms = Vec::with_capacity(group.path_indices.len() + 1);
    for (leg, atom_index) in group
        .path_indices
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .enumerate()
    {
        atoms.push(pathsd_paths_dat_atom(
            atom_positions,
            atom_potentials,
            potential_labels,
            atom_index,
            leg,
            &output,
        )?);
    }

    Ok(PathsDatPath {
        index: path_index,
        degeneracy: group.degeneracy as f64,
        effective_half_path_length_angstrom: total_path_length / 2.0,
        row_header:
            "      x           y           z     ipot  label      rleg      beta        eta"
                .to_string(),
        atoms,
    })
}

fn pathsd_paths_dat_atom(
    atom_positions: ArrayView2<'_, f64>,
    atom_potentials: &[usize],
    potential_labels: &[&str],
    atom_index: usize,
    leg: usize,
    output: &refeff_core::PathOutputParameters,
) -> Result<PathsDatAtom> {
    let position = atom_position_for_paths_dat(atom_positions, atom_index)?;
    let potential_index = *atom_potentials.get(atom_index).ok_or_else(|| {
        invalid_paths_dat(
            "ipot",
            format!(
                "atom {atom_index} has no potential entry in {} atom potentials",
                atom_potentials.len()
            ),
        )
    })?;
    let label = potential_labels.get(potential_index).ok_or_else(|| {
        invalid_paths_dat(
            "label",
            format!(
                "potential {potential_index} has no label in {} potential labels",
                potential_labels.len()
            ),
        )
    })?;
    let potential_index = i32::try_from(potential_index).map_err(|_| {
        invalid_paths_dat(
            "ipot",
            format!("potential index {potential_index} cannot be represented as i32"),
        )
    })?;
    Ok(PathsDatAtom {
        position_angstrom: position,
        potential_index,
        label: (*label).to_string(),
        leg_distance_angstrom: Some(output.leg_distances[leg]),
        beta_degrees: Some(output.scattering_angles[leg].to_degrees()),
        eta_degrees: Some(output.eta_angles[leg].to_degrees()),
    })
}

fn atom_position_for_paths_dat(
    atom_positions: ArrayView2<'_, f64>,
    atom_index: usize,
) -> Result<[f64; 3]> {
    if atom_index >= atom_positions.nrows() {
        return Err(invalid_paths_dat(
            "position",
            format!(
                "atom index {atom_index} is outside {} atom positions",
                atom_positions.nrows()
            ),
        ));
    }
    Ok([
        atom_positions[(atom_index, 0)],
        atom_positions[(atom_index, 1)],
        atom_positions[(atom_index, 2)],
    ])
}

/// Borrow GENFMT-ready paths as FEFF `rdpath` inputs in traversal order.
#[must_use]
pub fn genfmt_path_rotation_inputs<'a>(
    paths: &'a [PathsDatGenfmtPath],
    polarized: bool,
) -> Vec<PathRotationInput<'a>> {
    paths
        .iter()
        .map(|path| path.path_rotation_input(polarized))
        .collect()
}

/// Borrow GENFMT-ready paths as FEFF `nstar.dat` row inputs in traversal order.
#[must_use]
pub fn genfmt_nstar_path_inputs<'a>(
    paths: &'a [PathsDatGenfmtPath],
) -> Vec<GenfmtNStarPathInput<'a>> {
    paths
        .iter()
        .map(PathsDatGenfmtPath::nstar_path_input)
        .collect()
}

/// Render FEFF `paths.dat` text.
pub fn paths_dat_string(data: &PathsDatData) -> Result<String> {
    validate_paths_dat(data)?;

    let mut out = String::new();
    for title in &data.titles {
        writeln!(out, "{title}")?;
    }
    writeln!(out, " {}", "-".repeat(71))?;
    for path in &data.paths {
        writeln!(
            out,
            " {:5}{:5}{:8.3}  index, nleg, degeneracy, r={:8.4}",
            path.index,
            path.leg_count(),
            path.degeneracy,
            path.effective_half_path_length_angstrom
        )?;
        if path.row_header.trim().is_empty() {
            writeln!(
                out,
                "      x           y           z     ipot  label      rleg      beta        eta"
            )?;
        } else {
            writeln!(out, "{}", path.row_header)?;
        }
        for atom in &path.atoms {
            write!(
                out,
                "{:12.6}{:12.6}{:12.6}{:4} '{}'",
                atom.position_angstrom[0],
                atom.position_angstrom[1],
                atom.position_angstrom[2],
                atom.potential_index,
                fixed_label(&atom.label)
            )?;
            match (
                atom.leg_distance_angstrom,
                atom.beta_degrees,
                atom.eta_degrees,
            ) {
                (Some(rleg), Some(beta), Some(eta)) => {
                    writeln!(out, " {:10.4}{:10.4}{:10.4}", rleg, beta, eta)?;
                }
                _ => {
                    writeln!(out)?;
                }
            }
        }
    }
    Ok(out)
}

/// Parse FEFF `paths.dat` text.
pub fn parse_paths_dat(text: &str) -> Result<PathsDatData> {
    let mut lines = text.lines().enumerate().peekable();
    let mut titles = Vec::new();
    loop {
        let Some((_, text)) = lines.next() else {
            return Err(IoError::PathsDatMissing { field: "separator" });
        };
        if is_separator(text) {
            break;
        }
        titles.push(text.to_string());
    }

    let mut paths = Vec::new();
    while let Some((line, text)) = next_nonempty(&mut lines) {
        let header = parse_path_header(line, text)?;
        let (row_header_line, row_header) = lines
            .next()
            .map(|(line, text)| (line + 1, text))
            .ok_or(IoError::PathsDatMissing {
                field: "row header",
            })?;
        if row_header.trim().is_empty() {
            return Err(invalid_paths_dat(
                "row header",
                format!("line {row_header_line} is empty"),
            ));
        }

        let mut atoms = Vec::with_capacity(header.leg_count);
        for _ in 0..header.leg_count {
            let (atom_line, atom_text) = lines
                .next()
                .map(|(line, text)| (line + 1, text))
                .ok_or(IoError::PathsDatMissing { field: "atom row" })?;
            atoms.push(parse_atom_row(atom_line, atom_text)?);
        }
        paths.push(PathsDatPath {
            index: header.index,
            degeneracy: header.degeneracy,
            effective_half_path_length_angstrom: header.effective_half_path_length_angstrom,
            row_header: row_header.to_string(),
            atoms,
        });
    }

    let data = PathsDatData { titles, paths };
    validate_paths_dat(&data)?;
    Ok(data)
}

/// Write FEFF `paths.dat` text to a file.
pub fn write_paths_dat(path: impl AsRef<Path>, data: &PathsDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, paths_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `paths.dat` text from a file.
pub fn read_paths_dat(path: impl AsRef<Path>) -> Result<PathsDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_paths_dat(&text)
}

#[derive(Debug, Clone, Copy)]
struct PathHeader {
    index: usize,
    leg_count: usize,
    degeneracy: f64,
    effective_half_path_length_angstrom: f64,
}

fn validate_paths_dat(data: &PathsDatData) -> Result<()> {
    for path in &data.paths {
        if path.leg_count() == 0 {
            return Err(invalid_paths_dat(
                "nleg",
                "path must contain at least one atom",
            ));
        }
        if !path.degeneracy.is_finite() {
            return Err(invalid_paths_dat("degeneracy", "value must be finite"));
        }
        if !path.effective_half_path_length_angstrom.is_finite() {
            return Err(invalid_paths_dat("r", "value must be finite"));
        }
        for atom in &path.atoms {
            if atom.label.trim().is_empty() {
                return Err(invalid_paths_dat("label", "label must not be empty"));
            }
            let optional_geometry = [
                atom.leg_distance_angstrom,
                atom.beta_degrees,
                atom.eta_degrees,
            ];
            let optional_geometry_count = optional_geometry
                .iter()
                .filter(|value| value.is_some())
                .count();
            if !matches!(optional_geometry_count, 0 | 3) {
                return Err(invalid_paths_dat(
                    "atom row",
                    "rleg, beta, and eta must be all present or all absent",
                ));
            }
            for value in atom.position_angstrom {
                if !value.is_finite() {
                    return Err(invalid_paths_dat("position", "all values must be finite"));
                }
            }
            for value in optional_geometry.into_iter().flatten() {
                if !value.is_finite() {
                    return Err(invalid_paths_dat("atom row", "all values must be finite"));
                }
            }
        }
    }
    Ok(())
}

fn is_separator(line: &str) -> bool {
    let trimmed = line.trim();
    line.get(3..11).is_some_and(|text| text == "--------")
        || (!trimmed.is_empty() && trimmed.chars().all(|ch| ch == '-'))
}

fn next_nonempty<'a>(
    lines: &mut std::iter::Peekable<std::iter::Enumerate<std::str::Lines<'a>>>,
) -> Option<(usize, &'a str)> {
    for (line, text) in lines.by_ref() {
        if !text.trim().is_empty() {
            return Some((line + 1, text));
        }
    }
    None
}

fn parse_path_header(line: usize, text: &str) -> Result<PathHeader> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 3 {
        return Err(IoError::PathsDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 3,
        });
    }
    let effective_half_path_length_angstrom = parse_r_value(line, text)?;
    Ok(PathHeader {
        index: parse_usize(line, "index", tokens[0])?,
        leg_count: parse_usize(line, "nleg", tokens[1])?,
        degeneracy: parse_f64(line, "degeneracy", tokens[2])?,
        effective_half_path_length_angstrom,
    })
}

fn parse_r_value(line: usize, text: &str) -> Result<f64> {
    let Some((_, value)) = text.split_once("r=") else {
        return Err(IoError::PathsDatMissing { field: "r" });
    };
    let token = value
        .split_whitespace()
        .next()
        .ok_or(IoError::PathsDatMissing { field: "r" })?;
    parse_f64(line, "r", token)
}

fn parse_atom_row(line: usize, text: &str) -> Result<PathsDatAtom> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 4 {
        return Err(IoError::PathsDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 4,
        });
    }
    let after_numbers = after_tokens(text, 4).ok_or(IoError::PathsDatMissing { field: "label" })?;
    let (label, tail) = quoted_label(line, after_numbers)?;
    let tail_tokens = tail.split_whitespace().collect::<Vec<_>>();
    let trailing = parse_optional_atom_tail(line, &tail_tokens)?;
    Ok(PathsDatAtom {
        position_angstrom: [
            parse_f64(line, "x", tokens[0])?,
            parse_f64(line, "y", tokens[1])?,
            parse_f64(line, "z", tokens[2])?,
        ],
        potential_index: parse_i32(line, "ipot", tokens[3])?,
        label: label.trim().to_string(),
        leg_distance_angstrom: trailing.map(|values| values.0),
        beta_degrees: trailing.map(|values| values.1),
        eta_degrees: trailing.map(|values| values.2),
    })
}

fn after_tokens(text: &str, count: usize) -> Option<&str> {
    let mut offset = 0_usize;
    let mut seen = 0_usize;
    for token in text.split_whitespace() {
        let relative = text.get(offset..)?.find(token)?;
        offset += relative + token.len();
        seen += 1;
        if seen == count {
            return text.get(offset..);
        }
    }
    None
}

fn quoted_label(line: usize, text: &str) -> Result<(&str, &str)> {
    let start = text
        .find('\'')
        .ok_or(IoError::PathsDatMissing { field: "label" })?;
    let rest = &text[start + 1..];
    let end = rest
        .find('\'')
        .ok_or_else(|| invalid_paths_dat("label", format!("line {line} has unterminated quote")))?;
    Ok((&rest[..end], &rest[end + 1..]))
}

fn parse_optional_atom_tail(line: usize, tokens: &[&str]) -> Result<Option<(f64, f64, f64)>> {
    let Some(first) = tokens.first() else {
        return Ok(None);
    };
    if first.parse::<f64>().is_err() {
        return Ok(None);
    }
    if tokens.len() < 3 {
        return Err(IoError::PathsDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 3,
        });
    }
    Ok(Some((
        parse_f64(line, "rleg", tokens[0])?,
        parse_f64(line, "beta", tokens[1])?,
        parse_f64(line, "eta", tokens[2])?,
    )))
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::PathsDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token.parse::<i32>().map_err(|_| IoError::PathsDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::PathsDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn ensure_nonempty_path(path: &PathsDatPath) -> Result<()> {
    if path.leg_count() == 0 {
        Err(invalid_paths_dat(
            "nleg",
            "path must contain at least one atom",
        ))
    } else {
        Ok(())
    }
}

fn angstrom_to_bohr(field: &'static str, value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(invalid_paths_dat(field, "value must be finite"));
    }
    let converted = value / FEFF_BOHR_ANGSTROM;
    if !converted.is_finite() {
        return Err(invalid_paths_dat(
            field,
            "converted bohr value must be finite",
        ));
    }
    Ok(converted)
}

fn fixed_label(label: &str) -> String {
    format!("{:<6}", label).chars().take(6).collect()
}

fn invalid_paths_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidPathsDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use refeff_core::{
        GenfmtNStarInput, GenfmtNStarRowsInput, PathDegeneracyGroup, PathDegeneracyProcessedRange,
        PathDegeneracyRange, PathDegeneracyReduction, PathDegeneracyRetention,
        PathDegeneracyRetentionDecision, PathDegeneracyRetentionReference, PathOutputImportance,
        genfmt_nstar_row, genfmt_nstar_rows, path_rotation_angles,
    };

    #[test]
    fn parses_full_pathsd_output() -> Result<()> {
        let parsed = parse_paths_dat(FULL_PATHS_DAT)?;
        assert_eq!(parsed.titles, vec!["TITLE Cu crystal"]);
        assert_eq!(parsed.paths.len(), 1);
        let path = &parsed.paths[0];
        assert_eq!(path.index, 1);
        assert_eq!(path.leg_count(), 3);
        assert_eq!(path.potential_indices().to_vec(), vec![1, 1, 0]);
        assert_eq!(path.atoms[0].label, "Cu1");
        assert_eq!(path.atoms[0].leg_distance_angstrom, Some(2.5527));
        Ok(())
    }

    #[test]
    fn builds_genfmt_path_from_paths_dat_like_rdpath() -> Result<()> {
        let parsed = parse_paths_dat(FULL_PATHS_DAT)?;
        let path = &parsed.paths[0];

        let genfmt_path = path.to_genfmt_path(1)?;

        assert_eq!(genfmt_path.index, 1);
        assert_eq!(genfmt_path.degeneracy, 4.0);
        assert_eq!(genfmt_path.potential_indices.to_vec(), vec![1, 1, 0]);
        assert_eq!(genfmt_path.positions_bohr.shape(), &[3, 3]);
        assert_close(
            genfmt_path.positions_bohr[(0, 0)],
            1.805 / FEFF_BOHR_ANGSTROM,
        );
        assert_close(
            genfmt_path.positions_bohr[(1, 0)],
            -1.805 / FEFF_BOHR_ANGSTROM,
        );
        assert_close(genfmt_path.positions_bohr[(2, 2)], 0.0);
        assert_close(
            genfmt_path.effective_half_path_length_bohr,
            2.5527 / FEFF_BOHR_ANGSTROM,
        );
        Ok(())
    }

    #[test]
    fn builds_pathsd_paths_dat_path_from_retained_group() -> Result<()> {
        let atom_positions = pathsd_reference_positions();
        let group = PathDegeneracyGroup {
            path_indices: vec![1, 2, 3, 4],
            degeneracy: 7,
            degeneracy_hash: 1.540_019_626_331_394E8,
            member_count: 2,
            coordinates: arr2(&[[0.0, 0.0, 0.0]]),
            reversed: false,
            symmetry_case: 1,
        };

        let path = pathsd_paths_dat_path(PathsdPathsDatPathInput {
            path_index: 3,
            group: &group,
            atom_positions: atom_positions.view(),
            atom_potentials: &[0, 1, 2, 3, 0, 1, 2],
            potential_labels: &["Cu0", "Cu1", "Cu2", "Cu3"],
        })?;

        assert_eq!(path.index, 3);
        assert_eq!(path.leg_count(), 5);
        assert_eq!(path.degeneracy, 7.0);
        assert_close(
            path.effective_half_path_length_angstrom,
            (1.118_034_005_165_1
                + 1.268_857_717_514_038
                + 2.598_076_343_536_377
                + 3.178_049_802_780_151
                + 1.603_121_995_925_903)
                / 2.0,
        );
        assert_eq!(path.atoms[0].position_angstrom, [1.1, 0.2, 0.0]);
        assert_eq!(path.atoms[0].potential_index, 1);
        assert_eq!(path.atoms[0].label, "Cu1");
        assert_close(
            path.atoms[0].leg_distance_angstrom.expect("rleg"),
            1.118_034_005_165_1,
        );
        assert_close(
            path.atoms[0].beta_degrees.expect("beta"),
            0.625_546_110_572_241_1_f64.to_degrees(),
        );
        assert_eq!(path.atoms[4].position_angstrom, [0.0, 0.0, 0.0]);
        assert_eq!(path.atoms[4].potential_index, 0);

        let rendered = paths_dat_string(&PathsDatData {
            titles: vec!["TITLE pathsd retained group".to_string()],
            paths: vec![path],
        })?;
        let reparsed = parse_paths_dat(&rendered)?;
        assert_eq!(reparsed.paths[0].index, 3);
        assert_eq!(reparsed.paths[0].leg_count(), 5);
        assert_eq!(reparsed.paths[0].atoms[4].label, "Cu0");
        Ok(())
    }

    #[test]
    fn builds_pathsd_paths_dat_data_from_retained_reduction() -> Result<()> {
        let atom_positions = pathsd_reference_positions();
        let titles = vec!["TITLE pathsd retained groups".to_string()];
        let reduction = sample_pathsd_reduction();

        let data = pathsd_paths_dat_data(PathsdPathsDatDataInput {
            titles: &titles,
            reduction: &reduction,
            atom_positions: atom_positions.view(),
            atom_potentials: &[0, 1, 2, 3, 0, 1, 2],
            potential_labels: &["Cu0", "Cu1", "Cu2", "Cu3"],
        })?;

        assert_eq!(data.titles, titles);
        assert_eq!(data.paths.len(), 2);
        assert_eq!(data.paths[0].index, 1);
        assert_eq!(data.paths[0].leg_count(), 3);
        assert_eq!(data.paths[0].degeneracy, 5.0);
        assert_eq!(data.paths[0].effective_half_path_length_angstrom, 2.5);
        assert_eq!(data.paths[0].atoms[0].position_angstrom, [-0.5, 1.7, 0.3]);
        assert_eq!(data.paths[0].atoms[1].position_angstrom, [0.7, -1.2, 0.8]);
        assert_eq!(data.paths[1].index, 2);
        assert_eq!(data.paths[1].leg_count(), 2);
        assert_eq!(data.paths[1].degeneracy, 2.0);
        assert_eq!(data.paths[1].effective_half_path_length_angstrom, 1.5);
        assert_eq!(data.paths[1].atoms[0].position_angstrom, [1.1, 0.2, 0.0]);

        let reparsed = parse_paths_dat(&paths_dat_string(&data)?)?;
        assert_eq!(reparsed.paths.len(), 2);
        assert_eq!(reparsed.paths[0].index, 1);
        assert_eq!(reparsed.paths[1].index, 2);
        Ok(())
    }

    #[test]
    fn pathsd_paths_dat_data_rejects_inconsistent_reduction() {
        let atom_positions = pathsd_reference_positions();
        let titles = vec!["TITLE pathsd retained groups".to_string()];
        let mut reduction = sample_pathsd_reduction();
        reduction.ranges[0].range.retention.decisions[0] = PathDegeneracyRetentionDecision {
            group_index: 9,
            port_importance: 1.0,
            fraction_percent: 100.0,
            retained: true,
        };

        assert!(matches!(
            pathsd_paths_dat_data(PathsdPathsDatDataInput {
                titles: &titles,
                reduction: &reduction,
                atom_positions: atom_positions.view(),
                atom_potentials: &[0, 1, 2, 3, 0, 1, 2],
                potential_labels: &["Cu0", "Cu1", "Cu2", "Cu3"],
            }),
            Err(IoError::InvalidPathsDat {
                field: "pathsd reduction",
                ..
            })
        ));

        let mut reduction = sample_pathsd_reduction();
        reduction.retained_unique_count = 3;
        assert!(matches!(
            pathsd_paths_dat_data(PathsdPathsDatDataInput {
                titles: &titles,
                reduction: &reduction,
                atom_positions: atom_positions.view(),
                atom_potentials: &[0, 1, 2, 3, 0, 1, 2],
                potential_labels: &["Cu0", "Cu1", "Cu2", "Cu3"],
            }),
            Err(IoError::InvalidPathsDat {
                field: "pathsd reduction",
                ..
            })
        ));
    }

    #[test]
    fn genfmt_path_views_feed_core_path_helpers() -> Result<()> {
        let parsed = parse_paths_dat(FULL_PATHS_DAT)?;
        let genfmt_paths = parsed.to_genfmt_paths(1)?;
        let genfmt_path = &genfmt_paths[0];

        assert_eq!(genfmt_path.leg_count(), 3);
        assert_eq!(genfmt_path.scattering_count(), Some(2));

        let angles = path_rotation_angles(genfmt_path.path_rotation_input(false))
            .expect("paths.dat geometry should be valid rdpath input");
        assert_close(
            angles.leg_lengths[0],
            (1.805_f64.hypot(1.805)) / FEFF_BOHR_ANGSTROM,
        );
        assert_close(
            angles.leg_lengths[2],
            (1.805_f64.hypot(1.805)) / FEFF_BOHR_ANGSTROM,
        );

        let nstar_path = genfmt_path.nstar_path_input();
        let nstar_row = genfmt_nstar_row(GenfmtNStarInput {
            path_number: 1,
            positions: nstar_path.positions,
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            degeneracy: nstar_path.degeneracy,
            initial_l: 1,
            ellipticity: 0.0,
        })
        .expect("paths.dat geometry should be valid nstar input");

        assert_eq!(nstar_row.path_number, 1);
        assert!(nstar_row.nstar.is_finite());
        Ok(())
    }

    #[test]
    fn builds_traversal_order_genfmt_core_inputs() -> Result<()> {
        let mut data = sample_paths_dat();
        let mut second = data.paths[0].clone();
        second.index = 9;
        second.degeneracy = 2.6;
        second.atoms[0].position_angstrom = [0.0, 2.0, 0.5];
        data.paths.push(second);

        let genfmt_paths = data.to_genfmt_paths(1)?;
        let rotation_inputs = genfmt_path_rotation_inputs(&genfmt_paths, false);
        let nstar_inputs = genfmt_nstar_path_inputs(&genfmt_paths);

        assert_eq!(rotation_inputs.len(), 2);
        assert_eq!(nstar_inputs.len(), 2);
        assert_close(
            rotation_inputs[1].positions[(0, 1)],
            2.0 / FEFF_BOHR_ANGSTROM,
        );
        assert_eq!(nstar_inputs[0].degeneracy, 4.0);
        assert_eq!(nstar_inputs[1].degeneracy, 2.6);

        let rows = genfmt_nstar_rows(GenfmtNStarRowsInput {
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            initial_l: 1,
            ellipticity: 0.0,
            path_inputs: &nstar_inputs,
        })
        .expect("paths.dat traversal inputs should be valid nstar input");

        assert_eq!(rows.rows.len(), 2);
        assert_eq!(rows.rows[0].path_number, 1);
        assert_eq!(rows.rows[1].path_number, 2);
        assert!(rows.rows.iter().all(|row| row.nstar.is_finite()));
        Ok(())
    }

    #[test]
    fn parses_single_scattering_rdinp_output() -> Result<()> {
        let parsed = parse_paths_dat(SINGLE_SCATTERING_PATHS_DAT)?;
        assert_eq!(parsed.paths.len(), 1);
        let path = &parsed.paths[0];
        assert_eq!(path.index, 7);
        assert_eq!(path.leg_count(), 2);
        assert_eq!(path.atoms[0].leg_distance_angstrom, None);
        assert_eq!(path.atoms[1].potential_index, 0);
        Ok(())
    }

    #[test]
    fn roundtrips_paths_dat_text() -> Result<()> {
        let data = sample_paths_dat();
        let parsed = parse_paths_dat(&paths_dat_string(&data)?)?;
        assert_eq!(parsed.titles, data.titles);
        assert_eq!(parsed.paths.len(), data.paths.len());
        assert_eq!(parsed.paths[0].index, data.paths[0].index);
        assert_eq!(parsed.paths[0].atoms.len(), data.paths[0].atoms.len());
        Ok(())
    }

    #[test]
    fn rejects_bad_rows() {
        assert!(matches!(
            parse_paths_dat("bad\n"),
            Err(IoError::PathsDatMissing { field: "separator" })
        ));
        assert!(matches!(
            parse_paths_dat("\nTITLE Cu\n"),
            Err(IoError::PathsDatMissing { field: "separator" })
        ));
        assert!(matches!(
            parse_paths_dat(" ---\n  a 2 3 r= 4\n"),
            Err(IoError::PathsDatParse { field: "index", .. })
        ));
        assert!(matches!(
            parse_paths_dat(MALFORMED_TRAILING_PATHS_DAT),
            Err(IoError::PathsDatRowWidth {
                line: 5,
                expected: 3,
                ..
            })
        ));
        let mut partial_optional = sample_paths_dat();
        partial_optional.paths[0].atoms[0].beta_degrees = None;
        assert!(matches!(
            paths_dat_string(&partial_optional),
            Err(IoError::InvalidPathsDat {
                field: "atom row",
                ..
            })
        ));

        let sample = sample_paths_dat();
        let path = &sample.paths[0];
        assert!(matches!(
            path.genfmt_potential_indices(0),
            Err(IoError::InvalidPathsDat { field: "ipot", .. })
        ));

        let mut negative_potential = sample_paths_dat();
        negative_potential.paths[0].atoms[0].potential_index = -1;
        assert!(matches!(
            negative_potential.paths[0].genfmt_potential_indices(1),
            Err(IoError::InvalidPathsDat { field: "ipot", .. })
        ));
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1.0e-12;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differs from expected {expected}"
        );
    }

    fn sample_paths_dat() -> PathsDatData {
        PathsDatData {
            titles: vec!["TITLE Cu crystal".to_string()],
            paths: vec![PathsDatPath {
                index: 1,
                degeneracy: 4.0,
                effective_half_path_length_angstrom: 2.5527,
                row_header:
                    "      x           y           z     ipot  label      rleg      beta        eta"
                        .to_string(),
                atoms: vec![
                    PathsDatAtom {
                        position_angstrom: [1.805, 1.805, 0.0],
                        potential_index: 1,
                        label: "Cu1".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(90.0),
                        eta_degrees: Some(45.0),
                    },
                    PathsDatAtom {
                        position_angstrom: [0.0, 0.0, 0.0],
                        potential_index: 0,
                        label: "Cu0".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(90.0),
                        eta_degrees: Some(225.0),
                    },
                ],
            }],
        }
    }

    fn pathsd_reference_positions() -> ndarray::Array2<f64> {
        arr2(&[
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
            [2.0, 1.0, 0.4],
            [-0.5, 1.7, 0.3],
            [0.7, -1.2, 0.8],
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
        ])
    }

    fn sample_pathsd_reduction() -> PathDegeneracyReduction {
        let reference = PathDegeneracyRetentionReference {
            port_importance: 1.0,
            degeneracy: 2,
        };
        PathDegeneracyReduction {
            ranges: vec![
                PathDegeneracyProcessedRange {
                    representative_total_path_length: 5.0,
                    range: PathDegeneracyRange {
                        groups: vec![
                            sample_pathsd_group(vec![1, 2], 2),
                            sample_pathsd_group(vec![3, 4], 5),
                        ],
                        importances: vec![dummy_path_importance(), dummy_path_importance()],
                        retention: PathDegeneracyRetention {
                            decisions: vec![
                                PathDegeneracyRetentionDecision {
                                    group_index: 0,
                                    port_importance: 1.0,
                                    fraction_percent: 100.0,
                                    retained: false,
                                },
                                PathDegeneracyRetentionDecision {
                                    group_index: 1,
                                    port_importance: 0.75,
                                    fraction_percent: 187.5,
                                    retained: true,
                                },
                            ],
                            retained_unique_count: 1,
                            retained_total_degeneracy: 5,
                            reference_group_index: Some(0),
                            reference_port_importance: Some(reference.port_importance),
                            reference_degeneracy: Some(reference.degeneracy),
                            reference: Some(reference),
                        },
                        normalization: 1.0,
                    },
                },
                PathDegeneracyProcessedRange {
                    representative_total_path_length: 3.0,
                    range: PathDegeneracyRange {
                        groups: vec![sample_pathsd_group(vec![1], 2)],
                        importances: vec![dummy_path_importance()],
                        retention: PathDegeneracyRetention {
                            decisions: vec![PathDegeneracyRetentionDecision {
                                group_index: 0,
                                port_importance: 2.0,
                                fraction_percent: 200.0,
                                retained: true,
                            }],
                            retained_unique_count: 1,
                            retained_total_degeneracy: 2,
                            reference_group_index: None,
                            reference_port_importance: Some(reference.port_importance),
                            reference_degeneracy: Some(reference.degeneracy),
                            reference: Some(reference),
                        },
                        normalization: 1.0,
                    },
                },
            ],
            retained_unique_count: 2,
            retained_total_degeneracy: 7,
            normalization: 1.0,
            retention_reference: Some(reference),
        }
    }

    fn sample_pathsd_group(path_indices: Vec<usize>, degeneracy: usize) -> PathDegeneracyGroup {
        PathDegeneracyGroup {
            path_indices,
            degeneracy,
            degeneracy_hash: degeneracy as f64,
            member_count: 1,
            coordinates: arr2(&[[0.0, 0.0, 0.0]]),
            reversed: false,
            symmetry_case: 1,
        }
    }

    fn dummy_path_importance() -> PathOutputImportance {
        PathOutputImportance {
            port_importance: 1.0,
            heap_importance: None,
            reversed_heap_importance: None,
            output_importance: None,
            normalization: 1.0,
        }
    }

    const FULL_PATHS_DAT: &str = "\
TITLE Cu crystal
 -----------------------------------------------------------------------
     1    3   4.000  index, nleg, degeneracy, r=  2.5527
      x           y           z     ipot  label      rleg      beta        eta
    1.805000    1.805000    0.000000   1 'Cu1   '     2.5527   90.0000   45.0000
   -1.805000    1.805000    0.000000   1 'Cu1   '     2.5527   90.0000  135.0000
    0.000000    0.000000    0.000000   0 'Cu0   '     2.5527   90.0000  225.0000
";

    const SINGLE_SCATTERING_PATHS_DAT: &str = "\
 TITLE Cu crystal
 Single scattering paths from ss lines cards in feff input
 -----------------------------------------------------------------------
   7   2   1.000  index,nleg,degeneracy,r=  2.0000
 single scattering
    2.000000    0.000000    0.000000   1 'Cu1   '
    0.000000    0.000000    0.000000   0 'Cu0   '  x,y,z,ipot
";

    const MALFORMED_TRAILING_PATHS_DAT: &str = "\
TITLE Cu crystal
 -----------------------------------------------------------------------
     1    1   4.000  index, nleg, degeneracy, r=  2.5527
      x           y           z     ipot  label      rleg      beta        eta
    1.805000    1.805000    0.000000   1 'Cu1   '     2.5527   90.0000
";

    /// Round-trip property coverage (F7): generators snap floats to the
    /// exact decimals `paths_dat_string` renders (`{:.6}` positions,
    /// `{:.3}` degeneracy, `{:.4}` `r`/`rleg`/`beta`/`eta`) and restrict
    /// labels to the 1-6 character alphanumeric span `fixed_label` preserves
    /// exactly, so `parse_paths_dat(paths_dat_string(data)) == data` holds
    /// without expected precision loss from an arbitrary unsnapped `f64`.
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn snap_fixed(value: f64, precision: usize) -> f64 {
            format!("{value:.precision$}")
                .parse::<f64>()
                .unwrap_or(value)
        }

        fn position_component() -> impl Strategy<Value = f64> {
            (-50_000_000_i64..50_000_000).prop_map(|n| snap_fixed(n as f64 / 1_000_000.0, 6))
        }

        fn degeneracy_strategy() -> impl Strategy<Value = f64> {
            (1_i64..999_999).prop_map(|n| snap_fixed(n as f64 / 1000.0, 3))
        }

        fn length_strategy() -> impl Strategy<Value = f64> {
            (1_i64..99_999_999).prop_map(|n| snap_fixed(n as f64 / 10_000.0, 4))
        }

        fn angle_strategy() -> impl Strategy<Value = f64> {
            (-3_600_000_i64..3_600_000).prop_map(|n| snap_fixed(n as f64 / 10_000.0, 4))
        }

        fn label_strategy() -> impl Strategy<Value = String> {
            "[A-Za-z][A-Za-z0-9]{0,5}"
        }

        #[allow(clippy::type_complexity)]
        fn atom_fields_strategy()
        -> impl Strategy<Value = (f64, f64, f64, i32, String, f64, f64, f64)> {
            (
                position_component(),
                position_component(),
                position_component(),
                0_i32..20,
                label_strategy(),
                length_strategy(),
                angle_strategy(),
                angle_strategy(),
            )
        }

        fn path_strategy() -> impl Strategy<Value = PathsDatPath> {
            (
                1_usize..1000,
                degeneracy_strategy(),
                length_strategy(),
                any::<bool>(),
                prop::collection::vec(atom_fields_strategy(), 1..4),
            )
                .prop_map(|(index, degeneracy, r, has_optional, atoms)| {
                    let atoms = atoms
                        .into_iter()
                        .map(|(x, y, z, ipot, label, rleg, beta, eta)| PathsDatAtom {
                            position_angstrom: [x, y, z],
                            potential_index: ipot,
                            label,
                            leg_distance_angstrom: has_optional.then_some(rleg),
                            beta_degrees: has_optional.then_some(beta),
                            eta_degrees: has_optional.then_some(eta),
                        })
                        .collect();
                    PathsDatPath {
                        index,
                        degeneracy,
                        effective_half_path_length_angstrom: r,
                        row_header:
                            "      x           y           z     ipot  label      rleg      beta        eta"
                                .to_string(),
                        atoms,
                    }
                })
        }

        fn titles_strategy() -> impl Strategy<Value = Vec<String>> {
            prop::collection::vec(
                "[A-Za-z0-9]{1,20}".prop_map(|body| format!("TITLE {body}")),
                1..3,
            )
        }

        fn paths_dat_strategy() -> impl Strategy<Value = PathsDatData> {
            (
                titles_strategy(),
                prop::collection::vec(path_strategy(), 1..3),
            )
                .prop_map(|(titles, paths)| PathsDatData { titles, paths })
        }

        proptest! {
            #[test]
            fn roundtrips_paths_dat(data in paths_dat_strategy()) {
                let rendered = paths_dat_string(&data)?;
                let reparsed = parse_paths_dat(&rendered)?;
                prop_assert_eq!(reparsed, data);
            }
        }
    }
}
