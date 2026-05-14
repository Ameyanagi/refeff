//! Minimal Crystallographic Information File reader for FEFF inputs.
//!
//! FEFF's RDINP stage accepts `CIF` cards and expands cell, symmetry, and
//! atom-site data before writing downstream handoff files.  This module starts
//! with the CIF subset used by the FEFF10 reference examples: scalar cell
//! parameters, space-group metadata, symmetry-operation loops, and fractional
//! atom-site loops.

use std::path::Path;

use crate::{IoError, Result};
use refeff_core::atomic_symbol;

const CIF_FRACTION_TOLERANCE: f64 = 0.0002;
const CIF_POSITION_TOLERANCE: f64 = 1.0e-5;

/// Parsed CIF data needed by FEFF's structure import path.
#[derive(Debug, Clone, PartialEq)]
pub struct CifDocument {
    /// Optional CIF data block name without the `data_` prefix.
    pub data_block: Option<String>,
    /// Unit-cell parameters.
    pub cell: CifCell,
    /// International Tables space-group number, when present.
    pub space_group_number: Option<i32>,
    /// Hermann-Mauguin space-group label, when present.
    pub space_group_hm: Option<String>,
    /// Symmetry operations in CIF text form.
    pub symmetry_operations: Vec<String>,
    /// Fractional atom-site rows.
    pub atom_sites: Vec<CifAtomSite>,
}

/// Unit-cell lengths and angles from CIF scalar fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CifCell {
    /// `a` unit-cell length in angstroms.
    pub a: f64,
    /// `b` unit-cell length in angstroms.
    pub b: f64,
    /// `c` unit-cell length in angstroms.
    pub c: f64,
    /// `alpha` unit-cell angle in degrees.
    pub alpha: f64,
    /// `beta` unit-cell angle in degrees.
    pub beta: f64,
    /// `gamma` unit-cell angle in degrees.
    pub gamma: f64,
}

/// One CIF atom site with fractional coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CifAtomSite {
    /// Element symbol from `_atom_site_type_symbol`, or a stripped label fallback.
    pub symbol: String,
    /// Original atom-site label, when present.
    pub label: Option<String>,
    /// Fractional coordinate along the crystallographic `a` axis.
    pub fract_x: f64,
    /// Fractional coordinate along the crystallographic `b` axis.
    pub fract_y: f64,
    /// Fractional coordinate along the crystallographic `c` axis.
    pub fract_z: f64,
}

/// CIF structure expanded through symmetry operations into FEFF coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CifExpandedStructure {
    /// FEFF lattice type such as `P`, `F`, `H`, or `CXY`.
    pub lattice_name: String,
    /// Compacted Hermann-Mauguin space-group label.
    pub space_group_hm: String,
    /// International Tables space-group number.
    pub space_group: i32,
    /// Lattice vectors in Angstrom Cartesian coordinates.
    pub lattice_vectors: [[f64; 3]; 3],
    /// One-based atom position index for the absorbing atom.
    pub absorber: usize,
    /// Atomic number of the absorbing CIF site.
    pub absorber_atomic_number: i32,
    /// FEFF label of the absorbing CIF site.
    pub absorber_label: String,
    /// FEFF `ppos` coordinates relative to the absorber and divided by `a`.
    pub positions: Vec<[f64; 3]>,
    /// Potential index for each expanded atom.
    pub potentials: Vec<i32>,
    /// Multiplicity of each inequivalent CIF atom site in the imported unit cell.
    pub site_multiplicities: Vec<usize>,
    /// Atomic number of each inequivalent CIF atom site.
    pub site_atomic_numbers: Vec<i32>,
    /// FEFF label of each inequivalent CIF atom site.
    pub site_labels: Vec<String>,
    /// FEFF labels: absorbing label first, followed by each inequivalent site label.
    pub labels: Vec<String>,
}

/// Real-space cluster generated from a CIF unit cell.
#[derive(Debug, Clone, PartialEq)]
pub struct CifCluster {
    /// Atoms sorted by distance from the absorber.
    pub atoms: Vec<CifClusterAtom>,
    /// Generated potential metadata for CIF files without a `POTENTIALS` card.
    pub potentials: Vec<CifPotential>,
}

/// One atom in a CIF-generated real-space cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CifClusterAtom {
    /// Cartesian x coordinate in Angstrom, relative to the absorber.
    pub x: f64,
    /// Cartesian y coordinate in Angstrom, relative to the absorber.
    pub y: f64,
    /// Cartesian z coordinate in Angstrom, relative to the absorber.
    pub z: f64,
    /// FEFF potential index, with `0` for the absorbing atom.
    pub potential: i32,
}

/// One generated potential from CIF inequivalent atom-site metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CifPotential {
    /// FEFF potential index.
    pub ipot: i32,
    /// Atomic number.
    pub atomic_number: i32,
    /// FEFF potential label.
    pub label: String,
    /// Unit-cell multiplicity for this potential.
    pub multiplicity: usize,
    /// Whether this row is the absorbing potential.
    pub absorber: bool,
}

/// FEFF CIF potential-equivalence selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CifEquivalence {
    /// Keep crystallographically inequivalent CIF sites as separate potentials.
    Crystallographic,
    /// Collapse CIF sites with the same atomic number into one potential.
    AtomicNumber,
    /// Use crystallographic sites until FEFF's potential limit, then collapse by atomic number.
    AutomaticLimit,
}

const FEFF_CIF_EQUIVALENCE_LIMIT: usize = 31;

#[derive(Debug, Default)]
struct CifBuilder {
    data_block: Option<String>,
    a: Option<f64>,
    b: Option<f64>,
    c: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    space_group_number: Option<i32>,
    space_group_hm: Option<String>,
    symmetry_operations: Vec<String>,
    atom_sites: Vec<CifAtomSite>,
}

/// Parse CIF text into the FEFF structure subset.
pub fn parse_cif(text: &str) -> Result<CifDocument> {
    let lines = text.lines().collect::<Vec<_>>();
    let mut builder = CifBuilder::default();
    let mut index = 0_usize;

    while index < lines.len() {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if let Some(name) = line.strip_prefix("data_") {
            builder.data_block = Some(name.trim().to_string());
            index += 1;
            continue;
        }
        if line.eq_ignore_ascii_case("loop_") {
            index = parse_loop(&lines, index + 1, &mut builder)?;
            continue;
        }
        if line.starts_with('_') {
            index = parse_scalar(&lines, index, &mut builder)?;
            continue;
        }
        if is_cif_text_field_line(lines[index]) {
            let (_, next_index) = read_cif_text_field(&lines, index)?;
            index = next_index;
            continue;
        }
        index += 1;
    }

    let cell = CifCell {
        a: required_f64(builder.a, "cell_length_a")?,
        b: required_f64(builder.b, "cell_length_b")?,
        c: required_f64(builder.c, "cell_length_c")?,
        alpha: required_f64(builder.alpha, "cell_angle_alpha")?,
        beta: required_f64(builder.beta, "cell_angle_beta")?,
        gamma: required_f64(builder.gamma, "cell_angle_gamma")?,
    };
    if builder.atom_sites.is_empty() {
        return Err(invalid_cif("atom_site", "missing atom-site loop"));
    }

    Ok(CifDocument {
        data_block: builder.data_block,
        cell,
        space_group_number: builder.space_group_number,
        space_group_hm: builder.space_group_hm,
        symmetry_operations: builder.symmetry_operations,
        atom_sites: builder.atom_sites,
    })
}

/// Read and parse CIF text from a file.
pub fn read_cif(path: impl AsRef<Path>) -> Result<CifDocument> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_cif(&text)
}

/// Expand a parsed CIF into the structure values FEFF writes to `reciprocal.inp`.
///
/// `target` follows FEFF's CIF convention: it is a one-based inequivalent CIF
/// atom-site index, not an expanded atom-position index.
pub fn expand_cif_structure(cif: &CifDocument, target: usize) -> Result<CifExpandedStructure> {
    expand_cif_structure_with_equivalence(cif, target, CifEquivalence::Crystallographic)
}

/// Expand a parsed CIF using FEFF's `EQUIVALENCE` potential grouping mode.
///
/// `EQUIVALENCE 1` keeps crystallographic site types. `EQUIVALENCE 2`
/// collapses sites by atomic number while preserving FEFF's first-seen ordering.
pub fn expand_cif_structure_with_equivalence(
    cif: &CifDocument,
    target: usize,
    equivalence: CifEquivalence,
) -> Result<CifExpandedStructure> {
    if target == 0 || target > cif.atom_sites.len() {
        return Err(invalid_cif(
            "TARGET",
            format!(
                "target {target} is outside the CIF atom-site range 1..={}",
                cif.atom_sites.len()
            ),
        ));
    }
    let space_group = cif_space_group_number(cif)?;
    let space_group_hm = cif_space_group_hm(cif, space_group)?;
    let lattice_name = cif_lattice_name(cif, &space_group_hm, space_group);
    let symmetry_operations = cif_symmetry_operations(cif, space_group)?;

    let mut fractional_positions = Vec::new();
    let mut potentials = Vec::new();
    let mut first_positions = Vec::with_capacity(cif.atom_sites.len());
    let mut site_labels = Vec::with_capacity(cif.atom_sites.len());
    let mut site_multiplicities = Vec::with_capacity(cif.atom_sites.len());
    let mut site_atomic_numbers = Vec::with_capacity(cif.atom_sites.len());

    for (site_index, site) in cif.atom_sites.iter().enumerate() {
        let site_position = [
            snap_cif_fraction(site.fract_x),
            snap_cif_fraction(site.fract_y),
            snap_cif_fraction(site.fract_z),
        ];
        let transformed = symmetry_operations
            .iter()
            .map(|operation| apply_cif_symmetry_operation(operation, site_position))
            .collect::<Result<Vec<_>>>()?;
        let unique_indices = unique_symmetry_indices(&transformed, &lattice_name);

        first_positions.push(fractional_positions.len());
        site_multiplicities.push(unique_indices.len());
        site_atomic_numbers.push(cif_site_atomic_number(site)?);
        for index in unique_indices {
            fractional_positions.push(transformed[index]);
            potentials.push(
                i32::try_from(site_index + 1).map_err(|_| {
                    invalid_cif("atom_site", "too many inequivalent CIF atom sites")
                })?,
            );
        }
        site_labels.push(cif_site_label(site));
    }

    let absorber_atomic_number = site_atomic_numbers[target - 1];
    let absorber_label = site_labels[target - 1].clone();
    let equivalence = match equivalence {
        CifEquivalence::AutomaticLimit if cif.atom_sites.len() > FEFF_CIF_EQUIVALENCE_LIMIT => {
            CifEquivalence::AtomicNumber
        }
        CifEquivalence::AutomaticLimit => CifEquivalence::Crystallographic,
        value => value,
    };
    if equivalence == CifEquivalence::AtomicNumber {
        collapse_potentials_by_atomic_number(
            &mut potentials,
            &mut site_multiplicities,
            &mut site_atomic_numbers,
            &mut site_labels,
        )?;
    }

    let lattice_vectors = cif_lattice_vectors(cif.cell)?;
    let absorber_index = first_positions[target - 1];
    let mut cartesian_positions = fractional_positions
        .iter()
        .map(|position| fractional_to_cartesian(*position, lattice_vectors))
        .collect::<Vec<_>>();
    let origin = cartesian_positions[absorber_index];
    for position in &mut cartesian_positions {
        for axis in 0..3 {
            position[axis] -= origin[axis];
        }
    }
    wrap_to_nearest_images(&mut cartesian_positions, absorber_index, lattice_vectors);

    let positions = cartesian_positions
        .into_iter()
        .map(|position| {
            [
                position[0] / cif.cell.a,
                position[1] / cif.cell.a,
                position[2] / cif.cell.a,
            ]
        })
        .collect::<Vec<_>>();
    let mut labels = Vec::with_capacity(site_labels.len() + 1);
    labels.push(absorber_label.clone());
    labels.extend(site_labels.iter().cloned());

    Ok(CifExpandedStructure {
        lattice_name,
        space_group_hm,
        space_group,
        lattice_vectors,
        absorber: absorber_index + 1,
        absorber_atomic_number,
        absorber_label,
        positions,
        potentials,
        site_multiplicities,
        site_atomic_numbers,
        site_labels,
        labels,
    })
}

struct AtomicNumberGroup {
    atomic_number: i32,
    label: String,
    multiplicity: usize,
}

fn collapse_potentials_by_atomic_number(
    potentials: &mut [i32],
    site_multiplicities: &mut Vec<usize>,
    site_atomic_numbers: &mut Vec<i32>,
    site_labels: &mut Vec<String>,
) -> Result<()> {
    let original_atomic_numbers = site_atomic_numbers.clone();
    let original_labels = site_labels.clone();
    let mut groups: Vec<AtomicNumberGroup> = Vec::new();
    let mut site_to_group = vec![0_i32; original_atomic_numbers.len()];

    for potential in potentials.iter().copied() {
        let site_index = usize::try_from(potential - 1)
            .map_err(|_| invalid_cif("atom_site", "invalid CIF potential index"))?;
        let Some(atomic_number) = original_atomic_numbers.get(site_index).copied() else {
            return Err(invalid_cif(
                "atom_site",
                "CIF potential index is out of range",
            ));
        };

        if let Some(group_index) = groups
            .iter()
            .position(|group| group.atomic_number == atomic_number)
        {
            groups[group_index].multiplicity += 1;
            site_to_group[site_index] = i32::try_from(group_index + 1)
                .map_err(|_| invalid_cif("atom_site", "too many CIF potential rows"))?;
        } else {
            groups.push(AtomicNumberGroup {
                atomic_number,
                label: original_labels[site_index].clone(),
                multiplicity: 1,
            });
            site_to_group[site_index] = i32::try_from(groups.len())
                .map_err(|_| invalid_cif("atom_site", "too many CIF potential rows"))?;
        }
    }

    for potential in potentials {
        let site_index = usize::try_from(*potential - 1)
            .map_err(|_| invalid_cif("atom_site", "invalid CIF potential index"))?;
        let Some(group_index) = site_to_group.get(site_index).copied() else {
            return Err(invalid_cif(
                "atom_site",
                "CIF potential index is out of range",
            ));
        };
        *potential = group_index;
    }

    *site_multiplicities = groups.iter().map(|group| group.multiplicity).collect();
    *site_atomic_numbers = groups.iter().map(|group| group.atomic_number).collect();
    *site_labels = groups.into_iter().map(|group| group.label).collect();
    Ok(())
}

/// Expand a CIF into the real-space atom cluster FEFF's `rdinp` stage writes.
///
/// `rmax` is the largest requested cluster radius in Angstrom. FEFF builds at
/// least an 8 Angstrom periodic supercell before truncating to full shells.
pub fn expand_cif_cluster(cif: &CifDocument, target: usize, rmax: f64) -> Result<CifCluster> {
    expand_cif_cluster_with_equivalence(cif, target, rmax, CifEquivalence::Crystallographic)
}

/// Expand a CIF into a real-space cluster using FEFF's `EQUIVALENCE` mode.
pub fn expand_cif_cluster_with_equivalence(
    cif: &CifDocument,
    target: usize,
    rmax: f64,
    equivalence: CifEquivalence,
) -> Result<CifCluster> {
    let structure = expand_cif_structure_with_equivalence(cif, target, equivalence)?;
    let atoms = cif_cluster_atoms(
        &structure,
        vector_length(structure.lattice_vectors[0]),
        rmax,
    );
    let potentials = cif_cluster_potentials(&structure)?;
    Ok(CifCluster { atoms, potentials })
}

fn parse_scalar(lines: &[&str], index: usize, builder: &mut CifBuilder) -> Result<usize> {
    let tokens = tokenize_cif_line(lines[index]);
    let Some(key) = tokens.first() else {
        return Ok(index + 1);
    };
    let (value, next_index) = if let Some(value) = tokens.get(1) {
        (Some(value.clone()), index + 1)
    } else if lines
        .get(index + 1)
        .is_some_and(|line| is_cif_text_field_line(line))
    {
        let (value, next_index) = read_cif_text_field(lines, index + 1)?;
        (Some(value), next_index)
    } else {
        (None, index + 1)
    };
    let value = value.as_deref();

    match key.as_str() {
        "_cell_length_a" => builder.a = Some(parse_cif_f64(value, key)?),
        "_cell_length_b" => builder.b = Some(parse_cif_f64(value, key)?),
        "_cell_length_c" => builder.c = Some(parse_cif_f64(value, key)?),
        "_cell_angle_alpha" => builder.alpha = Some(parse_cif_f64(value, key)?),
        "_cell_angle_beta" => builder.beta = Some(parse_cif_f64(value, key)?),
        "_cell_angle_gamma" => builder.gamma = Some(parse_cif_f64(value, key)?),
        "_space_group_IT_number" | "_symmetry_Int_Tables_number" => {
            builder.space_group_number = Some(parse_cif_i32(value, key)?);
        }
        "_symmetry_space_group_name_H-M"
        | "_space_group_name_H-M_alt"
        | "_[local]_cod_cif_authors_sg_H-M"
            if builder.space_group_hm.is_none() =>
        {
            builder.space_group_hm = value.map(normalize_cif_string);
        }
        _ => {}
    }
    Ok(next_index)
}

fn parse_loop(lines: &[&str], mut index: usize, builder: &mut CifBuilder) -> Result<usize> {
    let mut headers = Vec::new();
    let mut row_tokens = Vec::new();
    while let Some(line) = lines.get(index).map(|line| line.trim()) {
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if !line.starts_with('_') {
            break;
        }
        let tokens = tokenize_cif_line(line);
        let header_count = tokens
            .iter()
            .take_while(|token| token.starts_with('_'))
            .count();
        headers.extend(tokens.iter().take(header_count).cloned());
        row_tokens.extend(tokens.into_iter().skip(header_count));
        index += 1;
        if !row_tokens.is_empty() {
            break;
        }
    }

    while !headers.is_empty() && row_tokens.len() >= headers.len() {
        let row = row_tokens.drain(..headers.len()).collect::<Vec<_>>();
        apply_loop_row(&headers, &row, builder)?;
    }

    while let Some(raw) = lines.get(index) {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if line.eq_ignore_ascii_case("loop_") || line.starts_with("data_") || line.starts_with('_')
        {
            break;
        }
        if is_cif_text_field_line(raw) {
            let (value, next_index) = read_cif_text_field(lines, index)?;
            row_tokens.push(value);
            while !headers.is_empty() && row_tokens.len() >= headers.len() {
                let row = row_tokens.drain(..headers.len()).collect::<Vec<_>>();
                apply_loop_row(&headers, &row, builder)?;
            }
            index = next_index;
            continue;
        }

        row_tokens.extend(tokenize_cif_line(line));
        while !headers.is_empty() && row_tokens.len() >= headers.len() {
            let row = row_tokens.drain(..headers.len()).collect::<Vec<_>>();
            apply_loop_row(&headers, &row, builder)?;
        }
        index += 1;
    }
    Ok(index)
}

fn cif_space_group_number(cif: &CifDocument) -> Result<i32> {
    cif.space_group_number
        .or_else(|| {
            cif.space_group_hm
                .as_deref()
                .and_then(compacted_space_group_number)
        })
        .ok_or_else(|| invalid_cif("space_group", "missing space-group number"))
}

fn cif_space_group_hm(cif: &CifDocument, space_group: i32) -> Result<String> {
    if let Some(space_group_hm) = &cif.space_group_hm {
        let compacted = compact_space_group_hm(space_group_hm);
        if !compacted.is_empty() {
            return Ok(compacted);
        }
    }
    compacted_space_group_name(space_group)
        .map(str::to_string)
        .ok_or_else(|| invalid_cif("space_group", "missing H-M space-group label"))
}

fn compact_space_group_hm(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch == '{' || ch == ':' {
            break;
        }
        if !ch.is_whitespace() && ch != '\'' && ch != '"' {
            let adjusted = if !out.is_empty() && matches!(ch, 'A' | 'B' | 'C' | 'M') {
                ch.to_ascii_lowercase()
            } else {
                ch
            };
            out.push(adjusted);
        }
        if out.len() >= 8 {
            break;
        }
    }
    out
}

fn compacted_space_group_number(space_group_hm: &str) -> Option<i32> {
    match compact_space_group_hm(space_group_hm).as_str() {
        "P1" => Some(1),
        "P63/mmc" => Some(194),
        "Fm-3m" => Some(225),
        _ => None,
    }
}

fn compacted_space_group_name(space_group: i32) -> Option<&'static str> {
    match space_group {
        1 => Some("P1"),
        194 => Some("P63/mmc"),
        225 => Some("Fm-3m"),
        _ => None,
    }
}

fn cif_lattice_name(cif: &CifDocument, space_group_hm: &str, space_group: i32) -> String {
    if (168..=194).contains(&space_group)
        && nearly_equal(cif.cell.alpha, 90.0, CIF_FRACTION_TOLERANCE)
        && nearly_equal(cif.cell.beta, 90.0, CIF_FRACTION_TOLERANCE)
        && nearly_equal(cif.cell.gamma, 120.0, CIF_FRACTION_TOLERANCE)
    {
        return "H".to_string();
    }

    match space_group_hm
        .chars()
        .next()
        .unwrap_or('P')
        .to_ascii_uppercase()
    {
        'A' => "CYZ",
        'B' => "CXZ",
        'C' => "CXY",
        'F' => "F",
        'I' => "B",
        'R' => "R",
        _ => "P",
    }
    .to_string()
}

fn cif_symmetry_operations(cif: &CifDocument, space_group: i32) -> Result<Vec<String>> {
    if !cif.symmetry_operations.is_empty() {
        return Ok(cif.symmetry_operations.clone());
    }
    if space_group == 1 {
        return Ok(vec!["x,y,z".to_string()]);
    }
    Err(invalid_cif("symmetry", "missing CIF symmetry operations"))
}

fn apply_cif_symmetry_operation(operation: &str, position: [f64; 3]) -> Result<[f64; 3]> {
    let parts = operation.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(invalid_cif(
            "symmetry",
            format!("operation {operation:?} must have three comma-separated fields"),
        ));
    }

    Ok([
        wrap_cif_fraction(evaluate_cif_axis_expr(parts[0], position, operation)?),
        wrap_cif_fraction(evaluate_cif_axis_expr(parts[1], position, operation)?),
        wrap_cif_fraction(evaluate_cif_axis_expr(parts[2], position, operation)?),
    ])
}

fn evaluate_cif_axis_expr(expr: &str, position: [f64; 3], operation: &str) -> Result<f64> {
    let expr = expr.replace(char::is_whitespace, "").to_ascii_lowercase();
    if expr.is_empty() {
        return Err(invalid_cif(
            "symmetry",
            format!("empty axis expression in {operation:?}"),
        ));
    }

    let mut value = 0.0;
    let mut start = 0_usize;
    let bytes = expr.as_bytes();
    while start < expr.len() {
        let mut sign = 1.0;
        if bytes[start] == b'+' {
            start += 1;
        } else if bytes[start] == b'-' {
            sign = -1.0;
            start += 1;
        }
        if start >= expr.len() {
            return Err(invalid_cif(
                "symmetry",
                format!("missing term after sign in {operation:?}"),
            ));
        }

        let mut end = start;
        while end < expr.len() && !matches!(bytes[end], b'+' | b'-') {
            end += 1;
        }
        let term = &expr[start..end];
        value += sign * evaluate_cif_axis_term(term, position, operation)?;
        start = end;
    }

    Ok(value)
}

fn evaluate_cif_axis_term(term: &str, position: [f64; 3], operation: &str) -> Result<f64> {
    match term {
        "x" => Ok(position[0]),
        "y" => Ok(position[1]),
        "z" => Ok(position[2]),
        _ if term.contains('/') => parse_cif_fraction_term(term, operation),
        _ => term.parse::<f64>().map(snap_cif_fraction).map_err(|_| {
            invalid_cif(
                "symmetry",
                format!("invalid term {term:?} in operation {operation:?}"),
            )
        }),
    }
}

fn parse_cif_fraction_term(term: &str, operation: &str) -> Result<f64> {
    let Some((numerator, denominator)) = term.split_once('/') else {
        return Err(invalid_cif(
            "symmetry",
            format!("invalid fraction {term:?} in operation {operation:?}"),
        ));
    };
    let numerator = numerator.parse::<f64>().map_err(|_| {
        invalid_cif(
            "symmetry",
            format!("invalid numerator {numerator:?} in operation {operation:?}"),
        )
    })?;
    let denominator = denominator.parse::<f64>().map_err(|_| {
        invalid_cif(
            "symmetry",
            format!("invalid denominator {denominator:?} in operation {operation:?}"),
        )
    })?;
    if denominator == 0.0 {
        return Err(invalid_cif(
            "symmetry",
            format!("zero denominator in operation {operation:?}"),
        ));
    }
    Ok(numerator / denominator)
}

fn unique_symmetry_indices(positions: &[[f64; 3]], lattice_name: &str) -> Vec<usize> {
    let mut unique = Vec::new();
    for (index, position) in positions.iter().enumerate() {
        if unique.iter().all(|unique_index| {
            !same_fractional_position(lattice_name, positions[*unique_index], *position)
        }) {
            unique.push(index);
        }
    }
    unique
}

fn same_fractional_position(lattice_name: &str, lhs: [f64; 3], rhs: [f64; 3]) -> bool {
    centering_translations(lattice_name)
        .iter()
        .any(|translation| {
            let dx = [
                periodic_delta(lhs[0] - rhs[0] - translation[0]),
                periodic_delta(lhs[1] - rhs[1] - translation[1]),
                periodic_delta(lhs[2] - rhs[2] - translation[2]),
            ];
            vector_length(dx) < CIF_POSITION_TOLERANCE
        })
}

fn centering_translations(lattice_name: &str) -> Vec<[f64; 3]> {
    let mut translations = Vec::with_capacity(39);
    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                translations.push([f64::from(i), f64::from(j), f64::from(k)]);
            }
        }
    }

    match lattice_name {
        "B" | "I" => {
            for i in [-1.0, 1.0] {
                for j in [-1.0, 1.0] {
                    for k in [-1.0, 1.0] {
                        translations.push([0.5 * i, 0.5 * j, 0.5 * k]);
                    }
                }
            }
        }
        "F" => {
            for i in [-1.0, 1.0] {
                for j in [-1.0, 1.0] {
                    translations.push([0.5 * i, 0.5 * j, 0.0]);
                    translations.push([0.5 * i, 0.0, 0.5 * j]);
                    translations.push([0.0, 0.5 * i, 0.5 * j]);
                }
            }
        }
        "CXY" => {
            for i in [-1.0, 1.0] {
                for j in [-1.0, 1.0] {
                    translations.push([0.5 * i, 0.5 * j, 0.0]);
                }
            }
        }
        "CXZ" => {
            for i in [-1.0, 1.0] {
                for j in [-1.0, 1.0] {
                    translations.push([0.5 * i, 0.0, 0.5 * j]);
                }
            }
        }
        "CYZ" => {
            for i in [-1.0, 1.0] {
                for j in [-1.0, 1.0] {
                    translations.push([0.0, 0.5 * i, 0.5 * j]);
                }
            }
        }
        _ => {}
    }
    translations
}

fn periodic_delta(value: f64) -> f64 {
    let small2 = CIF_POSITION_TOLERANCE / 2.0;
    let mut value = (value + 10.0 + small2).rem_euclid(1.0) - small2;
    if (value - 1.0).abs() < small2 {
        value = 0.0;
    }
    value
}

fn cif_lattice_vectors(cell: CifCell) -> Result<[[f64; 3]; 3]> {
    let alpha = cell.alpha.to_radians();
    let beta = cell.beta.to_radians();
    let gamma = cell.gamma.to_radians();
    let sin_alpha = alpha.sin();
    if sin_alpha.abs() < f64::EPSILON {
        return Err(invalid_cif(
            "cell_angle_alpha",
            "alpha angle cannot be 0 or 180 degrees",
        ));
    }

    let a_z = beta.cos() * cell.a;
    let a_y = ((gamma.cos() - beta.cos() * alpha.cos()) / sin_alpha) * cell.a;
    let a_x_squared = cell.a.mul_add(cell.a, -(a_y.mul_add(a_y, a_z * a_z)));
    if a_x_squared < -CIF_POSITION_TOLERANCE {
        return Err(invalid_cif(
            "cell",
            "cell parameters do not define a real lattice",
        ));
    }
    let a_x = a_x_squared.max(0.0).sqrt();

    Ok([
        [a_x, a_y, a_z],
        [0.0, sin_alpha * cell.b, alpha.cos() * cell.b],
        [0.0, 0.0, cell.c],
    ])
}

fn fractional_to_cartesian(position: [f64; 3], lattice_vectors: [[f64; 3]; 3]) -> [f64; 3] {
    [
        position[0].mul_add(
            lattice_vectors[0][0],
            position[1].mul_add(lattice_vectors[1][0], position[2] * lattice_vectors[2][0]),
        ),
        position[0].mul_add(
            lattice_vectors[0][1],
            position[1].mul_add(lattice_vectors[1][1], position[2] * lattice_vectors[2][1]),
        ),
        position[0].mul_add(
            lattice_vectors[0][2],
            position[1].mul_add(lattice_vectors[1][2], position[2] * lattice_vectors[2][2]),
        ),
    ]
}

fn wrap_to_nearest_images(
    positions: &mut [[f64; 3]],
    absorber_index: usize,
    lattice_vectors: [[f64; 3]; 3],
) {
    let translations = nearest_lattice_translations(lattice_vectors);
    for (index, position) in positions.iter_mut().enumerate() {
        if index == absorber_index {
            continue;
        }
        if let Some(translated) = translations.iter().min_by(|lhs, rhs| {
            let lhs_distance = vector_length(add_vectors(*position, **lhs));
            let rhs_distance = vector_length(add_vectors(*position, **rhs));
            lhs_distance.total_cmp(&rhs_distance)
        }) {
            *position = add_vectors(*position, *translated);
        }
    }
}

fn nearest_lattice_translations(lattice_vectors: [[f64; 3]; 3]) -> Vec<[f64; 3]> {
    let mut translations = Vec::with_capacity(27);
    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                translations.push([
                    f64::from(i).mul_add(
                        lattice_vectors[0][0],
                        f64::from(j)
                            .mul_add(lattice_vectors[1][0], f64::from(k) * lattice_vectors[2][0]),
                    ),
                    f64::from(i).mul_add(
                        lattice_vectors[0][1],
                        f64::from(j)
                            .mul_add(lattice_vectors[1][1], f64::from(k) * lattice_vectors[2][1]),
                    ),
                    f64::from(i).mul_add(
                        lattice_vectors[0][2],
                        f64::from(j)
                            .mul_add(lattice_vectors[1][2], f64::from(k) * lattice_vectors[2][2]),
                    ),
                ]);
            }
        }
    }
    translations
}

fn add_vectors(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn vector_length(vector: [f64; 3]) -> f64 {
    vector[0]
        .mul_add(
            vector[0],
            vector[1].mul_add(vector[1], vector[2] * vector[2]),
        )
        .sqrt()
}

fn snap_cif_fraction(value: f64) -> f64 {
    let value = snap_to_denominator(value, 12);
    snap_to_denominator(value, 8)
}

fn snap_to_denominator(value: f64, denominator: i32) -> f64 {
    (-2 * denominator..=2 * denominator)
        .map(|numerator| f64::from(numerator) / f64::from(denominator))
        .find(|candidate| (value - candidate).abs() < CIF_FRACTION_TOLERANCE)
        .unwrap_or(value)
}

fn wrap_cif_fraction(value: f64) -> f64 {
    let wrapped = snap_cif_fraction(value).rem_euclid(1.0);
    if nearly_equal(wrapped, 1.0, CIF_POSITION_TOLERANCE) {
        0.0
    } else {
        wrapped
    }
}

fn nearly_equal(lhs: f64, rhs: f64, tolerance: f64) -> bool {
    (lhs - rhs).abs() < tolerance
}

fn cif_site_label(site: &CifAtomSite) -> String {
    let candidate = if site.symbol.is_empty() {
        site.label.as_deref().unwrap_or("")
    } else {
        &site.symbol
    };
    let stripped = strip_element_label(candidate);
    let symbol: String = if stripped.is_empty() {
        candidate.chars().take(2).collect()
    } else {
        stripped.chars().take(2).collect()
    };
    canonical_element_symbol(&symbol)
}

fn cif_site_atomic_number(site: &CifAtomSite) -> Result<i32> {
    let label = cif_site_label(site);
    for atomic_number in 1..=139_usize {
        let symbol = atomic_symbol(atomic_number).map_err(|err| {
            invalid_cif(
                "atom_site",
                format!("failed to query atomic symbol table: {err}"),
            )
        })?;
        if symbol.eq_ignore_ascii_case(&label) {
            return i32::try_from(atomic_number)
                .map_err(|_| invalid_cif("atom_site", "atomic number out of range"));
        }
    }
    Err(invalid_cif(
        "atom_site",
        format!("unknown atom-site element label {label:?}"),
    ))
}

fn canonical_element_symbol(symbol: &str) -> String {
    let mut chars = symbol.chars();
    let mut out = String::new();
    if let Some(first) = chars.next() {
        out.push(first.to_ascii_uppercase());
    }
    if let Some(second) = chars.next() {
        out.push(second.to_ascii_lowercase());
    }
    out
}

fn cif_cluster_potentials(structure: &CifExpandedStructure) -> Result<Vec<CifPotential>> {
    let mut potentials = Vec::with_capacity(structure.site_labels.len() + 1);
    potentials.push(CifPotential {
        ipot: 0,
        atomic_number: structure.absorber_atomic_number,
        label: structure.absorber_label.clone(),
        multiplicity: 0,
        absorber: true,
    });

    for (index, ((atomic_number, label), multiplicity)) in structure
        .site_atomic_numbers
        .iter()
        .zip(&structure.site_labels)
        .zip(&structure.site_multiplicities)
        .enumerate()
    {
        potentials.push(CifPotential {
            ipot: i32::try_from(index + 1)
                .map_err(|_| invalid_cif("atom_site", "too many CIF potential rows"))?,
            atomic_number: *atomic_number,
            label: label.clone(),
            multiplicity: *multiplicity,
            absorber: false,
        });
    }
    Ok(potentials)
}

fn cif_cluster_atoms(
    structure: &CifExpandedStructure,
    lattice_scale: f64,
    rmax: f64,
) -> Vec<CifClusterAtom> {
    let [a1, a2, a3] = structure.lattice_vectors;
    let ratomslist = 8.0_f64.max(1.33 * rmax.max(0.0));
    let i1 = cell_repeat_count(ratomslist, a1);
    let i2 = cell_repeat_count(ratomslist, a2);
    let i3 = cell_repeat_count(ratomslist, a3);
    let shifts = lattice_centering_shifts(&structure.lattice_name);
    let absorber_index = structure.absorber - 1;

    let mut atoms = Vec::new();
    let mut index_absorber = 0_usize;
    for j1 in -i1..=i1 {
        for j2 in -i2..=i2 {
            for j3 in -i3..=i3 {
                for (index, (position, potential)) in structure
                    .positions
                    .iter()
                    .zip(&structure.potentials)
                    .enumerate()
                {
                    let base = add_vectors(
                        scale_vector(*position, lattice_scale),
                        lattice_translation(j1, j2, j3, a1, a2, a3),
                    );
                    let mut atom_potential = *potential;
                    if j1 == 0 && j2 == 0 && j3 == 0 && index == absorber_index {
                        atom_potential = 0;
                        index_absorber = atoms.len();
                    }
                    atoms.push(CifClusterAtom {
                        x: base[0],
                        y: base[1],
                        z: base[2],
                        potential: atom_potential,
                    });

                    for shift in &shifts {
                        let shifted =
                            add_vectors(base, fractional_to_cartesian(*shift, [a1, a2, a3]));
                        atoms.push(CifClusterAtom {
                            x: shifted[0],
                            y: shifted[1],
                            z: shifted[2],
                            potential: *potential,
                        });
                    }
                }
            }
        }
    }

    feff_sort_cluster_atoms(&mut atoms, index_absorber);
    let cutoff = (vector_length(a1) * f64::from(i1))
        .min(vector_length(a2) * f64::from(i1))
        .min(vector_length(a3) * f64::from(i1));
    let keep = atoms
        .iter()
        .position(|atom| cluster_atom_distance(*atom) > cutoff)
        .unwrap_or(atoms.len());
    atoms.truncate(keep);
    atoms
}

fn cell_repeat_count(radius: f64, lattice_vector: [f64; 3]) -> i32 {
    (radius / vector_length(lattice_vector)).trunc() as i32 + 1
}

fn lattice_centering_shifts(lattice_name: &str) -> Vec<[f64; 3]> {
    match lattice_name {
        "F" => vec![[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        "CXY" => vec![[0.5, 0.5, 0.0]],
        "CXZ" => vec![[0.5, 0.0, 0.5]],
        "CYZ" => vec![[0.0, 0.5, 0.5]],
        "B" | "I" => vec![[0.5, 0.5, 0.5]],
        _ => Vec::new(),
    }
}

fn lattice_translation(
    j1: i32,
    j2: i32,
    j3: i32,
    a1: [f64; 3],
    a2: [f64; 3],
    a3: [f64; 3],
) -> [f64; 3] {
    add_vectors(
        add_vectors(
            scale_vector(a1, f64::from(j1)),
            scale_vector(a2, f64::from(j2)),
        ),
        scale_vector(a3, f64::from(j3)),
    )
}

fn scale_vector(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn cluster_atom_distance(atom: CifClusterAtom) -> f64 {
    vector_length([atom.x, atom.y, atom.z])
}

fn feff_sort_cluster_atoms(atoms: &mut [CifClusterAtom], mut absorber_index: usize) {
    let mut distances = atoms
        .iter()
        .map(|atom| cluster_atom_distance(*atom))
        .collect::<Vec<_>>();
    for i in 0..atoms.len() {
        let mut min_index = i;
        let mut min_distance = distances[i];
        for (j, distance) in distances.iter().enumerate().skip(i) {
            if *distance < min_distance {
                min_index = j;
                min_distance = *distance;
            }
        }
        atoms.swap(i, min_index);
        distances[min_index] = distances[i];
        distances[i] = min_distance;
        if i == absorber_index {
            absorber_index = min_index;
        }
        if min_index == absorber_index {
            absorber_index = i;
        }
    }
}

fn apply_loop_row(headers: &[String], row: &[String], builder: &mut CifBuilder) -> Result<()> {
    if let Some(op_index) = headers.iter().position(|header| {
        header == "_symmetry_equiv_pos_as_xyz" || header == "_space_group_symop_operation_xyz"
    }) {
        if let Some(operation) = row.get(op_index) {
            builder
                .symmetry_operations
                .push(normalize_cif_string(operation));
        }
        return Ok(());
    }

    let x_index = headers
        .iter()
        .position(|header| header == "_atom_site_fract_x");
    let y_index = headers
        .iter()
        .position(|header| header == "_atom_site_fract_y");
    let z_index = headers
        .iter()
        .position(|header| header == "_atom_site_fract_z");
    let Some((x_index, y_index, z_index)) = x_index
        .zip(y_index)
        .zip(z_index)
        .map(|((x, y), z)| (x, y, z))
    else {
        return Ok(());
    };

    let label = headers
        .iter()
        .position(|header| header == "_atom_site_label")
        .and_then(|idx| row.get(idx))
        .map(|value| normalize_cif_string(value));
    let symbol = headers
        .iter()
        .position(|header| header == "_atom_site_type_symbol")
        .and_then(|idx| row.get(idx))
        .map(|value| normalize_cif_string(value))
        .or_else(|| label.as_deref().map(strip_element_label))
        .ok_or_else(|| invalid_cif("atom_site", "missing atom symbol or label"))?;

    builder.atom_sites.push(CifAtomSite {
        symbol,
        label,
        fract_x: parse_cif_f64(row.get(x_index).map(String::as_str), "_atom_site_fract_x")?,
        fract_y: parse_cif_f64(row.get(y_index).map(String::as_str), "_atom_site_fract_y")?,
        fract_z: parse_cif_f64(row.get(z_index).map(String::as_str), "_atom_site_fract_z")?,
    });
    Ok(())
}

fn tokenize_cif_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in line.chars() {
        if let Some(quote_char) = quote {
            if ch == quote_char {
                quote = None;
            } else {
                current.push(ch);
            }
            continue;
        }
        if ch == '#' {
            break;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_cif_f64(value: Option<&str>, field: &str) -> Result<f64> {
    let value = value.ok_or_else(|| invalid_cif(field, "missing numeric value"))?;
    let trimmed = numeric_without_uncertainty(value);
    trimmed
        .parse::<f64>()
        .map_err(|_| invalid_cif(field, format!("invalid numeric value {value:?}")))
}

fn parse_cif_i32(value: Option<&str>, field: &str) -> Result<i32> {
    let value = value.ok_or_else(|| invalid_cif(field, "missing integer value"))?;
    numeric_without_uncertainty(value)
        .parse::<i32>()
        .map_err(|_| invalid_cif(field, format!("invalid integer value {value:?}")))
}

fn numeric_without_uncertainty(value: &str) -> &str {
    let value = value.trim();
    value.find('(').map_or(value, |index| &value[..index])
}

fn normalize_cif_string(value: &str) -> String {
    value.trim_matches(['\'', '"']).trim().to_string()
}

fn strip_element_label(label: &str) -> String {
    label
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect::<String>()
}

fn required_f64(value: Option<f64>, field: &str) -> Result<f64> {
    value.ok_or_else(|| invalid_cif(field, "missing required cell field"))
}

fn is_cif_text_field_line(line: &str) -> bool {
    line.trim_start().starts_with(';')
}

fn read_cif_text_field(lines: &[&str], index: usize) -> Result<(String, usize)> {
    let Some(first_line) = lines.get(index) else {
        return Err(invalid_cif("text", "missing semicolon text field"));
    };
    let Some(semicolon_index) = first_line.find(';') else {
        return Err(invalid_cif("text", "missing semicolon text delimiter"));
    };

    let mut parts = Vec::new();
    let opening_remainder = &first_line[(semicolon_index + 1)..];
    if !opening_remainder.is_empty() {
        parts.push(opening_remainder);
    }

    for (offset, line) in lines.iter().enumerate().skip(index + 1) {
        if is_cif_text_field_line(line) {
            return Ok((parts.join("\n"), offset + 1));
        }
        parts.push(line);
    }

    Err(invalid_cif("text", "unterminated semicolon text field"))
}

fn invalid_cif(field: &str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "cif".into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CifEquivalence, apply_cif_symmetry_operation, expand_cif_cluster,
        expand_cif_cluster_with_equivalence, expand_cif_structure,
        expand_cif_structure_with_equivalence, parse_cif, tokenize_cif_line,
    };

    #[test]
    fn tokenizes_quoted_cif_values() {
        assert_eq!(
            tokenize_cif_line("_symmetry_space_group_name_H-M   'P 63/m m c' # comment"),
            ["_symmetry_space_group_name_H-M", "P 63/m m c"]
        );
    }

    #[test]
    fn parses_cell_uncertainties_and_atom_sites() -> crate::Result<()> {
        let cif = r#"
data_demo
_cell_length_a 2.9400(0)
_cell_length_b 2.9400(0)
_cell_length_c 12.1100(0)
_cell_angle_alpha 90.0000(0)
_cell_angle_beta 90.0000(0)
_cell_angle_gamma 120.0000(0)
_symmetry_space_group_name_H-M 'P 63/m m c'
_symmetry_Int_Tables_number 194
loop_
_symmetry_equiv_pos_as_xyz
'+x,+y,+z'
'-x,-y,1/2+z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C C1 0.0 0.0 0.0
Cr Cr1 0.6667 0.3333 0.0833
"#;
        let parsed = parse_cif(cif)?;
        assert_eq!(parsed.data_block.as_deref(), Some("demo"));
        assert_eq!(parsed.cell.a, 2.94);
        assert_eq!(parsed.cell.gamma, 120.0);
        assert_eq!(parsed.space_group_number, Some(194));
        assert_eq!(parsed.space_group_hm.as_deref(), Some("P 63/m m c"));
        assert_eq!(parsed.symmetry_operations.len(), 2);
        assert_eq!(parsed.atom_sites.len(), 2);
        assert_eq!(parsed.atom_sites[1].symbol, "Cr");
        Ok(())
    }

    #[test]
    fn parses_semicolon_cif_text_fields_in_scalars_and_loops() -> crate::Result<()> {
        let cif = r#"
data_text_fields
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M
;
P 1
;
loop_
_space_group_symop_operation_xyz
;
x,y,z
;
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1 0 0 0
"#;

        let parsed = parse_cif(cif)?;

        assert_eq!(parsed.space_group_hm.as_deref(), Some("P 1"));
        assert_eq!(parsed.symmetry_operations, ["x,y,z"]);
        assert_eq!(parsed.atom_sites.len(), 1);
        Ok(())
    }

    #[test]
    fn parses_loop_headers_and_values_on_shared_lines() -> crate::Result<()> {
        let cif = r#"
data_inline_loop
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_space_group_symop_operation_xyz 'x,y,z'
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z H1 0 0 0
O1 0.5 0.5 0.5
"#;

        let parsed = parse_cif(cif)?;

        assert_eq!(parsed.symmetry_operations, ["x,y,z"]);
        assert_eq!(parsed.atom_sites.len(), 2);
        assert_eq!(parsed.atom_sites[0].label.as_deref(), Some("H1"));
        assert_eq!(parsed.atom_sites[1].symbol, "O");
        Ok(())
    }

    #[test]
    fn parses_cif_data_past_ciftbx_page_boundaries() -> crate::Result<()> {
        let mut cif = String::from(
            r#"
data_page_boundary
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
_publ_section_comment
;
"#,
        );
        cif.push_str(&"x".repeat(9000));
        cif.push_str(
            r#"
;
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
"#,
        );
        for index in 0..180 {
            let x = f64::from(index) / 360.0;
            cif.push_str(&format!("H{index} {x:.6} 0 0\n"));
        }

        let parsed = parse_cif(&cif)?;

        assert!(cif.len() > 8192);
        assert_eq!(parsed.atom_sites.len(), 180);
        assert_eq!(
            parsed
                .atom_sites
                .last()
                .and_then(|site| site.label.as_deref()),
            Some("H179")
        );
        Ok(())
    }

    #[test]
    fn applies_symmetry_operations_to_fractional_coordinates() -> crate::Result<()> {
        let position = [2.0 / 3.0, 1.0 / 3.0, 1.0 / 12.0];
        let transformed = apply_cif_symmetry_operation("-x+y,1/2+z,0.3333-y", position)?;

        assert!((transformed[0] - (2.0 / 3.0)).abs() < 1.0e-12);
        assert!((transformed[1] - (7.0 / 12.0)).abs() < 1.0e-12);
        assert!(transformed[2].abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn expands_hexagonal_cif_like_feff_importcif() -> crate::Result<()> {
        let cif = parse_cif(
            r#"
data_Cr2GeC
_cell_length_a 2.9400(0)
_cell_length_b 2.9400(0)
_cell_length_c 12.1100(0)
_cell_angle_alpha 90.0000(0)
_cell_angle_beta 90.0000(0)
_cell_angle_gamma 120.0000(0)
_symmetry_space_group_name_H-M 'P 63/m m c'
_symmetry_Int_Tables_number 194
loop_
_symmetry_equiv_pos_as_xyz
'+x,+y,+z'
'-y,+x-y,+z'
'-x+y,-x,+z'
'-x,-y,1/2+z'
'+y,-x+y,1/2+z'
'+x-y,+x,1/2+z'
'-y,-x,+z'
'-x+y,+y,+z'
'+x,+x-y,+z'
'+y,+x,1/2+z'
'+x-y,-y,1/2+z'
'-x,-x+y,1/2+z'
'-x,-y,-z'
'+y,-x+y,-z'
'+x-y,+x,-z'
'+x,+y,1/2-z'
'-y,+x-y,1/2-z'
'-x+y,-x,1/2-z'
'+y,+x,-z'
'+x-y,-y,-z'
'-x,-x+y,-z'
'-y,-x,1/2-z'
'-x+y,+y,1/2-z'
'+x,+x-y,1/2-z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C C 0.0000 0.0000 0.0000
Cr Cr 0.6667 0.3333 0.0833
Ge Ge 0.6667 0.3333 0.7500
"#,
        )?;
        let structure = expand_cif_structure(&cif, 3)?;

        assert_eq!(structure.lattice_name, "H");
        assert_eq!(structure.space_group_hm, "P63/mmc");
        assert_eq!(structure.space_group, 194);
        assert_eq!(structure.absorber, 7);
        assert_eq!(structure.potentials, [1, 1, 2, 2, 2, 2, 3, 3]);
        assert_eq!(structure.labels, ["Ge", "C", "Cr", "Ge"]);
        assert_eq!(structure.positions.len(), 8);
        assert!((structure.positions[0][0] + 0.57735).abs() < 1.0e-5);
        assert!((structure.positions[0][2] - 1.02976).abs() < 1.0e-5);

        let cluster = expand_cif_cluster(&cif, 3, 6.0)?;
        assert!(cluster.atoms.len() > structure.positions.len());
        assert_eq!(cluster.atoms[0].potential, 0);
        assert_eq!(cluster.potentials[0].label, "Ge");
        Ok(())
    }

    #[test]
    fn cif_equivalence_two_collapses_potentials_by_atomic_number() -> crate::Result<()> {
        let cif = parse_cif(
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
        )?;

        let structure =
            expand_cif_structure_with_equivalence(&cif, 3, CifEquivalence::AtomicNumber)?;
        assert_eq!(structure.absorber_atomic_number, 8);
        assert_eq!(structure.absorber_label, "O");
        assert_eq!(structure.potentials, [1, 1, 2]);
        assert_eq!(structure.site_atomic_numbers, [1, 8]);
        assert_eq!(structure.site_labels, ["H", "O"]);
        assert_eq!(structure.site_multiplicities, [2, 1]);
        assert_eq!(structure.labels, ["O", "H", "O"]);

        let cluster =
            expand_cif_cluster_with_equivalence(&cif, 3, 4.0, CifEquivalence::AtomicNumber)?;
        assert_eq!(cluster.potentials.len(), 3);
        assert_eq!(cluster.potentials[0].atomic_number, 8);
        assert_eq!(cluster.potentials[1].atomic_number, 1);
        assert_eq!(cluster.potentials[1].multiplicity, 2);
        assert_eq!(cluster.potentials[2].atomic_number, 8);
        assert!(cluster.atoms.iter().any(|atom| atom.potential == 1));
        assert!(!cluster.atoms.iter().any(|atom| atom.potential == 3));
        Ok(())
    }

    #[test]
    fn cif_equivalence_four_uses_feff_potential_limit() -> crate::Result<()> {
        let small = parse_cif(
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
        )?;
        let small_structure =
            expand_cif_structure_with_equivalence(&small, 3, CifEquivalence::AutomaticLimit)?;
        assert_eq!(small_structure.potentials, [1, 2, 3]);

        let mut large_text = String::from(
            r#"
data_many_sites
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
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
"#,
        );
        for index in 0..32 {
            let symbol = if index % 2 == 0 { "H" } else { "O" };
            let x = index as f64 / 64.0;
            large_text.push_str(&format!("{symbol}{index} {x:.6} 0.0 0.0\n"));
        }
        let large = parse_cif(&large_text)?;
        let large_structure =
            expand_cif_structure_with_equivalence(&large, 2, CifEquivalence::AutomaticLimit)?;

        assert_eq!(large_structure.site_atomic_numbers.as_slice(), &[1, 8]);
        assert_eq!(large_structure.site_multiplicities.as_slice(), &[16, 16]);
        assert!(
            large_structure
                .potentials
                .iter()
                .all(|ipot| matches!(*ipot, 1 | 2))
        );
        Ok(())
    }
}
