use crate::Result;
use refeff_core::atomic_symbol;

use super::common::{
    CIF_FRACTION_TOLERANCE, CIF_POSITION_TOLERANCE, invalid_cif, strip_element_label,
};
use super::types::{
    CifAtomSite, CifCell, CifCluster, CifClusterAtom, CifDocument, CifEquivalence,
    CifExpandedStructure, CifPotential,
};

const FEFF_CIF_EQUIVALENCE_LIMIT: usize = 31;

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

pub(super) fn apply_cif_symmetry_operation(
    operation: &str,
    position: [f64; 3],
) -> Result<[f64; 3]> {
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
