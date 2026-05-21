use super::*;

pub(super) fn parse_reciprocal_space(input: &FeffInput) -> bool {
    input.cards().fold(false, |reciprocal, line| {
        let LineKind::Card { keyword, .. } = &line.kind else {
            return reciprocal;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("REAL") => false,
            Some("RECIPROCAL") => true,
            _ => reciprocal,
        }
    })
}

pub(super) fn parse_cif_equivalence(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "EQUIVALENCE") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    let Some(selector) = args.first() else {
        return Err(parse_error(line, "EQUIVALENCE requires a selector"));
    };
    let selector = parse_i32(line, selector)?;
    match selector {
        1 | 2 | 4 => Ok(selector),
        3 => Err(parse_error(
            line,
            "EQUIVALENCE 3 is not implemented by FEFF10",
        )),
        _ => Err(parse_error(line, "EQUIVALENCE must be 1, 2, 3, or 4")),
    }
}

fn cif_equivalence_mode(selector: i32) -> CifEquivalence {
    match selector {
        2 => CifEquivalence::AtomicNumber,
        4 => CifEquivalence::AutomaticLimit,
        _ => CifEquivalence::Crystallographic,
    }
}

pub(super) fn parse_coordinate_mode(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "COORDINATES") else {
        return Ok(3);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "COORDINATES requires a selector"));
    };
    let mode = parse_i32(line, value)?;
    if (1..=6).contains(&mode) {
        Ok(mode)
    } else {
        Err(parse_error(line, "COORDINATES must be between 1 and 6"))
    }
}

pub(super) fn parse_reciprocal_input(
    input: &FeffInput,
    nohole: i32,
    atoms: &[Atom],
    reciprocal: bool,
    cif_equivalence: i32,
) -> Result<Option<ReciprocalInput>> {
    if !reciprocal {
        return Ok(None);
    };
    let Some(reciprocal_line) = card_by_feff_name(input, "RECIPROCAL") else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "RECIPROCAL mode requires a RECIPROCAL card".to_string(),
        });
    };
    let k_mesh = parse_k_mesh(input)?;
    let absorber = parse_required_i32_card(input, "TARGET")?;
    let stretch = parse_strfac(input)?;

    let Some(lattice) = parse_lattice_block(input)? else {
        if let Some(cif_line) = card_by_feff_name(input, "CIF") {
            let cif_path = parse_cif_path(input, cif_line)?;
            let cif = read_cif(&cif_path)?;
            if absorber <= 0 {
                return Err(parse_error(
                    cif_line,
                    "TARGET must be positive for CIF input",
                ));
            }
            let target = usize::try_from(absorber)
                .map_err(|_| parse_error(cif_line, "TARGET is out of range for CIF input"))?;
            let structure = expand_cif_structure_with_equivalence(
                &cif,
                target,
                cif_equivalence_mode(cif_equivalence),
            )?;
            return Ok(Some(ReciprocalInput {
                ispace: 0,
                cell: Some(ReciprocalCell {
                    lattice_vectors: structure.lattice_vectors,
                    volume_scale: -1.0,
                    imaginary_energy: 0.0,
                    core_hole_strength: 1.0,
                    lattice_name: structure.lattice_name,
                    space_group_hm: structure.space_group_hm,
                    space_group: structure.space_group,
                    atom_count: structure.positions.len(),
                    absorber: i32::try_from(structure.absorber).map_err(|_| {
                        parse_error(cif_line, "expanded CIF absorber index is out of range")
                    })?,
                    core_hole: i32::from(nohole != 0),
                    k_mesh,
                    positions: structure.positions,
                    potentials: structure.potentials,
                    labels: structure.labels,
                    stretch,
                }),
            }));
        }
        return Err(parse_error(
            reciprocal_line,
            "RECIPROCAL requires LATTICE or CIF",
        ));
    };
    if atoms.is_empty() {
        return Err(parse_error(
            reciprocal_line,
            "RECIPROCAL with LATTICE requires ATOMS rows",
        ));
    }

    let space_group = parse_sgroup(input)?;
    let coordinate_mode = parse_coordinate_mode(input)?;
    let atoms = convert_lattice_atoms(input, &lattice, atoms, coordinate_mode)?;
    let positions = atoms.iter().map(|atom| [atom.x, atom.y, atom.z]).collect();
    let potentials = atoms.iter().map(|atom| atom.ipot).collect();

    Ok(Some(ReciprocalInput {
        ispace: 0,
        cell: Some(ReciprocalCell {
            lattice_vectors: lattice.vectors,
            volume_scale: -1.0,
            imaginary_energy: 0.0,
            core_hole_strength: 1.0,
            lattice_name: lattice.name,
            space_group_hm: "\0".repeat(8),
            space_group,
            atom_count: atoms.len(),
            absorber,
            core_hole: i32::from(nohole != 0),
            k_mesh,
            positions,
            potentials,
            labels: Vec::new(),
            stretch,
        }),
    }))
}

fn parse_cif_path(input: &FeffInput, line: &FeffLine) -> Result<PathBuf> {
    let args = card_args(line)?;
    let Some(path) = args.first() else {
        return Err(parse_error(line, "CIF requires a file path"));
    };
    let path = strip_card_delimiters(path);
    let path = PathBuf::from(path);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(input
            .source
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path))
    }
}

pub(super) fn strip_card_delimiters(value: &str) -> &str {
    let pairs = [
        ('"', '"'),
        ('\'', '\''),
        ('{', '}'),
        ('(', ')'),
        ('<', '>'),
        ('[', ']'),
    ];
    pairs
        .iter()
        .find_map(|(open, close)| {
            (value.starts_with(*open) && value.ends_with(*close) && value.len() >= 2)
                .then_some(&value[1..value.len() - 1])
        })
        .unwrap_or(value)
}

pub(super) fn parse_cif_cluster(
    input: &FeffInput,
    radius: f64,
    needed: bool,
    cif_equivalence: i32,
) -> Result<Option<CifCluster>> {
    if !needed {
        return Ok(None);
    }
    let Some(cif_line) = card_by_feff_name(input, "CIF") else {
        return Ok(None);
    };
    let cif_path = parse_cif_path(input, cif_line)?;
    let cif = read_cif(&cif_path)?;
    let target = parse_cif_target(input, cif_line)?;
    expand_cif_cluster_with_equivalence(&cif, target, radius, cif_equivalence_mode(cif_equivalence))
        .map(Some)
}

pub(super) fn cif_cluster_radius(scf: Option<&Scf>, fms: Option<&Fms>, rpath: Option<f64>) -> f64 {
    [scf.map(|scf| scf.radius), fms.map(|fms| fms.radius), rpath]
        .into_iter()
        .flatten()
        .fold(0.0, f64::max)
}

pub(super) fn cif_cluster_potentials(cluster: &CifCluster) -> Result<Vec<Potential>> {
    cluster
        .potentials
        .iter()
        .map(|potential| {
            let xnatph = if potential.absorber {
                Some(0.01)
            } else {
                Some(potential.multiplicity as f64)
            };
            Ok(Potential {
                ipot: potential.ipot,
                z: Some(potential.atomic_number),
                z_token: potential.atomic_number.to_string(),
                tag: Some(potential.label.clone()),
                lmax1: None,
                lmax2: None,
                xnatph,
                spinph: None,
            })
        })
        .collect()
}

pub(super) fn cif_cluster_atoms(cluster: &CifCluster) -> Vec<Atom> {
    cluster
        .atoms
        .iter()
        .map(|atom| Atom {
            x: atom.x,
            y: atom.y,
            z: atom.z,
            ipot: atom.potential,
            tag: None,
            distance: None,
            index: None,
        })
        .collect()
}

fn parse_cif_target(input: &FeffInput, cif_line: &FeffLine) -> Result<usize> {
    let Some(target_line) = card_by_feff_name(input, "TARGET") else {
        return Ok(1);
    };
    let args = card_args(target_line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(target_line, "TARGET requires a value"));
    };
    let target = parse_i32(target_line, value)?;
    if target <= 0 {
        return Err(parse_error(
            cif_line,
            "TARGET must be positive for CIF input",
        ));
    }
    usize::try_from(target)
        .map_err(|_| parse_error(cif_line, "TARGET is out of range for CIF input"))
}

struct LatticeBlock {
    name: String,
    vectors: [[f64; 3]; 3],
}

#[derive(Debug, Clone, Copy)]
struct PeriodicAtom {
    x: f64,
    y: f64,
    z: f64,
    ipot: i32,
    distance: f64,
}

pub(super) fn parse_lattice_cluster_atoms(
    input: &FeffInput,
    atoms: &[Atom],
    radius: f64,
    reciprocal: bool,
) -> Result<Option<Vec<Atom>>> {
    if !reciprocal || card_by_feff_name(input, "CIF").is_some() {
        return Ok(None);
    }
    let Some(lattice) = parse_lattice_block(input)? else {
        return Ok(None);
    };
    if atoms.is_empty() {
        return Ok(None);
    }
    let target = parse_required_i32_card(input, "TARGET")?;
    if target <= 0 {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "TARGET must be positive for LATTICE input".to_string(),
        });
    }
    let target = usize::try_from(target - 1).map_err(|_| IoError::Parse {
        path: input.source.clone(),
        line: 0,
        message: "TARGET is out of range for LATTICE input".to_string(),
    })?;
    if target >= atoms.len() {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!(
                "TARGET {} is outside the ATOMS row range 1..={}",
                target + 1,
                atoms.len()
            ),
        });
    }
    let coordinate_mode = parse_coordinate_mode(input)?;
    let atoms = convert_lattice_atoms(input, &lattice, atoms, coordinate_mode)?;
    Ok(Some(expand_lattice_cluster(
        &lattice, &atoms, target, radius,
    )))
}

fn convert_lattice_atoms(
    input: &FeffInput,
    lattice: &LatticeBlock,
    atoms: &[Atom],
    coordinate_mode: i32,
) -> Result<Vec<Atom>> {
    let lengths = lattice_vector_lengths(input, lattice)?;
    atoms
        .iter()
        .map(|atom| convert_lattice_atom(input, lattice, lengths, atom, coordinate_mode))
        .collect()
}

fn convert_lattice_atom(
    input: &FeffInput,
    lattice: &LatticeBlock,
    lengths: [f64; 3],
    atom: &Atom,
    coordinate_mode: i32,
) -> Result<Atom> {
    let [a1_len, a2_len, a3_len] = lengths;
    let position = match coordinate_mode {
        1 => [atom.x / a1_len, atom.y / a1_len, atom.z / a1_len],
        2 => [atom.x, atom.y * a2_len / a1_len, atom.z * a3_len / a1_len],
        3 => [atom.x, atom.y, atom.z],
        4 => scale_vector(
            fractional_to_cartesian([atom.x, atom.y, atom.z], lattice.vectors),
            1.0 / a1_len,
        ),
        5 => {
            let fractional = [atom.x, atom.y * a1_len / a2_len, atom.z * a1_len / a3_len];
            scale_vector(
                fractional_to_cartesian(fractional, lattice.vectors),
                1.0 / a1_len,
            )
        }
        6 => {
            let fractional = [atom.x / a1_len, atom.y / a2_len, atom.z / a3_len];
            scale_vector(
                fractional_to_cartesian(fractional, lattice.vectors),
                1.0 / a1_len,
            )
        }
        _ => {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "COORDINATES must be between 1 and 6".to_string(),
            });
        }
    };
    Ok(Atom {
        x: position[0],
        y: position[1],
        z: position[2],
        ipot: atom.ipot,
        tag: atom.tag.clone(),
        distance: atom.distance,
        index: atom.index,
    })
}

fn lattice_vector_lengths(input: &FeffInput, lattice: &LatticeBlock) -> Result<[f64; 3]> {
    let lengths = lattice.vectors.map(lattice_vector_length);
    if lengths
        .iter()
        .all(|length| length.is_finite() && *length > 0.0)
    {
        Ok(lengths)
    } else {
        Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "LATTICE vectors must have positive finite lengths".to_string(),
        })
    }
}

fn expand_lattice_cluster(
    lattice: &LatticeBlock,
    atoms: &[Atom],
    target: usize,
    radius: f64,
) -> Vec<Atom> {
    let [a1, a2, a3] = lattice.vectors;
    let ratomslist = 8.0_f64.max(1.33 * radius.max(0.0));
    let i1 = lattice_repeat_count(ratomslist, a1);
    let i2 = lattice_repeat_count(ratomslist, a2);
    let i3 = lattice_repeat_count(ratomslist, a3);
    let shifts = lattice_centering_shifts(&lattice.name);
    let lattice_scale = lattice_vector_length(a1);
    let absorber = lattice_atom_position(&atoms[target], lattice_scale);

    let mut expanded = Vec::new();
    let mut absorber_index = 0_usize;
    for j1 in -i1..=i1 {
        for j2 in -i2..=i2 {
            for j3 in -i3..=i3 {
                let translation = lattice_translation(j1, j2, j3, a1, a2, a3);
                for (index, atom) in atoms.iter().enumerate() {
                    let position =
                        add_vectors(lattice_atom_position(atom, lattice_scale), translation);
                    let mut ipot = atom.ipot;
                    if j1 == 0 && j2 == 0 && j3 == 0 && index == target {
                        ipot = 0;
                        absorber_index = expanded.len();
                    }
                    expanded.push(periodic_atom(position, ipot, absorber));

                    for shift in &shifts {
                        let shifted =
                            add_vectors(position, fractional_to_cartesian(*shift, [a1, a2, a3]));
                        expanded.push(periodic_atom(shifted, atom.ipot, absorber));
                    }
                }
            }
        }
    }

    feff_sort_periodic_atoms(&mut expanded, absorber_index);
    let cutoff = (lattice_vector_length(a1) * f64::from(i1))
        .min(lattice_vector_length(a2) * f64::from(i1))
        .min(lattice_vector_length(a3) * f64::from(i1));
    let keep = expanded
        .iter()
        .position(|atom| atom.distance > cutoff)
        .unwrap_or(expanded.len());
    expanded.truncate(keep);

    expanded
        .into_iter()
        .map(|atom| Atom {
            x: atom.x,
            y: atom.y,
            z: atom.z,
            ipot: atom.ipot,
            tag: None,
            distance: None,
            index: None,
        })
        .collect()
}

fn periodic_atom(position: [f64; 3], ipot: i32, absorber: [f64; 3]) -> PeriodicAtom {
    PeriodicAtom {
        x: position[0],
        y: position[1],
        z: position[2],
        ipot,
        distance: lattice_distance(position, absorber),
    }
}

fn lattice_atom_position(atom: &Atom, scale: f64) -> [f64; 3] {
    [atom.x * scale, atom.y * scale, atom.z * scale]
}

fn feff_sort_periodic_atoms(atoms: &mut [PeriodicAtom], mut absorber_index: usize) {
    for i in 0..atoms.len() {
        let mut min_index = i;
        let mut min_distance = atoms[i].distance;
        for (j, atom) in atoms.iter().enumerate().skip(i) {
            if atom.distance < min_distance {
                min_index = j;
                min_distance = atom.distance;
            }
        }
        atoms.swap(i, min_index);
        if i == absorber_index {
            absorber_index = min_index;
        }
        if min_index == absorber_index {
            absorber_index = i;
        }
    }
}

fn lattice_repeat_count(radius: f64, vector: [f64; 3]) -> i32 {
    (radius / lattice_vector_length(vector)).trunc() as i32 + 1
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

fn lattice_distance(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lattice_vector_length([lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]])
}

fn lattice_vector_length(vector: [f64; 3]) -> f64 {
    vector[0]
        .mul_add(
            vector[0],
            vector[1].mul_add(vector[1], vector[2] * vector[2]),
        )
        .sqrt()
}

fn add_vectors(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn scale_vector(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn parse_lattice_block(input: &FeffInput) -> Result<Option<LatticeBlock>> {
    let Some(line) = card_by_feff_name(input, "LATTICE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(name) = args.first() else {
        return Err(parse_error(line, "LATTICE requires a lattice type"));
    };
    let scale = parse_optional_f64(line, args.get(1))?.unwrap_or(1.0);
    let rows = input.section_rows("LATTICE").collect::<Vec<_>>();
    if rows.len() < 3 {
        return Err(parse_error(line, "LATTICE requires three vector rows"));
    }

    let mut vectors = [[0.0; 3]; 3];
    for (idx, row) in rows.iter().take(3).enumerate() {
        let fields = section_fields(row)?;
        if fields.len() < 3 {
            return Err(parse_error(row, "LATTICE vector rows require x y z"));
        }
        vectors[idx] = [
            parse_f64(row, &fields[0])? * scale,
            parse_f64(row, &fields[1])? * scale,
            parse_f64(row, &fields[2])? * scale,
        ];
    }

    Ok(Some(LatticeBlock {
        name: name.clone(),
        vectors,
    }))
}

fn parse_k_mesh(input: &FeffInput) -> Result<ReciprocalKMesh> {
    let Some(line) = card_by_feff_name(input, "KMESH") else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "RECIPROCAL requires KMESH".to_string(),
        });
    };
    let args = card_args(line)?;
    let Some(x) = args.first() else {
        return Err(parse_error(line, "KMESH requires at least one value"));
    };
    let x = parse_i32(line, x)?;
    let y = parse_optional_i32(line, args.get(1))?.unwrap_or(0);
    let z = parse_optional_i32(line, args.get(2))?.unwrap_or(0);
    let product = x * y * z;
    Ok(ReciprocalKMesh {
        total: if product == 0 { x } else { product },
        x,
        y,
        z,
        kind: parse_optional_i32(line, args.get(3))?.unwrap_or(1),
        use_symmetry: parse_optional_i32(line, args.get(4))?.unwrap_or(0) != 0,
    })
}

fn parse_required_i32_card(input: &FeffInput, keyword: &str) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!("RECIPROCAL requires {keyword}"),
        });
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, format!("{keyword} requires a value")));
    };
    parse_i32(line, value)
}

pub(super) fn parse_strfac(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = card_by_feff_name(input, "STRFAC") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "STRFAC requires three values"));
    }
    Ok([
        parse_f64(line, &args[0])?,
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
    ])
}

fn parse_sgroup(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "SGROUP") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(1);
    };
    parse_i32(line, value)
}
