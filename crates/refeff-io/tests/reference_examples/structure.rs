use super::*;

#[test]
fn roundtrips_generated_reference_structure_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping structural output roundtrip; reference-work/golden not found");
        return Ok(());
    }

    let mut compared = 0usize;

    let mut dimensions_outputs = Vec::new();
    collect_named_files(&golden_dir, ".dimensions.dat", &mut dimensions_outputs)?;
    dimensions_outputs.sort();
    for path in dimensions_outputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = DimensionsDat::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = dimensions_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                ".dimensions.dat mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut atoms_outputs = Vec::new();
    collect_named_files(&golden_dir, "atoms.dat", &mut atoms_outputs)?;
    atoms_outputs.sort();
    for path in atoms_outputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = AtomsDat::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = atoms_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "atoms.dat mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    let mut geom_outputs = Vec::new();
    collect_named_files(&golden_dir, "geom.dat", &mut geom_outputs)?;
    geom_outputs.sort();
    for path in geom_outputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed = GeomDat::parse_str(&path, &text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let rendered = geom_dat_string(&parsed)
            .with_context(|| format!("failed to render {}", path.display()))?;
        if rendered != text {
            let mismatch = first_mismatch(&text, &rendered);
            ensure!(
                false,
                "geom.dat mismatch for {}: {mismatch}",
                path.display()
            );
        }
        compared += 1;
    }

    ensure!(compared > 0, "no generated structural outputs found");
    Ok(())
}

#[test]
fn parses_generated_reference_cif_inputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated CIF parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut cif_inputs = Vec::new();
    collect_extension_files(&golden_dir, "cif", &mut cif_inputs)?;
    cif_inputs.sort();
    ensure!(!cif_inputs.is_empty(), "no generated CIF inputs found");

    for path in cif_inputs {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let parsed =
            parse_cif(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        ensure!(
            parsed.cell.a > 0.0 && parsed.cell.b > 0.0 && parsed.cell.c > 0.0,
            "{} has non-positive cell length",
            path.display()
        );
        ensure!(
            parsed.space_group_number.is_some() || parsed.space_group_hm.is_some(),
            "{} has no space-group metadata",
            path.display()
        );
        ensure!(
            !parsed.symmetry_operations.is_empty(),
            "{} has no symmetry operations",
            path.display()
        );
        ensure!(
            !parsed.atom_sites.is_empty(),
            "{} has no atom sites",
            path.display()
        );
    }

    Ok(())
}

#[test]
fn expands_generated_reference_cif_clusters_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated CIF cluster coverage; reference-work/golden not found");
        return Ok(());
    }

    struct Case {
        name: &'static str,
        cif_name: &'static str,
        target: usize,
        rmax: f64,
        potentials: &'static [(i32, i32, &'static str, bool)],
    }

    let cases = [
        Case {
            name: "CRPA",
            cif_name: "Ce-Cerium.cif",
            target: 1,
            rmax: 5.0,
            potentials: &[(0, 58, "Ce", true), (1, 58, "Ce", false)],
        },
        Case {
            name: "HUBBARD/CeO2",
            cif_name: "CeO2.cif",
            target: 2,
            rmax: 8.0,
            potentials: &[(0, 8, "O", true), (1, 58, "Ce", false), (2, 8, "O", false)],
        },
        Case {
            name: "KSPACE/Cr2GeC",
            cif_name: "Cr2GeC.cif",
            target: 3,
            rmax: 6.0,
            potentials: &[
                (0, 32, "Ge", true),
                (1, 6, "C", false),
                (2, 24, "Cr", false),
                (3, 32, "Ge", false),
            ],
        },
    ];

    let mut compared = 0_usize;
    for case in &cases {
        let case_dir = golden_dir.join(case.name);
        let cif_path = case_dir.join(case.cif_name);
        let atoms_path = case_dir.join("atoms.dat");
        if !cif_path.exists() || !atoms_path.exists() {
            continue;
        }

        let cif_text = std::fs::read_to_string(&cif_path)
            .with_context(|| format!("failed to read {}", cif_path.display()))?;
        let cif = parse_cif(&cif_text)
            .with_context(|| format!("failed to parse {}", cif_path.display()))?;
        let cluster = expand_cif_cluster(&cif, case.target, case.rmax)
            .with_context(|| format!("failed to expand {}", cif_path.display()))?;

        let atoms_text = std::fs::read_to_string(&atoms_path)
            .with_context(|| format!("failed to read {}", atoms_path.display()))?;
        let expected = AtomsDat::parse_str(atoms_path.clone(), &atoms_text)
            .with_context(|| format!("failed to parse {}", atoms_path.display()))?;

        ensure!(
            expected.natx == expected.atoms.len(),
            "{} atoms.dat natx does not match row count",
            case.name
        );
        ensure!(
            cluster.atoms.len() == expected.atoms.len(),
            "{} CIF cluster row count mismatch: expected {}, got {}",
            case.name,
            expected.atoms.len(),
            cluster.atoms.len()
        );
        ensure!(
            cluster.atoms.first().is_some_and(|atom| atom.potential == 0
                && rounded(atom.x) == 0
                && rounded(atom.y) == 0
                && rounded(atom.z) == 0),
            "{} CIF cluster does not start with the absorber at the origin",
            case.name
        );
        ensure!(
            cluster.atoms.windows(2).all(|pair| {
                cluster_atom_distance(pair[0]) <= cluster_atom_distance(pair[1]) + 1.0e-9
            }),
            "{} CIF cluster is not sorted by distance",
            case.name
        );

        let mut actual_atoms = cluster
            .atoms
            .iter()
            .copied()
            .map(rounded_cluster_atom_key)
            .collect::<Vec<_>>();
        let mut expected_atoms = expected
            .atoms
            .iter()
            .copied()
            .map(rounded_atoms_dat_key)
            .collect::<Vec<_>>();
        actual_atoms.sort_unstable();
        expected_atoms.sort_unstable();
        ensure!(
            actual_atoms == expected_atoms,
            "{} CIF cluster atom set differs from atoms.dat",
            case.name
        );

        let actual_potentials = cluster
            .potentials
            .iter()
            .map(|potential| {
                (
                    potential.ipot,
                    potential.atomic_number,
                    potential.label.as_str(),
                    potential.absorber,
                )
            })
            .collect::<Vec<_>>();
        ensure!(
            actual_potentials.as_slice() == case.potentials,
            "{} CIF potential metadata mismatch",
            case.name
        );

        compared += 1;
    }

    ensure!(
        compared == cases.len(),
        "expected {} CIF cluster cases, compared {compared}",
        cases.len()
    );
    Ok(())
}
