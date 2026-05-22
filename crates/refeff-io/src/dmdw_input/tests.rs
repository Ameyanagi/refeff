use crate::{FeffDocument, FeffInput, rdinp};

use super::{DmdwInput, DmdwPath, DmdwPdosOptions, DmdwSelfEnergyOptions, dmdw_input_string};
use refeff_core::{DMDW_ANGSTROM_TO_BOHR, DmdwPathDescriptor, dmdw_expand_path_descriptors};

#[test]
fn parses_disabled_dmdw_input() -> crate::Result<()> {
    let input = FeffInput::parse_str("feff.inp", "END\n")?;
    let document = FeffDocument::from_input(&input)?;
    let dmdw = DmdwInput::parse_str("dmdw.inp", &rdinp::dmdw_inp_string(&document)?)?;
    assert_eq!(dmdw, DmdwInput::Disabled);
    Ok(())
}

#[test]
fn renders_disabled_dmdw_input() -> crate::Result<()> {
    let input = FeffInput::parse_str("feff.inp", "END\n")?;
    let document = FeffDocument::from_input(&input)?;
    let expected = rdinp::dmdw_inp_string(&document)?;
    let dmdw = DmdwInput::parse_str("dmdw.inp", &expected)?;

    assert_eq!(dmdw_input_string(&dmdw)?, expected);
    Ok(())
}

#[test]
fn parses_generated_dmdw_routes() -> crate::Result<()> {
    let temp = tempfile::tempdir().map_err(|source| crate::IoError::io("tempdir", source))?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())
        .map_err(|source| crate::IoError::io("feff.dym", source))?;
    let input = FeffInput::parse_str(
        &input_path,
        r#"
DEBYE 450 315 5 feff.dym 6 7 13
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let dmdw = DmdwInput::parse_str("dmdw.inp", &rdinp::dmdw_inp_string(&document)?)?;
    let DmdwInput::Enabled(calculation) = dmdw else {
        return Err(crate::IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "expected enabled DMDW calculation".to_string(),
        });
    };

    assert_eq!(calculation.run, 1);
    assert_eq!(calculation.order, 6);
    assert_eq!(calculation.temperature_flag, 1);
    assert_eq!(calculation.temperature, 450.0);
    assert_eq!(calculation.temperature_max, None);
    assert_eq!(calculation.calculation_type, 7);
    assert_eq!(calculation.self_energy_options, None);
    assert_eq!(calculation.pdos_options, None);
    assert_eq!(calculation.dym_file, "feff.dym");
    assert_eq!(calculation.path_count, 3);
    assert_eq!(calculation.paths.len(), 3);
    assert_eq!(
        calculation.paths[0],
        DmdwPath {
            leg_count: 2,
            absorber_selector: 0,
            potentials: vec![0],
            max_distance: calculation.paths[0].max_distance,
        }
    );
    assert_eq!(calculation.paths[1].leg_count, 3);
    assert_eq!(calculation.paths[2].leg_count, 4);
    assert!(calculation.paths[2].max_distance > calculation.paths[1].max_distance);
    Ok(())
}

#[test]
fn parses_and_renders_multi_temperature_grid() -> crate::Result<()> {
    let text = concat!(
        "   1\n",
        "   2\n",
        "   3    100.000    500.000\n",
        "   0\n",
        "feff.dym\n",
        "   1\n",
        "   2   0   1          10.00\n",
    );
    let dmdw = DmdwInput::parse_str("dmdw.inp", text)?;
    let DmdwInput::Enabled(calculation) = &dmdw else {
        return Err(crate::IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "expected enabled DMDW calculation".to_string(),
        });
    };

    assert_eq!(calculation.temperature_flag, 3);
    assert_eq!(calculation.temperature, 100.0);
    assert_eq!(calculation.temperature_max, Some(500.0));
    assert_eq!(dmdw_input_string(&dmdw)?, text);
    Ok(())
}

#[test]
fn parses_single_atom_dmdw_descriptor() -> crate::Result<()> {
    let dmdw = DmdwInput::parse_str(
        "dmdw.inp",
        concat!(
            "   1\n",
            "   2\n",
            "   1     77.000\n",
            "   3\n",
            "feff.dym\n",
            "   1\n",
            "   1   2   10.00\n",
        ),
    )?;
    let DmdwInput::Enabled(calculation) = dmdw else {
        return Err(crate::IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "expected enabled DMDW calculation".to_string(),
        });
    };

    assert_eq!(
        calculation.paths[0],
        DmdwPath {
            leg_count: 1,
            absorber_selector: 2,
            potentials: Vec::new(),
            max_distance: 10.0,
        }
    );
    Ok(())
}

#[test]
fn parses_and_renders_projected_dos_options() -> crate::Result<()> {
    let text = concat!(
        "   1\n",
        "   4\n",
        "   1    300.000\n",
        "   5  10   T   T      0.750      0.050\n",
        "feff.dym\n",
        "   1\n",
        "   1   0   10.00\n",
    );
    let dmdw = DmdwInput::parse_str("dmdw.inp", text)?;
    let DmdwInput::Enabled(calculation) = &dmdw else {
        return Err(crate::IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "expected enabled DMDW calculation".to_string(),
        });
    };

    assert_eq!(
        calculation.pdos_options,
        Some(DmdwPdosOptions {
            format: 10,
            write_partial: true,
            drop_left_edges: true,
            gaussian_broadening_thz: 0.750,
            gaussian_resolution_thz: 0.050,
        })
    );

    let rendered = dmdw_input_string(&dmdw)?;
    let reparsed = DmdwInput::parse_str("dmdw.inp", &rendered)?;
    assert_eq!(reparsed, dmdw);
    Ok(())
}

#[test]
fn parses_and_renders_self_energy_options() -> crate::Result<()> {
    let text = concat!(
        "   1\n",
        "   8\n",
        "   1    300.000\n",
        "   2\n",
        "   3\n",
        "   1      0.125\n",
        "feff-se.dym\n",
        "phonon.pds\n",
        "coupling.a2f\n",
    );
    let dmdw = DmdwInput::parse_str("dmdw.inp", text)?;
    let DmdwInput::Enabled(calculation) = &dmdw else {
        return Err(crate::IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "expected enabled DMDW calculation".to_string(),
        });
    };

    assert_eq!(calculation.calculation_type, 2);
    assert_eq!(calculation.path_count, 0);
    assert!(calculation.paths.is_empty());
    assert_eq!(
        calculation.self_energy_options,
        Some(DmdwSelfEnergyOptions {
            displacement_option: 3,
            energy_option: 1,
            electron_energy: 0.125,
            pds_file: "phonon.pds".to_string(),
            a2f_file: "coupling.a2f".to_string(),
        })
    );
    assert_eq!(calculation.pdos_options, None);

    let rendered = dmdw_input_string(&dmdw)?;
    let reparsed = DmdwInput::parse_str("dmdw.inp", &rendered)?;
    assert_eq!(reparsed, dmdw);
    Ok(())
}

#[test]
fn renders_generated_dmdw_routes() -> crate::Result<()> {
    let temp = tempfile::tempdir().map_err(|source| crate::IoError::io("tempdir", source))?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())
        .map_err(|source| crate::IoError::io("feff.dym", source))?;
    let input = FeffInput::parse_str(
        &input_path,
        r#"
DEBYE 450 315 5 feff.dym 6 7 13
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;
    let expected = rdinp::dmdw_inp_string(&document)?;
    let dmdw = DmdwInput::parse_str("dmdw.inp", &expected)?;
    let rendered = dmdw_input_string(&dmdw)?;
    let reparsed = DmdwInput::parse_str("dmdw.inp", &rendered)?;

    assert_eq!(rendered, expected);
    assert_eq!(reparsed, dmdw);
    Ok(())
}

#[test]
fn expands_feff_h2o_reference_descriptors_when_available() -> crate::Result<()> {
    let Some(reference_dir) = reference_dmdw_test_dir()? else {
        eprintln!("skipping DMDW H2O descriptor expansion test; feff10 fixture not found");
        return Ok(());
    };

    let input_path = reference_dir.join("H2O.g03.dmdw.inp");
    let dym_path = reference_dir.join("H2O.g03.dym");
    let output_path = reference_dir
        .join("Reference_Results")
        .join("H2O.g03.dmdw.out");
    let input_text = std::fs::read_to_string(&input_path)
        .map_err(|source| crate::IoError::io(input_path.clone(), source))?;
    let dym_text = std::fs::read_to_string(&dym_path)
        .map_err(|source| crate::IoError::io(dym_path.clone(), source))?;
    let output_text = std::fs::read_to_string(&output_path)
        .map_err(|source| crate::IoError::io(output_path.clone(), source))?;
    let DmdwInput::Enabled(calculation) = DmdwInput::parse_str(&input_path, &input_text)? else {
        return Err(crate::IoError::Parse {
            path: input_path,
            line: 0,
            message: "expected enabled DMDW reference input".to_string(),
        });
    };
    let dym = crate::parse_dym(&dym_text)?;
    let positions = dym.coordinates.cartesian_positions();
    let descriptors = calculation
        .paths
        .iter()
        .map(|path| DmdwPathDescriptor {
            selectors: std::iter::once(path.absorber_selector)
                .chain(path.potentials.iter().copied())
                .collect(),
            max_effective_length: path.max_distance * DMDW_ANGSTROM_TO_BOHR,
        })
        .collect::<Vec<_>>();
    let expanded =
        dmdw_expand_path_descriptors(positions.view(), &descriptors).map_err(|source| {
            crate::IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: source.to_string(),
            }
        })?;
    let output = crate::parse_dmdw_out(&output_text)?;
    let expected_paths = output
        .sections
        .iter()
        .filter_map(|section| match &section.subject {
            crate::DmdwOutSubject::PathIndices(indices) => Some(indices.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let actual_paths = expanded
        .iter()
        .map(|path| path.atoms.iter().map(|&atom| atom + 1).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    assert_eq!(actual_paths, expected_paths);
    Ok(())
}

#[test]
fn rejects_invalid_dmdw_rendering() {
    let input = DmdwInput::Enabled(super::DmdwCalculation {
        run: 1,
        order: 6,
        temperature_flag: 1,
        temperature: 450.0,
        temperature_max: None,
        calculation_type: 7,
        self_energy_options: None,
        pdos_options: None,
        dym_file: "feff.dym".to_string(),
        path_count: 2,
        paths: vec![DmdwPath {
            leg_count: 2,
            absorber_selector: 0,
            potentials: vec![0],
            max_distance: 10.0,
        }],
    });
    assert!(dmdw_input_string(&input).is_err());
}

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

fn reference_dmdw_test_dir() -> crate::Result<Option<std::path::PathBuf>> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or_else(|| crate::IoError::Parse {
            path: manifest_dir.into(),
            line: 0,
            message: "failed to find workspace root".to_string(),
        })?;
    let path = workspace
        .join("feff10")
        .join("src")
        .join("DMDW")
        .join("Test");
    let required = [
        "H2O.g03.dmdw.inp",
        "H2O.g03.dym",
        "Reference_Results/H2O.g03.dmdw.out",
    ];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}
