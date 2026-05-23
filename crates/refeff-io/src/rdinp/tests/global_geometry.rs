use super::*;

#[test]
fn writes_reciprocal_inp_from_lattice_atoms() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 1000 0
TARGET 1
LATTICE P 2.456
0.86603 -0.5000 0.00000
0.00000 1.00000 0.00000
0.00000 0.00000 2.72638
ATOMS
0.00000 0.00000 0.68160 1 C1
0.00000 0.00000 2.04479 1 C1
0.57735 0.00000 0.68160 2 C2
0.28868 0.50000 2.04479 2 C2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let reciprocal = outputs
        .get("reciprocal.inp")
        .ok_or_else(|| IoError::Parse {
            path: "test".into(),
            line: 0,
            message: "missing reciprocal.inp".to_string(),
        })?;

    assert!(reciprocal.starts_with("ispace\n   0\n"));
    assert!(reciprocal.contains("      2.12697     -1.22800      0.00000\n"));
    assert!(
        reciprocal
            .contains("        1000        1000           0           0           1           0\n")
    );
    assert!(reciprocal.contains("           1           1           2           2\n"));
    Ok(())
}

#[test]
fn writes_reciprocal_coordinates_one_like_feff() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE COORDINATES smoke
EDGE K
RECIPROCAL
KMESH 10 0
TARGET 1
FMS 2.0
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
COORDINATES 1
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;
    let geom = crate::GeomDat::parse_str("geom.dat", &geom_dat_string(&doc)?)?;
    let reciprocal_input = doc
        .reciprocal_input
        .as_ref()
        .ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "missing reciprocal input".to_string(),
        })?;
    let reciprocal = crate::reciprocal_input_string(reciprocal_input)?;

    assert_eq!(atoms.atoms[0].iph, 0);
    assert_eq!(atoms.atoms[1].distance, 1.0);
    assert_eq!(atoms.atoms[2].distance, 1.0);
    assert_eq!(geom.atoms[1].x, -1.0);
    assert_eq!(geom.atoms[2].x, 1.0);
    assert!(reciprocal.contains("      0.50000      0.00000      0.00000\n"));
    Ok(())
}

#[test]
fn logs_real_and_reciprocal_cards_in_input_order() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
REAL
EDGE K
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let expected_reciprocal = reciprocal_inp_string();

    assert!(!doc.reciprocal);
    assert!(doc.reciprocal_input.is_none());
    assert!(
        rdinp_stdout_string(&doc)?
            .contains("Working in reciprocal space.\nWorking in real space.\n")
    );
    assert_eq!(
        outputs.get("reciprocal.inp").map(String::as_str),
        Some(expected_reciprocal.as_str())
    );
    Ok(())
}

#[test]
fn writes_global_inp_defaults() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let global = global_inp_string(&doc)?;

    assert!(global.contains(" nabs, iphabs - CFAVERAGE data\n       1       0 100000.00000\n"));
    assert!(global.contains(" polarization tensor \n      0.33333"));
    Ok(())
}

#[test]
fn writes_cfaverage_into_global_pot_and_geom_outputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE CFAVERAGE smoke
EDGE K
CFAVERAGE 1 0 0
POTENTIALS
1 29 Cu
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
2.0 0.0 0.0 1 Cu2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let global = GlobalInput::parse_str("global.inp", &global_inp_string(&doc)?)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;
    let geom = crate::GeomDat::parse_str("geom.dat", &geom_dat_string(&doc)?)?;

    assert_eq!(global.cfaverage.nabs, 3);
    assert_eq!(global.cfaverage.iphabs, 1);
    assert_eq!(global.cfaverage.rclabs, 100000.0);
    assert_eq!(pot.control.nph, 1);
    assert_eq!(pot.potentials[0].z, 29);
    assert_eq!(pot.potentials[0].xnatph, 1.0);
    assert_eq!(pot.potentials[1].z, 29);
    assert_eq!(pot.potentials[1].xnatph, 2.0);
    assert_eq!(atoms.atoms[0].iph, 1);
    assert_eq!(geom.model_atoms, [1, 2]);
    assert_eq!(geom.atoms[0].iph, 0);
    assert_eq!(geom.atoms[1].iph, 1);
    Ok(())
}

#[test]
fn writes_global_inp_linear_polarization_tensor() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POLARIZATION 1 0 0
MULTIPOLE 2
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let text = global_inp_string(&doc)?;
    let global = GlobalInput::parse_str("global.inp", &text)?;

    assert_eq!(global.control.ipol, 1);
    assert_eq!(global.control.le2, 0);
    assert_eq!(
        global.polarization_tensor,
        [
            [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
            [0.0; 6],
            [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        ]
    );
    Ok(())
}

#[test]
fn writes_spectroscopy_aliases_into_global_and_xsph_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EDGE K
XANE 20.0 1.25 0.2
POLA 1.0 0.0 0.0
ELLI 0.25 0.0 1.0 0.0
MULT 2 1
POTENTIALS
0 29 Cu
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let global = GlobalInput::parse_str("global.inp", &global_inp_string(&doc)?)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;

    assert_eq!(global.control.ipol, 1);
    assert_eq!(global.control.le2, 2);
    assert_eq!(global.control.l2lp, 1);
    assert_eq!(global.control.elpty, 0.25);
    assert_eq!(global.xivec, [0.0, 1.0, 0.0]);
    assert_eq!(global.norms.xivnorm, 1.0);
    assert_eq!(xsph.control.ispec, 1);
    assert_eq!(xsph.grid.xkstep, 1.25);
    assert_eq!(xsph.grid.xkmax, 20.0);
    assert_eq!(xsph.grid.vixan, 0.2);
    Ok(())
}

#[test]
fn rejects_zero_length_global_polarization_vector() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POLARIZATION 0 0 0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    assert!(global_inp_string(&doc).is_err());
    Ok(())
}

#[test]
fn writes_dimensions_dat_from_cluster_radius() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RPATH 2.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
3.0 0.0 0.0 1 Cu2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        dimensions_dat_string(&doc)?,
        "           2           3           1           1\n"
    );
    Ok(())
}

#[test]
fn writes_geom_dat_with_sorted_relative_cluster() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
2.0 0.0 0.0 1 Cu2
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        geom_dat_string(&doc)?,
        concat!(
            "nat, nph =     3    1\n",
            "    1    2\n",
            " iat     x       y        z       iph  \n",
            " -----------------------------------------------------------------------\n",
            "   1      0.00000      0.00000      0.00000   0   1\n",
            "   2      1.00000      0.00000      0.00000   1   1\n",
            "   3      2.00000      0.00000      0.00000   1   1\n",
        )
    );
    Ok(())
}

#[test]
fn nogeom_suppresses_geom_dat_output_like_feff() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
NOGEOM
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;

    assert!(doc.no_geom);
    assert!(outputs.contains_key("atoms.dat"));
    assert!(!outputs.contains_key("geom.dat"));
    Ok(())
}
