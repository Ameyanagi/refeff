use super::*;

#[test]
fn generates_potentials_for_cif_without_potentials_card() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("two-site.cif");
    std::fs::write(
        &cif_path,
        r#"
data_two_site
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
H 0.0 0.0 0.0
O 0.5 0.5 0.5
"#,
    )?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF two-site.cif
TARGET 2
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.potentials.len(), 3);
    assert_eq!(doc.potentials[0].ipot, 0);
    assert_eq!(doc.potentials[0].z, Some(8));
    assert_eq!(doc.potentials[0].tag.as_deref(), Some("O"));
    assert_eq!(doc.potentials[0].xnatph, Some(0.01));
    assert_eq!(doc.potentials[1].ipot, 1);
    assert_eq!(doc.potentials[1].z, Some(1));
    assert_eq!(doc.potentials[1].tag.as_deref(), Some("H"));
    assert_eq!(doc.potentials[1].xnatph, Some(1.0));
    assert_eq!(doc.potentials[2].ipot, 2);
    assert_eq!(doc.potentials[2].z, Some(8));
    assert_eq!(doc.potentials[2].tag.as_deref(), Some("O"));
    assert_eq!(doc.potentials[2].xnatph, Some(1.0));
    Ok(())
}

#[test]
fn cif_equivalence_two_generates_atomic_number_potentials() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("three-site.cif");
    std::fs::write(
        &cif_path,
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
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF three-site.cif
TARGET 3
EQUI 2
FMS 4.0
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.cif_equivalence, 2);
    assert_eq!(doc.potentials.len(), 3);
    assert_eq!(doc.potentials[0].ipot, 0);
    assert_eq!(doc.potentials[0].z, Some(8));
    assert_eq!(doc.potentials[0].tag.as_deref(), Some("O"));
    assert_eq!(doc.potentials[1].ipot, 1);
    assert_eq!(doc.potentials[1].z, Some(1));
    assert_eq!(doc.potentials[1].xnatph, Some(2.0));
    assert_eq!(doc.potentials[2].ipot, 2);
    assert_eq!(doc.potentials[2].z, Some(8));
    assert!(doc.atoms.iter().any(|atom| atom.ipot == 1));
    assert!(!doc.atoms.iter().any(|atom| atom.ipot == 3));
    assert!(doc.active_cards.iter().any(|card| card == "EQUIVALENCE"));
    Ok(())
}

#[test]
fn rejects_bare_cif_equivalence_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EQUIVALENCE
END
"#,
    )?;

    let error = FeffDocument::from_input(&input).expect_err("bare EQUIVALENCE should fail");
    assert!(
        error
            .to_string()
            .contains("EQUIVALENCE requires a selector")
    );
    Ok(())
}

#[test]
fn cif_equivalence_four_collapses_when_potential_limit_is_exceeded() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("many-site.cif");
    let mut cif = String::from(
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
        cif.push_str(&format!("{symbol}{index} {x:.6} 0.0 0.0\n"));
    }
    std::fs::write(&cif_path, cif)?;

    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF many-site.cif
TARGET 2
EQUIVALENCE 4
FMS 4.0
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.cif_equivalence, 4);
    assert_eq!(doc.potentials.len(), 3);
    assert_eq!(doc.potentials[0].z, Some(8));
    assert_eq!(doc.potentials[1].z, Some(1));
    assert_eq!(doc.potentials[1].xnatph, Some(16.0));
    assert_eq!(doc.potentials[2].z, Some(8));
    assert_eq!(doc.potentials[2].xnatph, Some(16.0));
    assert!(doc.atoms.iter().any(|atom| atom.ipot == 1));
    assert!(!doc.atoms.iter().any(|atom| atom.ipot == 3));
    Ok(())
}

#[test]
fn generates_atoms_for_cif_without_atoms_card() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("two-site.cif");
    std::fs::write(
        &cif_path,
        r#"
data_two_site
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
H 0.0 0.0 0.0
O 0.5 0.5 0.5
"#,
    )?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF two-site.cif
TARGET 2
FMS 4.0
RMULTIPLIER 2.0
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert!(!doc.atoms.is_empty());
    assert_eq!(doc.atoms[0].ipot, 0);
    assert_eq!(
        (
            doc.atoms[0].x.round() as i32,
            doc.atoms[0].y.round() as i32,
            doc.atoms[0].z.round() as i32,
        ),
        (0, 0, 0)
    );
    assert!(
        doc.atoms
            .iter()
            .any(|atom| atom.ipot == 1 && (atom.x.abs() - 4.0).abs() < 1.0e-9)
    );
    Ok(())
}
