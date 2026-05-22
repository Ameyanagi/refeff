use super::expand::apply_cif_symmetry_operation;
use super::parse::tokenize_cif_line;
use super::{
    CifEquivalence, expand_cif_cluster, expand_cif_cluster_with_equivalence, expand_cif_structure,
    expand_cif_structure_with_equivalence, parse_cif,
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
fn parses_first_cif_data_block_like_ciftbx_blank_data_call() -> crate::Result<()> {
    let cif = r#"
data_first
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
H1 0 0 0
data_second
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
O1 0.5 0.5 0.5
"#;

    let parsed = parse_cif(cif)?;

    assert_eq!(parsed.data_block.as_deref(), Some("first"));
    assert_eq!(parsed.cell.a, 4.0);
    assert_eq!(parsed.atom_sites.len(), 1);
    assert_eq!(parsed.atom_sites[0].label.as_deref(), Some("H1"));
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

    let structure = expand_cif_structure_with_equivalence(&cif, 3, CifEquivalence::AtomicNumber)?;
    assert_eq!(structure.absorber_atomic_number, 8);
    assert_eq!(structure.absorber_label, "O");
    assert_eq!(structure.potentials, [1, 1, 2]);
    assert_eq!(structure.site_atomic_numbers, [1, 8]);
    assert_eq!(structure.site_labels, ["H", "O"]);
    assert_eq!(structure.site_multiplicities, [2, 1]);
    assert_eq!(structure.labels, ["O", "H", "O"]);

    let cluster = expand_cif_cluster_with_equivalence(&cif, 3, 4.0, CifEquivalence::AtomicNumber)?;
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
