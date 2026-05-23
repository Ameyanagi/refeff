use super::*;

#[test]
fn writes_compton_inp_from_cards() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
COMPTON
RHOZZP
CGRID 10 32 32 32 120
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        compton_inp_string(&doc)?,
        concat!(
            "run compton module?\n",
            "           1\n",
            "pqmax, npq\n",
            "   5.00000000            1000\n",
            "ns, nphi, nz, nzp\n",
            "  32  32  32 120\n",
            "smax, phimax, zmax, zpmax\n",
            "      0.00000      6.28319      0.00000     10.00000\n",
            "jpq? rhozzp? force_recalc_jzzp?\n",
            " T T F\n",
            "window_type (0=Step, 1=Hann), window_cutoff\n",
            "           1   0.00000000    \n",
            "temperature (in eV)\n",
            "      0.00000\n",
            "set_chemical_potential? chemical_potential(eV)\n",
            " F   0.00000000    \n",
            "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?\n",
            " F F F F F\n",
            "qhat_x qhat_y qhat_z\n",
            "   0.0000000000000000        0.0000000000000000        1.0000000000000000     \n",
        )
    );
    Ok(())
}

#[test]
fn writes_compton_aliases_into_compton_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
COMP 7.0 300 1
RHOZ
CGRI 12.0 20 21 22 23
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let compton_text = compton_inp_string(&doc)?;
    let compton = crate::ComptonInput::parse_str("compton.inp", &compton_text)?;

    assert!(compton.run);
    assert_eq!(compton.momentum.pqmax, 7.0);
    assert_eq!(compton.momentum.npq, 300);
    assert_eq!(compton.grid.ns, 20);
    assert_eq!(compton.grid.nphi, 21);
    assert_eq!(compton.grid.nz, 22);
    assert_eq!(compton.grid.nzp, 23);
    assert_eq!(compton.limits.zpmax, 12.0);
    assert!(compton.switches.jpq);
    assert!(compton.switches.rhozzp);
    assert!(compton.switches.force_recalc_jzzp);
    assert_eq!(crate::compton_input_string(&compton)?, compton_text);
    Ok(())
}

#[test]
fn writes_pot_inp_for_copper_defaults() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
EDGE K
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains("   1   1   1   1   0   0   0   0  11\n"));
    assert!(pot.contains("      1.72919      0.05000      0.00000    -40.00000"));
    assert!(pot.contains("   29    2        1.0000000000"));
    Ok(())
}

#[test]
fn writes_block_alias_rows_into_structure_outputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;

    assert_eq!(pot.control.nph, 1);
    assert_eq!(pot.potentials.len(), 2);
    assert_eq!(pot.potentials[0].z, 29);
    assert_eq!(atoms.atoms.len(), 2);
    assert_eq!(atoms.atoms[1].iph, 1);
    Ok(())
}

#[test]
fn writes_cif_equivalence_two_into_structure_outputs() -> Result<()> {
    let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
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
    )
    .map_err(|source| IoError::io(&cif_path, source))?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF three-site.cif
TARGET 3
EQUIVALENCE 2
FMS 4.0
EDGE K
XANES
END
"#,
    )
    .map_err(|source| IoError::io(&input_path, source))?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;

    assert_eq!(pot.control.nph, 2);
    assert_eq!(pot.potentials.len(), 3);
    assert_eq!(pot.potentials[0].z, 8);
    assert_eq!(pot.potentials[1].z, 1);
    assert_eq!(pot.potentials[1].xnatph, 200.0);
    assert_eq!(pot.potentials[2].z, 8);
    assert!(atoms.atoms.iter().any(|atom| atom.iph == 1));
    assert!(atoms.atoms.iter().all(|atom| atom.iph <= 2));
    Ok(())
}
