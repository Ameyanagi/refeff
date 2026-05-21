use super::*;

pub(in crate::tests) fn write_minimal_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu smoke test
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_bandstructure_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu band smoke test
BANDSTRUCTURE -5.0 10.0 0.25 2 64 T
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_screen_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu SCREEN cache run
EDGE K
COREHOLE RPA
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_dmdw_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
DEBYE 450 315 5 dym/force.dym 6 0 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_highz_template_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE test_element
HIGHZ
POTENTIALS
   0    XXX   Te
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_opcons_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu opcons run
OPCONS
NUMDENS 0 1.0
POTENTIALS
0 29 Cu
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_xsph_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu XSPH cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_self_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu SELF cache run
SELF
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_fms_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu FMS cache run
CONTROL 1 1 1 1 1 1
FMS 5.5
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_rixs_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu RIXS cache run
EDGE L3 VAL
RIXS 0.1 0.1
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_rhorrp_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu RHORRP cache run
EDGE K
DENSITY
line density.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 2
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_compton_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu compton cache run
COMPTON 1.0 3 0
CGRID 1.0 2 2 3 3
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_compton_rhozzp_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu compton rhozzp cache run
COMPTON 1.0 3 0
RHOZZP
CGRID 1.0 2 2 3 3
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_crpa_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Ce CRPA cache run
CRPA 2 3.5
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_ldos_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu LDOS cache run
LDOS -1 1 0.1 3 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_eels_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu EELS cache run
ELNES
300
0 1 0
2.4 0.0
5 3
0.0 0.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_eelsmdff_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu EELS-MDFF cache run
ELNES
300
0 1 0
2.4 0.0
5 3
0.0 0.0
MDFF 3
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_dmdw_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
DEBYE 450 315 5 feff.dym 2 0 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_path_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu PATH cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_genfmt_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu GENFMT cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_ff2x_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu FF2X cache run
CONTROL 1 1 1 1 1 1
RPATH 5.5
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

pub(in crate::tests) fn write_fullspectrum_cached_input(path: &std::path::Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu fullspectrum cache run
FULLSPECTRUM
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}
