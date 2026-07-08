use super::*;

pub(in crate::tests) fn minimal_dym_text() -> &'static str {
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

pub(in crate::tests) fn reference_opcons_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("MPSE/Cu_OPCONS").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_self_mpse_cu_case() -> Result<Option<(PathBuf, PathBuf)>> {
    let Some(case) = GoldenCase::locate("MPSE/Cu") else {
        return Ok(None);
    };
    if !case.require_files(&["xsph.inp", "loss.dat"]) {
        return Ok(None);
    }
    let Some(zip_path) = case.zip() else {
        return Ok(None);
    };
    Ok(Some((case.path().to_path_buf(), zip_path)))
}

pub(in crate::tests) fn reference_crpa_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("CRPA").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_atomic_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("EXAFS/Cu")
        .filter(|case| case.require_files(&["feff.inp", "apot.bin", "fort.16", "fpf0.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_bn_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XANES/BN")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_bn_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XANES/BN").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_bn_positive_totvol_bounded_feff_pot_bin()
-> Result<Option<PathBuf>> {
    if let Some(path) = std::env::var_os("REFEFF_BN_POSITIVE_TOTVOL_BOUNDED_FEFF_POT_BIN") {
        let path = PathBuf::from(path);
        return Ok(path.is_file().then_some(path));
    }
    Ok(
        latest_generated_tmp_dir("feff-pot-bn-positive-totvol-bounded.", &["pot.bin"])?
            .map(|dir| dir.join("pot.bin")),
    )
}

pub(in crate::tests) fn reference_ybco_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("EXAFS/YBCO")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_ybco_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("EXAFS/YBCO").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_sf6_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("EXAFS/SF6")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_sf6_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("EXAFS/SF6").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_xanes_gecl4_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XANES/GeCl_4")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_xanes_gecl4_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XANES/GeCl_4").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_hubbard_nio_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("HUBBARD/NiO")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_hubbard_nio_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("HUBBARD/NiO").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_hubbard_nio_bounded_feff_pot_bin() -> Result<Option<PathBuf>> {
    if let Some(path) = std::env::var_os("REFEFF_NIO_BOUNDED_FEFF_POT_BIN") {
        let path = PathBuf::from(path);
        return Ok(path.is_file().then_some(path));
    }
    Ok(
        latest_generated_tmp_dir("feff-pot-nio-bounded.", &["pot.bin"])?
            .map(|dir| dir.join("pot.bin")),
    )
}

pub(in crate::tests) fn reference_xmcd_mnf2_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XMCD/MnF2_SPXAS")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_xmcd_mnf2_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XMCD/MnF2_SPXAS").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_xmcd_gd_l1_source_dir() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XMCD/Gd_L1")
        .filter(|case| case.require_files(&["pot.inp", "geom.dat"]))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_xmcd_gd_l1_pot_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XMCD/Gd_L1").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_nrixs_gecl4_xsph_phase_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("NRIXS/GeCl_4")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_xanes_cu_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "phase.bin",
        "xsect.dat",
    ];
    Ok(GoldenCase::locate("XANES/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn stock_xanes_cu_feff_input() -> Result<Option<PathBuf>> {
    let Some(root) = workspace_root() else {
        return Ok(None);
    };
    let input = root.join("feff10/examples/XANES/Cu/feff.inp");
    Ok(input.is_file().then_some(input))
}

pub(in crate::tests) fn reference_debye_dm_xanes_cu_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("DEBYE/DM/XANES/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_exafs_cu_scf_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("EXAFS/Cu_SCF")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_danes_cu_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("DANES/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_ldos_xanes_cu_spin_no_fms_xsph_source_dir()
-> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("LDOS/XANES_Cu_spin_no_fms")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_ldos_xanes_cu_fms_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("LDOS/XANES_Cu_fms")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_ldos_xanes_cu_spin_fms_short_xsph_source_dir()
-> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("LDOS/XANES_Cu_spin_fms_short")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_xes_cu_xsph_zip() -> Result<Option<PathBuf>> {
    Ok(GoldenCase::locate("XES/Cu").and_then(|case| case.zip()))
}

pub(in crate::tests) fn reference_fprime_gecl4_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("FPRIME/GeCl4")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_elnes_cu_xsph_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(GoldenCase::locate("ELNES/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_xanes_cu_screen_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "screen.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "fms.inp",
        "geom.dat",
        "wscrn.dat",
        "vtot.dat",
    ];
    Ok(GoldenCase::locate("XANES/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_exafs_cu_ff2x_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "ff2x.inp",
        "feff.bin",
        "list.dat",
        "xsect.dat",
        "chi.dat",
        "xmu.dat",
    ];
    Ok(GoldenCase::locate("EXAFS/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_elnes_cu_eels_source_dir() -> Result<Option<PathBuf>> {
    let required = [
        "eels.inp",
        "eels.dat",
        "xmu.dat",
        "xmu02.dat",
        "xmu03.dat",
        "xmu04.dat",
        "xmu05.dat",
        "xmu06.dat",
        "xmu07.dat",
        "xmu08.dat",
        "xmu09.dat",
    ];
    Ok(GoldenCase::locate("ELNES/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_debye_dm_exafs_cu_dmdw_source_dir() -> Result<Option<PathBuf>> {
    let required = ["dmdw.inp", "feff.dym", "dmdw.out"];
    Ok(GoldenCase::locate("DEBYE/DM/EXAFS/Cu")
        .filter(|case| case.require_files(&required))
        .map(|case| case.path().to_path_buf()))
}

pub(in crate::tests) fn reference_graphite_band_handoff() -> Result<Option<(PathBuf, PathBuf)>> {
    let Some(case) = GoldenCase::locate("KSPACE/Graphite") else {
        return Ok(None);
    };
    if !case.require_files(&["reciprocal.inp"]) {
        return Ok(None);
    }
    let Some(zip_path) = case.zip() else {
        return Ok(None);
    };
    Ok(Some((case.path().to_path_buf(), zip_path)))
}

pub(in crate::tests) fn reference_cr2gec_generated_band_output() -> Result<Option<PathBuf>> {
    let required = [
        "band.inp",
        "reciprocal.inp",
        "fms.inp",
        "global.inp",
        "phase.bin",
        "bandstructure.dat",
    ];
    latest_generated_tmp_dir("feff-band-cr2gec.", &required)
}

pub(in crate::tests) fn reference_ldos_xanes_cu_no_fms_source_case()
-> Result<Option<(PathBuf, PathBuf)>> {
    let source_required = [
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
    ];
    let expected_required = [
        "ldos.inp",
        "ldos00.dat",
        "ldos01.dat",
        "rhoc00.dat",
        "rhoc01.dat",
    ];
    let Some(source) =
        GoldenCase::locate("XANES/Cu").filter(|case| case.require_files(&source_required))
    else {
        return Ok(None);
    };
    let Some(expected) = GoldenCase::locate("LDOS/XANES_Cu_no_fms")
        .filter(|case| case.require_files(&expected_required))
    else {
        return Ok(None);
    };
    Ok(Some((
        source.path().to_path_buf(),
        expected.path().to_path_buf(),
    )))
}
