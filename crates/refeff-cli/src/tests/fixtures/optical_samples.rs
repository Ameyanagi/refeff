use super::*;

pub(in crate::tests) fn sample_fullspectrum_eps_dat() -> EpsDatData {
    EpsDatData {
        header_lines: vec!["# sample eps.dat".to_string()],
        omega: Array1::from_vec(vec![1.0, 2.0, 4.0, 7.0]),
        epsilon: Array1::from_vec(vec![
            Complex64::new(0.2, 0.05),
            Complex64::new(0.4, 0.12),
            Complex64::new(0.1, 0.07),
            Complex64::new(0.3, 0.03),
        ]),
        background_epsilon: Array1::from_vec(vec![
            Complex64::new(0.1, 0.02),
            Complex64::new(0.2, 0.04),
            Complex64::new(0.05, 0.025),
            Complex64::new(0.15, 0.01),
        ]),
        sigma: Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04]),
    }
}

pub(in crate::tests) fn sample_fullspectrum_osc_str_dat() -> OscStrDatData {
    OscStrDatData {
        header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
        rows: vec![OscStrRow {
            component: "Cu".to_string(),
            edge: "K".to_string(),
            core_hole_index: 1,
            effective_electron_count: 5.123,
        }],
    }
}

pub(in crate::tests) fn sample_fullspectrum_hamaker_dat() -> HamakerDatData {
    HamakerDatData {
        header_lines: vec!["# cached hamaker transform".to_string()],
        omega: Array1::from_vec(vec![1.0, 2.0, 4.0]),
        imaginary_axis_epsilon: Array1::from_vec(vec![
            Complex64::new(0.35, 0.0),
            Complex64::new(0.25, 0.0),
            Complex64::new(0.10, 0.0),
        ]),
    }
}

pub(in crate::tests) fn sample_crpa_dat() -> CrpaDatData {
    CrpaDatData {
        header_lines: vec!["U, n, U_Bare".to_string()],
        hubbard_u: 0.197_879_035_252_010,
        occupation: 1.0,
        bare_u: 0.694_283_422_651_496,
    }
}

pub(in crate::tests) fn sample_wscrn_dat() -> WscrnDatData {
    WscrnDatData {
        header_lines: vec![" # r       w_scrn(r)      v_ch(r)".to_string()],
        radius_bohr: Array1::from_vec(vec![
            0.150_733_046_3E-03,
            0.158_461_294_9E-03,
            0.166_585_779_2E-03,
        ]),
        screened_potential: Array1::from_vec(vec![
            0.267_288_234_6E+02,
            0.267_288_167_8E+02,
            0.267_288_030_6E+02,
        ]),
        core_hole_potential: Array1::from_vec(vec![
            0.291_616_524_4E+02,
            0.291_616_457_6E+02,
            0.291_616_320_4E+02,
        ]),
    }
}

pub(in crate::tests) fn sample_vtot_dat() -> VtotDatData {
    VtotDatData {
        header_lines: Vec::new(),
        radius_bohr: Array1::from_vec(vec![
            0.150_733_046_3E-03,
            0.158_461_294_9E-03,
            0.166_585_779_2E-03,
        ]),
        total_potential: Array1::from_vec(vec![
            -0.182_900_150_0E+06,
            -0.182_900_133_6E+06,
            -0.182_900_100_2E+06,
        ]),
        screened_core_hole_potential: Array1::from_vec(vec![
            0.267_288_234_6E+02,
            0.267_288_167_8E+02,
            0.267_288_030_6E+02,
        ]),
    }
}

pub(in crate::tests) fn sample_screen_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating screened core-hole potential ...".to_string(),
            "Done with module: screened core-hole potential.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_ldos_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating local density of states ...".to_string(),
            "Done with module: LDOS.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_eels_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EELS spectrum ...".to_string(),
            "Done with module: EELS.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_rixs_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating RIXS spectrum ...".to_string(),
            "Done with module: RIXS.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_compton_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating Compton scattering ...".to_string(),
            "Done with module: COMPTON.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_fullspectrum_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating full spectrum optical constants ...".to_string(),
            "Done with module: FULLSPECTRUM.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_xsph_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating potentials and phases ...".to_string(),
            "Done with module: potentials and phases.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_fms_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "FMS calculation of full Green's function ...".to_string(),
            "Done with module: FMS.".to_string(),
            "MKGTR: Tracing over Green's function ...".to_string(),
            "Done with module: MKGTR.".to_string(),
        ],
        line_terminators: vec![
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
        ],
    }
}

pub(in crate::tests) fn sample_path_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Pathfinder: finding scattering paths...".to_string(),
            "Preparing plane wave scattering amplitudes".to_string(),
            "Searching for paths".to_string(),
            "Done with module: pathfinder.".to_string(),
        ],
        line_terminators: vec![
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
        ],
    }
}

pub(in crate::tests) fn sample_genfmt_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EXAFS parameters ...".to_string(),
            "Done with module: EXAFS parameters (GENFMT).".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

pub(in crate::tests) fn sample_ff2x_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EXAFS ...".to_string(),
            "Done with module: EXAFS spectra.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}
