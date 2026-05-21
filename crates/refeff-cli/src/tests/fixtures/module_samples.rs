use super::*;

pub(in crate::tests) fn sample_ldos_dat() -> Result<LdosDatData> {
    Ok(LdosDatData {
        header_lines: vec![
            "#  Fermi level (eV):  -3.777".to_string(),
            "#      e        sDOS           pDOS          dDOS          fDOS".to_string(),
        ],
        fermi_level_ev: Some(-3.777),
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: Array1::from_vec(vec![-1.0, 0.0, 1.0]),
        density: Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 1.2E-4, 2.2E-4,
                3.2E-4, 4.2E-4,
            ],
        )?,
    })
}

pub(in crate::tests) fn sample_eels_dat() -> EelsDatData {
    EelsDatData {
        header_lines: vec![
            "# Orientation averaged EELS calculation".to_string(),
            "#  Energy       total         atomic-bg     fine-struct".to_string(),
        ],
        energy_loss_ev: Array1::from_vec(vec![8979.41, 8980.98, 8982.40]),
        total: Array1::from_vec(vec![0.123_014E-12, 0.146_285E-12, 0.176_683E-12]),
        atomic_background: Array1::from_vec(vec![0.138_430E-12, 0.166_322E-12, 0.203_202E-12]),
        fine_structure: Array1::from_vec(vec![-0.154_167E-13, -0.200_377E-13, -0.265_188E-13]),
        tensor: None,
    }
}

pub(in crate::tests) fn sample_mdff_dat() -> Result<MdffDatData> {
    Ok(MdffDatData {
        header_lines: vec![
            "# Orientation sensitive EELS calculation - beam energy =    300keV".to_string(),
            "#  Energy       total".to_string(),
        ],
        energy_loss_ev: Array1::from_vec(vec![10.0, 12.5]),
        spectrum: Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.25),
                Complex64::new(0.5, -0.1),
                Complex64::new(1.2, 0.2),
                Complex64::new(0.8, -0.05),
            ],
        )?,
    })
}

pub(in crate::tests) fn sample_mpse_dat() -> MpseDatData {
    MpseDatData {
        header_lines: vec!["# XSPH MPSE self-energy sidecar".to_string()],
        energy_ev: Array1::from_vec(vec![0.038_099_840_30, 0.152_399_361_2]),
        self_energy: Array1::from_vec(vec![
            Complex64::new(0.001_436_696_198, -0.000_007_842_984_015),
            Complex64::new(0.005_774_807_411, -0.000_124_742_315_9),
        ]),
        renormalization: Some(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ])),
        renormalization_magnitude: Some(Array1::from_vec(vec![1.0, 1.0])),
        renormalization_phase: Some(Array1::from_vec(vec![0.0, 0.0])),
        inelastic_mean_free_path: Some(Array1::from_vec(vec![48_578.245_52, 6_108.567_091])),
    }
}

pub(in crate::tests) fn sample_emesh_dat() -> EmeshDatData {
    EmeshDatData {
        edge_hartree: 333.333,
        bohr_angstrom: 0.529_177_249,
        edge_ev: 9_071.2,
        spectrum: 0,
        fermi_index: 1,
        indices: Array1::from_vec(vec![1, 2, 3]),
        energy_ev: Array1::from_vec(vec![0.0, 1.5, 3.0]),
        wave_number_inverse_angstrom: Array1::from_vec(vec![0.0, 0.627, 0.887]),
    }
}

pub(in crate::tests) fn sample_emesh_bin() -> EmeshBinData {
    EmeshBinData {
        point_count_declared: 3,
        horizontal_count: 2,
        danes_extension_count: 1,
        energy_hartree: Array1::from_vec(vec![
            Complex64::new(-0.25, 0.01),
            Complex64::new(0.0, 0.02),
            Complex64::new(0.5, 0.03),
        ]),
    }
}

pub(in crate::tests) fn sample_exc_dat() -> ExcDatData {
    ExcDatData {
        header_lines: vec!["# SELF excitation poles".to_string()],
        energy_ev: Array1::from_vec(vec![15.0, 27.5]),
        broadening_ev: Array1::from_vec(vec![0.15, 0.275]),
        oscillator_strength: Array1::from_vec(vec![0.75, 0.25]),
        auxiliary_weight: Some(Array1::from_vec(vec![1.0, 0.5])),
    }
}

pub(in crate::tests) fn sample_paths_dat() -> PathsDatData {
    PathsDatData {
        titles: vec![
            "PATH  Rmax= 5.500,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%".to_string(),
        ],
        paths: vec![PathsDatPath {
            index: 1,
            degeneracy: 12.0,
            effective_half_path_length_angstrom: 2.5527,
            row_header:
                "      x           y           z     ipot  label      rleg      beta        eta"
                    .to_string(),
            atoms: vec![
                PathsDatAtom {
                    position_angstrom: [-1.805, -1.805, 0.0],
                    potential_index: 1,
                    label: "Cu".to_string(),
                    leg_distance_angstrom: Some(2.5527),
                    beta_degrees: Some(180.0),
                    eta_degrees: Some(0.0),
                },
                PathsDatAtom {
                    position_angstrom: [0.0, 0.0, 0.0],
                    potential_index: 0,
                    label: "Cu".to_string(),
                    leg_distance_angstrom: Some(2.5527),
                    beta_degrees: Some(180.0),
                    eta_degrees: Some(0.0),
                },
            ],
        }],
    }
}

pub(in crate::tests) fn sample_phase_bin_data() -> PhaseBinData {
    let spin_count = 1;
    let energy_count = 2;
    let transition_count = 2;
    let q_count = 1;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 2,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: 4,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: -0.35,
            edge_energy: 9.8,
        },
        energy_grid: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + energy as f64, 0.01 * energy as f64)
        }),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
            Complex64::new(-1.0 + 0.2 * energy as f64, 0.0)
        }),
        potentials: vec![
            sample_phase_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
            sample_phase_potential(1, 8, "O", energy_count, spin_count, 0.2),
        ],
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

pub(in crate::tests) fn sample_phase_potential(
    lmax: usize,
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
    scale: f64,
) -> PhaseBinPotential {
    let l_count = 2 * lmax + 1;
    PhaseBinPotential {
        lmax,
        atomic_number,
        label: label.to_string(),
        phase_shifts: Array3::from_shape_fn(
            (energy_count, l_count, spin_count),
            |(energy, l_slot, spin)| {
                Complex64::new(
                    scale + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        ),
    }
}

pub(in crate::tests) fn sample_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu crystal".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            edge_energy: 9.1,
            chemical_potential: -0.4,
        },
        core_hole_width_ev: 1.23,
        main_energy_count: 2,
        fermi_index: 1,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(1.25, 0.01),
            Complex64::new(1.5, 0.02),
        ]),
        normalized_background: Array1::from_vec(vec![2.0, 2.5]),
        cross_section: Array1::from_vec(vec![Complex64::new(3.0, -0.4), Complex64::new(3.5, -0.5)]),
    }
}

pub(in crate::tests) fn sample_fms_bin_data() -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 2,
        main_energy_count: 1,
        auxiliary_energy_count: 0,
        highest_potential_index: 1,
        pad_width: 8,
        declared_spectrum_count: Some(2),
        spectra: Array2::from_shape_fn((2, 2), |(spectrum, energy)| {
            Complex64::new(
                0.25 * (energy + 1) as f64 + spectrum as f64,
                -0.05 * (energy + 1) as f64 - spectrum as f64,
            )
        }),
    }
}

pub(in crate::tests) fn sample_rixs_map_data() -> RixsMapData {
    RixsMapData {
        header_lines: vec!["# sample RIXS map".to_string()],
        block_lengths: vec![2, 2],
        first_energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0, 11_540.0, 11_541.0]),
        second_energy_ev: Array1::from_vec(vec![-15.0, -15.0, -14.0, -14.0]),
        channels: Array2::from_shape_fn((4, 2), |(row, channel)| {
            1.0e-6 * (row + 1) as f64 + 2.0e-7 * channel as f64
        }),
    }
}

pub(in crate::tests) fn sample_rhorrp_density_text_data() -> RhorrpDensityTextData {
    RhorrpDensityTextData {
        points_angstrom: Array2::from_shape_fn((2, 3), |(row, coordinate)| {
            if row == 1 && coordinate == 0 {
                0.529_177_249
            } else {
                0.0
            }
        }),
        density_per_angstrom3: Array1::from_vec(vec![1.0, 2.0]),
        nearest: Some(RhorrpNearestAtomColumns {
            displacement_bohr: Array2::from_shape_fn((2, 3), |(row, coordinate)| {
                if row == 1 && coordinate == 0 {
                    1.0
                } else {
                    0.0
                }
            }),
            atom_indices: Array1::from_vec(vec![0, 0]),
            potential_indices: Array1::from_vec(vec![0, 0]),
        }),
    }
}

pub(in crate::tests) fn sample_feff_bin_data() -> FeffBinData {
    FeffBinData {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        ihole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 9.1,
        potentials: vec![
            FeffBinPotential {
                label: "Cu".to_string(),
                atomic_number: 29,
            },
            FeffBinPotential {
                label: "O".to_string(),
                atomic_number: 8,
            },
        ],
        central_phase_shift: Array1::from_vec(vec![
            Complex64::new(0.1, -0.01),
            Complex64::new(0.2, -0.02),
            Complex64::new(0.3, -0.03),
        ]),
        complex_momentum: Array1::from_vec(vec![
            Complex64::new(1.0, 0.1),
            Complex64::new(1.1, 0.2),
            Complex64::new(1.2, 0.3),
        ]),
        real_momentum: Array1::from_vec(vec![0.5, 0.6, 0.7]),
        paths: vec![FeffBinPath {
            index: 17,
            degeneracy: 4.0,
            effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
            criterion: 12.5,
            potential_indices: Array1::from_vec(vec![0, 1, 0]),
            positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                (0, 0..=2) => 0.0,
                (1, 0) => 1.0,
                (1, 1) => 0.5,
                (1, 2) => 0.0,
                (2, 0) => -1.0,
                (2, 1) => 0.25,
                (2, 2) => 0.0,
                _ => 0.0,
            }),
            beta: Array1::from_vec(vec![0.1, 0.2, 0.3]),
            eta: Array1::from_vec(vec![0.4, 0.5, 0.6]),
            leg_distances: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
            phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
        }],
        raw_text: None,
    }
}

pub(in crate::tests) fn sample_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: vec![ListDatEntry {
            path_index: 17,
            sigma2: 0.0,
            amplitude_ratio: 12.5,
            degeneracy: 4.0,
            leg_count: 3,
            effective_half_path_length_angstrom: 2.5,
        }],
    }
}

pub(in crate::tests) fn sample_xmu_dat() -> XmuDatData {
    XmuDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0"
                .to_string(),
            "# xsedge+ 50, used to normalize mu           1.234500E+00".to_string(),
        ],
        normalization: Some(1.2345),
        photon_energy_ev: Array1::from_vec(vec![8979.0, 8980.0, 8981.0]),
        relative_energy_ev: Array1::from_vec(vec![0.0, 1.0, 2.0]),
        wave_number: Array1::from_vec(vec![0.0, 0.512, 0.724]),
        mu: Array1::from_vec(vec![1.0, 1.1, 1.2]),
        mu0: Array1::from_vec(vec![0.9, 0.95, 1.0]),
        chi: Array1::from_vec(vec![0.1, 0.15, 0.2]),
    }
}

pub(in crate::tests) fn sample_chi_dat() -> ChiDatData {
    ChiDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0"
                .to_string(),
            "#       k          chi          mag           phase @#".to_string(),
        ],
        wave_number: Array1::from_vec(vec![0.0, 0.05, 0.1]),
        chi: Array1::from_vec(vec![-0.115_938_3, -0.119_413_8, -0.122_912_6]),
        magnitude: Array1::from_vec(vec![0.270_227_8, 0.272_670_8, 0.275_083_6]),
        phase: Array1::from_vec(vec![-2.698_164, -2.688_285, -2.678_386]),
        phase_minus_2kr: None,
        ckp_real: None,
        ckp_imag: None,
    }
}

pub(in crate::tests) fn sample_danes_dat() -> DanesDatData {
    DanesDatData {
        header_lines: vec!["# E  matsub. sommerf. anomal. tale, total, differ.".to_string()],
        energy_ev: Array1::from_vec(vec![-18.690, -17.122, -15.703]),
        matsubara: Array1::from_vec(vec![0.0, 0.0, 0.0]),
        sommerfeld: Array1::from_vec(vec![0.0, 0.0, 0.0]),
        anomalous: Array1::from_vec(vec![10.097, 10.603, 11.159]),
        tail: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
        total: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
        difference: Array1::from_vec(vec![-5.4576, -5.6591, -5.8651]),
    }
}

pub(in crate::tests) fn sample_xscorr_complex_table() -> XscorrComplexTable {
    XscorrComplexTable {
        energy_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
        values: Array1::from_vec(vec![
            Complex64::new(-0.000_020_637_731_56, 0.000_120_322_770_8),
            Complex64::new(-0.000_021_177_763_91, 0.000_123_685_052_9),
        ]),
    }
}

pub(in crate::tests) fn sample_xscorr_curve_dat() -> XscorrCurveDatData {
    XscorrCurveDatData {
        energy: Array1::from_vec(vec![
            Complex64::new(-0.138_801_301_5, 0.000_183_746_545),
            Complex64::new(-0.138_801_301_5, 0.000_367_493_09),
        ]),
        values: Array1::from_vec(vec![
            Complex64::new(-0.000_028_662, 0.000_237_48),
            Complex64::new(-0.000_028_683, 0.000_237_44),
        ]),
    }
}

pub(in crate::tests) fn sample_xscorr_raw_dat() -> XscorrRawDatData {
    XscorrRawDatData {
        temperature_hartree: 0.0,
        electronic_temperature_ev: 0.0,
        loss_ev: 0.864_59,
        fermi_energy_ev: -3.776_977_18,
        pole_count: 0,
        omega_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
        cchi: Array1::from_vec(vec![
            Complex64::new(-0.000_016_299_5, 0.000_115_24),
            Complex64::new(-0.000_016_898_337_65, 0.000_118_558_222_9),
        ]),
        one_minus_fermi: Array1::from_vec(vec![0.5, 0.514_017_875_2]),
        xmu0: Array1::from_vec(vec![
            Complex64::new(-0.000_032_599, 0.000_230_48),
            Complex64::new(-0.000_032_875, 0.000_230_65),
        ]),
    }
}

pub(in crate::tests) fn sample_dmdw_out() -> DmdwOutData {
    let mut section = DmdwOutSection::new(DmdwOutSubject::PathIndices(vec![1, 2]));
    section.reduced_mass_amu = Some(31.773);
    section.path_length_angstrom = Some(2.5323);
    section.sigma2_1e_minus_3_angstrom2 = Some(11.8576);

    DmdwOutData {
        header: Some(DmdwOutHeader {
            lanczos_recursion_order: 2,
            temperature: DmdwOutTemperature::Single(450.0),
            dynamical_matrix_file: "feff.dym".to_string(),
        }),
        mass_enhancement_header: false,
        sections: vec![section],
    }
}
