use super::*;

pub(crate) fn list_dat_bench_data() -> ListDatData {
    ListDatData {
        titles: vec![
            "PATH  Rmax= 6.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%".to_string(),
        ],
        entries: (0..256)
            .map(|path| ListDatEntry {
                path_index: path + 1,
                sigma2: 0.0,
                amplitude_ratio: 100.0 / (path + 1) as f64,
                degeneracy: 2.0 + (path % 8) as f64,
                leg_count: 2 + path % 6,
                effective_half_path_length_angstrom: 1.5 + path as f64 * 0.015,
            })
            .collect(),
    }
}

pub(crate) fn log_dat_bench_data() -> LogDatData {
    LogDatData {
        version: "FEFF 10.0.0".to_string(),
        preamble_lines: vec![
            "Resetting lmaxsc to 2 for iph =    0.  Use  UNFREEZE to prevent this.".to_string(),
            "Resetting lmaxsc to 2 for iph =    1.  Use  UNFREEZE to prevent this.".to_string(),
        ],
        core_hole_lifetime_ev: Some(1.729),
        post_core_lines: Vec::new(),
        titles: vec![" Cu crystal".to_string()],
        calculation_summary: Some("Cu K edge XANES using RPA corehole.".to_string()),
        features: vec![
            "Debye-Waller factors".to_string(),
            "Many-Pole Self-Energy".to_string(),
            "Self-Consistent Field potentials".to_string(),
        ],
        cards: [
            "ATOMS",
            "CONTROL",
            "EXCHANGE",
            "TITLE",
            "DEBYE",
            "POTENTIALS",
            "XANES",
            "CORRECTIONS",
            "SCF",
            "FMS",
            "MPSE",
            "SFCONV",
            "COREHOLE",
            "OPCONS",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        trailing_lines: Vec::new(),
    }
}

pub(crate) fn run_stdout_bench_data() -> RunStdoutData {
    let mut lines = Vec::new();
    for cycle in 0..128 {
        lines.push(format!("Calculating synthetic module {cycle} ..."));
        lines.push("FEFF-serial using 1 thread.".to_string());
        lines.push(format!("Done with module: synthetic module {cycle}."));
    }
    let text = lines.join("\n");
    match parse_run_stdout(&text) {
        Ok(data) => data,
        Err(_) => RunStdoutData {
            lines,
            line_endings: Vec::new(),
            module_events: Vec::new(),
        },
    }
}

pub(crate) fn run_stderr_bench_data() -> RunStderrData {
    let lines = (0..128)
        .map(|index| {
            if index % 7 == 0 {
                "Note: The following floating-point exceptions are signalling: IEEE_INVALID_FLAG"
                    .to_string()
            } else {
                "Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG"
                    .to_string()
            }
        })
        .collect::<Vec<_>>();
    let text = lines.join("\n");
    match parse_run_stderr(&text) {
        Ok(data) => data,
        Err(_) => RunStderrData {
            lines,
            line_endings: Vec::new(),
            floating_point_notes: Vec::new(),
        },
    }
}

pub(crate) fn paths_dat_bench_data() -> PathsDatData {
    let paths = (0..256)
        .map(|path| PathsDatPath {
            index: path + 1,
            degeneracy: 4.0 + (path % 8) as f64,
            effective_half_path_length_angstrom: 2.0 + 0.01 * path as f64,
            row_header:
                "      x           y           z     ipot  label      rleg      beta        eta"
                    .to_string(),
            atoms: vec![
                PathsDatAtom {
                    position_angstrom: [1.0 + 0.01 * path as f64, 0.5, 0.0],
                    potential_index: 1,
                    label: "Cu1".to_string(),
                    leg_distance_angstrom: Some(2.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(45.0),
                },
                PathsDatAtom {
                    position_angstrom: [-1.0 - 0.01 * path as f64, 0.5, 0.0],
                    potential_index: 1,
                    label: "Cu1".to_string(),
                    leg_distance_angstrom: Some(2.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(135.0),
                },
                PathsDatAtom {
                    position_angstrom: [0.0, 0.0, 0.0],
                    potential_index: 0,
                    label: "Cu0".to_string(),
                    leg_distance_angstrom: Some(2.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(225.0),
                },
            ],
        })
        .collect();
    PathsDatData {
        titles: vec!["TITLE Cu crystal".to_string()],
        paths,
    }
}

pub(crate) fn dym_bench_data() -> DymData {
    let atom_count = 32_usize;
    let atomic_numbers =
        Array1::from_iter((0..atom_count).map(|index| if index % 2 == 0 { 29 } else { 8 }));
    let atomic_masses = Array1::from_iter(
        (0..atom_count).map(|index| if index % 2 == 0 { 63.546 } else { 15.999 }),
    );
    let positions = Array2::from_shape_fn((atom_count, 3), |(atom, axis)| match axis {
        0 => atom as f64 * 0.25,
        1 => (atom % 7) as f64 * 0.1,
        _ => (atom % 5) as f64 * 0.05,
    });
    let mut force_constants = Array4::zeros((atom_count, atom_count, 3, 3));
    for i_atom in 0..atom_count {
        for j_atom in 0..atom_count {
            let diagonal = if i_atom == j_atom { 0.2 } else { -0.002 };
            for row in 0..3 {
                for column in 0..3 {
                    force_constants[[i_atom, j_atom, row, column]] = if row == column {
                        diagonal
                    } else {
                        0.0001 * (i_atom as f64 - j_atom as f64)
                    };
                }
            }
        }
    }

    DymData {
        dym_type: 1,
        atomic_numbers,
        atomic_masses,
        coordinates: DymCoordinates::Cartesian(positions),
        force_constants,
        type2_metadata: None,
        dipole_derivatives: None,
    }
}

pub(crate) fn grid_inp_bench_data() -> GridInput {
    let mut records = (0..8)
        .map(|index| {
            GridRecord::Regular(GridRegularRecord {
                kind: if index % 2 == 0 {
                    GridKind::Energy
                } else {
                    GridKind::WaveNumber
                },
                minimum: if index == 0 {
                    GridMinimum::Value(-15.0)
                } else {
                    GridMinimum::Last
                },
                maximum: 5.0 + index as f64,
                step: 0.05 + 0.01 * index as f64,
            })
        })
        .collect::<Vec<_>>();
    records.push(GridRecord::User(GridUserRecord {
        points: (0..64)
            .map(|index| GridPoint {
                real: -2.0 + 0.1 * index as f64,
                imaginary: if index % 3 == 0 { 0.05 } else { 0.0 },
            })
            .collect(),
    }));
    GridInput { records }
}

pub(crate) fn config_inp_bench_data() -> ConfigInput {
    ConfigInput {
        records: (0..16)
            .map(|index| ConfigRecord {
                potential_index: index,
                element: if index % 2 == 0 {
                    "Cu".to_string()
                } else {
                    "Ge".to_string()
                },
                noble_gas: (index % 3 == 0).then(|| "Ar".to_string()),
                states: vec![
                    ConfigState {
                        orbital: "3d".to_string(),
                        occupations: vec![
                            ConfigOccupation {
                                occupation: 4.0 + (index % 3) as f64,
                                spin: None,
                            },
                            ConfigOccupation {
                                occupation: 6.0,
                                spin: None,
                            },
                        ],
                    },
                    ConfigState {
                        orbital: "4s".to_string(),
                        occupations: vec![ConfigOccupation {
                            occupation: 1.0,
                            spin: Some((index % 2) as f64),
                        }],
                    },
                    ConfigState {
                        orbital: "4p".to_string(),
                        occupations: vec![
                            ConfigOccupation {
                                occupation: 0.0,
                                spin: Some(1.0),
                            },
                            ConfigOccupation {
                                occupation: 0.0,
                                spin: Some(0.0),
                            },
                        ],
                    },
                ],
            })
            .collect(),
    }
}

pub(crate) fn spring_inp_bench_data() -> SpringInput {
    SpringInput {
        vdos: Some(SpringVdos {
            resolution: 0.02,
            wmax: 20.0,
            dosfit: 0.1,
            acut: 3.0,
        }),
        print_projected: Some(8),
        stretches: (0..64)
            .map(|index| SpringStretch {
                first_atom: index,
                second_atom: index + 1,
                force_constant: 25.0 + index as f64,
                distance_tolerance_percent: 2.0 + (index % 4) as f64,
            })
            .collect(),
        angles: (0..64)
            .map(|index| SpringAngle {
                first_atom: index,
                center_atom: index + 1,
                third_atom: index + 2,
                force_constant: 40.0 + 3.0 * index as f64,
                angle_tolerance_percent: 5.0 + (index % 5) as f64,
            })
            .collect(),
    }
}
