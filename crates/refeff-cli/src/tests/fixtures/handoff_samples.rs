use super::*;

pub(in crate::tests) fn sample_jzzp_data() -> JzzpDatData {
    JzzpDatData {
        ns: 2,
        nphi: 2,
        nz: 3,
        nzp: 3,
        smax: 1.0,
        phimax: std::f64::consts::PI,
        zmax: 1.0,
        zpmax: 1.0,
        values: Array2::from_shape_fn((3, 3), |(z, zp)| 0.2 + z as f64 * 0.1 + zp as f64 * 0.05),
    }
}

pub(in crate::tests) fn sample_rhozzp_data() -> RhozzpDatData {
    RhozzpDatData {
        header_lines: vec![" # rhozzp diagnostic".to_string()],
        z_prime: Array1::from_vec(vec![0.01, 0.51, 1.01]),
        density: Array1::from_vec(vec![0.45, 0.35, 0.15]),
    }
}

pub(in crate::tests) fn sample_misc_dat() -> MiscDatData {
    MiscDatData {
        titles: vec![
            "Cu".to_string(),
            "absorbing".to_string(),
            " POT  SCF 100  5.5000   0, core-hole, AFOLP (folp(0)= 1.150)".to_string(),
        ],
    }
}

pub(in crate::tests) fn sample_convergence_scf() -> ScfConvergenceData {
    let header = " # it. E_fermi(eV)  Charge Distance  Partial Chg. D.  Convergence".to_string();
    let first = ScfConvergenceRow {
        iteration: 0,
        fermi_level_ev: -4.006,
        charge_distance: 0.0,
        partial_charge_distance: 0.0,
        converged: false,
    };
    let second = ScfConvergenceRow {
        iteration: 1,
        fermi_level_ev: -4.125,
        charge_distance: 0.3252,
        partial_charge_distance: 0.5599,
        converged: true,
    };
    ScfConvergenceData {
        detail_lines: vec![header.clone()],
        rows: vec![first.clone(), second.clone()],
        lines: vec![
            ScfConvergenceLine::Detail(header),
            ScfConvergenceLine::Row(first),
            ScfConvergenceLine::Row(second),
        ],
    }
}

pub(in crate::tests) fn sample_convergence_scf_fine() -> ScfConvergenceData {
    let title = " Electronic configuration".to_string();
    let detail = " 0     2   10.466".to_string();
    let row = ScfConvergenceRow {
        iteration: 2,
        fermi_level_ev: -4.250,
        charge_distance: 0.1025,
        partial_charge_distance: 0.2250,
        converged: true,
    };
    ScfConvergenceData {
        detail_lines: vec![title.clone(), detail.clone()],
        rows: vec![row.clone()],
        lines: vec![
            ScfConvergenceLine::Detail(title),
            ScfConvergenceLine::Detail(detail),
            ScfConvergenceLine::Row(row),
        ],
    }
}

pub(in crate::tests) fn sample_fort16() -> Fort16Data {
    Fort16Data {
        total_energy_hartree: Array1::from_vec(vec![
            -1_322.522_518_926_127_5,
            -1_652.786_043_284_159_6,
        ]),
    }
}

pub(in crate::tests) fn sample_pot_bin_data() -> PotBinData {
    let potentials = 1;
    PotBinData {
        titles: vec!["CLI wpot smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.0,
            fermi_level: 0.0,
            interstitial_potential: 0.0,
            interstitial_density: 0.0,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![12]),
        muffin_tin_radii: Array1::from_vec(vec![1.1]),
        norman_indices: Array1::from_vec(vec![40]),
        atomic_numbers: Array1::from_vec(vec![29]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.1]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        electron_density: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            0.035 * (row + 1) as f64
        }),
        coulomb_potential: Array2::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, potentials),
            |(row, _)| -1.2 - 0.02 * (row + 1) as f64,
        ),
        total_potential: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            -0.45 + 0.003 * (row + 1) as f64
        }),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

pub(in crate::tests) fn sample_apot_bin_data() -> ApotBinData {
    ApotBinData {
        sections: vec![
            apot_matrix_section(
                8,
                "rho(r,0:nphx+1) - atomic density for each unique potential",
                Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, 2), |(row, potential)| {
                    0.015 * (row + 1) as f64 + 0.25 * potential as f64
                }),
            ),
            apot_matrix_section(
                11,
                "vcoul(r,nph) - coulomb potential for each unique potential.",
                Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, 2), |(row, potential)| {
                    -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
                }),
            ),
        ],
    }
}

pub(in crate::tests) fn sample_bandstructure_dat() -> BandstructureDatData {
    BandstructureDatData {
        header_lines: vec![
            " # grid of            2  k-points.".to_string(),
            " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000".to_string(),
            " # Found between            1  and            2  number of bands.".to_string(),
        ],
        rows: vec![
            BandstructureRow {
                index: 1,
                k_point: [0.0, 0.5, 0.25],
                bands: Array1::from_vec(vec![-5.0, 1.25]),
            },
            BandstructureRow {
                index: 2,
                k_point: [0.5, 0.25, 0.0],
                bands: Array1::from_vec(vec![0.75]),
            },
        ],
    }
}

pub(in crate::tests) fn apot_matrix_section(
    section_number: usize,
    header: &str,
    values: Array2<f64>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: vec![header.to_string()],
        header_texts: vec![format!(" {header}")],
        column_labels: vec![],
        column_label_text: None,
        payload: ApotBinPayload::Matrix(ApotBinMatrix {
            value_type: ApotBinType::Double,
            values: ApotBinMatrixValues::Real(values),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    }
}
