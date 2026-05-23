use super::*;

pub(crate) fn pot_bin_bench_data() -> PotBinData {
    let potentials = 6;
    let angular_count = 5;
    PotBinData {
        titles: vec![
            "Cu crystal".to_string(),
            "Gam_ch=1.000E+00 H-L exch Vi=0.000E+00 Vr=0.000E+00".to_string(),
        ],
        pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 1,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            interstitial_potential: -1.2,
            interstitial_density: 0.03,
            edge_position: 9.1,
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            core_valence_energy: -3.0,
            density_radius: 1.7,
            fermi_momentum: 0.9,
            total_charge: 42.0,
            total_volume: 11.0,
        },
        muffin_tin_indices: Array1::from_shape_fn(potentials, |potential| 12 + potential),
        muffin_tin_radii: Array1::from_shape_fn(potentials, |potential| {
            1.1 + potential as f64 * 0.02
        }),
        norman_indices: Array1::from_shape_fn(potentials, |potential| 30 + potential),
        atomic_numbers: Array1::from_shape_fn(
            potentials,
            |potential| {
                if potential % 2 == 0 { 29 } else { 8 }
            },
        ),
        kappa: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| orbital as i32 - 20),
        norman_radii: Array1::from_shape_fn(potentials, |potential| 2.0 + potential as f64 * 0.03),
        overlap_factors: Array1::from_shape_fn(potentials, |potential| {
            0.85 + potential as f64 * 0.01
        }),
        max_overlap_factors: Array1::from_shape_fn(potentials, |potential| {
            1.15 + potential as f64 * 0.01
        }),
        potential_multiplicities: Array1::from_shape_fn(potentials, |potential| {
            1.0 + potential as f64
        }),
        ionization: Array1::from_shape_fn(potentials, |potential| potential as f64 * 0.25),
        initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            0.001 * (row + 1) as f64
        }),
        initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            -0.001 * (row + 1) as f64
        }),
        large_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        large_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                0.01 * (coef + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                -0.01 * (coef + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        electron_density: pot_bin_radial_matrix(potentials, 0.01),
        coulomb_potential: pot_bin_radial_matrix(potentials, -0.02),
        total_potential: pot_bin_radial_matrix(potentials, -0.03),
        valence_density: pot_bin_radial_matrix(potentials, 0.004),
        valence_potential: pot_bin_radial_matrix(potentials, -0.005),
        magnetization_density: pot_bin_radial_matrix(potentials, 0.0002),
        orbital_occupancy: Array2::from_shape_fn(
            (POT_BIN_ORBITALS, potentials),
            |(orbital, potential)| 0.2 * orbital as f64 + potential as f64,
        ),
        orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
            -10.0 + orbital as f64 * 0.25
        }),
        occupied_orbital_indices: Array2::from_shape_fn(
            (POT_BIN_IORB_SLOTS, potentials),
            |(slot, _)| slot as i32 - 5,
        ),
        norman_charges: Array1::from_shape_fn(potentials, |potential| 8.0 + potential as f64 * 0.5),
        valence_occupancy: Array2::from_shape_fn(
            (angular_count, potentials),
            |(angular, potential)| 0.5 * angular as f64 + potential as f64,
        ),
        raw_text: None,
    }
}

pub(crate) fn pot_bin_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

pub(crate) fn phase_bin_bench_data() -> PhaseBinData {
    let spin_count = 2;
    let energy_count = 64;
    let potentials = 6;
    let q_count = 1;
    let transition_count = PHASE_BIN_DEFAULT_TRANSITION_COUNT;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 48,
        auxiliary_energy_count: 8,
        ihole: 1,
        fermi_index: 24,
        pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
        final_state_count: transition_count,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
        },
        energy_grid: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.1 * energy as f64, 0.01 * energy as f64)
        }),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, spin)| {
            Complex64::new(-1.0 + energy as f64 * 0.05, 0.02 * spin as f64)
        }),
        potentials: (0..potentials)
            .map(|potential| {
                phase_bin_bench_potential(
                    3,
                    if potential % 2 == 0 { 29 } else { 8 },
                    if potential % 2 == 0 { "Cu" } else { "O" },
                    energy_count,
                    spin_count,
                    potential as f64 * 0.01,
                )
            })
            .collect(),
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.001 * (energy + 1) as f64 + 0.1 * q_index as f64 + 0.01 * transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

pub(crate) fn phase_bin_bench_potential(
    lmax: usize,
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
    offset: f64,
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
                    offset + 0.0005 * energy as f64 + 0.01 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        ),
    }
}

pub(crate) fn feff_bin_bench_data() -> FeffBinData {
    let energy_count = 64;
    let path_count = 24;
    FeffBinData {
        version: "refeff-bench".to_string(),
        pad_width: 8,
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
        central_phase_shift: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.01 * energy as f64, -0.001 * energy as f64)
        }),
        complex_momentum: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + 0.02 * energy as f64, 0.01 * energy as f64)
        }),
        real_momentum: Array1::from_shape_fn(energy_count, |energy| 0.1 + 0.02 * energy as f64),
        paths: (0..path_count)
            .map(|path| feff_bin_bench_path(path, energy_count))
            .collect(),
        raw_text: None,
    }
}

pub(crate) fn feff_bin_bench_path(path: usize, energy_count: usize) -> FeffBinPath {
    let leg_count = 3 + path % 4;
    FeffBinPath {
        index: path + 1,
        degeneracy: 2.0 + path as f64 * 0.25,
        effective_half_path_length_bohr: 3.0 + path as f64 * 0.05,
        criterion: 100.0 / (path + 1) as f64,
        potential_indices: Array1::from_shape_fn(leg_count, |leg| leg % 2),
        positions: Array2::from_shape_fn((leg_count, 3), |(leg, axis)| {
            leg as f64 * 0.4 + axis as f64 * 0.125 + path as f64 * 0.01
        }),
        beta: Array1::from_shape_fn(leg_count, |leg| 0.1 * leg as f64),
        eta: Array1::from_shape_fn(leg_count, |leg| 0.2 * leg as f64),
        leg_distances: Array1::from_shape_fn(leg_count, |leg| 1.0 + 0.05 * leg as f64),
        amplitude: Array1::from_shape_fn(energy_count, |energy| {
            0.001 * (energy + 1) as f64 + path as f64 * 0.0001
        }),
        phase: Array1::from_shape_fn(energy_count, |energy| -0.01 * energy as f64),
    }
}
