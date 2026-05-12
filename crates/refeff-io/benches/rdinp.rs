use std::path::Path;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
    POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    FeffDocument, FeffInput, MtdpData, PotBinData, PotBinScalars, PotentialDatSetInput,
    mtdp_string, parse_mtdp, parse_pot_bin, pot_bin_string, potential_dat_outputs, rdinp,
};

const FALLBACK_INPUT: &str = r#"
TITLE Cu crystal
EDGE K
SCF 5.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.805 1.805 0.0 1 Cu1 2.55266 1
-1.805 1.805 0.0 1 Cu1 2.55266 2
1.805 -1.805 0.0 1 Cu1 2.55266 3
-1.805 -1.805 0.0 1 Cu1 2.55266 4
END
"#;

fn bench_parse(c: &mut Criterion) {
    let input = bench_input();
    if let Err(err) = FeffInput::parse_str("bench.inp", &input) {
        eprintln!("skipping parse_cu_feff_input benchmark: {err}");
        return;
    }
    c.bench_function("parse_cu_feff_input", |b| {
        b.iter(|| black_box(FeffInput::parse_str("bench.inp", black_box(&input))));
    });
}

fn bench_rdinp_outputs(c: &mut Criterion) {
    let input = bench_input();
    let parsed = match FeffInput::parse_str("bench.inp", &input) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("skipping render_rdinp_text_outputs benchmark: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&parsed) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping render_rdinp_text_outputs benchmark: {err}");
            return;
        }
    };

    c.bench_function("render_rdinp_text_outputs", |b| {
        b.iter(|| black_box(rdinp::text_outputs(black_box(&document))));
    });
}

fn bench_potential_outputs(c: &mut Criterion) {
    let state = PotOutputBenchState::new();
    c.bench_function("render_wpot_potential_dat_outputs", |b| {
        b.iter(|| black_box(potential_dat_outputs(black_box(state.input()))));
    });
}

fn bench_mtdp(c: &mut Criterion) {
    let data = mtdp_bench_data();
    let text = match mtdp_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping mtdp benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_mtdp_text", |b| {
        b.iter(|| black_box(mtdp_string(black_box(&data))));
    });
    c.bench_function("parse_mtdp_text", |b| {
        b.iter(|| black_box(parse_mtdp(black_box(&text))));
    });
}

fn bench_pot_bin(c: &mut Criterion) {
    let data = pot_bin_bench_data();
    let text = match pot_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping pot.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_pot_bin_text", |b| {
        b.iter(|| black_box(pot_bin_string(black_box(&data))));
    });
    c.bench_function("parse_pot_bin_text", |b| {
        b.iter(|| black_box(parse_pot_bin(black_box(&text))));
    });
}

fn bench_input() -> String {
    let local_cu =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples/EXAFS/Cu/feff.inp");
    std::fs::read_to_string(local_cu).unwrap_or_else(|_| FALLBACK_INPUT.to_string())
}

struct PotOutputBenchState {
    muffin_tin_indices: Vec<usize>,
    norman_indices: Vec<usize>,
    titles: Vec<String>,
    electron_density: Array2<f64>,
    free_density: Array2<f64>,
    overlapped_coulomb: Array2<f64>,
    free_coulomb: Array2<f64>,
    total_potential: Array2<f64>,
}

impl PotOutputBenchState {
    fn new() -> Self {
        let rows = 251;
        let potentials = 6;
        Self {
            muffin_tin_indices: (0..potentials).map(|potential| 12 + potential).collect(),
            norman_indices: (0..potentials)
                .map(|potential| 40 + 2 * potential)
                .collect(),
            titles: vec![
                "Cu crystal".to_string(),
                "Gam_ch=1.000E+00 H-L exch Vi=0.000E+00 Vr=0.000E+00".to_string(),
            ],
            electron_density: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                0.035 * (row + 1) as f64 + 0.125 * potential as f64
            }),
            free_density: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                0.015 * (row + 1) as f64 + 0.25 * potential as f64
            }),
            overlapped_coulomb: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -1.2 * (potential + 1) as f64 - 0.02 * (row + 1) as f64
            }),
            free_coulomb: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
            }),
            total_potential: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -0.45 * (potential + 1) as f64 + 0.003 * (row + 1) as f64
            }),
        }
    }

    fn input(&self) -> PotentialDatSetInput<'_> {
        PotentialDatSetInput {
            highest_potential_index: self.muffin_tin_indices.len() - 1,
            muffin_tin_indices: &self.muffin_tin_indices,
            norman_indices: &self.norman_indices,
            titles: &self.titles,
            electron_density: self.electron_density.view(),
            free_density: self.free_density.view(),
            overlapped_coulomb: self.overlapped_coulomb.view(),
            free_coulomb: self.free_coulomb.view(),
            total_potential: self.total_potential.view(),
        }
    }
}

fn mtdp_bench_data() -> MtdpData {
    let radial_count = 251;
    let atom_count = 12;
    let empty_count = 4;
    MtdpData {
        radial_count,
        atomic_numbers: Array1::from_shape_fn(
            atom_count,
            |atom| if atom % 3 == 0 { 29 } else { 8 },
        ),
        atom_coordinates: Array2::from_shape_fn((atom_count, 3), |(atom, axis)| {
            atom as f64 * 0.25 + axis as f64 * 0.125
        }),
        atom_radii: Array1::from_shape_fn(atom_count, |atom| 0.4 + atom as f64 * 0.01),
        atom_radius_indices: Array1::from_shape_fn(atom_count, |atom| 40 + atom),
        atom_density: Array2::from_shape_fn((radial_count, atom_count), |(radial, atom)| {
            0.001 * (radial + 1) as f64 + 0.0001 * atom as f64
        }),
        atom_potential: Array2::from_shape_fn((radial_count, atom_count), |(radial, atom)| {
            -1.0 - 0.01 * radial as f64 - 0.05 * atom as f64
        }),
        empty_sphere_coordinates: Array2::from_shape_fn((empty_count, 3), |(sphere, axis)| {
            sphere as f64 * 0.5 + axis as f64 * 0.2
        }),
        empty_sphere_radii: Array1::from_shape_fn(empty_count, |sphere| 0.2 + sphere as f64 * 0.02),
        empty_sphere_radius_indices: Array1::from_shape_fn(empty_count, |sphere| 25 + sphere),
        empty_sphere_density: Array2::from_shape_fn(
            (radial_count, empty_count),
            |(radial, sphere)| 0.0005 * (radial + 1) as f64 + 0.0002 * sphere as f64,
        ),
        empty_sphere_potential: Array2::from_shape_fn(
            (radial_count, empty_count),
            |(radial, sphere)| -0.5 - 0.006 * radial as f64 - 0.025 * sphere as f64,
        ),
        interstitial_potential: -0.75,
        homo_energy: -0.12,
        lumo_energy: 0.34,
    }
}

fn pot_bin_bench_data() -> PotBinData {
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
    }
}

fn pot_bin_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

criterion_group!(
    benches,
    bench_parse,
    bench_rdinp_outputs,
    bench_potential_outputs,
    bench_mtdp,
    bench_pot_bin
);
criterion_main!(benches);
