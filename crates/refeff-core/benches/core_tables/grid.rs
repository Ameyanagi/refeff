use super::*;

pub(super) fn bench_grid_helpers(c: &mut Criterion) {
    let mut large = vec![0.0; 251];
    let mut small = vec![0.0; 251];
    for i in 1..=80 {
        let i_real = i as f64;
        large[i - 1] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
        small[i - 1] = (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
    }
    let large = Array1::from_vec(large);
    let small = Array1::from_vec(small);

    c.bench_function("grid_fix_dirac_spinor_251_to_180", |b| {
        b.iter(|| {
            black_box(fix_dirac_spinor_grid(black_box(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: large.view(),
                small_component: small.view(),
                output_len: 180,
            })))
        });
    });

    let mut orbital_large = Array2::<f64>::zeros((251, 4).f());
    let mut orbital_small = Array2::<f64>::zeros((251, 4).f());
    for i in 1..=80 {
        let i_real = i as f64;
        orbital_large[(i - 1, 0)] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
        orbital_small[(i - 1, 0)] =
            (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
        orbital_large[(i - 1, 2)] = 0.2 * (0.11 * i_real).sin() + 0.002 * i_real;
        orbital_small[(i - 1, 2)] = 0.3 * (0.09 * i_real).cos() - 0.001 * i_real;
    }
    c.bench_function("grid_fix_dirac_spinor_orbitals_251x4", |b| {
        b.iter(|| {
            black_box(fix_dirac_spinor_orbitals_grid(black_box(
                DiracSpinorOrbitalsGridInput {
                    original_delta: 0.05,
                    new_delta: 0.025,
                    large_components: orbital_large.view(),
                    small_components: orbital_small.view(),
                    output_len: 260,
                },
            )))
        });
    });

    let source_len = 251;
    let density = (1..=source_len)
        .map(|index| {
            let i = index as f64;
            0.4 + 0.002 * i + 0.03 * (0.04 * i).sin()
        })
        .collect::<Array1<_>>();
    let potential = (1..=source_len)
        .map(|index| {
            let i = index as f64;
            -2.0 + 0.015 * i + 0.05 * (0.03 * i).cos()
        })
        .collect::<Array1<_>>();
    let magnetization = (1..=source_len)
        .map(|index| {
            let i = index as f64;
            0.01 * (0.08 * i).sin() - 0.0001 * i
        })
        .collect::<Array1<_>>();

    c.bench_function("grid_fix_potential_251_to_180", |b| {
        b.iter(|| {
            black_box(fix_potential_grid(black_box(PotentialGridInput {
                muffin_tin_radius: (-8.8 + 60.4 * 0.05_f64).exp(),
                electron_density: density.view(),
                total_potential: potential.view(),
                magnetization: magnetization.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 1,
                potential_jump: 0.125,
                output_len: 180,
            })))
        });
    });

    let atom_radii = (1..=source_len)
        .map(|index| {
            (-8.85 + 0.051 * (index - 1) as f64 + 1.0e-4 * (0.37 * index as f64).cos()).exp()
        })
        .collect::<Array1<_>>();
    let atom_coulomb = (1..=source_len)
        .map(|index| 0.2 + 0.01 * index as f64 + (0.03 * index as f64).sin())
        .collect::<Array1<_>>();
    let atom_density = (1..=source_len)
        .map(|index| 0.1 * index as f64 + 0.25 * (0.02 * index as f64).cos())
        .collect::<Array1<_>>();
    let atom_magnetization = (1..=source_len)
        .map(|index| -0.04 * index as f64 + 0.1 * (0.05 * index as f64).sin())
        .collect::<Array1<_>>();
    let atom_valence = (1..=source_len)
        .map(|index| 0.05 * (index as f64).sqrt() + 0.002 * (index % 5) as f64)
        .collect::<Array1<_>>();
    let atom_initial_large = (1..=source_len)
        .map(|index| 0.003 * index as f64 + 1.0e-5 * (index * index) as f64)
        .collect::<Array1<_>>();
    let atom_initial_small = (1..=source_len)
        .map(|index| -0.002 * index as f64 + 2.0e-6 * (index * index) as f64)
        .collect::<Array1<_>>();
    let atom_large = Array2::from_shape_fn((source_len, 41).f(), |(row, col)| {
        let i = (row + 1) as f64;
        let j = (col + 1) as f64;
        0.001 * i * j + 0.02 * (0.01 * (i + j)).sin()
    });
    let atom_small = Array2::from_shape_fn((source_len, 41).f(), |(row, col)| {
        let i = (row + 1) as f64;
        let j = (col + 1) as f64;
        -0.0007 * i * j + 0.015 * (0.012 * (i + 2.0 * j)).cos()
    });
    c.bench_function("grid_atom_fix_quantities_scfdat_251x41", |b| {
        b.iter(|| {
            black_box(fix_atomic_quantities_grid(black_box(
                AtomicQuantitiesGridInput {
                    source_radii: atom_radii.view(),
                    coulomb_potential: atom_coulomb.view(),
                    charge_density: atom_density.view(),
                    magnetization: atom_magnetization.view(),
                    valence_density: atom_valence.view(),
                    initial_large_component: atom_initial_large.view(),
                    initial_small_component: atom_initial_small.view(),
                    large_components: atom_large.view(),
                    small_components: atom_small.view(),
                    output_len: source_len,
                },
            )))
        });
    });

    let coulomb_radii = (1..=source_len)
        .map(|index| (-8.8 + 0.05 * (index - 1) as f64).exp())
        .collect::<Array1<_>>();
    let coulomb_density = (1..=source_len)
        .map(|index| {
            let radius = coulomb_radii[index - 1];
            (0.015 * index as f64 + 0.002 * (index % 5) as f64) * radius * radius
        })
        .collect::<Array1<_>>();
    c.bench_function("grid_coulomb_potslw_251", |b| {
        b.iter(|| {
            black_box(coulomb_potential_slw(black_box(CoulombPotentialSlwInput {
                density: coulomb_density.view(),
                radii: coulomb_radii.view(),
                delta: 0.05,
                active_len: source_len,
            })))
        });
    });

    c.bench_function("grid_scmt_energy_80x17", |b| {
        b.iter(|| {
            black_box(scmt_energy_grid(black_box(ScmtEnergyGridInput {
                core_valence_energy: -0.5,
                fermi_energy: 0.2,
                max_points: 80,
                step_count: 17,
            })))
        });
    });

    let overlap_source = (1..=250)
        .map(|index| {
            let i = index as f64;
            0.2 + 0.004 * i + 0.03 * (0.035 * i).sin()
        })
        .collect::<Array1<_>>();
    let overlap_base = (1..=250)
        .map(|index| {
            let i = index as f64;
            0.01 * (0.027 * i).cos()
        })
        .collect::<Array1<_>>();
    c.bench_function("grid_sum_loucks_overlap_250", |b| {
        b.iter(|| {
            black_box(sum_loucks_spherical_overlap(black_box(
                LoucksSphericalOverlapInput {
                    neighbor_distance: 2.35,
                    multiplicity: 1.75,
                    source: overlap_source.view(),
                    accumulated: overlap_base.view(),
                },
            )))
        });
    });
    c.bench_function("grid_sphere_overlap_lens_volume", |b| {
        b.iter(|| {
            black_box(sphere_overlap_lens_volume(
                black_box(2.40),
                black_box(1.70),
                black_box(2.15),
            ))
        });
    });
    let movrlp_atom_potentials = Array1::from_vec(vec![0, 1]);
    let movrlp_atom_positions = Array2::<f64>::zeros((2, 3));
    let movrlp_representatives = Array1::from_vec(vec![0, 1]);
    let movrlp_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let movrlp_neighbors0 = [MuffinTinOverlapNeighbor {
        source_potential: 1,
        multiplicity: 2,
        distance: 0.030,
    }];
    let movrlp_neighbors1 = [MuffinTinOverlapNeighbor {
        source_potential: 0,
        multiplicity: 1,
        distance: 0.031,
    }];
    let movrlp_explicit: [&[MuffinTinOverlapNeighbor]; 2] =
        [&movrlp_neighbors0, &movrlp_neighbors1];
    let movrlp_imt = Array1::from_vec(vec![95, 100]);
    let movrlp_inrm = Array1::from_vec(vec![90, 92]);
    let movrlp_rmt = Array1::from_vec(vec![0.020, 0.024]);
    let movrlp_rnrm = Array1::from_vec(vec![0.015, 0.018]);
    let movrlp_lnear = Array1::from_vec(vec![false, false]);
    let movrlp_input = MuffinTinOverlapMatrixInput {
        highest_potential_index: 1,
        atom_potentials: movrlp_atom_potentials.view(),
        atom_positions: movrlp_atom_positions.view(),
        representative_atoms: movrlp_representatives.view(),
        potential_multiplicities: movrlp_multiplicities.view(),
        explicit_overlaps: &movrlp_explicit,
        muffin_tin_indices: movrlp_imt.view(),
        muffin_tin_radii: movrlp_rmt.view(),
        norman_radii: movrlp_rnrm.view(),
        near_neighbor_flags: movrlp_lnear.view(),
        interstitial_selector: 0,
        interstitial_volume: 12.5,
    };
    c.bench_function("grid_movrlp_overlap_matrix_2pot", |b| {
        b.iter(|| black_box(muffin_tin_overlap_matrix(black_box(movrlp_input))));
    });
    let movrlp_overlap = match muffin_tin_overlap_matrix(movrlp_input) {
        Ok(overlap) => overlap,
        Err(error) => {
            eprintln!("skipping ovp2mt projection bench: {error}");
            return;
        }
    };
    let ovp2mt_values = Array2::from_shape_fn((251, 2), |(radial, potential)| {
        let index = (radial + 1) as f64;
        0.1 * (potential + 1) as f64
            + 0.001 * index
            + 0.00001 * index * index
            + 0.02 * movrlp_overlap.radii[radial]
    });
    c.bench_function("grid_ovp2mt_project_potential_2pot", |b| {
        b.iter(|| {
            black_box(project_muffin_tin_overlap(black_box(
                MuffinTinOverlapProjectionInput {
                    highest_potential_index: 1,
                    values: ovp2mt_values.view(),
                    radii: movrlp_overlap.radii.view(),
                    potential_multiplicities: movrlp_multiplicities.view(),
                    norman_indices: movrlp_inrm.view(),
                    muffin_tin_indices: movrlp_imt.view(),
                    muffin_tin_radii: movrlp_rmt.view(),
                    norman_radii: movrlp_rnrm.view(),
                    near_neighbor_flags: movrlp_lnear.view(),
                    overlap_matrix: &movrlp_overlap,
                    interstitial_selector: 0,
                    interstitial_value: 0.0,
                    mode: MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial,
                },
            )))
        });
    });

    let shell_len = 1251;
    let shell_potential = (1..=shell_len)
        .map(|index| {
            let i = index as f64;
            -1.5 + 0.002 * i + 0.04 * (0.017 * i).cos()
        })
        .collect::<Array1<_>>();
    let shell_density = (1..=shell_len)
        .map(|index| {
            let i = index as f64;
            0.5 + 0.003 * i + 0.02 * (0.023 * i).sin()
        })
        .collect::<Array1<_>>();
    c.bench_function("grid_interstitial_shell_values_1251", |b| {
        b.iter(|| {
            black_box(interstitial_shell_values(black_box(
                InterstitialShellValuesInput {
                    total_potential: shell_potential.view(),
                    overlapped_density: shell_density.view(),
                    muffin_tin_radius: (-8.8 + 44.0 * 0.05_f64 + 0.021).exp(),
                    muffin_tin_index: 45,
                    wigner_seitz_radius: (-8.8 + 115.0 * 0.05_f64 + 0.034).exp(),
                    wigner_seitz_index: 116,
                },
            )))
        });
    });

    let sidx_density = (1..=250)
        .map(|index| {
            let i = index as f64;
            if index <= 92 {
                0.04 + 0.0002 * i
            } else {
                1.0e-6
            }
        })
        .collect::<Array1<_>>();
    c.bench_function("grid_overlap_density_indices_250", |b| {
        b.iter(|| {
            black_box(overlap_density_indices(black_box(
                OverlapDensityIndicesInput {
                    overlapped_density: sidx_density.view(),
                    muffin_tin_radius: (0.05_f32 as f64 * 29.0 - 8.8_f32 as f64 + 0.020).exp(),
                    norman_radius: (0.05_f32 as f64 * 129.0 - 8.8_f32 as f64 + 0.010).exp(),
                },
            )))
        });
    });

    let frnrm_density = (1..=251)
        .map(|index| {
            let radius = (0.05_f32 as f64 * (index as f64 - 1.0) - 8.8_f32 as f64).exp();
            220.0 * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
        })
        .collect::<Array1<_>>();
    c.bench_function("grid_norman_radius_frnrm_251", |b| {
        b.iter(|| {
            black_box(norman_radius_from_density(black_box(NormanRadiusInput {
                overlapped_density: frnrm_density.view(),
                atomic_number: 26,
            })))
        });
    });

    c.bench_function("grid_interstitial_fermi_level", |b| {
        b.iter(|| {
            black_box(interstitial_fermi_level(black_box(FermiLevelInput {
                interstitial_density: 8.430_358_921_763_391e-1,
                interstitial_potential: -1.294_131_834_592_241_2,
            })))
        });
    });
}
