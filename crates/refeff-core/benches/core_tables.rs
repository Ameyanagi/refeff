use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3, Array4, Array6, ShapeBuilder, arr2, array};
use num_complex::Complex32;
use refeff_core::{
    BasisTransformMode, BravaisLattice, BroydenMixInput, BroydenWorkspace, Complex,
    CoulombPotentialSlwInput, CoulombPotentialUpdateInput, CoulombUpdateMode,
    CurvedWavePolynomialInput, DiracSpinorGridInput, DiracSpinorOrbitalsGridInput, EelsMeshInput,
    EelsMeshMode, EnergyIndependentMatrixInput, FermiLevelInput, FmsAtom, FmsBiCgStabInput,
    FmsFreePropagatorInput, FmsFreePropagatorMatrixInput, FmsFullPotentialLuInput,
    FmsGravesMorrisInput, FmsIterativeSystemInput, FmsLuInput, FmsRecursionInput,
    FmsRotationDirection, FmsTMatrixInput, FmsTMatrixTableInput, FmsTfqmrInput,
    GenfmtLegendreNormalizationInput, HydrogenBondAdjustmentInput, InitialStateRotationInput,
    InterstitialShellValuesInput, LambdaIndexInput, LoucksSphericalOverlapInput,
    MuffinTinOverlapMatrixInput, MuffinTinOverlapNeighbor, MuffinTinOverlapProjectionInput,
    MuffinTinOverlapProjectionMode, NormanRadiusInput, OverlapDensityIndicesInput,
    PathCanonicalRepresentationInput, PathCriteriaDecisionInput, PathOutputCriterionInput,
    PathOutputImportanceInput, PathPhaseCriteriaInput, PathRotationInput,
    PathStandardCoordinatesInput, PolarizationTensorMode, PolarizedScatteringAmplitudeInput,
    PotentialGridInput, PotentialOverlapInput, PotentialOverlapNeighbor,
    ScatteringAmplitudeMatrixInput, ScmtEnergyGridInput, SelfEnergyIntegrandInput,
    SingularityFunction, StateKet, TransitionBMatrixInput, TransitionRotationInput,
    ValenceDensityUpdateInput, XStarInput, adjust_hydrogen_bonds, basis_transform_matrices, besjh,
    besjn, bilinear_interpolate_complex, bracket_table_minimum, brent_table_minimum, cgratr,
    change_basis_representation, change_cartesian_basis, classical_debye_correlation,
    construct_state_kets, conv, coulomb_potential_slw, cubic_zeros, curved_wave_polynomials,
    define_k_path, depressed_quartic_roots, dirac_hara_exchange_potential, distance_between,
    eels_euler_rotation_matrix, eels_integration_mesh, electron_wavelength_atomic_units,
    energy_independent_transition_matrix, exjlnl, find_self_energy_singularities,
    fix_dirac_spinor_grid, fix_dirac_spinor_orbitals_grid, fix_potential_grid,
    fms_bicgstab_scattering, fms_free_propagator_element, fms_free_propagator_matrix,
    fms_full_potential_lu_scattering, fms_graves_morris_scattering, fms_iterative_system_matrix,
    fms_lu_scattering, fms_pair_tables, fms_recursion_scattering, fms_rotation_matrix,
    fms_t_matrix_element, fms_t_matrix_table, fms_tfqmr_scattering, gamma_q,
    gauss_legendre_quadrature, genfmt_legendre_normalization_table, hartree_fock_exchange,
    hedin_lundqvist_ffq, hedin_lundqvist_imaginary_self_energy, hedin_lundqvist_self_energy,
    initial_state_rotation, integrated_double_lorentz, interpolation_polynomial_coefficients,
    interstitial_fermi_level, interstitial_shell_values, karasiev_sjostrom_dufty_trickey_vxc,
    kk_integral, kmesh_arbitrary_mesh, kmesh_basis_divisions, kmesh_bravais_basis,
    kmesh_tetrahedron_division, kmesh_tetrahedron_records, lambda_indices,
    legendre_normalization_table, legendre_polynomials, lint, log_i, make_excitation_poles,
    mix_broyden_density, morse_einstein_cumulants, muffin_tin_overlap_matrix,
    muffin_tin_phase_amplitude, norman_radius_from_density, nuclear_mass, omega_q,
    overlap_density_indices, overlap_potential_density, pack_path_indices, pair_polar_angles,
    path_canonical_representation, path_criteria_decision, path_degeneracy_hash, path_geometry,
    path_heap_bubble_down, path_heap_bubble_up, path_heap_criterion, path_output_criterion,
    path_output_importance, path_output_parameters, path_phase_criteria_tables,
    path_rotation_angles, path_standard_coordinates, perdew_zunger_vxc, perrot_dharma_wardana_vxc,
    point_group_operations, polarization_tensor, polarized_scattering_amplitude_matrix,
    project_muffin_tin_overlap, qsortd_order_1based, quadratic_zeros, quantum_debye_correlation,
    quantum_debye_waller_factor, quinn_imaginary_self_energy, real_polynomial_roots,
    reciprocal_lattice_vectors, reciprocal_metric, redefine_lattice_symmetry_operations,
    reduce_kmesh_common_divisor, reduce_kmesh_irreducible_points, reduce_to_lattice_cell,
    rehr_albers_polynomials, rehr_albers_z_axis_propagator,
    relativistic_clebsch_gordan_coefficients, scattering_amplitude_matrix, scmt_energy_grid,
    self_energy_r1_integrand, somm2, sort_atoms_by_radius, sort_representative_atoms,
    sortid_order_1based, sortii_order_1based, sortir_order_1based, sphere_overlap_lens_volume,
    spherical_harmonics, spin_orbit_coupling_tables, subtract_lattice_translation,
    sum_loucks_spherical_overlap, symmetry_check, terp, terpc, thermal_expansion_cumulants,
    transform_lapw_symmetry_operations, transition_b_matrix, trap, unpack_path_indices,
    update_coulomb_potential, update_valence_density, von_barth_hedin_potential, wigner_rotation,
    x_log_x, xstar,
};

fn bench_angular_tables(c: &mut Criterion) {
    c.bench_function("build_legendre_xnlm_lmax8", |b| {
        b.iter(|| black_box(legendre_normalization_table(black_box(8))));
    });
    c.bench_function("build_spin_orbit_tables_lmax8", |b| {
        b.iter(|| black_box(spin_orbit_coupling_tables(black_box(8))));
    });
    c.bench_function("build_relativistic_cgc_lmax8", |b| {
        b.iter(|| black_box(relativistic_clebsch_gordan_coefficients(black_box(8))));
    });
    c.bench_function("build_basis_transform_lmax4", |b| {
        b.iter(|| black_box(basis_transform_matrices(black_box(4))));
    });
    let Ok(basis_transforms) = basis_transform_matrices(3) else {
        return;
    };
    let basis_input = Array2::from_shape_fn(
        (basis_transforms.order, basis_transforms.order).f(),
        |(row, column)| {
            Complex::new(
                0.01 * (row as f64 + 1.0) + 0.003 * (column as f64 + 1.0),
                -0.002 * (row as f64 + 1.0) + 0.007 * (column as f64 + 1.0),
            )
        },
    );
    c.bench_function("change_basis_representation_lmax3_rel_to_real", |b| {
        b.iter(|| {
            black_box(change_basis_representation(
                black_box(basis_input.view()),
                black_box(BasisTransformMode::RelativisticToReal),
                black_box(&basis_transforms),
            ))
        });
    });
    c.bench_function("build_legendre_polynomials_lmax32", |b| {
        b.iter(|| black_box(legendre_polynomials(black_box(0.25), black_box(32))));
    });
    c.bench_function("wigner_rotation_half_integer", |b| {
        b.iter(|| {
            black_box(wigner_rotation(
                black_box(0.7),
                black_box(3),
                black_box(1),
                black_box(-1),
                black_box(2),
            ))
        });
    });
    c.bench_function("spherical_harmonics_l8", |b| {
        b.iter(|| {
            black_box(spherical_harmonics(
                black_box([1.0, 2.0, 3.0]),
                black_box(8),
            ))
        });
    });
    c.bench_function("polarization_tensor_cartesian", |b| {
        b.iter(|| {
            black_box(polarization_tensor(
                black_box(5),
                black_box(PolarizationTensorMode::Cartesian),
            ))
        });
    });
    c.bench_function("transition_b_matrix_l3", |b| {
        b.iter(|| {
            black_box(transition_b_matrix(black_box(TransitionBMatrixInput {
                lmax: 3,
                initial_kappa: -1,
                polarization: 1,
                polarization_tensor: sample_polarization_tensor(),
                multipole: 2,
                trace_orbital: false,
                spin: 1,
                spin_channels: 1,
                spin_vector_angle: 0.3,
            })))
        });
    });
}

fn sample_polarization_tensor() -> [[Complex; 3]; 3] {
    [
        [
            Complex::new(0.20, -0.05),
            Complex::new(-0.10, 0.04),
            Complex::new(0.03, 0.02),
        ],
        [
            Complex::new(0.11, -0.07),
            Complex::new(0.50, 0.00),
            Complex::new(-0.08, 0.09),
        ],
        [
            Complex::new(0.06, 0.01),
            Complex::new(0.13, -0.02),
            Complex::new(0.17, 0.03),
        ],
    ]
}

fn bench_state_kets(c: &mut Criterion) {
    let atom_potentials = vec![0, 1, 1, 2, 2, 2, 1, 0, 3, 3, 2, 1];
    let potential_lmax = vec![0, 2, 3, 1];

    c.bench_function("construct_state_kets_small_cluster", |b| {
        b.iter(|| {
            black_box(construct_state_kets(
                black_box(2),
                black_box(&atom_potentials),
                black_box(&potential_lmax),
                black_box(3),
            ))
        });
    });
}

fn bench_kspace_helpers(c: &mut Criterion) {
    let basis = [[1.1, 0.2, 0.05], [-0.1, 1.3, 0.04], [0.03, 0.2, 0.9]];
    c.bench_function("define_k_path_fcc_default", |b| {
        b.iter(|| {
            black_box(define_k_path(
                black_box(BravaisLattice::CubicFaceCentered),
                black_box(0),
                black_box(basis),
            ))
        });
    });
    c.bench_function("define_k_path_orthorhombic_full", |b| {
        b.iter(|| {
            black_box(define_k_path(
                black_box(BravaisLattice::OrthorhombicPrimitive),
                black_box(1),
                black_box(basis),
            ))
        });
    });

    let direct = arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
    let reciprocal = arr2(&[
        [std::f64::consts::TAU / 2.0, 0.0, 0.0],
        [0.0, std::f64::consts::TAU / 3.0, 0.0],
        [0.0, 0.0, std::f64::consts::TAU / 4.0],
    ]);
    let operation = arr2(&[[1, -2, 0], [3, 0, 1], [-1, 2, 1]]);
    let vector = [3.2, -1.55, 8.2];
    let skew_lattice = arr2(&[[2.0, 0.3, -0.2], [0.1, 3.0, 0.5], [0.2, 0.4, 4.0]]);
    c.bench_function("reciprocal_lattice_vectors_skew_3x3", |b| {
        b.iter(|| black_box(reciprocal_lattice_vectors(black_box(skew_lattice.view()))));
    });
    let bravais_right_angle = 1_570_796.0 / 1_000_000.0;
    c.bench_function("kmesh_bravais_basis_cxz", |b| {
        b.iter(|| {
            black_box(kmesh_bravais_basis(
                black_box("CXZ"),
                black_box([2.0, 3.0, 4.0]),
                black_box([bravais_right_angle; 3]),
            ))
        });
    });
    let Ok(skew_reciprocal) = reciprocal_lattice_vectors(skew_lattice.view()) else {
        return;
    };
    c.bench_function("kmesh_basis_divisions_skew_120", |b| {
        b.iter(|| {
            black_box(kmesh_basis_divisions(
                black_box(skew_reciprocal.view()),
                black_box(120),
                black_box([false, false, false]),
            ))
        });
    });
    let tetdiv_reciprocal = arr2(&[[2.0, 0.5, 0.0], [0.0, 3.0, 0.25], [0.1, 0.0, 4.0]]);
    c.bench_function("kmesh_tetrahedron_division_skew", |b| {
        b.iter(|| {
            black_box(kmesh_tetrahedron_division(
                black_box([2, 3, 4]),
                black_box(tetdiv_reciprocal.view()),
            ))
        });
    });
    let Ok(tetdiv_offsets) = kmesh_tetrahedron_division([2, 3, 4], tetdiv_reciprocal.view()) else {
        return;
    };
    let tetdiv_links = (1..=60).collect::<Vec<_>>();
    c.bench_function("kmesh_tetrahedron_records_2x3x4_identity", |b| {
        b.iter(|| {
            black_box(kmesh_tetrahedron_records(
                black_box(tetdiv_offsets.view()),
                black_box([2, 3, 4]),
                black_box(&tetdiv_links),
                black_box(60),
            ))
        });
    });
    let reduz_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ];
    let reduz_reciprocal = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    c.bench_function("reduce_kmesh_irreducible_points_2x1x1_sign", |b| {
        b.iter(|| {
            black_box(reduce_kmesh_irreducible_points(
                black_box([2, 1, 1]),
                black_box(reduz_operations.view()),
                black_box(reduz_reciprocal.view()),
            ))
        });
    });
    c.bench_function("kmesh_arbitrary_mesh_4_sign_tetrahedra", |b| {
        b.iter(|| {
            black_box(kmesh_arbitrary_mesh(
                black_box(tetdiv_reciprocal.view()),
                black_box(reduz_operations.view()),
                black_box(4),
                black_box([false, false, false]),
                black_box(true),
            ))
        });
    });
    let klist = arr2(&[[6, 12, 18], [24, 30, 36]]);
    c.bench_function("reduce_kmesh_common_divisor_2x3", |b| {
        b.iter(|| {
            black_box(reduce_kmesh_common_divisor(
                black_box(klist.view()),
                black_box(12),
            ))
        });
    });
    let sdef_operations = array![
        [[111, 112, 113], [121, 122, 123], [131, 132, 133]],
        [[211, 212, 213], [221, 222, 223], [231, 232, 233]]
    ];
    c.bench_function("redefine_lattice_symmetry_cxz_2", |b| {
        b.iter(|| {
            black_box(redefine_lattice_symmetry_operations(
                black_box(sdef_operations.view()),
                black_box("CXZ"),
            ))
        });
    });
    let sdefl_direct = arr2(&[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let Ok(sdefl_reciprocal) = reciprocal_lattice_vectors(sdefl_direct.view()) else {
        return;
    };
    let sdefl_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ];
    c.bench_function("transform_lapw_symmetry_shear_2", |b| {
        b.iter(|| {
            black_box(transform_lapw_symmetry_operations(
                black_box(sdefl_direct.view()),
                black_box(sdefl_reciprocal.view()),
                black_box(sdefl_operations.view()),
                black_box("P  "),
                black_box(true),
            ))
        });
    });
    c.bench_function("subtract_lattice_translation_3d", |b| {
        b.iter(|| {
            black_box(subtract_lattice_translation(
                black_box(reciprocal.view()),
                vector,
            ))
        });
    });
    c.bench_function("reduce_to_lattice_cell_3d", |b| {
        b.iter(|| {
            black_box(reduce_to_lattice_cell(
                black_box(direct.view()),
                black_box(reciprocal.view()),
                black_box(vector),
            ))
        });
    });
    c.bench_function("change_cartesian_basis_3x3", |b| {
        b.iter(|| {
            black_box(change_cartesian_basis(
                black_box(reciprocal.view()),
                black_box(direct.view()),
                black_box(operation.view()),
            ))
        });
    });

    let cubic = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let Ok(cubic_metric) = reciprocal_metric(cubic.view()) else {
        return;
    };
    c.bench_function("point_group_cubic_48", |b| {
        b.iter(|| {
            black_box(point_group_operations(
                black_box(cubic.view()),
                black_box(cubic_metric.view()),
                black_box(64),
            ))
        });
    });

    let sign_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ];
    let sign_translations = Array2::<f64>::zeros((4, 3));
    c.bench_function("symmetry_check_sign_group_4", |b| {
        b.iter(|| {
            black_box(symmetry_check(
                black_box(sign_operations.view()),
                black_box(sign_translations.view()),
            ))
        });
    });
}

fn bench_density_helpers(c: &mut Criterion) {
    let l_count = 4;
    let potential_count = 3;
    let radial_count = 251;
    let scattering_trace = (0..l_count)
        .map(|angular| {
            let l = angular as f64;
            Complex32::new(
                ((0.05_f32 as f64) * l + 0.11_f32 as f64) as f32,
                ((-0.03_f32 as f64) * l + 0.07_f32 as f64) as f32,
            )
        })
        .collect::<Array1<_>>();
    let scattering_ldos = (0..l_count)
        .map(|angular| {
            let l = angular as f64;
            Complex::new(0.2 + 0.04 * l, -0.13 + 0.02 * l)
        })
        .collect::<Array1<_>>();
    let embedded_ldos =
        Array2::from_shape_fn((l_count, potential_count), |(angular, potential)| {
            let l = angular as f64;
            let p = potential as f64;
            Complex::new(0.4 + 0.03 * l + 0.02 * p, -0.2 + 0.01 * l - 0.015 * p)
        });
    let previous_ldos =
        Array2::from_shape_fn((l_count, potential_count), |(angular, potential)| {
            let l = angular as f64;
            let p = potential as f64;
            Complex::new(-0.1 + 0.025 * l + 0.01 * p, 0.08 - 0.02 * l + 0.005 * p)
        });
    let scattering_density = Array2::from_shape_fn((radial_count, l_count), |(radial, angular)| {
        let r = (radial + 1) as f64;
        let l = angular as f64;
        Complex::new(0.006 * r + 0.02 * l, -0.004 * r + 0.015 * l)
    });
    let embedded_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as f64;
            Complex::new(0.05 * r, -0.02 * r)
        })
        .collect::<Array1<_>>();
    let previous_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as f64;
            Complex::new(-0.03 * r, 0.04 * r)
        })
        .collect::<Array1<_>>();
    let valence_density = (1..=radial_count)
        .map(|radial| 0.01 * radial as f64)
        .collect::<Array1<_>>();
    let occupancy_by_l = (0..l_count)
        .map(|angular| -0.03 + 0.015 * angular as f64)
        .collect::<Array1<_>>();

    c.bench_function("density_update_ff2g_251_l4", |b| {
        b.iter(|| {
            black_box(update_valence_density(black_box(
                ValenceDensityUpdateInput {
                    scattering_trace: scattering_trace.view(),
                    potential_index: 1,
                    energy_index: 1,
                    last_radial_index: radial_count,
                    scattering_ldos: scattering_ldos.view(),
                    embedded_ldos: embedded_ldos.view(),
                    previous_ldos: previous_ldos.view(),
                    scattering_density: scattering_density.view(),
                    embedded_density: embedded_density.view(),
                    previous_density: previous_density.view(),
                    valence_density: valence_density.view(),
                    occupancy_by_l: occupancy_by_l.view(),
                    current_energy: Complex::new(0.72, 0.11),
                    previous_energy: Complex::new(0.61, -0.04),
                    potential_multiplicity: 2.5,
                    current_floor: 1,
                    previous_floor: 0,
                    left_sum: Complex::new(0.2, -0.1),
                    right_sum: Complex::new(-0.3, 0.25),
                    total_electron_count: 1.25,
                    include_high_l: false,
                },
            )))
        });
    });

    let atom_potentials = Array1::from_vec(vec![0, 1, 2, 1]);
    let atom_positions = arr2(&[
        [0.0, 0.0, 0.0],
        [1.35, 0.2, -0.15],
        [3.10, -0.4, 0.25],
        [13.5, 0.0, 0.0],
    ]);
    let representative_atoms = Array1::from_vec(vec![0, 1, 2]);
    let atomic_numbers = Array1::from_vec(vec![6, 8, 14]);
    let explicit_overlaps = [
        PotentialOverlapNeighbor {
            source_potential: 0,
            multiplicity: 2.0,
            distance: 1.6,
        },
        PotentialOverlapNeighbor {
            source_potential: 2,
            multiplicity: 1.0,
            distance: 2.4,
        },
    ];
    let electron_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            let radius = ((0.05_f32 as f64) * radial as f64 - 8.8_f32 as f64).exp();
            let i = (radial + 1) as f64;
            let p = potential as f64;
            (45.0 + 18.0 * p) * (-(1.0 + 0.08 * p) * radius).exp() + 0.05 * (i + p)
        });
    let spin_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            0.02 + 0.0003 * (radial + 1) as f64 + 0.005 * potential as f64
        });
    let valence_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            let density = electron_density[(radial, potential)];
            0.65 * density + 0.01 * potential as f64 + 0.0002 * (radial + 1) as f64
        });
    let coulomb_potential =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            let i = (radial + 1) as f64;
            let p = potential as f64;
            -2.0 - 0.12 * p + 0.004 * i + 0.03 * (0.05 * i + p).cos()
        });

    c.bench_function("density_overlap_ovrlp_251_explicit", |b| {
        b.iter(|| {
            black_box(overlap_potential_density(black_box(
                PotentialOverlapInput {
                    potential_index: 1,
                    atom_potentials: atom_potentials.view(),
                    atom_positions: atom_positions.view(),
                    representative_atoms: representative_atoms.view(),
                    atomic_numbers: atomic_numbers.view(),
                    explicit_overlaps: &explicit_overlaps,
                    electron_density: electron_density.view(),
                    spin_density: spin_density.view(),
                    valence_density: valence_density.view(),
                    coulomb_potential: coulomb_potential.view(),
                },
            )))
        });
    });

    let last_indices = Array1::from_vec(vec![140, 132]);
    let coulom_atom_potentials = Array1::from_vec(vec![0, 1, 1]);
    let coulom_representatives = Array1::from_vec(vec![0, 1]);
    let coulom_atomic_numbers = Array1::from_vec(vec![8, 14]);
    let coulom_norman_radii = Array1::from_vec(vec![0.65, 0.82]);
    let coulom_charge_deltas = Array1::from_vec(vec![0.15, -0.07]);
    let coulom_atom_positions = arr2(&[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.1, 0.0]]);
    let coulom_density = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        let radius = (-8.8 + 0.05 * radial as f64).exp();
        (80.0 + 15.0 * potential as f64) * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
    });
    let coulom_edenvl = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        (0.42 + 0.03 * potential as f64) * coulom_density[(radial, potential)]
    });
    let coulom_rhoval = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        (0.36 + 0.02 * potential as f64) * coulom_density[(radial, potential)]
    });
    let coulom_vclap = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        -1.7 - 0.25 * potential as f64 + 0.004 * (radial + 1) as f64
    });
    c.bench_function("density_coulom_update_251x2", |b| {
        b.iter(|| {
            black_box(update_coulomb_potential(black_box(
                CoulombPotentialUpdateInput {
                    mode: CoulombUpdateMode::Norman,
                    highest_potential_index: 1,
                    last_indices: last_indices.view(),
                    valence_density: coulom_rhoval.view(),
                    overlapped_valence_density: coulom_edenvl.view(),
                    overlapped_density: coulom_density.view(),
                    atom_positions: coulom_atom_positions.view(),
                    representative_atoms: coulom_representatives.view(),
                    atom_potentials: coulom_atom_potentials.view(),
                    norman_radii: coulom_norman_radii.view(),
                    charge_deltas: coulom_charge_deltas.view(),
                    atomic_numbers: coulom_atomic_numbers.view(),
                    coulomb_potential: coulom_vclap.view(),
                },
            )))
        });
    });

    let broydn_last_indices = Array1::from_vec(vec![190, 196]);
    let broydn_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let broydn_norman_radii = Array1::from_vec(vec![0.72, 0.88]);
    let broydn_initial_charges = Array1::from_vec(vec![1.40, 2.10]);
    let mut broydn_occupancy = Array2::<f64>::zeros((3, 2));
    broydn_occupancy[(0, 0)] = 1.10;
    broydn_occupancy[(1, 0)] = 0.60;
    broydn_occupancy[(0, 1)] = 1.45;
    broydn_occupancy[(1, 1)] = 0.80;
    broydn_occupancy[(2, 1)] = 0.30;
    let broydn_edenvl = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        let radius = (-8.8 + 0.05 * radial as f64).exp();
        (45.0 + 8.0 * potential as f64) * (-0.92 * radius).exp() / (1.0 + 0.10 * radius)
    });
    let broydn_density_for_iteration = |iteration: usize| {
        Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
            let radius = (-8.8 + 0.05 * radial as f64).exp();
            broydn_edenvl[(radial, potential)]
                * (0.97 + 0.018 * iteration as f64 + 0.004 * potential as f64)
                + (0.015 * iteration as f64 + 0.003 * potential as f64) * (-0.35 * radius).exp()
        })
    };
    let broydn_rhoval1 = broydn_density_for_iteration(1);
    let broydn_rhoval2 = broydn_density_for_iteration(2);
    let broydn_rhoval3 = broydn_density_for_iteration(3);
    let broydn_workspace0 = BroydenWorkspace::zeros(4, 2);
    let broydn_iter2_setup = match mix_broyden_density(BroydenMixInput {
        iteration: 1,
        accelerator: 0.35,
        highest_potential_index: 1,
        valence_occupancy: broydn_occupancy.view(),
        last_indices: broydn_last_indices.view(),
        potential_multiplicities: broydn_multiplicities.view(),
        norman_radii: broydn_norman_radii.view(),
        norman_charges: broydn_initial_charges.view(),
        overlapped_valence_density: broydn_edenvl.view(),
        valence_density: broydn_rhoval1.view(),
        workspace: &broydn_workspace0,
    }) {
        Ok(first_mix) => mix_broyden_density(BroydenMixInput {
            iteration: 2,
            accelerator: 0.35,
            highest_potential_index: 1,
            valence_occupancy: broydn_occupancy.view(),
            last_indices: broydn_last_indices.view(),
            potential_multiplicities: broydn_multiplicities.view(),
            norman_radii: broydn_norman_radii.view(),
            norman_charges: first_mix.norman_charges.view(),
            overlapped_valence_density: broydn_edenvl.view(),
            valence_density: broydn_rhoval2.view(),
            workspace: &first_mix.workspace,
        }),
        Err(error) => Err(error),
    };
    if let Ok(second_mix) = broydn_iter2_setup {
        c.bench_function("density_broydn_mix_251x2_iter3", |b| {
            b.iter(|| {
                black_box(mix_broyden_density(black_box(BroydenMixInput {
                    iteration: 3,
                    accelerator: 0.35,
                    highest_potential_index: 1,
                    valence_occupancy: broydn_occupancy.view(),
                    last_indices: broydn_last_indices.view(),
                    potential_multiplicities: broydn_multiplicities.view(),
                    norman_radii: broydn_norman_radii.view(),
                    norman_charges: second_mix.norman_charges.view(),
                    overlapped_valence_density: broydn_edenvl.view(),
                    valence_density: broydn_rhoval3.view(),
                    workspace: &second_mix.workspace,
                })))
            });
        });
    }
}

fn bench_grid_helpers(c: &mut Criterion) {
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

    c.bench_function("grid_scmt_energy_120x9", |b| {
        b.iter(|| {
            black_box(scmt_energy_grid(black_box(ScmtEnergyGridInput {
                core_valence_energy: -0.5,
                fermi_energy: 0.2,
                max_points: 120,
                step_count: 9,
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

fn bench_genfmt_helpers(c: &mut Criterion) {
    let beta_angles = [0.0, 0.25, std::f64::consts::PI];
    c.bench_function("genfmt_lambda_indices_cute_high", |b| {
        b.iter(|| {
            black_box(lambda_indices(black_box(LambdaIndexInput {
                calculation: 10,
                energy_index: 42,
                scattering_count: 2,
                initial_l: 4,
                beta_angles: &beta_angles,
                lambda_capacity: 80,
                max_m: 10,
                max_n: 10,
            })))
        });
    });
    c.bench_function("genfmt_xstar_elliptic", |b| {
        b.iter(|| {
            black_box(xstar(black_box(XStarInput {
                primary_polarization: [0.3, 1.0, -0.2],
                secondary_polarization: [-0.4, 0.2, 1.5],
                first_leg: [1.2, -0.5, 0.8],
                last_leg: [-0.7, 1.4, 0.6],
                degeneracy: 2.25,
                initial_l: 2,
                ellipticity: 0.7,
            })))
        });
    });
    c.bench_function("genfmt_initial_state_rotation_l3", |b| {
        b.iter(|| {
            black_box(initial_state_rotation(black_box(
                InitialStateRotationInput {
                    lmaxp1: 4,
                    mmaxp1: 4,
                    beta_angle: 0.7,
                },
            )))
        });
    });
    let path_positions = arr2(&[
        [1.2, -0.4, 0.7],
        [-0.3, 1.1, 1.5],
        [0.5, 0.2, -0.6],
        [0.0, 0.0, 0.0],
    ]);
    c.bench_function("genfmt_path_rotation_angles_polarized", |b| {
        b.iter(|| {
            black_box(path_rotation_angles(black_box(PathRotationInput {
                positions: path_positions.view(),
                polarized: true,
            })))
        });
    });
    c.bench_function("genfmt_legendre_normalization_l16_m8", |b| {
        b.iter(|| {
            black_box(genfmt_legendre_normalization_table(black_box(
                GenfmtLegendreNormalizationInput {
                    lmaxp1: 17,
                    mmaxp1: 9,
                },
            )))
        });
    });
    c.bench_function("genfmt_curved_wave_polynomials_l4", |b| {
        b.iter(|| {
            black_box(curved_wave_polynomials(black_box(
                CurvedWavePolynomialInput {
                    lmaxp1: 5,
                    mmaxp1: 4,
                    rho: Complex::new(1.25, 0.4),
                },
            )))
        });
    });

    let Ok(scattering) = sample_scattering_amplitude_inputs() else {
        return;
    };
    c.bench_function("genfmt_scattering_amplitude_matrix_6x5", |b| {
        b.iter(|| black_box(scattering_amplitude_matrix(black_box(scattering.input()))));
    });

    let Ok(polarized) = sample_polarized_scattering_amplitude_inputs() else {
        return;
    };
    c.bench_function("genfmt_polarized_scattering_amplitude_matrix_6", |b| {
        b.iter(|| {
            black_box(polarized_scattering_amplitude_matrix(black_box(
                polarized.input(),
            )))
        });
    });

    let transition = sample_energy_independent_transition_inputs();
    c.bench_function("genfmt_energy_independent_transition_matrix", |b| {
        b.iter(|| {
            black_box(energy_independent_transition_matrix(black_box(
                transition.input(),
            )))
        });
    });
    c.bench_function("genfmt_energy_independent_transition_matrix_avg", |b| {
        b.iter(|| {
            black_box(energy_independent_transition_matrix(black_box(
                transition.unpolarized_input(),
            )))
        });
    });
}

struct SampleScatteringAmplitude {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    phase_shifts: Array1<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    rotation: Array3<f64>,
    xnlm: Array2<f64>,
}

impl SampleScatteringAmplitude {
    fn input(&self) -> ScatteringAmplitudeMatrixInput<'_> {
        ScatteringAmplitudeMatrixInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            left_lambda_count: 6,
            right_lambda_count: 5,
            phase_shifts: self.phase_shifts.view(),
            angular_limit: 3,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            rotation: self.rotation.view(),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta: 0.37,
        }
    }
}

fn sample_scattering_amplitude_inputs()
-> Result<SampleScatteringAmplitude, Box<dyn std::error::Error>> {
    let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let phase_shifts = Array1::from_iter((-4..=4).map(|l| {
        let l = l as f64;
        Complex::new(0.015 * l + 0.02, -0.01 * l + 0.03)
    }));
    let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(1.25, 0.4),
    })?;
    let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(-0.8, 1.1),
    })?;
    let mut rotation = Array3::zeros((5, 9, 9).f());
    for l in 0..=4 {
        let il = (l + 1) as f64;
        for m1 in -4_i32..=4 {
            for m2 in -4_i32..=4 {
                if (m1.unsigned_abs() as usize) <= l && (m2.unsigned_abs() as usize) <= l {
                    let row = (m1 + 4) as usize;
                    let column = (m2 + 4) as usize;
                    rotation[(l, row, column)] =
                        (0.11 * il + 0.07 * (m1 as f64) - 0.05 * (m2 as f64)).cos();
                }
            }
        }
    }

    Ok(SampleScatteringAmplitude {
        m_indices,
        n_indices,
        phase_shifts,
        first_polynomials,
        second_polynomials,
        rotation,
        xnlm: legendre_normalization_table(4)?,
    })
}

struct SamplePolarizedScatteringAmplitude {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    transition_angular_momenta: Array1<i32>,
    radial_factors: Array1<Complex>,
    transition_matrix: Array4<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    xnlm: Array2<f64>,
}

impl SamplePolarizedScatteringAmplitude {
    fn input(&self) -> PolarizedScatteringAmplitudeInput<'_> {
        PolarizedScatteringAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 6,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            transition_matrix: self.transition_matrix.view(),
            transition_magnetic_offset: 4,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            xnlm: self.xnlm.view(),
            eta: 0.37,
        }
    }
}

fn sample_polarized_scattering_amplitude_inputs()
-> Result<SamplePolarizedScatteringAmplitude, Box<dyn std::error::Error>> {
    let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
    let radial_factors = Array1::from_iter((1..=8).map(|k| {
        let k = k as f64;
        Complex::new(0.9 + 0.07 * k, -0.02 * k)
    }));
    let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(1.25, 0.4),
    })?;
    let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(-0.8, 1.1),
    })?;
    let mut transition_matrix = Array4::zeros((9, 8, 9, 8).f());
    for k2 in 1..=8 {
        for m2 in -4_i32..=4 {
            for k1 in 1..=8 {
                for m1 in -4_i32..=4 {
                    let first_m = (m1 + 4) as usize;
                    let second_m = (m2 + 4) as usize;
                    transition_matrix[(first_m, k1 - 1, second_m, k2 - 1)] = Complex::new(
                        0.01 * (m1 as f64) + 0.02 * (m2 as f64) + 0.03 * (k1 as f64)
                            - 0.015 * (k2 as f64),
                        0.02 * ((m1 - m2) as f64) + 0.01 * (k1 as f64) + 0.04 * (k2 as f64),
                    );
                }
            }
        }
    }

    Ok(SamplePolarizedScatteringAmplitude {
        m_indices,
        n_indices,
        transition_angular_momenta,
        radial_factors,
        transition_matrix,
        first_polynomials,
        second_polynomials,
        xnlm: legendre_normalization_table(4)?,
    })
}

struct SampleEnergyIndependentTransition {
    transition_angular_momenta: Array1<i32>,
    transition_b_matrix: Array6<Complex>,
    combined_rotation: Array3<f64>,
    first_rotation: Array3<f64>,
    last_rotation: Array3<f64>,
}

impl SampleEnergyIndependentTransition {
    fn input(&self) -> EnergyIndependentMatrixInput<'_> {
        EnergyIndependentMatrixInput {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            transition_b_matrix: self.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            spin_index: 1,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: TransitionRotationInput::Polarized {
                first_rotation: self.first_rotation.view(),
                last_rotation: self.last_rotation.view(),
                first_eta: 0.23,
                last_eta: 0.41,
            },
        }
    }

    fn unpolarized_input(&self) -> EnergyIndependentMatrixInput<'_> {
        EnergyIndependentMatrixInput {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            transition_b_matrix: self.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            spin_index: 0,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: TransitionRotationInput::Unpolarized {
                combined_rotation: self.combined_rotation.view(),
            },
        }
    }
}

fn sample_energy_independent_transition_inputs() -> SampleEnergyIndependentTransition {
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
    let mut transition_b_matrix = Array6::zeros((7, 2, 8, 7, 2, 8).f());
    for k2 in 1..=8 {
        for s2 in 0..=1 {
            for m2 in -3_i32..=3 {
                for k1 in 1..=8 {
                    for s1 in 0..=1 {
                        for m1 in -3_i32..=3 {
                            let first_m = (m1 + 3) as usize;
                            let second_m = (m2 + 3) as usize;
                            transition_b_matrix[(first_m, s1, k1 - 1, second_m, s2, k2 - 1)] =
                                Complex::new(
                                    0.01 * (m1 as f64) + 0.02 * (m2 as f64) + 0.03 * (k1 as f64)
                                        - 0.015 * (k2 as f64)
                                        + 0.04 * (s1 as f64)
                                        - 0.025 * (s2 as f64),
                                    0.02 * ((m1 - m2) as f64)
                                        + 0.01 * (k1 as f64)
                                        + 0.04 * (k2 as f64)
                                        + 0.03 * (s1 as f64)
                                        + 0.02 * (s2 as f64),
                                );
                        }
                    }
                }
            }
        }
    }

    SampleEnergyIndependentTransition {
        transition_angular_momenta,
        transition_b_matrix,
        combined_rotation: sample_mmtr_rotation(1),
        first_rotation: sample_mmtr_rotation(2),
        last_rotation: sample_mmtr_rotation(3),
    }
}

fn sample_mmtr_rotation(leg: usize) -> Array3<f64> {
    let mut rotation = Array3::zeros((4, 7, 7).f());
    for l in 0..=3 {
        let il = (l + 1) as f64;
        for m1 in -3_i32..=3 {
            for m2 in -3_i32..=3 {
                if (m1.unsigned_abs() as usize) <= l && (m2.unsigned_abs() as usize) <= l {
                    let row = (m1 + 3) as usize;
                    let column = (m2 + 3) as usize;
                    rotation[(l, row, column)] =
                        (0.13 * il + 0.07 * (m1 as f64) - 0.05 * (m2 as f64) + 0.17 * (leg as f64))
                            .cos();
                }
            }
        }
    }
    rotation
}

fn bench_interpolation(c: &mut Criterion) {
    let xs: Vec<_> = (0..128).map(|index| index as f64 * 0.05).collect();
    let ys: Vec<_> = xs
        .iter()
        .map(|&x| (x * x * x) - (0.5 * x * x) + (2.0 * x) + 1.0)
        .collect();
    let complex_ys: Vec<_> = xs.iter().map(|&x| Complex::new(x.sin(), x.cos())).collect();

    c.bench_function("terp_cubic_128_points", |b| {
        b.iter(|| {
            black_box(terp(
                black_box(&xs),
                black_box(&ys),
                black_box(3),
                black_box(2.75),
            ))
        });
    });
    c.bench_function("terpc_cubic_128_points", |b| {
        b.iter(|| {
            black_box(terpc(
                black_box(&xs),
                black_box(&complex_ys),
                black_box(3),
                black_box(2.75),
            ))
        });
    });
    c.bench_function("lint_128_points", |b| {
        b.iter(|| black_box(lint(black_box(&xs), black_box(&ys), black_box(2.75))));
    });
    c.bench_function("polcoe_15_points", |b| {
        b.iter(|| {
            black_box(interpolation_polynomial_coefficients(
                black_box(&xs[..15]),
                black_box(&ys[..15]),
            ))
        });
    });

    let min_xs: Vec<_> = (1..=13)
        .map(|index| -1.0 + 0.5 * (index as f64 - 1.0))
        .collect();
    let min_ys: Vec<_> = min_xs
        .iter()
        .map(|&x| (x - 2.15).powi(2) + 0.02 * (x - 2.15).powi(4) + 0.1)
        .collect();
    let bracket = match bracket_table_minimum(&min_xs, &min_ys, 3, 0.0, 0.75) {
        Ok(bracket) => bracket,
        Err(error) => {
            eprintln!("skipping table minimization benches: {error}");
            return;
        }
    };
    c.bench_function("mnbrak_table_cubic_13_points", |b| {
        b.iter(|| {
            black_box(bracket_table_minimum(
                black_box(&min_xs),
                black_box(&min_ys),
                black_box(3),
                black_box(0.0),
                black_box(0.75),
            ))
        });
    });
    c.bench_function("brent_table_cubic_13_points", |b| {
        b.iter(|| {
            black_box(brent_table_minimum(
                black_box(&min_xs),
                black_box(&min_ys),
                black_box(3),
                black_box(bracket),
                black_box(1.0e-5),
            ))
        });
    });
}

fn bench_quadrature(c: &mut Criterion) {
    let xs: Vec<_> = (0..1024).map(|index| index as f64 * 0.01).collect();
    let ys: Vec<_> = xs.iter().map(|&x| x.sin() * x.exp()).collect();
    c.bench_function("trap_1024_points", |b| {
        b.iter(|| black_box(trap(black_box(&xs), black_box(&ys))));
    });
    c.bench_function("gauleg_64_points", |b| {
        b.iter(|| {
            black_box(gauss_legendre_quadrature(
                black_box(-1.0),
                black_box(1.0),
                black_box(64),
            ))
        });
    });

    let radii: Vec<_> = (0..128)
        .map(|index| (-8.8 + index as f64 * 0.05).exp())
        .collect();
    let values: Vec<_> = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| radius * (1.0 + index as f64 * 0.001))
        .collect();
    let rnrm = radii[100] * 0.02_f64.exp();
    c.bench_function("somm2_128_points", |b| {
        b.iter(|| {
            black_box(somm2(
                black_box(&radii),
                black_box(&values),
                black_box(0.05),
                black_box(0.5),
                black_box(rnrm),
                black_box(0),
            ))
        });
    });
}

fn bench_bessel(c: &mut Criterion) {
    c.bench_function("besjn_medium_l17", |b| {
        b.iter(|| black_box(besjn(black_box(Complex::new(3.5, 0.4)), black_box(17))));
    });
    c.bench_function("besjh_large_l8", |b| {
        b.iter(|| black_box(besjh(black_box(Complex::new(12.0, 0.5)), black_box(8))));
    });
    c.bench_function("exjlnl_l9", |b| {
        b.iter(|| black_box(exjlnl(black_box(Complex::new(6.1, 0.8)), black_box(9))));
    });
}

fn bench_convolution(c: &mut Criterion) {
    let omega: Vec<_> = (0..128).map(|index| -5.0 + index as f64 * 0.1).collect();
    let spectrum: Vec<_> = omega
        .iter()
        .map(|&energy| Complex::new((energy * 0.7).sin(), (energy * 0.4).cos()))
        .collect();

    c.bench_function("conv_128_points", |b| {
        b.iter(|| {
            black_box(conv(
                black_box(&omega),
                black_box(&spectrum),
                black_box(0.2),
            ))
        });
    });
}

fn bench_fms(c: &mut Criterion) {
    c.bench_function("rehr_albers_polynomials_lx3", |b| {
        b.iter(|| {
            black_box(rehr_albers_polynomials(
                black_box(3),
                black_box(4),
                black_box(4),
                black_box(Complex32::new(1.25, 0.4)),
            ))
        });
    });

    let Ok(clm) = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4)) else {
        return;
    };
    let Ok(xnlm) = legendre_normalization_table(3) else {
        return;
    };
    let mut xclm = Array4::zeros((4, 4, 2, 2).f());
    for l in 0..=3 {
        for m in 0..=3 {
            xclm[(m, l, 1, 0)] = clm[(l, m)];
        }
    }
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };
    c.bench_function("rehr_albers_z_axis_propagator_mu1", |b| {
        b.iter(|| {
            black_box(rehr_albers_z_axis_propagator(
                black_box(1),
                black_box(first),
                black_box(second),
                black_box(xclm.view()),
                black_box(xnlm.view()),
            ))
        });
    });

    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 2.0],
        [0.0, 5.0e-8, 2.0e-7],
        [0.0, 2.0e-7, 0.0],
    ];
    c.bench_function("fms_pair_polar_angles", |b| {
        b.iter(|| {
            black_box(pair_polar_angles(
                black_box(&positions),
                black_box(1),
                black_box(0),
            ))
        });
    });

    c.bench_function("fms_sort_atoms_by_radius", |b| {
        b.iter(|| {
            let mut atoms = sample_fms_atoms();
            black_box(sort_atoms_by_radius(black_box(&mut atoms[..])))
        });
    });
    c.bench_function("fms_sort_representative_atoms", |b| {
        b.iter(|| {
            let mut atoms = sample_representative_atoms();
            black_box(sort_representative_atoms(
                black_box(0),
                black_box(3),
                black_box(&mut atoms[..]),
            ))
        });
    });
    c.bench_function("fms_rotation_matrix_l3", |b| {
        b.iter(|| {
            black_box(fms_rotation_matrix(
                black_box(3),
                black_box(3),
                black_box(0.7),
                black_box(1.1),
                black_box(FmsRotationDirection::Forward),
            ))
        });
    });
    c.bench_function("fms_pair_tables_l2_atoms3", |b| {
        b.iter(|| {
            black_box(fms_pair_tables(
                black_box(2),
                black_box(Complex32::new(1.2, 0.3)),
                black_box(&sample_pair_table_atoms()),
            ))
        });
    });

    let pair_atoms = sample_pair_table_atoms();
    let free_wave_number = Complex32::new(1.2, 0.3);
    let Ok(pair_tables) = fms_pair_tables(2, free_wave_number, &pair_atoms) else {
        return;
    };
    let Ok(free_xnlm) = legendre_normalization_table(2) else {
        return;
    };
    let Ok(backward_rotation) = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)
    else {
        return;
    };
    let Ok(forward_rotation) = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)
    else {
        return;
    };
    let free_first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };
    let free_second = StateKet {
        atom: 2,
        angular_momentum: 2,
        magnetic: -1,
        spin: 1,
    };
    c.bench_function("fms_free_propagator_element_l2", |b| {
        b.iter(|| {
            black_box(fms_free_propagator_element(FmsFreePropagatorInput {
                first: black_box(free_first),
                second: black_box(free_second),
                rho: black_box(pair_tables.rho[(0, 1)]),
                wave_number: black_box(free_wave_number),
                mean_square_displacement: black_box(0.05),
                xclm: black_box(pair_tables.polynomials.view()),
                xnlm: black_box(free_xnlm.view()),
                backward_rotation: black_box(backward_rotation.view()),
                forward_rotation: black_box(forward_rotation.view()),
            }))
        });
    });

    let mut free_rotations = Array6::zeros((5, 5, 3, 2, 3, 3).f());
    copy_rotation_pair(
        &mut free_rotations,
        1,
        0,
        FmsRotationDirection::Backward,
        &backward_rotation,
    );
    copy_rotation_pair(
        &mut free_rotations,
        1,
        0,
        FmsRotationDirection::Forward,
        &forward_rotation,
    );
    let mut free_sigsqr = Array2::zeros((3, 3).f());
    free_sigsqr[(1, 0)] = 0.05;
    let free_states = [free_first, free_second];
    c.bench_function("fms_free_propagator_matrix_states2", |b| {
        b.iter(|| {
            black_box(fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
                states: black_box(&free_states),
                atoms: black_box(&pair_atoms),
                direct_cutoff: black_box(3.0),
                rho: black_box(pair_tables.rho.view()),
                wave_number: black_box(free_wave_number),
                mean_square_displacements: black_box(free_sigsqr.view()),
                xclm: black_box(pair_tables.polynomials.view()),
                xnlm: black_box(free_xnlm.view()),
                rotations: black_box(free_rotations.view()),
            }))
        });
    });

    let mut phase_shifts = Array3::zeros((2, 5, 2).f());
    phase_shifts[(0, 4, 1)] = Complex32::new(0.2, 0.05);
    phase_shifts[(0, 0, 1)] = Complex32::new(-0.1, 0.03);
    phase_shifts[(1, 4, 1)] = Complex32::new(0.15, -0.02);
    phase_shifts[(1, 0, 1)] = Complex32::new(0.07, 0.04);
    let Ok(t_matrix_spin_orbit) = spin_orbit_coupling_tables(2) else {
        return;
    };
    c.bench_function("fms_t_matrix_element_spin_mix_l2", |b| {
        b.iter(|| {
            black_box(fms_t_matrix_element(FmsTMatrixInput {
                first: black_box(free_first),
                second: black_box(StateKet {
                    magnetic: 0,
                    spin: 2,
                    ..free_first
                }),
                spin_channels: black_box(2),
                spin_selector: black_box(0),
                potential: black_box(1),
                phase_shifts: black_box(phase_shifts.view()),
                spin_orbit: black_box(&t_matrix_spin_orbit),
            }))
        });
    });

    let t_matrix_atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 1,
    }];
    let t_matrix_states = [
        free_first,
        StateKet {
            magnetic: 0,
            spin: 2,
            ..free_first
        },
    ];
    c.bench_function("fms_t_matrix_table_states2", |b| {
        b.iter(|| {
            black_box(fms_t_matrix_table(FmsTMatrixTableInput {
                states: black_box(&t_matrix_states),
                atoms: black_box(&t_matrix_atoms),
                spin_channels: black_box(2),
                spin_selector: black_box(0),
                phase_shifts: black_box(phase_shifts.view()),
                spin_orbit: black_box(&t_matrix_spin_orbit),
            }))
        });
    });

    let Ok(lu_states) = construct_state_kets(2, &[0], &[1], 1) else {
        return;
    };
    let (lu_g0, lu_t) = reference_gglu_inputs(lu_states.states.len());
    c.bench_function("fms_iterative_system_states8", |b| {
        b.iter(|| {
            black_box(fms_iterative_system_matrix(FmsIterativeSystemInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_bicgstab_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_bicgstab_scattering(FmsBiCgStabInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_tfqmr_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_tfqmr_scattering(FmsTfqmrInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_recursion_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_recursion_scattering(FmsRecursionInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_graves_morris_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_graves_morris_scattering(FmsGravesMorrisInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_lu_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_lu_scattering(FmsLuInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
            }))
        });
    });
    let lu_t_full = reference_full_potential_t_matrix(lu_states.states.len());
    c.bench_function("fms_full_potential_lu_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t_full.view()),
            }))
        });
    });
}

fn sample_fms_atoms() -> [FmsAtom; 5] {
    [
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [-1.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 3,
        },
        FmsAtom {
            position: [0.0, 2.0, 0.0],
            potential: 4,
        },
    ]
}

fn sample_representative_atoms() -> [FmsAtom; 6] {
    [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [3.0, 0.0, 0.0],
            potential: 3,
        },
        FmsAtom {
            position: [4.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [5.0, 0.0, 0.0],
            potential: 1,
        },
    ]
}

fn sample_pair_table_atoms() -> [FmsAtom; 3] {
    [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
        FmsAtom {
            position: [-1.0, 0.0, 0.5],
            potential: 2,
        },
    ]
}

fn copy_rotation_pair(
    rotations: &mut Array6<Complex32>,
    atom2: usize,
    atom1: usize,
    direction: FmsRotationDirection,
    table: &Array3<Complex32>,
) {
    let branch = match direction {
        FmsRotationDirection::Forward => 0,
        FmsRotationDirection::Backward => 1,
    };
    for l in 0..table.shape()[2] {
        for m1 in 0..table.shape()[1] {
            for m2 in 0..table.shape()[0] {
                rotations[(m2, m1, l, branch, atom2, atom1)] = table[(m2, m1, l)];
            }
        }
    }
}

fn reference_gglu_inputs(state_count: usize) -> (Array2<Complex32>, Array2<Complex32>) {
    let mut free_propagator = Array2::zeros((state_count, state_count).f());
    let mut t_matrix = Array2::zeros((2, state_count).f());
    for column in 0..state_count {
        for row in 0..state_count {
            let row_feff = row as f32 + 1.0;
            let column_feff = column as f32 + 1.0;
            if row != column {
                free_propagator[(row, column)] = Complex32::new(
                    0.01 * row_feff - 0.02 * column_feff,
                    0.015 * row_feff + 0.005 * column_feff,
                );
            }
        }
        let column_feff = column as f32 + 1.0;
        t_matrix[(0, column)] = Complex32::new(0.02 * column_feff, -0.01 * column_feff);
        t_matrix[(1, column)] = Complex32::new(-0.005 * column_feff, 0.003 * column_feff);
    }
    (free_propagator, t_matrix)
}

fn reference_full_potential_t_matrix(state_count: usize) -> Array2<Complex32> {
    let mut t_matrix = Array2::zeros((state_count, state_count).f());
    for column in 0..state_count {
        for row in 0..state_count {
            let row_feff = row as f32 + 1.0;
            let column_feff = column as f32 + 1.0;
            t_matrix[(row, column)] = Complex32::new(
                0.002 * row_feff + 0.001 * column_feff,
                -0.0015 * row_feff + 0.0007 * column_feff,
            );
        }
    }
    t_matrix
}

fn bench_scalar_helpers(c: &mut Criterion) {
    c.bench_function("nuclear_mass", |b| {
        b.iter(|| black_box(nuclear_mass(black_box(92))));
    });
    c.bench_function("distance_between", |b| {
        b.iter(|| {
            black_box(distance_between(
                black_box([1.0, -2.0, 0.5]),
                black_box([-3.0, 4.0, 2.5]),
            ))
        });
    });
    c.bench_function("eels_electron_wavelength", |b| {
        b.iter(|| black_box(electron_wavelength_atomic_units(black_box(300_000.0))));
    });
    c.bench_function("eels_euler_rotation_matrix", |b| {
        b.iter(|| {
            black_box(eels_euler_rotation_matrix(
                black_box(0.3),
                black_box(0.4),
                black_box(-0.2),
            ))
        });
    });
    c.bench_function("eels_integration_mesh_log", |b| {
        b.iter(|| {
            black_box(eels_integration_mesh(black_box(EelsMeshInput {
                collection_angle: 0.015,
                convergence_angle: 0.008,
                theta0: 0.001,
                theta_x_center: -0.0015,
                theta_y_center: 0.0005,
                radial_count: 3,
                angular_count: 2,
                mode: EelsMeshMode::Logarithmic,
            })))
        });
    });
    let hydrogen_potentials = Array1::from_vec(vec![0, 1, 0]);
    let potential_atomic_numbers = Array1::from_vec(vec![8, 1]);
    let hydrogen_positions = arr2(&[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    c.bench_function("adjust_hydrogen_bonds_moveh", |b| {
        b.iter(|| {
            black_box(adjust_hydrogen_bonds(black_box(
                HydrogenBondAdjustmentInput {
                    atom_potentials: hydrogen_potentials.view(),
                    potential_atomic_numbers: potential_atomic_numbers.view(),
                    positions: hydrogen_positions.view(),
                },
            )))
        });
    });
    c.bench_function("x_log_x", |b| {
        b.iter(|| black_box(x_log_x(black_box(2.5))));
    });
    c.bench_function("dirac_hara_exchange_potential", |b| {
        b.iter(|| {
            black_box(dirac_hara_exchange_potential(
                black_box(2.0),
                black_box(1.3),
            ))
        });
    });
    c.bench_function("hedin_lundqvist_ffq", |b| {
        b.iter(|| {
            black_box(hedin_lundqvist_ffq(
                black_box(0.8),
                black_box(0.42),
                black_box(1.2),
                black_box(0.7),
                black_box(4.0 / 3.0),
            ))
        });
    });
    c.bench_function("von_barth_hedin_potential", |b| {
        b.iter(|| black_box(von_barth_hedin_potential(black_box(2.5), black_box(1.2))));
    });
    c.bench_function("perdew_zunger_vxc", |b| {
        b.iter(|| black_box(perdew_zunger_vxc(black_box(2.0))));
    });
    c.bench_function("perrot_dharma_wardana_vxc", |b| {
        b.iter(|| black_box(perrot_dharma_wardana_vxc(black_box(2.0), black_box(0.05))));
    });
    c.bench_function("karasiev_sjostrom_dufty_trickey_vxc", |b| {
        b.iter(|| {
            black_box(karasiev_sjostrom_dufty_trickey_vxc(
                black_box(2.0),
                black_box(0.05),
            ))
        });
    });
    c.bench_function("quinn_imaginary_self_energy", |b| {
        b.iter(|| {
            black_box(quinn_imaginary_self_energy(
                black_box(1.15),
                black_box(2.0),
                black_box(0.65),
                black_box(0.42),
            ))
        });
    });
    c.bench_function("hedin_lundqvist_imaginary_self_energy", |b| {
        b.iter(|| {
            black_box(hedin_lundqvist_imaginary_self_energy(
                black_box(2.0),
                black_box(1.3),
            ))
        });
    });
    c.bench_function("hedin_lundqvist_self_energy", |b| {
        b.iter(|| black_box(hedin_lundqvist_self_energy(black_box(2.0), black_box(1.3))));
    });
    c.bench_function("muffin_tin_phase_amplitude", |b| {
        b.iter(|| {
            black_box(muffin_tin_phase_amplitude(
                black_box(1.7),
                black_box(Complex::new(0.8, 0.2)),
                black_box(Complex::new(-0.3, 0.4)),
                black_box(Complex::new(1.1, 0.15)),
                black_box(Complex::new(0.9, -0.1)),
                black_box(Complex::new(-0.2, 0.7)),
                black_box(Complex::new(0.4, 0.3)),
                black_box(Complex::new(-0.6, 0.25)),
                black_box(-2),
            ))
        });
    });
    c.bench_function("depressed_quartic_roots", |b| {
        b.iter(|| {
            black_box(depressed_quartic_roots(black_box([
                Complex::new(0.75, -0.2),
                Complex::new(-1.5, 0.6),
                Complex::new(0.3, 0.4),
                Complex::new(2.2, -0.7),
            ])))
        });
    });
    c.bench_function("quadratic_zeros", |b| {
        b.iter(|| {
            black_box(quadratic_zeros(black_box([
                Complex::new(1.0, 0.5),
                Complex::new(-2.0, 1.0),
                Complex::new(0.25, -0.75),
            ])))
        });
    });
    c.bench_function("cubic_zeros", |b| {
        b.iter(|| {
            black_box(cubic_zeros(black_box([
                Complex::new(0.75, -0.2),
                Complex::new(-1.5, 0.6),
                Complex::new(0.3, 0.4),
                Complex::new(2.2, -0.7),
            ])))
        });
    });
    c.bench_function("real_polynomial_roots_croots", |b| {
        b.iter(|| black_box(real_polynomial_roots(black_box([1.0, 0.0, -1.0, 1.0]))));
    });
    c.bench_function("find_self_energy_singularities", |b| {
        b.iter(|| {
            black_box(find_self_energy_singularities(
                black_box([Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)]),
                black_box([0.35, 0.02, 0.8, 0.0]),
                black_box([Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)]),
                black_box(SingularityFunction::First),
            ))
        });
    });
    c.bench_function("omega_q", |b| {
        b.iter(|| black_box(omega_q(black_box(0.7), black_box(0.2))));
    });
    c.bench_function("gamma_q", |b| {
        b.iter(|| black_box(gamma_q(black_box(0.08), black_box(0.2))));
    });
    c.bench_function("log_i", |b| {
        b.iter(|| black_box(log_i(black_box(Complex::new(-1.0, 0.5)), black_box(-1))));
    });
    c.bench_function("hartree_fock_exchange", |b| {
        b.iter(|| {
            black_box(hartree_fock_exchange(
                black_box(Complex::new(1.6, 0.2)),
                black_box(0.8),
                black_box(1.1),
            ))
        });
    });
    let integrand_input = SelfEnergyIntegrandInput {
        q: Complex::new(0.8, 0.0),
        normalized_momentum: Complex::new(0.7, 0.0),
        normalized_energy: Complex::new(0.9, 0.02),
        plasmon_over_fermi: 0.35,
        width_over_fermi: 0.02,
        gap_energy: 0.0,
    };
    c.bench_function("self_energy_r1_integrand", |b| {
        b.iter(|| black_box(self_energy_r1_integrand(black_box(integrand_input))));
    });
    c.bench_function("cgratr_oscillatory", |b| {
        b.iter(|| {
            black_box(cgratr(
                |q| Ok((Complex::new(0.0, 3.0) * q).exp() / (Complex::new(1.0, 0.0) + q * q)),
                black_box(Complex::new(0.0, 0.0)),
                black_box(Complex::new(4.0, 0.0)),
                black_box(1.0e-5),
                black_box(1.0e-4),
                black_box(&[]),
            ))
        });
    });
    let mkexc_energy = ndarray::arr1(&[5.0, 12.0, 25.0, 60.0, 120.0, 250.0, 500.0]);
    let mkexc_loss = ndarray::arr1(&[0.18, 0.45, 0.32, 0.20, 0.11, 0.05, 0.02]);
    c.bench_function("make_excitation_poles_4", |b| {
        b.iter(|| {
            black_box(make_excitation_poles(
                black_box(mkexc_energy.view()),
                black_box(mkexc_loss.view()),
                black_box(12.0),
                black_box(4),
            ))
        });
    });
    c.bench_function("integrated_double_lorentz", |b| {
        b.iter(|| {
            black_box(integrated_double_lorentz(
                black_box(3.1),
                black_box(2.7),
                black_box(0.45),
                black_box(0.3),
                black_box(1.2),
                black_box(-0.08),
                black_box(Some(5.0)),
            ))
        });
    });
    c.bench_function("kk_integral", |b| {
        b.iter(|| {
            black_box(kk_integral(
                black_box(Complex::new(0.7, -0.2)),
                black_box(Complex::new(1.1, 0.3)),
                black_box(-1.0),
                black_box(2.0),
                black_box(0.25),
                black_box(0.4),
            ))
        });
    });
    let rixs_x = ndarray::arr1(&[0.0, 1.0, 2.5]);
    let rixs_y = ndarray::arr1(&[-1.0, 0.5, 2.0, 4.0]);
    let rixs_values = Array2::from_shape_fn((rixs_x.len(), rixs_y.len()).f(), |(row, col)| {
        let fortran_row = row as f64 + 1.0;
        let fortran_col = col as f64 + 1.0;
        Complex::new(
            10.0 * fortran_row + fortran_col,
            -1.5 * fortran_row + 0.25 * fortran_col,
        )
    });
    c.bench_function("bilinear_interpolate_complex", |b| {
        b.iter(|| {
            black_box(bilinear_interpolate_complex(
                black_box(rixs_x.view()),
                black_box(rixs_y.view()),
                black_box(rixs_values.view()),
                black_box(0.4),
                black_box(1.1),
            ))
        });
    });
    c.bench_function("morse_einstein_cumulants", |b| {
        b.iter(|| {
            black_box(morse_einstein_cumulants(
                black_box(0.003),
                black_box(300.0),
                black_box(1.0e-5),
                black_box(400.0),
            ))
        });
    });
    c.bench_function("thermal_expansion_cumulants", |b| {
        b.iter(|| {
            black_box(thermal_expansion_cumulants(
                black_box(29),
                black_box(29),
                black_box(0.003),
                black_box(1.0e-5),
                black_box(400.0),
                black_box(2.55),
            ))
        });
    });
    c.bench_function("quantum_debye_correlation", |b| {
        b.iter(|| {
            black_box(quantum_debye_correlation(
                black_box(2.55),
                black_box(400.0),
                black_box(300.0),
                black_box(29),
                black_box(29),
                black_box(2.7),
            ))
        });
    });
    c.bench_function("classical_debye_correlation", |b| {
        b.iter(|| {
            black_box(classical_debye_correlation(
                black_box(2.55),
                black_box(400.0),
                black_box(300.0),
                black_box(29),
                black_box(29),
                black_box(2.7),
            ))
        });
    });
    let debye_path = Array2::from_shape_fn((3, 3), |(row, col)| match (row, col) {
        (1, 0) => 2.55,
        _ => 0.0,
    });
    let debye_atomic_numbers = [29, 29, 29];
    c.bench_function("quantum_debye_waller_factor", |b| {
        b.iter(|| {
            black_box(quantum_debye_waller_factor(
                black_box(300.0),
                black_box(400.0),
                black_box(2.7),
                black_box(debye_path.view()),
                black_box(&debye_atomic_numbers),
            ))
        });
    });
}

fn bench_sort_helpers(c: &mut Criterion) {
    let values: Vec<_> = (0..256)
        .map(|index| ((index * 37) % 256) as f64 - 128.0)
        .collect();
    c.bench_function("qsortd_order_256", |b| {
        b.iter(|| black_box(qsortd_order_1based(black_box(&values))));
    });
    c.bench_function("sortid_order_256", |b| {
        b.iter(|| black_box(sortid_order_1based(black_box(&values))));
    });
    c.bench_function("sortir_order_256", |b| {
        b.iter(|| black_box(sortir_order_1based(black_box(&values))));
    });

    let int_values: Vec<_> = (0..256).map(|index| ((index * 37) % 256) - 128).collect();
    c.bench_function("sortii_order_256", |b| {
        b.iter(|| black_box(sortii_order_1based(black_box(&int_values))));
    });
}

fn bench_path_helpers(c: &mut Criterion) {
    let path = [1, 2, 3, 4, 5, 6, 7, 8];
    c.bench_function("pack_path_indices_8", |b| {
        b.iter(|| black_box(pack_path_indices(black_box(&path))));
    });
    let packed = [3_329_498, 8_325_663, 13_321_836];
    c.bench_function("unpack_path_indices_8", |b| {
        b.iter(|| black_box(unpack_path_indices(black_box(packed), black_box(8))));
    });
    let (phase_energies, reference_energies, phase_shifts, angular_limits) =
        sample_path_phase_criteria_inputs();
    c.bench_function("path_phase_criteria_tables_43", |b| {
        b.iter(|| {
            black_box(path_phase_criteria_tables(black_box(
                PathPhaseCriteriaInput {
                    energies: &phase_energies,
                    reference_energies: &reference_energies,
                    phase_shifts: phase_shifts.view(),
                    angular_limits: angular_limits.view(),
                    output_energy_count: 38,
                    zero_wave_energy_index: 1,
                },
            )))
        });
    });
    c.bench_function("path_heap_bubble_up", |b| {
        b.iter(|| {
            let mut keys = black_box([1.0, 3.0, 2.0, 5.0, 4.0, 0.5]);
            let mut indices = black_box([10, 30, 20, 50, 40, 5]);
            black_box(path_heap_bubble_up(&mut keys, &mut indices))
        });
    });
    c.bench_function("path_heap_bubble_down", |b| {
        b.iter(|| {
            let mut keys = black_box([6.0, 2.0, 3.0, 4.0, 5.0]);
            let mut indices = black_box([60, 20, 30, 40, 50]);
            black_box(path_heap_bubble_down(&mut keys, &mut indices))
        });
    });

    let atom_positions = ndarray::arr2(&[
        [0.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
        [2.0, 1.0, 0.4],
        [-0.5, 1.7, 0.3],
        [0.7, -1.2, 0.8],
    ]);
    let path_indices = [1_usize, 2, 3, 4];
    c.bench_function("path_geometry_4_scatterers", |b| {
        b.iter(|| {
            black_box(path_geometry(
                black_box(atom_positions.view()),
                black_box(&path_indices),
            ))
        });
    });
    c.bench_function("path_output_parameters_4", |b| {
        b.iter(|| {
            black_box(path_output_parameters(
                black_box(atom_positions.view()),
                black_box(&path_indices),
            ))
        });
    });
    c.bench_function("path_standard_coordinates_4", |b| {
        b.iter(|| {
            black_box(path_standard_coordinates(black_box(
                PathStandardCoordinatesInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    polarization: 0,
                    spin: 0,
                    electric_vector: [0.0, 0.0, 1.0],
                    incident_vector: [0.0, 0.0, 0.0],
                    symmetry_case_override: None,
                },
            )))
        });
    });

    let atom_potentials: Vec<_> = (0..=8).map(|index| index % 4).collect();
    c.bench_function("path_canonical_representation_4", |b| {
        b.iter(|| {
            black_box(path_canonical_representation(black_box(
                PathCanonicalRepresentationInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    atom_potentials: &atom_potentials,
                    polarization: 0,
                    spin: 0,
                    electric_vector: [0.0, 0.0, 1.0],
                    incident_vector: [0.0, 0.0, 0.0],
                    symmetry_case_override: None,
                    force_no_symmetry: false,
                },
            )))
        });
    });

    let hash_positions = ndarray::arr2(&[
        [1.23456, -0.34567, 0.12549],
        [-2.25, 1.5004, -0.9995],
        [0.0, 2.4996, 3.3333],
        [0.75, -0.25, 0.5],
    ]);
    let potential_indices = [1, 3, 0, 2];
    c.bench_function("path_degeneracy_hash_4", |b| {
        b.iter(|| {
            black_box(path_degeneracy_hash(
                black_box(hash_positions.view()),
                black_box(&potential_indices),
            ))
        });
    });

    let criteria_distances = [1.10, 1.25, 1.40, 1.60, 1.20];
    let criteria_angles = [0.80, -0.35, 0.55, -0.10, 0.25];
    let criteria_beta = [-3, 4, 10, -2, 0];
    let fbeta = Array3::from_shape_fn((81, 4, 3), |(beta_row, potential, criterion)| {
        let beta_index = beta_row as i32 - 40;
        f64::from(
            0.5_f32
                + 0.01_f32 * potential as f32
                + 0.002_f32 * (criterion + 1) as f32
                + 0.003_f32 * beta_index.abs() as f32
                + 0.0001_f32 * beta_index as f32,
        )
    });
    let criteria_waves = [2.0, 3.5, 5.0];
    let mean_free_paths = [7.5, 10.0, 12.0];
    c.bench_function("path_heap_criterion_4", |b| {
        b.iter(|| {
            black_box(path_heap_criterion(
                black_box(&path_indices),
                black_box(&criteria_distances),
                black_box(&criteria_beta),
                black_box(&atom_potentials),
                black_box(fbeta.view()),
                black_box(&criteria_waves),
            ))
        });
    });
    c.bench_function("path_output_criterion_4", |b| {
        b.iter(|| {
            black_box(path_output_criterion(black_box(PathOutputCriterionInput {
                path_indices: &path_indices,
                leg_distances: &criteria_distances,
                angle_cosines: &criteria_angles,
                beta_indices: &criteria_beta,
                atom_potentials: &atom_potentials,
                fbeta_critical: fbeta.view(),
                mean_free_paths: &mean_free_paths,
                wave_numbers: &criteria_waves,
                current_normalization: 0.004,
            })))
        });
    });

    let mut cluster_outside = vec![false; atom_potentials.len()];
    cluster_outside[4] = true;
    c.bench_function("path_criteria_decision_4", |b| {
        b.iter(|| {
            black_box(path_criteria_decision(black_box(
                PathCriteriaDecisionInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    atom_potentials: &atom_potentials,
                    cluster_outside: &cluster_outside,
                    fbeta_critical: fbeta.view(),
                    mean_free_paths: &mean_free_paths,
                    wave_numbers: &criteria_waves,
                    max_path_length: 20.0,
                    heap_cutoff: 0.0,
                    output_cutoff: 50.0,
                    current_normalization: -1.0,
                },
            )))
        });
    });

    let fbeta_output = Array3::from_shape_fn((81, 4, 5), |(beta_row, potential, energy)| {
        let beta_index = beta_row as i32 - 40;
        f64::from(
            0.45_f32
                + 0.008_f32 * potential as f32
                + 0.015_f32 * (energy + 1) as f32
                + 0.0025_f32 * beta_index.abs() as f32
                + 0.0002_f32 * beta_index as f32,
        )
    });
    let output_waves = [1.2, 2.0, 3.25, 4.5, 6.0];
    let output_mean_free_paths = [6.0, 7.5, 9.0, 11.0, 14.0];
    c.bench_function("path_output_importance_4", |b| {
        b.iter(|| {
            black_box(path_output_importance(black_box(
                PathOutputImportanceInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    atom_potentials: &atom_potentials,
                    fbeta: fbeta_output.view(),
                    wave_numbers: &output_waves,
                    mean_free_paths: &output_mean_free_paths,
                    start_energy_index: 1,
                    fbeta_critical: fbeta.view(),
                    critical_wave_numbers: &criteria_waves,
                    critical_mean_free_paths: &mean_free_paths,
                    current_normalization: 0.004,
                },
            )))
        });
    });
}

fn sample_path_phase_criteria_inputs()
-> (Vec<Complex>, Vec<Complex>, Array3<Complex>, Array2<usize>) {
    let energy_count = 43;
    let potential_count = 3;
    let angular_channels = 4;
    let energies = (0..energy_count)
        .map(|index| {
            let ie = (index + 1) as f64;
            Complex::new(0.02 * (ie - 2.0) + 0.001 * (ie - 1.0), 0.005 + 0.0003 * ie)
        })
        .collect::<Vec<_>>();
    let references = vec![Complex::new(-0.015, -0.002); energy_count];
    let phase_shifts = Array3::from_shape_fn(
        (energy_count, angular_channels, potential_count).f(),
        |(energy, angular, potential)| {
            let ie = (energy + 1) as f64;
            let il = (angular + 1) as f64;
            let iph = potential as f64;
            Complex::new(
                0.02 * ie + 0.11 * il + 0.03 * iph,
                0.004 * ie - 0.002 * il + 0.001 * iph,
            )
        },
    );
    let angular_limits = Array2::from_shape_fn(
        (energy_count, potential_count).f(),
        |(energy, potential)| (energy + 1 + potential) % angular_channels,
    );
    (energies, references, phase_shifts, angular_limits)
}

criterion_group!(
    benches,
    bench_angular_tables,
    bench_state_kets,
    bench_kspace_helpers,
    bench_density_helpers,
    bench_grid_helpers,
    bench_genfmt_helpers,
    bench_interpolation,
    bench_quadrature,
    bench_bessel,
    bench_convolution,
    bench_fms,
    bench_scalar_helpers,
    bench_sort_helpers,
    bench_path_helpers
);
criterion_main!(benches);
