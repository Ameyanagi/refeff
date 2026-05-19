use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3, Array4, Array6, ShapeBuilder, arr1, arr2, array};
use num_complex::Complex32;
use refeff_core::{
    AtomicCoulombCoefficientInput, AtomicFormFactorInput, AtomicLagrangeParametersInput,
    AtomicNuclearPotentialInput, AtomicOverlapAmplitudeReductionInput, AtomicRadialIntegralRequest,
    AtomicSchmidtIntegralRequest, AtomicSchmidtOrthogonalizationInput, AtomicTabulationInput,
    AtomicTabulationIntegralRequest, AtomicTotalEnergyInput, BasisTransformMode, BravaisLattice,
    BroydenMixInput, BroydenWorkspace, Complex, ComptonGridInput, ComptonProfileInput,
    ComptonRhoZzpInput, ComptonWindow, CoulombPotentialSlwInput, CoulombPotentialUpdateInput,
    CoulombUpdateMode, CurvedWavePolynomialInput, DiracSpinorGridInput,
    DiracSpinorOrbitalsGridInput, DmdwPathDescriptor, DmdwPhononCoupling, DmdwPoleWeightedA2f,
    DmdwType2AtomGroup, EelsMeshInput, EelsMeshMode, EnergyIndependentMatrixInput, EpsilonTable,
    FEFF_BOHR_ANGSTROM, FermiLevelInput, Ff2xAtanCorrectionInput, Ff2xExcitationConvolutionInput,
    FmsAtom, FmsBiCgStabInput, FmsFreePropagatorInput, FmsFreePropagatorMatrixInput,
    FmsFullPotentialLuInput, FmsGravesMorrisInput, FmsIterativeSystemInput, FmsLuInput,
    FmsRecursionInput, FmsRotationDirection, FmsTMatrixInput, FmsTMatrixTableInput, FmsTfqmrInput,
    FprimeContourIntegralInput, FprimeLogCase, FprimePositiveAxisIntegralInput,
    FullSpectrumBackgroundInput, FullSpectrumBackgroundSegmentInput, FullSpectrumDefaultGridEdge,
    FullSpectrumDrudeInput, FullSpectrumEdgeAssemblyInput, FullSpectrumEdgeGridInput,
    FullSpectrumEdgeSelectionInput, FullSpectrumFineStructureInput,
    FullSpectrumFineStructureSegmentInput, FullSpectrumHamakerInput,
    FullSpectrumKramersKronigInput, FullSpectrumLinearGridInput, FullSpectrumNumberDensityInput,
    FullSpectrumOpticalConstantsInput, FullSpectrumQSumInput,
    FullSpectrumScatteringDielectricInput, FullSpectrumSumRulesInput, FullSpectrumValenceInput,
    GenfmtLegendreNormalizationInput, HydrogenBondAdjustmentInput, InitialStateRotationInput,
    InterstitialShellValuesInput, LambdaIndexInput, LoucksSphericalOverlapInput, MinimumBracket,
    MuffinTinOverlapMatrixInput, MuffinTinOverlapNeighbor, MuffinTinOverlapProjectionInput,
    MuffinTinOverlapProjectionMode, NormanRadiusInput, OverlapDensityIndicesInput,
    PathCanonicalRepresentationInput, PathCriteriaDecisionInput, PathOutputCriterionInput,
    PathOutputImportanceInput, PathPhaseCriteriaInput, PathRotationInput,
    PathStandardCoordinatesInput, PolarizationTensorMode, PolarizedScatteringAmplitudeInput,
    PotentialGridInput, PotentialOverlapInput, PotentialOverlapNeighbor, RhorrpAtomicDensityInput,
    RhorrpDensityGridInput, RhorrpDensityIntegrationInput, RhorrpEnergyDensityInput,
    RhorrpEnergyPrefactorInput, RhorrpFermiDistributionInput, RhorrpFmsInclusionInput,
    RhorrpIrregularFixInput, RhorrpNearestAtomInput, RhorrpNearestAtomTableInput,
    RhorrpPairDensityInput, RhorrpPairEnergyDensityInput, RhorrpRadialInterpolationInput,
    RhorrpRadialInterpolationLocation, RhorrpSameSiteGreenInput, RhorrpScatteringGreenInput,
    RhorrpWavefunctionInterpolationInput, ScatteringAmplitudeMatrixInput, ScmtEnergyGridInput,
    SelfEnergyIntegrandInput, SfconvExafsConvolutionInput, SfconvExtrinsicSatelliteSplitInput,
    SfconvFeffPathInterpolationInput, SfconvFeffPathSignalInput,
    SfconvMomentumSpectralInterpolationInput, SfconvPathAverageInput,
    SfconvPhotoelectronMomentumInput, SfconvQuasiparticlePeakInput, SfconvQuasiparticleTableInput,
    SfconvSatelliteContext, SfconvSatelliteCorrectionInput, SfconvSatelliteTableInput,
    SfconvSelfEnergyContext, SfconvSo2convExafsEnergyPaddingInput,
    SfconvSo2convExafsPreparationInput, SfconvSo2convMaterialInput,
    SfconvSo2convXanesPreparationInput, SfconvSpectralInterpolationInput,
    SfconvSpectralWeightsInput, SfconvXanesConvolutionInput, SingularityFunction, StateKet,
    TransitionBMatrixInput, TransitionRotationInput, ValenceDensityUpdateInput, XStarInput,
    XsphAxafsInput, XsphHoleOrbitalInput, XsphLgSpectrumUpdateInput, XsphLjSpectrumUpdateInput,
    XsphPhaseEnergyMesh84Input, XsphPhaseUserGridInput, XsphPhaseUserGridKind,
    XsphPhaseUserGridMinimum, XsphPhaseUserGridRecord, XsphPhaseUserRegularGrid,
    XsphSpectrumUpdateMode, XsphThermalPhaseEnergyMeshInput, adjust_hydrogen_bonds,
    atomic_breit_angular_coefficients, atomic_convergence_mix, atomic_coulomb_coefficients,
    atomic_direct_coulomb_coefficient, atomic_exchange_coulomb_coefficient, atomic_form_factor,
    atomic_lagrange_parameters, atomic_nuclear_potential, atomic_occupation_product,
    atomic_overlap_amplitude_reduction, atomic_polynomial_product_coefficient,
    atomic_schmidt_orthogonalization, atomic_tabulation, atomic_total_energy,
    basis_transform_matrices, besjh, besjn, bilinear_interpolate_complex, bracket_table_minimum,
    brent_derivative_minimum, brent_table_minimum, cgratr, change_basis_representation,
    change_cartesian_basis, classical_debye_correlation, combine_epsilon_tables,
    compton_build_grid, compton_jzzp, compton_profile, compton_profiles, compton_rhozzp_slice,
    compton_rotation_axis_angle, construct_state_kets, conv, coulomb_potential_slw, csommjas,
    cubic_zeros, curved_wave_polynomials, define_k_path, depressed_quartic_roots,
    dirac_hara_exchange_potential, distance_between, dmdw_expand_path_descriptor,
    dmdw_moment_summaries_from_poles, dmdw_self_energy_grid_from_a2f_poles,
    dmdw_spectral_function_from_a2f_poles, dmdw_type2_pole_weighted_a2f,
    eels_euler_rotation_matrix, eels_integration_mesh, elam_edge_energy_hartree,
    electron_wavelength_atomic_units, energy_independent_transition_matrix, exjlnl,
    ff2x_atan_correction, ff2x_excitation_convolve, find_self_energy_singularities,
    fix_dirac_spinor_grid, fix_dirac_spinor_orbitals_grid, fix_potential_grid,
    fms_bicgstab_scattering, fms_free_propagator_element, fms_free_propagator_matrix,
    fms_full_potential_lu_scattering, fms_graves_morris_scattering, fms_iterative_system_matrix,
    fms_lu_scattering, fms_pair_tables, fms_recursion_scattering, fms_rotation_matrix,
    fms_t_matrix_element, fms_t_matrix_table, fms_tfqmr_scattering, fprime_contour_integral,
    fprime_log_correction, fprime_positive_axis_integral, full_spectrum_assemble_edge,
    full_spectrum_background_from_fprime, full_spectrum_default_energy_grid,
    full_spectrum_drude_term, full_spectrum_edge_energy_grid, full_spectrum_edges_from_occupations,
    full_spectrum_effective_electron_count, full_spectrum_elam_edge_energies,
    full_spectrum_fine_structure_from_segments, full_spectrum_hamaker_transform,
    full_spectrum_kramers_kronig, full_spectrum_linear_energy_grid, full_spectrum_number_density,
    full_spectrum_optical_constants, full_spectrum_scattering_to_dielectric,
    full_spectrum_sum_rules, full_spectrum_valence_epsilon2, gamma_q, gauss_legendre_quadrature,
    genfmt_legendre_normalization_table, hartree_fock_exchange, hedin_lundqvist_ffq,
    hedin_lundqvist_imaginary_self_energy, hedin_lundqvist_self_energy, initial_state_rotation,
    integrated_double_lorentz, interpolation_polynomial_coefficients, interstitial_fermi_level,
    interstitial_shell_values, karasiev_sjostrom_dufty_trickey_vxc, kk_integral,
    kmesh_arbitrary_mesh, kmesh_basis_divisions, kmesh_bravais_basis, kmesh_tetrahedron_division,
    kmesh_tetrahedron_records, lambda_indices, legendre_normalization_table, legendre_polynomials,
    lint, log_i, make_excitation_poles, mix_broyden_density, mkgtr_clebsch_gordan_coefficients,
    morse_einstein_cumulants, muffin_tin_overlap_matrix, muffin_tin_phase_amplitude,
    next_elam_edge_hartree, norman_radius_from_density, nuclear_mass, omega_q,
    overlap_density_indices, overlap_potential_density, pack_path_indices, pair_polar_angles,
    path_canonical_representation, path_criteria_decision, path_degeneracy_hash, path_geometry,
    path_heap_bubble_down, path_heap_bubble_up, path_heap_criterion, path_output_criterion,
    path_output_importance, path_output_parameters, path_phase_criteria_tables,
    path_rotation_angles, path_standard_coordinates, perdew_zunger_vxc, perrot_dharma_wardana_vxc,
    point_group_operations, polarization_tensor, polarized_scattering_amplitude_matrix,
    previous_elam_edge_hartree, project_muffin_tin_overlap, qsortd_order_1based, quadratic_zeros,
    quantum_debye_correlation, quantum_debye_waller_factor, quinn_imaginary_self_energy,
    real_polynomial_roots, reciprocal_lattice_vectors, reciprocal_metric,
    redefine_lattice_symmetry_operations, reduce_kmesh_common_divisor,
    reduce_kmesh_irreducible_points, reduce_to_lattice_cell, rehr_albers_polynomials,
    rehr_albers_z_axis_propagator, relativistic_clebsch_gordan_coefficients,
    relativistic_state_index_1based, rhorrp_atomic_density, rhorrp_density_grid_points,
    rhorrp_energy_prefactor, rhorrp_evaluate_density_grid, rhorrp_fermi_distribution,
    rhorrp_finish_energy_density, rhorrp_fix_irregular_origin, rhorrp_fms_inclusion_counts,
    rhorrp_integrate_density, rhorrp_interpolate_wavefunction, rhorrp_nearest_atom,
    rhorrp_nearest_atom_table, rhorrp_pair_density, rhorrp_pair_energy_density,
    rhorrp_process_ranges, rhorrp_radial_interpolation_location, rhorrp_same_site_green,
    rhorrp_scattering_green, scattering_amplitude_matrix, scmt_energy_grid,
    self_energy_r1_integrand, sfconv_correct_satellite_weights, sfconv_exafs_convolution,
    sfconv_extrinsic_beta, sfconv_feff_path_signal, sfconv_grater_integrate,
    sfconv_imaginary_self_energy, sfconv_imaginary_self_energy_derivative,
    sfconv_interference_satellite, sfconv_interpolate_feff_path,
    sfconv_interpolate_momentum_spectral_function, sfconv_interpolate_spectral_function,
    sfconv_intrinsic_satellite, sfconv_path_average, sfconv_plasma_parameters,
    sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion, sfconv_q_limits,
    sfconv_quasiparticle_main_peak, sfconv_quasiparticle_table, sfconv_real_self_energy,
    sfconv_real_self_energy_derivative, sfconv_satellite_table, sfconv_select_pole,
    sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
    sfconv_so2conv_pad_exafs_energy_grid, sfconv_so2conv_photoelectron_momentum,
    sfconv_so2conv_prepare_exafs_signal, sfconv_so2conv_prepare_xanes_signal,
    sfconv_spectral_energy_grid, sfconv_spectral_weights, sfconv_split_extrinsic_satellite,
    sfconv_xanes_convolution, somm2, sort_atoms_by_radius, sort_representative_atoms,
    sortid_order_1based, sortii_order_1based, sortir_order_1based, sphere_overlap_lens_volume,
    spherical_harmonics, spin_orbit_coupling_tables, subtract_lattice_translation,
    sum_loucks_spherical_overlap, symmetry_check, terp, terpc, thermal_expansion_cumulants,
    thomas_fermi_density_potential, transform_lapw_symmetry_operations, transition_b_matrix, trap,
    unpack_path_indices, update_coulomb_potential, update_valence_density,
    von_barth_hedin_potential, wigner_rotation, x_log_x, xscorr_arctangent_step,
    xscorr_lorentz_kernel, xsph_angular_density_coefficients, xsph_axafs, xsph_even_energy_mesh,
    xsph_exafs_energy_grid_84, xsph_exponential_energy_mesh, xsph_fprime_energy_grid_84,
    xsph_initial_hole_orbital, xsph_k_energy_mesh, xsph_lj_needed_flags,
    xsph_longitudinal_multipole_factor, xsph_minimize_calculations, xsph_nrixs_transition_weights,
    xsph_occupation_normalization, xsph_phase_energy_mesh_84, xsph_phase_energy_mesh_user,
    xsph_q_bessel_table, xsph_relativistic_multipole_factors, xsph_reverse_energy_grid,
    xsph_sort_energy_grid, xsph_thermal_phase_energy_mesh, xsph_update_nrixs_atom_spectrum,
    xsph_update_nrixs_lg_spectrum, xsph_update_nrixs_lj_spectrum, xsph_vertical_energy_mesh_84,
    xsph_xanes_energy_grid_84, xsph_xes_energy_grid_84, xstar,
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
    c.bench_function("build_mkgtr_clbcoef_lmax8", |b| {
        b.iter(|| {
            black_box(mkgtr_clebsch_gordan_coefficients(
                black_box(8),
                black_box(9),
                black_box(18),
            ))
        });
    });
    c.bench_function("relativistic_state_index_kappa_grid", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for kappa in black_box([-1, 1, -2, 2, -3, 3, -4, 4]) {
                let jp05 = i32::abs(kappa);
                for mu_minus_half in -jp05..jp05 {
                    if let Ok(index) =
                        relativistic_state_index_1based(black_box(kappa), black_box(mu_minus_half))
                    {
                        total = total.saturating_add(index);
                    }
                }
            }
            black_box(total)
        });
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

fn bench_compton_helpers(c: &mut Criterion) {
    c.bench_function("compton_rotation_axis_angle", |b| {
        b.iter(|| {
            black_box(compton_rotation_axis_angle(
                black_box([0.0, 0.0, 1.0]),
                black_box([0.35, -0.25, 0.92]),
            ))
        });
    });

    let grid_input = ComptonGridInput {
        ns: 16,
        nphi: 17,
        nz: 32,
        nzp: 33,
        smax: 0.0,
        phimax: std::f64::consts::PI,
        zmax: 1.2,
        zpmax: 1.5,
        norman_radius: 2.25,
        qhat: [0.35, -0.25, 0.92],
    };
    c.bench_function("compton_build_grid_16_17_32_33", |b| {
        b.iter(|| black_box(compton_build_grid(black_box(grid_input))));
    });

    let Ok(grid) = compton_build_grid(grid_input) else {
        return;
    };
    let jzzp = Array2::from_shape_fn((grid.nz(), grid.nzp()).f(), |(iz, izp)| {
        let iz = iz as f64 + 1.0;
        let izp = izp as f64 + 1.0;
        0.12 * iz + 0.07 * izp + 0.015 * iz * izp
    });
    c.bench_function("compton_profile_cosine_32x33", |b| {
        b.iter(|| {
            black_box(compton_profile(
                black_box(&grid),
                black_box(jzzp.view()),
                black_box(ComptonProfileInput {
                    pq: 1.35,
                    window: ComptonWindow::CosineSquared,
                    window_cutoff: 1.0,
                }),
            ))
        });
    });

    let Ok(profile_grid) = compton_build_grid(ComptonGridInput {
        ns: 32,
        nphi: 32,
        nz: 32,
        nzp: 120,
        smax: 2.642_889_499_664_3,
        phimax: std::f64::consts::TAU,
        zmax: 2.642_889_499_664_3,
        zpmax: 10.0,
        norman_radius: 0.0,
        qhat: [0.0, 0.0, 1.0],
    }) else {
        return;
    };
    let profile_jzzp =
        Array2::from_shape_fn((profile_grid.nz(), profile_grid.nzp()).f(), |(iz, izp)| {
            let z = profile_grid.z[iz];
            let zp = profile_grid.zp[izp];
            (-(z * z) * 0.18 - (zp * zp) * 0.025).exp() * (1.0 + 0.01 * izp as f64)
        });
    let momentum = Array1::from_iter((0..1000).map(|index| 5.0 * index as f64 / 999.0));
    c.bench_function("compton_profile_loop_cosine_1000x32x120", |b| {
        b.iter(|| {
            black_box(
                momentum
                    .iter()
                    .map(|&pq| {
                        compton_profile(
                            black_box(&profile_grid),
                            black_box(profile_jzzp.view()),
                            black_box(ComptonProfileInput {
                                pq,
                                window: ComptonWindow::CosineSquared,
                                window_cutoff: 0.0,
                            }),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>(),
            )
        });
    });
    c.bench_function("compton_profiles_batch_cosine_1000x32x120", |b| {
        b.iter(|| {
            black_box(compton_profiles(
                black_box(&profile_grid),
                black_box(profile_jzzp.view()),
                black_box(momentum.view()),
                black_box(ComptonWindow::CosineSquared),
                black_box(0.0),
            ))
        });
    });

    let Ok(jzzp_grid) = compton_build_grid(ComptonGridInput {
        ns: 8,
        nphi: 9,
        nz: 8,
        nzp: 9,
        ..grid_input
    }) else {
        return;
    };
    c.bench_function("compton_jzzp_stub_8_9_8_9", |b| {
        b.iter(|| black_box(compton_jzzp(black_box(&jzzp_grid), sample_compton_density)));
    });
    c.bench_function("compton_rhozzp_slice_stub_1000", |b| {
        b.iter(|| {
            black_box(compton_rhozzp_slice(
                black_box(&jzzp_grid),
                black_box(ComptonRhoZzpInput {
                    sample_count: 1000,
                    base_z: 0.01,
                }),
                sample_compton_density,
            ))
        });
    });
}

fn sample_compton_density(r: [f64; 3], rp: [f64; 3]) -> Result<f64, refeff_core::ComptonError> {
    let r2 = r.iter().map(|value| value * value).sum::<f64>();
    let rp2 = rp.iter().map(|value| value * value).sum::<f64>();
    let dot = r
        .iter()
        .zip(rp)
        .map(|(left, right)| *left * right)
        .sum::<f64>();
    Ok((-r2 - 0.5 * rp2).exp() + 0.1 * dot)
}

fn bench_rhorrp_helpers(c: &mut Criterion) {
    let axes = arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]]);
    let points_per_axis = [40, 30, 20];
    c.bench_function("rhorrp_density_grid_points_40x30x20", |b| {
        b.iter(|| {
            black_box(rhorrp_density_grid_points(black_box(
                RhorrpDensityGridInput {
                    origin: [0.1, -0.2, 0.3],
                    axes: axes.view(),
                    points_per_axis: &points_per_axis,
                },
            )))
        });
    });
    c.bench_function("rhorrp_evaluate_density_grid_40x30x20", |b| {
        b.iter(|| {
            black_box(rhorrp_evaluate_density_grid(
                black_box(RhorrpDensityGridInput {
                    origin: [0.1, -0.2, 0.3],
                    axes: axes.view(),
                    points_per_axis: &points_per_axis,
                }),
                |point| Ok(point[0] + 2.0 * point[1] - 0.5 * point[2] + point[0] * point[1]),
            ))
        });
    });
    c.bench_function("rhorrp_process_ranges_1000000x64", |b| {
        b.iter(|| black_box(rhorrp_process_ranges(black_box(1_000_000), black_box(64))));
    });

    let positions = Array2::from_shape_fn((128, 3), |(atom, axis)| {
        let atom = atom as f64;
        match axis {
            0 => (atom * 0.37).sin(),
            1 => (atom * 0.23).cos(),
            _ => atom * 0.015,
        }
    });
    let potentials = (0..128).map(|atom| atom % 5).collect::<Vec<_>>();
    let representative_atoms = [0, 7, 31, 63, 95, 127];
    c.bench_function("rhorrp_fms_inclusion_counts_6x128", |b| {
        b.iter(|| {
            black_box(rhorrp_fms_inclusion_counts(black_box(
                RhorrpFmsInclusionInput {
                    atom_positions: positions.view(),
                    representative_atoms: &representative_atoms,
                    fms_radius: 1.25,
                },
            )))
        });
    });
    c.bench_function("rhorrp_nearest_atom_128", |b| {
        b.iter(|| {
            black_box(rhorrp_nearest_atom(black_box(RhorrpNearestAtomInput {
                point: [0.25, -0.15, 0.75],
                atom_positions: positions.view(),
                atom_potentials: &potentials,
                fms_atom_count: None,
            })))
        });
    });
    let nearest_points = Array2::from_shape_fn((3, 4096), |(axis, point)| {
        let point = point as f64;
        match axis {
            0 => 0.001 * point,
            1 => (0.003 * point).sin(),
            _ => (0.002 * point).cos(),
        }
    });
    c.bench_function("rhorrp_nearest_atom_table_4096x128", |b| {
        b.iter(|| {
            black_box(rhorrp_nearest_atom_table(black_box(
                RhorrpNearestAtomTableInput {
                    points: nearest_points.view(),
                    atom_positions: positions.view(),
                    atom_potentials: &potentials,
                    fms_atom_count: None,
                },
            )))
        });
    });

    c.bench_function("rhorrp_radial_interpolation_location", |b| {
        b.iter(|| {
            black_box(rhorrp_radial_interpolation_location(black_box(
                RhorrpRadialInterpolationInput {
                    radius: 0.522_045_776_761_016,
                    x0: 0.7,
                    dx: 0.2,
                    radial_count: 251,
                },
            )))
        });
    });

    c.bench_function("rhorrp_energy_prefactor", |b| {
        b.iter(|| {
            black_box(rhorrp_energy_prefactor(black_box(
                RhorrpEnergyPrefactorInput {
                    energy_hartree: Complex::new(0.2, 0.05),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                },
            )))
        });
    });

    let finish_energies = Array1::from_shape_fn(256, |index| {
        let index = index as f64;
        Complex::new(-0.1 + 0.006 * index, 0.02 * (0.05 * index).sin())
    });
    let finish_green = Array1::from_shape_fn(256, |index| {
        let index = index as f64;
        Complex::new(0.001 * (0.03 * index).cos(), -0.0007 * (0.02 * index).sin())
    });
    c.bench_function("rhorrp_finish_energy_density_256", |b| {
        b.iter(|| {
            black_box(rhorrp_finish_energy_density(black_box(
                RhorrpEnergyDensityInput {
                    energies_hartree: finish_energies.view(),
                    green_function: finish_green.view(),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                    radius: 0.85,
                    prime_radius: 1.25,
                },
            )))
        });
    });

    let same_regular_large = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            0.001 * energy + 0.03 * angular + (0.01 * radial).sin(),
            -0.0007 * energy + 0.02 * angular - (0.008 * radial).cos(),
        )
    });
    let same_irregular_large = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            -0.0008 * energy + 0.04 * angular + (0.012 * radial).cos(),
            0.0005 * energy - 0.01 * angular + (0.009 * radial).sin(),
        )
    });
    let same_regular_small = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            0.0007 * energy - 0.02 * angular + (0.006 * radial).sin(),
            0.0004 * energy + 0.015 * angular - (0.011 * radial).cos(),
        )
    });
    let same_irregular_small = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            -0.0003 * energy + 0.025 * angular - (0.007 * radial).cos(),
            0.0002 * energy + 0.018 * angular + (0.005 * radial).sin(),
        )
    });
    c.bench_function("rhorrp_same_site_green_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_same_site_green(black_box(
                RhorrpSameSiteGreenInput {
                    regular_large: same_regular_large.view(),
                    irregular_large: same_irregular_large.view(),
                    regular_small: same_regular_small.view(),
                    irregular_small: same_irregular_small.view(),
                    first_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 34,
                        fraction: 0.25,
                    },
                    second_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 61,
                        fraction: 0.60,
                    },
                    cosine_between: 0.35,
                },
            )))
        });
    });

    let scattering_phase = Array2::from_shape_fn((64, 4), |(energy, angular)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        Complex::new(
            0.00015 * energy + 0.04 * angular,
            -0.00006 * energy + 0.02 * angular,
        )
    });
    let scattering_phase_prime = Array2::from_shape_fn((64, 4), |(energy, angular)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        Complex::new(
            -0.00011 * energy + 0.03 * angular,
            0.00007 * energy - 0.015 * angular,
        )
    });
    let scattering_matrix = Array3::from_shape_fn((64, 16, 16), |(energy, row, column)| {
        let energy = (energy + 1) as f64;
        let row = (row + 1) as f64;
        let column = (column + 1) as f64;
        Complex::new(
            0.00002 * energy + 0.0004 * row - 0.0003 * column,
            -0.000015 * energy + 0.00025 * row + 0.0001 * column,
        )
    });
    c.bench_function("rhorrp_scattering_green_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_scattering_green(black_box(
                RhorrpScatteringGreenInput {
                    first_regular_large: same_regular_large.view(),
                    first_regular_small: same_regular_small.view(),
                    second_regular_large: same_irregular_large.view(),
                    second_regular_small: same_irregular_small.view(),
                    first_phase: scattering_phase.view(),
                    second_phase: scattering_phase_prime.view(),
                    scattering_matrix: scattering_matrix.view(),
                    first_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 34,
                        fraction: 0.25,
                    },
                    second_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 61,
                        fraction: 0.60,
                    },
                    first_displacement: [0.4, -0.2, 0.6],
                    second_displacement: [-0.3, 0.5, 0.7],
                },
            )))
        });
    });

    let pair_energies = Array1::from_shape_fn(64, |index| {
        let index = index as f64;
        Complex::new(-0.08 + 0.004 * index, 0.015 * (0.05 * index).cos())
    });
    c.bench_function("rhorrp_pair_energy_density_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_pair_energy_density(black_box(
                RhorrpPairEnergyDensityInput {
                    energies_hartree: pair_energies.view(),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                    first_regular_large: same_regular_large.view(),
                    first_irregular_large: same_irregular_large.view(),
                    first_regular_small: same_regular_small.view(),
                    first_irregular_small: same_irregular_small.view(),
                    second_regular_large: same_irregular_large.view(),
                    second_regular_small: same_irregular_small.view(),
                    first_phase: scattering_phase.view(),
                    second_phase: scattering_phase_prime.view(),
                    scattering_matrix: Some(scattering_matrix.view()),
                    same_atom: true,
                    first_displacement: [0.22, -0.18, 0.44],
                    second_displacement: [-0.31, 0.28, 0.36],
                    radial_x0: 0.7,
                    radial_dx: 0.2,
                    radial_count: 96,
                },
            )))
        });
    });

    let wavefunctions = Array3::from_shape_fn((192, 5, 251), |(energy, angular, radial)| {
        let real = 0.001 * energy as f64 + 0.01 * angular as f64 + (0.002 * radial as f64).sin();
        let imag = -0.0005 * energy as f64 + 0.005 * angular as f64 - (0.001 * radial as f64).cos();
        Complex::new(real, imag)
    });
    c.bench_function("rhorrp_interpolate_wavefunction_192x5", |b| {
        b.iter(|| {
            black_box(rhorrp_interpolate_wavefunction(black_box(
                RhorrpWavefunctionInterpolationInput {
                    wavefunctions: wavefunctions.view(),
                    index_below_1based: 120,
                    fraction: 0.375,
                },
            )))
        });
    });

    c.bench_function("rhorrp_fermi_distribution_complex", |b| {
        b.iter(|| {
            black_box(rhorrp_fermi_distribution(black_box(
                RhorrpFermiDistributionInput {
                    energy_hartree: Complex::new(0.2, 0.05),
                    chemical_potential_hartree: 0.1,
                    temperature_hartree: 0.025,
                    chemical_potential_override_hartree: Some(0.22),
                },
            )))
        });
    });

    let irregular_radii = (1..=251)
        .map(|index| {
            let index = index as f64;
            0.02 * index + 0.0001 * index * index
        })
        .collect::<Vec<_>>();
    let irregular_values = Array1::from_shape_fn(251, |index| {
        let one_based = (index + 1) as f64;
        Complex::new(
            (0.07 * one_based).sin() + 0.002 * one_based,
            (0.05 * one_based).cos() - 0.001 * one_based,
        )
    });
    c.bench_function("rhorrp_fix_irregular_origin_251", |b| {
        b.iter(|| {
            black_box(rhorrp_fix_irregular_origin(black_box(
                RhorrpIrregularFixInput {
                    radii: &irregular_radii,
                    values: irregular_values.view(),
                },
            )))
        });
    });

    let atomic_radii = (1..=251)
        .map(|index| {
            let index = index as f64;
            0.015 + 0.035 * index + 0.0002 * (index - 1.0) * (index - 1.0)
        })
        .collect::<Vec<_>>();
    let atomic_positions = Array2::from_shape_fn((128, 3), |(atom, axis)| {
        let atom = atom as f64;
        match axis {
            0 => 1.25 * (0.17 * atom).sin(),
            1 => 1.10 * (0.13 * atom).cos(),
            _ => -0.85 + 0.013 * atom,
        }
    });
    let atomic_potentials = (0..128).map(|atom| atom % 5).collect::<Vec<_>>();
    let atomic_large = Array3::from_shape_fn((251, 4, 5), |(radial, orbital, potential)| {
        let index = (radial + 1) as f64;
        (0.017 * index).sin()
            + 0.021 * (orbital + 1) as f64
            + 0.012 * potential as f64
            + 0.03 * atomic_radii[radial]
    });
    let atomic_small = Array3::from_shape_fn((251, 4, 5), |(radial, orbital, potential)| {
        let index = (radial + 1) as f64;
        (0.011 * index).cos() - 0.014 * (orbital + 1) as f64 + 0.009 * potential as f64
            - 0.02 * atomic_radii[radial]
    });
    c.bench_function("rhorrp_atomic_density_128_atoms", |b| {
        b.iter(|| {
            black_box(rhorrp_atomic_density(black_box(RhorrpAtomicDensityInput {
                point: [0.22, -0.15, 0.18],
                orbital_index_1based: 2,
                atom_positions: atomic_positions.view(),
                atom_potentials: &atomic_potentials,
                radii: &atomic_radii,
                large_components: atomic_large.view(),
                small_components: atomic_small.view(),
            })))
        });
    });

    let contour_energies = Array1::from_shape_fn(64, |index| {
        if index < 16 {
            Complex::new(-0.025, 0.075 - 0.005 * index as f64)
        } else if index < 56 {
            Complex::new(-0.025 + 0.006 * (index - 15) as f64, 0.0)
        } else {
            Complex::new(0.045, 0.0035 * std::f64::consts::TAU * (index - 55) as f64)
        }
    });
    let contour_density = Array1::from_shape_fn(64, |index| {
        let energy = contour_energies[index];
        let one_based = (index + 1) as f64;
        Complex::new(
            0.40 + 0.003 * one_based + 0.04 * energy.re - 0.05 * energy.im,
            -0.20 + 0.002 * one_based + 0.03 * energy.re + 0.04 * energy.im,
        )
    });
    c.bench_function("rhorrp_integrate_density_64_energy", |b| {
        b.iter(|| {
            black_box(rhorrp_integrate_density(black_box(
                RhorrpDensityIntegrationInput {
                    energies_hartree: contour_energies.view(),
                    energy_density: contour_density.view(),
                    real_axis_count: 56,
                    chemical_potential_hartree: 0.045,
                    temperature_hartree: 0.0035,
                    chemical_potential_override_hartree: None,
                },
            )))
        });
    });
    c.bench_function("rhorrp_pair_density_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_pair_density(black_box(RhorrpPairDensityInput {
                pair_energy: RhorrpPairEnergyDensityInput {
                    energies_hartree: contour_energies.view(),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                    first_regular_large: same_regular_large.view(),
                    first_irregular_large: same_irregular_large.view(),
                    first_regular_small: same_regular_small.view(),
                    first_irregular_small: same_irregular_small.view(),
                    second_regular_large: same_irregular_large.view(),
                    second_regular_small: same_irregular_small.view(),
                    first_phase: scattering_phase.view(),
                    second_phase: scattering_phase_prime.view(),
                    scattering_matrix: Some(scattering_matrix.view()),
                    same_atom: true,
                    first_displacement: [0.22, -0.18, 0.44],
                    second_displacement: [-0.31, 0.28, 0.36],
                    radial_x0: 0.7,
                    radial_dx: 0.2,
                    radial_count: 96,
                },
                real_axis_count: 56,
                chemical_potential_hartree: 0.045,
                temperature_hartree: 0.0035,
                chemical_potential_override_hartree: None,
            })))
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
    c.bench_function("dbrent_quartic", |b| {
        b.iter(|| {
            black_box(brent_derivative_minimum(
                black_box(MinimumBracket {
                    ax: -1.0,
                    bx: 1.0,
                    cx: 3.0,
                    fa: 0.0,
                    fb: 0.0,
                    fc: 0.0,
                }),
                black_box(1.0e-8),
                |x| (x - 1.35).powi(2) + 0.05 * (x + 0.5).powi(4) + 0.25,
                |x| 2.0 * (x - 1.35) + 0.2 * (x + 0.5).powi(3),
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
    let complex_dp = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| {
            let row = index as f64 + 1.0;
            Complex::new(radius * (0.3 + 0.002 * row), -0.02 * row)
        })
        .collect::<Vec<_>>();
    let complex_dq = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| {
            let row = index as f64 + 1.0;
            Complex::new(-0.05 * radius * row, 0.004 * row * row)
        })
        .collect::<Vec<_>>();
    c.bench_function("csommjas_128_points", |b| {
        b.iter(|| {
            black_box(csommjas(
                black_box(&radii),
                black_box(&complex_dp),
                black_box(&complex_dq),
                black_box(0.05),
                black_box(0.5),
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

    let excitation_energy = Array1::from_iter((0..256).map(|index| -1.0 + index as f64 * 0.02));
    let excitation_xmu = Array1::from_iter(
        excitation_energy
            .iter()
            .map(|&energy| 0.8 + (energy * 0.9).sin() * 0.2 + (energy * 0.25).cos() * 0.08),
    );
    c.bench_function("ff2x_exconv_256_points", |b| {
        b.iter(|| {
            black_box(ff2x_excitation_convolve(black_box(
                Ff2xExcitationConvolutionInput {
                    energy: excitation_energy.view(),
                    xmu: excitation_xmu.view(),
                    fermi_energy: 0.05,
                    amplitude_reduction: 0.72,
                    relaxation_energy: 0.18,
                    plasmon_frequency: 0.55,
                },
            )))
        });
    });

    let atan_energy = Array1::from_iter((0..320).map(|index| {
        if index < 256 {
            Complex::new(-1.2 + index as f64 * 0.01, 0.08)
        } else {
            Complex::new(1.35, 0.001 + (index - 256) as f64 * 0.002)
        }
    }));
    let atan_xsec = Array1::from_iter(
        atan_energy
            .iter()
            .map(|energy| Complex::new(0.9 + (energy.re * 0.6).sin() * 0.2, energy.re * 0.01)),
    );
    let atan_xsnorm = Array1::from_iter((0..320).map(|index| 0.85 + index as f64 * 0.0005));
    let atan_chia = Array1::from_iter(atan_energy.iter().map(|energy| {
        Complex::new(
            (energy.re * 1.3).cos() * 0.08,
            (energy.re * 0.9).sin() * 0.03,
        )
    }));
    c.bench_function("ff2x_xscorratan_256_horizontal_points", |b| {
        b.iter(|| {
            black_box(ff2x_atan_correction(black_box(Ff2xAtanCorrectionInput {
                spectroscopy: 1,
                energy: atan_energy.view(),
                horizontal_len: 256,
                fermi_index: 120,
                xsec: atan_xsec.view(),
                xsnorm: atan_xsnorm.view(),
                chia: atan_chia.view(),
                real_correction: 0.03,
                imaginary_correction: 0.0,
            })))
        });
    });
}

fn bench_fprime_helpers(c: &mut Criterion) {
    c.bench_function("fprime_funlog_real_branch", |b| {
        b.iter(|| {
            black_box(fprime_log_correction(
                black_box(FprimeLogCase::RealFrequency),
                black_box(0.08),
                black_box(0.50),
                black_box(0.21),
            ))
        });
    });
    c.bench_function("xscorr_lorentz_kernel", |b| {
        b.iter(|| {
            black_box(xscorr_lorentz_kernel(
                black_box(0.08),
                black_box(0.13),
                black_box(0.02),
            ))
        });
    });
    c.bench_function("xscorr_arctangent_step", |b| {
        b.iter(|| black_box(xscorr_arctangent_step(black_box(0.08), black_box(-0.11))));
    });

    let contour_energy = Array1::from_iter(
        (0..96)
            .map(|index| Complex::new(-0.05 + 0.012 * index as f64, 0.02 + 0.0015 * index as f64)),
    );
    let contour_xmu = Array1::from_iter((0..96).map(|index| {
        let row = index as f64 + 1.0;
        Complex::new(0.7 + 0.006 * row + 0.0001 * row * row, -0.04 + 0.001 * row)
    }));
    c.bench_function("fprime_fpint_96_points", |b| {
        b.iter(|| {
            black_box(fprime_contour_integral(black_box(
                FprimeContourIntegralInput {
                    energy: contour_energy.view(),
                    xmu: contour_xmu.view(),
                    start_index: 1,
                    end_index: 95,
                    delta: 0.11,
                    loss: 0.08,
                    epsilon: 1.0e-4,
                    fermi_energy: 0.03,
                },
            )))
        });
    });

    let axis_energy = Array1::from_iter((0..96).map(|index| 0.03 + 0.025 * index as f64));
    let axis_xmu = Array1::from_iter((0..96).map(|index| {
        let row = index as f64 + 1.0;
        Complex::new(0.5 + 0.008 * row + 0.00015 * row * row, 0.01 + 0.002 * row)
    }));
    c.bench_function("fprime_fpintp_96_points", |b| {
        b.iter(|| {
            black_box(fprime_positive_axis_integral(black_box(
                FprimePositiveAxisIntegralInput {
                    energy: axis_energy.view(),
                    xmu: axis_xmu.view(),
                    delta: 0.09,
                    loss: 0.08,
                    fermi_energy: 0.03,
                },
            )))
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
    c.bench_function("atom_nucdev_finite_nucleus", |b| {
        b.iter(|| {
            black_box(atomic_nuclear_potential(black_box(
                AtomicNuclearPotentialInput {
                    nuclear_charge: 92.0,
                    step: 0.05,
                    requested_nucleus_index: -11,
                    radial_count: 251,
                    coefficient_count: 10,
                    first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
                },
            )))
        });
    });
    let elam_components = [29, 8, 79];
    c.bench_function("elam_edge_lookup", |b| {
        b.iter(|| {
            black_box((
                elam_edge_energy_hartree(black_box(29), black_box(1)),
                previous_elam_edge_hartree(black_box(35.0), black_box(&elam_components)),
                next_elam_edge_hartree(black_box(35.0), black_box(&elam_components)),
            ))
        });
    });

    let left = (1..=10)
        .map(|index| 0.1 * index as f64 + 0.03)
        .collect::<Vec<_>>();
    let right = (1..=10)
        .map(|index| -0.04 * index as f64 + 0.25)
        .collect::<Vec<_>>();
    c.bench_function("atom_aprdev_product_l7", |b| {
        b.iter(|| {
            black_box(atomic_polynomial_product_coefficient(
                black_box(&left),
                black_box(&right),
                black_box(7),
            ))
        });
    });
    c.bench_function("atom_cofcon_mix", |b| {
        b.iter(|| {
            black_box(atomic_convergence_mix(
                black_box(0.5),
                black_box(0.3),
                black_box(0.2),
            ))
        });
    });
    c.bench_function("atom_dentfa_density", |b| {
        b.iter(|| {
            black_box(thomas_fermi_density_potential(
                black_box(0.45),
                black_box(29.0),
                black_box(-1.0),
            ))
        });
    });

    let mut occupations = vec![0.0; 41];
    let mut kappas = vec![1; 41];
    occupations[1] = 1.5;
    occupations[4] = 3.0;
    kappas[1] = -1;
    kappas[4] = -3;
    c.bench_function("atom_fdmocc_same_orbital", |b| {
        b.iter(|| {
            black_box(atomic_occupation_product(
                black_box(&occupations),
                black_box(&kappas),
                black_box(4),
                black_box(4),
            ))
        });
    });

    let muatco_kappas = [-1, 1, -2, 2, -3];
    let muatco_occupations = [2.0, 1.5, 3.0, 0.5, 4.0];
    let muatco_valence = [0.0, 0.5, 0.0, 0.25, 0.0];
    c.bench_function("atom_muatco_coefficients_5_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_coulomb_coefficients(black_box(
                AtomicCoulombCoefficientInput {
                    kappas: &muatco_kappas,
                    occupations: &muatco_occupations,
                    valence_occupations: &muatco_valence,
                },
            )))
        });
    });
    let muatco_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &muatco_kappas,
        occupations: &muatco_occupations,
        valence_occupations: &muatco_valence,
    });
    if let Ok(muatco_coefficients) = muatco_coefficients {
        let lagdat_shell_markers = [-1, 1, 1, 1, -1];
        c.bench_function("atom_lagdat_parameters_5_orbitals", |b| {
            b.iter(|| {
                black_box(atomic_lagrange_parameters(
                    black_box(AtomicLagrangeParametersInput {
                        active_orbital_1based: None,
                        include_exchange: true,
                        kappas: &muatco_kappas,
                        occupations: &muatco_occupations,
                        shell_markers: &lagdat_shell_markers,
                        coulomb_coefficients: muatco_coefficients.view(),
                    }),
                    |request: AtomicRadialIntegralRequest| {
                        Ok(0.0001 * (request.rank + 1) as f64
                            + 0.001 * request.first_left as f64
                            + 0.0002 * request.first_right as f64
                            + 0.00003 * request.second_left as f64
                            + 0.000004 * request.second_right as f64)
                    },
                ))
            });
        });
    }

    let tabrat_principal = [1, 2, 2, 3, 3];
    let tabrat_kappas = [-1, -1, 1, -2, 1];
    let tabrat_occupations = [2.0, 1.5, 0.5, 3.0, 0.25];
    let tabrat_energies = [-0.70, -0.25, -0.18, -0.09, -0.04];
    c.bench_function("atom_tabrat_tabulation_5_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_tabulation(
                black_box(AtomicTabulationInput {
                    principal_quantum_numbers: &tabrat_principal,
                    kappas: &tabrat_kappas,
                    occupations: &tabrat_occupations,
                    orbital_energies: &tabrat_energies,
                }),
                |request: AtomicTabulationIntegralRequest| {
                    Ok(0.01 * (request.left + 1) as f64
                        + 0.02 * (request.right + 1) as f64
                        + 0.001 * request.power as f64
                        + 0.1)
                },
            ))
        });
    });

    let fpf0_radial_count = 251;
    let fpf0_orbital_count = 5;
    let fpf0_radial_step = 0.05;
    let fpf0_radii = Array1::from_shape_fn(fpf0_radial_count, |index| {
        (-8.8 + fpf0_radial_step * index as f64).exp()
    });
    let fpf0_density = Array1::from_shape_fn(fpf0_radial_count, |index| {
        0.3 * (-0.7 * fpf0_radii[index]).exp() + 0.01 * (index + 1).rem_euclid(7) as f64
    });
    let fpf0_initial_large = Array1::from_shape_fn(fpf0_radial_count, |index| {
        0.2 * (-0.4 * fpf0_radii[index]).exp() + 0.001 * (index + 1) as f64
    });
    let fpf0_initial_small = Array1::from_shape_fn(fpf0_radial_count, |index| {
        -0.05 * (-0.3 * fpf0_radii[index]).exp() + 0.0002 * (index + 1) as f64
    });
    let fpf0_large =
        Array2::from_shape_fn((fpf0_radial_count, fpf0_orbital_count), |(row, col)| {
            let orbital = (col + 1) as f64;
            (0.03 * orbital + 0.0007 * (row + 1) as f64) * (-0.05 * orbital * fpf0_radii[row]).exp()
        });
    let fpf0_small =
        Array2::from_shape_fn((fpf0_radial_count, fpf0_orbital_count), |(row, col)| {
            let orbital = (col + 1) as f64;
            (-0.01 * orbital + 0.0003 * (row + 1) as f64)
                * (-0.03 * orbital * fpf0_radii[row]).exp()
        });
    let fpf0_occupations = [2.0, 2.0, 1.5, 0.5, 0.0];
    let fpf0_energies = [-0.85, -0.55, -0.21, -0.08, 0.04];
    let fpf0_kappas = [-1, 1, -2, 2, -1];
    c.bench_function("atom_fpf0_form_factor_5_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_form_factor(black_box(AtomicFormFactorInput {
                atomic_number: 26,
                hole_orbital_1based: 2,
                radial_step: fpf0_radial_step,
                total_energy: -2.345,
                radii: fpf0_radii.view(),
                density_4pi: fpf0_density.view(),
                initial_large_component: fpf0_initial_large.view(),
                initial_small_component: fpf0_initial_small.view(),
                large_components: fpf0_large.view(),
                small_components: fpf0_small.view(),
                occupations: &fpf0_occupations,
                orbital_energies: &fpf0_energies,
                kappas: &fpf0_kappas,
            })))
        });
    });

    let ortdat_kappas = [-1, -1, 1, -1];
    let ortdat_active_lengths = [3, 4, 3, 5];
    let ortdat_powers = [0.1, 0.2, 0.3, 0.4];
    let ortdat_large_components = Array2::from_shape_fn((5, 4), |(row, orbital)| {
        0.07 * (row + 1) as f64 + 0.11 * (orbital + 1) as f64
    });
    let ortdat_small_components = Array2::from_shape_fn((5, 4), |(row, orbital)| {
        0.03 * (row + 1) as f64 - 0.02 * (orbital + 1) as f64
    });
    let ortdat_large_coefficients = Array2::from_shape_fn((4, 4), |(row, orbital)| {
        0.2 * (row + 1) as f64 + 0.05 * (orbital + 1) as f64
    });
    let ortdat_small_coefficients = Array2::from_shape_fn((4, 4), |(row, orbital)| {
        -0.03 * (row + 1) as f64 + 0.04 * (orbital + 1) as f64
    });
    c.bench_function("atom_ortdat_schmidt_4_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_schmidt_orthogonalization(
                black_box(AtomicSchmidtOrthogonalizationInput {
                    active_orbital_1based: None,
                    kappas: &ortdat_kappas,
                    active_lengths: &ortdat_active_lengths,
                    orbital_powers: &ortdat_powers,
                    large_components: ortdat_large_components.view(),
                    small_components: ortdat_small_components.view(),
                    large_coefficients: ortdat_large_coefficients.view(),
                    small_coefficients: ortdat_small_coefficients.view(),
                }),
                |request| match request {
                    AtomicSchmidtIntegralRequest::Projection(request) => Ok(request
                        .target_large
                        .iter()
                        .zip(request.reference_large.iter())
                        .map(|(&target, &reference)| target * reference)
                        .sum::<f64>()
                        + request
                            .target_small
                            .iter()
                            .zip(request.reference_small.iter())
                            .map(|(&target, &reference)| target * reference)
                            .sum::<f64>()),
                    AtomicSchmidtIntegralRequest::Norm(request) => Ok(request
                        .target_large
                        .iter()
                        .map(|&value| value * value)
                        .sum::<f64>()
                        + request
                            .target_small
                            .iter()
                            .map(|&value| value * value)
                            .sum::<f64>()),
                },
            ))
        });
    });

    let coefficients = Array3::from_shape_fn((41, 41, 5), |(row, column, channel)| {
        1000.0 * (row + 1) as f64 + 10.0 * (column + 1) as f64 + channel as f64
    });
    c.bench_function("atom_akeato_lookup", |b| {
        b.iter(|| {
            black_box(atomic_direct_coulomb_coefficient(
                black_box(coefficients.view()),
                black_box(4),
                black_box(1),
                black_box(4),
            ))
        });
    });
    c.bench_function("atom_bkeato_lookup", |b| {
        b.iter(|| {
            black_box(atomic_exchange_coulomb_coefficient(
                black_box(coefficients.view()),
                black_box(1),
                black_box(4),
                black_box(4),
            ))
        });
    });
    c.bench_function("atom_bkmrdf_coefficients", |b| {
        b.iter(|| {
            black_box(atomic_breit_angular_coefficients(
                black_box(2),
                black_box(-4),
                black_box(3),
            ))
        });
    });
    let total_kappas = [-1, 1, -2, 2];
    let total_occupations = [2.0, 1.5, 3.0, 0.5];
    let total_valence = [0.0, 0.0, 1.0, 0.0];
    let total_energies = [-0.7, -0.3, -0.12, -0.05];
    let total_coefficients = Array3::from_shape_fn((4, 4, 6), |(row, column, channel)| {
        0.01 * (100 * (row + 1) + 10 * (column + 1) + channel + 1) as f64
    });
    c.bench_function("atom_etotal_accumulation_4_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_total_energy(
                black_box(AtomicTotalEnergyInput {
                    kappas: &total_kappas,
                    occupations: &total_occupations,
                    valence_occupations: &total_valence,
                    orbital_energies: &total_energies,
                    coulomb_coefficients: total_coefficients.view(),
                }),
                |request: AtomicRadialIntegralRequest| {
                    Ok(0.0001 * (request.rank + 1) as f64
                        + 0.001 * request.first_left as f64
                        + 0.0002 * request.first_right as f64
                        + 0.00003 * request.second_left as f64
                        + 0.000004 * request.second_right as f64)
                },
            ))
        });
    });
    let s02_kappas = [-1, -1, 1, 1, -2, -3];
    let s02_occupations = [2.0, 1.0, 1.5, 0.5, 3.0, 2.5];
    let mut s02_overlaps =
        Array2::from_shape_fn((6, 6), |(row, column)| 0.02 * (row + column + 2) as f64);
    for index in 0..6 {
        s02_overlaps[(index, index)] = 1.0;
    }
    s02_overlaps[(0, 1)] = 0.91;
    s02_overlaps[(1, 0)] = 0.91;
    s02_overlaps[(2, 3)] = 0.82;
    s02_overlaps[(3, 2)] = 0.82;
    c.bench_function("atom_s02at_overlap_reduction_6_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_overlap_amplitude_reduction(black_box(
                AtomicOverlapAmplitudeReductionInput {
                    hole_orbital_1based: Some(4),
                    kappas: &s02_kappas,
                    occupations: &s02_occupations,
                    overlap_integrals: s02_overlaps.view(),
                },
            )))
        });
    });

    let xsph_kind = array![2, 4, 2, -3, 4, 5, -3, 2];
    let xsph_orbital_l = array![1, 2, 3, 1, 4, 0, 5, 6];
    let xsph_final_lj = array![2, 1, 5, 3, 4, 0, 6, 1];
    let xsph_index_map = array![1, 2, -1, 3, -2, 4, -3, -1];
    c.bench_function("xsph_mincalc_plan", |b| {
        b.iter(|| {
            black_box(xsph_minimize_calculations(
                black_box(xsph_kind.view()),
                black_box(xsph_orbital_l.view()),
                black_box(xsph_final_lj.view()),
                black_box(8),
            ))
        });
    });
    c.bench_function("xsph_ljneeded0_flags", |b| {
        b.iter(|| {
            black_box(xsph_lj_needed_flags(
                black_box(6),
                black_box(xsph_final_lj.view()),
                black_box(xsph_index_map.view()),
                black_box(8),
                black_box(1),
            ))
        });
    });
    c.bench_function("xsph_xmultjas_factor", |b| {
        b.iter(|| {
            black_box(xsph_longitudinal_multipole_factor(
                black_box(3),
                black_box(-2),
                black_box(2),
            ))
        });
    });
    c.bench_function("xsph_xmult_factors", |b| {
        b.iter(|| {
            black_box(xsph_relativistic_multipole_factors(
                black_box(-3),
                black_box(2),
                black_box(1),
                black_box(2),
            ))
        });
    });
    c.bench_function("xsph_acoef_lx4", |b| {
        b.iter(|| {
            black_box(xsph_angular_density_coefficients(
                black_box(1),
                black_box(4),
            ))
        });
    });
    let xsph_lgind = array![0, 1, 2, 1, 3, 2, 4];
    let xsph_ljind = array![0, 1, 1, 2, 2, 3, 3];
    c.bench_function("xsph_bcoefjas_weights", |b| {
        b.iter(|| {
            black_box(xsph_nrixs_transition_weights(
                black_box(-1),
                black_box(1),
                black_box(4),
                black_box(9),
                black_box(3),
                black_box(xsph_lgind.view()),
                black_box(xsph_ljind.view()),
                black_box(7),
            ))
        });
    });
    let xsph_spec_index_map = array![1, -1, 2, 1, -2];
    let xsph_spec_lind = array![0, 1, 2, 3, 4];
    let xsph_spec_ljind = array![0, 1, 2, 3, 1];
    let xsph_spec_radial = array![
        Complex::new(0.12, -0.03),
        Complex::new(-0.08, 0.19),
        Complex::new(0.31, 0.07),
        Complex::new(-0.22, -0.11)
    ];
    let xsph_spec_qweights = array![Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)];
    let xsph_spec_cosines = arr2(&[[0.25, -0.35], [0.60, -0.40]]);
    let xsph_spec_hbmat = Array3::from_shape_fn((2, 5, 4).f(), |(spin, state, magnetic)| {
        let state_feff = state as f64 + 1.0;
        let magnetic_j2 = [-3.0, -1.0, 1.0, 3.0][magnetic];
        0.05 * state_feff
            + 0.11 * spin as f64
            + 0.017 * magnetic_j2
            + 0.003 * state_feff * magnetic_j2
    });
    c.bench_function("xsph_specupdlg_update", |b| {
        b.iter(|| {
            let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
            black_box(xsph_update_nrixs_lg_spectrum(
                XsphLgSpectrumUpdateInput {
                    calculation_index: black_box(1),
                    spin_index: black_box(1),
                    index_map: black_box(xsph_spec_index_map.view()),
                    orbital_l: black_box(xsph_spec_lind.view()),
                    final_lj: black_box(xsph_spec_ljind.view()),
                    initial_j2: black_box(3),
                    transition_weights: black_box(xsph_spec_hbmat.view()),
                    radial_integrals: black_box(xsph_spec_radial.view()),
                    q_weights: black_box(xsph_spec_qweights.view()),
                    q_cosines: black_box(xsph_spec_cosines.view()),
                    mix_dff: black_box(false),
                    mdff_mode: black_box(0),
                    ljmax: black_box(3),
                    active_len: black_box(5),
                    mode: XsphSpectrumUpdateMode::Regular,
                },
                black_box(spectrum.view_mut()),
            ))
        });
    });
    c.bench_function("xsph_specupd_lj_update", |b| {
        b.iter(|| {
            let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
            let mut spectrum_norm = 0.02;
            black_box(xsph_update_nrixs_lj_spectrum(
                XsphLjSpectrumUpdateInput {
                    calculation_index: black_box(1),
                    spin_index: black_box(1),
                    index_map: black_box(xsph_spec_index_map.view()),
                    final_lj: black_box(xsph_spec_ljind.view()),
                    initial_j2: black_box(3),
                    transition_weights: black_box(xsph_spec_hbmat.view()),
                    radial_integrals: black_box(xsph_spec_radial.view()),
                    q_weights: black_box(xsph_spec_qweights.view()),
                    q_cosines: black_box(xsph_spec_cosines.view()),
                    mix_dff: black_box(false),
                    mdff_mode: black_box(0),
                    ljmax: black_box(3),
                    active_len: black_box(5),
                    mode: XsphSpectrumUpdateMode::Regular,
                },
                black_box(spectrum.view_mut()),
                black_box(&mut spectrum_norm),
            ))
        });
    });
    c.bench_function("xsph_specupdatom_update", |b| {
        b.iter(|| {
            let mut spectrum = Array1::from_elem(5, Complex::new(0.02, 0.01));
            let mut spectrum_norm = 0.005;
            black_box(xsph_update_nrixs_atom_spectrum(
                XsphLjSpectrumUpdateInput {
                    calculation_index: black_box(2),
                    spin_index: black_box(1),
                    index_map: black_box(xsph_spec_index_map.view()),
                    final_lj: black_box(xsph_spec_ljind.view()),
                    initial_j2: black_box(3),
                    transition_weights: black_box(xsph_spec_hbmat.view()),
                    radial_integrals: black_box(xsph_spec_radial.view()),
                    q_weights: black_box(xsph_spec_qweights.view()),
                    q_cosines: black_box(xsph_spec_cosines.view()),
                    mix_dff: black_box(false),
                    mdff_mode: black_box(0),
                    ljmax: black_box(3),
                    active_len: black_box(5),
                    mode: XsphSpectrumUpdateMode::Regular,
                },
                black_box(spectrum.view_mut()),
                black_box(&mut spectrum_norm),
            ))
        });
    });
    let xsph_axafs_energies = Array1::from_shape_fn(64, |index| {
        let i = index as f64 + 1.0;
        Complex::new(0.015 * (i - 3.0).powi(2) + 0.012 * (i - 1.0), 0.002 * i)
    });
    let xsph_axafs_xsec = Array1::from_shape_fn(64, |index| {
        let i = index as f64 + 1.0;
        Complex::new(
            -0.03 * i,
            0.42 + 0.021 * i + 0.004 * i * i + 0.025 * (0.7 * i).sin(),
        )
    });
    c.bench_function("xsph_axafs_table", |b| {
        b.iter(|| {
            black_box(xsph_axafs(black_box(XsphAxafsInput {
                energies: black_box(xsph_axafs_energies.view()),
                cross_section: black_box(xsph_axafs_xsec.view()),
                fermi_energy: black_box(0.37),
                horizontal_count: black_box(48),
                zero_wave_index: black_box(2),
            })))
        });
    });
    c.bench_function("xsph_getoccnorm", |b| {
        b.iter(|| black_box(xsph_occupation_normalization(black_box(92), black_box(22))));
    });
    let mut xsph_hole_large = Array1::<f64>::zeros(251);
    let mut xsph_hole_small = Array1::<f64>::zeros(251);
    for index in 0..15 {
        let i = index as f64 + 1.0;
        xsph_hole_large[index] = 0.1 + 0.017 * i + 0.0009 * i * i + 0.002 * (0.3 * i).sin();
        xsph_hole_small[index] = -0.04 + 0.011 * i - 0.0004 * i * i + 0.001 * (0.25 * i).cos();
    }
    c.bench_function("xsph_getholeorb0", |b| {
        b.iter(|| {
            black_box(xsph_initial_hole_orbital(black_box(XsphHoleOrbitalInput {
                large_component: black_box(xsph_hole_large.view()),
                small_component: black_box(xsph_hole_small.view()),
                original_step: black_box(0.05),
                new_step: black_box(0.035),
                output_count: black_box(64),
                output_capacity: black_box(96),
            })))
        });
    });
    let xsph_phase_sort_input = arr1(&[
        Complex::new(0.002, 9.0),
        Complex::new(-0.004, 8.0),
        Complex::new(0.0004, 7.0),
        Complex::new(0.0012, 6.0),
        Complex::new(-0.0036, 5.0),
        Complex::new(0.25, 4.0),
    ]);
    let xsph_user_phase_points = arr1(&[
        Complex::new(-5.0, 0.2),
        Complex::new(0.0004, 0.0),
        Complex::new(12.0, -0.1),
    ]);
    let xsph_user_phase_records = [
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::Energy,
            minimum: XsphPhaseUserGridMinimum::Value(-2.0),
            maximum: 2.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::WaveNumber,
            minimum: XsphPhaseUserGridMinimum::Last,
            maximum: 3.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::User(xsph_user_phase_points.view()),
    ];
    c.bench_function("xsph_phmesh2_primitives", |b| {
        b.iter(|| {
            let even = black_box(xsph_even_energy_mesh(
                black_box(-0.2),
                black_box(0.35),
                black_box(0.11),
                black_box(64),
            ));
            let k_mesh = black_box(xsph_k_energy_mesh(
                black_box(-1.2),
                black_box(-0.2),
                black_box(0.25),
                black_box(64),
            ));
            let exp_mesh = black_box(xsph_exponential_energy_mesh(
                black_box(0.02),
                black_box(0.5),
                black_box(0.4),
                black_box(64),
            ));
            let vertical = black_box(xsph_vertical_energy_mesh_84(black_box(0.05), black_box(64)));
            let exafs84 = black_box(xsph_exafs_energy_grid_84(
                black_box(18.0 * FEFF_BOHR_ANGSTROM),
                black_box(100),
            ));
            let xanes84 = black_box(xsph_xanes_energy_grid_84(
                black_box(4.0 * FEFF_BOHR_ANGSTROM),
                black_box(0.5 * FEFF_BOHR_ANGSTROM),
                black_box(0.02),
                black_box(80),
            ));
            let fprime84 = black_box(xsph_fprime_energy_grid_84(
                black_box(-5.0),
                black_box(10.0),
                black_box(0.25),
                black_box(9.0),
                black_box(-0.4),
                black_box(64),
            ));
            let xes84 = black_box(xsph_xes_energy_grid_84(
                black_box(-5.0),
                black_box(10.0),
                black_box(0.25),
                black_box(-0.4),
                black_box(64),
            ));
            let phase84 = black_box(xsph_phase_energy_mesh_84(black_box(
                XsphPhaseEnergyMesh84Input {
                    spectroscopy: 1,
                    edge: -0.4,
                    reference_energy: 9.0,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    core_valence_separation: -1.5,
                    max_wave_number: 18.0 * FEFF_BOHR_ANGSTROM,
                    wave_number_step: 0.5 * FEFF_BOHR_ANGSTROM,
                    xanes_energy_step: 0.02,
                    capacity: 120,
                },
            )));
            let no_fms_phase84 = black_box(xsph_phase_energy_mesh_84(black_box(
                XsphPhaseEnergyMesh84Input {
                    spectroscopy: -1,
                    edge: -0.4,
                    reference_energy: 9.0,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    core_valence_separation: -1.5,
                    max_wave_number: 18.0 * FEFF_BOHR_ANGSTROM,
                    wave_number_step: 0.5 * FEFF_BOHR_ANGSTROM,
                    xanes_energy_step: 0.02,
                    capacity: 120,
                },
            )));
            let user_phase = black_box(xsph_phase_energy_mesh_user(black_box(
                XsphPhaseUserGridInput {
                    spectroscopy: 1,
                    edge: -0.4,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    records: &xsph_user_phase_records,
                    capacity: 120,
                },
            )));
            let thermal_phase = black_box(xsph_thermal_phase_energy_mesh(black_box(
                XsphThermalPhaseEnergyMeshInput {
                    edge: -0.4,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    core_valence_separation: -1.5,
                    electronic_temperature: 5.0,
                    user_records: Some(&xsph_user_phase_records),
                    capacity: 240,
                },
            )));
            let reversed = black_box(xsph_reverse_energy_grid(
                black_box(xsph_phase_sort_input.view()),
                black_box(0.25),
            ));
            let sorted = black_box(xsph_sort_energy_grid(black_box(
                xsph_phase_sort_input.view(),
            )));
            black_box((
                even,
                k_mesh,
                exp_mesh,
                vertical,
                exafs84,
                xanes84,
                fprime84,
                xes84,
                phase84,
                no_fms_phase84,
                user_phase,
                thermal_phase,
                reversed,
                sorted,
            ))
        });
    });
    let xsph_radii = array![0.1, 1.0, 3.0, 20.0, 40.0, 80.0];
    c.bench_function("xsph_qbesselget_table", |b| {
        b.iter(|| {
            black_box(xsph_q_bessel_table(
                black_box(0.35),
                black_box(xsph_radii.view()),
                black_box(6),
            ))
        });
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
    let epsilon_tables = sample_epsilon_tables();
    let epsilon_weights = [1.0, 0.35, 0.15];
    c.bench_function("opcons_combine_epsilon_tables_3x160", |b| {
        b.iter(|| {
            black_box(combine_epsilon_tables(
                black_box(&epsilon_tables),
                black_box(&epsilon_weights),
            ))
        });
    });
    let fullspectrum_energy = Array1::from_shape_fn(4096, |index| 5.0 + 0.05 * index as f64);
    let fullspectrum_epsilon2 = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        0.2 + 0.05 * x.sin().abs()
    });
    let fullspectrum_epsilon = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        Complex::new(0.1 + 0.02 * x.cos(), fullspectrum_epsilon2[index])
    });
    let fullspectrum_refractive = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        Complex::new(0.02 + 0.005 * x.sin(), 0.01 + 0.002 * x.cos())
    });
    let fullspectrum_absorption = Array1::from_shape_fn(4096, |index| {
        1000.0 + 5.0 * index as f64 + 20.0 * (index as f64 * 0.005).sin()
    });
    let fullspectrum_valence_energy_ev =
        Array1::from_shape_fn(4096, |index| (5.0 + 0.05 * index as f64) * 27.211_396);
    let fullspectrum_valence_absorption = Array1::from_shape_fn(4096, |index| {
        0.5 + 0.01 * index as f64 + 0.05 * (index as f64 * 0.01).cos()
    });
    let fullspectrum_background_energy_ev =
        Array1::from_shape_fn(2048, |index| (5.0 + 0.2 * index as f64) * 27.211_396);
    let fullspectrum_background_f_prime = Array1::from_shape_fn(2048, |index| {
        1.0 + 0.002 * index as f64 + 0.05 * (index as f64 * 0.01).sin()
    });
    let fullspectrum_background_f_double_prime = Array1::from_shape_fn(2048, |index| {
        0.05 + 0.0005 * index as f64 + 0.01 * (index as f64 * 0.02).cos().abs()
    });
    let fullspectrum_background_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: fullspectrum_background_energy_ev.view(),
        f_prime: fullspectrum_background_f_prime.view(),
        f_double_prime: fullspectrum_background_f_double_prime.view(),
    }];
    let fullspectrum_epsilon_minus_one = Array1::from_shape_fn(4096, |index| {
        Complex::new(
            0.2 + 0.01 * (index as f64 * 0.01).sin(),
            0.4 + 0.05 * (index as f64 * 0.02).cos().abs(),
        )
    });
    let fullspectrum_scattering_factor = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        Complex::new(1.0 + 0.02 * x.sin(), 0.3 + 0.04 * x.cos().abs())
    });
    let fullspectrum_background_scattering_factor =
        fullspectrum_scattering_factor.mapv(|value| value * Complex::new(0.85, 0.02));
    let fullspectrum_fine_fms_energy_ev =
        Array1::from_shape_fn(2048, |index| (5.0 + 0.05 * index as f64) * 27.211_396);
    let fullspectrum_fine_path_energy_ev =
        Array1::from_shape_fn(2048, |index| (8.0 + 0.08 * index as f64) * 27.211_396);
    let fullspectrum_fine_fms_wave =
        Array1::from_shape_fn(2048, |index| 0.25 + 0.003 * index as f64);
    let fullspectrum_fine_path_wave =
        Array1::from_shape_fn(2048, |index| 3.0 + 0.004 * index as f64);
    let fullspectrum_fine_real_fms = Array1::from_shape_fn(2048, |index| {
        1.0 + 0.002 * index as f64 + 0.02 * (index as f64 * 0.01).sin()
    });
    let fullspectrum_fine_real_path = Array1::from_shape_fn(2048, |index| {
        1.5 + 0.0025 * index as f64 + 0.03 * (index as f64 * 0.015).cos()
    });
    let fullspectrum_fine_imag_fms = Array1::from_shape_fn(2048, |index| {
        0.05 + 0.001 * index as f64 + 0.01 * (index as f64 * 0.02).sin().abs()
    });
    let fullspectrum_fine_imag_path = Array1::from_shape_fn(2048, |index| {
        0.08 + 0.0012 * index as f64 + 0.01 * (index as f64 * 0.02).cos().abs()
    });
    let fullspectrum_fine_real_fms_background =
        fullspectrum_fine_real_fms.mapv(|value| value * 0.85);
    let fullspectrum_fine_real_path_background =
        fullspectrum_fine_real_path.mapv(|value| value * 0.9);
    let fullspectrum_fine_imag_fms_background =
        fullspectrum_fine_imag_fms.mapv(|value| value * 0.8);
    let fullspectrum_fine_imag_path_background =
        fullspectrum_fine_imag_path.mapv(|value| value * 0.82);
    let fullspectrum_atomic_numbers = array![29_usize, 8, 29, 14, 8, 29];
    let fullspectrum_multiplicities = array![0.01, 2.0, 3.0, 1.0, 4.0, 2.0];
    let fullspectrum_norman_radii = array![2.0, 1.5, 2.5, 1.8, 1.6, 2.2];
    let fullspectrum_occupations = Array1::from_shape_fn(40, |index| match index {
        0 => 2.0,
        1 => 1.0,
        2 => 2.0,
        3 => 1.0,
        4 => 0.5,
        _ => 0.0,
    });
    let fullspectrum_edge_onsets = Array1::from_shape_fn(40, |index| 0.2 + 0.1 * index as f64);
    let fullspectrum_edges = array![25.0, 75.0, 140.0, 210.0];
    let fullspectrum_elam_components = [29, 8, 79];
    let fullspectrum_default_edges = [
        FullSpectrumDefaultGridEdge {
            atomic_number: 8,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 1,
            fine_structure: true,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 4,
            fine_structure: false,
        },
    ];
    c.bench_function("fullspectrum_elam_edge_adapter", |b| {
        b.iter(|| {
            black_box(full_spectrum_elam_edge_energies(black_box(
                &fullspectrum_elam_components,
            )))
        });
    });
    c.bench_function("fullspectrum_rdop_default_grid", |b| {
        b.iter(|| {
            black_box(full_spectrum_default_energy_grid(black_box(
                &fullspectrum_default_edges,
            )))
        });
    });
    c.bench_function("fullspectrum_edge_grid_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_edge_energy_grid(black_box(
                FullSpectrumEdgeGridInput {
                    min_energy: 0.0,
                    max_energy: 250.0,
                    edge_energies: fullspectrum_edges.view(),
                    wave_number_step: 0.2,
                    max_points: 4096,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_linear_grid_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_linear_energy_grid(black_box(
                FullSpectrumLinearGridInput {
                    point_count: 4096,
                    min_energy: 0.0,
                    max_energy: 250.0,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_qsum_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_effective_electron_count(black_box(
                FullSpectrumQSumInput {
                    number_density: 0.075,
                    epsilon2: fullspectrum_epsilon2.view(),
                    omega: fullspectrum_energy.view(),
                    active_len: fullspectrum_energy.len(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_valence_epsilon2_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_valence_epsilon2(black_box(
                FullSpectrumValenceInput {
                    number_density: 0.075,
                    omega: fullspectrum_energy.view(),
                    source_energy_ev: fullspectrum_valence_energy_ev.view(),
                    source_absorption_angstrom2: fullspectrum_valence_absorption.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_background_from_fprime_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_background_from_fprime(black_box(
                FullSpectrumBackgroundInput {
                    omega: fullspectrum_energy.view(),
                    segments: &fullspectrum_background_segments,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_optical_constants_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_optical_constants(black_box(
                FullSpectrumOpticalConstantsInput {
                    omega: fullspectrum_energy.view(),
                    epsilon_minus_one: fullspectrum_epsilon_minus_one.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_scattering_to_dielectric_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_scattering_to_dielectric(black_box(
                FullSpectrumScatteringDielectricInput {
                    number_density: 0.075,
                    omega: fullspectrum_energy.view(),
                    scattering_factor: fullspectrum_scattering_factor.view(),
                    background_scattering_factor: fullspectrum_background_scattering_factor.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_fine_structure_from_segments_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_fine_structure_from_segments(black_box(
                FullSpectrumFineStructureInput {
                    omega: fullspectrum_energy.view(),
                    real_fms: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                        scattering_factor: fullspectrum_fine_real_fms.view(),
                        background: fullspectrum_fine_real_fms_background.view(),
                    },
                    real_path: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                        scattering_factor: fullspectrum_fine_real_path.view(),
                        background: fullspectrum_fine_real_path_background.view(),
                    },
                    imaginary_fms: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                        scattering_factor: fullspectrum_fine_imag_fms.view(),
                        background: fullspectrum_fine_imag_fms_background.view(),
                    },
                    imaginary_path: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                        scattering_factor: fullspectrum_fine_imag_path.view(),
                        background: fullspectrum_fine_imag_path_background.view(),
                    },
                    low_wave_number: 3.0,
                    high_wave_number: 4.0,
                },
            )))
        });
    });
    let fullspectrum_background_result =
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: fullspectrum_energy.view(),
            segments: &fullspectrum_background_segments,
        });
    let fullspectrum_fine_structure_result =
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: fullspectrum_energy.view(),
            real_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                scattering_factor: fullspectrum_fine_real_fms.view(),
                background: fullspectrum_fine_real_fms_background.view(),
            },
            real_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                scattering_factor: fullspectrum_fine_real_path.view(),
                background: fullspectrum_fine_real_path_background.view(),
            },
            imaginary_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                scattering_factor: fullspectrum_fine_imag_fms.view(),
                background: fullspectrum_fine_imag_fms_background.view(),
            },
            imaginary_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                scattering_factor: fullspectrum_fine_imag_path.view(),
                background: fullspectrum_fine_imag_path_background.view(),
            },
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        });
    if let (Ok(background_result), Ok(fine_structure_result)) = (
        fullspectrum_background_result,
        fullspectrum_fine_structure_result,
    ) {
        c.bench_function("fullspectrum_assemble_edge_4096", |b| {
            b.iter(|| {
                black_box(full_spectrum_assemble_edge(black_box(
                    FullSpectrumEdgeAssemblyInput {
                        omega: fullspectrum_energy.view(),
                        background: &background_result,
                        fine_structure: &fine_structure_result,
                        transition_size: 0.05,
                    },
                )))
            });
        });
    }
    c.bench_function("fullspectrum_number_density", |b| {
        b.iter(|| {
            black_box(full_spectrum_number_density(black_box(
                FullSpectrumNumberDensityInput {
                    target_atomic_number: 29,
                    atomic_numbers: fullspectrum_atomic_numbers.view(),
                    potential_multiplicities: fullspectrum_multiplicities.view(),
                    norman_radii: fullspectrum_norman_radii.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_edge_selection", |b| {
        b.iter(|| {
            black_box(full_spectrum_edges_from_occupations(black_box(
                FullSpectrumEdgeSelectionInput {
                    occupations: fullspectrum_occupations.view(),
                    edge_onsets_hartree: fullspectrum_edge_onsets.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_drude_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_drude_term(black_box(
                FullSpectrumDrudeInput {
                    omega: fullspectrum_energy.view(),
                    lifetime_seconds: 1.0e-15,
                    number_density: 0.075,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_kk_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_kramers_kronig(black_box(
                FullSpectrumKramersKronigInput {
                    omega: fullspectrum_energy.view(),
                    epsilon2: fullspectrum_epsilon2.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_hamaker_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_hamaker_transform(black_box(
                FullSpectrumHamakerInput {
                    omega: fullspectrum_energy.view(),
                    epsilon: fullspectrum_epsilon.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_sumrules_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_sum_rules(black_box(
                FullSpectrumSumRulesInput {
                    number_density: 0.075,
                    energy_ev: fullspectrum_energy.view(),
                    epsilon_minus_one: fullspectrum_epsilon.view(),
                    refractive_index_minus_one: fullspectrum_refractive.view(),
                    absorption_coefficient: fullspectrum_absorption.view(),
                },
            )))
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
    let sfconv_energy = Array1::from_shape_fn(110, |index| {
        let i = index as f64;
        -2.0 + 0.018 * i + 0.000_11 * i * i
    });
    let sfconv_spectral = Array2::from_shape_fn((8, 110).f(), |(row, column)| {
        let fortran_row = row as f64 + 1.0;
        let i = column as f64;
        0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
    });
    let sfconv_pole_energy = Array1::from_shape_fn(5, |index| {
        let i = index as f64 + 1.0;
        0.12 * i + 0.015 * i * i
    });
    let sfconv_pole_weight = Array1::from_shape_fn(5, |index| 0.25 + 0.07 * (index as f64 + 1.0));
    let sfconv_pole_broadening = Array1::from_shape_fn(5, |index| {
        let i = index as f64 + 1.0;
        0.01 * i + 0.002 * i * i
    });
    c.bench_function("sfconv_plset_ppset", |b| {
        b.iter(|| {
            let pole = black_box(sfconv_select_pole(
                black_box(3),
                black_box(sfconv_pole_energy.view()),
                black_box(sfconv_pole_weight.view()),
                black_box(sfconv_pole_broadening.view()),
            ));
            let plasma = black_box(sfconv_plasma_parameters(black_box(2.35)));
            black_box((pole, plasma))
        });
    });
    c.bench_function("sfconv_ppole_qlimits", |b| {
        b.iter(|| {
            let dispersion = black_box(sfconv_pole_dispersion(
                black_box(0.35),
                black_box(0.47),
                black_box(0.28),
            ));
            let limits = black_box(sfconv_q_limits(
                black_box(1.15),
                black_box(1.05),
                black_box(0.47),
                black_box(0.28),
                black_box(12.0),
            ));
            let threshold = black_box(sfconv_plasmon_threshold_momentum(
                black_box(0.47),
                black_box(0.28),
                black_box(0.42),
                black_box(0.88),
            ));
            black_box((dispersion, limits, threshold))
        });
    });
    c.bench_function("sfconv_so2conv_momentum_grid", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_momentum_grid(
                black_box(0.816_663_103_267_026_7),
                black_box(1.733_25),
            ))
        });
    });
    c.bench_function("sfconv_so2conv_material_parameters", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_material_parameters(black_box(
                SfconvSo2convMaterialInput {
                    core_hole_width_ev: 1.729,
                    wigner_seitz_radius: 2.05,
                    interstitial_potential_ev: 12.34,
                    chemical_potential_ev: 18.76,
                    fermi_wave_number_inv_angstrom: 1.23,
                },
            )))
        });
    });
    let photoelectron_momentum_grid = array![0.0, 0.35, -0.40, 0.82, 1.10, 1.45];
    let photoelectron_self_energy = array![0.090, 0.105, 0.120, 0.150, 0.190, 0.250];
    c.bench_function("sfconv_so2conv_photoelectron_momentum", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_photoelectron_momentum(black_box(
                SfconvPhotoelectronMomentumInput {
                    momentum: photoelectron_momentum_grid.view(),
                    chemical_potential: 0.47,
                    fermi_momentum: 0.92,
                    fermi_level: 0.36,
                    fermi_self_energy: 0.115,
                    self_energy: photoelectron_self_energy.view(),
                },
            )))
        });
    });
    let exafs_padding_energy = array![0.10, 0.22, 0.37, 0.55];
    c.bench_function("sfconv_so2conv_pad_exafs_energy_grid", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_pad_exafs_energy_grid(black_box(
                SfconvSo2convExafsEnergyPaddingInput {
                    energy: exafs_padding_energy.view(),
                    active_len: 4,
                    output_len: 401,
                },
            )))
        });
    });
    let exafs_prep_count = 112;
    let exafs_prep_momentum = Array1::from_shape_fn(exafs_prep_count, |index| 0.02 * index as f64);
    let exafs_prep_magnitude =
        Array1::from_shape_fn(exafs_prep_count, |index| 1.0 + 0.001 * index as f64);
    let exafs_prep_phase = Array1::from_shape_fn(exafs_prep_count, |index| 0.02 * index as f64);
    c.bench_function("sfconv_so2conv_prepare_exafs_signal", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_prepare_exafs_signal(black_box(
                SfconvSo2convExafsPreparationInput {
                    momentum: exafs_prep_momentum.view(),
                    magnitude: exafs_prep_magnitude.view(),
                    phase: exafs_prep_phase.view(),
                    phase_minus_2kr: None,
                    chemical_potential: 0.5,
                    active_len: exafs_prep_count,
                    output_len: 401,
                },
            )))
        });
    });
    let xanes_prep_count = 112;
    let xanes_prep_incident = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        0.2 + 0.13 * (i - 1.0) + 0.002 * ((i as usize) % 3) as f64
    });
    let xanes_prep_energy = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        -0.4 + 0.11 * (i - 1.0) + 0.001 * ((i as usize) % 4) as f64
    });
    let xanes_prep_background = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        1.0 + 0.015 * (i - 1.0) + 0.0008 * ((i as usize) % 2) as f64
    });
    let xanes_prep_absorption = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        xanes_prep_background[index] + 0.04 * (0.31 * i).sin() + 0.002 * (i - 1.0)
    });
    c.bench_function("sfconv_so2conv_prepare_xanes_signal", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_prepare_xanes_signal(black_box(
                SfconvSo2convXanesPreparationInput {
                    incident_energy: xanes_prep_incident.view(),
                    excitation_energy: xanes_prep_energy.view(),
                    absorption: xanes_prep_absorption.view(),
                    embedded_background: xanes_prep_background.view(),
                    active_len: xanes_prep_count,
                    output_len: 401,
                },
            )))
        });
    });
    let momentum_spectral_grid = array![0.50, 1.00, 2.00, 4.00];
    let momentum_spectral_energy = array![
        [0.11, 0.12, 0.13, 0.14],
        [0.21, 0.22, 0.23, 0.24],
        [0.31, 0.32, 0.33, 0.34],
        [0.41, 0.42, 0.43, 0.44],
    ];
    let momentum_spectral_emsf = array![
        [1.11, 1.12, 1.13, 1.14],
        [1.21, 1.22, 1.23, 1.24],
        [1.31, 1.32, 1.33, 1.34],
        [1.41, 1.42, 1.43, 1.44],
    ];
    let momentum_spectral_essf = array![
        [2.22, 2.24, 2.26, 2.28],
        [2.42, 2.44, 2.46, 2.48],
        [2.62, 2.64, 2.66, 2.68],
        [2.82, 2.84, 2.86, 2.88],
    ];
    let momentum_spectral_xmsf = array![
        [3.33, 3.36, 3.39, 3.42],
        [3.63, 3.66, 3.69, 3.72],
        [3.93, 3.96, 3.99, 4.02],
        [4.23, 4.26, 4.29, 4.32],
    ];
    let momentum_spectral_xssf = array![
        [0.444, 0.448, 0.452, 0.456],
        [0.484, 0.488, 0.492, 0.496],
        [0.524, 0.528, 0.532, 0.536],
        [0.564, 0.568, 0.572, 0.576],
    ];
    let momentum_spectral_xissf = array![
        [0.555, 0.560, 0.565, 0.570],
        [0.605, 0.610, 0.615, 0.620],
        [0.655, 0.660, 0.665, 0.670],
        [0.705, 0.710, 0.715, 0.720],
    ];
    let momentum_spectral_escsf = array![
        [0.666, 0.672, 0.678, 0.684],
        [0.726, 0.732, 0.738, 0.744],
        [0.786, 0.792, 0.798, 0.804],
        [0.846, 0.852, 0.858, 0.864],
    ];
    let momentum_spectral_weights = array![
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
        [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
        [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
    ];
    let momentum_spectral_self = array![41.0, 42.0, 43.0, 44.0];
    let momentum_spectral_correction = array![51.0, 52.0, 53.0, 54.0];
    let momentum_spectral_width = array![61.0, 62.0, 63.0, 64.0];
    let momentum_spectral_z1 = array![71.0, 72.0, 73.0, 74.0];
    let momentum_spectral_z1i = array![81.0, 82.0, 83.0, 84.0];
    c.bench_function("sfconv_so2conv_momentum_spectral_interpolation", |b| {
        b.iter(|| {
            black_box(sfconv_interpolate_momentum_spectral_function(black_box(
                SfconvMomentumSpectralInterpolationInput {
                    photoelectron_momentum: 0.75,
                    momentum_grid: momentum_spectral_grid.view(),
                    energy_grid: momentum_spectral_energy.view(),
                    extrinsic_quasiparticle: momentum_spectral_emsf.view(),
                    extrinsic_satellite: momentum_spectral_essf.view(),
                    interference_quasiparticle: momentum_spectral_xmsf.view(),
                    interference_satellite: momentum_spectral_xssf.view(),
                    intrinsic_satellite: momentum_spectral_xissf.view(),
                    clipped_extrinsic_satellite: momentum_spectral_escsf.view(),
                    weights: momentum_spectral_weights.view(),
                    self_energy_real: momentum_spectral_self.view(),
                    energy_correction: momentum_spectral_correction.view(),
                    width: momentum_spectral_width.view(),
                    renormalization_real: momentum_spectral_z1.view(),
                    renormalization_imag: momentum_spectral_z1i.view(),
                },
            )))
        });
    });
    let path_interp_source = array![0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00];
    let path_interp_momentum = array![0.25, 0.75, 1.25, 1.75];
    let path_interp_central_phase = array![0.10, 0.20, 0.10, 0.30];
    let path_interp_amplitude = array![1.00, 1.40, 1.10, 1.80];
    let path_interp_phase = array![0.50, 0.70, 0.60, 1.00];
    let path_interp_reduction = array![0.80, 0.90, 0.85, 0.95];
    let path_interp_lambda = array![6.00, 7.00, 8.00, 9.00];
    c.bench_function("sfconv_so2conv_path_interpolation", |b| {
        b.iter(|| {
            black_box(sfconv_interpolate_feff_path(black_box(
                SfconvFeffPathInterpolationInput {
                    source_momentum: path_interp_source.view(),
                    path_momentum: path_interp_momentum.view(),
                    central_phase: path_interp_central_phase.view(),
                    effective_amplitude: path_interp_amplitude.view(),
                    effective_phase: path_interp_phase.view(),
                    reduction_factor: path_interp_reduction.view(),
                    mean_free_path: path_interp_lambda.view(),
                },
            )))
        });
    });
    let path_signal_central_phase = array![0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0];
    let path_signal_amplitude = array![0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0];
    let path_signal_phase = array![0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0];
    let path_signal_reduction = array![0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0];
    let path_signal_lambda = array![0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0];
    c.bench_function("sfconv_so2conv_path_signal", |b| {
        b.iter(|| {
            black_box(sfconv_feff_path_signal(black_box(
                SfconvFeffPathSignalInput {
                    momentum: path_interp_source.view(),
                    central_phase: path_signal_central_phase.view(),
                    effective_amplitude: path_signal_amplitude.view(),
                    effective_phase: path_signal_phase.view(),
                    reduction_factor: path_signal_reduction.view(),
                    mean_free_path: path_signal_lambda.view(),
                    degeneracy: 4.0,
                    half_path_length: 3.25,
                },
            )))
        });
    });
    c.bench_function("sfconv_so2conv_exafs_convolution", |b| {
        b.iter(|| {
            black_box(sfconv_exafs_convolution(black_box(
                SfconvExafsConvolutionInput {
                    real_convolution_amplitude: -1.494_388_190_129_498_7,
                    real_convolution_phase: 0.0,
                    imaginary_convolution_amplitude: -0.137_577_673_742_690_1,
                    imaginary_convolution_phase: 0.0,
                    original_magnitude: 1.7,
                    original_phase: 0.25,
                    phase_minus_2kr: 0.03,
                    previous_phase: 3.050_020_434_612_271,
                    phase_jump_count: 0,
                },
            )))
        });
    });
    c.bench_function("sfconv_so2conv_xanes_convolution", |b| {
        b.iter(|| {
            black_box(sfconv_xanes_convolution(black_box(
                SfconvXanesConvolutionInput {
                    asymmetric_phase: false,
                    absorption_convolution: 0.0,
                    embedded_background: 3.40,
                    fine_structure_imaginary_amplitude: 1.80,
                    fine_structure_imaginary_phase: 0.20,
                    fine_structure_real_amplitude: 0.70,
                    fine_structure_real_phase: 0.90,
                },
            )))
        });
    });
    c.bench_function("sfconv_grater_oscillatory", |b| {
        b.iter(|| {
            black_box(sfconv_grater_integrate(
                |x| Ok((5.0 * x).sin() / (1.0 + x * x)),
                black_box(0.0),
                black_box(4.0),
                black_box(1.0e-6),
                black_box(1.0e-6),
                black_box(&[]),
            ))
        });
    });
    c.bench_function("sfconv_mkspectf_energy_grid", |b| {
        b.iter(|| black_box(sfconv_spectral_energy_grid(black_box(0.62))));
    });
    let quasiparticle_peak_input = SfconvQuasiparticlePeakInput {
        center_energy: -0.000_206_666_666_666_666_66,
        lower_boundary: -0.000_31,
        upper_boundary: 0.0,
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.9348,
        quasiparticle_width: 0.0656,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
    };
    c.bench_function("sfconv_mkspectf_quasiparticle_peak", |b| {
        b.iter(|| {
            black_box(sfconv_quasiparticle_main_peak(black_box(
                quasiparticle_peak_input,
            )))
        });
    });
    let quasiparticle_table_energy = array![-0.40, -0.12, -0.01, 0.02, 0.20, 0.55];
    let quasiparticle_table_boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
    c.bench_function("sfconv_mkspectf_quasiparticle_table", |b| {
        b.iter(|| {
            black_box(sfconv_quasiparticle_table(black_box(
                SfconvQuasiparticleTableInput {
                    energy: quasiparticle_table_energy.view(),
                    boundaries: quasiparticle_table_boundaries.view(),
                    photoelectron_energy: 0.93,
                    quasiparticle_energy: 0.944,
                    endpoint_width: 0.073,
                    quasiparticle_width: 0.073 * 0.82,
                    plasma_frequency: 0.62,
                    renormalization_real: 0.82,
                    renormalization_imag: 0.06,
                    renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
                    interference_amplitude: 0.135,
                    exponential_reduction: 0.74,
                },
            )))
        });
    });
    let satellite_main_peak = array![
        0.144_118_631_068_914_32,
        0.796_854_020_052_775_2,
        3.306_037_878_829_96,
        2.944_827_731_705_054,
        0.351_606_691_790_681_77,
        0.027_414_131_538_569_52,
    ];
    let satellite_quasiparticle_interference = array![
        0.031_993_167_546_517_99,
        0.176_895_131_355_183_62,
        0.733_913_602_898_189_5,
        0.653_727_879_020_868,
        0.078_053_834_660_399_79,
        0.006_085_714_920_760_973,
    ];
    let satellite_extrinsic = array![0.04, 0.09, -0.02, 0.18, 0.13, 0.07];
    let satellite_interference = array![0.01, 0.025, 0.006, 0.055, 0.04, 0.015];
    let satellite_intrinsic = array![0.02, 0.035, 0.012, 0.08, 0.065, 0.025];
    c.bench_function("sfconv_mkspectf_satellite_table", |b| {
        b.iter(|| {
            black_box(sfconv_satellite_table(black_box(
                SfconvSatelliteTableInput {
                    main_peak: satellite_main_peak.view(),
                    quasiparticle_interference: satellite_quasiparticle_interference.view(),
                    extrinsic_satellite: satellite_extrinsic.view(),
                    interference_satellite: satellite_interference.view(),
                    intrinsic_satellite: satellite_intrinsic.view(),
                    boundaries: quasiparticle_table_boundaries.view(),
                    quasiparticle_lower_column_1based: 3,
                    quasiparticle_upper_column_1based: 4,
                    include_full_broadening_quasiparticle: true,
                    exponential_reduction: 0.74,
                },
            )))
        });
    });
    let mut split_table = Array2::<f64>::zeros((8, 8).f());
    for (row, values) in [
        (1, [0.10, 0.18, 0.35, 0.30, 0.22, 0.15, 0.25, 0.20]),
        (4, [0.02, 0.05, 0.11, 0.16, 0.13, 0.09, 0.12, 0.07]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            split_table[(row, column)] = value;
        }
    }
    let split_energy = array![-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0];
    let split_boundaries = array![-0.75, -0.45, -0.20, -0.05, 0.05, 0.20, 0.45, 0.80, 1.20];
    c.bench_function("sfconv_mkspectf_extrinsic_satellite_split", |b| {
        b.iter(|| {
            black_box(sfconv_split_extrinsic_satellite(black_box(
                SfconvExtrinsicSatelliteSplitInput {
                    spectral_function: split_table.view(),
                    energy: split_energy.view(),
                    boundaries: split_boundaries.view(),
                    photoelectron_energy: 0.05,
                    beta_zero: 1.0,
                },
            )))
        });
    });
    let mut satellite_table = Array2::<f64>::zeros((8, 6).f());
    for (row, values) in [
        (1, [0.40, 0.18, 0.06, 0.50, 0.28, 0.08]),
        (3, [0.10, 0.16, 0.08, 0.35, 0.05, 0.03]),
        (4, [0.05, 0.04, 0.20, 0.03, 0.30, 0.20]),
        (6, [0.08, 0.05, 0.03, 0.12, 0.07, 0.02]),
        (7, [0.04, 0.02, 0.01, 0.06, 0.09, 0.03]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            satellite_table[(row, column)] = value;
        }
    }
    let satellite_boundaries = array![-0.4, -0.2, 0.0, 0.15, 0.35, 0.7, 1.1];
    c.bench_function("sfconv_mkspectf_satellite_correction", |b| {
        b.iter(|| {
            black_box(sfconv_correct_satellite_weights(black_box(
                SfconvSatelliteCorrectionInput {
                    spectral_function: satellite_table.view(),
                    boundaries: satellite_boundaries.view(),
                    uniform_width: 0.2,
                    exponential_reduction: 0.73,
                },
            )))
        });
    });
    let spectral_satellite_weights = array![0.259_15, 0.119_355, 0.174_47, 0.036_5, 0.054_02];
    c.bench_function("sfconv_mkspectf_spectral_weights", |b| {
        b.iter(|| {
            black_box(sfconv_spectral_weights(black_box(
                SfconvSpectralWeightsInput {
                    renormalization_real: 0.82,
                    renormalization_imag: 0.06,
                    renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
                    interference_amplitude: 0.135,
                    interference_reduction: 0.43,
                    exponential_reduction: 0.74,
                    satellite_weights: spectral_satellite_weights.view(),
                },
            )))
        });
    });
    let path_average_source = array![0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25];
    let path_average_amplitude = array![0.82, 0.84, 0.88, 0.91, 0.89, 0.86, 0.83];
    let path_average_phase = array![0.05, 0.08, 0.13, 0.17, 0.14, 0.09, 0.02];
    c.bench_function("sfconv_so2conv_path_average", |b| {
        b.iter(|| {
            black_box(sfconv_path_average(black_box(SfconvPathAverageInput {
                source_momentum: path_average_source.view(),
                amplitude_reduction: path_average_amplitude.view(),
                phase_shift: path_average_phase.view(),
                previous_momentum: 1.00,
                center_momentum: 1.60,
                next_momentum: 2.30,
                momentum_step: 0.05,
            })))
        });
    });
    let senergies_context = SfconvSelfEnergyContext {
        fermi_energy: 0.50,
        fermi_momentum: 1.00,
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        quasiparticle_energy: 0.91,
        photoelectron_momentum: (2.0_f64 * 0.85).sqrt(),
        accuracy: 1.0e-4,
        pole_broadening: 0.035,
        dispersion_parameter: 0.28,
        include_below_fermi: false,
    };
    c.bench_function("sfconv_senergies_beta", |b| {
        b.iter(|| {
            let beta = black_box(sfconv_extrinsic_beta(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let imaginary = black_box(sfconv_imaginary_self_energy(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let real = black_box(sfconv_real_self_energy(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let real_derivative = black_box(sfconv_real_self_energy_derivative(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let imaginary_derivative = black_box(sfconv_imaginary_self_energy_derivative(
                black_box(0.36),
                black_box(senergies_context),
            ));
            black_box((beta, imaginary, real, real_derivative, imaginary_derivative))
        });
    });
    let satellite_context = SfconvSatelliteContext {
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        dispersion_parameter: 0.28,
        photoelectron_energy: 0.85,
        accuracy: 1.0e-4,
    };
    c.bench_function("sfconv_mksat_satellites", |b| {
        b.iter(|| {
            let interference = black_box(sfconv_interference_satellite(
                black_box(0.75),
                black_box(0.045),
                black_box(satellite_context),
            ));
            let intrinsic = black_box(sfconv_intrinsic_satellite(
                black_box(0.75),
                black_box(0.045),
                black_box(satellite_context),
            ));
            black_box((interference, intrinsic))
        });
    });
    c.bench_function("sfconv_interpsf_512_points", |b| {
        b.iter(|| {
            black_box(sfconv_interpolate_spectral_function(black_box(
                SfconvSpectralInterpolationInput {
                    energy: sfconv_energy.view(),
                    spectral_function: sfconv_spectral.view(),
                    output_len: black_box(512),
                },
            )))
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
    let dmdw_positions = Array2::from_shape_fn((48, 3), |(atom, component)| {
        let shell = atom / 12;
        let slot = atom % 12;
        match component {
            0 => shell as f64 * 1.7 + (slot % 3) as f64 * 0.9,
            1 => (slot / 3) as f64 * 1.1 - shell as f64 * 0.2,
            _ => (slot % 2) as f64 * 1.3 + shell as f64 * 0.4,
        }
    });
    let dmdw_descriptor = DmdwPathDescriptor {
        selectors: vec![1, 0, 0],
        max_effective_length: 7.0,
    };
    c.bench_function("dmdw_expand_path_descriptor_triple_wildcards", |b| {
        b.iter(|| {
            black_box(dmdw_expand_path_descriptor(
                black_box(dmdw_positions.view()),
                black_box(&dmdw_descriptor),
            ))
        });
    });
    let dmdw_pole_frequencies = Array1::from_shape_fn(64, |index| {
        if index % 17 == 0 {
            -0.25 * (index as f64 + 1.0)
        } else {
            1.0 + index as f64 * 0.125
        }
    });
    let dmdw_pole_weights = Array1::from_shape_fn(64, |index| 1.0 / (64.0 + index as f64));
    c.bench_function("dmdw_moment_summaries_from_poles_64", |b| {
        b.iter(|| {
            black_box(dmdw_moment_summaries_from_poles(
                black_box(31.773),
                black_box(dmdw_pole_frequencies.view()),
                black_box(dmdw_pole_weights.view()),
            ))
        });
    });
    let dmdw_type2_masses = arr1(&[63.546, 63.546, 63.546]);
    let mut dmdw_type2_force_blocks = Array4::zeros((3, 3, 3, 3));
    for atom in 0..3 {
        for component in 0..3 {
            dmdw_type2_force_blocks[(atom, atom, component, component)] =
                0.02 + 0.003 * atom as f64 + 0.001 * component as f64;
        }
    }
    for component in 0..3 {
        dmdw_type2_force_blocks[(0, 1, component, component)] = -0.004;
        dmdw_type2_force_blocks[(1, 0, component, component)] = -0.004;
        dmdw_type2_force_blocks[(1, 2, component, component)] = -0.003;
        dmdw_type2_force_blocks[(2, 1, component, component)] = -0.003;
    }
    let dmdw_type2_groups = vec![DmdwType2AtomGroup {
        center_atom_indices: vec![0],
    }];
    let dmdw_type2_coupling = DmdwPhononCoupling {
        energy_hartree: arr1(&[0.001, 0.002, 0.004]),
        energy_ev: arr1(&[0.027_211_396_132, 0.054_422_792_264, 0.108_845_584_528]),
        eliashberg: arr1(&[0.5, 1.0, 1.5]),
        matrix_element: arr1(&[0.05, 0.05, 0.05]),
        normalization: 1.0,
    };
    c.bench_function("dmdw_type2_a2f_single_group_order1", |b| {
        b.iter(|| {
            black_box(dmdw_type2_pole_weighted_a2f(
                black_box(dmdw_type2_force_blocks.view()),
                black_box(dmdw_type2_masses.view()),
                black_box(&dmdw_type2_groups),
                black_box(0),
                black_box(1),
                black_box(&dmdw_type2_coupling),
            ))
        });
    });
    let dmdw_a2f = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: Array1::from_shape_fn(32, |index| 2.0 + index as f64 * 0.5),
        lanczos_weight: Array1::from_shape_fn(32, |index| 1.0 / (32.0 + index as f64)),
        normalization: 1.0,
        pole_energy_ev: Array1::from_shape_fn(32, |index| 0.008 + index as f64 * 0.0015),
        pole_weight: Array1::from_shape_fn(32, |index| 0.02 / (1.0 + index as f64 * 0.1)),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.024,
    };
    let dmdw_self_energy_grid = Array1::from_shape_fn(257, |index| -0.18 + index as f64 * 0.0015);
    c.bench_function("dmdw_self_energy_grid_a2f_257x32", |b| {
        b.iter(|| {
            black_box(dmdw_self_energy_grid_from_a2f_poles(
                black_box(300.0),
                black_box(dmdw_self_energy_grid.view()),
                black_box(&dmdw_a2f),
            ))
        });
    });
    let dmdw_spectral_grid = Array1::from_shape_fn(65, |index| -2.0 + index as f64 * 0.0625);
    c.bench_function("dmdw_spectral_function_a2f_65x65x32", |b| {
        b.iter(|| {
            black_box(dmdw_spectral_function_from_a2f_poles(
                black_box(300.0),
                black_box(dmdw_spectral_grid.view()),
                black_box(0.0),
                black_box(dmdw_a2f.characteristic_energy_ev),
                black_box(&dmdw_a2f),
                black_box(20.0),
                black_box(65),
            ))
        });
    });
}

fn sample_epsilon_tables() -> Vec<EpsilonTable> {
    (0..3)
        .map(|table| {
            let table_f = table as f64;
            let energy_ev =
                Array1::from_shape_fn(160, |index| 0.02 + 0.018 * index as f64 + table_f * 0.003);
            let epsilon1_minus_one = energy_ev
                .mapv(|energy| (0.2 + 0.03 * table_f) * (-0.4 * energy).exp() + 0.01 * energy);
            let epsilon2 =
                energy_ev.mapv(|energy| (0.08 + 0.01 * table_f) * (1.0 + energy).recip());
            EpsilonTable {
                energy_ev,
                epsilon1_minus_one,
                epsilon2,
            }
        })
        .collect()
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
    bench_compton_helpers,
    bench_rhorrp_helpers,
    bench_kspace_helpers,
    bench_density_helpers,
    bench_grid_helpers,
    bench_genfmt_helpers,
    bench_interpolation,
    bench_quadrature,
    bench_bessel,
    bench_convolution,
    bench_fprime_helpers,
    bench_fms,
    bench_scalar_helpers,
    bench_sort_helpers,
    bench_path_helpers
);
criterion_main!(benches);
