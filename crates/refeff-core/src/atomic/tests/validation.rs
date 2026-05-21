#![allow(clippy::excessive_precision)]

use super::*;

#[test]
fn atom_helper_kernels_reject_invalid_inputs() {
    assert!(matches!(
        atomic_polynomial_product_coefficient(&[1.0], &[2.0], 2),
        Err(AtomMathError::InvalidPolynomialTerm { .. })
    ));
    assert!(matches!(
        atomic_convergence_mix(0.5, Real::INFINITY, 1.0),
        Err(AtomMathError::NonFiniteScalar {
            field: "current_error",
            ..
        })
    ));
    assert!(matches!(
        thomas_fermi_density_potential(0.0, 1.0, 0.0),
        Err(AtomMathError::NonPositiveRadius { .. })
    ));
    assert!(matches!(
        atomic_occupation_product(&[1.0], &[], 0, 0),
        Err(AtomMathError::OccupationKappaLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_occupation_product(&[1.0], &[0], 0, 0),
        Err(AtomMathError::ZeroKappa)
    ));

    let coefficients = Array3::zeros((2, 2, 1));
    assert!(matches!(
        atomic_direct_coulomb_coefficient(coefficients.view(), 0, 0, 4),
        Err(AtomMathError::CoefficientChannelOutOfRange { .. })
    ));

    let coefficients = Array3::zeros((2, 3, 1));
    assert!(matches!(
        atomic_direct_coulomb_coefficient(coefficients.view(), 1, 2, 0),
        Err(AtomMathError::CoefficientTableShape { .. })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[1],
            occupations: &[],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[0],
            occupations: &[1.0],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[1],
            occupations: &[Real::NAN],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::NonFiniteScalar {
            field: "occupation",
            ..
        })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[6],
            occupations: &[2.0],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::CoefficientChannelOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 0.0,
            step: 0.05,
            requested_nucleus_index: 1,
            radial_count: 251,
            coefficient_count: 10,
            first_radius_times_charge: 1.0,
        }),
        Err(AtomMathError::InvalidNuclearPotentialScalar { .. })
    ));
    assert!(matches!(
        atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 92.0,
            step: 0.05,
            requested_nucleus_index: -11,
            radial_count: 5,
            coefficient_count: 10,
            first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
        }),
        Err(AtomMathError::NuclearRadiusOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 26.0,
            step: 0.05,
            requested_nucleus_index: 1,
            radial_count: 251,
            coefficient_count: 4,
            first_radius_times_charge: 26.0 * (-8.8_f64).exp(),
        }),
        Err(AtomMathError::InvalidNuclearPotentialCount { .. })
    ));
    let dsordf = sample_dsordf_fixture();
    assert!(matches!(
        atomic_differential_integral(dsordf.input(
            AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 8 },
            0,
            0.45,
        )),
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength { .. })
    ));
    assert!(matches!(
        atomic_differential_integral(dsordf.input(
            AtomicDifferentialIntegralKind::ComponentOverlap {
                left_orbital_1based: 0,
                right_orbital_1based: 1,
                multiply_by_derivative: false,
            },
            0,
            0.0,
        )),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_differential_integral(dsordf.input(
            AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
            0,
            -1.0,
        )),
        Err(AtomMathError::ZeroDifferentialIntegralOriginExponent)
    ));
    let bad_radii = Array1::from_vec(vec![0.0; 11]);
    assert!(matches!(
        atomic_differential_integral(AtomicDifferentialIntegralInput {
            radii: bad_radii.view(),
            ..dsordf.input(
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                0,
                0.45,
            )
        }),
        Err(AtomMathError::NonPositiveRadius { .. })
    ));
    let bad_derivative_coefficients = Array1::from_vec(vec![0.1; 5]);
    assert!(matches!(
        atomic_differential_integral(AtomicDifferentialIntegralInput {
            derivative_large_coefficients: bad_derivative_coefficients.view(),
            ..dsordf.input(
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                0,
                0.45,
            )
        }),
        Err(AtomMathError::CoefficientTableLengthMismatch { .. })
    ));
    let yzkteg = sample_yzkteg_fixture();
    assert!(matches!(
        atomic_yk_zk_transform(AtomicYkZkTransformInput {
            active_len: 3,
            ..yzkteg.input()
        }),
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength { .. })
    ));
    assert!(matches!(
        atomic_yk_zk_transform(AtomicYkZkTransformInput {
            step: 0.0,
            ..yzkteg.input()
        }),
        Err(AtomMathError::ZeroYkZkDenominator { field: "step" })
    ));
    assert!(matches!(
        atomic_yk_zk_transform(AtomicYkZkTransformInput {
            initial_power: 2.0,
            ..yzkteg.input()
        }),
        Err(AtomMathError::ZeroYkZkDenominator { field: "yk_origin" })
    ));
    let yzkrdf = sample_yzkrdf_fixture();
    assert!(matches!(
        atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
            left_orbital_1based: 0,
            ..yzkrdf.yzkrdf_input(1, 2, 2, false)
        }),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    assert!(matches!(
        AtomicLocalDensityExchangeMode::try_from(4),
        Err(AtomMathError::InvalidExchangeMode { idfock: 4 })
    ));
    let vlda = sample_vlda_fixture();
    assert!(matches!(
        atomic_local_density_potential(AtomicLocalDensityPotentialInput {
            speed_of_light: 0.0,
            ..vlda.input(AtomicLocalDensityExchangeMode::TotalDensity, false)
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_orbital_initialization(AtomicOrbitalInitializationInput {
            atomic_number: 4,
            ionicity: 0.0,
            principal_quantum_numbers: &[2],
            kappas: &[1],
            occupations: &[1.0],
        }),
        Err(AtomMathError::ElectronCountMismatch { .. })
    ));
    assert!(matches!(
        atomic_orbital_initialization(AtomicOrbitalInitializationInput {
            atomic_number: 1,
            ionicity: 0.0,
            principal_quantum_numbers: &[1],
            kappas: &[1],
            occupations: &[1.0],
        }),
        Err(AtomMathError::OrbitalAngularMomentumOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_entry_state(AtomicDiracEntryStateInput {
            asymptotic_large_component: Real::NAN,
            method: 1,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    let soldir_norm = sample_soldir_norm_fixture();
    assert!(matches!(
        atomic_dirac_normalization(soldir_norm.input(1, 6, 0.177, 0.82, 10, 5)),
        Err(AtomMathError::InvalidDiracNormalizationActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_normalization(soldir_norm.input(1, 6, 0.177, 0.82, 11, 0)),
        Err(AtomMathError::DiracNormalizationMatchingIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_normalization(soldir_norm.input(2, 6, 0.177, -0.5, 11, 5)),
        Err(AtomMathError::ZeroDiracNormalizationOriginExponent)
    ));
    let soldir_solution_norm = sample_soldir_solution_normalization_fixture(false, false);
    assert!(matches!(
        atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
            active_len: 0,
            ..soldir_solution_norm.input(6.25, 0.8, -0.4)
        }),
        Err(AtomMathError::InvalidDiracSolutionNormalizationActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
            norm: 0.0,
            ..soldir_solution_norm.input(6.25, 0.8, -0.4)
        }),
        Err(AtomMathError::NonPositiveScalar {
            field: "soldir_solution_norm",
            ..
        })
    ));
    let soldir_nodes = sample_soldir_node_count_component();
    assert!(matches!(
        atomic_dirac_node_count(AtomicDiracNodeCountInput {
            large_component: soldir_nodes.view(),
            matching_index_1based: 0,
            scan_index_1based: 3,
        }),
        Err(AtomMathError::InvalidDiracNodeCountIndex { .. })
    ));
    assert!(matches!(
        atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
            speed_of_light: 137.0373,
            norm: 1.0,
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            matching_small_component: 0.0,
            matching_index_1based: 0,
        },),
        Err(AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
            energy: 0.0,
            correction: 0.0,
            mismatch: 0.0,
            energy_sup: -1.0,
            energy_inf: -0.1,
            mismatch_precision: 0.1,
            zero_energy_precision: 1.0e-7,
        }),
        Err(AtomMathError::ZeroDiracEnergyCorrectionDenominator)
    ));
    assert!(matches!(
        atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: Real::NAN,
            mismatch_precision: 0.1,
            match_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: 1.0,
            mismatch_precision: 0.1,
            match_attempt_count: usize::MAX,
            max_attempt_count: usize::MAX,
        }),
        Err(AtomMathError::DiracRematchAttemptCountOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: 0.0,
            previous_energy: -0.1,
        }),
        Err(AtomMathError::ZeroDiracShootingPassEnergy)
    ));
    assert!(matches!(
        atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: Real::NAN,
            previous_energy: -0.1,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: Real::NAN,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 0.0,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: usize::MAX,
            max_attempt_count: usize::MAX,
        }),
        Err(AtomMathError::DiracNodeEnergyAttemptCountOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
            method: 1,
            primary_matching_precision: 0.0,
            secondary_matching_precision: 1.0e-5,
            energy_floor: -5.0,
            reference_energy: -0.5,
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
            method: 1,
            primary_matching_precision: 1.0e-5,
            secondary_matching_precision: 2.0e-5,
            energy_floor: Real::NAN,
            reference_energy: -0.5,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: 1,
            method: i32::MAX,
        }),
        Err(AtomMathError::DiracMethodOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
            large_source: soldir_nodes.view(),
            small_source: soldir_nodes
                .view()
                .slice_axis(Axis(0), Slice::from(..soldir_nodes.len() - 1)),
            large_source_coefficients: soldir_nodes.view(),
            small_source_coefficients: soldir_nodes.view(),
            coefficient_count: 1,
        }),
        Err(AtomMathError::RadialTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
            large_source: soldir_nodes.view(),
            small_source: soldir_nodes.view(),
            large_source_coefficients: soldir_nodes.view().slice_axis(Axis(0), Slice::from(..1)),
            small_source_coefficients: soldir_nodes.view(),
            coefficient_count: 2,
        }),
        Err(AtomMathError::CoefficientTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
            radial_len: 0,
            coefficient_len: 1,
        }),
        Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_radial_len",
            ..
        })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
            radial_len: 1,
            coefficient_len: 0,
        }),
        Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_coefficient_len",
            ..
        })
    ));
    assert!(matches!(
        atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: 0,
            matching_index_1based: 1,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: soldir_nodes.len(),
            matching_index_1based: 0,
        }),
        Err(AtomMathError::DiracMatchMatchingIndexOutOfRange { .. })
    ));
    let zero_match_denominator = Array1::<Real>::zeros(soldir_nodes.len());
    assert!(matches!(
        atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            homogeneous_large_component: zero_match_denominator.view(),
            homogeneous_small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: soldir_nodes.len(),
            matching_index_1based: 1,
        }),
        Err(AtomMathError::ZeroDiracMatchDenominator { .. })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: 0,
            matching_index_1based: 1,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: soldir_nodes.len(),
            matching_index_1based: 4,
        }),
        Err(AtomMathError::ZeroDiracMatchDenominator { .. })
    ));
    assert!(matches!(
        atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            homogeneous_large_coefficients: soldir_nodes.view(),
            homogeneous_small_coefficients: soldir_nodes.view(),
            matching_large_component: 0.0,
            matching_small_component: 0.0,
            homogeneous_matching_large_component: 1.0,
            homogeneous_matching_small_component: 1.0,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
            matching_index_1based: 1,
        }),
        Err(AtomMathError::ZeroDiracMatchDenominator { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_match(AtomicDiracEnergyDisagreementMatchInput {
            large_derivative: soldir_nodes.view(),
            small_derivative: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            homogeneous_large_coefficients: soldir_nodes.view(),
            homogeneous_small_coefficients: soldir_nodes.view(),
            matching_large_derivative: 0.0,
            matching_small_derivative: 0.0,
            homogeneous_matching_large_component: 1.0,
            homogeneous_matching_small_component: 1.0,
            coefficient_count: 1,
            active_len: 0,
            matching_index_1based: 1,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    let positive_radii = Array1::from_elem(soldir_nodes.len(), 0.1);
    assert!(matches!(
        atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            radii: positive_radii.view(),
            speed_of_light: 137.0373,
            coefficient_count: 1,
            active_len: 0,
        }),
        Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            radii: positive_radii.view(),
            speed_of_light: 0.0,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_correction(AtomicDiracEnergyDisagreementCorrectionInput {
            radii: positive_radii.view(),
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_derivative: soldir_nodes.view(),
            small_derivative: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            norm: 1.0,
            step: 0.11,
            origin_power: 1.0,
            coefficient_count: 1,
            active_len: 8,
        },),
        Err(AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_correction(AtomicDiracEnergyDisagreementCorrectionInput {
            radii: positive_radii.view(),
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_derivative: soldir_nodes.view(),
            small_derivative: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            norm: 1.0,
            step: 0.11,
            origin_power: -0.5,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
        },),
        Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionOriginExponent)
    ));
    let zero_derivative = Array1::<Real>::zeros(soldir_nodes.len());
    assert!(matches!(
        atomic_dirac_energy_disagreement_correction(AtomicDiracEnergyDisagreementCorrectionInput {
            radii: positive_radii.view(),
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_derivative: zero_derivative.view(),
            small_derivative: zero_derivative.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            norm: 0.9,
            step: 0.11,
            origin_power: 1.0,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
        },),
        Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionIntegral)
    ));
    assert!(matches!(
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: soldir_nodes.view(),
            active_len: 9,
            matching_index_1based: 5,
            already_relocated: false,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    let zero_matching_component = Array1::<Real>::zeros(13);
    assert!(matches!(
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: zero_matching_component.view(),
            active_len: 13,
            matching_index_1based: 5,
            already_relocated: false,
        }),
        Err(AtomMathError::DiracMatchingPointNotFound { .. })
    ));
    let intdir = sample_intdir_fixture();
    assert!(matches!(
        atomic_dirac_integration(AtomicDiracIntegrationInput {
            active_len: 12,
            ..intdir.input(AtomicDiracIntegrationMode::SearchMatchingPoint, 0, 0)
        }),
        Err(AtomMathError::InvalidDiracIntegrationActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_integration(intdir.input(
            AtomicDiracIntegrationMode::FixedMatchingPoint,
            5,
            139
        )),
        Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_integration(intdir.input(AtomicDiracIntegrationMode::InwardOnly, 65, 68)),
        Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_integration(AtomicDiracIntegrationInput {
            energy: 0.01,
            ..intdir.input(AtomicDiracIntegrationMode::InwardOnly, 65, 139)
        }),
        Err(AtomMathError::InvalidDiracIntegrationEnergy { .. })
    ));
    let soldir_setup = sample_soldir_setup_fixture();
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            active_len: 0,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::InvalidDiracSolverSetupActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            kappa: 0,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            principal_quantum_number: 0,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::InvalidDiracSolverPrincipalQuantumNumber { .. })
    ));
    let positive_potential = Array1::from_vec(vec![0.25; 7]);
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            potential: positive_potential.view(),
            kappa: 2,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::DiracSolverPotentialNotAttractive { .. })
    ));
    let potrdf = sample_potrdf_fixture();
    assert!(matches!(
        atomic_orbital_potential(AtomicOrbitalPotentialInput {
            active_orbital_1based: 0,
            ..potrdf.input(true, true)
        }),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    let kappas = [-1, 1, -2];
    assert!(matches!(
        atomic_radial_integral(yzkrdf.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 0,
                first_right: 0,
                second_left: 1,
                second_right: 2,
                rank: 1,
            },
            &kappas,
            false,
            None,
        )),
        Err(AtomMathError::MissingRadialFirstFactor)
    ));
    let coefficients = Array3::zeros((2, 2, 1));
    assert!(matches!(
        atomic_lagrange_parameters(
            AtomicLagrangeParametersInput {
                active_orbital_1based: Some(0),
                kappas: &[1, 1],
                occupations: &[1.0, 2.0],
                shell_markers: &[1, 1],
                include_exchange: true,
                coulomb_coefficients: coefficients.view(),
            },
            |_| Ok(0.0),
        ),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_lagrange_parameters(
            AtomicLagrangeParametersInput {
                active_orbital_1based: None,
                kappas: &[1, 1],
                occupations: &[0.0, 2.0],
                shell_markers: &[1, 1],
                include_exchange: true,
                coulomb_coefficients: coefficients.view(),
            },
            |_| Ok(0.0),
        ),
        Err(AtomMathError::NonPositiveOccupation { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[1],
                kappas: &[1, 1],
                occupations: &[1.0, 2.0],
                orbital_energies: &[-0.1, -0.2],
            },
            sample_atomic_tabrat_integral,
        ),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[0],
                kappas: &[1],
                occupations: &[1.0],
                orbital_energies: &[-0.1],
            },
            sample_atomic_tabrat_integral,
        ),
        Err(AtomMathError::InvalidPrincipalQuantumNumber { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[1],
                kappas: &[5],
                occupations: &[1.0],
                orbital_energies: &[-0.1],
            },
            sample_atomic_tabrat_integral,
        ),
        Err(AtomMathError::OrbitalLabelKappaOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[1],
                kappas: &[1],
                occupations: &[1.0],
                orbital_energies: &[-0.1],
            },
            |_| Ok(Real::NAN),
        ),
        Err(AtomMathError::NonFiniteScalar {
            field: "tabrat_integral",
            ..
        })
    ));
    let fpf0_radii = Array1::from_vec(vec![1.0, 1.2]);
    let fpf0_values = Array1::from_vec(vec![0.1, 0.2]);
    let fpf0_components = Array2::zeros((2, 1));
    let fpf0_input = AtomicFormFactorInput {
        atomic_number: 26,
        hole_orbital_1based: 1,
        radial_step: 0.05,
        total_energy: -1.0,
        radii: fpf0_radii.view(),
        density_4pi: fpf0_values.view(),
        initial_large_component: fpf0_values.view(),
        initial_small_component: fpf0_values.view(),
        large_components: fpf0_components.view(),
        small_components: fpf0_components.view(),
        occupations: &[1.0],
        orbital_energies: &[-0.2],
        kappas: &[1],
    };
    assert!(matches!(
        atomic_form_factor(AtomicFormFactorInput {
            atomic_number: 0,
            ..fpf0_input
        }),
        Err(AtomMathError::InvalidFormFactorAtomicNumber { .. })
    ));
    assert!(matches!(
        atomic_form_factor(AtomicFormFactorInput {
            hole_orbital_1based: 2,
            ..fpf0_input
        }),
        Err(AtomMathError::HoleOrbitalOutOfRange { .. })
    ));
    let bad_fpf0_density = Array1::from_vec(vec![0.1]);
    assert!(matches!(
        atomic_form_factor(AtomicFormFactorInput {
            density_4pi: bad_fpf0_density.view(),
            ..fpf0_input
        }),
        Err(AtomMathError::RadialTableLengthMismatch { .. })
    ));
    let schmidt = sample_schmidt_fixture();
    assert!(matches!(
        atomic_schmidt_orthogonalization(schmidt.as_input(Some(5)), sample_schmidt_integral),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));

    let bad_small_components = Array2::<Real>::zeros((4, 4));
    assert!(matches!(
        atomic_schmidt_orthogonalization(
            AtomicSchmidtOrthogonalizationInput {
                small_components: bad_small_components.view(),
                ..schmidt.as_input(None)
            },
            sample_schmidt_integral,
        ),
        Err(AtomMathError::MatrixShape { .. })
    ));

    let bad_active_lengths = [6, 4, 3, 5];
    assert!(matches!(
        atomic_schmidt_orthogonalization(
            AtomicSchmidtOrthogonalizationInput {
                active_lengths: &bad_active_lengths,
                ..schmidt.as_input(None)
            },
            sample_schmidt_integral,
        ),
        Err(AtomMathError::ActiveLengthOutOfRange { .. })
    ));

    assert!(matches!(
        atomic_schmidt_orthogonalization(schmidt.as_input(Some(1)), |request| match request {
            AtomicSchmidtIntegralRequest::Projection(_) => Ok(0.0),
            AtomicSchmidtIntegralRequest::Norm(_) => Ok(0.0),
        }),
        Err(AtomMathError::NonPositiveNorm { .. })
    ));
    assert!(matches!(
        atomic_breit_angular_coefficients(0, -1, 1),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_breit_angular_coefficients(i32::MIN, -1, 1),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_breit_angular_coefficients(1, -1, usize::MAX),
        Err(AtomMathError::BreitRankOutOfRange { .. })
    ));

    let coefficients = Array3::zeros((2, 2, 1));
    let input = AtomicTotalEnergyInput {
        kappas: &[1],
        occupations: &[],
        valence_occupations: &[0.0],
        orbital_energies: &[0.0],
        coulomb_coefficients: coefficients.view(),
    };
    assert!(matches!(
        atomic_total_energy(input, |_| Ok(0.0)),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));

    let input = AtomicTotalEnergyInput {
        kappas: &[1],
        occupations: &[1.0],
        valence_occupations: &[0.0],
        orbital_energies: &[0.0],
        coulomb_coefficients: coefficients.view(),
    };
    assert!(matches!(
        atomic_total_energy(input, |_| Ok(Real::NAN)),
        Err(AtomMathError::NonFiniteScalar {
            field: "radial_integral",
            ..
        })
    ));

    let single_overlap = Array2::from_elem((1, 1), 1.0);
    let input = AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: Some(0),
        kappas: &[1],
        occupations: &[1.0],
        overlap_integrals: single_overlap.view(),
    };
    assert!(matches!(
        atomic_overlap_amplitude_reduction(input),
        Err(AtomMathError::HoleOrbitalOutOfRange { .. })
    ));

    let input = AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: None,
        kappas: &[1, 1],
        occupations: &[1.0, 1.0],
        overlap_integrals: single_overlap.view(),
    };
    assert!(matches!(
        atomic_overlap_amplitude_reduction(input),
        Err(AtomMathError::OverlapMatrixShape { .. })
    ));

    let too_many_kappas = [1; 9];
    let too_many_occupations = [1.0; 9];
    let too_many_overlaps = Array2::from_diag(&Array1::from_elem(9, 1.0));
    let input = AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: None,
        kappas: &too_many_kappas,
        occupations: &too_many_occupations,
        overlap_integrals: too_many_overlaps.view(),
    };
    assert!(matches!(
        atomic_overlap_amplitude_reduction(input),
        Err(AtomMathError::KappaGroupTooLarge { .. })
    ));
}
