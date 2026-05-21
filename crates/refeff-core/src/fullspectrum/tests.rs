use ndarray::{Array1, array};
use num_complex::Complex64;

use crate::{ElamError, Real};

use super::{
    FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY, FEFF_HARTREE_EV, FullSpectrumBackground,
    FullSpectrumBackgroundInput, FullSpectrumBackgroundSegmentInput, FullSpectrumDefaultGridEdge,
    FullSpectrumDrudeInput, FullSpectrumEdgeAssemblyInput, FullSpectrumEdgeGridInput,
    FullSpectrumEdgeSelectionInput, FullSpectrumError, FullSpectrumFineStructure,
    FullSpectrumFineStructureInput, FullSpectrumFineStructureSegmentInput,
    FullSpectrumHamakerInput, FullSpectrumKramersKronigInput, FullSpectrumLinearGridInput,
    FullSpectrumNumberDensityInput, FullSpectrumOpticalConstantsInput, FullSpectrumQSumInput,
    FullSpectrumScatteringDielectricInput, FullSpectrumSumRulesInput, FullSpectrumValenceInput,
    full_spectrum_assemble_edge, full_spectrum_background_from_fprime,
    full_spectrum_default_energy_grid, full_spectrum_drude_term, full_spectrum_edge_energy_grid,
    full_spectrum_edges_from_occupations, full_spectrum_effective_electron_count,
    full_spectrum_elam_edge_energies, full_spectrum_fine_structure_from_segments,
    full_spectrum_hamaker_transform, full_spectrum_kramers_kronig,
    full_spectrum_linear_energy_grid, full_spectrum_number_density,
    full_spectrum_optical_constants, full_spectrum_scattering_to_dielectric,
    full_spectrum_sum_rules, full_spectrum_valence_epsilon2,
};

#[test]
fn effective_electron_count_matches_feff_qsum_reference() -> Result<(), FullSpectrumError> {
    let omega = array![0.0, 0.1, 0.2, 0.5, 1.0, 1.8];
    let epsilon2 = array![0.0, 0.5, 1.0, 0.25, 0.75, 0.1];

    let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
        number_density: 0.075,
        epsilon2: epsilon2.view(),
        omega: omega.view(),
        active_len: 6,
    })?;

    assert_close(neff, 0.442_098_097_959_400_5, 1.0e-14);
    Ok(())
}

#[test]
fn effective_electron_count_matches_feff_single_point_reference() -> Result<(), FullSpectrumError> {
    let omega = array![0.0, 0.1];
    let epsilon2 = array![1.0, 2.0];

    let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
        number_density: 0.075,
        epsilon2: epsilon2.view(),
        omega: omega.view(),
        active_len: 1,
    })?;

    assert_eq!(neff, 0.0);
    Ok(())
}

#[test]
fn effective_electron_count_rejects_invalid_inputs() {
    let omega = array![0.0, 0.1, 0.2];
    let epsilon2 = array![0.0, 0.5, 1.0];

    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.0,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 3,
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "number_density",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 4,
        }),
        Err(FullSpectrumError::ActiveCountOutOfRange {
            field: "epsilon2",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: array![0.0, f64::NAN, 1.0].view(),
            omega: omega.view(),
            active_len: 3,
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "epsilon2",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: array![0.0, 0.2, 0.1].view(),
            active_len: 3,
        }),
        Err(FullSpectrumError::DecreasingOmega { row: 2, .. })
    ));
}

#[test]
fn drude_term_matches_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![0.1, 0.2, 0.5];

    let drude = full_spectrum_drude_term(FullSpectrumDrudeInput {
        omega: omega.view(),
        lifetime_seconds: 1.0e-15,
        number_density: 0.075,
    })?;

    assert_eq!(drude.point_count(), 3);
    assert_close(drude.gamma_ev, 0.658, 1.0e-14);
    assert_close(drude.plasma_frequency_ev, 26.417_175_795_207_253, 1.0e-14);
    assert_close(drude.epsilon[0].re, -89.041_328_740_125_08, 1.0e-14);
    assert_close(drude.epsilon[0].im, 21.531_124_059_567_656, 1.0e-14);
    assert_close(drude.epsilon[2].re, -3.761_114_344_763_512, 1.0e-14);
    assert_close(drude.epsilon[2].im, 0.181_895_352_877_477_57, 1.0e-14);
    Ok(())
}

#[test]
fn drude_term_rejects_invalid_inputs() {
    assert!(matches!(
        full_spectrum_drude_term(FullSpectrumDrudeInput {
            omega: array![0.1].view(),
            lifetime_seconds: 0.0,
            number_density: 0.075,
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "lifetime_seconds",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_drude_term(FullSpectrumDrudeInput {
            omega: Array1::<Real>::zeros(0).view(),
            lifetime_seconds: 1.0e-15,
            number_density: 0.075,
        }),
        Err(FullSpectrumError::EmptyTable { name: "drude_term" })
    ));
    assert!(matches!(
        full_spectrum_drude_term(FullSpectrumDrudeInput {
            omega: array![0.0].view(),
            lifetime_seconds: 1.0e-15,
            number_density: 0.075,
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 0,
            ..
        })
    ));
}

#[test]
fn valence_epsilon2_matches_feff_rdval_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![
        5.0 / FEFF_HARTREE_EV,
        10.0 / FEFF_HARTREE_EV,
        15.0 / FEFF_HARTREE_EV,
        25.0 / FEFF_HARTREE_EV,
        40.0 / FEFF_HARTREE_EV,
        50.0 / FEFF_HARTREE_EV,
    ];
    let source_energy_ev = array![10.0, 20.0, 40.0];
    let source_absorption = array![1.0, 3.0, 7.0];

    let epsilon2 = full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
        number_density: 0.075,
        omega: omega.view(),
        source_energy_ev: source_energy_ev.view(),
        source_absorption_angstrom2: source_absorption.view(),
    })?;

    assert_eq!(epsilon2.len(), omega.len());
    assert_close(epsilon2[0], 0.0, 0.0);
    assert_close(epsilon2[1], 0.0, 0.0);
    assert_close(epsilon2[2], 131.219_281_455_964_96, 1.0e-12);
    assert_close(epsilon2[3], 157.463_137_747_157_93, 1.0e-12);
    assert_close(epsilon2[4], 0.0, 0.0);
    assert_close(epsilon2[5], 0.0, 0.0);
    Ok(())
}

#[test]
fn valence_epsilon2_rejects_invalid_inputs() {
    let omega = array![0.1, 0.2, 0.3];
    let source_energy_ev = array![10.0, 20.0, 40.0];
    let source_absorption = array![1.0, 3.0, 7.0];

    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.0,
            omega: omega.view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: source_absorption.view(),
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "number_density",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: array![0.1, 0.0, 0.3].view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: source_absorption.view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: omega.view(),
            source_energy_ev: array![10.0, 10.0, 40.0].view(),
            source_absorption_angstrom2: source_absorption.view(),
        }),
        Err(FullSpectrumError::NonIncreasingOmega { row: 1, .. })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: omega.view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: array![1.0, 3.0].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "source_absorption_angstrom2",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: omega.view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: array![1.0, f64::NAN, 7.0].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "source_absorption_angstrom2",
            row: 1,
            ..
        })
    ));
}

#[test]
fn number_density_matches_feff_rddens_reference_algorithm() -> Result<(), FullSpectrumError> {
    let atomic_numbers = array![29_usize, 8, 29];
    let multiplicities = array![0.01, 2.0, 3.0];
    let norman_radii = array![2.0, 1.5, 2.5];

    let copper_density = full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number: 29,
        atomic_numbers: atomic_numbers.view(),
        potential_multiplicities: multiplicities.view(),
        norman_radii: norman_radii.view(),
    })?;
    let oxygen_density = full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number: 8,
        atomic_numbers: atomic_numbers.view(),
        potential_multiplicities: multiplicities.view(),
        norman_radii: norman_radii.view(),
    })?;
    let missing_density = full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number: 26,
        atomic_numbers: atomic_numbers.view(),
        potential_multiplicities: multiplicities.view(),
        norman_radii: norman_radii.view(),
    })?;

    assert_close(copper_density, 0.013_380_217_262_078_158, 1.0e-16);
    assert_close(oxygen_density, 0.008_890_509_808_689_806, 1.0e-16);
    assert_close(missing_density, 0.0, 0.0);
    Ok(())
}

#[test]
fn number_density_rejects_invalid_inputs() {
    let atomic_numbers = array![29_usize, 8];
    let multiplicities = array![0.01, 2.0];
    let norman_radii = array![2.0, 1.5];

    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 0,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: multiplicities.view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::InvalidAtomicNumber { atomic_number: 0 })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: array![29_usize, 0].view(),
            potential_multiplicities: multiplicities.view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::InvalidAtomicNumber { atomic_number: 0 })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: array![0.01].view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "potential_multiplicities",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: array![0.01, f64::NAN].view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "potential_multiplicities",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: array![0.01, 0.0].view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "potential_multiplicities",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: multiplicities.view(),
            norman_radii: array![2.0, -1.5].view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "norman_radii",
            row: 1,
            ..
        })
    ));
}

#[test]
fn edge_selection_matches_feff_gtedgs_occupied_slots() -> Result<(), FullSpectrumError> {
    let mut occupations = Array1::<Real>::zeros(super::FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT);
    occupations[0] = 2.0;
    occupations[1] = 1.0;
    occupations[2] = 2.0;
    occupations[3] = 1.0;
    occupations[4] = 0.5;
    let edge_onsets = Array1::from_elem(super::FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT, 1.0 as Real);

    let selected = full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
        occupations: occupations.view(),
        edge_onsets_hartree: edge_onsets.view(),
    })?;

    assert_eq!(selected.edge_count(), 4);
    let labels: Vec<_> = selected.edges.iter().map(|edge| edge.label).collect();
    assert_eq!(labels, ["K", "L1", "L2", "L3"]);
    assert!(selected.edges.iter().all(|edge| edge.convolve));
    assert_close(selected.edges[2].occupation, 2.0, 0.0);
    Ok(())
}

#[test]
fn edge_selection_applies_feff_convolution_threshold() -> Result<(), FullSpectrumError> {
    let mut occupations = Array1::<Real>::zeros(super::FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT);
    occupations[0] = 2.0;
    occupations[2] = 1.0;
    occupations[4] = 1.0;
    let mut edge_onsets = Array1::from_elem(super::FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT, 10.0 as Real);
    edge_onsets[0] = super::FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE + 1.0e-7;
    edge_onsets[2] = super::FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE;
    edge_onsets[4] = -1.0;

    let selected = full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
        occupations: occupations.view(),
        edge_onsets_hartree: edge_onsets.view(),
    })?;

    assert_eq!(selected.edge_count(), 3);
    assert_eq!(selected.edges[0].label, "K");
    assert!(!selected.edges[0].convolve);
    assert_eq!(selected.edges[1].label, "L2");
    assert!(selected.edges[1].convolve);
    assert_eq!(selected.edges[2].label, "M1");
    assert!(selected.edges[2].convolve);
    Ok(())
}

#[test]
fn edge_selection_rejects_invalid_inputs() {
    let occupations = Array1::<Real>::zeros(super::FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT);
    let edge_onsets = Array1::<Real>::zeros(super::FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT);

    assert!(matches!(
        full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
            occupations: Array1::<Real>::zeros(39).view(),
            edge_onsets_hartree: edge_onsets.view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "occupations",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
            occupations: occupations.view(),
            edge_onsets_hartree: Array1::<Real>::zeros(39).view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "edge_onsets_hartree",
            ..
        })
    ));
    let mut nonfinite_occupations = occupations.clone();
    nonfinite_occupations[1] = f64::NAN;
    assert!(matches!(
        full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
            occupations: nonfinite_occupations.view(),
            edge_onsets_hartree: edge_onsets.view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "occupations",
            row: 1,
            ..
        })
    ));

    let mut negative_occupations = occupations.clone();
    negative_occupations[1] = -0.1;
    assert!(matches!(
        full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
            occupations: negative_occupations.view(),
            edge_onsets_hartree: edge_onsets.view(),
        }),
        Err(FullSpectrumError::NegativeValue {
            field: "occupations",
            row: 1,
            ..
        })
    ));

    let mut nonfinite_edge_onsets = edge_onsets.clone();
    nonfinite_edge_onsets[1] = f64::NAN;
    assert!(matches!(
        full_spectrum_edges_from_occupations(FullSpectrumEdgeSelectionInput {
            occupations: occupations.view(),
            edge_onsets_hartree: nonfinite_edge_onsets.view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "edge_onsets_hartree",
            row: 1,
            ..
        })
    ));
}

#[test]
fn sum_rules_match_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let energy_ev = array![10.0, 20.0, 40.0];
    let epsilon_minus_one = array![
        Complex64::new(0.10, 0.20),
        Complex64::new(0.15, 0.25),
        Complex64::new(0.20, 0.35),
    ];
    let refractive_index_minus_one = array![
        Complex64::new(0.01, 0.02),
        Complex64::new(0.02, 0.03),
        Complex64::new(0.03, 0.04),
    ];
    let absorption_coefficient = array![1000.0, 2000.0, 3000.0];

    let rules = full_spectrum_sum_rules(FullSpectrumSumRulesInput {
        number_density: 0.075,
        energy_ev: energy_ev.view(),
        epsilon_minus_one: epsilon_minus_one.view(),
        refractive_index_minus_one: refractive_index_minus_one.view(),
        absorption_coefficient: absorption_coefficient.view(),
    })?;

    assert_eq!(rules.point_count(), 3);
    assert_close(rules.energy_ev[2], 40.0, 0.0);
    assert_close(
        rules.epsilon2_effective_electrons[2],
        0.214_375_530_814_624_3,
        1.0e-14,
    );
    assert_close(
        rules.absorption_effective_electrons[2],
        162.008_009_804_073_2,
        1.0e-14,
    );
    assert_close(
        rules.loss_effective_electrons[2],
        0.145_731_231_111_208_3,
        1.0e-14,
    );
    assert_close(
        rules.absorption_refractive_sum[2],
        4.140_204_694_992_98,
        1.0e-14,
    );
    assert_close(rules.refractive_index_sum_ratio[2], 1.0, 1.0e-14);
    assert_close(
        rules.log_loss_moment_ratio[2],
        -0.038_690_505_695_948_56,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn sum_rules_reject_invalid_inputs() {
    let energy_ev = array![10.0, 5.0];
    let epsilon_minus_one = array![Complex64::new(0.10, 0.20), Complex64::new(0.15, 0.25)];
    let refractive_index_minus_one = array![Complex64::new(0.01, 0.02), Complex64::new(0.02, 0.03)];
    let absorption_coefficient = array![1000.0, 2000.0];

    assert!(matches!(
        full_spectrum_sum_rules(FullSpectrumSumRulesInput {
            number_density: 0.075,
            energy_ev: energy_ev.view(),
            epsilon_minus_one: epsilon_minus_one.view(),
            refractive_index_minus_one: refractive_index_minus_one.view(),
            absorption_coefficient: absorption_coefficient.view(),
        }),
        Err(FullSpectrumError::DecreasingOmega { row: 1, .. })
    ));
    assert!(matches!(
        full_spectrum_sum_rules(FullSpectrumSumRulesInput {
            number_density: 0.075,
            energy_ev: Array1::<Real>::zeros(0).view(),
            epsilon_minus_one: Array1::<Complex64>::zeros(0).view(),
            refractive_index_minus_one: Array1::<Complex64>::zeros(0).view(),
            absorption_coefficient: Array1::<Real>::zeros(0).view(),
        }),
        Err(FullSpectrumError::EmptyTable { name: "sum_rules" })
    ));
    assert!(matches!(
        full_spectrum_sum_rules(FullSpectrumSumRulesInput {
            number_density: 0.075,
            energy_ev: array![10.0, 20.0].view(),
            epsilon_minus_one: array![Complex64::new(0.10, 0.20)].view(),
            refractive_index_minus_one: refractive_index_minus_one.view(),
            absorption_coefficient: absorption_coefficient.view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "epsilon_minus_one",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_sum_rules(FullSpectrumSumRulesInput {
            number_density: 0.075,
            energy_ev: array![10.0].view(),
            epsilon_minus_one: array![Complex64::new(0.10, 0.20)].view(),
            refractive_index_minus_one: array![Complex64::new(0.0, 0.0)].view(),
            absorption_coefficient: array![1000.0].view(),
        }),
        Err(FullSpectrumError::NonFiniteSumRule {
            field: "refractive_index_sum_ratio",
            ..
        })
    ));
}

#[test]
fn linear_energy_grid_matches_active_feff_egrid_lin_branch() -> Result<(), FullSpectrumError> {
    let grid = full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
        point_count: 5,
        min_energy: 0.1,
        max_energy: 1.1,
    })?;

    assert_close(grid[0], 0.1, 0.0);
    assert_close(grid[1], 0.35, 1.0e-15);
    assert_close(grid[2], 0.6, 1.0e-15);
    assert_close(grid[3], 0.85, 1.0e-15);
    assert_close(grid[4], 1.1, 1.0e-15);
    Ok(())
}

#[test]
fn linear_energy_grid_applies_feff_positive_floor() -> Result<(), FullSpectrumError> {
    let grid = full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
        point_count: 3,
        min_energy: -1.0,
        max_energy: 0.5,
    })?;

    assert_close(grid[0], super::FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY, 0.0);
    assert_close(
        grid[1],
        (super::FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY + 0.5) / 2.0,
        1.0e-15,
    );
    assert_close(grid[2], 0.5, 1.0e-15);
    Ok(())
}

#[test]
fn linear_energy_grid_rejects_invalid_inputs() {
    assert!(matches!(
        full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
            point_count: 1,
            min_energy: 0.1,
            max_energy: 1.1,
        }),
        Err(FullSpectrumError::TooFewRows {
            name: "linear_grid",
            len: 1
        })
    ));
    assert!(matches!(
        full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
            point_count: 2,
            min_energy: f64::NAN,
            max_energy: 1.1,
        }),
        Err(FullSpectrumError::NonFiniteInput {
            name: "min_energy",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_linear_energy_grid(FullSpectrumLinearGridInput {
            point_count: 2,
            min_energy: 1.1,
            max_energy: 1.1,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "linear_grid",
            ..
        })
    ));
}

#[test]
fn default_energy_grid_prefers_fine_structure_edges_like_feff_rdop() -> Result<(), FullSpectrumError>
{
    let grid = full_spectrum_default_energy_grid(&[
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
    ])?;

    assert!(grid.used_fine_structure_edges);
    assert_close(grid.min_energy, 328.134_580_448_300_37, 1.0e-12);
    assert_close(grid.max_energy, 366.721_354_901_180_35, 1.0e-12);
    assert_eq!(grid.point_count, 2100);
    Ok(())
}

#[test]
fn default_energy_grid_falls_back_to_all_edges_like_feff_rdop() -> Result<(), FullSpectrumError> {
    let grid = full_spectrum_default_energy_grid(&[
        FullSpectrumDefaultGridEdge {
            atomic_number: 8,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 4,
            fine_structure: false,
        },
    ])?;

    assert!(!grid.used_fine_structure_edges);
    assert_close(grid.min_energy, 18.121_084_049_374_577, 1.0e-12);
    assert_close(grid.max_energy, 366.721_354_901_180_35, 1.0e-12);
    assert_eq!(grid.point_count, 18_971);
    Ok(())
}

#[test]
fn default_energy_grid_rejects_invalid_or_missing_edges() {
    assert!(matches!(
        full_spectrum_default_energy_grid(&[]),
        Err(FullSpectrumError::EmptyTable {
            name: "rdop_edge_grid"
        })
    ));
    assert!(matches!(
        full_spectrum_default_energy_grid(&[FullSpectrumDefaultGridEdge {
            atomic_number: 6,
            hole_index: 20,
            fine_structure: true,
        }]),
        Err(FullSpectrumError::MissingElamEdge {
            atomic_number: 6,
            hole_index: 20,
        })
    ));
    assert!(matches!(
        full_spectrum_default_energy_grid(&[FullSpectrumDefaultGridEdge {
            atomic_number: 101,
            hole_index: 1,
            fine_structure: true,
        }]),
        Err(FullSpectrumError::ElamEdgeTable {
            component: 0,
            source: ElamError::AtomicNumberOutOfRange { z: 101, .. },
        })
    ));
}

#[test]
fn edge_energy_grid_matches_feff_egrid_edge_restarts() -> Result<(), FullSpectrumError> {
    let edges = array![0.4, 0.8];

    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        min_energy: 0.0,
        max_energy: 1.0,
        edge_energies: edges.view(),
        wave_number_step: 0.2,
        max_points: 20,
    })?;

    assert!(!grid.clipped);
    assert_eq!(grid.point_count(), 15);
    assert_close(grid.energy[0], FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY, 0.0);
    assert_close(grid.energy[4], 0.326_895_256_108_847_07, 1.0e-14);
    assert_close(grid.energy[5], 0.4, 1.0e-14);
    assert_close(grid.energy[10], 0.8, 1.0e-14);
    assert_close(grid.energy[14], 1.0, 1.0e-14);
    Ok(())
}

#[test]
fn edge_energy_grid_matches_feff_egrid_without_edges() -> Result<(), FullSpectrumError> {
    let edges = Array1::<Real>::zeros(0);

    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        min_energy: 0.1,
        max_energy: 1.0,
        edge_energies: edges.view(),
        wave_number_step: 0.2,
        max_points: 20,
    })?;

    assert!(!grid.clipped);
    assert_eq!(grid.point_count(), 6);
    assert_close(grid.energy[0], 0.1, 0.0);
    assert_close(grid.energy[1], 0.209_442_719_099_991_6, 1.0e-14);
    assert_close(grid.energy[4], 0.777_770_876_399_966_3, 1.0e-14);
    assert_close(grid.energy[5], 1.0, 1.0e-14);
    Ok(())
}

#[test]
fn edge_energy_grid_reports_feff_capacity_clipping() -> Result<(), FullSpectrumError> {
    let edges = Array1::<Real>::zeros(0);

    let grid = full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
        min_energy: 0.1,
        max_energy: 10.0,
        edge_energies: edges.view(),
        wave_number_step: 0.2,
        max_points: 5,
    })?;

    assert!(grid.clipped);
    assert_eq!(grid.point_count(), 5);
    assert_close(grid.energy[4], 0.777_770_876_399_966_3, 1.0e-14);
    Ok(())
}

#[test]
fn elam_edge_energy_adapter_matches_feff_preved_nexted_scan() -> Result<(), FullSpectrumError> {
    let edges = full_spectrum_elam_edge_energies(&[29, 8, 79])?;

    assert_eq!(edges.len(), 30);
    assert!(
        edges
            .iter()
            .zip(edges.iter().skip(1))
            .all(|(left, right)| left < right)
    );

    let previous = edges
        .iter()
        .copied()
        .filter(|&energy| energy < 35.0)
        .fold(0.0, Real::max);
    let next = edges
        .iter()
        .copied()
        .filter(|&energy| energy > 35.0)
        .fold(1.0e8, Real::min);

    assert_close(previous, 3.499_636_651_471_202e1, 1.0e-14);
    assert_close(next, 4.030_296_538_890_82e1, 1.0e-14);
    Ok(())
}

#[test]
fn elam_edge_energy_adapter_rejects_out_of_range_components() {
    assert!(matches!(
        full_spectrum_elam_edge_energies(&[29, 101]),
        Err(FullSpectrumError::ElamEdgeTable {
            component: 1,
            source: ElamError::AtomicNumberOutOfRange { z: 101, .. },
        })
    ));
}

#[test]
fn edge_energy_grid_rejects_invalid_inputs() {
    let edges = array![0.4];

    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 0.1,
            max_energy: 1.0,
            edge_energies: edges.view(),
            wave_number_step: 0.2,
            max_points: 1,
        }),
        Err(FullSpectrumError::TooFewRows {
            name: "edge_grid",
            len: 1
        })
    ));
    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 0.1,
            max_energy: 1.0,
            edge_energies: edges.view(),
            wave_number_step: 0.0,
            max_points: 2,
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "wave_number_step",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 0.1,
            max_energy: 1.0,
            edge_energies: array![f64::NAN].view(),
            wave_number_step: 0.2,
            max_points: 2,
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "edge_energies",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_edge_energy_grid(FullSpectrumEdgeGridInput {
            min_energy: 1.0,
            max_energy: 1.0,
            edge_energies: edges.view(),
            wave_number_step: 0.2,
            max_points: 2,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "edge_grid",
            ..
        })
    ));
}

#[test]
fn background_from_fprime_matches_feff_rdbkg_reference_algorithm() -> Result<(), FullSpectrumError>
{
    let omega = array![
        5.0 / FEFF_HARTREE_EV,
        10.0 / FEFF_HARTREE_EV,
        20.0 / FEFF_HARTREE_EV,
        30.0 / FEFF_HARTREE_EV,
    ];
    let photon_energy_ev = array![5.0, 15.0, 25.0];
    let f_prime = array![1.0, 3.0, 5.0];
    let f_double_prime = array![0.1, 0.3, 0.5];
    let segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: photon_energy_ev.view(),
        f_prime: f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];

    let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
        omega: omega.view(),
        segments: &segments,
    })?;

    let expected_second = 1.0 + 0.5 * (3.0 - 1.0) * (10.0_f64 / 15.0).powi(2);
    let expected_third =
        1.0 + ((2.0 / 15.0_f64.powi(2) + 4.0 / 25.0_f64.powi(2)) / 2.0) * 20.0_f64.powi(2);

    assert_eq!(background.point_count(), 4);
    assert_close(background.zero_energy_fprime, 1.0, 0.0);
    assert_close(background.scattering_factor[0].re, 1.0, 0.0);
    assert_close(background.scattering_factor[0].im, 0.0, 0.0);
    assert_close(background.scattering_factor[1].re, expected_second, 1.0e-14);
    assert_close(background.scattering_factor[1].im, 0.2, 1.0e-14);
    assert_close(background.scattering_factor[2].re, expected_third, 1.0e-14);
    assert_close(background.scattering_factor[2].im, 0.4, 1.0e-14);
    assert_close(background.scattering_factor[3].re, 0.0, 0.0);
    assert_close(background.scattering_factor[3].im, 0.0, 0.0);
    assert!(background.effective_electron_count.is_finite());
    assert!(background.effective_electron_count > 0.0);
    Ok(())
}

#[test]
fn background_from_fprime_applies_feff_file_priority() -> Result<(), FullSpectrumError> {
    let omega = array![10.0 / FEFF_HARTREE_EV];
    let photon_energy_ev = array![5.0, 15.0, 25.0];
    let high_priority_f_prime = array![1.0, 3.0, 5.0];
    let high_priority_f_double_prime = array![0.1, 0.3, 0.5];
    let low_priority_f_prime = array![1.0, 12.0, 14.0];
    let low_priority_f_double_prime = array![9.0, 9.0, 9.0];
    let segments = [
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: photon_energy_ev.view(),
            f_prime: high_priority_f_prime.view(),
            f_double_prime: high_priority_f_double_prime.view(),
        },
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: photon_energy_ev.view(),
            f_prime: low_priority_f_prime.view(),
            f_double_prime: low_priority_f_double_prime.view(),
        },
    ];

    let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
        omega: omega.view(),
        segments: &segments,
    })?;

    assert_close(
        background.scattering_factor[0].re,
        1.444_444_444_444_444_4,
        1.0e-14,
    );
    assert_close(background.scattering_factor[0].im, 0.2, 1.0e-14);
    Ok(())
}

#[test]
fn background_from_fprime_updates_fp0_in_feff_reverse_order() -> Result<(), FullSpectrumError> {
    let omega = array![30.0 / FEFF_HARTREE_EV];
    let high_priority_energy_ev = array![5.0, 15.0];
    let high_priority_f_prime = array![1.0, 3.0];
    let high_priority_f_double_prime = array![0.1, 0.3];
    let low_priority_energy_ev = array![20.0, 40.0];
    let low_priority_f_prime = array![10.0, 18.0];
    let low_priority_f_double_prime = array![1.0, 3.0];
    let segments = [
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: high_priority_energy_ev.view(),
            f_prime: high_priority_f_prime.view(),
            f_double_prime: high_priority_f_double_prime.view(),
        },
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: low_priority_energy_ev.view(),
            f_prime: low_priority_f_prime.view(),
            f_double_prime: low_priority_f_double_prime.view(),
        },
    ];

    let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
        omega: omega.view(),
        segments: &segments,
    })?;

    assert_close(background.zero_energy_fprime, 1.0, 0.0);
    assert_close(background.scattering_factor[0].re, 12.25, 1.0e-14);
    assert_close(background.scattering_factor[0].im, 2.0, 1.0e-14);
    Ok(())
}

#[test]
fn background_from_fprime_rejects_invalid_inputs() {
    let omega = array![10.0 / FEFF_HARTREE_EV];
    let photon_energy_ev = array![5.0, 15.0, 25.0];
    let f_prime = array![1.0, 3.0, 5.0];
    let f_double_prime = array![0.1, 0.3, 0.5];
    let valid_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: photon_energy_ev.view(),
        f_prime: f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];

    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: Array1::<Real>::zeros(0).view(),
            segments: &valid_segments,
        }),
        Err(FullSpectrumError::EmptyTable { name: "omega" })
    ));
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &[],
        }),
        Err(FullSpectrumError::EmptyTable {
            name: "background_segments"
        })
    ));

    let too_short_energy = array![5.0];
    let too_short_f_prime = array![1.0];
    let too_short_f_double_prime = array![0.1];
    let too_short_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: too_short_energy.view(),
        f_prime: too_short_f_prime.view(),
        f_double_prime: too_short_f_double_prime.view(),
    }];
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &too_short_segments,
        }),
        Err(FullSpectrumError::SegmentTooShort {
            name: "background",
            segment: 0,
            len: 1
        })
    ));

    let mismatched_f_prime = array![1.0, 3.0];
    let mismatched_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: photon_energy_ev.view(),
        f_prime: mismatched_f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &mismatched_segments,
        }),
        Err(FullSpectrumError::SegmentLengthMismatch {
            field: "f_prime",
            segment: 0,
            actual: 2,
            expected: 3
        })
    ));

    let non_increasing_energy = array![5.0, 5.0, 25.0];
    let non_increasing_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: non_increasing_energy.view(),
        f_prime: f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &non_increasing_segments,
        }),
        Err(FullSpectrumError::SegmentNonIncreasingEnergy {
            segment: 0,
            row: 1,
            ..
        })
    ));
}

#[test]
fn fine_structure_from_segments_matches_feff_rdst_reference_algorithm()
-> Result<(), FullSpectrumError> {
    let omega = array![
        10.0 / FEFF_HARTREE_EV,
        12.5 / FEFF_HARTREE_EV,
        15.0 / FEFF_HARTREE_EV,
        20.0 / FEFF_HARTREE_EV,
    ];
    let fms_energy_ev = array![5.0, 10.0, 15.0];
    let fms_wave_number = array![1.0, 3.0, 4.0];
    let real_fms = array![1.0, 2.0, 3.0];
    let real_fms_background = array![0.5, 1.0, 1.5];
    let imaginary_fms = array![0.0, 2.0, 4.0];
    let imaginary_fms_background = array![0.0, 1.0, 2.0];
    let path_energy_ev = array![10.0, 20.0, 30.0];
    let path_wave_number = array![3.0, 5.0, 7.0];
    let real_path = array![10.0, 20.0, 30.0];
    let real_path_background = array![5.0, 10.0, 15.0];
    let imaginary_path = array![20.0, 40.0, 60.0];
    let imaginary_path_background = array![10.0, 20.0, 30.0];

    let fine_structure =
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: real_fms.view(),
                background: real_fms_background.view(),
            },
            real_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: real_path.view(),
                background: real_path_background.view(),
            },
            imaginary_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: imaginary_fms.view(),
                background: imaginary_fms_background.view(),
            },
            imaginary_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: imaginary_path.view(),
                background: imaginary_path_background.view(),
            },
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        })?;

    assert_eq!(fine_structure.point_count(), 4);
    assert_close(
        fine_structure.real_transition_interval[0],
        10.0 / FEFF_HARTREE_EV,
        1.0e-14,
    );
    assert_close(
        fine_structure.real_transition_interval[1],
        15.0 / FEFF_HARTREE_EV,
        1.0e-14,
    );
    assert_close(fine_structure.scattering_factor[0].re, 2.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[0].im, 2.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[1].re, 7.5, 1.0e-14);
    assert_close(fine_structure.scattering_factor[1].im, 14.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[2].re, 15.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[2].im, 30.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[3].re, 20.0, 1.0e-14);
    assert_close(fine_structure.scattering_factor[3].im, 40.0, 1.0e-14);
    assert_close(fine_structure.background[1].re, 3.75, 1.0e-14);
    assert_close(fine_structure.background[1].im, 7.0, 1.0e-14);
    assert_close(fine_structure.background[3].re, 10.0, 1.0e-14);
    assert_close(fine_structure.background[3].im, 20.0, 1.0e-14);
    Ok(())
}

#[test]
fn fine_structure_from_segments_clamps_negative_imaginary_parts() -> Result<(), FullSpectrumError> {
    let omega = array![12.5 / FEFF_HARTREE_EV];
    let fms_energy_ev = array![5.0, 10.0, 15.0];
    let fms_wave_number = array![1.0, 3.0, 4.0];
    let real = array![1.0, 2.0, 3.0];
    let real_background = array![0.5, 1.0, 1.5];
    let imaginary_fms = array![0.0, -2.0, -4.0];
    let imaginary_fms_background = array![0.0, -1.0, -2.0];
    let path_energy_ev = array![10.0, 20.0, 30.0];
    let path_wave_number = array![3.0, 5.0, 7.0];
    let imaginary_path = array![-20.0, -40.0, -60.0];
    let imaginary_path_background = array![-10.0, -20.0, -30.0];

    let fine_structure =
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: real.view(),
                background: real_background.view(),
            },
            real_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: real.view(),
                background: real_background.view(),
            },
            imaginary_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fms_energy_ev.view(),
                wave_number_inverse_angstrom: fms_wave_number.view(),
                scattering_factor: imaginary_fms.view(),
                background: imaginary_fms_background.view(),
            },
            imaginary_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: path_energy_ev.view(),
                wave_number_inverse_angstrom: path_wave_number.view(),
                scattering_factor: imaginary_path.view(),
                background: imaginary_path_background.view(),
            },
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        })?;

    assert_close(fine_structure.scattering_factor[0].im, 0.0, 0.0);
    assert_close(fine_structure.background[0].im, 0.0, 0.0);
    Ok(())
}

#[test]
fn fine_structure_from_segments_rejects_invalid_inputs() {
    let omega = array![10.0 / FEFF_HARTREE_EV];
    let energy_ev = array![5.0, 10.0, 15.0];
    let wave_number = array![1.0, 3.0, 4.0];
    let signal = array![1.0, 2.0, 3.0];
    let background = array![0.5, 1.0, 1.5];
    let valid_segment = FullSpectrumFineStructureSegmentInput {
        photon_energy_ev: energy_ev.view(),
        wave_number_inverse_angstrom: wave_number.view(),
        scattering_factor: signal.view(),
        background: background.view(),
    };
    let empty_omega = Array1::<Real>::zeros(0);

    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: empty_omega.view(),
            real_fms: valid_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        }),
        Err(FullSpectrumError::EmptyTable { name: "omega" })
    ));
    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: valid_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 4.0,
            high_wave_number: 3.0,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "fine_structure_wave_number",
            ..
        })
    ));

    let too_short_energy = array![5.0];
    let too_short_wave_number = array![1.0];
    let too_short_signal = array![1.0];
    let too_short_background = array![0.5];
    let too_short_segment = FullSpectrumFineStructureSegmentInput {
        photon_energy_ev: too_short_energy.view(),
        wave_number_inverse_angstrom: too_short_wave_number.view(),
        scattering_factor: too_short_signal.view(),
        background: too_short_background.view(),
    };
    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: too_short_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        }),
        Err(FullSpectrumError::SegmentTooShort {
            name: "fine_structure",
            segment: 0,
            len: 1
        })
    ));

    let high_wave_number_only = array![5.0, 6.0, 7.0];
    let missing_transition_segment = FullSpectrumFineStructureSegmentInput {
        photon_energy_ev: energy_ev.view(),
        wave_number_inverse_angstrom: high_wave_number_only.view(),
        scattering_factor: signal.view(),
        background: background.view(),
    };
    assert!(matches!(
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: omega.view(),
            real_fms: missing_transition_segment,
            real_path: valid_segment,
            imaginary_fms: valid_segment,
            imaginary_path: valid_segment,
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        }),
        Err(FullSpectrumError::MissingTransitionThreshold {
            name: "real_fms",
            ..
        })
    ));
}

#[test]
fn edge_assembly_matches_feff_addedg_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = Array1::from_shape_fn(9, |row| row as Real);
    let background = sample_edge_background(omega.len());
    let fine_structure = sample_edge_fine_structure(omega.len());

    let edge = full_spectrum_assemble_edge(FullSpectrumEdgeAssemblyInput {
        omega: omega.view(),
        background: &background,
        fine_structure: &fine_structure,
        transition_size: 4.0,
    })?;

    let theta = 0.4 * std::f64::consts::FRAC_PI_2;
    let cos_squared = theta.cos().powi(2);
    let sin_squared = theta.sin().powi(2);
    let row_four_entry_real = 304.0 * cos_squared + 204.0 * sin_squared;
    let row_four_entry_imag = -34.0 * cos_squared - 24.0 * sin_squared;
    let row_four_real = 304.0 * cos_squared + row_four_entry_real * sin_squared - 1.0;

    assert_eq!(edge.point_count(), omega.len());
    assert_eq!(edge.overlap_points, 5);
    assert_close(edge.effective_electron_count, 2.5, 0.0);
    assert_close(edge.zero_energy_fprime, 1.0, 0.0);
    assert_close(edge.scattering_factor[0].re, 99.0, 0.0);
    assert_close(edge.scattering_factor[0].im, 0.0, 0.0);
    assert_close(edge.background[0].re, 99.0, 0.0);
    assert_close(edge.background[0].im, 0.0, 0.0);
    assert_close(edge.scattering_factor[4].re, row_four_real, 1.0e-14);
    assert_close(edge.scattering_factor[4].im, row_four_entry_imag, 1.0e-14);
    assert_close(edge.background[4].re, 303.0, 1.0e-14);
    assert_close(edge.background[4].im, -34.0, 1.0e-14);
    assert_close(edge.scattering_factor[8].re, 107.0, 0.0);
    assert_close(edge.scattering_factor[8].im, -18.0, 0.0);
    assert_close(edge.background[8].re, 107.0, 0.0);
    assert_close(edge.background[8].im, -18.0, 0.0);
    Ok(())
}

#[test]
fn edge_assembly_rejects_invalid_inputs() {
    let omega = Array1::from_shape_fn(9, |row| row as Real);
    let background = sample_edge_background(omega.len());
    let fine_structure = sample_edge_fine_structure(omega.len());
    let empty_omega = Array1::<Real>::zeros(0);

    assert!(matches!(
        full_spectrum_assemble_edge(FullSpectrumEdgeAssemblyInput {
            omega: empty_omega.view(),
            background: &background,
            fine_structure: &fine_structure,
            transition_size: 4.0,
        }),
        Err(FullSpectrumError::EmptyTable {
            name: "edge_assembly"
        })
    ));

    let short_background = FullSpectrumBackground {
        scattering_factor: Array1::from_elem(omega.len() - 1, Complex64::new(0.0, 0.0)),
        effective_electron_count: 2.5,
        zero_energy_fprime: 1.0,
    };
    assert!(matches!(
        full_spectrum_assemble_edge(FullSpectrumEdgeAssemblyInput {
            omega: omega.view(),
            background: &short_background,
            fine_structure: &fine_structure,
            transition_size: 4.0,
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "background scattering_factor",
            actual: 8,
            expected: 9
        })
    ));

    let mut nonfinite_fine = fine_structure.clone();
    nonfinite_fine.scattering_factor[1] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        full_spectrum_assemble_edge(FullSpectrumEdgeAssemblyInput {
            omega: omega.view(),
            background: &background,
            fine_structure: &nonfinite_fine,
            transition_size: 4.0,
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "fine_structure scattering_factor",
            row: 1,
            ..
        })
    ));

    let mut bad_interval = fine_structure;
    bad_interval.real_energy_interval = [5.0, 4.0];
    assert!(matches!(
        full_spectrum_assemble_edge(FullSpectrumEdgeAssemblyInput {
            omega: omega.view(),
            background: &background,
            fine_structure: &bad_interval,
            transition_size: 4.0,
        }),
        Err(FullSpectrumError::InvalidEnergyRange {
            name: "real_energy_interval",
            ..
        })
    ));
}

#[test]
fn scattering_to_dielectric_matches_feff_fullspectrum_reference_algorithm()
-> Result<(), FullSpectrumError> {
    let omega = array![1.0, 2.0];
    let scattering_factor = array![Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.25)];
    let background_scattering_factor = array![Complex64::new(0.25, 0.5), Complex64::new(0.1, 0.05)];

    let dielectric =
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: omega.view(),
            scattering_factor: scattering_factor.view(),
            background_scattering_factor: background_scattering_factor.view(),
        })?;

    assert_eq!(dielectric.point_count(), 2);
    assert_eq!(dielectric.omega[1], omega[1]);
    assert_close(
        dielectric.epsilon_minus_one[0].re,
        -0.125_663_706_143_591_74,
        1.0e-15,
    );
    assert_close(
        dielectric.epsilon_minus_one[0].im,
        -0.251_327_412_287_183_47,
        1.0e-15,
    );
    assert_close(
        dielectric.background_epsilon_minus_one[0].re,
        -0.031_415_926_535_897_934,
        1.0e-15,
    );
    assert_close(
        dielectric.background_epsilon_minus_one[0].im,
        -0.062_831_853_071_795_87,
        1.0e-15,
    );
    assert_close(dielectric.sigma[0], -0.052_118_634_285_441_2, 1.0e-15);
    assert_close(
        dielectric.epsilon_minus_one[1].re,
        0.015_707_963_267_948_967,
        1.0e-15,
    );
    assert_close(
        dielectric.epsilon_minus_one[1].im,
        -0.007_853_981_633_974_483,
        1.0e-15,
    );
    assert_close(dielectric.sigma[1], -0.003_257_414_642_840_075, 1.0e-15);
    Ok(())
}

#[test]
fn kramers_kronig_matches_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![1.0, 2.0, 4.0, 7.0, 11.0];
    let epsilon2 = array![0.2, 0.5, 0.25, 0.4, 0.15];

    let epsilon1 = full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
        omega: omega.view(),
        epsilon2: epsilon2.view(),
    })?;

    assert_close(epsilon1[0], 0.459_691_458_481_860_66, 1.0e-14);
    assert_close(epsilon1[1], 0.459_691_458_481_860_66, 1.0e-14);
    assert_close(epsilon1[2], 0.160_612_887_833_759_8, 1.0e-14);
    assert_close(epsilon1[3], 0.014_010_911_454_772_346, 1.0e-14);
    assert_close(epsilon1[4], 0.014_010_911_454_772_346, 1.0e-14);
    Ok(())
}

#[test]
fn optical_constants_match_feff_eps2opt_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![0.5, 1.0];
    let epsilon_minus_one = array![Complex64::new(3.0, 4.0), Complex64::new(-0.5, 0.25)];

    let constants = full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
        omega: omega.view(),
        epsilon_minus_one: epsilon_minus_one.view(),
    })?;

    assert_eq!(constants.point_count(), 2);
    assert_eq!(constants.omega[1], omega[1]);
    assert_eq!(constants.epsilon_minus_one[0], epsilon_minus_one[0]);
    assert_close(
        constants.refractive_index_minus_one[0].re,
        1.197_368_226_935_62,
        1.0e-14,
    );
    assert_close(
        constants.refractive_index_minus_one[0].im,
        0.910_179_721_124_454_7,
        1.0e-14,
    );
    assert_close(
        constants.absorption_coefficient[0],
        12.551_376_312_230_127,
        1.0e-14,
    );
    assert_close(constants.reflectivity[0], 0.204_687_076_850_633_86, 1.0e-14);
    assert_close(constants.loss[0], 0.125, 1.0e-14);
    assert_close(
        constants.refractive_index_minus_one[1].re,
        -0.272_326_654_887_322_55,
        1.0e-14,
    );
    assert_close(
        constants.refractive_index_minus_one[1].im,
        0.171_780_374_861_256_22,
        1.0e-14,
    );
    assert_close(
        constants.absorption_coefficient[1],
        4.737_701_967_861_726,
        1.0e-14,
    );
    assert_close(constants.reflectivity[1], 0.034_392_102_279_900_92, 1.0e-14);
    assert_close(constants.loss[1], 0.8, 1.0e-14);
    Ok(())
}

#[test]
fn hamaker_transform_matches_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![1.0, 2.0, 4.0, 7.0, 11.0];
    let epsilon = array![
        Complex64::new(0.1, 0.2),
        Complex64::new(0.3, 0.5),
        Complex64::new(0.2, 0.25),
        Complex64::new(0.4, 0.4),
        Complex64::new(0.1, 0.15),
    ];

    let imaginary_axis = full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
        omega: omega.view(),
        epsilon: epsilon.view(),
    })?;

    assert_close(imaginary_axis[0], 0.354_646_982_533_894_65, 1.0e-14);
    assert_close(imaginary_axis[1], 0.354_646_982_533_894_65, 1.0e-14);
    assert_close(imaginary_axis[2], 0.223_104_490_219_296_74, 1.0e-14);
    assert_close(imaginary_axis[3], 0.126_886_660_083_068_56, 1.0e-14);
    assert_close(imaginary_axis[4], 0.126_886_660_083_068_56, 1.0e-14);
    Ok(())
}

#[test]
fn fullspectrum_transforms_reject_invalid_inputs() {
    assert!(matches!(
        full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
            omega: array![1.0].view(),
            epsilon2: array![0.2].view(),
        }),
        Err(FullSpectrumError::TooFewRows {
            name: "kramers_kronig",
            len: 1
        })
    ));
    assert!(matches!(
        full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
            omega: array![1.0, 1.0].view(),
            epsilon2: array![0.2, 0.3].view(),
        }),
        Err(FullSpectrumError::NonIncreasingOmega { row: 1, .. })
    ));
    assert!(matches!(
        full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
            omega: array![1.0, 2.0].view(),
            epsilon: array![Complex64::new(0.1, f64::NAN), Complex64::new(0.2, 0.3)].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "epsilon imaginary",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
            omega: array![1.0, 2.0].view(),
            epsilon: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "epsilon",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
            omega: array![0.0].view(),
            epsilon_minus_one: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
            omega: array![1.0].view(),
            epsilon_minus_one: array![Complex64::new(f64::NAN, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "epsilon real",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
            omega: array![1.0, 2.0].view(),
            epsilon_minus_one: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "epsilon_minus_one",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.0,
            omega: array![1.0].view(),
            scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "number_density",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: array![0.0].view(),
            scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: array![1.0].view(),
            scattering_factor: array![Complex64::new(0.1, f64::NAN)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "scattering_factor imaginary",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: array![1.0, 2.0].view(),
            scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "scattering_factor",
            ..
        })
    ));
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn sample_edge_background(point_count: usize) -> FullSpectrumBackground {
    FullSpectrumBackground {
        scattering_factor: Array1::from_shape_fn(point_count, |row| {
            Complex64::new(100.0 + row as Real, 10.0 + row as Real)
        }),
        effective_electron_count: 2.5,
        zero_energy_fprime: 1.0,
    }
}

fn sample_edge_fine_structure(point_count: usize) -> FullSpectrumFineStructure {
    FullSpectrumFineStructure {
        scattering_factor: Array1::from_shape_fn(point_count, |row| {
            Complex64::new(200.0 + row as Real, 20.0 + row as Real)
        }),
        background: Array1::from_shape_fn(point_count, |row| {
            Complex64::new(300.0 + row as Real, 30.0 + row as Real)
        }),
        real_energy_interval: [2.0, 6.0],
        imaginary_energy_interval: [2.0, 6.0],
        real_transition_interval: [2.0, 6.0],
        imaginary_transition_interval: [2.0, 6.0],
    }
}
