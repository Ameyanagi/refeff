use super::{support::*, *};

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
