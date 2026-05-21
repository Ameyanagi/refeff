use ndarray::{Array1, Array2, Array3, ShapeBuilder, arr1, arr2};

use crate::BesselError;

use super::*;

fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual {actual} expected {expected}"
    );
}

fn assert_close_tol(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() < tolerance,
        "actual {actual} expected {expected}"
    );
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

fn hole_orbital_source() -> (Array1<Real>, Array1<Real>) {
    let mut large = Array1::<Real>::zeros(251);
    let mut small = Array1::<Real>::zeros(251);
    for index in 0..15 {
        let i = index as Real + 1.0;
        large[index] = 0.1 + 0.017 * i + 0.0009 * i * i + 0.002 * (0.3 * i).sin();
        small[index] = -0.04 + 0.011 * i - 0.0004 * i * i + 0.001 * (0.25 * i).cos();
    }
    (large, small)
}

fn phase_mesh84_input(spectroscopy: i32) -> XsphPhaseEnergyMesh84Input {
    XsphPhaseEnergyMesh84Input {
        spectroscopy,
        edge: -0.4,
        reference_energy: 9.0,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        core_valence_separation: -1.5,
        max_wave_number: 18.0 * XSPH_BOHR_ANGSTROM,
        wave_number_step: 0.5 * XSPH_BOHR_ANGSTROM,
        xanes_energy_step: 0.02,
        capacity: 120,
    }
}

struct XsphSpectrumFixture {
    index_map: Array1<i32>,
    final_lj: Array1<i32>,
    radial_integrals: Array1<Complex>,
    q_cosines: Array2<Real>,
    transition_weights: Array3<Real>,
}

fn xsph_spectrum_fixture() -> XsphSpectrumFixture {
    let index_map = arr1(&[1, -1, 2, 1, -2]);
    let final_lj = arr1(&[0, 1, 2, 3, 1]);
    let radial_integrals = arr1(&[
        Complex::new(0.12, -0.03),
        Complex::new(-0.08, 0.19),
        Complex::new(0.31, 0.07),
        Complex::new(-0.22, -0.11),
    ]);
    let q_cosines = arr2(&[[0.25, -0.35], [0.60, -0.40]]);
    let mut transition_weights = Array3::<Real>::zeros((2, 5, 4).f());
    for state in 0..5 {
        let state_feff = state as Real + 1.0;
        for spin in 0..2 {
            let spin_feff = spin as Real;
            for (magnetic_index, magnetic_j2) in [-3, -1, 1, 3].iter().enumerate() {
                let magnetic = Real::from(*magnetic_j2);
                transition_weights[(spin, state, magnetic_index)] = 0.05 * state_feff
                    + 0.11 * spin_feff
                    + 0.017 * magnetic
                    + 0.003 * state_feff * magnetic;
            }
        }
    }

    XsphSpectrumFixture {
        index_map,
        final_lj,
        radial_integrals,
        q_cosines,
        transition_weights,
    }
}

fn acoef_sum(coefficients: &Array5<Real>, operator: usize, lmax: usize) -> Real {
    let mut total = 0.0;
    let ml_count = 2 * lmax + 1;
    for l in 0..=lmax {
        for branch_2 in 0..2 {
            for branch_1 in 0..2 {
                for ml_index in 0..ml_count {
                    total += coefficients[(ml_index, branch_1, branch_2, operator, l)];
                }
            }
        }
    }
    total
}

fn acoef_entry(
    coefficients: &Array5<Real>,
    lmax: usize,
    magnetic_l: i32,
    branch_1: usize,
    branch_2: usize,
    operator: usize,
    l: usize,
) -> Real {
    let ml_index = (magnetic_l + lmax as i32) as usize;
    coefficients[(ml_index, branch_1 - 1, branch_2 - 1, operator - 1, l)]
}

#[test]
fn xsph_minimize_calculations_matches_feff_reference() -> Result<(), XsphError> {
    let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
    let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

    let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 8)?;

    assert_eq!(plan.max_lj, 6);
    assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2, 4, -3, -1]));
    assert_eq!(
        plan.calculations,
        arr2(&[[2, 5, 1], [4, 4, 2], [-3, 6, 1], [5, 0, 0]])
    );
    Ok(())
}

#[test]
fn xsph_minimize_calculations_honors_active_prefix() -> Result<(), XsphError> {
    let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
    let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

    let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 5)?;

    assert_eq!(plan.max_lj, 5);
    assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2]));
    assert_eq!(plan.calculations, arr2(&[[2, 5, 1], [4, 4, 2], [-3, 3, 1]]));
    Ok(())
}

#[test]
fn xsph_lj_needed_flags_match_feff_reference() -> Result<(), XsphError> {
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);
    let index_map = arr1(&[1, 2, -1, 3, -2, 4, -3, -1]);

    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 1)?,
        arr1(&[0, 1, 1, 0, 0, 1, 0])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 2)?,
        arr1(&[0, 1, 0, 0, 1, 0, 0])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 3)?,
        arr1(&[0, 0, 0, 1, 0, 0, 1])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 4)?,
        arr1(&[1, 0, 0, 0, 0, 0, 0])
    );
    Ok(())
}

#[test]
fn xsph_q_bessel_table_matches_feff_reference() -> Result<(), XsphError> {
    let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
    let table = xsph_q_bessel_table(0.35, radii.view(), 4)?;

    assert_eq!(table.shape(), &[4, 5]);
    assert_eq!(table.strides(), &[1, 4]);
    let expected = arr2(&[
        [
            9.997_958_458_381_769e-1,
            1.166_523_756_252_462e-2,
            8.165_952_107_648_562e-5,
            4.083_055_447_551_5e-7,
            1.587_874_544_380_937_5e-9,
        ],
        [
            9.797_080_213_012_896e-1,
            1.152_437_384_397_447_3e-1,
            8.095_451_039_379_387e-3,
            4.055_621_228_179_726_3e-4,
            1.579_141_698_006_595_3e-5,
        ],
        [
            8.261_173_577_085_878e-1,
            3.129_012_474_446_291e-1,
            6.788_620_641_892_411e-2,
            1.036_640_216_929_531_6e-2,
            1.223_141_376_378_009_1e-3,
        ],
        [
            9.385_522_838_839_835e-2,
            -9.429_243_227_927_261e-2,
            -1.342_662_707_938_009e-1,
            -1.612_046_859_156_612_8e-3,
            1.326_542_239_346_443e-1,
        ],
    ]);
    for ((row, column), &expected_value) in expected.indexed_iter() {
        assert_close(table[(row, column)], expected_value);
    }
    Ok(())
}

#[test]
fn xsph_q_bessel_table_applies_feff_large_argument_cutoff() -> Result<(), XsphError> {
    let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
    let table = xsph_q_bessel_table(1.0e8, radii.view(), 4)?;

    let expected_first_row = [
        4.205_477_931_907_825e-8,
        9.072_704_282_365_188e-8,
        -4.205_475_210_096_54e-8,
        -9.072_706_385_102_794e-8,
        4.205_468_859_202_071e-8,
    ];
    for (column, &expected_value) in expected_first_row.iter().enumerate() {
        assert_close(table[(0, column)], expected_value);
    }
    for row in 1..4 {
        for column in 0..5 {
            assert_close(table[(row, column)], 0.0);
        }
    }
    Ok(())
}

#[test]
fn xsph_occupation_normalization_matches_feff_getoccnorm_reference() -> Result<(), XsphError> {
    let cases = [
        (1, 1, 0.5),
        (6, 4, 0.25),
        (8, 4, 0.5),
        (26, 9, 1.0 / 3.0),
        (29, 10, 0.5),
        (47, 17, 0.5),
        (58, 15, 1.0 / 6.0),
        (79, 24, 0.5),
        (80, 24, 1.0),
        (92, 22, 0.5),
        (100, 16, 1.0),
        (100, 29, 1.0),
    ];

    for (atomic_number, hole_index, expected) in cases {
        let actual = xsph_occupation_normalization(atomic_number, hole_index)?;
        assert_close_tol(actual, expected, 5.0e-13);
    }
    Ok(())
}

#[test]
fn xsph_occupation_normalization_rejects_invalid_inputs() {
    assert_eq!(
        xsph_occupation_normalization(0, 1),
        Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number: 0,
            max_atomic_number: 100,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(101, 1),
        Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number: 101,
            max_atomic_number: 100,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(26, 0),
        Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index: 0,
            max_hole_index: 29,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(26, 30),
        Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index: 30,
            max_hole_index: 29,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(92, 27),
        Err(XsphError::ZeroOccupationNormDenominator { hole_index: 27 })
    );
}

#[test]
fn xsph_initial_hole_orbital_matches_feff_getholeorb0_reference() -> Result<(), XsphError> {
    let (large_source, small_source) = hole_orbital_source();
    let orbital = xsph_initial_hole_orbital(XsphHoleOrbitalInput {
        large_component: large_source.view(),
        small_component: small_source.view(),
        original_step: 0.05,
        new_step: 0.035,
        output_count: 12,
        output_capacity: 16,
    })?;

    assert_eq!(orbital.active_count, 12);
    assert_eq!(orbital.source_count, 16);
    let expected_large = [
        1.184_910_404_133_227e-1,
        1.324_776_258_980_802e-1,
        1.473_025_254_311_882e-1,
        1.629_521_321_304_354e-1,
        1.794_130_641_284_993e-1,
        1.966_760_790_140_799e-1,
        2.147_356_523_616_736e-1,
        2.335_893_236_221_459e-1,
        2.532_385_421_404_321e-1,
        2.736_894_375_417_349e-1,
        2.949_509_263_611_023e-1,
        3.170_346_434_301_027e-1,
    ];
    let expected_small = [
        -2.843_108_757_828_936e-2,
        -2.154_487_654_885_214e-2,
        -1.507_873_522_927_23e-2,
        -9.029_598_949_854_223e-3,
        -3.394_352_127_428_596e-3,
        1.831_137_246_489_437e-3,
        6.651_487_119_769_044e-3,
        1.107_164_454_957_789e-2,
        1.509_688_426_219_543e-2,
        1.873_254_703_597_43e-2,
        2.198_385_316_345_285e-2,
        2.485_593_326_716_564e-2,
    ];
    for index in 0..12 {
        assert_close_tol(
            orbital.large_component[index],
            expected_large[index],
            5.0e-14,
        );
        assert_close_tol(
            orbital.small_component[index],
            expected_small[index],
            5.0e-14,
        );
    }
    for index in 12..16 {
        assert_close(orbital.large_component[index], 0.0);
        assert_close(orbital.small_component[index], 0.0);
    }
    Ok(())
}

#[test]
fn xsph_initial_hole_orbital_rejects_invalid_inputs() {
    let (large_source, small_source) = hole_orbital_source();
    let small_short: Array1<_> = small_source.iter().take(250).copied().collect();
    assert_eq!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: large_source.view(),
            small_component: small_short.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 12,
            output_capacity: 16,
        }),
        Err(XsphError::HoleOrbitalLengthMismatch {
            large_len: 251,
            small_len: 250,
        })
    );
    assert_eq!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: large_source.view(),
            small_component: small_source.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 17,
            output_capacity: 16,
        }),
        Err(XsphError::InvalidHoleOrbitalOutputCount {
            output_count: 17,
            output_capacity: 16,
        })
    );
    let zero = Array1::<Real>::zeros(251);
    assert_eq!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: zero.view(),
            small_component: zero.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 12,
            output_capacity: 16,
        }),
        Err(XsphError::EmptyHoleOrbital)
    );
    let mut bad = large_source.clone();
    bad[4] = Real::NAN;
    assert!(matches!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: bad.view(),
            small_component: small_source.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 12,
            output_capacity: 16,
        }),
        Err(XsphError::NonFiniteScalar {
            name: "large_component",
            ..
        })
    ));
}

#[test]
fn xsph_phase_energy_mesh_84_matches_feff_phmesh2_reference() -> Result<(), XsphError> {
    let exafs = xsph_phase_energy_mesh_84(phase_mesh84_input(0))?;
    assert_eq!(exafs.energies.len(), 79);
    assert_eq!(exafs.horizontal_count, 57);
    assert_eq!(exafs.extension_count, 0);
    assert_eq!(exafs.zero_index, 0);
    assert_close(exafs.xloss, 0.05);
    assert_complex_close(exafs.energies[0], Complex::new(-0.4, 0.05));
    assert_complex_close(
        exafs.energies[56],
        Complex::new(44.964_626_859_191_69, 0.05),
    );
    assert_complex_close(
        exafs.energies[57],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        exafs.energies[78],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let xanes = xsph_phase_energy_mesh_84(phase_mesh84_input(1))?;
    assert_eq!(xanes.energies.len(), 69);
    assert_eq!(xanes.horizontal_count, 47);
    assert_eq!(xanes.extension_count, 0);
    assert_eq!(xanes.zero_index, 10);
    assert_complex_close(
        xanes.energies[0],
        Complex::new(-11.741_156_714_797_922, 0.05),
    );
    assert_complex_close(xanes.energies[10], Complex::new(-0.4, 0.05));
    assert_complex_close(
        xanes.energies[46],
        Complex::new(44.964_626_859_191_69, 0.05),
    );
    assert_complex_close(
        xanes.energies[47],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    let no_fms = xsph_phase_energy_mesh_84(phase_mesh84_input(-1))?;
    assert_eq!(no_fms.energies.len(), 89);
    assert_eq!(no_fms.horizontal_count, 67);
    assert_eq!(no_fms.extension_count, 0);
    assert_eq!(no_fms.zero_index, 10);
    assert_complex_close(
        no_fms.energies[0],
        Complex::new(-11.741_156_714_797_922, 0.05),
    );
    assert_complex_close(no_fms.energies[10], Complex::new(-0.4, 0.05));
    assert_complex_close(
        no_fms.energies[11],
        Complex::new(-3.985_998_571_957_039_5e-1, 0.05),
    );
    assert_complex_close(
        no_fms.energies[66],
        Complex::new(44.964_626_859_191_69, 0.05),
    );
    assert_complex_close(
        no_fms.energies[67],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        no_fms.energies[88],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let xes_input = XsphPhaseEnergyMesh84Input {
        spectroscopy: 2,
        max_wave_number: -5.0,
        wave_number_step: 10.0,
        xanes_energy_step: 0.25,
        ..phase_mesh84_input(2)
    };
    let xes = xsph_phase_energy_mesh_84(xes_input)?;
    assert_eq!(xes.energies.len(), 27);
    assert_eq!(xes.horizontal_count, 5);
    assert_eq!(xes.extension_count, 0);
    assert_eq!(xes.zero_index, 0);
    assert_complex_close(xes.energies[0], Complex::new(-0.4, 0.05));
    assert_complex_close(
        xes.energies[1],
        Complex::new(-9.723_062_142_400_252e-2, 0.05),
    );
    assert_complex_close(
        xes.energies[4],
        Complex::new(6.527_693_785_759_748e-1, 0.05),
    );
    assert_complex_close(
        xes.energies[5],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    let danes = xsph_phase_energy_mesh_84(phase_mesh84_input(3))?;
    assert_eq!(danes.energies.len(), 119);
    assert_eq!(danes.horizontal_count, 47);
    assert_eq!(danes.extension_count, 50);
    assert_eq!(danes.zero_index, 10);
    assert_complex_close(
        danes.energies[69],
        Complex::new(47.449_880_336_817_16, 2.0e-8),
    );
    assert_complex_close(
        danes.energies[118],
        Complex::new(60_495.181_164_572_314, 2.0e-8),
    );

    let no_fms_danes = xsph_phase_energy_mesh_84(phase_mesh84_input(-3))?;
    assert_eq!(no_fms_danes.energies.len(), 119);
    assert_eq!(no_fms_danes.horizontal_count, 67);
    assert_eq!(no_fms_danes.extension_count, 30);
    assert_eq!(no_fms_danes.zero_index, 10);
    assert_complex_close(
        no_fms_danes.energies[89],
        Complex::new(49.865_126_674_227_82, 2.0e-8),
    );
    assert_complex_close(
        no_fms_danes.energies[118],
        Complex::new(54_977.881_757_676_99, 2.0e-8),
    );

    let fprime_input = XsphPhaseEnergyMesh84Input {
        spectroscopy: 4,
        max_wave_number: -5.0,
        wave_number_step: 10.0,
        xanes_energy_step: 0.25,
        ..phase_mesh84_input(4)
    };
    let fprime = xsph_phase_energy_mesh_84(fprime_input)?;
    assert_eq!(fprime.energies.len(), 105);
    assert_eq!(fprime.horizontal_count, 5);
    assert_eq!(fprime.extension_count, 100);
    assert_eq!(fprime.zero_index, 0);
    assert_complex_close(
        fprime.energies[0],
        Complex::new(-9.347_230_621_424_002, 0.0),
    );
    assert_complex_close(fprime.energies[5], Complex::new(-0.4, 0.0));
    assert_complex_close(fprime.energies[104], Complex::new(171.0, 0.0));

    Ok(())
}

#[test]
fn xsph_user_phase_energy_mesh_matches_feff_phmesh2_reference() -> Result<(), XsphError> {
    let points = arr1(&[
        Complex::new(-5.0, 0.2),
        Complex::new(0.0004, 0.0),
        Complex::new(12.0, -0.1),
    ]);
    let records = [
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
        XsphPhaseUserGridRecord::User(points.view()),
    ];
    let mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
        spectroscopy: 1,
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        records: &records,
        capacity: 120,
    })?;
    assert_eq!(mesh.energies.len(), 31);
    assert_eq!(mesh.horizontal_count, 9);
    assert_eq!(mesh.extension_count, 0);
    assert_eq!(mesh.zero_index, 3);
    assert_complex_close(
        mesh.energies[0],
        Complex::new(-5.837_465_450_137_141e-1, 0.05),
    );
    assert_complex_close(mesh.energies[3], Complex::new(-0.4, 0.05));
    assert_complex_close(
        mesh.energies[6],
        Complex::new(1.640_061_233_229_339_6e-2, 0.05),
    );
    assert_complex_close(
        mesh.energies[8],
        Complex::new(6.393_311_675_183_092e-1, 0.05),
    );
    assert_complex_close(
        mesh.energies[9],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        mesh.energies[30],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let exp_points = arr1(&[Complex::new(-1.0, 0.0), Complex::new(0.002, 0.0)]);
    let exp_records = [
        XsphPhaseUserGridRecord::User(exp_points.view()),
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::Exponential,
            minimum: XsphPhaseUserGridMinimum::Last,
            maximum: 20.0,
            step: 0.5,
        }),
    ];
    let exp_mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
        spectroscopy: 1,
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        records: &exp_records,
        capacity: 120,
    })?;
    assert_eq!(exp_mesh.energies.len(), 54);
    assert_eq!(exp_mesh.horizontal_count, 32);
    assert_eq!(exp_mesh.extension_count, 0);
    assert_eq!(exp_mesh.zero_index, 1);
    assert_complex_close(
        exp_mesh.energies[0],
        Complex::new(-4.367_493_090_027_428_4e-1, 0.05),
    );
    assert_complex_close(exp_mesh.energies[1], Complex::new(-0.4, 0.05));
    assert_complex_close(
        exp_mesh.energies[31],
        Complex::new(3.222_548_489_185_893_5e-1, 0.05),
    );
    assert_complex_close(
        exp_mesh.energies[32],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        exp_mesh.energies[53],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let danes_mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
        spectroscopy: -3,
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        records: &exp_records,
        capacity: 120,
    })?;
    assert_eq!(danes_mesh.energies.len(), 119);
    assert_eq!(danes_mesh.horizontal_count, 32);
    assert_eq!(danes_mesh.extension_count, 65);
    assert_eq!(danes_mesh.zero_index, 1);
    assert_complex_close(
        danes_mesh.energies[54],
        Complex::new(3.532_758_355_442_124e-1, 2.0e-8),
    );
    assert_complex_close(
        danes_mesh.energies[118],
        Complex::new(58_023.774_523_482_745, 2.0e-8),
    );

    Ok(())
}

#[test]
fn xsph_thermal_phase_energy_mesh_matches_feff_phmesh2t_reference() -> Result<(), XsphError> {
    let thermal = xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        core_valence_separation: -1.5,
        electronic_temperature: 5.0,
        user_records: None,
        capacity: 240,
    })?;
    assert_eq!(thermal.energies.len(), 72);
    assert_eq!(thermal.horizontal_count, 30);
    assert_eq!(thermal.pole_count, 1);
    assert_eq!(thermal.zero_index, 5);
    assert_complex_close(
        thermal.energies[0],
        Complex::new(-1.25, 1.154_513_591_875_180_8),
    );
    assert_complex_close(
        thermal.energies[5],
        Complex::new(0.0, 1.154_513_591_875_180_8),
    );
    assert_complex_close(thermal.energies[30], Complex::new(-1.25, 0.05));
    assert_complex_close(
        thermal.energies[60],
        Complex::new(-1.5, 1.154_513_591_875_180_7e-2),
    );
    assert_complex_close(
        thermal.energies[70],
        Complex::new(-0.4, 5.772_567_959_375_904e-1),
    );
    assert_complex_close(
        thermal.energies[71],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    let points = arr1(&[
        Complex::new(-5.0, 0.2),
        Complex::new(0.0004, 0.0),
        Complex::new(12.0, -0.1),
    ]);
    let records = [
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
        XsphPhaseUserGridRecord::User(points.view()),
    ];
    let user = xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        core_valence_separation: -1.5,
        electronic_temperature: 5.0,
        user_records: Some(&records),
        capacity: 240,
    })?;
    assert_eq!(user.energies.len(), 30);
    assert_eq!(user.horizontal_count, 9);
    assert_eq!(user.pole_count, 1);
    assert_eq!(user.zero_index, 3);
    assert_complex_close(
        user.energies[0],
        Complex::new(-5.837_465_450_137_141e-1, 1.154_513_591_875_180_8),
    );
    assert_complex_close(
        user.energies[8],
        Complex::new(6.393_311_675_183_092e-1, 1.154_513_591_875_180_8),
    );
    assert_complex_close(
        user.energies[18],
        Complex::new(-1.5, 1.154_513_591_875_180_7e-2),
    );
    assert_complex_close(
        user.energies[28],
        Complex::new(-0.4, 5.772_567_959_375_904e-1),
    );
    assert_complex_close(
        user.energies[29],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    Ok(())
}

#[test]
fn xsph_phase_mesh_primitives_match_feff_phmesh2_reference() -> Result<(), XsphError> {
    let even = xsph_even_energy_mesh(-0.2, 0.35, 0.11, 4)?;
    assert_eq!(even.len(), 4);
    for (&actual, expected) in even.iter().zip([-0.2, -0.09, 0.02, 0.13]) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let k_mesh = xsph_k_energy_mesh(-1.2, -0.2, 0.25, 32)?;
    let expected_k = [-0.72, -0.45125, -0.245, -0.10125, -0.02];
    assert_eq!(k_mesh.len(), expected_k.len());
    for (&actual, expected) in k_mesh.iter().zip(expected_k) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let exp_mesh = xsph_exponential_energy_mesh(0.02, 0.5, 0.4, 32)?;
    let expected_exp = [
        2e-2,
        2.983_649_395_282_541e-2,
        4.451_081_856_984_936e-2,
        6.640_233_845_473_097e-2,
        9.906_064_848_790_23e-2,
        1.477_811_219_786_13e-1,
        2.204_635_276_128_321e-1,
        3.288_929_354_219_411e-1,
        4.906_506_039_421_871e-1,
    ];
    assert_eq!(exp_mesh.len(), expected_exp.len());
    for (&actual, expected) in exp_mesh.iter().zip(expected_exp) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let vertical = xsph_vertical_energy_mesh_84(0.05, 32)?;
    let expected_vertical = [
        1.837_465_409_066_59e-4,
        3.674_930_818_133_18e-4,
        4.927_048_004_095_16e-4,
        7.350_291_898_973_28e-4,
        1.096_534_698_976_09e-3,
        1.635_837_545_753_17e-3,
        2.440_382_852_083_45e-3,
        3.640_623_410_438_34e-3,
        5.431_171_918_502_91e-3,
        8.102_356_405_158_36e-3,
        1.208_729_539_430_72e-2,
        1.803_212_579_691_30e-2,
        2.690_077_061_480_91e-2,
        4.013_123_398_875_48e-2,
        5.986_876_601_124_52e-2,
        8.931_370_375_288_18e-2,
        1.332_403_890_963_65e-1,
        1.987_713_031_772_90e-1,
        2.965_319_392_622_22e-1,
        4.423_736_706_308_43e-1,
        6.599_439_674_333_17e-1,
        9.845_207_096_763_88e-1,
    ];
    assert_eq!(vertical.len(), expected_vertical.len());
    for (&actual, expected) in vertical.iter().zip(expected_vertical) {
        assert_complex_close(actual, Complex::new(0.0, expected));
    }
    let clipped_vertical = xsph_vertical_energy_mesh_84(0.05, 4)?;
    assert_eq!(clipped_vertical.len(), 4);
    for (&actual, expected) in clipped_vertical.iter().zip(expected_vertical) {
        assert_complex_close(actual, Complex::new(0.0, expected));
    }

    let exafs84 = xsph_exafs_energy_grid_84(18.0 * XSPH_BOHR_ANGSTROM, 160)?;
    let expected_exafs84 = [
        0.0,
        1.400_142_804_296_04e-3,
        5.600_571_217_184_16e-3,
        1.260_128_523_866_44e-2,
        2.240_228_486_873_66e-2,
        3.500_357_010_740_10e-2,
        5.040_514_095_465_74e-2,
        6.860_699_741_050_60e-2,
        8.960_913_947_494_66e-2,
        1.134_115_671_479_79e-1,
        1.400_142_804_296_04e-1,
        1.694_172_793_198_21e-1,
        2.016_205_638_186_30e-1,
        2.366_241_339_260_31e-1,
        2.744_279_896_420_24e-1,
        3.150_321_309_666_09e-1,
        3.584_365_578_997_86e-1,
        4.046_412_704_415_56e-1,
        4.536_462_685_919_17e-1,
        5.054_515_523_508_70e-1,
        5.600_571_217_184_16e-1,
        6.776_691_172_792_83e-1,
        8.064_822_552_745_19e-1,
        9.464_965_357_041_23e-1,
        1.097_711_958_568_10,
        1.260_128_523_866_44,
        1.433_746_231_599_14,
        1.618_565_081_766_22,
        1.814_585_074_367_67,
        2.021_806_209_403_48,
        2.240_228_486_873_66,
        2.469_851_906_778_21,
        2.710_676_469_117_13,
        2.962_702_173_890_42,
        3.225_929_021_098_08,
        3.500_357_010_740_10,
        3.785_986_142_816_49,
        4.082_816_417_327_25,
        4.390_847_834_272_38,
        4.710_080_393_651_88,
        5.040_514_095_465_74,
        5.915_603_348_150_77,
        6.860_699_741_050_60,
        7.875_803_274_165_22,
        8.960_913_947_494_65,
        10.116_031_761_038_9,
        11.341_156_714_797_9,
        12.636_288_808_771_8,
        14.001_428_042_960_4,
        16.941_727_931_982_1,
        20.162_056_381_863_0,
        23.662_413_392_603_1,
        27.442_798_964_202_4,
        31.503_213_096_660_9,
        35.843_655_789_978_6,
        40.464_127_044_155_6,
        45.364_626_859_191_7,
    ];
    assert_eq!(exafs84.len(), expected_exafs84.len());
    for (&actual, expected) in exafs84.iter().zip(expected_exafs84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    let clipped_exafs84 = xsph_exafs_energy_grid_84(18.0 * XSPH_BOHR_ANGSTROM, 45)?;
    assert_eq!(clipped_exafs84.len(), 45);
    for (&actual, expected) in clipped_exafs84.iter().zip(expected_exafs84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let xanes84 =
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.5 * XSPH_BOHR_ANGSTROM, 0.02, 80)?;
    let expected_xanes84 = [
        -11.341_156_714_797_9,
        -8.960_913_947_494_65,
        -6.860_699_741_050_60,
        -5.040_514_095_465_74,
        -3.500_357_010_740_10,
        -2.240_228_486_873_66,
        -1.260_128_523_866_44,
        -5.600_571_217_184_16e-1,
        -1.400_142_804_296_04e-1,
        -2.0e-2,
        0.0,
        3.500_357_010_740_10e-2,
        1.400_142_804_296_04e-1,
        3.150_321_309_666_09e-1,
        5.600_571_217_184_16e-1,
        8.750_892_526_850_25e-1,
        1.260_128_523_866_44,
        1.715_174_935_262_65,
        2.240_228_486_873_66,
    ];
    assert_eq!(xanes84.zero_index, 10);
    assert_eq!(xanes84.energies.len(), expected_xanes84.len());
    for (&actual, expected) in xanes84.energies.iter().zip(expected_xanes84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    let clipped_xanes84 =
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.5 * XSPH_BOHR_ANGSTROM, 0.02, 15)?;
    assert_eq!(clipped_xanes84.zero_index, 10);
    assert_eq!(clipped_xanes84.energies.len(), 13);
    for (&actual, expected) in clipped_xanes84.energies.iter().zip(expected_xanes84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let xes84 = xsph_xes_energy_grid_84(-5.0, 10.0, 0.25, -0.4, 64)?;
    let expected_xes84 = [
        0.0,
        3.027_693_785_759_975e-1,
        5.527_693_785_759_975e-1,
        8.027_693_785_759_975e-1,
        1.052_769_378_575_997_5,
    ];
    assert_eq!(xes84.zero_index, 0);
    assert_eq!(xes84.energies.len(), expected_xes84.len());
    for (&actual, expected) in xes84.energies.iter().zip(expected_xes84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let fprime84 = xsph_fprime_energy_grid_84(-5.0, 10.0, 0.25, 9.0, -0.4, 32)?;
    let expected_fprime84 = [
        -9.347_230_621_424_00,
        -9.097_230_621_424_00,
        -8.847_230_621_424_00,
        -8.597_230_621_424_00,
        -8.347_230_621_424_00,
        -4.0e-1,
        -2.897_520_729_917_72e-1,
        -1.795_041_459_835_43e-1,
        -6.925_621_897_531_47e-2,
        4.099_170_803_291_38e-2,
        1.512_396_350_411_42e-1,
        2.614_875_620_493_71e-1,
        3.717_354_890_575_99e-1,
        5.133_096_128_907_97e-1,
        7.088_017_325_278_11e-1,
        9.787_463_227_214_28e-1,
        1.351_498_339_069_21,
        1.866_211_620_012_10,
        2.576_951_602_520_49,
        3.558_374_350_755_49,
        4.913_568_422_367_72,
        6.784_883_281_367_88,
        9.368_881_673_088_11,
        12.936_986_557_362_0,
        17.863_991_351_936_7,
        24.667_428_199_534_5,
        34.061_929_497_811_9,
        47.034_292_822_459_8,
        64.947_134_056_249_1,
        89.682_016_439_421_3,
        123.837_089_804_069,
        171.0,
    ];
    assert_eq!(fprime84.regular_count, 5);
    assert_eq!(fprime84.kk_count, 27);
    assert_eq!(fprime84.energies.len(), expected_fprime84.len());
    for (&actual, expected) in fprime84.energies.iter().zip(expected_fprime84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    let fprime_auto_step = xsph_fprime_energy_grid_84(-5.0, 10.0, 0.0, 9.0, -0.4, 140)?;
    assert_eq!(fprime_auto_step.regular_count, 100);
    assert_eq!(fprime_auto_step.kk_count, 40);
    assert_eq!(fprime_auto_step.energies.len(), 140);
    assert_complex_close(
        fprime_auto_step.energies[0],
        Complex::new(expected_fprime84[0], 0.0),
    );
    assert_complex_close(fprime_auto_step.energies[139], Complex::new(171.0, 0.0));

    let reverse_input = arr1(&[
        Complex::new(1.0, 0.2),
        Complex::new(2.0, -0.1),
        Complex::new(-0.5, 0.0),
    ]);
    let reversed = xsph_reverse_energy_grid(reverse_input.view(), 0.25)?;
    let expected_reverse = [
        Complex::new(0.75, 0.0),
        Complex::new(-1.75, 0.1),
        Complex::new(-0.75, -0.2),
    ];
    for (&actual, expected) in reversed.iter().zip(expected_reverse) {
        assert_complex_close(actual, expected);
    }

    let sort_input = arr1(&[
        Complex::new(0.002, 9.0),
        Complex::new(-0.004, 8.0),
        Complex::new(0.0004, 7.0),
        Complex::new(0.0012, 6.0),
        Complex::new(-0.0036, 5.0),
        Complex::new(0.25, 4.0),
    ]);
    let sorted = xsph_sort_energy_grid(sort_input.view())?;
    assert_eq!(sorted.zero_index, 1);
    let expected_sorted = [-0.004, 0.0, 0.002, 0.25];
    assert_eq!(sorted.energies.len(), expected_sorted.len());
    for (&actual, expected) in sorted.energies.iter().zip(expected_sorted) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    Ok(())
}

#[test]
fn xsph_phase_mesh_primitives_reject_invalid_inputs() {
    assert_eq!(
        xsph_even_energy_mesh(0.0, 1.0, 0.1, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_even_energy_mesh(0.0, 1.0, 0.0, 4),
        Err(XsphError::InvalidPhaseMeshStep {
            name: "energy_step",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_exponential_energy_mesh(0.0, 1.0, 0.4, 4),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "min_energy",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_vertical_energy_mesh_84(0.05, 1),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 1 })
    );
    assert_eq!(
        xsph_vertical_energy_mesh_84(0.0, 4),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "xloss",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_exafs_energy_grid_84(18.0 * XSPH_BOHR_ANGSTROM, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_exafs_energy_grid_84(0.0, 4),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "max_wave_number",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.5 * XSPH_BOHR_ANGSTROM, 0.02, 10),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 10 })
    );
    assert_eq!(
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.0, 0.02, 80),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "wave_number_step",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_fprime_energy_grid_84(-5.0, 10.0, 0.25, 9.0, -0.4, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_xes_energy_grid_84(-5.0, 10.0, 0.25, -0.4, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert!(matches!(
        xsph_fprime_energy_grid_84(-5.0, 10.0, Real::NAN, 9.0, -0.4, 32),
        Err(XsphError::NonFiniteScalar {
            name: "energy_step",
            value,
        }) if value.is_nan()
    ));
    assert_eq!(
        xsph_phase_energy_mesh_84(XsphPhaseEnergyMesh84Input {
            capacity: 0,
            ..phase_mesh84_input(0)
        }),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_phase_energy_mesh_84(phase_mesh84_input(6)),
        Err(XsphError::UnsupportedPhaseMeshSpectroscopy { spectroscopy: 6 })
    );
    assert_eq!(
        xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: 1,
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            records: &[],
            capacity: 120,
        }),
        Err(XsphError::EmptyPhaseGridRecords)
    );
    let too_many = [XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
        kind: XsphPhaseUserGridKind::Energy,
        minimum: XsphPhaseUserGridMinimum::Value(0.0),
        maximum: 1.0,
        step: 0.1,
    }); 11];
    assert_eq!(
        xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: 1,
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            records: &too_many,
            capacity: 120,
        }),
        Err(XsphError::TooManyPhaseGridRecords { count: 11, max: 10 })
    );
    assert_eq!(
        xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: 6,
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            records: &too_many[..1],
            capacity: 120,
        }),
        Err(XsphError::UnsupportedPhaseMeshSpectroscopy { spectroscopy: 6 })
    );
    assert_eq!(
        xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            core_valence_separation: -1.5,
            electronic_temperature: 0.0,
            user_records: None,
            capacity: 240,
        }),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "electronic_temperature",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            core_valence_separation: -1.5,
            electronic_temperature: 5.0,
            user_records: None,
            capacity: 4,
        }),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 4 })
    );
    let descending = xsph_even_energy_mesh(1.0, 0.0, 0.1, 4);
    assert_eq!(descending, Ok(Array1::zeros(0)));

    let empty = Array1::<Complex>::zeros(0);
    assert_eq!(
        xsph_sort_energy_grid(empty.view()),
        Err(XsphError::EmptyPhaseMesh)
    );

    let bad = arr1(&[Complex::new(0.0, 0.0), Complex::new(Real::NAN, 0.0)]);
    assert!(matches!(
        xsph_sort_energy_grid(bad.view()),
        Err(XsphError::NonFiniteComplex {
            name: "energies",
            index: 1,
            ..
        })
    ));
}

#[test]
fn xsph_longitudinal_multipole_factor_matches_feff_reference() -> Result<(), XsphError> {
    let cases = [
        (-1, -1, 0, -std::f64::consts::SQRT_2),
        (-1, 1, 1, 2.449_489_742_783_178),
        (1, -1, 1, 2.449_489_742_783_178),
        (-2, 1, 1, 0.0),
        (2, -1, 2, -4.472_135_954_999_58),
        (-3, 2, 3, 0.0),
        (3, -2, 2, 2.927_700_218_845_598),
        (-2, -2, 5, 0.0),
    ];

    for (kappa, kappa_prime, multipole_l, expected) in cases {
        let value = xsph_longitudinal_multipole_factor(kappa, kappa_prime, multipole_l)?;
        assert_close(value.re, expected);
        assert_close(value.im, 0.0);
    }
    Ok(())
}

#[test]
fn xsph_relativistic_multipole_factors_match_feff_reference() -> Result<(), XsphError> {
    let cases = [
        (-1, -1, 0, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        (
            1,
            -1,
            0,
            1,
            Complex::new(0.0, -8.164_965_809_277_261e-1),
            Complex::new(0.0, -2.449_489_742_783_178),
        ),
        (
            -2,
            -1,
            0,
            1,
            Complex::new(0.0, -2.309_401_076_758_503_4),
            Complex::new(0.0, 0.0),
        ),
        (2, -1, 2, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        (
            -2,
            1,
            1,
            2,
            Complex::new(-3.872_983_346_207_417_5, 0.0),
            Complex::new(-7.745_966_692_414_837e-1, 0.0),
        ),
        (
            3,
            -2,
            1,
            1,
            Complex::new(2.323_790_007_724_448_4, 0.0),
            Complex::new(2.323_790_007_724_45, 0.0),
        ),
        (
            -3,
            2,
            1,
            2,
            Complex::new(-3.549_647_869_859_77, 0.0),
            Complex::new(-1.521_277_658_511_329_2, 0.0),
        ),
        (2, -3, 3, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        (
            1,
            1,
            1,
            1,
            Complex::new(-2.449_489_742_783_178, 0.0),
            Complex::new(-2.449_489_742_783_178, 0.0),
        ),
        (
            -2,
            -2,
            1,
            1,
            Complex::new(3.098_386_676_965_933_6, 0.0),
            Complex::new(3.098_386_676_965_934, 0.0),
        ),
    ];

    for (kappa, kappa_prime, bessel_l, multipole_l, expected_pq, expected_qp) in cases {
        let factors =
            xsph_relativistic_multipole_factors(kappa, kappa_prime, bessel_l, multipole_l)?;
        assert_close(factors.p_q_prime.re, expected_pq.re);
        assert_close(factors.p_q_prime.im, expected_pq.im);
        assert_close(factors.q_p_prime.re, expected_qp.re);
        assert_close(factors.q_p_prime.im, expected_qp.im);
    }
    Ok(())
}

#[test]
fn xsph_relativistic_multipole_factors_return_zero_for_unmatched_orders() -> Result<(), XsphError> {
    let factors = xsph_relativistic_multipole_factors(-1, 1, 4, 1)?;

    assert_close(factors.p_q_prime.re, 0.0);
    assert_close(factors.p_q_prime.im, 0.0);
    assert_close(factors.q_p_prime.re, 0.0);
    assert_close(factors.q_p_prime.im, 0.0);
    Ok(())
}

#[test]
fn xsph_angular_density_coefficients_match_feff_acoef_reference() -> Result<(), XsphError> {
    let cases = [
        (
            0,
            [
                3.199_999_994_039_535_5e1,
                3.199_999_994_039_535_5e1,
                3.199_999_991_059_303_3e1,
            ],
            [
                (-3, 1, 1, 1, 3, 1.714_285_731_315_612_8),
                (0, 2, 2, 1, 0, 1.999_999_523_162_841_8),
                (-1, 1, 1, 3, 1, 1.333_333_134_651_184),
                (-2, 2, 2, 2, 3, 5.714_284_777_641_296e-1),
                (3, 1, 2, 1, 3, 0.0),
            ],
        ),
        (
            1,
            [
                7.999_999_996_274_71,
                -6.109_476_089_477_539e-7,
                -1.369_044_184_684_753_4e-7,
            ],
            [
                (-2, 1, 2, 3, 2, 2.285_714_261_233_806_6e-2),
                (-1, 2, 1, 2, 2, -2.400_000_095_367_431_6e-1),
                (0, 1, 2, 3, 1, -4.444_444_924_592_972e-2),
                (2, 2, 2, 3, 3, -4.081_631_451_845_169e-2),
                (-1, 1, 1, 3, 1, 1.777_777_522_802_353e-1),
            ],
        ),
        (
            -1,
            [
                -7.999_999_996_274_71,
                6.109_476_089_477_539e-7,
                1.406_297_087_669_372_6e-7,
            ],
            [
                (-1, 2, 1, 2, 2, 1.599_999_964_237_213e-1),
                (0, 1, 2, 3, 1, 4.444_444_924_592_972e-2),
                (1, 2, 1, 3, 1, 4.444_444_179_534_912e-2),
                (3, 1, 2, 1, 3, -1.224_489_733_576_774_6e-1),
                (2, 1, 1, 1, 2, -2.399_999_946_355_819_7e-1),
            ],
        ),
        (
            2,
            [
                7.999_999_996_274_71,
                1.599_999_997_019_767_8e1,
                9.999_999_787_658_453,
            ],
            [
                (-3, 1, 1, 1, 3, 3.061_224_520_206_451_4e-1),
                (-2, 1, 2, 3, 2, -3.725_290_298_461_914e-9),
                (1, 1, 1, 2, 2, 1.999_999_880_790_710_4e-1),
                (2, 2, 2, 3, 3, 8.571_425_676_345_825e-1),
                (-2, 2, 2, 2, 3, 2.857_142_388_820_648e-1),
            ],
        ),
        (
            -2,
            [
                -7.999_999_996_274_71,
                1.599_999_997_019_767_8e1,
                9.999_999_674_037_099,
            ],
            [
                (0, 2, 2, 1, 0, -4.999_998_807_907_104_5e-1),
                (1, 1, 1, 2, 2, 6.000_000_834_465_027e-1),
                (2, 2, 2, 3, 3, 2.857_142_388_820_648e-1),
                (3, 1, 2, 1, 3, -1.224_489_733_576_774_6e-1),
                (-2, 2, 2, 2, 3, 8.571_426_868_438_721e-1),
            ],
        ),
    ];

    for (spin_selector, expected_sums, expected_entries) in cases {
        let coefficients = xsph_angular_density_coefficients(spin_selector, 3)?;
        assert_eq!(coefficients.shape(), &[7, 2, 2, 3, 4]);
        assert_eq!(coefficients.strides(), &[1, 7, 14, 28, 84]);
        for (operator, &expected_sum) in expected_sums.iter().enumerate() {
            assert_close_tol(acoef_sum(&coefficients, operator, 3), expected_sum, 1.0e-6);
        }
        for (magnetic_l, branch_1, branch_2, operator, l, expected) in expected_entries {
            assert_close_tol(
                acoef_entry(
                    &coefficients,
                    3,
                    magnetic_l,
                    branch_1,
                    branch_2,
                    operator,
                    l,
                ),
                expected,
                1.0e-7,
            );
        }
    }
    Ok(())
}

#[test]
fn xsph_angular_density_coefficients_reject_invalid_inputs() {
    assert!(matches!(
        xsph_angular_density_coefficients(1, XSPH_MAX_LX + 1),
        Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum,
            ljmax
        }) if angular_momentum == XSPH_MAX_LX + 1 && ljmax == XSPH_MAX_LX
    ));
    assert!(matches!(
        xsph_angular_density_coefficients(i32::MIN, 1),
        Err(XsphError::IntegerOutOfRange {
            name: "spin_selector",
            value: i32::MIN
        })
    ));
}

#[test]
fn xsph_nrixs_transition_weights_match_feff_reference() -> Result<(), XsphError> {
    let lgind = arr1(&[0, 1, 2, 1, 3, 2, 4]);
    let ljind = arr1(&[0, 1, 1, 2, 2, 3, 3]);
    let weights = xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, lgind.view(), ljind.view(), 7)?;
    assert_eq!(weights.shape(), &[2, 7]);
    assert_eq!(weights.strides(), &[1, 2]);
    let expected = arr2(&[
        [
            0.0,
            -3.333_333_333_333_333_7e-1,
            3.162_277_660_168_380_5e-1,
            1.825_741_858_350_554_4e-1,
            -2.390_457_218_668_785e-1,
            -1.690_308_509_457_032e-1,
            1.992_047_682_223_989_4e-1,
        ],
        [
            -7.071_067_811_865_477e-1,
            2.357_022_603_955_158_7e-1,
            -2.581_988_897_471_612_6e-1,
            2.581_988_897_471_612_6e-1,
            2.070_196_678_027_061_4e-1,
            -2.070_196_678_027_061_4e-1,
            -1.781_741_612_749_495_3e-1,
        ],
    ]);
    for ((spin, channel), &expected_value) in expected.indexed_iter() {
        assert_close(weights[(spin, channel)], expected_value);
    }

    let lgind = arr1(&[1, 2, 1, 3, 2, 4, 3, 4]);
    let ljind = arr1(&[0, 1, 1, 2, 2, 3, 3, 4]);
    let weights = xsph_nrixs_transition_weights(2, -1, 4, 11, 4, lgind.view(), ljind.view(), 8)?;
    let expected = arr2(&[
        [
            4.082_482_904_638_632_4e-1,
            0.0,
            -1.054_092_553_389_460_6e-1,
            7.824_607_964_359_512e-2,
            0.0,
            0.0,
            1.106_566_670_344_975_2e-1,
            -9.390_602_830_316_835e-2,
        ],
        [
            2.886_751_345_948_13e-1,
            0.0,
            -7.453_559_924_999_303e-2,
            -9.035_079_029_052_508e-2,
            0.0,
            0.0,
            -1.277_753_129_999_878_7e-1,
            1.049_901_313_914_518_7e-1,
        ],
    ]);
    for ((spin, channel), &expected_value) in expected.indexed_iter() {
        assert_close(weights[(spin, channel)], expected_value);
    }

    let lgind = arr1(&[0, 1, 2, 2, 3]);
    let ljind = arr1(&[0, 1, 2, 2, 3]);
    let weights = xsph_nrixs_transition_weights(-2, 3, 4, 9, 3, lgind.view(), ljind.view(), 5)?;
    let expected = arr2(&[
        [0.0, 0.0, 2.0e-1, -1.309_307_341_415_953e-1, 0.0],
        [
            0.0,
            0.0,
            -1.000_000_000_000_000_2e-1,
            -2.618_614_682_831_905e-1,
            0.0,
        ],
    ]);
    for ((spin, channel), &expected_value) in expected.indexed_iter() {
        assert_close(weights[(spin, channel)], expected_value);
    }
    Ok(())
}

#[test]
fn xsph_update_nrixs_lg_spectrum_matches_feff_reference() -> Result<(), XsphError> {
    let index_map = arr1(&[1, -1, 2, 1, -2]);
    let orbital_l = arr1(&[0, 1, 2, 3, 4]);
    let final_lj = arr1(&[0, 1, 2, 3, 1]);
    let radial_integrals = arr1(&[
        Complex::new(0.12, -0.03),
        Complex::new(-0.08, 0.19),
        Complex::new(0.31, 0.07),
        Complex::new(-0.22, -0.11),
    ]);
    let q_cosines = arr2(&[[0.25, -0.35], [0.60, -0.40]]);
    let mut transition_weights = Array3::<Real>::zeros((2, 5, 4).f());
    for state in 0..5 {
        let state_feff = state as Real + 1.0;
        for spin in 0..2 {
            let spin_feff = spin as Real;
            for (magnetic_index, magnetic_j2) in [-3, -1, 1, 3].iter().enumerate() {
                let magnetic = Real::from(*magnetic_j2);
                transition_weights[(spin, state, magnetic_index)] = 0.05 * state_feff
                    + 0.11 * spin_feff
                    + 0.017 * magnetic
                    + 0.003 * state_feff * magnetic;
            }
        }
    }

    let q_weights = arr1(&[Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
    xsph_update_nrixs_lg_spectrum(
        XsphLgSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 1,
            index_map: index_map.view(),
            orbital_l: orbital_l.view(),
            final_lj: final_lj.view(),
            initial_j2: 3,
            transition_weights: transition_weights.view(),
            radial_integrals: radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: q_cosines.view(),
            mix_dff: false,
            mdff_mode: 0,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
    )?;
    let expected = [
        Complex::new(1.100_552_32e-2, -1.768_391_84e-2),
        Complex::new(1.004_038_768e-2, -2.057_271_974e-2),
        Complex::new(1.0e-2, -2.0e-2),
        Complex::new(1.156_784_538_79e-2, -2.277_795_550_092_5e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }

    let q_weights = arr1(&[Complex::new(0.7, -0.2), Complex::new(-0.15, 0.45)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(-0.03, 0.04));
    xsph_update_nrixs_lg_spectrum(
        XsphLgSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 0,
            index_map: index_map.view(),
            orbital_l: orbital_l.view(),
            final_lj: final_lj.view(),
            initial_j2: 3,
            transition_weights: transition_weights.view(),
            radial_integrals: radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: q_cosines.view(),
            mix_dff: true,
            mdff_mode: 1,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Irregular,
        },
        spectrum.view_mut(),
    )?;
    let expected = [
        Complex::new(-3.066_69e-2, 3.953_56e-2),
        Complex::new(-2.859_349_665e-2, 3.854_721_595e-2),
        Complex::new(-3.0e-2, 4.0e-2),
        Complex::new(-4.005_742_490_156_249e-2, 3.762_661_973_593_751e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }

    let q_weights = arr1(&[Complex::new(0.25, 0.1), Complex::new(0.3, -0.35)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(0.02, 0.01));
    xsph_update_nrixs_lg_spectrum(
        XsphLgSpectrumUpdateInput {
            calculation_index: 2,
            spin_index: 1,
            index_map: index_map.view(),
            orbital_l: orbital_l.view(),
            final_lj: final_lj.view(),
            initial_j2: 3,
            transition_weights: transition_weights.view(),
            radial_integrals: radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: q_cosines.view(),
            mix_dff: true,
            mdff_mode: 2,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
    )?;
    let expected = [
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(1.995_779_884_1e-2, 8.875_159_533_25e-3),
        Complex::new(2.0e-2, 1.0e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn xsph_update_nrixs_lj_and_atom_spectra_match_feff_reference() -> Result<(), XsphError> {
    let fixture = xsph_spectrum_fixture();

    let q_weights = arr1(&[Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
    let mut spectrum_norm = 0.02;
    xsph_update_nrixs_lj_spectrum(
        XsphLjSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 1,
            index_map: fixture.index_map.view(),
            final_lj: fixture.final_lj.view(),
            initial_j2: 3,
            transition_weights: fixture.transition_weights.view(),
            radial_integrals: fixture.radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: fixture.q_cosines.view(),
            mix_dff: false,
            mdff_mode: 0,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
        &mut spectrum_norm,
    )?;
    let expected = [
        Complex::new(1.100_552_319_999_999_9e-2, -1.768_391_84e-2),
        Complex::new(1.004_038_767_999_999_9e-2, -2.057_271_974e-2),
        Complex::new(1.0e-2, -2.0e-2),
        Complex::new(1.156_784_538_790_000_2e-2, -2.277_795_550_092_500_2e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    assert_close(spectrum_norm, 7.678_319_047_619_049e-2);

    let q_weights = arr1(&[Complex::new(0.7, -0.2), Complex::new(-0.15, 0.45)]);
    let mut spectrum = Array1::from_elem(4, Complex::new(-0.03, 0.04));
    let mut spectrum_norm = -0.01;
    xsph_update_nrixs_lj_spectrum(
        XsphLjSpectrumUpdateInput {
            calculation_index: 1,
            spin_index: 0,
            index_map: fixture.index_map.view(),
            final_lj: fixture.final_lj.view(),
            initial_j2: 3,
            transition_weights: fixture.transition_weights.view(),
            radial_integrals: fixture.radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: fixture.q_cosines.view(),
            mix_dff: true,
            mdff_mode: 1,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Irregular,
        },
        spectrum.view_mut(),
        &mut spectrum_norm,
    )?;
    let expected = [
        Complex::new(-3.066_689_999_999_999_7e-2, 3.953_560_000_000_000_4e-2),
        Complex::new(-2.859_349_665e-2, 3.854_721_595_000_000_5e-2),
        Complex::new(-3e-2, 4e-2),
        Complex::new(-4.005_742_490_156_249e-2, 3.762_661_973_593_750_5e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    assert_close(spectrum_norm, -1.0e-2);

    let q_weights = arr1(&[Complex::new(0.25, 0.1), Complex::new(0.3, -0.35)]);
    let mut spectrum = Array1::from_elem(5, Complex::new(0.02, 0.01));
    let mut spectrum_norm = 0.005;
    xsph_update_nrixs_atom_spectrum(
        XsphLjSpectrumUpdateInput {
            calculation_index: 2,
            spin_index: 1,
            index_map: fixture.index_map.view(),
            final_lj: fixture.final_lj.view(),
            initial_j2: 3,
            transition_weights: fixture.transition_weights.view(),
            radial_integrals: fixture.radial_integrals.view(),
            q_weights: q_weights.view(),
            q_cosines: fixture.q_cosines.view(),
            mix_dff: true,
            mdff_mode: 2,
            ljmax: 3,
            active_len: 5,
            mode: XsphSpectrumUpdateMode::Regular,
        },
        spectrum.view_mut(),
        &mut spectrum_norm,
    )?;
    let expected = [
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(1.995_779_884_1e-2, 8.875_159_533_25e-3),
        Complex::new(2.0e-2, 1.0e-2),
        Complex::new(1.969_139_016e-2, 1.094_586_912e-2),
    ];
    for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
    assert_close(spectrum_norm, 8.780_333_333_333_334e-3);

    Ok(())
}

#[test]
fn xsph_axafs_matches_feff_reference_output() -> Result<(), XsphError> {
    let energies = Array1::from_shape_fn(12, |index| {
        let i = index as Real + 1.0;
        Complex::new(0.015 * (i - 3.0).powi(2) + 0.012 * (i - 1.0), 0.002 * i)
    });
    let cross_section = Array1::from_shape_fn(12, |index| {
        let i = index as Real + 1.0;
        Complex::new(
            -0.03 * i,
            0.42 + 0.021 * i + 0.004 * i * i + 0.025 * (0.7 * i).sin(),
        )
    });

    let axafs = xsph_axafs(XsphAxafsInput {
        energies: energies.view(),
        cross_section: cross_section.view(),
        fermi_energy: 0.37,
        horizontal_count: 10,
        zero_wave_index: 2,
    })?;

    let expected = arr2(&[
        [10.803, 0.735, 0.439, 2.861_61e-1, 2.853_50e-1, 2.839_94e-3],
        [12.354, 2.286, 0.775, 3.059_48e-1, 3.037_03e-1, 7.391_87e-3],
        [14.721, 4.653, 1.105, 3.317_56e-1, 3.312_74e-1, 1.452_30e-3],
        [17.905, 7.837, 1.434, 3.666_23e-1, 3.675_12e-1, -2.418_88e-3],
        [
            21.905,
            11.837,
            1.763,
            4.111_97e-1,
            4.116_72e-1,
            -1.155_65e-3,
        ],
        [26.722, 16.653, 2.091, 4.634_28e-1, 4.628_25e-1, 1.303_51e-3],
        [
            32.354,
            22.286,
            2.419,
            5.195_33e-1,
            5.198_45e-1,
            -6.003_98e-4,
        ],
    ]);
    assert_eq!(axafs.rows.dim(), (7, XSPH_AXAFS_COLUMN_COUNT));
    for ((row, column), &expected_value) in expected.indexed_iter() {
        let tolerance = if column < 3 { 5.0e-4 } else { 5.0e-7 };
        assert_close_tol(axafs.rows[(row, column)], expected_value, tolerance);
    }
    Ok(())
}

#[test]
fn xsph_axafs_rejects_invalid_inputs() {
    let energies = Array1::from_vec(vec![Complex::new(0.0, 0.0); 5]);
    let cross_section = Array1::from_vec(vec![Complex::new(0.0, 1.0); 5]);

    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: energies.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 5,
            zero_wave_index: 5,
        }),
        Err(XsphError::InvalidAxafsGridIndex { .. })
    ));
    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: energies.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 4,
            zero_wave_index: 1,
        }),
        Err(XsphError::InsufficientAxafsPoints { point_count: 2 })
    ));

    let mut bad_energy = energies.clone();
    bad_energy[2] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: bad_energy.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 5,
            zero_wave_index: 1,
        }),
        Err(XsphError::NonFiniteComplex {
            name: "energies",
            index: 2,
            ..
        })
    ));

    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: energies.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 5,
            zero_wave_index: 1,
        }),
        Err(XsphError::SingularAxafsFit)
    ));
}

#[test]
fn xsph_planning_helpers_reject_invalid_inputs() {
    let kind = arr1(&[2]);
    let orbital_l = arr1(&[1]);
    let final_lj = arr1(&[2]);

    assert!(matches!(
        xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 0),
        Err(XsphError::EmptyIndexSet)
    ));
    assert!(matches!(
        xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 2),
        Err(XsphError::LengthTooShort { name: "kind", .. })
    ));

    let bad_lj = arr1(&[-1]);
    assert!(matches!(
        xsph_minimize_calculations(kind.view(), orbital_l.view(), bad_lj.view(), 1),
        Err(XsphError::NegativeAngularMomentum { .. })
    ));

    let index_map = arr1(&[1]);
    assert!(matches!(
        xsph_lj_needed_flags(1, final_lj.view(), index_map.view(), 1, 1),
        Err(XsphError::AngularMomentumOutOfRange { .. })
    ));
    assert!(matches!(
        xsph_lj_needed_flags(2, final_lj.view(), index_map.view(), 1, 0),
        Err(XsphError::NonPositiveCalculationIndex { .. })
    ));

    let overflow_map = arr1(&[i32::MIN]);
    assert!(matches!(
        xsph_lj_needed_flags(2, final_lj.view(), overflow_map.view(), 1, 1),
        Err(XsphError::IndexMapOverflow { .. })
    ));

    let radii = arr1(&[1.0]);
    assert!(matches!(
        xsph_q_bessel_table(Real::NAN, radii.view(), 4),
        Err(XsphError::NonFiniteScalar { name: "qtrans", .. })
    ));
    let bad_radii = arr1(&[Real::INFINITY]);
    assert!(matches!(
        xsph_q_bessel_table(1.0, bad_radii.view(), 4),
        Err(XsphError::NonFiniteScalar { name: "radius", .. })
    ));
    assert!(matches!(
        xsph_q_bessel_table(1.0, radii.view(), 40),
        Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: 40,
            ljmax: 39,
        })
    ));
    assert!(matches!(
        xsph_q_bessel_table(0.0, radii.view(), 4),
        Err(XsphError::Bessel(
            BesselError::NonPositiveRealArgument { .. }
        ))
    ));

    assert!(matches!(
        xsph_longitudinal_multipole_factor(0, 1, 1),
        Err(XsphError::ZeroKappa)
    ));
    assert!(matches!(
        xsph_longitudinal_multipole_factor(1, 1, -1),
        Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_longitudinal_multipole_factor(i32::MIN, 1, 1),
        Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
    ));
    assert!(matches!(
        xsph_longitudinal_multipole_factor(60, -60, 1),
        Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(0, 1, 0, 1),
        Err(XsphError::ZeroKappa)
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(1, 1, -1, 1),
        Err(XsphError::NegativeAngularMomentum {
            name: "bessel_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(1, 1, 0, -1),
        Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(1, 1, 59, 59),
        Err(XsphError::IntegerOutOfRange {
            name: "bessel_l",
            ..
        })
    ));

    let lgind = arr1(&[0]);
    let ljind = arr1(&[0]);
    assert!(matches!(
        xsph_nrixs_transition_weights(0, 1, 4, 9, 3, lgind.view(), ljind.view(), 1),
        Err(XsphError::ZeroKappa)
    ));
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 4, -1, 3, lgind.view(), ljind.view(), 1),
        Err(XsphError::NegativeAngularMomentum { name: "jmax", .. })
    ));
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, lgind.view(), ljind.view(), 2),
        Err(XsphError::LengthTooShort { name: "lgind", .. })
    ));
    let bad_lgind = arr1(&[-1]);
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, bad_lgind.view(), ljind.view(), 1),
        Err(XsphError::NegativeAngularMomentum { name: "lgind", .. })
    ));
    let two_lgind = arr1(&[0, 0]);
    let two_ljind = arr1(&[0, 0]);
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 0, 1, 0, two_lgind.view(), two_ljind.view(), 2),
        Err(XsphError::InsufficientGeneratedStates { .. })
    ));
}
