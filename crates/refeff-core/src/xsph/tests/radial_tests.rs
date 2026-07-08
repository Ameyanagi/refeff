use super::{support::*, *};

const RADINT_ACTIVE_LEN: usize = 7;
const RADINT_TOTAL_LEN: usize = 16;
const JAS_CORRECTION_ACTIVE_LEN: usize = 7;
const JAS_CORRECTION_TOTAL_LEN: usize = 9;
const JAS_CORRECTION_LJMAX: usize = 4;
const JAS_CORRECTION_Q_COUNT: usize = 2;
const JAS_RADIAL_ACTIVE_LEN: usize = 7;
const JAS_RADIAL_TOTAL_LEN: usize = 9;
const JAS_RADIAL_LJMAX: usize = 4;

fn reciprocal32(value: num_complex::Complex32) -> Complex {
    let denominator = value.re * value.re + value.im * value.im;
    Complex::new(
        (value.re / denominator) as Real,
        (-value.im / denominator) as Real,
    )
}

fn interpolated_pair(first: Complex, second: Complex) -> [Complex; 6] {
    [
        first,
        (first * 4.0 + second) / 5.0,
        (first * 3.0 + second * 2.0) / 5.0,
        (first * 2.0 + second * 3.0) / 5.0,
        (first + second * 4.0) / 5.0,
        second,
    ]
}

fn phiscf_loucks_grid(len: usize, log_step: Real, origin_shift: Real) -> Array1<Real> {
    Array1::from_iter((0..len).map(|index| (-origin_shift + index as Real * log_step).exp()))
}

fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
    assert_close_tol(actual.re, expected.re, tolerance);
    assert_close_tol(actual.im, expected.im, tolerance);
}

struct PhiscfRadialContributionFixture {
    radii: Array1<Real>,
    orbital_large: Array1<Real>,
    orbital_small: Array1<Real>,
    regular_large: Array1<Complex>,
    regular_small: Array1<Complex>,
    irregular_large: Array1<Complex>,
    irregular_small: Array1<Complex>,
    local_field: Array1<Real>,
    basis_fields: Array2<Complex>,
    wave_number: Complex,
}

struct PhiscfWfirdcContributionFixture {
    energy: Complex,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    orbital_lengths: Array1<usize>,
    exchange_correlation_potential: Array1<Complex>,
    c3_potential: Array1<Complex>,
    orbital_large: Array1<Real>,
    orbital_small: Array1<Real>,
    local_field: Array1<Real>,
    step: Real,
    nuclear_charge: Real,
    muffin_tin_radius: Real,
    radial_match_index: usize,
    wkb_index: usize,
    active_len: usize,
}

impl PhiscfWfirdcContributionFixture {
    fn to_input(&self) -> crate::FovrgInitialPhotoelectronInput<'_> {
        crate::FovrgInitialPhotoelectronInput {
            energy: self.energy,
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            orbital_lengths: self.orbital_lengths.view(),
            exchange_correlation_potential: self.exchange_correlation_potential.view(),
            c3_potential: self.c3_potential.view(),
            initial_large_coefficient: Complex::new(0.0, 0.0),
            initial_small_coefficient: Complex::new(0.0, 0.0),
            nuclear_charge: self.nuclear_charge,
            muffin_tin_radius: self.muffin_tin_radius,
            step: self.step,
            speed_of_light: 137.0373,
            c3_scale: 0,
            irregular: false,
            radial_match_index: self.radial_match_index,
            wkb_index: self.wkb_index,
            coefficient_count: 3,
            orbital_count: self.kappa.len(),
            active_len: self.active_len,
        }
    }
}

fn phiscf_radial_contribution_fixture() -> PhiscfRadialContributionFixture {
    let radii = arr1(&[1.0, 1.2, 1.5, 1.9, 2.4, 3.0]);
    let orbital_large = arr1(&[0.8, 0.82, 0.85, 0.9, 0.95, 1.0]);
    let orbital_small = arr1(&[0.1, 0.11, 0.12, 0.13, 0.14, 0.15]);
    let regular_large = arr1(&[
        Complex::new(0.2, 0.05),
        Complex::new(0.23, 0.04),
        Complex::new(0.27, 0.03),
        Complex::new(0.30, 0.02),
        Complex::new(0.34, 0.01),
        Complex::new(0.38, -0.01),
    ]);
    let regular_small = arr1(&[
        Complex::new(0.05, -0.02),
        Complex::new(0.045, -0.015),
        Complex::new(0.04, -0.01),
        Complex::new(0.035, -0.005),
        Complex::new(0.03, 0.0),
        Complex::new(0.025, 0.005),
    ]);
    let irregular_large = arr1(&[
        Complex::new(0.3, -0.01),
        Complex::new(0.28, -0.015),
        Complex::new(0.25, -0.02),
        Complex::new(0.22, -0.025),
        Complex::new(0.19, -0.03),
        Complex::new(0.16, -0.035),
    ]);
    let irregular_small = arr1(&[
        Complex::new(-0.02, 0.04),
        Complex::new(-0.015, 0.035),
        Complex::new(-0.01, 0.03),
        Complex::new(-0.005, 0.025),
        Complex::new(0.0, 0.02),
        Complex::new(0.005, 0.015),
    ]);
    let local_field = arr1(&[0.1, 0.08, 0.06, 0.04, 0.02, 0.01]);
    let mut basis_fields = Array2::<Complex>::zeros((radii.len(), 2));
    for index in 0..radii.len() {
        basis_fields[(index, 0)] = Complex::new(0.2 + index as Real * 0.03, -0.1);
        basis_fields[(index, 1)] = Complex::new(-0.4 + index as Real * 0.02, 0.15);
    }

    PhiscfRadialContributionFixture {
        radii,
        orbital_large,
        orbital_small,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        local_field,
        basis_fields,
        wave_number: Complex::new(0.74, 0.06),
    }
}

fn phiscf_wfirdc_contribution_fixture() -> PhiscfWfirdcContributionFixture {
    let active_len = 15;
    let bound_orbitals = 2;
    let step = 0.045;
    let radial_match_index = 8;
    let muffin_tin_radius = (-8.8_f64 + step * radial_match_index as Real).exp() - 1.0e-20;
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.01 * row + 0.0015 * orbital * (0.25 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.007 * row + 0.001 * orbital * (0.18 * row * orbital).sin()
    });
    let exchange_correlation_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.22 + 0.015 * row, 0.003 * (0.37 * row).cos())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));
    let orbital_large = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        0.8 + 0.03 * (0.2 * row).sin()
    }));
    let orbital_small = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        0.1 + 0.01 * (0.17 * row).cos()
    }));
    let local_field = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        0.06 * (-0.03 * row).exp()
    }));

    PhiscfWfirdcContributionFixture {
        energy: Complex::new(0.42, 0.018),
        bound_large_coefficients,
        bound_small_coefficients,
        electron_counts: Array1::from_vec(vec![1.25, 0.65]),
        kappa: Array1::from_vec(vec![-1, 1, -2]),
        orbital_lengths: Array1::from_vec(vec![0, 0, 12]),
        exchange_correlation_potential,
        c3_potential,
        orbital_large,
        orbital_small,
        local_field,
        step,
        nuclear_charge: 29.0,
        muffin_tin_radius,
        radial_match_index,
        wkb_index: 6,
        active_len,
    }
}

#[test]
fn xsph_xray_bessel_table_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let radii = arr1(&[0.12, 0.85, 2.7, 3.1, 4.8, 6.2]);
    let result = xsph_xray_bessel_table(XsphXrayBesselTableInput {
        photon_wave_number: 0.42,
        radii: radii.view(),
        active_len: radii.len(),
    })?;

    assert_eq!(result.values.shape(), &[3, radii.len()]);
    let expected = [
        [
            9.995_766_937_668_55e-1,
            9.788_934_503_717_904e-1,
            7.990_402_017_386_633e-1,
            7.404_694_743_487_317e-1,
            4.476_800_465_968_245e-1,
            1.966_473_509_764_408_2e-1,
        ],
        [
            1.679_573_291_832_655_8e-2,
            1.174_902_440_106_764_8e-1,
            3.315_709_402_053_599e-1,
            3.647_452_348_753_561_5e-1,
            4.356_754_739_957_095e-1,
            4.053_729_835_764_632_3e-1,
        ],
        [
            1.693_132_763_925_672e-4,
            8.419_524_507_843_872e-3,
            7.813_159_774_641_576e-2,
            9.995_733_411_983_071e-2,
            2.006_465_516_110_765_8e-1,
            2.703_722_153_558_9e-1,
        ],
    ];
    for (row, expected_row) in expected.iter().enumerate() {
        for (index, expected_value) in expected_row.iter().enumerate() {
            assert_close(result.values[(row, index)], *expected_value);
        }
    }
    Ok(())
}

#[test]
fn xsph_xray_bessel_table_rejects_invalid_inputs() {
    let radii = arr1(&[0.12, 0.85]);
    let error = xsph_xray_bessel_table(XsphXrayBesselTableInput {
        photon_wave_number: 0.0,
        radii: radii.view(),
        active_len: radii.len(),
    })
    .expect_err("FEFF xk0 is positive after the omega floor");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "photon_wave_number",
            value: 0.0
        }
    ));

    let invalid_radii = arr1(&[0.12, -0.85]);
    let error = xsph_xray_bessel_table(XsphXrayBesselTableInput {
        photon_wave_number: 0.42,
        radii: invalid_radii.view(),
        active_len: invalid_radii.len(),
    })
    .expect_err("radial grid entries must be positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveRadius {
            name: "radius",
            value: -0.85
        }
    ));

    let error = xsph_xray_bessel_table(XsphXrayBesselTableInput {
        photon_wave_number: 0.42,
        radii: radii.view(),
        active_len: radii.len() + 1,
    })
    .expect_err("the active prefix must be present");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "radii",
            required: 3,
            actual: 2
        }
    ));
}

#[test]
fn xsph_xsect_hole_normalization_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let radii = arr1(&[0.12, 0.18, 0.27, 0.405, 0.6075, 0.91125, 1.366875]);
    let large = arr1(&[
        3.453_995_925_725_288,
        2.734_413_441_199_186,
        2.014_830_956_673_084_6,
        1.295_248_472_146_983,
        0.805_932_382_669_234,
        0.460_532_790_096_705_06,
        0.230_266_395_048_352_53,
    ]);
    let small = arr1(&[
        0.230_266_395_048_352_53,
        0.201_483_095_667_308_5,
        0.158_308_146_595_742_36,
        0.115_133_197_524_176_26,
        0.080_593_238_266_923_38,
        0.051_809_938_885_879_314,
        0.028_783_299_381_044_066,
    ]);

    let result = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l: 1,
        log_step: 0.1,
        radii: radii.view(),
        initial_large: large.view(),
        initial_small: small.view(),
        norman_index_1based: radii.len(),
    })?;

    assert_close(result.near_origin_power, 4.0);
    assert_close(result.normalization, 0.999_999_999_999_999_8);
    assert_close(result.deviation, 2.220_446_049_250_313e-16);
    assert!(!result.warning_required);

    Ok(())
}

#[test]
fn xsph_xsect_hole_normalization_preserves_feff_warning_condition() -> Result<(), XsphError> {
    let radii = arr1(&[0.12, 0.18, 0.27, 0.405, 0.6075, 0.91125, 1.366875]);
    let large = arr1(&[1.2, 0.95, 0.7, 0.45, 0.28, 0.16, 0.08]);
    let small = arr1(&[0.08, 0.07, 0.055, 0.04, 0.028, 0.018, 0.01]);

    let result = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l: 1,
        log_step: 0.1,
        radii: radii.view(),
        initial_large: large.view(),
        initial_small: small.view(),
        norman_index_1based: radii.len(),
    })?;

    assert_close(result.near_origin_power, 4.0);
    assert_close(result.normalization, 0.120_703_218_409_687_45);
    assert_close(result.deviation, 0.879_296_781_590_312_5);
    assert!(result.warning_required);

    Ok(())
}

#[test]
fn xsph_xsect_hole_normalization_rejects_invalid_inputs() {
    let radii = arr1(&[0.12, 0.18]);
    let large = arr1(&[1.0, 0.8]);
    let small = arr1(&[0.1, 0.08]);
    let error = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l: 1,
        log_step: 0.1,
        radii: radii.view(),
        initial_large: large.view(),
        initial_small: small.view(),
        norman_index_1based: 3,
    })
    .expect_err("FEFF jnrm prefix must be present");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "radii",
            required: 3,
            actual: 2
        }
    ));

    let error = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l: 1,
        log_step: 0.0,
        radii: radii.view(),
        initial_large: large.view(),
        initial_small: small.view(),
        norman_index_1based: radii.len(),
    })
    .expect_err("Loucks log step must be positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "log_step",
            value: 0.0
        }
    ));

    let invalid_large = arr1(&[1.0, Real::NAN]);
    let error = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l: 1,
        log_step: 0.1,
        radii: radii.view(),
        initial_large: invalid_large.view(),
        initial_small: small.view(),
        norman_index_1based: radii.len(),
    })
    .expect_err("core-hole spinor components must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteScalar {
            name: "initial_large",
            ..
        }
    ));
}

#[test]
fn xsph_xsect_energy_setup_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let active = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(1.4, 0.25),
        reference_energy: Complex::new(0.2, 0.05),
        edge_energy: 0.3,
        chemical_potential: -0.02,
        muffin_tin_radius: 2.4,
        exchange_selector: 15,
        norman_index_1based: 42,
        new_grid_index_1based: 55,
        radial_capacity: 80,
    })?;
    assert_eq!(active.decision, XsphXsectEnergyDecision::Active);
    assert_complex_close(active.momentum_squared, Complex::new(1.2, 0.2));
    assert_close(active.edge_momentum_squared, 0.1);
    assert_complex_close(
        active.wave_number,
        Complex::new(1.554_550_948_626_939, 0.128_662_737_302_707),
    );
    assert_complex_close(
        active.muffin_tin_argument,
        Complex::new(3.730_922_276_704_653_5, 0.308_790_569_526_496_8),
    );
    assert_eq!(active.cycle_count, 3);
    assert_close(active.photon_energy, 1.08);
    assert_close(active.photon_wave_number, 0.007_881_141_322_565_715);
    assert_eq!(active.active_radial_len, 55);

    let floored = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(0.1, 0.0),
        reference_energy: Complex::new(0.02, 0.01),
        edge_energy: 0.2,
        chemical_potential: -0.5,
        muffin_tin_radius: 1.8,
        exchange_selector: 4,
        norman_index_1based: 12,
        new_grid_index_1based: 10,
        radial_capacity: 40,
    })?;
    assert_eq!(floored.decision, XsphXsectEnergyDecision::Active);
    assert_complex_close(floored.momentum_squared, Complex::new(0.08, -0.01));
    assert_close(floored.edge_momentum_squared, 0.18);
    assert_complex_close(
        floored.wave_number,
        Complex::new(0.400_777_889_803_392_85, -0.024_951_582_548_616_03),
    );
    assert_complex_close(
        floored.muffin_tin_argument,
        Complex::new(0.721_400_201_646_107_2, -0.044_912_848_587_508_86),
    );
    assert_eq!(floored.cycle_count, 0);
    assert_close(floored.photon_energy, 0.000_036_749_309_002_742_823);
    assert_close(floored.photon_wave_number, 0.000_000_268_172_683_108_567_4);
    assert_eq!(floored.active_radial_len, 19);

    Ok(())
}

#[test]
fn xsph_xsect_energy_setup_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let below = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(-11.2, 0.1),
        reference_energy: Complex::new(0.3, 0.0),
        edge_energy: 0.2,
        chemical_potential: 0.1,
        muffin_tin_radius: 2.0,
        exchange_selector: 5,
        norman_index_1based: 4,
        new_grid_index_1based: 20,
        radial_capacity: 25,
    })?;
    assert_eq!(below.decision, XsphXsectEnergyDecision::BelowEnergyWindow);
    assert_complex_close(below.momentum_squared, Complex::new(-11.5, 0.1));
    assert_eq!(below.cycle_count, 3);
    assert_eq!(below.active_radial_len, 20);

    let nonpositive = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(0.1, -0.2),
        reference_energy: Complex::new(0.5, 0.0),
        edge_energy: 0.2,
        chemical_potential: 0.1,
        muffin_tin_radius: 2.0,
        exchange_selector: 0,
        norman_index_1based: 4,
        new_grid_index_1based: 20,
        radial_capacity: 25,
    })?;
    assert_eq!(
        nonpositive.decision,
        XsphXsectEnergyDecision::NonPositiveMomentum
    );
    assert_complex_close(nonpositive.momentum_squared, Complex::new(-0.4, -0.2));
    assert_close(nonpositive.photon_energy, 0.000_036_749_309_002_742_823);

    let capped = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(1.0, 0.1),
        reference_energy: Complex::new(0.2, 0.0),
        edge_energy: 0.2,
        chemical_potential: 0.0,
        muffin_tin_radius: 2.0,
        exchange_selector: -6,
        norman_index_1based: 24,
        new_grid_index_1based: 31,
        radial_capacity: 27,
    })?;
    assert_eq!(capped.cycle_count, 0);
    assert_eq!(capped.active_radial_len, 27);

    Ok(())
}

#[test]
fn xsph_xsect_energy_setup_rejects_invalid_inputs() {
    let error = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(Real::NAN, 0.0),
        reference_energy: Complex::new(0.2, 0.0),
        edge_energy: 0.2,
        chemical_potential: 0.0,
        muffin_tin_radius: 2.0,
        exchange_selector: 0,
        norman_index_1based: 4,
        new_grid_index_1based: 20,
        radial_capacity: 25,
    })
    .expect_err("energy must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_energy",
            ..
        }
    ));

    let error = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(1.0, 0.0),
        reference_energy: Complex::new(0.2, 0.0),
        edge_energy: 0.2,
        chemical_potential: 0.0,
        muffin_tin_radius: 0.0,
        exchange_selector: 0,
        norman_index_1based: 4,
        new_grid_index_1based: 20,
        radial_capacity: 25,
    })
    .expect_err("muffin-tin radius must be positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: 0.0
        }
    ));

    let error = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
        energy: Complex::new(1.0, 0.0),
        reference_energy: Complex::new(0.2, 0.0),
        edge_energy: 0.2,
        chemical_potential: 0.0,
        muffin_tin_radius: 2.0,
        exchange_selector: 0,
        norman_index_1based: 0,
        new_grid_index_1based: 20,
        radial_capacity: 25,
    })
    .expect_err("jnrm is one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "norman_index",
            index_1based: 0,
            active_len: 25
        }
    ));
}

#[test]
fn xsph_xsect_transition_plan_matches_feff_loop_order() -> Result<(), XsphError> {
    let final_kappas = arr1(&[101, 102, 103, 104, 105, 106, 0, 108]);
    let orbital_l = arr1(&[0, 1, 2, 3, 4, 5, 6, 7]);

    let result = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 1.0,
        selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
        transition_direction: 0,
        initial_kappa: -1,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: final_kappas.len(),
    })?;

    assert_eq!(
        result.transitions,
        vec![
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricDipole,
                transition_delta: -1,
                transition_index_1based: 1,
                final_kappa: 101,
                final_l: 0,
                multipole_order: 1,
            },
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricDipole,
                transition_delta: 0,
                transition_index_1based: 2,
                final_kappa: 102,
                final_l: 1,
                multipole_order: 1,
            },
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricDipole,
                transition_delta: 1,
                transition_index_1based: 3,
                final_kappa: 103,
                final_l: 2,
                multipole_order: 1,
            },
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricQuadrupole,
                transition_delta: -2,
                transition_index_1based: 4,
                final_kappa: 104,
                final_l: 3,
                multipole_order: 2,
            },
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricQuadrupole,
                transition_delta: -1,
                transition_index_1based: 5,
                final_kappa: 105,
                final_l: 4,
                multipole_order: 2,
            },
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricQuadrupole,
                transition_delta: 0,
                transition_index_1based: 6,
                final_kappa: 106,
                final_l: 5,
                multipole_order: 2,
            },
            XsphXsectTransition {
                multipole: XsphTransitionMultipole::ElectricQuadrupole,
                transition_delta: 2,
                transition_index_1based: 8,
                final_kappa: 108,
                final_l: 7,
                multipole_order: 2,
            },
        ]
    );

    let magnetic = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 1.0,
        selected_higher_multipole: Some(XsphTransitionMultipole::MagneticDipole),
        transition_direction: 0,
        initial_kappa: -1,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: final_kappas.len(),
    })?;
    assert_eq!(
        magnetic
            .transitions
            .iter()
            .map(|transition| transition.transition_index_1based)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 5, 6]
    );
    assert_eq!(
        magnetic
            .transitions
            .iter()
            .filter(|transition| transition.multipole == XsphTransitionMultipole::MagneticDipole)
            .count(),
        2
    );

    Ok(())
}

#[test]
fn xsph_xsect_transition_plan_preserves_feff_l2lp_filters() -> Result<(), XsphError> {
    let final_kappas = arr1(&[11, 12, 13, 14, 15, 16, 17, 18]);
    let orbital_l = arr1(&[0, 1, 2, 3, 4, 5, 6, 7]);

    for (initial_kappa, transition_direction, expected_indices) in [
        (-1, 1, vec![1, 2]),
        (-1, -1, vec![3]),
        (1, 1, vec![3]),
        (1, -1, vec![1, 2]),
    ] {
        let result = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
            photon_energy: 1.0,
            selected_higher_multipole: None,
            transition_direction,
            initial_kappa,
            final_kappas: final_kappas.view(),
            orbital_l: orbital_l.view(),
            active_len: final_kappas.len(),
        })?;
        assert_eq!(
            result
                .transitions
                .iter()
                .map(|transition| transition.transition_index_1based)
                .collect::<Vec<_>>(),
            expected_indices
        );
    }

    let skipped = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 0.0,
        selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
        transition_direction: 0,
        initial_kappa: -1,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: final_kappas.len(),
    })?;
    assert!(skipped.transitions.is_empty());

    Ok(())
}

#[test]
fn xsph_xsect_transition_plan_rejects_invalid_inputs() {
    let final_kappas = arr1(&[11, 12, 13]);
    let orbital_l = arr1(&[0, 1, 2]);

    let error = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: Real::NAN,
        selected_higher_multipole: None,
        transition_direction: 0,
        initial_kappa: -1,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: final_kappas.len(),
    })
    .expect_err("photon energy must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteScalar {
            name: "photon_energy",
            ..
        }
    ));

    let error = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 1.0,
        selected_higher_multipole: None,
        transition_direction: 2,
        initial_kappa: -1,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: final_kappas.len(),
    })
    .expect_err("FEFF l2lp filters only support -1, 0, and 1");
    assert!(matches!(
        error,
        XsphError::IntegerOutOfRange {
            name: "l2lp",
            value: 2
        }
    ));

    let error = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 1.0,
        selected_higher_multipole: None,
        transition_direction: 0,
        initial_kappa: 0,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: final_kappas.len(),
    })
    .expect_err("initial kappa comes from FEFF setkap and must be nonzero");
    assert_eq!(error, XsphError::ZeroKappa);

    let error = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 1.0,
        selected_higher_multipole: None,
        transition_direction: 0,
        initial_kappa: -1,
        final_kappas: final_kappas.view(),
        orbital_l: orbital_l.view(),
        active_len: 9,
    })
    .expect_err("FEFF xsect transition tables contain eight slots");
    assert!(matches!(
        error,
        XsphError::SizeOutOfRange {
            name: "xsect_transition_count",
            value: 9
        }
    ));
}

#[test]
fn xsph_xsect_screened_field_setup_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let screened = xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        standard_potential: true,
        orbital_correction_pending: true,
        momentum_squared: Complex::new(1.4, 0.2),
        edge_energy: 0.5,
        chemical_potential: 0.3,
        screened_orbital_energy: 0.8,
    })?;
    assert_eq!(screened.mode, XsphXsectScreenedFieldMode::ScreenedDipole);
    assert_close(screened.work_energy, 1.2);
    assert_close(screened.screened_transition_energy, 0.6);
    assert_close(screened.field_scale, std::f64::consts::FRAC_1_SQRT_2);
    assert!(!screened.unity_fscf);
    assert!(screened.orbital_correction_required);
    assert!(!screened.orbital_correction_pending_after);
    assert_eq!(
        screened.phiscf_workspace,
        Some(XsphXsectPhiscfWorkspace {
            max_size: 1,
            matrix_size: 0,
            scale_function: 1.0,
        })
    );

    let already_corrected = xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        standard_potential: true,
        orbital_correction_pending: false,
        momentum_squared: Complex::new(1.4, 0.2),
        edge_energy: 0.5,
        chemical_potential: 0.3,
        screened_orbital_energy: 0.8,
    })?;
    assert!(!already_corrected.orbital_correction_required);
    assert!(!already_corrected.orbital_correction_pending_after);

    Ok(())
}

#[test]
fn xsph_xsect_screened_field_setup_preserves_feff_unity_branches() -> Result<(), XsphError> {
    let poison_orbital_energy = Real::NAN;
    for (multipole, standard_potential) in [
        (XsphTransitionMultipole::MagneticDipole, true),
        (XsphTransitionMultipole::ElectricQuadrupole, true),
        (XsphTransitionMultipole::ElectricDipole, false),
    ] {
        let unity = xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
            multipole,
            standard_potential,
            orbital_correction_pending: true,
            momentum_squared: Complex::new(1.4, 0.2),
            edge_energy: 0.5,
            chemical_potential: 0.3,
            screened_orbital_energy: poison_orbital_energy,
        })?;
        assert_eq!(unity.mode, XsphXsectScreenedFieldMode::UnityField);
        assert_close(unity.work_energy, 1.2);
        assert_close(unity.screened_transition_energy, 1.2);
        assert_close(unity.field_scale, 1.0);
        assert!(unity.unity_fscf);
        assert!(!unity.orbital_correction_required);
        assert!(unity.orbital_correction_pending_after);
        assert_eq!(unity.phiscf_workspace, None);
    }

    Ok(())
}

#[test]
fn xsph_xsect_screened_field_setup_rejects_invalid_inputs() {
    let error = xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        standard_potential: true,
        orbital_correction_pending: true,
        momentum_squared: Complex::new(1.4, 0.2),
        edge_energy: 0.5,
        chemical_potential: 0.3,
        screened_orbital_energy: Real::NAN,
    })
    .expect_err("screened dipoles require a finite corrected orbital energy");
    assert!(matches!(
        error,
        XsphError::NonFiniteScalar {
            name: "screened_orbital_energy",
            ..
        }
    ));

    let error = xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        standard_potential: false,
        orbital_correction_pending: true,
        momentum_squared: Complex::new(Real::NAN, 0.2),
        edge_energy: 0.5,
        chemical_potential: 0.3,
        screened_orbital_energy: 0.8,
    })
    .expect_err("p2 must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_momentum_squared",
            ..
        }
    ));

    let error = xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        standard_potential: false,
        orbital_correction_pending: true,
        momentum_squared: Complex::new(0.2, 0.0),
        edge_energy: 0.5,
        chemical_potential: 0.3,
        screened_orbital_energy: 0.8,
    })
    .expect_err("FEFF field scale must stay finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteScalar {
            name: "xsect_screened_field_scale",
            ..
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_local_field_matches_feff_reference() -> Result<(), XsphError> {
    let radii = arr1(&[0.5, 1.2, 2.0, 3.1, 9.9]);
    let density = arr1(&[0.0, 0.05, 0.2, -1.0, 0.4]);

    let zangwill_soven = xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
        exchange_correlation_selector: 1,
        radii: radii.view(),
        electron_density: density.view(),
        active_len: 4,
    })?;
    let expected = [
        -12_690.293_237_582_286,
        -0.433_105_413_693_821_33,
        -0.060_325_097_856_270_62,
        -330.132_498_376_230_1,
    ];
    assert_eq!(zangwill_soven.values.len(), expected.len());
    for (index, expected_value) in expected.iter().copied().enumerate() {
        assert_close(zangwill_soven.values[index], expected_value);
    }

    let rpa = xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
        exchange_correlation_selector: 0,
        radii: radii.view(),
        electron_density: density.view(),
        active_len: 4,
    })?;
    assert_eq!(rpa.values, arr1(&[0.0, 0.0, 0.0, 0.0]));

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_local_field_rejects_invalid_inputs() {
    let radii = arr1(&[0.5, 1.2]);
    let density = arr1(&[0.0]);
    let error = xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
        exchange_correlation_selector: 1,
        radii: radii.view(),
        electron_density: density.view(),
        active_len: 2,
    })
    .expect_err("density must cover the active radial prefix");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "electron_density",
            required: 2,
            actual: 1
        }
    ));

    let invalid_radius = arr1(&[0.5, 0.0]);
    let density = arr1(&[0.0, 0.1]);
    let error = xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
        exchange_correlation_selector: 1,
        radii: invalid_radius.view(),
        electron_density: density.view(),
        active_len: 2,
    })
    .expect_err("phiscf radial grid must stay positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "xsect_phiscf_radius",
            value: 0.0
        }
    ));

    let radii = arr1(&[0.5, 1.2]);
    let invalid_density = arr1(&[0.0, Real::NAN]);
    let error = xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
        exchange_correlation_selector: 0,
        radii: radii.view(),
        electron_density: invalid_density.view(),
        active_len: 2,
    })
    .expect_err("density validation runs before the RPA zero branch");
    assert!(matches!(
        error,
        XsphError::NonFiniteScalar {
            name: "electron_density",
            ..
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_linear_solve_matches_feff_chiklu_reference() -> Result<(), XsphError> {
    let radii = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let response = arr2(&[
        [Complex::new(0.2, 0.1), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(-0.1, 0.2)],
    ]);
    let mut basis_fields = Array2::<Complex>::zeros((6, 1));
    basis_fields[(0, 0)] = Complex::new(2.0, 9.0);
    basis_fields[(5, 0)] = Complex::new(-1.0, 3.0);

    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 2,
        radii: radii.view(),
        response: response.view(),
        basis_fields: basis_fields.view(),
        basis_count: 1,
    })?;

    let first = reciprocal32(num_complex::Complex32::new(0.8, -0.1));
    let second = reciprocal32(num_complex::Complex32::new(1.1, -0.2));
    let expected_field = interpolated_pair(first, second);
    assert_eq!(solved.screened_field.len(), expected_field.len());
    for (index, expected) in expected_field.iter().copied().enumerate() {
        assert_complex_close_tol(solved.screened_field[index], expected, 2.0e-6);
    }

    let expected_basis = interpolated_pair(first * 2.0, second * (-1.0 / 6.0));
    assert_eq!(
        solved.screened_basis_fields.dim(),
        (expected_basis.len(), 1)
    );
    for (index, expected) in expected_basis.iter().copied().enumerate() {
        assert_complex_close_tol(solved.screened_basis_fields[(index, 0)], expected, 2.0e-6);
    }

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_linear_solve_scales_large_response_system() -> Result<(), XsphError> {
    let radii = arr1(&[1.0]);
    let response = arr2(&[[Complex::new(1.0e200, -2.0e199)]]);
    let basis_fields = Array2::<Complex>::zeros((1, 0));

    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 1,
        radii: radii.view(),
        response: response.view(),
        basis_fields: basis_fields.view(),
        basis_count: 0,
    })?;

    let expected = Complex::new(-9.615_384_615_384_615e-201, -1.923_076_923_076_923e-201);
    assert_complex_close_tol(solved.screened_field[0], expected, 1.0e-210);
    Ok(())
}

#[test]
fn xsph_xsect_phiscf_linear_solve_rejects_invalid_inputs() {
    let short_radii = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let response = Array2::<Complex>::zeros((2, 2));
    let basis_fields = Array2::<Complex>::zeros((0, 0));
    let error = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 2,
        radii: short_radii.view(),
        response: response.view(),
        basis_fields: basis_fields.view(),
        basis_count: 0,
    })
    .expect_err("six fine-grid points are needed for two chiklu coarse rows");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "radii",
            required: 6,
            actual: 5
        }
    ));

    let radii = arr1(&[1.0]);
    let singular_response = arr2(&[[Complex::new(1.0, 0.0)]]);
    let error = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 1,
        radii: radii.view(),
        response: singular_response.view(),
        basis_fields: basis_fields.view(),
        basis_count: 0,
    })
    .expect_err("1 - chik must be nonsingular");
    assert!(matches!(
        error,
        XsphError::Linalg(refeff_linalg::LinalgError::SingularMatrix { pivot: 0 })
    ));
}

#[test]
fn xsph_xsect_phiscf_lipman_response_matches_feff_reference() -> Result<(), XsphError> {
    let radii = arr1(&[1.0, 1.2, 1.5, 1.9, 2.4, 3.0]);
    let orbital_large = arr1(&[0.8, 0.82, 0.85, 0.9, 0.95, 1.0]);
    let orbital_small = arr1(&[0.1, 0.11, 0.12, 0.13, 0.14, 0.15]);
    let regular_large = arr1(&[
        Complex::new(0.2, 0.05),
        Complex::new(0.23, 0.04),
        Complex::new(0.27, 0.03),
        Complex::new(0.30, 0.02),
        Complex::new(0.34, 0.01),
        Complex::new(0.38, -0.01),
    ]);
    let regular_small = arr1(&[
        Complex::new(0.05, -0.02),
        Complex::new(0.045, -0.015),
        Complex::new(0.04, -0.01),
        Complex::new(0.035, -0.005),
        Complex::new(0.03, 0.0),
        Complex::new(0.025, 0.005),
    ]);
    let irregular_large = arr1(&[
        Complex::new(0.3, -0.01),
        Complex::new(0.28, -0.015),
        Complex::new(0.25, -0.02),
        Complex::new(0.22, -0.025),
        Complex::new(0.19, -0.03),
        Complex::new(0.16, -0.035),
    ]);
    let irregular_small = arr1(&[
        Complex::new(-0.02, 0.04),
        Complex::new(-0.015, 0.035),
        Complex::new(-0.01, 0.03),
        Complex::new(-0.005, 0.025),
        Complex::new(0.0, 0.02),
        Complex::new(0.005, 0.015),
    ]);
    let local_field = arr1(&[0.1, 0.08, 0.06, 0.04, 0.02, 0.01]);

    let response = xsph_xsect_phiscf_lipman_response(XsphXsectPhiscfLipmanInput {
        coarse_count: 2,
        active_len: radii.len(),
        match_index_1based: 4,
        radii: radii.view(),
        orbital_large: orbital_large.view(),
        orbital_small: orbital_small.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        local_field: local_field.view(),
    })?;
    let expected = arr2(&[
        [
            Complex::new(0.001_090_854_828_918_513_7, 0.000_204_158_405_672_899_4),
            Complex::new(0.004_467_814_165_552_142, -0.000_419_285_474_293_773_24),
        ],
        [
            Complex::new(0.000_272_467_805_555_555_56, 0.000_043_220_008_333_333_335),
            Complex::new(0.003_357_311_372_916_667_6, -0.000_560_884_391_666_666_8),
        ],
    ]);
    assert_eq!(response.response.dim(), expected.dim());
    for row in 0..expected.nrows() {
        for column in 0..expected.ncols() {
            assert_complex_close(response.response[(row, column)], expected[(row, column)]);
        }
    }

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_lipman_response_rejects_invalid_inputs() {
    let radii = arr1(&[1.0, 1.2]);
    let real = arr1(&[1.0, 1.0]);
    let complex = arr1(&[Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)]);
    let error = xsph_xsect_phiscf_lipman_response(XsphXsectPhiscfLipmanInput {
        coarse_count: 1,
        active_len: radii.len(),
        match_index_1based: 3,
        radii: radii.view(),
        orbital_large: real.view(),
        orbital_small: real.view(),
        regular_large: complex.view(),
        regular_small: complex.view(),
        irregular_large: complex.view(),
        irregular_small: complex.view(),
        local_field: real.view(),
    })
    .expect_err("match index is one-based and must be inside the active prefix");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_match_index",
            index_1based: 3,
            active_len: 2
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_accumulated_response_matches_feff_cchik_branch() -> Result<(), XsphError> {
    let first = arr2(&[
        [Complex::new(1.25, 0.5), Complex::new(-2.0, 3.0)],
        [Complex::new(4.0, -1.0), Complex::new(0.125, 0.25)],
    ]);
    let second = arr2(&[
        [Complex::new(-0.5, 2.0), Complex::new(1.0, -4.0)],
        [Complex::new(0.75, 0.5), Complex::new(-1.5, -1.0)],
    ]);
    let contributions = [
        XsphXsectPhiscfResponseContributionInput {
            response: first.view(),
            scale: 2.0,
            include_imaginary: true,
        },
        XsphXsectPhiscfResponseContributionInput {
            response: second.view(),
            scale: -0.5,
            include_imaginary: false,
        },
    ];

    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: 2,
            contributions: &contributions,
        })?;
    let expected = arr2(&[
        [Complex::new(2.75, 1.0), Complex::new(-4.5, 6.0)],
        [Complex::new(7.625, -2.0), Complex::new(1.0, 0.5)],
    ]);
    assert_eq!(accumulated.response.dim(), expected.dim());
    for row in 0..expected.nrows() {
        for column in 0..expected.ncols() {
            assert_complex_close(accumulated.response[(row, column)], expected[(row, column)]);
        }
    }

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_accumulated_response_preserves_large_double_precision_rows()
-> Result<(), XsphError> {
    let source = Complex::new(1.0e40, -2.0e40);
    let scale = 1.0e-4;
    let response = arr2(&[[source]]);
    let contributions = [XsphXsectPhiscfResponseContributionInput {
        response: response.view(),
        scale,
        include_imaginary: true,
    }];

    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: 1,
            contributions: &contributions,
        })?;

    let expected = Complex::new(source.re * scale, source.im * scale);
    assert_complex_close(accumulated.response[(0, 0)], expected);
    Ok(())
}

#[test]
fn xsph_xsect_phiscf_accumulated_response_rejects_invalid_inputs() {
    let response = arr2(&[[Complex::new(1.0, 0.0)]]);
    let contribution = [XsphXsectPhiscfResponseContributionInput {
        response: response.view(),
        scale: Real::NAN,
        include_imaginary: true,
    }];
    let error = xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
        coarse_count: 1,
        contributions: &contribution,
    })
    .expect_err("response scale must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteScalar {
            name: "xsect_phiscf_response_scale",
            ..
        }
    ));

    let short = Array2::<Complex>::zeros((1, 0));
    let contribution = [XsphXsectPhiscfResponseContributionInput {
        response: short.view(),
        scale: 1.0,
        include_imaginary: true,
    }];
    let error = xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
        coarse_count: 1,
        contributions: &contribution,
    })
    .expect_err("response contribution must cover the coarse grid");
    assert!(matches!(
        error,
        XsphError::MatrixTooSmall {
            name: "xsect_phiscf_response_contribution",
            required: [1, 1],
            actual: [1, 0]
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_contribution_rule_matches_feff_aa_branch() -> Result<(), XsphError> {
    let input = XsphXsectPhiscfContributionRuleInput {
        initial_kappa: 1,
        final_kappa: -2,
        shell_occupation_fraction: 0.7,
        photon_energy_correction: 1.2,
        scale_function: 0.8,
        pole_index_1based: 1,
        pole_energy: Complex::new(0.6, 0.03),
        edge_energy: 0.5,
    };
    let rule = xsph_xsect_phiscf_contribution_rule(input)?;

    let jfin2 = 2 * input.final_kappa.abs() - 1;
    let jin2 = 2 * input.initial_kappa.abs() - 1;
    let expected_angular =
        -wigner_3j(jfin2, 2, jin2, 1, 0, 2)?.powi(2) * ((jfin2 + 1) * (jin2 + 1)) as Real / 3.0;
    assert_close(rule.angular_scale, expected_angular);
    assert_close(
        rule.scale,
        expected_angular
            * input.shell_occupation_fraction
            * input.photon_energy_correction
            * input.scale_function,
    );
    assert!(rule.include_imaginary);

    let backward_pole =
        xsph_xsect_phiscf_contribution_rule(XsphXsectPhiscfContributionRuleInput {
            pole_index_1based: 2,
            ..input
        })?;
    assert_close(backward_pole.scale, rule.scale);
    assert!(!backward_pole.include_imaginary);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_contribution_rule_rejects_invalid_inputs() {
    let error = xsph_xsect_phiscf_contribution_rule(XsphXsectPhiscfContributionRuleInput {
        initial_kappa: 1,
        final_kappa: -2,
        shell_occupation_fraction: 1.0,
        photon_energy_correction: 1.0,
        scale_function: 1.0,
        pole_index_1based: 3,
        pole_energy: Complex::new(0.6, 0.03),
        edge_energy: 0.5,
    })
    .expect_err("FEFF phiscf has two one-based poles");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_pole_index",
            index_1based: 3,
            active_len: 2
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_pole_energy_matches_feff_forward_pole() -> Result<(), XsphError> {
    let pole = xsph_xsect_phiscf_pole_energy(XsphXsectPhiscfPoleEnergyInput {
        momentum_squared: Complex::new(1.0, 0.04),
        edge_energy: 0.2,
        chemical_potential: 0.3,
        hole_orbital_energy: 0.25,
        occupied_orbital_energy: -0.1,
        pole_index_1based: 1,
    })?;

    assert_close(pole.photon_energy, 1.1);
    assert_complex_close(pole.response_energy, Complex::new(0.75, 0.04));
    assert_close(pole.photon_energy_correction, 0.75 / 1.1);
    assert_complex_close(pole.raw_pole_energy, Complex::new(0.65, 0.04));
    assert_complex_close(pole.pole_energy, Complex::new(0.65, 0.04));
    assert!(!pole.below_edge_broadening_applied);
    assert_close(pole.broadening, 0.04);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_pole_energy_matches_feff_below_edge_broadening() -> Result<(), XsphError> {
    let pole = xsph_xsect_phiscf_pole_energy(XsphXsectPhiscfPoleEnergyInput {
        momentum_squared: Complex::new(1.0, 0.04),
        edge_energy: 0.8,
        chemical_potential: 0.3,
        hole_orbital_energy: 0.25,
        occupied_orbital_energy: -0.1,
        pole_index_1based: 2,
    })?;

    assert_close(pole.photon_energy, 0.5);
    assert_complex_close(pole.response_energy, Complex::new(0.75, 0.04));
    assert_close(pole.photon_energy_correction, 1.5);
    assert_complex_close(pole.raw_pole_energy, Complex::new(-0.85, 0.04));
    assert_complex_close(pole.pole_energy, Complex::new(-0.85, 0.09));
    assert!(pole.below_edge_broadening_applied);
    assert_close(pole.broadening, 0.09);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_pole_energy_rejects_invalid_inputs() {
    let error = xsph_xsect_phiscf_pole_energy(XsphXsectPhiscfPoleEnergyInput {
        momentum_squared: Complex::new(1.0, 0.04),
        edge_energy: 0.2,
        chemical_potential: 0.3,
        hole_orbital_energy: 0.25,
        occupied_orbital_energy: -0.1,
        pole_index_1based: 0,
    })
    .expect_err("FEFF phiscf pole indices are one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_pole_index",
            index_1based: 0,
            active_len: 2
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_contribution_plan_matches_feff_loop_order() -> Result<(), XsphError> {
    let orbital_kappas = arr1(&[1, -2]);
    let orbital_energy_counts = arr1(&[2_usize, 1]);
    let occupied_energies = arr2(&[[-0.1, 0.05], [0.2, 99.0]]);
    let occupation_fractions = arr2(&[[0.7, 0.4], [0.6, 99.0]]);

    let plan = xsph_xsect_phiscf_contribution_plan(XsphXsectPhiscfContributionPlanInput {
        momentum_squared: Complex::new(1.0, 0.04),
        edge_energy: 0.2,
        chemical_potential: 0.3,
        hole_orbital_energy: 0.25,
        scale_function: 0.8,
        orbital_kappas: orbital_kappas.view(),
        orbital_energy_counts: orbital_energy_counts.view(),
        occupied_energies: occupied_energies.view(),
        occupation_fractions: occupation_fractions.view(),
        active_orbital_count: orbital_kappas.len(),
    })?;

    assert_eq!(plan.rows.len(), 14);
    assert_eq!(plan.rows[0].orbital_index_1based, 1);
    assert_eq!(plan.rows[0].energy_index_1based, 1);
    assert_eq!(plan.rows[0].pole_index_1based, 1);
    assert_eq!(plan.rows[0].dipole_delta, 0);
    assert_eq!(plan.rows[0].initial_kappa, 1);
    assert_eq!(plan.rows[0].final_kappa, -1);
    assert_close(plan.rows[0].occupied_orbital_energy, -0.1);
    assert_close(plan.rows[0].shell_occupation_fraction, 0.7);
    assert_complex_close(plan.rows[0].pole.pole_energy, Complex::new(0.65, 0.04));
    assert!(plan.rows[0].rule.include_imaginary);
    assert_close(plan.rows[0].pole.photon_energy_correction, 0.75 / 1.1);

    assert_eq!(plan.rows[1].dipole_delta, 1);
    assert_eq!(plan.rows[1].final_kappa, 2);
    assert_eq!(plan.rows[2].pole_index_1based, 2);
    assert_eq!(plan.rows[2].dipole_delta, 0);
    assert_eq!(plan.rows[2].final_kappa, -1);
    assert_complex_close(plan.rows[2].pole.pole_energy, Complex::new(-0.85, 0.04));
    assert!(!plan.rows[2].rule.include_imaginary);

    assert_eq!(plan.rows[8].orbital_index_1based, 2);
    assert_eq!(plan.rows[8].initial_kappa, -2);
    assert_eq!(
        plan.rows[8..11]
            .iter()
            .map(|row| (row.dipole_delta, row.final_kappa))
            .collect::<Vec<_>>(),
        vec![(-1, -3), (0, 2), (1, -1)]
    );
    assert_eq!(plan.rows[13].pole_index_1based, 2);
    assert_eq!(plan.rows[13].dipole_delta, 1);
    assert_eq!(plan.rows[13].final_kappa, -1);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_contribution_plan_rejects_invalid_inputs() {
    let orbital_kappas = arr1(&[0]);
    let orbital_energy_counts = arr1(&[1_usize]);
    let occupied_energies = arr2(&[[0.1]]);
    let occupation_fractions = arr2(&[[0.7]]);
    let error = xsph_xsect_phiscf_contribution_plan(XsphXsectPhiscfContributionPlanInput {
        momentum_squared: Complex::new(1.0, 0.04),
        edge_energy: 0.2,
        chemical_potential: 0.3,
        hole_orbital_energy: 0.25,
        scale_function: 0.8,
        orbital_kappas: orbital_kappas.view(),
        orbital_energy_counts: orbital_energy_counts.view(),
        occupied_energies: occupied_energies.view(),
        occupation_fractions: occupation_fractions.view(),
        active_orbital_count: 1,
    })
    .expect_err("occupied orbital kappa values must be nonzero");
    assert_eq!(error, XsphError::ZeroKappa);

    let orbital_kappas = arr1(&[1]);
    let orbital_energy_counts = arr1(&[2_usize]);
    let short_energies = arr2(&[[0.1]]);
    let occupation_fractions = arr2(&[[0.7], [0.6]]);
    let error = xsph_xsect_phiscf_contribution_plan(XsphXsectPhiscfContributionPlanInput {
        momentum_squared: Complex::new(1.0, 0.04),
        edge_energy: 0.2,
        chemical_potential: 0.3,
        hole_orbital_energy: 0.25,
        scale_function: 0.8,
        orbital_kappas: orbital_kappas.view(),
        orbital_energy_counts: orbital_energy_counts.view(),
        occupied_energies: short_energies.view(),
        occupation_fractions: occupation_fractions.view(),
        active_orbital_count: 1,
    })
    .expect_err("occupied energy table must cover each active FEFF neg row");
    assert!(matches!(
        error,
        XsphError::MatrixTooSmall {
            name: "xsect_phiscf_occupied_energies",
            required: [2, 1],
            actual: [1, 1]
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_radial_solver_setup_matches_feff_reference() -> Result<(), XsphError> {
    let radii = phiscf_loucks_grid(251, 0.05, 8.8);
    let setup = xsph_xsect_phiscf_radial_solver_setup(XsphXsectPhiscfRadialSolverSetupInput {
        pole_energy: Complex::new(0.65, 0.04),
        muffin_tin_radius: 2.0,
        radii: radii.view(),
        log_step: 0.05,
        origin_shift: 8.8,
        active_len: radii.len(),
        target_last_index_1based: 251,
    })?;

    assert_complex_close_tol(
        setup.wave_number,
        Complex::new(1.140_724_367_928_864_1, 0.035_066_652_085_321_28),
        1.0e-12,
    );
    assert_close(setup.matching_radius_limit, 2.0);
    assert_eq!(setup.match_index_1based, 191);
    assert_eq!(setup.match_index, 190);
    assert_close(setup.match_radius, radii[190] - 1.0e-20);
    assert_eq!(setup.wkb_index_1based, 221);
    assert_eq!(setup.wkb_index, 220);
    assert_complex_close_tol(
        setup.match_argument_inside,
        Complex::new(2.297_136_784_394_296_7, 0.070_615_565_578_740_92),
        1.0e-12,
    );
    assert_complex_close_tol(
        setup.match_argument_grid,
        Complex::new(2.297_136_784_394_296_7, 0.070_615_565_578_740_92),
        1.0e-12,
    );

    let clamped = xsph_xsect_phiscf_radial_solver_setup(XsphXsectPhiscfRadialSolverSetupInput {
        target_last_index_1based: 220,
        ..XsphXsectPhiscfRadialSolverSetupInput {
            pole_energy: Complex::new(0.65, 0.04),
            muffin_tin_radius: 2.0,
            radii: radii.view(),
            log_step: 0.05,
            origin_shift: 8.8,
            active_len: radii.len(),
            target_last_index_1based: 251,
        }
    })?;
    assert_eq!(clamped.wkb_index_1based, 251);
    assert_eq!(clamped.wkb_index, 250);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_radial_solver_setup_rejects_invalid_inputs() {
    let radii = phiscf_loucks_grid(12, 0.05, 8.8);
    let error = xsph_xsect_phiscf_radial_solver_setup(XsphXsectPhiscfRadialSolverSetupInput {
        pole_energy: Complex::new(0.65, 0.04),
        muffin_tin_radius: 2.0,
        radii: radii.view(),
        log_step: 0.05,
        origin_shift: 8.8,
        active_len: radii.len(),
        target_last_index_1based: 0,
    })
    .expect_err("FEFF jlast is one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "xsect_phiscf_target_last_index",
            index_1based: 0,
            active_len: 12
        }
    ));

    let error = xsph_xsect_phiscf_radial_solver_setup(XsphXsectPhiscfRadialSolverSetupInput {
        pole_energy: Complex::new(0.65, 0.04),
        muffin_tin_radius: 2.0,
        radii: radii.view(),
        log_step: 0.0,
        origin_shift: 8.8,
        active_len: radii.len(),
        target_last_index_1based: radii.len(),
    })
    .expect_err("FEFF log step must be positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "xsect_phiscf_log_step",
            value: 0.0
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_irregular_seed_matches_feff_reference() -> Result<(), XsphError> {
    let positive_channels = xsph_xsect_phiscf_angular_channels(2)?;
    assert_eq!(positive_channels.large_l, 2);
    assert_eq!(positive_channels.small_l, 1);
    let negative_channels = xsph_xsect_phiscf_angular_channels(-2)?;
    assert_eq!(negative_channels.large_l, 1);
    assert_eq!(negative_channels.small_l, 2);

    let wave_number = Complex::new(0.74, 0.06);
    let match_radius = 1.85;
    let seed = xsph_xsect_phiscf_irregular_seed(XsphXsectPhiscfIrregularSeedInput {
        final_kappa: -2,
        wave_number,
        match_radius,
    })?;

    let alpha_scaled = wave_number * XSPH_FINE_STRUCTURE_ALPHA;
    let small_component_factor = -alpha_scaled
        / (Complex::new(1.0, 0.0) + (Complex::new(1.0, 0.0) + alpha_scaled * alpha_scaled).sqrt());
    let relativistic_scale = Complex::new(1.0, 0.0)
        / (Complex::new(1.0, 0.0) + small_component_factor * small_component_factor).sqrt();
    let hankel = crate::besjh(wave_number * match_radius, 2)?;
    let expected_large = hankel.h[1] * match_radius * relativistic_scale;
    let expected_small = hankel.h[2] * match_radius * relativistic_scale * small_component_factor;

    assert_eq!(seed.channels, negative_channels);
    assert_complex_close_tol(seed.small_component_factor, small_component_factor, 1.0e-14);
    assert_complex_close_tol(seed.relativistic_scale, relativistic_scale, 1.0e-14);
    assert_complex_close_tol(seed.large_coefficient, expected_large, 1.0e-12);
    assert_complex_close_tol(seed.small_coefficient, expected_small, 1.0e-12);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_field_assembly_matches_feff_reference() -> Result<(), XsphError> {
    let radii = arr1(&[1.0, 1.2, 1.5, 1.9, 2.4, 3.0]);
    let regular_large = arr1(&[
        Complex::new(0.2, 0.05),
        Complex::new(0.23, 0.04),
        Complex::new(0.27, 0.03),
        Complex::new(0.30, 0.02),
        Complex::new(0.34, 0.01),
        Complex::new(0.38, -0.01),
    ]);
    let regular_small = arr1(&[
        Complex::new(0.05, -0.02),
        Complex::new(0.045, -0.015),
        Complex::new(0.04, -0.01),
        Complex::new(0.035, -0.005),
        Complex::new(0.03, 0.0),
        Complex::new(0.025, 0.005),
    ]);
    let irregular_large = arr1(&[
        Complex::new(0.3, -0.01),
        Complex::new(0.28, -0.015),
        Complex::new(0.25, -0.02),
        Complex::new(0.22, -0.025),
        Complex::new(0.19, -0.03),
        Complex::new(0.16, -0.035),
    ]);
    let irregular_small = arr1(&[
        Complex::new(-0.02, 0.04),
        Complex::new(-0.015, 0.035),
        Complex::new(-0.01, 0.03),
        Complex::new(-0.005, 0.025),
        Complex::new(0.0, 0.02),
        Complex::new(0.005, 0.015),
    ]);
    let wave_number = Complex::new(0.74, 0.06);

    let fields = xsph_xsect_phiscf_field_assembly(XsphXsectPhiscfFieldAssemblyInput {
        final_kappa: -2,
        wave_number,
        radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        active_len: radii.len(),
        match_index_1based: 3,
    })?;

    assert_eq!(
        fields.channels,
        XsphXsectPhiscfAngularChannels {
            large_l: 1,
            small_l: 2
        }
    );

    let match_index = 2;
    let match_argument = wave_number * radii[match_index];
    let match_hankel = crate::besjh(match_argument, 2)?;
    let wronskian_denominator = 2.0
        * (1.0 / XSPH_FINE_STRUCTURE_ALPHA)
        * (irregular_large[match_index] * regular_small[match_index]
            - regular_large[match_index] * irregular_small[match_index]);
    let wronskian_scale = Complex::new(2.0, 0.0) / wronskian_denominator;
    assert_complex_close_tol(fields.wronskian_scale, wronskian_scale, 1.0e-14);

    for index in 0..=match_index {
        assert_complex_close_tol(
            fields.regular_large[index],
            regular_large[index] * wronskian_scale,
            1.0e-12,
        );
        assert_complex_close_tol(
            fields.regular_small[index],
            regular_small[index] * wronskian_scale,
            1.0e-12,
        );
        assert_complex_close_tol(
            fields.irregular_large[index],
            irregular_large[index],
            1.0e-12,
        );
        assert_complex_close_tol(
            fields.irregular_small[index],
            irregular_small[index],
            1.0e-12,
        );
    }

    let tail_coefficient = (fields.regular_large[match_index] / (match_argument * 2.0)
        - match_hankel.j[1])
        / match_hankel.h[1];
    assert_complex_close_tol(fields.tail_coefficient, tail_coefficient, 1.0e-12);

    for index in (match_index + 1)..radii.len() {
        let argument = wave_number * radii[index];
        let hankel = crate::besjh(argument, 2)?;
        let radius_ratio = radii[index] / radii[match_index];
        let expected_irregular_large =
            irregular_large[match_index] * radius_ratio * hankel.h[1] / match_hankel.h[1];
        let expected_irregular_small =
            irregular_small[match_index] * radius_ratio * hankel.h[2] / match_hankel.h[2];
        let expected_regular_large =
            2.0 * argument * (hankel.j[1] + tail_coefficient * hankel.h[1]);
        let expected_regular_small =
            2.0 * argument * (hankel.j[2] + tail_coefficient * hankel.h[2]);

        assert_complex_close_tol(
            fields.irregular_large[index],
            expected_irregular_large,
            1.0e-12,
        );
        assert_complex_close_tol(
            fields.irregular_small[index],
            expected_irregular_small,
            1.0e-12,
        );
        assert_complex_close_tol(fields.regular_large[index], expected_regular_large, 1.0e-12);
        assert_complex_close_tol(fields.regular_small[index], expected_regular_small, 1.0e-12);
    }

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_field_assembly_rejects_invalid_inputs() {
    let error = xsph_xsect_phiscf_angular_channels(0)
        .expect_err("FEFF kfin must select a physical relativistic channel");
    assert!(matches!(error, XsphError::ZeroKappa));

    let error = xsph_xsect_phiscf_irregular_seed(XsphXsectPhiscfIrregularSeedInput {
        final_kappa: -1,
        wave_number: Complex::new(0.7, 0.02),
        match_radius: 0.0,
    })
    .expect_err("FEFF rmtp is positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveRadius {
            name: "xsect_phiscf_match_radius",
            value: 0.0
        }
    ));

    let radii = arr1(&[1.0, 1.2]);
    let regular_large = arr1(&[Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)]);
    let regular_small = arr1(&[Complex::new(2.0, 0.0), Complex::new(2.0, 0.0)]);
    let irregular_large = arr1(&[Complex::new(3.0, 0.0), Complex::new(3.0, 0.0)]);
    let irregular_small = arr1(&[Complex::new(6.0, 0.0), Complex::new(6.0, 0.0)]);
    let error = xsph_xsect_phiscf_field_assembly(XsphXsectPhiscfFieldAssemblyInput {
        final_kappa: -1,
        wave_number: Complex::new(0.7, 0.02),
        radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        active_len: radii.len(),
        match_index_1based: 1,
    })
    .expect_err("FEFF Wronskian denominator must be nonzero");
    assert!(matches!(
        error,
        XsphError::ZeroComplexResult {
            name: "xsect_phiscf_wronskian_denominator"
        }
    ));
}

#[test]
fn xsph_xsect_phiscf_radial_contribution_matches_field_lipman_chain() -> Result<(), XsphError> {
    let fixture = phiscf_radial_contribution_fixture();
    let contribution =
        xsph_xsect_phiscf_radial_contribution(XsphXsectPhiscfRadialContributionInput {
            coarse_count: 2,
            active_len: fixture.radii.len(),
            match_index_1based: 3,
            final_kappa: -2,
            wave_number: fixture.wave_number,
            radii: fixture.radii.view(),
            orbital_large: fixture.orbital_large.view(),
            orbital_small: fixture.orbital_small.view(),
            regular_large: fixture.regular_large.view(),
            regular_small: fixture.regular_small.view(),
            irregular_large: fixture.irregular_large.view(),
            irregular_small: fixture.irregular_small.view(),
            local_field: fixture.local_field.view(),
            response_scale: 0.75,
            include_response_imaginary: false,
        })?;

    let fields = xsph_xsect_phiscf_field_assembly(XsphXsectPhiscfFieldAssemblyInput {
        final_kappa: -2,
        wave_number: fixture.wave_number,
        radii: fixture.radii.view(),
        regular_large: fixture.regular_large.view(),
        regular_small: fixture.regular_small.view(),
        irregular_large: fixture.irregular_large.view(),
        irregular_small: fixture.irregular_small.view(),
        active_len: fixture.radii.len(),
        match_index_1based: 3,
    })?;
    let response = xsph_xsect_phiscf_lipman_response(XsphXsectPhiscfLipmanInput {
        coarse_count: 2,
        active_len: fixture.radii.len(),
        match_index_1based: 3,
        radii: fixture.radii.view(),
        orbital_large: fixture.orbital_large.view(),
        orbital_small: fixture.orbital_small.view(),
        regular_large: fields.regular_large.view(),
        regular_small: fields.regular_small.view(),
        irregular_large: fields.irregular_large.view(),
        irregular_small: fields.irregular_small.view(),
        local_field: fixture.local_field.view(),
    })?;

    assert_complex_close_tol(
        contribution.fields.wronskian_scale,
        fields.wronskian_scale,
        1.0e-14,
    );
    assert_complex_close_tol(
        contribution.fields.tail_coefficient,
        fields.tail_coefficient,
        1.0e-12,
    );
    for row in 0..response.response.nrows() {
        for column in 0..response.response.ncols() {
            assert_complex_close_tol(
                contribution.response.response[(row, column)],
                response.response[(row, column)],
                1.0e-12,
            );
        }
    }
    assert_close(contribution.scale, 0.75);
    assert!(!contribution.include_imaginary);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_wfirdc_contribution_matches_explicit_source_chain() -> Result<(), XsphError> {
    let fixture = phiscf_wfirdc_contribution_fixture();
    let alpha_scaled = fixture.energy * XSPH_FINE_STRUCTURE_ALPHA;
    let wave_number = (fixture.energy * 2.0 + alpha_scaled * alpha_scaled).sqrt();

    let generated =
        xsph_xsect_phiscf_wfirdc_contribution(XsphXsectPhiscfWfirdcContributionInput {
            coarse_count: 3,
            wave_number,
            wfirdc_input: fixture.to_input(),
            orbital_large: fixture.orbital_large.view(),
            orbital_small: fixture.orbital_small.view(),
            local_field: fixture.local_field.view(),
            response_scale: 0.6,
            include_response_imaginary: true,
        })?;

    let regular_solution =
        crate::fovrg_initial_photoelectron(crate::FovrgInitialPhotoelectronInput {
            irregular: false,
            initial_large_coefficient: Complex::new(0.0, 0.0),
            initial_small_coefficient: Complex::new(0.0, 0.0),
            coefficient_count: 3,
            ..fixture.to_input()
        })?;
    let irregular_seed = xsph_xsect_phiscf_irregular_seed(XsphXsectPhiscfIrregularSeedInput {
        final_kappa: fixture.kappa[fixture.kappa.len() - 1],
        wave_number,
        match_radius: fixture.muffin_tin_radius,
    })?;
    let irregular_solution =
        crate::fovrg_initial_photoelectron(crate::FovrgInitialPhotoelectronInput {
            irregular: true,
            initial_large_coefficient: irregular_seed.large_coefficient,
            initial_small_coefficient: irregular_seed.small_coefficient,
            coefficient_count: 2,
            ..fixture.to_input()
        })?;
    let expected = xsph_xsect_phiscf_radial_contribution(XsphXsectPhiscfRadialContributionInput {
        coarse_count: 3,
        active_len: fixture.active_len,
        match_index_1based: fixture.radial_match_index + 1,
        final_kappa: fixture.kappa[fixture.kappa.len() - 1],
        wave_number,
        radii: regular_solution.nuclear_potential.radii.view(),
        orbital_large: fixture.orbital_large.view(),
        orbital_small: fixture.orbital_small.view(),
        regular_large: regular_solution.large_component.view(),
        regular_small: regular_solution.small_component.view(),
        irregular_large: irregular_solution.large_component.view(),
        irregular_small: irregular_solution.small_component.view(),
        local_field: fixture.local_field.view(),
        response_scale: 0.6,
        include_response_imaginary: true,
    })?;

    assert_complex_close_tol(
        generated.irregular_seed.large_coefficient,
        irregular_seed.large_coefficient,
        1.0e-12,
    );
    assert_complex_close_tol(
        generated.irregular_seed.small_coefficient,
        irregular_seed.small_coefficient,
        1.0e-12,
    );
    assert_complex_close_tol(
        generated.regular_solution.large_component[fixture.radial_match_index],
        regular_solution.large_component[fixture.radial_match_index],
        1.0e-12,
    );
    assert_complex_close_tol(
        generated.irregular_solution.small_component[fixture.radial_match_index],
        irregular_solution.small_component[fixture.radial_match_index],
        1.0e-12,
    );
    assert_complex_close_tol(
        generated.contribution.fields.wronskian_scale,
        expected.fields.wronskian_scale,
        1.0e-12,
    );
    for row in 0..expected.response.response.nrows() {
        for column in 0..expected.response.response.ncols() {
            assert_complex_close_tol(
                generated.contribution.response.response[(row, column)],
                expected.response.response[(row, column)],
                1.0e-10,
            );
        }
    }
    assert_close(generated.contribution.scale, 0.6);
    assert!(generated.contribution.include_imaginary);

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_wfirdc_contributions_collects_and_solves_source_rows() -> Result<(), XsphError>
{
    let fixture = phiscf_wfirdc_contribution_fixture();
    let alpha_scaled = fixture.energy * XSPH_FINE_STRUCTURE_ALPHA;
    let wave_number = (fixture.energy * 2.0 + alpha_scaled * alpha_scaled).sqrt();
    let radii = phiscf_loucks_grid(fixture.active_len, fixture.step, XSPH_HOLE_ORBITAL_X0);
    let basis_fields = Array2::from_shape_fn((fixture.active_len, 2), |(row, column)| {
        Complex::new(
            0.1 + row as Real * 0.02 - column as Real * 0.03,
            -0.05 + column as Real * 0.04,
        )
    });

    let first_input = XsphXsectPhiscfWfirdcContributionInput {
        coarse_count: 3,
        wave_number,
        wfirdc_input: fixture.to_input(),
        orbital_large: fixture.orbital_large.view(),
        orbital_small: fixture.orbital_small.view(),
        local_field: fixture.local_field.view(),
        response_scale: 0.6,
        include_response_imaginary: true,
    };
    let second_input = XsphXsectPhiscfWfirdcContributionInput {
        response_scale: -0.2,
        include_response_imaginary: false,
        ..first_input
    };
    let contribution_inputs = [first_input, second_input];

    let collected =
        xsph_xsect_phiscf_wfirdc_contributions(XsphXsectPhiscfWfirdcContributionsInput {
            coarse_count: 3,
            radii: radii.view(),
            contribution_inputs: &contribution_inputs,
            basis_fields: basis_fields.view(),
            basis_count: 2,
        })?;

    let expected_first = xsph_xsect_phiscf_wfirdc_contribution(first_input)?;
    let expected_second = xsph_xsect_phiscf_wfirdc_contribution(second_input)?;
    let response_contributions = [
        XsphXsectPhiscfResponseContributionInput {
            response: expected_first.contribution.response.response.view(),
            scale: expected_first.contribution.scale,
            include_imaginary: expected_first.contribution.include_imaginary,
        },
        XsphXsectPhiscfResponseContributionInput {
            response: expected_second.contribution.response.response.view(),
            scale: expected_second.contribution.scale,
            include_imaginary: expected_second.contribution.include_imaginary,
        },
    ];
    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: 3,
            contributions: &response_contributions,
        })?;
    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 3,
        radii: radii.view(),
        response: accumulated.response.view(),
        basis_fields: basis_fields.view(),
        basis_count: 2,
    })?;

    assert_eq!(collected.contributions.len(), 2);
    assert_complex_close_tol(
        collected.contributions[0]
            .contribution
            .fields
            .wronskian_scale,
        expected_first.contribution.fields.wronskian_scale,
        1.0e-12,
    );
    assert_complex_close_tol(
        collected.contributions[1]
            .contribution
            .fields
            .wronskian_scale,
        expected_second.contribution.fields.wronskian_scale,
        1.0e-12,
    );
    for row in 0..accumulated.response.nrows() {
        for column in 0..accumulated.response.ncols() {
            assert_complex_close_tol(
                collected.screened_solution.response[(row, column)],
                accumulated.response[(row, column)],
                1.0e-12,
            );
        }
    }
    for index in 0..solved.screened_field.len() {
        assert_complex_close_tol(
            collected.screened_solution.screened_field[index],
            solved.screened_field[index],
            2.0e-6,
        );
    }
    for row in 0..solved.screened_basis_fields.nrows() {
        for column in 0..solved.screened_basis_fields.ncols() {
            assert_complex_close_tol(
                collected.screened_solution.screened_basis_fields[(row, column)],
                solved.screened_basis_fields[(row, column)],
                2.0e-6,
            );
        }
    }

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_screened_contributions_matches_feff_accumulation_chain()
-> Result<(), XsphError> {
    let fixture = phiscf_radial_contribution_fixture();
    let first = xsph_xsect_phiscf_radial_contribution(XsphXsectPhiscfRadialContributionInput {
        coarse_count: 2,
        active_len: fixture.radii.len(),
        match_index_1based: 3,
        final_kappa: -2,
        wave_number: fixture.wave_number,
        radii: fixture.radii.view(),
        orbital_large: fixture.orbital_large.view(),
        orbital_small: fixture.orbital_small.view(),
        regular_large: fixture.regular_large.view(),
        regular_small: fixture.regular_small.view(),
        irregular_large: fixture.irregular_large.view(),
        irregular_small: fixture.irregular_small.view(),
        local_field: fixture.local_field.view(),
        response_scale: 1.0,
        include_response_imaginary: true,
    })?;
    let second = xsph_xsect_phiscf_radial_contribution(XsphXsectPhiscfRadialContributionInput {
        response_scale: -0.25,
        include_response_imaginary: false,
        ..XsphXsectPhiscfRadialContributionInput {
            coarse_count: 2,
            active_len: fixture.radii.len(),
            match_index_1based: 3,
            final_kappa: -2,
            wave_number: fixture.wave_number,
            radii: fixture.radii.view(),
            orbital_large: fixture.orbital_large.view(),
            orbital_small: fixture.orbital_small.view(),
            regular_large: fixture.regular_large.view(),
            regular_small: fixture.regular_small.view(),
            irregular_large: fixture.irregular_large.view(),
            irregular_small: fixture.irregular_small.view(),
            local_field: fixture.local_field.view(),
            response_scale: 1.0,
            include_response_imaginary: true,
        }
    })?;
    let contributions = vec![first, second];

    let screened =
        xsph_xsect_phiscf_screened_contributions(XsphXsectPhiscfScreenedContributionsInput {
            coarse_count: 2,
            radii: fixture.radii.view(),
            contributions: &contributions,
            basis_fields: fixture.basis_fields.view(),
            basis_count: 2,
        })?;

    let contribution_inputs = contributions
        .iter()
        .map(|contribution| XsphXsectPhiscfResponseContributionInput {
            response: contribution.response.response.view(),
            scale: contribution.scale,
            include_imaginary: contribution.include_imaginary,
        })
        .collect::<Vec<_>>();
    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: 2,
            contributions: &contribution_inputs,
        })?;
    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 2,
        radii: fixture.radii.view(),
        response: accumulated.response.view(),
        basis_fields: fixture.basis_fields.view(),
        basis_count: 2,
    })?;

    for row in 0..accumulated.response.nrows() {
        for column in 0..accumulated.response.ncols() {
            assert_complex_close_tol(
                screened.response[(row, column)],
                accumulated.response[(row, column)],
                1.0e-12,
            );
        }
    }
    for index in 0..solved.screened_field.len() {
        assert_complex_close_tol(
            screened.screened_field[index],
            solved.screened_field[index],
            2.0e-6,
        );
    }
    for row in 0..solved.screened_basis_fields.nrows() {
        for column in 0..solved.screened_basis_fields.ncols() {
            assert_complex_close_tol(
                screened.screened_basis_fields[(row, column)],
                solved.screened_basis_fields[(row, column)],
                2.0e-6,
            );
        }
    }

    Ok(())
}

#[test]
fn xsph_xsect_phiscf_screened_solution_matches_lipman_chiklu_chain() -> Result<(), XsphError> {
    let radii = arr1(&[1.0, 1.2, 1.5, 1.9, 2.4, 3.0]);
    let orbital_large = arr1(&[0.8, 0.82, 0.85, 0.9, 0.95, 1.0]);
    let orbital_small = arr1(&[0.1, 0.11, 0.12, 0.13, 0.14, 0.15]);
    let regular_large = arr1(&[
        Complex::new(0.2, 0.05),
        Complex::new(0.23, 0.04),
        Complex::new(0.27, 0.03),
        Complex::new(0.30, 0.02),
        Complex::new(0.34, 0.01),
        Complex::new(0.38, -0.01),
    ]);
    let regular_small = arr1(&[
        Complex::new(0.05, -0.02),
        Complex::new(0.045, -0.015),
        Complex::new(0.04, -0.01),
        Complex::new(0.035, -0.005),
        Complex::new(0.03, 0.0),
        Complex::new(0.025, 0.005),
    ]);
    let irregular_large = arr1(&[
        Complex::new(0.3, -0.01),
        Complex::new(0.28, -0.015),
        Complex::new(0.25, -0.02),
        Complex::new(0.22, -0.025),
        Complex::new(0.19, -0.03),
        Complex::new(0.16, -0.035),
    ]);
    let irregular_small = arr1(&[
        Complex::new(-0.02, 0.04),
        Complex::new(-0.015, 0.035),
        Complex::new(-0.01, 0.03),
        Complex::new(-0.005, 0.025),
        Complex::new(0.0, 0.02),
        Complex::new(0.005, 0.015),
    ]);
    let local_field = arr1(&[0.1, 0.08, 0.06, 0.04, 0.02, 0.01]);
    let mut basis_fields = Array2::<Complex>::zeros((radii.len(), 2));
    for index in 0..radii.len() {
        basis_fields[(index, 0)] = Complex::new(0.2 + index as Real * 0.03, -0.1);
        basis_fields[(index, 1)] = Complex::new(-0.4 + index as Real * 0.02, 0.15);
    }

    let screened = xsph_xsect_phiscf_screened_solution(XsphXsectPhiscfScreenedSolutionInput {
        coarse_count: 2,
        active_len: radii.len(),
        match_index_1based: 4,
        radii: radii.view(),
        orbital_large: orbital_large.view(),
        orbital_small: orbital_small.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        local_field: local_field.view(),
        response_scale: 1.0,
        include_response_imaginary: true,
        basis_fields: basis_fields.view(),
        basis_count: 2,
    })?;

    let response = xsph_xsect_phiscf_lipman_response(XsphXsectPhiscfLipmanInput {
        coarse_count: 2,
        active_len: radii.len(),
        match_index_1based: 4,
        radii: radii.view(),
        orbital_large: orbital_large.view(),
        orbital_small: orbital_small.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        local_field: local_field.view(),
    })?;
    let contribution = [XsphXsectPhiscfResponseContributionInput {
        response: response.response.view(),
        scale: 1.0,
        include_imaginary: true,
    }];
    let accumulated =
        xsph_xsect_phiscf_accumulated_response(XsphXsectPhiscfAccumulatedResponseInput {
            coarse_count: 2,
            contributions: &contribution,
        })?;
    let solved = xsph_xsect_phiscf_linear_solve(XsphXsectPhiscfLinearSolveInput {
        coarse_count: 2,
        radii: radii.view(),
        response: accumulated.response.view(),
        basis_fields: basis_fields.view(),
        basis_count: 2,
    })?;

    assert_eq!(screened.response.dim(), accumulated.response.dim());
    for row in 0..accumulated.response.nrows() {
        for column in 0..accumulated.response.ncols() {
            assert_complex_close(
                screened.response[(row, column)],
                accumulated.response[(row, column)],
            );
        }
    }
    assert_eq!(screened.screened_field.len(), solved.screened_field.len());
    for index in 0..solved.screened_field.len() {
        assert_complex_close_tol(
            screened.screened_field[index],
            solved.screened_field[index],
            2.0e-6,
        );
    }
    assert_eq!(
        screened.screened_basis_fields.dim(),
        solved.screened_basis_fields.dim()
    );
    for row in 0..solved.screened_basis_fields.nrows() {
        for column in 0..solved.screened_basis_fields.ncols() {
            assert_complex_close_tol(
                screened.screened_basis_fields[(row, column)],
                solved.screened_basis_fields[(row, column)],
                2.0e-6,
            );
        }
    }

    Ok(())
}

#[test]
fn xsph_xsect_fscf_weights_match_feff_id_loop() -> Result<(), XsphError> {
    let fscf = arr1(&[
        Complex::new(1.0, 0.2),
        Complex::new(-0.5, 0.3),
        Complex::new(2.0, -0.1),
        Complex::new(99.0, 99.0),
    ]);

    let standard = xsph_xsect_fscf_weights(XsphXsectFscfWeightsInput {
        standard_potential: true,
        fscf: fscf.view(),
        active_len: 3,
    })?;
    assert_eq!(standard.components.len(), 2);
    assert_eq!(standard.components[0].component_id, 1);
    assert_eq!(
        standard.components[0].part,
        XsphXsectFscfComponentPart::Real
    );
    assert_eq!(standard.components[0].weights, arr1(&[1.0, -0.5, 2.0]));
    assert_eq!(standard.components[1].component_id, 2);
    assert_eq!(
        standard.components[1].part,
        XsphXsectFscfComponentPart::Imaginary
    );
    assert_eq!(standard.components[1].weights, arr1(&[0.2, 0.3, -0.1]));

    let nonstandard = xsph_xsect_fscf_weights(XsphXsectFscfWeightsInput {
        standard_potential: false,
        fscf: fscf.view(),
        active_len: 3,
    })?;
    assert_eq!(nonstandard.components.len(), 1);
    assert_eq!(nonstandard.components[0].component_id, 1);
    assert_eq!(
        nonstandard.components[0].part,
        XsphXsectFscfComponentPart::Real
    );
    assert_eq!(nonstandard.components[0].weights, arr1(&[1.0, -0.5, 2.0]));

    Ok(())
}

#[test]
fn xsph_xsect_fscf_weights_reject_invalid_inputs() {
    let fscf = arr1(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    let error = xsph_xsect_fscf_weights(XsphXsectFscfWeightsInput {
        standard_potential: true,
        fscf: fscf.view(),
        active_len: 3,
    })
    .expect_err("active fscf prefix must be present");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "fscf",
            required: 3,
            actual: 2
        }
    ));

    let invalid_fscf = arr1(&[Complex::new(1.0, 0.0), Complex::new(Real::NAN, 0.0)]);
    let error = xsph_xsect_fscf_weights(XsphXsectFscfWeightsInput {
        standard_potential: true,
        fscf: invalid_fscf.view(),
        active_len: invalid_fscf.len(),
    })
    .expect_err("fscf samples must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex { name: "fscf", .. }
    ));
}

#[test]
fn xsph_xsect_radial_pass_matches_feff_ifl_scaling() -> Result<(), XsphError> {
    let reduced_standard = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::ReducedMatrixElement,
        standard_potential: true,
        photon_wave_number: 0.4,
        screened_field_scale: 0.7,
    })?;
    assert_eq!(reduced_standard.feff_ifl, -1);
    assert_close(reduced_standard.post_radint_scale, 0.4);

    let reduced_nonstandard = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::ReducedMatrixElement,
        standard_potential: false,
        photon_wave_number: 0.4,
        screened_field_scale: 0.7,
    })?;
    assert_eq!(reduced_nonstandard.feff_ifl, 1);
    assert_close(reduced_nonstandard.post_radint_scale, 1.0);

    let central_standard = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::CentralCrossSection,
        standard_potential: true,
        photon_wave_number: 0.4,
        screened_field_scale: 0.7,
    })?;
    assert_eq!(central_standard.feff_ifl, -2);
    assert_close(central_standard.post_radint_scale, 0.078_4);

    let central_nonstandard = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::CentralCrossSection,
        standard_potential: false,
        photon_wave_number: 0.4,
        screened_field_scale: 0.7,
    })?;
    assert_eq!(central_nonstandard.feff_ifl, 2);
    assert_close(central_nonstandard.post_radint_scale, 1.0);

    Ok(())
}

#[test]
fn xsph_xsect_radial_pass_rejects_invalid_inputs() {
    let error = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::ReducedMatrixElement,
        standard_potential: true,
        photon_wave_number: 0.0,
        screened_field_scale: 0.7,
    })
    .expect_err("standard-atom negative-ifl scaling needs FEFF xk0");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "photon_wave_number",
            value: 0.0
        }
    ));

    let error = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::CentralCrossSection,
        standard_potential: true,
        photon_wave_number: 0.4,
        screened_field_scale: 0.0,
    })
    .expect_err("standard central cross-section scaling needs FEFF ww");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "screened_field_scale",
            value: 0.0
        }
    ));
}

#[test]
fn xsph_xsect_regular_solution_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let regular_large = arr1(&[
        Complex::new(1.0, 0.2),
        Complex::new(-0.4, 0.5),
        Complex::new(0.3, -0.7),
        Complex::new(99.0, -99.0),
    ]);
    let regular_small = arr1(&[
        Complex::new(-0.2, 0.1),
        Complex::new(0.8, -0.1),
        Complex::new(-1.0, 0.0),
        Complex::new(88.0, -88.0),
    ]);

    let result = xsph_xsect_regular_solution(XsphXsectRegularSolutionInput {
        wave_number: Complex::new(0.4, 0.5),
        phase_amplitude: Complex::new(1.25, -0.4),
        final_kappa: -2,
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        active_len: 3,
    })?;

    assert_eq!(result.regular_large.len(), 3);
    assert_eq!(result.regular_small.len(), 3);
    assert_complex_close(
        result.small_component_factor,
        Complex::new(-0.001_459_482_078_780_620_7, -0.001_824_332_682_938_356_4),
    );
    assert_complex_close(
        result.relativistic_scale,
        Complex::new(1.000_000_599_040_804_3, -0.000_002_662_585_641_506_650_3),
    );
    assert_complex_close(
        result.regular_solution_scale,
        Complex::new(0.725_690_457_959_513_5, 0.232_218_816_478_531_07),
    );
    assert_complex_close(
        result.regular_large[0],
        Complex::new(0.679_246_694_663_807_3, 0.377_356_908_070_433_76),
    );
    assert_complex_close(
        result.regular_large[2],
        Complex::new(0.380_260_308_922_825_8, -0.438_317_675_628_100_1),
    );
    assert_complex_close(
        result.regular_small[0],
        Complex::new(-0.168_359_973_239_755_8, 0.026_125_282_500_245_13),
    );
    assert_complex_close(
        result.regular_small[2],
        Complex::new(-0.725_690_457_959_513_5, -0.232_218_816_478_531_07),
    );

    Ok(())
}

#[test]
fn xsph_xsect_irregular_initial_condition_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let result = xsph_xsect_irregular_initial_condition(XsphXsectIrregularInitialConditionInput {
        muffin_tin_radius: 1.7,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        final_kappa: 3,
        bessel_j_l: Complex::new(0.8, 0.1),
        neumann_l: Complex::new(-0.3, 0.05),
        bessel_j_l_plus_1: Complex::new(0.25, -0.03),
        neumann_l_plus_1: Complex::new(-0.6, 0.2),
    })?;

    assert_complex_close(
        result.small_component_factor,
        Complex::new(0.001_459_482_078_780_620_7, 0.001_824_332_682_938_356_4),
    );
    assert_complex_close(
        result.relativistic_scale,
        Complex::new(1.000_000_599_040_804_3, -0.000_002_662_585_641_506_650_3),
    );
    assert_complex_close(
        result.large_component,
        Complex::new(-0.215_795_629_731_268_1, -0.025_994_455_746_676_363),
    );
    assert_complex_close(
        result.small_component,
        Complex::new(-0.001_838_866_245_442_667_3, -0.001_316_132_001_240_696_6),
    );

    Ok(())
}

#[test]
fn xsph_xsect_irregular_transform_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let regular_large = arr1(&[
        Complex::new(0.6, 0.1),
        Complex::new(-0.2, 0.4),
        Complex::new(0.3, -0.5),
    ]);
    let regular_small = arr1(&[
        Complex::new(0.01, -0.02),
        Complex::new(-0.03, 0.04),
        Complex::new(0.05, 0.06),
    ]);
    let irregular_large = arr1(&[
        Complex::new(-0.7, 0.2),
        Complex::new(0.4, -0.3),
        Complex::new(0.8, 0.1),
    ]);
    let irregular_small = arr1(&[
        Complex::new(0.02, 0.03),
        Complex::new(-0.05, 0.07),
        Complex::new(0.11, -0.09),
    ]);

    let result = xsph_xsect_irregular_transform(XsphXsectIrregularTransformInput {
        phase_shift: Complex::new(0.2, -0.1),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        active_len: 3,
    })?;

    assert_complex_close(
        result.phase_factor,
        Complex::new(1.083_141_079_608_063_2, 0.219_563_566_708_252_36),
    );
    assert_complex_close(
        result.irregular_large[0],
        Complex::new(0.702_111_469_067_294_7, 0.537_066_280_774_164),
    );
    assert_complex_close(
        result.irregular_large[2],
        Complex::new(-0.344_556_507_015_625_35, 0.016_035_038_672_591_76),
    );
    assert_complex_close(
        result.irregular_small[0],
        Complex::new(0.004_924_085_409_086_308, -0.026_885_503_722_406_938),
    );
    assert_complex_close(
        result.irregular_small[2],
        Complex::new(-0.198_906_239_760_629_66, 0.123_330_704_826_817_92),
    );

    Ok(())
}

#[test]
fn xsph_xsect_output_normalization_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let reduced_matrix_elements = arr1(&[
        Complex::new(0.1, 0.05),
        Complex::new(-0.2, 0.3),
        Complex::new(0.0, -0.1),
        Complex::new(99.0, -99.0),
    ]);
    let phase_shifts = arr1(&[
        Complex::new(0.2, -0.1),
        Complex::new(0.0, 0.3),
        Complex::new(-0.4, 0.2),
        Complex::new(88.0, -88.0),
    ]);

    let result = xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
        photon_energy: 1.3,
        wave_number: Complex::new(0.4, 0.5),
        spectrum_norm: 0.75,
        cross_section: Complex::new(0.2, -0.3),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        phase_shifts: phase_shifts.view(),
        active_channel_count: 3,
    })?;

    assert_close(result.prefactor, 370.939_840_103_318_15);
    assert_close(result.spectrum_norm, 356.276_082_119_253_3);
    assert_close(result.spectrum_norm_sqrt, 18.875_277_007_748_874);
    assert_complex_close(
        result.cross_section,
        Complex::new(170.632_326_447_526_34, -14.837_593_604_132_72),
    );
    assert_complex_close(
        result.reduced_matrix_root_scale,
        Complex::new(19.644_167_687_149_015, 9.441_475_098_636_596),
    );
    assert_complex_close(
        result.reduced_matrix_scale,
        Complex::new(1.040_735_332_206_488_3, 0.500_203_260_315_627_9),
    );
    assert_eq!(result.reduced_matrix_elements.len(), 3);
    assert_complex_close(
        result.reduced_matrix_elements[0],
        Complex::new(0.063_228_764_892_824_8, 0.127_901_665_063_949_48),
    );
    assert_complex_close(
        result.reduced_matrix_elements[1],
        Complex::new(-0.265_367_046_187_026_67, 0.157_186_771_244_498_65),
    );
    assert_complex_close(
        result.reduced_matrix_elements[2],
        Complex::new(0.004_538_739_079_270_451, -0.094_429_870_599_185_03),
    );

    Ok(())
}

#[test]
fn xsph_xsect_embedded_density_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let fixture = xsect_density_fixture();
    let result = xsph_xsect_embedded_density(fixture.embedded_input())?;

    assert_complex_close(
        result.prefactor,
        Complex::new(0.056_149_249_180_007_24, 0.070_185_795_281_871_84),
    );
    assert_complex_close(
        result.density_samples[0],
        Complex::new(0.154_400_000_000_000_04, -0.111_200_000_000_000_02),
    );
    assert_complex_close(
        result.density_samples[4],
        Complex::new(0.026_260_000_000_000_01, 0.056_830_000_000_000_006),
    );
    assert_complex_close(
        result.integral,
        Complex::new(0.034_286_264_435_008_63, -0.031_573_731_704_600_635),
    );
    assert_complex_close(
        result.density,
        Complex::new(-0.004_141_175_474_916_766, -0.000_633_567_407_591_323_6),
    );

    Ok(())
}

#[test]
fn xsph_xsect_projected_density_matches_feff_xsect_reference() -> Result<(), XsphError> {
    let fixture = xsect_density_fixture();
    let result = xsph_xsect_projected_density(fixture.projected_input())?;

    assert_complex_close(
        result.prefactor,
        Complex::new(0.056_149_249_180_007_24, 0.070_185_795_281_871_84),
    );
    assert_close(result.atomic_norm_integral, 0.098_524_764_950_738_17);
    assert_close(result.atomic_norm_sqrt, 0.313_886_547_897_067_7);
    assert_close(result.normalized_atomic_large[0], 2.867_278_021_405_159);
    assert_close(result.normalized_atomic_large[5], 0.796_466_117_056_988_7);
    assert_close(result.normalized_atomic_small[0], 0.159_293_223_411_397_73);
    assert_close(result.normalized_atomic_small[5], 0.254_869_157_458_236_35);
    assert_complex_close(
        result.cumulative_overlap[0],
        Complex::new(0.230_656_587_499_703_94, 0.056_708_387_534_457_595),
    );
    assert_complex_close(
        result.cumulative_overlap[4],
        Complex::new(0.591_802_997_753_556_6, 0.259_974_186_682_124_9),
    );
    assert_complex_close(
        result.density_samples[0],
        Complex::new(0.255_587_516_626_816_7, -0.183_797_444_318_230_06),
    );
    assert_complex_close(
        result.density_samples[4],
        Complex::new(0.218_252_817_408_410_3, 0.105_942_034_677_463_08),
    );
    assert_complex_close(
        result.integral,
        Complex::new(0.040_181_675_071_598_35, -0.038_046_390_244_283_78),
    );
    assert_complex_close(
        result.density,
        Complex::new(-0.004_926_487_042_964_769, -0.000_683_906_574_431_808_7),
    );

    Ok(())
}

#[test]
fn xsph_xsect_density_branch_matches_feff_projector_selection() -> Result<(), XsphError> {
    let projector_map = arr1(&[0, 0, 0, 0, 0, 5, 7, 0, 0, 0]);

    let fallback = xsph_xsect_density_branch(XsphXsectDensityBranchInput {
        initial_kappa: 1,
        final_kappa: -2,
        transition_delta: 1,
        spin_orbit_removed_pass: false,
        orbital_projector_map: projector_map.view(),
        min_projector_kappa: -5,
    })?
    .expect("FEFF falls back from iorb(-2) to iorb(1)");
    assert_eq!(fallback.required_transition_delta, 1);
    assert_eq!(fallback.projector_index_1based, 7);

    let direct = xsph_xsect_density_branch(XsphXsectDensityBranchInput {
        initial_kappa: -1,
        final_kappa: 0,
        transition_delta: -1,
        spin_orbit_removed_pass: false,
        orbital_projector_map: projector_map.view(),
        min_projector_kappa: -5,
    });
    assert!(matches!(direct, Err(XsphError::ZeroKappa)));

    let direct = xsph_xsect_density_branch(XsphXsectDensityBranchInput {
        initial_kappa: -1,
        final_kappa: 1,
        transition_delta: -1,
        spin_orbit_removed_pass: false,
        orbital_projector_map: projector_map.view(),
        min_projector_kappa: -5,
    })?
    .expect("FEFF uses direct iorb(1) when present");
    assert_eq!(direct.required_transition_delta, -1);
    assert_eq!(direct.projector_index_1based, 7);

    Ok(())
}

#[test]
fn xsph_xsect_density_branch_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let projector_map = arr1(&[0, 0, 0, 0, 0, 5, 7, 0, 0, 0]);

    for (transition_delta, spin_orbit_removed_pass, final_kappa) in
        [(-1, false, -2), (1, true, -2), (1, false, -3)]
    {
        let result = xsph_xsect_density_branch(XsphXsectDensityBranchInput {
            initial_kappa: 1,
            final_kappa,
            transition_delta,
            spin_orbit_removed_pass,
            orbital_projector_map: projector_map.view(),
            min_projector_kappa: -5,
        })?;
        assert_eq!(result, None);
    }

    let short = arr1(&[0, 0]);
    let error = xsph_xsect_density_branch(XsphXsectDensityBranchInput {
        initial_kappa: 1,
        final_kappa: 4,
        transition_delta: 1,
        spin_orbit_removed_pass: false,
        orbital_projector_map: short.view(),
        min_projector_kappa: -5,
    })
    .expect_err("iorb map must cover requested final kappa");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "orbital_projector_map",
            required: 10,
            actual: 2
        }
    ));

    Ok(())
}

#[test]
fn xsph_xsect_fscf_integral_matches_feff_magnitude_combination() -> Result<(), XsphError> {
    let first = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
        accumulated: Complex::new(9.0, 9.0),
        contribution: Complex::new(0.2, -0.4),
        first_component: true,
    })?;
    assert_eq!(first.selection, XsphXsectFscfSelection::FirstComponent);
    assert_close(first.scale, 1.0);
    assert_complex_close(first.value, Complex::new(0.2, -0.4));

    let accumulated_zero = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
        accumulated: Complex::new(0.0, 0.0),
        contribution: Complex::new(-0.3, 0.7),
        first_component: false,
    })?;
    assert_eq!(
        accumulated_zero.selection,
        XsphXsectFscfSelection::AccumulatedZero
    );
    assert_complex_close(accumulated_zero.value, Complex::new(-0.3, 0.7));

    let contribution_zero = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
        accumulated: Complex::new(0.5, -0.6),
        contribution: Complex::new(0.0, 0.0),
        first_component: false,
    })?;
    assert_eq!(
        contribution_zero.selection,
        XsphXsectFscfSelection::ContributionZero
    );
    assert_complex_close(contribution_zero.value, Complex::new(0.5, -0.6));

    let accumulated_dominant = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
        accumulated: Complex::new(3.0, 4.0),
        contribution: Complex::new(1.0, 1.0),
        first_component: false,
    })?;
    assert_eq!(
        accumulated_dominant.selection,
        XsphXsectFscfSelection::AccumulatedDominant
    );
    assert_close(accumulated_dominant.scale, 1.039_230_484_541_326_5);
    assert_complex_close(
        accumulated_dominant.value,
        Complex::new(3.117_691_453_623_979, 4.156_921_938_165_306),
    );

    let contribution_dominant = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
        accumulated: Complex::new(1.0, 1.0),
        contribution: Complex::new(3.0, 4.0),
        first_component: false,
    })?;
    assert_eq!(
        contribution_dominant.selection,
        XsphXsectFscfSelection::ContributionDominant
    );
    assert_close(contribution_dominant.scale, 1.039_230_484_541_326_5);
    assert_complex_close(
        contribution_dominant.value,
        Complex::new(3.117_691_453_623_979, 4.156_921_938_165_306),
    );

    Ok(())
}

#[test]
fn xsph_xsect_direct_transition_matches_feff_accumulation() -> Result<(), XsphError> {
    let dipole = xsph_xsect_direct_transition(XsphXsectDirectTransitionInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        radial_integral: Complex::new(0.3, -0.4),
        phase_shift: Complex::new(0.2, 0.05),
        angular_weight: Complex::new(0.6, 0.2),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
    })?;
    assert!(dipole.store_reduced_matrix);
    assert_eq!(dipole.reduced_matrix_element, Some(Complex::new(0.3, -0.4)));
    assert_eq!(dipole.phase_shift, Some(Complex::new(0.2, 0.05)));
    assert_close(dipole.spectrum_norm_increment, 1.0 / 12.0);
    assert_close(dipole.spectrum_norm, 1.583_333_333_333_333_3);
    assert_complex_close(dipole.cross_section_increment, Complex::new(0.158, 0.006));
    assert_complex_close(dipole.cross_section, Complex::new(0.358, -0.094));

    let quadrupole = xsph_xsect_direct_transition(XsphXsectDirectTransitionInput {
        multipole: XsphTransitionMultipole::ElectricQuadrupole,
        selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
        radial_integral: Complex::new(0.3, -0.4),
        phase_shift: Complex::new(-0.1, 0.02),
        angular_weight: Complex::new(-0.1, 0.3),
        spectrum_norm: 0.5,
        cross_section: Complex::new(-0.2, 0.05),
    })?;
    assert!(quadrupole.store_reduced_matrix);
    assert_eq!(
        quadrupole.reduced_matrix_element,
        Some(Complex::new(0.3, -0.4))
    );
    assert_close(quadrupole.spectrum_norm_increment, 0.05);
    assert_close(quadrupole.spectrum_norm, 0.55);
    assert_complex_close(
        quadrupole.cross_section_increment,
        Complex::new(-0.003, 0.079),
    );
    assert_complex_close(quadrupole.cross_section, Complex::new(-0.203, 0.129));

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_direct_transition_uses_traced_weights() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let result = xsph_xsect_bcoef_direct_transition(XsphXsectBcoefDirectTransitionInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        transition_index_1based: 1,
        diagonal_weights: weights.diagonal_weights.view(),
        radial_integral: Complex::new(0.3, -0.4),
        phase_shift: Complex::new(0.2, 0.05),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
    })?;

    assert_eq!(result.reduced_matrix_element, Some(Complex::new(0.3, -0.4)));
    assert_eq!(result.phase_shift, Some(Complex::new(0.2, 0.05)));
    assert_close(result.spectrum_norm_increment, 1.0 / 12.0);
    assert_complex_close(
        result.cross_section_increment,
        Complex::new(-0.08, 0.023_333_333_333_333_33),
    );
    assert_complex_close(
        result.cross_section,
        Complex::new(0.120_000_000_000_000_01, -0.076_666_666_666_666_66),
    );

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_direct_transition_update_stores_rkk_and_phx() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let reduced_matrix_elements = arr1(&[
        Complex::new(9.0, 0.0),
        Complex::new(8.0, 0.0),
        Complex::new(7.0, 0.0),
        Complex::new(6.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(4.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(2.0, 0.0),
    ]);
    let phase_shifts = arr1(&[
        Complex::new(0.9, 0.0),
        Complex::new(0.8, 0.0),
        Complex::new(0.7, 0.0),
        Complex::new(0.6, 0.0),
        Complex::new(0.5, 0.0),
        Complex::new(0.4, 0.0),
        Complex::new(0.3, 0.0),
        Complex::new(0.2, 0.0),
    ]);

    let result =
        xsph_xsect_bcoef_direct_transition_update(XsphXsectBcoefDirectTransitionUpdateInput {
            multipole: XsphTransitionMultipole::ElectricDipole,
            selected_higher_multipole: None,
            transition_index_1based: 1,
            diagonal_weights: weights.diagonal_weights.view(),
            radial_integral: Complex::new(0.3, -0.4),
            phase_shift: Complex::new(0.2, 0.05),
            spectrum_norm: 1.5,
            cross_section: Complex::new(0.2, -0.1),
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
        })?;

    assert!(result.transition.store_reduced_matrix);
    assert_close(result.spectrum_norm, 1.583_333_333_333_333_3);
    assert_complex_close(
        result.cross_section,
        Complex::new(0.120_000_000_000_000_01, -0.076_666_666_666_666_66),
    );
    assert_complex_close(result.reduced_matrix_elements[0], Complex::new(0.3, -0.4));
    assert_complex_close(result.phase_shifts[0], Complex::new(0.2, 0.05));
    assert_complex_close(
        result.reduced_matrix_elements[1],
        reduced_matrix_elements[1],
    );
    assert_complex_close(result.phase_shifts[1], phase_shifts[1]);

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_direct_transition_update_preserves_unstored_rkk_phx() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let reduced_matrix_elements = arr1(&[
        Complex::new(1.0, 0.1),
        Complex::new(2.0, 0.2),
        Complex::new(3.0, 0.3),
        Complex::new(4.0, 0.4),
        Complex::new(5.0, 0.5),
        Complex::new(6.0, 0.6),
        Complex::new(7.0, 0.7),
        Complex::new(8.0, 0.8),
    ]);
    let phase_shifts = arr1(&[
        Complex::new(-1.0, 0.1),
        Complex::new(-2.0, 0.2),
        Complex::new(-3.0, 0.3),
        Complex::new(-4.0, 0.4),
        Complex::new(-5.0, 0.5),
        Complex::new(-6.0, 0.6),
        Complex::new(-7.0, 0.7),
        Complex::new(-8.0, 0.8),
    ]);

    let result =
        xsph_xsect_bcoef_direct_transition_update(XsphXsectBcoefDirectTransitionUpdateInput {
            multipole: XsphTransitionMultipole::MagneticDipole,
            selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
            transition_index_1based: 6,
            diagonal_weights: weights.diagonal_weights.view(),
            radial_integral: Complex::new(0.1, 0.2),
            phase_shift: Complex::new(0.4, -0.3),
            spectrum_norm: 0.0,
            cross_section: Complex::new(0.0, 0.0),
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
        })?;

    assert!(!result.transition.store_reduced_matrix);
    assert_close(result.spectrum_norm, 1.0 / 60.0);
    assert_complex_close(result.cross_section, Complex::new(-0.008, -0.006));
    assert_eq!(result.reduced_matrix_elements, reduced_matrix_elements);
    assert_eq!(result.phase_shifts, phase_shifts);

    Ok(())
}

#[test]
fn xsph_xsect_direct_transition_preserves_feff_storage_branch() -> Result<(), XsphError> {
    let magnetic = xsph_xsect_direct_transition(XsphXsectDirectTransitionInput {
        multipole: XsphTransitionMultipole::MagneticDipole,
        selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
        radial_integral: Complex::new(0.1, 0.2),
        phase_shift: Complex::new(0.4, -0.3),
        angular_weight: Complex::new(0.5, 0.0),
        spectrum_norm: 0.0,
        cross_section: Complex::new(0.0, 0.0),
    })?;
    assert!(!magnetic.store_reduced_matrix);
    assert_eq!(magnetic.reduced_matrix_element, None);
    assert_eq!(magnetic.phase_shift, None);
    assert_close(magnetic.spectrum_norm_increment, 1.0 / 60.0);
    assert_complex_close(
        magnetic.cross_section_increment,
        Complex::new(-0.02, -0.015),
    );

    Ok(())
}

#[test]
fn xsph_xsect_direct_transition_rejects_invalid_inputs() {
    let error = xsph_xsect_direct_transition(XsphXsectDirectTransitionInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        radial_integral: Complex::new(0.3, -0.4),
        phase_shift: Complex::new(0.2, 0.05),
        angular_weight: Complex::new(Real::NAN, 0.2),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
    })
    .expect_err("angular coefficient must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_transition_angular_weight",
            ..
        }
    ));
}

#[test]
fn xsph_xsect_bcoef_direct_transition_rejects_invalid_inputs() {
    let diagonal_weights = Array1::<Complex>::zeros(8);
    let error = xsph_xsect_bcoef_direct_transition(XsphXsectBcoefDirectTransitionInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        transition_index_1based: 0,
        diagonal_weights: diagonal_weights.view(),
        radial_integral: Complex::new(0.3, -0.4),
        phase_shift: Complex::new(0.2, 0.05),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
    })
    .expect_err("bcoef transition slots are one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: 0,
            active_len: 8
        }
    ));

    let short_weights = Array1::<Complex>::zeros(7);
    let error = xsph_xsect_bcoef_direct_transition(XsphXsectBcoefDirectTransitionInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        transition_index_1based: 1,
        diagonal_weights: short_weights.view(),
        radial_integral: Complex::new(0.3, -0.4),
        phase_shift: Complex::new(0.2, 0.05),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
    })
    .expect_err("source-backed bcoef diagonal table has eight slots");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "xsect_bcoef_diagonal_weights",
            required: 8,
            actual: 7
        }
    ));
}

#[test]
fn xsph_xsect_bcoef_direct_transition_update_rejects_invalid_workspace() {
    let weights = Array1::<Complex>::zeros(8);
    let short_workspace = Array1::<Complex>::zeros(7);
    let phase_shifts = Array1::<Complex>::zeros(8);
    let error =
        xsph_xsect_bcoef_direct_transition_update(XsphXsectBcoefDirectTransitionUpdateInput {
            multipole: XsphTransitionMultipole::ElectricDipole,
            selected_higher_multipole: None,
            transition_index_1based: 1,
            diagonal_weights: weights.view(),
            radial_integral: Complex::new(0.3, -0.4),
            phase_shift: Complex::new(0.2, 0.05),
            spectrum_norm: 1.5,
            cross_section: Complex::new(0.2, -0.1),
            reduced_matrix_elements: short_workspace.view(),
            phase_shifts: phase_shifts.view(),
        })
        .expect_err("FEFF rkk workspace has eight slots");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "xsect_reduced_matrix_workspace",
            required: 8,
            actual: 7
        }
    ));

    let reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let mut nonfinite_phase = Array1::<Complex>::zeros(8);
    nonfinite_phase[3] = Complex::new(Real::NAN, 0.0);
    let error =
        xsph_xsect_bcoef_direct_transition_update(XsphXsectBcoefDirectTransitionUpdateInput {
            multipole: XsphTransitionMultipole::ElectricDipole,
            selected_higher_multipole: None,
            transition_index_1based: 1,
            diagonal_weights: weights.view(),
            radial_integral: Complex::new(0.3, -0.4),
            phase_shift: Complex::new(0.2, 0.05),
            spectrum_norm: 1.5,
            cross_section: Complex::new(0.2, -0.1),
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: nonfinite_phase.view(),
        })
        .expect_err("FEFF phx workspace must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_phase_workspace",
            index: 3,
            ..
        }
    ));
}

#[test]
fn xsph_xsect_central_cross_section_matches_feff_diagonal_update() -> Result<(), XsphError> {
    let result = xsph_xsect_central_cross_section(XsphXsectCentralCrossSectionInput {
        spin_orbit_removed_pass: false,
        radial_integral: Complex::new(0.3, -0.4),
        angular_weight: Complex::new(0.6, 0.2),
        cross_section: Complex::new(0.2, -0.1),
    })?
    .expect("FEFF ic3=0 applies the diagonal central xsec contribution");

    assert_complex_close(result.cross_section_increment, Complex::new(-0.26, 0.18));
    assert_complex_close(result.cross_section, Complex::new(-0.06, 0.08));

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_central_cross_section_uses_traced_weights() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let result = xsph_xsect_bcoef_central_cross_section(XsphXsectBcoefCentralCrossSectionInput {
        spin_orbit_removed_pass: false,
        transition_index_1based: 1,
        diagonal_weights: weights.diagonal_weights.view(),
        radial_integral: Complex::new(0.3, -0.4),
        cross_section: Complex::new(0.2, -0.1),
    })?
    .expect("FEFF ic3=0 applies the traced diagonal bcoef central update");

    assert_complex_close(
        result.cross_section_increment,
        Complex::new(0.1, -0.133_333_333_333_333_33),
    );
    assert_complex_close(
        result.cross_section,
        Complex::new(0.3, -0.233_333_333_333_333_34),
    );

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_ordinary_row_applies_direct_then_central_updates() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let phase_shifts = Array1::<Complex>::zeros(8);

    let result = xsph_xsect_bcoef_ordinary_row(XsphXsectBcoefOrdinaryRowInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        transition_index_1based: 1,
        diagonal_weights: weights.diagonal_weights.view(),
        reduced_matrix_integral: Complex::new(0.3, -0.4),
        central_cross_integral: Complex::new(0.05, 0.07),
        phase_shift: Complex::new(0.2, 0.05),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        phase_shifts: phase_shifts.view(),
    })?;

    assert_close(result.spectrum_norm, 1.583_333_333_333_333_3);
    assert_complex_close(
        result.direct_transition.cross_section,
        Complex::new(0.120_000_000_000_000_01, -0.076_666_666_666_666_66),
    );
    assert_complex_close(
        result.central_cross_section.cross_section_increment,
        Complex::new(0.016_666_666_666_666_666, 0.023_333_333_333_333_334),
    );
    assert_complex_close(
        result.cross_section,
        Complex::new(0.136_666_666_666_666_7, -0.053_333_333_333_333_32),
    );
    assert_complex_close(result.reduced_matrix_elements[0], Complex::new(0.3, -0.4));
    assert_complex_close(result.phase_shifts[0], Complex::new(0.2, 0.05));

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_standard_channel_row_combines_fscf_passes() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let transition = XsphXsectTransition {
        multipole: XsphTransitionMultipole::ElectricDipole,
        transition_delta: -1,
        transition_index_1based: 1,
        final_kappa: 1,
        final_l: 0,
        multipole_order: 1,
    };
    let fscf = arr1(&[
        Complex::new(1.0, 0.25),
        Complex::new(0.8, -0.15),
        Complex::new(1.1, 0.35),
        Complex::new(0.6, 0.05),
        Complex::new(0.9, -0.2),
        Complex::new(1.2, 0.4),
        Complex::new(0.7, 0.1),
    ]);
    let regular_channel = xsect_test_regular_channel(&fixture, Complex::new(0.2, 0.05));
    let irregular_channel = xsect_test_irregular_channel(&fixture);
    let reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let phase_shifts = Array1::<Complex>::zeros(8);
    let photon_wave_number = 0.37;
    let screened_field_scale = 0.8;

    let result = xsph_xsect_bcoef_standard_channel_row(XsphXsectBcoefStandardChannelRowInput {
        transition,
        selected_higher_multipole: None,
        initial_kappa: -1,
        initial_large: fixture.initial_large.view(),
        initial_small: fixture.initial_small.view(),
        regular_channel: &regular_channel,
        irregular_channel: &irregular_channel,
        xray_bessel: fixture.bessel.view(),
        radii: fixture.radii.view(),
        log_step: 0.137,
        photon_wave_number,
        screened_field_scale,
        fscf: fscf.view(),
        diagonal_weights: weights.diagonal_weights.view(),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        phase_shifts: phase_shifts.view(),
    })?;

    let fscf_weights = xsph_xsect_fscf_weights(XsphXsectFscfWeightsInput {
        standard_potential: true,
        fscf: fscf.view(),
        active_len: RADINT_ACTIVE_LEN,
    })?;
    let reduced_pass = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::ReducedMatrixElement,
        standard_potential: true,
        photon_wave_number,
        screened_field_scale,
    })?;
    let central_pass = xsph_xsect_radial_pass(XsphXsectRadialPassInput {
        kind: XsphXsectRadialPassKind::CentralCrossSection,
        standard_potential: true,
        photon_wave_number,
        screened_field_scale,
    })?;

    let mut reduced_matrix_integral = Complex::new(0.0, 0.0);
    let mut expected_reduced_components = Vec::new();
    let mut expected_reduced_fscf = Vec::new();
    for (component_index, component) in fscf_weights.components.iter().enumerate() {
        let integral = xsph_xsect_weighted_radial_integral(XsphXsectWeightedRadialIntegralInput {
            mode: XsphRadialIntegralMode::NonRelativisticMatrixElement,
            multipole: transition.multipole,
            initial_kappa: -1,
            final_kappa: transition.final_kappa,
            initial_large: fixture.initial_large.view(),
            initial_small: fixture.initial_small.view(),
            final_large_regular: fixture.final_large.view(),
            final_small_regular: fixture.final_small.view(),
            xray_bessel: fixture.bessel.view(),
            radii: fixture.radii.view(),
            log_step: 0.137,
            radial_weights: component.weights.view(),
            active_len: RADINT_ACTIVE_LEN,
        })?;
        let combined = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
            accumulated: reduced_matrix_integral,
            contribution: integral.integral.value * reduced_pass.post_radint_scale,
            first_component: component_index == 0,
        })?;
        reduced_matrix_integral = combined.value;
        expected_reduced_components.push(integral);
        expected_reduced_fscf.push(combined);
    }

    let mut central_cross_integral = Complex::new(0.0, 0.0);
    let mut expected_central_components = Vec::new();
    let mut expected_central_fscf = Vec::new();
    for (component_index, component) in fscf_weights.components.iter().enumerate() {
        let integral =
            xsph_xsect_weighted_radial_cross_integral(XsphXsectWeightedRadialCrossIntegralInput {
                mode: XsphRadialIntegralMode::NonRelativisticMatrixElement,
                branch: XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
                multipole: transition.multipole,
                initial_kappa: -1,
                final_kappa: transition.final_kappa,
                initial_large: fixture.initial_large.view(),
                initial_small: fixture.initial_small.view(),
                final_large_regular: fixture.final_large.view(),
                final_small_regular: fixture.final_small.view(),
                final_large_irregular: fixture.irregular_large.view(),
                final_small_irregular: fixture.irregular_small.view(),
                xray_bessel: fixture.bessel.view(),
                radii: fixture.radii.view(),
                log_step: 0.137,
                regular_weights: component.weights.view(),
                irregular_weights: component.weights.view(),
                active_len: RADINT_ACTIVE_LEN,
            })?;
        let combined = xsph_xsect_fscf_integral(XsphXsectFscfIntegralInput {
            accumulated: central_cross_integral,
            contribution: integral.integral.value * central_pass.post_radint_scale,
            first_component: component_index == 0,
        })?;
        central_cross_integral = combined.value;
        expected_central_components.push(integral);
        expected_central_fscf.push(combined);
    }

    let expected_row = xsph_xsect_bcoef_ordinary_row(XsphXsectBcoefOrdinaryRowInput {
        multipole: transition.multipole,
        selected_higher_multipole: None,
        transition_index_1based: transition.transition_index_1based,
        diagonal_weights: weights.diagonal_weights.view(),
        reduced_matrix_integral,
        central_cross_integral,
        phase_shift: regular_channel.phase.phase_shift,
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        phase_shifts: phase_shifts.view(),
    })?;

    assert_eq!(result.fscf_weights.components.len(), 2);
    assert_eq!(result.reduced_radial_pass.feff_ifl, -1);
    assert_eq!(result.central_radial_pass.feff_ifl, -2);
    assert_close(
        result.reduced_radial_pass.post_radint_scale,
        reduced_pass.post_radint_scale,
    );
    assert_close(
        result.central_radial_pass.post_radint_scale,
        central_pass.post_radint_scale,
    );
    assert_eq!(
        result.reduced_fscf_integrals[0].selection,
        XsphXsectFscfSelection::FirstComponent
    );
    assert_eq!(
        result.central_fscf_integrals[0].selection,
        XsphXsectFscfSelection::FirstComponent
    );

    for index in 0..2 {
        assert_complex_close(
            result.reduced_component_integrals[index].integral.value,
            expected_reduced_components[index].integral.value,
        );
        assert_complex_close(
            result.reduced_fscf_integrals[index].value,
            expected_reduced_fscf[index].value,
        );
        assert_complex_close(
            result.central_component_integrals[index].integral.value,
            expected_central_components[index].integral.value,
        );
        assert_complex_close(
            result.central_fscf_integrals[index].value,
            expected_central_fscf[index].value,
        );
    }
    assert_close(result.row.spectrum_norm, expected_row.spectrum_norm);
    assert_complex_close(result.row.cross_section, expected_row.cross_section);
    assert_complex_close(
        result.row.reduced_matrix_elements[0],
        expected_row.reduced_matrix_elements[0],
    );
    assert_complex_close(result.row.phase_shifts[0], expected_row.phase_shifts[0]);

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_standard_energy_row_folds_transition_rows() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let transitions = vec![
        XsphXsectTransition {
            multipole: XsphTransitionMultipole::ElectricDipole,
            transition_delta: -1,
            transition_index_1based: 1,
            final_kappa: 1,
            final_l: 0,
            multipole_order: 1,
        },
        XsphXsectTransition {
            multipole: XsphTransitionMultipole::ElectricDipole,
            transition_delta: 0,
            transition_index_1based: 2,
            final_kappa: 1,
            final_l: 0,
            multipole_order: 1,
        },
    ];
    let regular_channels = vec![
        xsect_test_regular_channel(&fixture, Complex::new(0.2, 0.05)),
        xsect_test_regular_channel(&fixture, Complex::new(-0.1, 0.08)),
    ];
    let irregular_channels = vec![
        xsect_test_irregular_channel(&fixture),
        xsect_test_irregular_channel(&fixture),
    ];
    let fscf = arr1(&[
        Complex::new(1.0, 0.25),
        Complex::new(0.8, -0.15),
        Complex::new(1.1, 0.35),
        Complex::new(0.6, 0.05),
        Complex::new(0.9, -0.2),
        Complex::new(1.2, 0.4),
        Complex::new(0.7, 0.1),
    ]);
    let photon_wave_number = 0.37;
    let screened_field_scale = 0.8;
    let photon_energy = 1.3;
    let wave_number = Complex::new(0.4, 0.5);

    let result = xsph_xsect_bcoef_standard_energy_row(XsphXsectBcoefStandardEnergyRowInput {
        transitions: &transitions,
        regular_channels: &regular_channels,
        irregular_channels: &irregular_channels,
        selected_higher_multipole: None,
        initial_kappa: -1,
        initial_large: fixture.initial_large.view(),
        initial_small: fixture.initial_small.view(),
        xray_bessel: fixture.bessel.view(),
        radii: fixture.radii.view(),
        log_step: 0.137,
        photon_wave_number,
        screened_field_scale,
        fscf: fscf.view(),
        diagonal_weights: weights.diagonal_weights.view(),
        spin_polarized_cross_terms: false,
        orbital_l: weights.orbital_l.view(),
        trace_weights: weights.trace_weights.view(),
        spin_orbit_removed_regular_channels: None,
        spin_orbit_removed_irregular_channels: None,
        photon_energy,
        wave_number,
        active_channel_count: transitions.len(),
    })?;
    let transition_fields = vec![
        XsphXsectBcoefStandardTransitionField {
            screened_field_scale,
            fscf: fscf.view(),
        };
        transitions.len()
    ];
    let fields_result = xsph_xsect_bcoef_standard_energy_row_with_transition_fields(
        XsphXsectBcoefStandardEnergyRowFieldsInput {
            transitions: &transitions,
            regular_channels: &regular_channels,
            irregular_channels: &irregular_channels,
            transition_fields: &transition_fields,
            selected_higher_multipole: None,
            initial_kappa: -1,
            initial_large: fixture.initial_large.view(),
            initial_small: fixture.initial_small.view(),
            xray_bessel: fixture.bessel.view(),
            radii: fixture.radii.view(),
            log_step: 0.137,
            photon_wave_number,
            diagonal_weights: weights.diagonal_weights.view(),
            spin_polarized_cross_terms: false,
            orbital_l: weights.orbital_l.view(),
            trace_weights: weights.trace_weights.view(),
            spin_orbit_removed_regular_channels: None,
            spin_orbit_removed_irregular_channels: None,
            photon_energy,
            wave_number,
            active_channel_count: transitions.len(),
        },
    )?;
    assert_eq!(fields_result, result);

    let mut spectrum_norm = 0.0;
    let mut cross_section = Complex::new(0.0, 0.0);
    let mut reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let mut phase_shifts = Array1::<Complex>::zeros(8);
    let mut expected_rows = Vec::new();
    for (index, transition) in transitions.iter().copied().enumerate() {
        let row = xsph_xsect_bcoef_standard_channel_row(XsphXsectBcoefStandardChannelRowInput {
            transition,
            selected_higher_multipole: None,
            initial_kappa: -1,
            initial_large: fixture.initial_large.view(),
            initial_small: fixture.initial_small.view(),
            regular_channel: &regular_channels[index],
            irregular_channel: &irregular_channels[index],
            xray_bessel: fixture.bessel.view(),
            radii: fixture.radii.view(),
            log_step: 0.137,
            photon_wave_number,
            screened_field_scale,
            fscf: fscf.view(),
            diagonal_weights: weights.diagonal_weights.view(),
            spectrum_norm,
            cross_section,
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
        })?;
        spectrum_norm = row.row.spectrum_norm;
        cross_section = row.row.cross_section;
        reduced_matrix_elements.assign(&row.row.reduced_matrix_elements);
        phase_shifts.assign(&row.row.phase_shifts);
        expected_rows.push(row);
    }
    let expected_output = xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
        photon_energy,
        wave_number,
        spectrum_norm,
        cross_section,
        reduced_matrix_elements: reduced_matrix_elements.view(),
        phase_shifts: phase_shifts.view(),
        active_channel_count: transitions.len(),
    })?;

    assert_eq!(result.transition_rows.len(), transitions.len());
    assert!(result.cross_term_updates.is_empty());
    for (index, expected) in expected_rows.iter().enumerate().take(transitions.len()) {
        assert_complex_close(
            result.transition_rows[index].row.cross_section,
            expected.row.cross_section,
        );
        assert_close(
            result.transition_rows[index].row.spectrum_norm,
            expected.row.spectrum_norm,
        );
    }
    assert_close(result.unnormalized_spectrum_norm, spectrum_norm);
    assert_complex_close(result.unnormalized_cross_section, cross_section);
    for index in 0..transitions.len() {
        assert_complex_close(
            result.unnormalized_reduced_matrix_elements[index],
            reduced_matrix_elements[index],
        );
        assert_complex_close(result.phase_shifts[index], phase_shifts[index]);
        assert_complex_close(
            result.output_normalization.reduced_matrix_elements[index],
            expected_output.reduced_matrix_elements[index],
        );
    }
    assert_close(
        result.output_normalization.spectrum_norm,
        expected_output.spectrum_norm,
    );
    assert_complex_close(
        result.output_normalization.cross_section,
        expected_output.cross_section,
    );

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_standard_energy_row_uses_per_transition_fields() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let transitions = vec![
        XsphXsectTransition {
            multipole: XsphTransitionMultipole::ElectricDipole,
            transition_delta: -1,
            transition_index_1based: 1,
            final_kappa: 1,
            final_l: 0,
            multipole_order: 1,
        },
        XsphXsectTransition {
            multipole: XsphTransitionMultipole::ElectricQuadrupole,
            transition_delta: -2,
            transition_index_1based: 4,
            final_kappa: 1,
            final_l: 2,
            multipole_order: 2,
        },
    ];
    let regular_channels = vec![
        xsect_test_regular_channel(&fixture, Complex::new(0.2, 0.05)),
        xsect_test_regular_channel(&fixture, Complex::new(-0.1, 0.08)),
    ];
    let irregular_channels = vec![
        xsect_test_irregular_channel(&fixture),
        xsect_test_irregular_channel(&fixture),
    ];
    let screened_fscf = arr1(&[
        Complex::new(1.0, 0.25),
        Complex::new(0.8, -0.15),
        Complex::new(1.1, 0.35),
        Complex::new(0.6, 0.05),
        Complex::new(0.9, -0.2),
        Complex::new(1.2, 0.4),
        Complex::new(0.7, 0.1),
    ]);
    let unity_fscf = Array1::from(vec![Complex::new(1.0, 0.0); RADINT_ACTIVE_LEN]);
    let transition_fields = vec![
        XsphXsectBcoefStandardTransitionField {
            screened_field_scale: 0.8,
            fscf: screened_fscf.view(),
        },
        XsphXsectBcoefStandardTransitionField {
            screened_field_scale: 1.0,
            fscf: unity_fscf.view(),
        },
    ];

    let result = xsph_xsect_bcoef_standard_energy_row_with_transition_fields(
        XsphXsectBcoefStandardEnergyRowFieldsInput {
            transitions: &transitions,
            regular_channels: &regular_channels,
            irregular_channels: &irregular_channels,
            transition_fields: &transition_fields,
            selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
            initial_kappa: -1,
            initial_large: fixture.initial_large.view(),
            initial_small: fixture.initial_small.view(),
            xray_bessel: fixture.bessel.view(),
            radii: fixture.radii.view(),
            log_step: 0.137,
            photon_wave_number: 0.37,
            diagonal_weights: weights.diagonal_weights.view(),
            spin_polarized_cross_terms: false,
            orbital_l: weights.orbital_l.view(),
            trace_weights: weights.trace_weights.view(),
            spin_orbit_removed_regular_channels: None,
            spin_orbit_removed_irregular_channels: None,
            photon_energy: 1.3,
            wave_number: Complex::new(0.4, 0.5),
            active_channel_count: 8,
        },
    )?;

    assert_eq!(result.transition_rows.len(), 2);
    assert_eq!(
        result.transition_rows[0].fscf_weights.components[0].weights,
        screened_fscf.mapv(|value| value.re)
    );
    assert_eq!(
        result.transition_rows[0].fscf_weights.components[1].weights,
        screened_fscf.mapv(|value| value.im)
    );
    assert_eq!(
        result.transition_rows[1].fscf_weights.components[0].weights,
        Array1::<Real>::ones(RADINT_ACTIVE_LEN)
    );
    assert_eq!(
        result.transition_rows[1].fscf_weights.components[1].weights,
        Array1::<Real>::zeros(RADINT_ACTIVE_LEN)
    );
    assert!(result.output_normalization.spectrum_norm.is_finite());
    Ok(())
}

#[test]
fn xsph_xsect_bcoef_standard_energy_row_applies_spin_orbit_retry() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let diagonal_weights = Array1::from(vec![Complex::new(-1.0 / 3.0, 0.0); 8]);
    let orbital_l = arr1(&[1, 1, 0, 2, 3, 4, 5, 6]);
    let mut trace_weights = Array2::<Complex>::zeros((8, 8));
    trace_weights[(0, 1)] = Complex::new(0.6, -0.1);
    trace_weights[(1, 0)] = Complex::new(-0.2, 0.3);
    let transitions = vec![
        XsphXsectTransition {
            multipole: XsphTransitionMultipole::ElectricDipole,
            transition_delta: -1,
            transition_index_1based: 1,
            final_kappa: 1,
            final_l: 1,
            multipole_order: 1,
        },
        XsphXsectTransition {
            multipole: XsphTransitionMultipole::ElectricDipole,
            transition_delta: 0,
            transition_index_1based: 2,
            final_kappa: 1,
            final_l: 1,
            multipole_order: 1,
        },
    ];
    let regular_channels = vec![
        xsect_test_regular_channel(&fixture, Complex::new(0.2, 0.05)),
        xsect_test_regular_channel(&fixture, Complex::new(-0.1, 0.08)),
    ];
    let irregular_channels = vec![
        xsect_test_irregular_channel(&fixture),
        xsect_test_irregular_channel(&fixture),
    ];
    let retry_regular_channels = vec![
        xsect_test_regular_channel(&fixture, Complex::new(0.3, -0.04)),
        xsect_test_regular_channel(&fixture, Complex::new(-0.25, 0.12)),
    ];
    let retry_irregular_channels = vec![
        xsect_test_irregular_channel(&fixture),
        xsect_test_irregular_channel(&fixture),
    ];
    let fscf = arr1(&[
        Complex::new(1.0, 0.25),
        Complex::new(0.8, -0.15),
        Complex::new(1.1, 0.35),
        Complex::new(0.6, 0.05),
        Complex::new(0.9, -0.2),
        Complex::new(1.2, 0.4),
        Complex::new(0.7, 0.1),
    ]);
    let photon_wave_number = 0.37;
    let screened_field_scale = 0.8;
    let photon_energy = 1.3;
    let wave_number = Complex::new(0.4, 0.5);

    let base = xsph_xsect_bcoef_standard_energy_row(XsphXsectBcoefStandardEnergyRowInput {
        transitions: &transitions,
        regular_channels: &regular_channels,
        irregular_channels: &irregular_channels,
        selected_higher_multipole: None,
        initial_kappa: -1,
        initial_large: fixture.initial_large.view(),
        initial_small: fixture.initial_small.view(),
        xray_bessel: fixture.bessel.view(),
        radii: fixture.radii.view(),
        log_step: 0.137,
        photon_wave_number,
        screened_field_scale,
        fscf: fscf.view(),
        diagonal_weights: diagonal_weights.view(),
        spin_polarized_cross_terms: false,
        orbital_l: orbital_l.view(),
        trace_weights: trace_weights.view(),
        spin_orbit_removed_regular_channels: None,
        spin_orbit_removed_irregular_channels: None,
        photon_energy,
        wave_number,
        active_channel_count: transitions.len(),
    })?;
    let retry = xsph_xsect_bcoef_standard_energy_row(XsphXsectBcoefStandardEnergyRowInput {
        transitions: &transitions,
        regular_channels: &regular_channels,
        irregular_channels: &irregular_channels,
        selected_higher_multipole: None,
        initial_kappa: -1,
        initial_large: fixture.initial_large.view(),
        initial_small: fixture.initial_small.view(),
        xray_bessel: fixture.bessel.view(),
        radii: fixture.radii.view(),
        log_step: 0.137,
        photon_wave_number,
        screened_field_scale,
        fscf: fscf.view(),
        diagonal_weights: diagonal_weights.view(),
        spin_polarized_cross_terms: true,
        orbital_l: orbital_l.view(),
        trace_weights: trace_weights.view(),
        spin_orbit_removed_regular_channels: Some(&retry_regular_channels),
        spin_orbit_removed_irregular_channels: Some(&retry_irregular_channels),
        photon_energy,
        wave_number,
        active_channel_count: transitions.len(),
    })?;

    assert_eq!(retry.cross_term_updates.len(), 1);
    let update = &retry.cross_term_updates[0];
    assert_eq!(update.partner_index_1based, 1);
    assert_complex_close(update.angular_coupling, Complex::new(-0.2, -0.1));
    assert!(update.cross_section_increment.norm() > 0.0);
    assert_complex_close(
        retry.unnormalized_cross_section,
        base.unnormalized_cross_section + update.cross_section_increment,
    );
    assert_complex_close(
        retry.output_normalization.cross_section,
        xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
            photon_energy,
            wave_number,
            spectrum_norm: retry.unnormalized_spectrum_norm,
            cross_section: retry.unnormalized_cross_section,
            reduced_matrix_elements: retry.unnormalized_reduced_matrix_elements.view(),
            phase_shifts: retry.phase_shifts.view(),
            active_channel_count: transitions.len(),
        })?
        .cross_section,
    );

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_ordinary_row_rejects_invalid_central_integral() {
    let weights = Array1::from(vec![Complex::new(-1.0 / 3.0, 0.0); 8]);
    let reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let phase_shifts = Array1::<Complex>::zeros(8);

    let error = xsph_xsect_bcoef_ordinary_row(XsphXsectBcoefOrdinaryRowInput {
        multipole: XsphTransitionMultipole::ElectricDipole,
        selected_higher_multipole: None,
        transition_index_1based: 1,
        diagonal_weights: weights.view(),
        reduced_matrix_integral: Complex::new(0.3, -0.4),
        central_cross_integral: Complex::new(Real::NAN, 0.0),
        phase_shift: Complex::new(0.2, 0.05),
        spectrum_norm: 1.5,
        cross_section: Complex::new(0.2, -0.1),
        reduced_matrix_elements: reduced_matrix_elements.view(),
        phase_shifts: phase_shifts.view(),
    })
    .expect_err("ordinary-row central cross-section integral must be finite");

    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_central_cross_integral",
            ..
        }
    ));
}

#[test]
fn xsph_xsect_central_cross_section_preserves_feff_ic3_skip() -> Result<(), XsphError> {
    let poison = Complex::new(Real::NAN, Real::NAN);
    let result = xsph_xsect_central_cross_section(XsphXsectCentralCrossSectionInput {
        spin_orbit_removed_pass: true,
        radial_integral: poison,
        angular_weight: poison,
        cross_section: poison,
    })?;
    assert_eq!(result, None);

    let empty_weights = Array1::<Complex>::zeros(0);
    let result = xsph_xsect_bcoef_central_cross_section(XsphXsectBcoefCentralCrossSectionInput {
        spin_orbit_removed_pass: true,
        transition_index_1based: 0,
        diagonal_weights: empty_weights.view(),
        radial_integral: poison,
        cross_section: poison,
    })?;
    assert_eq!(result, None);

    Ok(())
}

#[test]
fn xsph_xsect_central_cross_section_rejects_invalid_inputs() {
    let error = xsph_xsect_central_cross_section(XsphXsectCentralCrossSectionInput {
        spin_orbit_removed_pass: false,
        radial_integral: Complex::new(Real::NAN, 0.0),
        angular_weight: Complex::new(0.6, 0.2),
        cross_section: Complex::new(0.2, -0.1),
    })
    .expect_err("active central xsec integral must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_central_cross_integral",
            ..
        }
    ));

    let short_weights = Array1::<Complex>::zeros(7);
    let error = xsph_xsect_bcoef_central_cross_section(XsphXsectBcoefCentralCrossSectionInput {
        spin_orbit_removed_pass: false,
        transition_index_1based: 1,
        diagonal_weights: short_weights.view(),
        radial_integral: Complex::new(0.3, -0.4),
        cross_section: Complex::new(0.2, -0.1),
    })
    .expect_err("source-backed bcoef diagonal table has eight slots");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "xsect_bcoef_diagonal_weights",
            required: 8,
            actual: 7
        }
    ));
}

fn xsect_bcoef_average_input(spin: i32, spin_channels: usize) -> XsphXsectBcoefWeightsInput {
    XsphXsectBcoefWeightsInput {
        max_angular_momentum: 4,
        initial_kappa: -1,
        polarization: 0,
        polarization_tensor: [[Complex::new(0.0, 0.0); 3]; 3],
        higher_multipole_selector: 2,
        spin,
        spin_channels,
        spin_vector_angle: 0.0,
    }
}

#[test]
fn xsph_xsect_cross_term_plan_matches_feff_iold_logic() -> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1, 0, 2, 2]);

    let first = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
        spin_polarized: true,
        spin_orbit_removed_pass: false,
        transition_index_1based: 1,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
    })?
    .expect("FEFF sets iold=1 for the first row of an adjacent same-l pair");
    assert_eq!(first.iold, 1);
    assert_eq!(first.mode, XsphXsectCrossTermMode::SaveCurrentForNext);
    assert_eq!(first.partner_index_1based, 2);

    let second = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
        spin_polarized: true,
        spin_orbit_removed_pass: false,
        transition_index_1based: 2,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
    })?
    .expect("FEFF sets iold=2 when the previous row has the same l");
    assert_eq!(second.iold, 2);
    assert_eq!(second.mode, XsphXsectCrossTermMode::UsePreviousForCurrent);
    assert_eq!(second.partner_index_1based, 1);

    let later_pair = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
        spin_polarized: true,
        spin_orbit_removed_pass: false,
        transition_index_1based: 5,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
    })?
    .expect("FEFF later same-l rows reuse the previous row");
    assert_eq!(later_pair.iold, 2);
    assert_eq!(later_pair.partner_index_1based, 4);

    Ok(())
}

#[test]
fn xsph_xsect_cross_term_plan_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1, 0, 2, 2]);

    for (spin_polarized, spin_orbit_removed_pass, transition_index_1based) in [
        (false, false, 1),
        (true, true, 1),
        (true, false, 3),
        (true, false, 4),
    ] {
        let plan = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
            spin_polarized,
            spin_orbit_removed_pass,
            transition_index_1based,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
        })?;
        assert_eq!(plan, None);
    }

    let error = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
        spin_polarized: true,
        spin_orbit_removed_pass: false,
        transition_index_1based: 0,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
    })
    .expect_err("transition index is one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: 0,
            active_len: 5
        }
    ));

    Ok(())
}

#[test]
fn xsph_xsect_cross_term_state_save_and_reuse_build_radint_branches() -> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1]);
    let save_plan = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
        spin_polarized: true,
        spin_orbit_removed_pass: false,
        transition_index_1based: 1,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
    })?
    .expect("first same-l retry stores FEFF xrcold/xncold");

    let stored_fixture = radint_fixture();
    let stored_regular = xsph_radial_integral(stored_fixture.input(
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?
    .coupling;
    let stored_irregular = xsph_radial_cross_integral(stored_fixture.cross_input(
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?
    .irregular_coupling;

    let state = xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
        plan: save_plan,
        transition_index_1based: 1,
        radial_integral: Complex::new(0.2, 0.1),
        phase_shift: Complex::new(0.1, 0.05),
        regular_coupling: stored_regular.view(),
        irregular_coupling: stored_irregular.view(),
        active_len: RADINT_ACTIVE_LEN,
    })?
    .expect("iold=1 saves FEFF cross-term state");
    assert_eq!(state.transition_index_1based, 1);
    assert_eq!(state.partner_index_1based, 2);
    assert_complex_close(state.radial_integral, Complex::new(0.2, 0.1));
    assert_complex_close(state.phase_shift, Complex::new(0.1, 0.05));
    assert_eq!(state.regular_coupling.len(), RADINT_ACTIVE_LEN);
    assert_eq!(state.irregular_coupling.len(), RADINT_ACTIVE_LEN);

    let reuse_plan = xsph_xsect_cross_term_plan(XsphXsectCrossTermPlanInput {
        spin_polarized: true,
        spin_orbit_removed_pass: false,
        transition_index_1based: 2,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
    })?
    .expect("second same-l retry reuses FEFF xrcold/xncold");
    let reuse = xsph_xsect_cross_term_state_reuse(XsphXsectCrossTermStateReuseInput {
        plan: reuse_plan,
        transition_index_1based: 2,
        state: &state,
    })?
    .expect("iold=2 exposes FEFF radint cross-term branches");

    assert_eq!(reuse.saved_transition_index_1based, 1);
    assert_complex_close(reuse.saved_radial_integral, Complex::new(0.2, 0.1));
    assert_complex_close(reuse.saved_phase_shift, Complex::new(0.1, 0.05));

    let XsphXsectCrossTermStateReuse {
        radint3_branch,
        radint4_branch,
        ..
    } = reuse;
    let current_fixture = radint_fixture_solution_set(2);
    let stored_regular_result = xsph_radial_cross_integral(current_fixture.cross_input(
        radint3_branch,
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?;
    assert_complex_close(
        stored_regular_result.value,
        Complex::new(3.178_514_079_941_580_5e-6, -4.005_466_192_553_954_6e-7),
    );

    let stored_irregular_result = xsph_radial_cross_integral(current_fixture.cross_input(
        radint4_branch,
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?;
    assert_complex_close(
        stored_irregular_result.value,
        Complex::new(2.399_554_659_493_643e-6, -5.997_356_168_103_543e-7),
    );

    Ok(())
}

#[test]
fn xsph_xsect_cross_term_state_helpers_preserve_inactive_modes() -> Result<(), XsphError> {
    let poison = Complex::new(Real::NAN, Real::NAN);
    let empty = Array1::<Complex>::zeros(0);
    let use_plan = XsphXsectCrossTermPlan {
        iold: 2,
        mode: XsphXsectCrossTermMode::UsePreviousForCurrent,
        partner_index_1based: 1,
    };

    let saved = xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
        plan: use_plan,
        transition_index_1based: 2,
        radial_integral: poison,
        phase_shift: poison,
        regular_coupling: empty.view(),
        irregular_coupling: empty.view(),
        active_len: 0,
    })?;
    assert_eq!(saved, None);

    let save_plan = XsphXsectCrossTermPlan {
        iold: 1,
        mode: XsphXsectCrossTermMode::SaveCurrentForNext,
        partner_index_1based: 2,
    };
    let invalid_state = XsphXsectCrossTermState {
        transition_index_1based: 0,
        partner_index_1based: 0,
        radial_integral: poison,
        phase_shift: poison,
        regular_coupling: empty.clone(),
        irregular_coupling: empty,
    };
    let reused = xsph_xsect_cross_term_state_reuse(XsphXsectCrossTermStateReuseInput {
        plan: save_plan,
        transition_index_1based: 1,
        state: &invalid_state,
    })?;
    assert!(reused.is_none());

    Ok(())
}

#[test]
fn xsph_xsect_cross_term_state_helpers_reject_invalid_state() {
    let regular = arr1(&[Complex::new(0.1, 0.2), Complex::new(0.3, 0.4)]);
    let short_irregular = arr1(&[Complex::new(0.5, 0.6)]);
    let save_plan = XsphXsectCrossTermPlan {
        iold: 1,
        mode: XsphXsectCrossTermMode::SaveCurrentForNext,
        partner_index_1based: 2,
    };

    let error = xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
        plan: save_plan,
        transition_index_1based: 1,
        radial_integral: Complex::new(0.2, 0.1),
        phase_shift: Complex::new(0.1, 0.05),
        regular_coupling: regular.view(),
        irregular_coupling: short_irregular.view(),
        active_len: 2,
    })
    .expect_err("FEFF xncold must cover ilast");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "xsect_cross_term_irregular_coupling",
            required: 2,
            actual: 1
        }
    ));

    let bad_iold = XsphXsectCrossTermPlan {
        iold: 0,
        mode: XsphXsectCrossTermMode::SaveCurrentForNext,
        partner_index_1based: 2,
    };
    let error = xsph_xsect_cross_term_state_save(XsphXsectCrossTermStateSaveInput {
        plan: bad_iold,
        transition_index_1based: 1,
        radial_integral: Complex::new(0.2, 0.1),
        phase_shift: Complex::new(0.1, 0.05),
        regular_coupling: regular.view(),
        irregular_coupling: regular.view(),
        active_len: 2,
    })
    .expect_err("typed save state requires FEFF iold=1");
    assert!(matches!(
        error,
        XsphError::IntegerOutOfRange {
            name: "xsect_cross_term_iold",
            value: 0
        }
    ));

    let mismatched_state = XsphXsectCrossTermState {
        transition_index_1based: 1,
        partner_index_1based: 3,
        radial_integral: Complex::new(0.2, 0.1),
        phase_shift: Complex::new(0.1, 0.05),
        regular_coupling: regular.clone(),
        irregular_coupling: regular,
    };
    let reuse_plan = XsphXsectCrossTermPlan {
        iold: 2,
        mode: XsphXsectCrossTermMode::UsePreviousForCurrent,
        partner_index_1based: 1,
    };
    let error = xsph_xsect_cross_term_state_reuse(XsphXsectCrossTermStateReuseInput {
        plan: reuse_plan,
        transition_index_1based: 2,
        state: &mismatched_state,
    })
    .expect_err("saved state must target the current transition");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "xsect_cross_term_state_partner",
            index_1based: 3,
            active_len: 2
        }
    ));
}

#[test]
fn xsph_xsect_bcoef_cross_term_state_accumulation_uses_saved_state() -> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1, 0]);
    let state = XsphXsectCrossTermState {
        transition_index_1based: 1,
        partner_index_1based: 2,
        radial_integral: Complex::new(0.2, 0.1),
        phase_shift: Complex::new(0.1, 0.05),
        regular_coupling: arr1(&[Complex::new(0.1, 0.0)]),
        irregular_coupling: arr1(&[Complex::new(0.2, 0.0)]),
    };
    let reuse_plan = XsphXsectCrossTermPlan {
        iold: 2,
        mode: XsphXsectCrossTermMode::UsePreviousForCurrent,
        partner_index_1based: 1,
    };
    let state_reuse = xsph_xsect_cross_term_state_reuse(XsphXsectCrossTermStateReuseInput {
        plan: reuse_plan,
        transition_index_1based: 2,
        state: &state,
    })?
    .expect("iold=2 exposes saved rkk1/phold state");

    let mut trace_weights = Array2::<Complex>::zeros((8, 8));
    trace_weights[(0, 1)] = Complex::new(0.6, -0.1);
    trace_weights[(1, 0)] = Complex::new(-0.2, 0.3);
    let result = xsph_xsect_bcoef_cross_term_state_accumulation(
        XsphXsectBcoefCrossTermStateAccumulationInput {
            transition_index_1based: 2,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            trace_weights: trace_weights.view(),
            state_reuse: &state_reuse,
            current_radial_integral: Complex::new(-0.3, 0.4),
            current_phase_shift: Complex::new(0.5, -0.2),
            radint3_integral: Complex::new(0.05, 0.07),
            radint4_integral: Complex::new(-0.02, 0.04),
            cross_section: Complex::new(1.0, -0.5),
        },
    )?
    .expect("FEFF updates the previous same-l neighbor cross term");

    assert_eq!(result.partner_index_1based, 1);
    assert_complex_close(result.angular_coupling, Complex::new(-0.2, -0.1));
    assert_complex_close(
        result.cross_section_increment,
        Complex::new(0.009_465_781_632_484_065, -0.063_314_828_230_396_86),
    );
    assert_complex_close(
        result.cross_section,
        Complex::new(1.009_465_781_632_484_1, -0.563_314_828_230_396_8),
    );

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_cross_term_state_accumulation_preserves_skip_conditions()
-> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1]);
    let trace_weights = Array2::<Complex>::zeros((8, 8));
    let empty = Array1::<Complex>::zeros(0);
    let poison = Complex::new(Real::NAN, Real::NAN);
    let state_reuse = XsphXsectCrossTermStateReuse {
        saved_transition_index_1based: 99,
        saved_radial_integral: poison,
        saved_phase_shift: poison,
        radint3_branch: XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling: empty.view(),
        },
        radint4_branch: XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling: empty.view(),
        },
    };

    let result = xsph_xsect_bcoef_cross_term_state_accumulation(
        XsphXsectBcoefCrossTermStateAccumulationInput {
            transition_index_1based: 1,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            trace_weights: trace_weights.view(),
            state_reuse: &state_reuse,
            current_radial_integral: poison,
            current_phase_shift: poison,
            radint3_integral: poison,
            radint4_integral: poison,
            cross_section: poison,
        },
    )?;
    assert_eq!(result, None);

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_cross_term_state_accumulation_rejects_mismatched_state() {
    let orbital_l = arr1(&[1, 1]);
    let trace_weights = Array2::<Complex>::zeros((8, 8));
    let empty = Array1::<Complex>::zeros(1);
    let state_reuse = XsphXsectCrossTermStateReuse {
        saved_transition_index_1based: 2,
        saved_radial_integral: Complex::new(0.2, 0.1),
        saved_phase_shift: Complex::new(0.1, 0.05),
        radint3_branch: XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling: empty.view(),
        },
        radint4_branch: XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling: empty.view(),
        },
    };

    let error = xsph_xsect_bcoef_cross_term_state_accumulation(
        XsphXsectBcoefCrossTermStateAccumulationInput {
            transition_index_1based: 2,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            trace_weights: trace_weights.view(),
            state_reuse: &state_reuse,
            current_radial_integral: Complex::new(-0.3, 0.4),
            current_phase_shift: Complex::new(0.5, -0.2),
            radint3_integral: Complex::new(0.05, 0.07),
            radint4_integral: Complex::new(-0.02, 0.04),
            cross_section: Complex::new(1.0, -0.5),
        },
    )
    .expect_err("saved state must be from FEFF k1 = ind - 1");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "xsect_cross_term_saved_transition",
            index_1based: 2,
            active_len: 1
        }
    ));
}

#[test]
fn xsph_xsect_cross_term_accumulation_matches_feff_xmcd_block() -> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1, 0]);
    let result = xsph_xsect_cross_term_accumulation(XsphXsectCrossTermAccumulationInput {
        transition_index_1based: 2,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
        saved_radial_integral: Complex::new(0.2, 0.1),
        current_radial_integral: Complex::new(-0.3, 0.4),
        saved_phase_shift: Complex::new(0.1, 0.05),
        current_phase_shift: Complex::new(0.5, -0.2),
        partner_current_weight: Complex::new(0.6, -0.1),
        current_partner_weight: Complex::new(-0.2, 0.3),
        radint3_integral: Complex::new(0.05, 0.07),
        radint4_integral: Complex::new(-0.02, 0.04),
        cross_section: Complex::new(1.0, -0.5),
    })?
    .expect("FEFF updates the previous same-l neighbor cross term");

    assert_eq!(result.partner_index_1based, 1);
    assert_complex_close(
        result.phase_factor,
        Complex::new(1.182_665_726_619_379_9, 0.500_023_049_248_714_5),
    );
    assert_complex_close(
        result.inverse_phase_factor,
        Complex::new(0.717_323_023_385_973_4, -0.303_279_309_932_345_4),
    );
    assert_complex_close(result.angular_coupling, Complex::new(-0.2, -0.1));
    assert_complex_close(
        result.matrix_cross_term_increment,
        Complex::new(0.004_918_593_482_909_222, -0.047_499_718_750_133_84),
    );
    assert_complex_close(
        result.radint3_increment,
        Complex::new(-0.007_914_275_958_872_483, -0.012_719_299_514_536_458),
    );
    assert_complex_close(
        result.radint4_increment,
        Complex::new(0.012_461_464_108_447_326, -0.003_095_809_965_726_562_7),
    );
    assert_complex_close(
        result.cross_section_increment,
        Complex::new(0.009_465_781_632_484_065, -0.063_314_828_230_396_86),
    );
    assert_complex_close(
        result.cross_section,
        Complex::new(1.009_465_781_632_484_1, -0.563_314_828_230_396_8),
    );

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_cross_term_accumulation_uses_traced_weights() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;
    let result =
        xsph_xsect_bcoef_cross_term_accumulation(XsphXsectBcoefCrossTermAccumulationInput {
            transition_index_1based: 2,
            orbital_l: weights.orbital_l.view(),
            active_len: weights.orbital_l.len(),
            trace_weights: weights.trace_weights.view(),
            saved_radial_integral: Complex::new(0.2, 0.1),
            current_radial_integral: Complex::new(-0.3, 0.4),
            saved_phase_shift: Complex::new(0.1, 0.05),
            current_phase_shift: Complex::new(0.5, -0.2),
            radint3_integral: Complex::new(0.05, 0.07),
            radint4_integral: Complex::new(-0.02, 0.04),
            cross_section: Complex::new(1.0, -0.5),
        })?
        .expect("slots 1 and 2 form FEFF's adjacent same-l pair");

    assert_eq!(result.partner_index_1based, 1);
    assert_complex_close(result.angular_coupling, Complex::new(0.0, 0.0));
    assert_complex_close(result.matrix_cross_term_increment, Complex::new(0.0, 0.0));
    assert_complex_close(result.radint3_increment, Complex::new(0.0, 0.0));
    assert_complex_close(result.radint4_increment, Complex::new(0.0, 0.0));
    assert_complex_close(result.cross_section_increment, Complex::new(0.0, 0.0));
    assert_complex_close(result.cross_section, Complex::new(1.0, -0.5));

    Ok(())
}

#[test]
fn xsph_xsect_bcoef_cross_term_accumulation_preserves_feff_weight_order() -> Result<(), XsphError> {
    let orbital_l = arr1(&[1, 1, 0]);
    let mut trace_weights = Array2::<Complex>::zeros((8, 8));
    trace_weights[(0, 1)] = Complex::new(0.6, -0.1);
    trace_weights[(1, 0)] = Complex::new(-0.2, 0.3);

    let result =
        xsph_xsect_bcoef_cross_term_accumulation(XsphXsectBcoefCrossTermAccumulationInput {
            transition_index_1based: 2,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            trace_weights: trace_weights.view(),
            saved_radial_integral: Complex::new(0.2, 0.1),
            current_radial_integral: Complex::new(-0.3, 0.4),
            saved_phase_shift: Complex::new(0.1, 0.05),
            current_phase_shift: Complex::new(0.5, -0.2),
            radint3_integral: Complex::new(0.05, 0.07),
            radint4_integral: Complex::new(-0.02, 0.04),
            cross_section: Complex::new(1.0, -0.5),
        })?
        .expect("FEFF updates the previous same-l neighbor cross term");

    assert_complex_close(result.angular_coupling, Complex::new(-0.2, -0.1));
    assert_complex_close(
        result.cross_section_increment,
        Complex::new(0.009_465_781_632_484_065, -0.063_314_828_230_396_86),
    );
    assert_complex_close(
        result.cross_section,
        Complex::new(1.009_465_781_632_484_1, -0.563_314_828_230_396_8),
    );

    Ok(())
}

#[test]
fn xsph_xsect_cross_term_accumulation_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let poison = Complex::new(Real::NAN, Real::NAN);
    for (transition_index_1based, orbital_l) in [
        (1, arr1(&[1, 1, 0])),
        (2, arr1(&[1, 2, 0])),
        (2, arr1(&[0, 0, 1])),
    ] {
        let result = xsph_xsect_cross_term_accumulation(XsphXsectCrossTermAccumulationInput {
            transition_index_1based,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            saved_radial_integral: poison,
            current_radial_integral: poison,
            saved_phase_shift: poison,
            current_phase_shift: poison,
            partner_current_weight: poison,
            current_partner_weight: poison,
            radint3_integral: poison,
            radint4_integral: poison,
            cross_section: poison,
        })?;
        assert_eq!(result, None);
    }

    Ok(())
}

#[test]
fn xsph_xsect_cross_term_accumulation_rejects_invalid_inputs() {
    let orbital_l = arr1(&[1, 1]);
    let error = xsph_xsect_cross_term_accumulation(XsphXsectCrossTermAccumulationInput {
        transition_index_1based: 0,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
        saved_radial_integral: Complex::new(0.2, 0.1),
        current_radial_integral: Complex::new(-0.3, 0.4),
        saved_phase_shift: Complex::new(0.1, 0.05),
        current_phase_shift: Complex::new(0.5, -0.2),
        partner_current_weight: Complex::new(0.6, -0.1),
        current_partner_weight: Complex::new(-0.2, 0.3),
        radint3_integral: Complex::new(0.05, 0.07),
        radint4_integral: Complex::new(-0.02, 0.04),
        cross_section: Complex::new(1.0, -0.5),
    })
    .expect_err("transition index is one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: 0,
            active_len: 2
        }
    ));

    let error = xsph_xsect_cross_term_accumulation(XsphXsectCrossTermAccumulationInput {
        transition_index_1based: 2,
        orbital_l: orbital_l.view(),
        active_len: orbital_l.len(),
        saved_radial_integral: Complex::new(Real::NAN, 0.1),
        current_radial_integral: Complex::new(-0.3, 0.4),
        saved_phase_shift: Complex::new(0.1, 0.05),
        current_phase_shift: Complex::new(0.5, -0.2),
        partner_current_weight: Complex::new(0.6, -0.1),
        current_partner_weight: Complex::new(-0.2, 0.3),
        radint3_integral: Complex::new(0.05, 0.07),
        radint4_integral: Complex::new(-0.02, 0.04),
        cross_section: Complex::new(1.0, -0.5),
    })
    .expect_err("active cross-term inputs must be finite");
    assert!(matches!(
        error,
        XsphError::NonFiniteComplex {
            name: "xsect_cross_term_saved_integral",
            ..
        }
    ));
}

#[test]
fn xsph_xsect_bcoef_cross_term_accumulation_rejects_invalid_inputs() {
    let orbital_l = arr1(&[1, 1]);
    let trace_weights = Array2::<Complex>::zeros((8, 8));
    let error =
        xsph_xsect_bcoef_cross_term_accumulation(XsphXsectBcoefCrossTermAccumulationInput {
            transition_index_1based: 0,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            trace_weights: trace_weights.view(),
            saved_radial_integral: Complex::new(0.2, 0.1),
            current_radial_integral: Complex::new(-0.3, 0.4),
            saved_phase_shift: Complex::new(0.1, 0.05),
            current_phase_shift: Complex::new(0.5, -0.2),
            radint3_integral: Complex::new(0.05, 0.07),
            radint4_integral: Complex::new(-0.02, 0.04),
            cross_section: Complex::new(1.0, -0.5),
        })
        .expect_err("transition index is one-based");
    assert!(matches!(
        error,
        XsphError::InvalidOneBasedIndex {
            name: "transition_index",
            index_1based: 0,
            active_len: 2
        }
    ));

    let short_trace_weights = Array2::<Complex>::zeros((7, 8));
    let error =
        xsph_xsect_bcoef_cross_term_accumulation(XsphXsectBcoefCrossTermAccumulationInput {
            transition_index_1based: 2,
            orbital_l: orbital_l.view(),
            active_len: orbital_l.len(),
            trace_weights: short_trace_weights.view(),
            saved_radial_integral: Complex::new(0.2, 0.1),
            current_radial_integral: Complex::new(-0.3, 0.4),
            saved_phase_shift: Complex::new(0.1, 0.05),
            current_phase_shift: Complex::new(0.5, -0.2),
            radint3_integral: Complex::new(0.05, 0.07),
            radint4_integral: Complex::new(-0.02, 0.04),
            cross_section: Complex::new(1.0, -0.5),
        })
        .expect_err("source-backed bcoef trace table is 8x8");
    assert!(matches!(
        error,
        XsphError::MatrixTooSmall {
            name: "xsect_bcoef_trace_weights",
            required: [8, 8],
            actual: [7, 8]
        }
    ));
}

#[test]
fn xsph_xsect_radial_helpers_reject_invalid_inputs() {
    let regular_large = arr1(&[Complex::new(1.0, 0.0)]);
    let regular_small = arr1(&[Complex::new(1.0, 0.0)]);

    let error = xsph_xsect_regular_solution(XsphXsectRegularSolutionInput {
        wave_number: Complex::new(0.4, 0.5),
        phase_amplitude: Complex::new(0.0, 0.0),
        final_kappa: -2,
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        active_len: 1,
    })
    .expect_err("xfnorm cannot divide by a zero phase amplitude");
    assert!(matches!(error, XsphError::ZeroPhaseAmplitude));

    let error = xsph_xsect_irregular_initial_condition(XsphXsectIrregularInitialConditionInput {
        muffin_tin_radius: 0.0,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        final_kappa: 3,
        bessel_j_l: Complex::new(0.8, 0.1),
        neumann_l: Complex::new(-0.3, 0.05),
        bessel_j_l_plus_1: Complex::new(0.25, -0.03),
        neumann_l_plus_1: Complex::new(-0.6, 0.2),
    })
    .expect_err("muffin-tin radius must be positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: 0.0
        }
    ));

    let error = xsph_xsect_irregular_transform(XsphXsectIrregularTransformInput {
        phase_shift: Complex::new(0.2, -0.1),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: regular_large.view(),
        irregular_small: regular_small.view(),
        active_len: 2,
    })
    .expect_err("all active radial rows must be present");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "regular_large",
            required: 2,
            actual: 1
        }
    ));

    let error = xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
        photon_energy: -1.0,
        wave_number: Complex::new(0.4, 0.5),
        spectrum_norm: 0.75,
        cross_section: Complex::new(0.2, -0.3),
        reduced_matrix_elements: regular_large.view(),
        phase_shifts: regular_small.view(),
        active_channel_count: 1,
    })
    .expect_err("positive-omega FEFF branch requires a positive photon energy");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "photon_energy",
            value: -1.0
        }
    ));

    let error = xsph_xsect_output_normalization(XsphXsectOutputNormalizationInput {
        photon_energy: 1.3,
        wave_number: Complex::new(0.4, 0.5),
        spectrum_norm: 0.0,
        cross_section: Complex::new(0.2, -0.3),
        reduced_matrix_elements: regular_large.view(),
        phase_shifts: regular_small.view(),
        active_channel_count: 1,
    })
    .expect_err("normalized xsnorm must be positive before reduced-matrix scaling");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "xsect_scaled_spectrum_norm",
            value: 0.0
        }
    ));

    let fixture = xsect_density_fixture();
    let error = xsph_xsect_embedded_density(XsphXsectEmbeddedDensityInput {
        integration_len: fixture.radii.len() + 1,
        ..fixture.embedded_input()
    })
    .expect_err("the integration prefix cannot exceed the active radial prefix");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "active_len",
            required: 7,
            actual: 6
        }
    ));

    let zeros = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let error = xsph_xsect_projected_density(XsphXsectProjectedDensityInput {
        atomic_large: zeros.view(),
        atomic_small: zeros.view(),
        ..fixture.projected_input()
    })
    .expect_err("projector normalization must be positive");
    assert!(matches!(
        error,
        XsphError::InvalidPositiveScalar {
            name: "xsect_projected_atomic_norm",
            value: 0.0
        }
    ));
}

#[test]
fn xsph_jas_orthogonality_correction_matches_feff_reference() -> Result<(), XsphError> {
    let fixture = jas_correction_fixture();
    let result = xsph_jas_orthogonality_correction(fixture.input())?;

    assert_eq!(result.corrections.shape(), &[5, 2]);
    assert_eq!(result.corrections.strides(), &[1, 5]);
    let expected = arr2(&[
        [
            Complex::new(8.532_814_982_278_092e-1, 0.0),
            Complex::new(5.383_065_623_201_823e-1, 0.0),
        ],
        [
            Complex::new(6.253_990_661_980_002e-1, 0.0),
            Complex::new(-1.548_198_156_243_805_7e-1, 0.0),
        ],
        [
            Complex::new(3.052_006_063_171_550_7e-1, 0.0),
            Complex::new(-5.104_841_082_979_479e-1, 0.0),
        ],
        [
            Complex::new(-1.292_008_855_802_578e-2, 0.0),
            Complex::new(-2.267_415_597_527_204e-1, 0.0),
        ],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ]);
    for ((angular_l, q_index), &expected_value) in expected.indexed_iter() {
        assert_complex_close(result.corrections[(angular_l, q_index)], expected_value);
    }
    assert!(result.normalization.norm() > 0.0);
    Ok(())
}

#[test]
fn xsph_jas_orthogonality_correction_rejects_invalid_inputs() {
    let fixture = jas_correction_fixture();
    let short_q_bessel = Array3::<Real>::zeros(
        (
            JAS_CORRECTION_ACTIVE_LEN - 1,
            JAS_CORRECTION_LJMAX + 1,
            JAS_CORRECTION_Q_COUNT,
        )
            .f(),
    );
    let error = xsph_jas_orthogonality_correction(XsphJasOrthogonalityCorrectionInput {
        q_bessel: short_q_bessel.view(),
        ..fixture.input()
    })
    .expect_err("q-Bessel table must cover the active radial prefix");
    assert!(matches!(
        error,
        XsphError::ShapeTooSmall {
            name: "q_bessel",
            required: [JAS_CORRECTION_ACTIVE_LEN, 5, 1],
            actual: [6, 5, 2],
        }
    ));

    let zero_q_bessel =
        Array3::<Real>::zeros((JAS_CORRECTION_ACTIVE_LEN, JAS_CORRECTION_LJMAX + 1, 0).f());
    let error = xsph_jas_orthogonality_correction(XsphJasOrthogonalityCorrectionInput {
        q_bessel: zero_q_bessel.view(),
        ..fixture.input()
    })
    .expect_err("at least one q-vector table is required");
    assert!(matches!(
        error,
        XsphError::ShapeTooSmall {
            name: "q_bessel",
            required: [JAS_CORRECTION_ACTIVE_LEN, 5, 1],
            actual: [JAS_CORRECTION_ACTIVE_LEN, 5, 0],
        }
    ));

    let zeros = Array1::<Real>::zeros(JAS_CORRECTION_TOTAL_LEN);
    let error = xsph_jas_orthogonality_correction(XsphJasOrthogonalityCorrectionInput {
        large_component: zeros.view(),
        small_component: zeros.view(),
        ..fixture.input()
    })
    .expect_err("zero spinors cannot produce a normalization denominator");
    assert_eq!(error, XsphError::ZeroJasOrthogonalityNormalization);
}

#[test]
fn xsph_jas_overlap_matches_feff_getorthg_reference() -> Result<(), XsphError> {
    let fixture = jas_radial_fixture();

    let same = xsph_jas_overlap(fixture.same_overlap_input(0, 0, 0))?;
    assert_complex_close(
        same.large_overlap,
        Complex::new(3.111_103_137_426_832_5e-2, -1.939_327_586_493_520_9e-3),
    );
    assert_complex_close(
        same.small_overlap,
        Complex::new(5.470_577_673_537_446e-4, -2.750_209_476_779_58e-4),
    );
    assert_complex_close(same.total_overlap, same.large_overlap + same.small_overlap);
    assert_close(same.near_origin_power, 1.0);

    let different = xsph_jas_overlap(fixture.different_overlap_input(0, 2, 0))?;
    assert_complex_close(
        different.large_overlap,
        Complex::new(2.486_260_786_107_731e-2, -5.101_315_014_281_199e-4),
    );
    assert_complex_close(
        different.small_overlap,
        Complex::new(5.295_956_881_896_171e-5, 1.068_650_088_400_366_3e-4),
    );
    assert_close(different.near_origin_power, 3.0);

    let powered = xsph_jas_overlap(fixture.different_overlap_input(1, 2, 2))?;
    assert_complex_close(
        powered.large_overlap,
        Complex::new(1.975_368_971_994_53e-3, -8.755_367_812_898_518e-5),
    );
    assert_complex_close(
        powered.small_overlap,
        Complex::new(-3.904_513_356_471_819e-6, 2.062_914_036_995_52e-6),
    );
    assert_close(powered.near_origin_power, 4.0);

    Ok(())
}

#[test]
fn xsph_jas_overlap_rejects_invalid_inputs() {
    let fixture = jas_radial_fixture();
    let short_final = Array1::<Complex>::zeros(JAS_RADIAL_ACTIVE_LEN - 1);
    let error = xsph_jas_overlap(XsphJasOverlapInput {
        final_large: short_final.view(),
        ..fixture.same_overlap_input(0, 0, 0)
    })
    .expect_err("final spinor must cover the active prefix");
    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "final_large",
            required: JAS_RADIAL_ACTIVE_LEN,
            actual: 6,
        }
    ));
}

#[test]
fn xsph_jas_radial_integral_matches_feff_radjas_reference() -> Result<(), XsphError> {
    let fixture = jas_radial_fixture();

    let same = xsph_jas_radial_integral(fixture.same_input())?;
    for (index, expected) in [
        Complex::new(-3.271_120_082_718_515_6e-2, 2.611_535_040_033_983e-3),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_complex_close(same.radial_integrals[index], expected);
    }
    assert_eq!(same.regular_coupling.shape(), &[JAS_RADIAL_ACTIVE_LEN, 5]);
    assert_eq!(
        same.regular_coupling.strides(),
        &[1, JAS_RADIAL_ACTIVE_LEN as isize]
    );
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(7.795_698_734_655_833e-2, -1.004_667_714_016_403_2e-2),
        Complex::new(1.083_287_943_729_738e-1, -5.989_497_958_479_377e-3),
        Complex::new(1.182_149_936_483_546_4e-1, -9.426_991_839_898_332e-4),
    ]) {
        assert_complex_close(same.regular_coupling[(index, 0)], expected);
    }
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(8.473_971_887_447_507e-2, -8.197_234_323_199_5e-3),
        Complex::new(4.986_061_020_468_446e-2, 1.483_586_621_086_033_1e-3),
        Complex::new(-1.039_040_713_505_878_5e-1, 7.289_875_354_170_097e-3),
    ]) {
        assert_complex_close(same.regular_coupling[(index, 2)], expected);
    }
    assert_eq!(same.near_origin_powers, arr1(&[1.0, 2.0, 3.0, 4.0, 0.0]));

    let different = xsph_jas_radial_integral(fixture.different_input())?;
    for (index, expected) in [
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(-1.050_500_041_540_072_3e-2, -2.125_014_849_058_739e-3),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_complex_close(different.radial_integrals[index], expected);
    }
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(6.523_513_042_794_926e-2, 6.633_244_621_180_122e-3),
        Complex::new(1.053_789_292_073_257_8e-1, -1.554_248_908_639_072e-3),
        Complex::new(1.306_661_473_499_400_3e-1, -9.184_879_295_494_825e-3),
    ]) {
        assert_complex_close(different.regular_coupling[(index, 0)], expected);
    }
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(6.471_521_338_089_362e-2, 6.580_378_367_472_684e-3),
        Complex::new(4.650_008_643_224_12e-2, -6.858_364_298_496_697e-4),
        Complex::new(-8.647_769_940_642_204e-2, 6.078_752_966_312_475e-3),
    ]) {
        assert_complex_close(different.regular_coupling[(index, 2)], expected);
    }
    assert_eq!(
        different.near_origin_powers,
        arr1(&[3.0, 4.0, 5.0, 6.0, 0.0])
    );
    Ok(())
}

#[test]
fn xsph_jas_radial_integral_rejects_invalid_inputs() {
    let fixture = jas_radial_fixture();
    let short_q_bessel =
        Array2::<Real>::zeros((JAS_RADIAL_ACTIVE_LEN - 1, JAS_RADIAL_LJMAX + 1).f());
    let error = xsph_jas_radial_integral(XsphJasRadialIntegralInput {
        q_bessel: short_q_bessel.view(),
        ..fixture.same_input()
    })
    .expect_err("q-Bessel table must cover active radii");
    assert!(matches!(
        error,
        XsphError::MatrixTooSmall {
            name: "q_bessel",
            required: [JAS_RADIAL_ACTIVE_LEN, 5],
            actual: [6, 5],
        }
    ));

    let bad_needed = arr1(&[1, 1, -1, 1, 0]);
    let error = xsph_jas_radial_integral(XsphJasRadialIntegralInput {
        needed_multipoles: bad_needed.view(),
        ..fixture.same_input()
    })
    .expect_err("ljneeded flags are nonnegative");
    assert!(matches!(
        error,
        XsphError::NegativeAngularMomentum {
            name: "needed_multipoles",
            index: 2,
            value: -1,
        }
    ));

    let error = xsph_jas_radial_integral(XsphJasRadialIntegralInput {
        initial_kappa: 0,
        ..fixture.same_input()
    })
    .expect_err("relativistic kappa is nonzero");
    assert_eq!(error, XsphError::ZeroKappa);
}

#[test]
fn xsph_jas_radial_cross_integral_matches_feff_radjas_reference() -> Result<(), XsphError> {
    let fixture = jas_radial_fixture();

    let same_regular = xsph_jas_radial_integral(fixture.same_input())?;
    let same = xsph_jas_radial_cross_integral(
        fixture.same_cross_input(same_regular.regular_coupling.view()),
    )?;
    for (index, expected) in [
        Complex::new(6.182_156_685_544_322e-4, -6.340_795_369_028_393e-5),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_complex_close(same.radial_integrals[index], expected);
    }
    assert_eq!(same.irregular_coupling.shape(), &[JAS_RADIAL_ACTIVE_LEN, 5]);
    assert_eq!(
        same.irregular_coupling.strides(),
        &[1, JAS_RADIAL_ACTIVE_LEN as isize]
    );
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(2.670_425_068_846_008e-4, -2.443_212_836_160_869_5e-6),
        Complex::new(1.263_905_607_115_131_7e-3, -1.271_040_466_380_665e-4),
        Complex::new(3.359_778_644_415_626e-3, -3.677_209_495_998_954_4e-4),
    ]) {
        assert_complex_close(same.weighted_irregular_coupling[(index, 0)], expected);
    }
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(3.128_331_076_601_623e-4, 1.700_951_624_254_111_6e-5),
        Complex::new(5.258_936_067_341_301e-4, 9.465_010_931_338_373e-6),
        Complex::new(-8.985_161_656_018_706e-4, 1.097_532_929_782_511_6e-4),
    ]) {
        assert_complex_close(same.weighted_irregular_coupling[(index, 2)], expected);
    }
    assert_eq!(
        same.first_near_origin_powers,
        arr1(&[2.0, 2.0, 2.0, 2.0, 0.0])
    );
    assert_eq!(
        same.second_near_origin_powers,
        arr1(&[4.0, 4.0, 4.0, 4.0, 0.0])
    );

    let different_regular = xsph_jas_radial_integral(fixture.different_input())?;
    let different = xsph_jas_radial_cross_integral(
        fixture.different_cross_input(different_regular.regular_coupling.view()),
    )?;
    for (index, expected) in [
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(2.640_673_704_702_674_8e-5, -3.975_738_199_122_718e-5),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_complex_close(different.radial_integrals[index], expected);
    }
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(1.314_793_748_182_110_2e-4, -1.237_438_461_482_972_8e-5),
        Complex::new(9.762_210_528_461_665e-4, -1.451_084_481_729_865_9e-5),
        Complex::new(3.286_620_741_566_653e-3, -5.710_968_623_336_948e-5),
    ]) {
        assert_complex_close(different.weighted_irregular_coupling[(index, 0)], expected);
    }
    for (index, expected) in [0, 3, JAS_RADIAL_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(1.293_919_730_333_971_5e-4, -1.217_792_556_893_996_6e-5),
        Complex::new(3.411_590_293_366_064_4e-4, -1.524_209_094_135_544_7e-6),
        Complex::new(-4.859_634_443_247_96e-4, -5.129_040_195_947_568_5e-5),
    ]) {
        assert_complex_close(different.weighted_irregular_coupling[(index, 2)], expected);
    }
    assert_eq!(
        different.first_near_origin_powers,
        arr1(&[4.0, 4.0, 4.0, 4.0, 0.0])
    );
    assert_eq!(
        different.second_near_origin_powers,
        arr1(&[4.0, 4.0, 4.0, 4.0, 0.0])
    );
    Ok(())
}

#[test]
fn xsph_jas_radial_cross_integral_rejects_invalid_inputs() {
    let fixture = jas_radial_fixture();
    let short_regular =
        Array2::<Complex>::zeros((JAS_RADIAL_ACTIVE_LEN - 1, JAS_RADIAL_LJMAX + 1).f());
    let error = xsph_jas_radial_cross_integral(fixture.same_cross_input(short_regular.view()))
        .expect_err("regular coupling must cover active radii and multipoles");
    assert!(matches!(
        error,
        XsphError::MatrixTooSmall {
            name: "regular_coupling",
            required: [JAS_RADIAL_ACTIVE_LEN, 5],
            actual: [6, 5],
        }
    ));
}

#[test]
fn xsph_radial_integral_matches_feff_radint_reference() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let cases = [
        (
            XsphRadialIntegralMode::RelativisticMatrixElement,
            XsphTransitionMultipole::ElectricDipole,
            -1,
            1,
            Complex::new(2.892_099_727_715_220_5e-3, 1.724_172_211_728_283_7e-3),
            3.0,
            [
                Complex::new(6.831_422_947_130_478e-2, 1.490_419_187_090_019e-1),
                Complex::new(8.419_899_406_144_694e-2, 1.182_014_210_097_877_6e-2),
                Complex::new(4.532_121_678_112_812e-2, -6.931_728_587_675_957e-2),
            ],
        ),
        (
            XsphRadialIntegralMode::RelativisticMatrixElement,
            XsphTransitionMultipole::ElectricQuadrupole,
            -2,
            1,
            Complex::new(-1.301_604_031_312_044e-3, -5.851_564_073_370_349e-3),
            5.0,
            [
                Complex::new(4.702_146_149_554_148e-2, -2.247_713_703_488_716_7e-2),
                Complex::new(2.960_937_672_178_702_2e-2, -1.452_815_136_801_924_6e-1),
                Complex::new(-3.066_367_843_002_329e-1, -3.562_704_488_551_439e-1),
            ],
        ),
        (
            XsphRadialIntegralMode::RelativisticMatrixElement,
            XsphTransitionMultipole::MagneticDipole,
            1,
            1,
            Complex::new(4.212_200_875_501_714_7e-4, -3.742_492_656_576_937e-3),
            5.0,
            [
                Complex::new(4.143_733_946_190_214_5e-2, -1.617_328_349_712_892e-2),
                Complex::new(5.764_888_732_857_414e-2, -9.358_128_917_552_017e-2),
                Complex::new(-1.470_587_491_344_930_2e-1, -2.240_380_204_500_566e-1),
            ],
        ),
        (
            XsphRadialIntegralMode::NonRelativisticMatrixElement,
            XsphTransitionMultipole::ElectricDipole,
            -1,
            1,
            Complex::new(1.401_404_826_116_659_5e-5, 1.961_818_457_058_759_6e-4),
            3.0,
            [
                Complex::new(6.156_176_748_515_444e-4, 3.289_173_886_367_542_6e-3),
                Complex::new(3.554_395_115_247_475e-4, 5.901_109_827_752_765_5e-3),
                Complex::new(-8.763_148_707_213_764e-5, 4.582_857_460_352_856e-3),
            ],
        ),
        (
            XsphRadialIntegralMode::NonRelativisticMatrixElement,
            XsphTransitionMultipole::ElectricQuadrupole,
            -2,
            1,
            Complex::new(5.864_289_140_110_974e-4, -1.183_481_681_757_632_4e-5),
            5.0,
            [
                Complex::new(1.329_381_892_417_094_1e-3, -2.488_135_373_418_527e-4),
                Complex::new(1.218_451_501_674_969_5e-2, -7.339_056_875_965_178e-4),
                Complex::new(4.405_978_641_986_449e-2, 8.424_928_415_199_33e-4),
            ],
        ),
    ];

    for (mode, multipole, initial_kappa, final_kappa, expected, power, expected_coupling) in cases {
        let result =
            xsph_radial_integral(fixture.input(mode, multipole, initial_kappa, final_kappa))?;
        assert_complex_close(result.value, expected);
        assert_close(result.near_origin_power, power);
        for (index, expected) in [0, 3, RADINT_ACTIVE_LEN - 1]
            .into_iter()
            .zip(expected_coupling)
        {
            assert_complex_close(result.coupling[index], expected);
        }
    }
    Ok(())
}

#[test]
fn xsph_radial_integral_rejects_feff_nonrelativistic_m1_stop() {
    let fixture = radint_fixture();
    let error = xsph_radial_integral(fixture.input(
        XsphRadialIntegralMode::NonRelativisticMatrixElement,
        XsphTransitionMultipole::MagneticDipole,
        -1,
        1,
    ))
    .expect_err("FEFF stops for nonrelativistic M1 radial integrals");

    assert!(matches!(
        error,
        XsphError::UnsupportedRadialMultipole {
            mode: XsphRadialIntegralMode::NonRelativisticMatrixElement,
            multipole: XsphTransitionMultipole::MagneticDipole
        }
    ));
}

#[test]
fn xsph_xsect_weighted_radial_integral_matches_scaled_final_state() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let weights = arr1(&[1.0, 0.95, 1.1, -0.4, 0.7, 1.25, -0.2]);
    let mode = XsphRadialIntegralMode::RelativisticMatrixElement;
    let multipole = XsphTransitionMultipole::ElectricDipole;
    let initial_kappa = -1;
    let final_kappa = 1;

    let baseline =
        xsph_radial_integral(fixture.input(mode, multipole, initial_kappa, final_kappa))?;
    let weighted = xsph_xsect_weighted_radial_integral(XsphXsectWeightedRadialIntegralInput {
        mode,
        multipole,
        initial_kappa,
        final_kappa,
        initial_large: fixture.initial_large.view(),
        initial_small: fixture.initial_small.view(),
        final_large_regular: fixture.final_large.view(),
        final_small_regular: fixture.final_small.view(),
        xray_bessel: fixture.bessel.view(),
        radii: fixture.radii.view(),
        log_step: 0.137,
        radial_weights: weights.view(),
        active_len: RADINT_ACTIVE_LEN,
    })?;

    let mut scaled_large = fixture.final_large.clone();
    let mut scaled_small = fixture.final_small.clone();
    for index in 0..RADINT_ACTIVE_LEN {
        scaled_large[index] *= weights[index];
        scaled_small[index] *= weights[index];
    }
    let expected = xsph_radial_integral(XsphRadialIntegralInput {
        final_large_regular: scaled_large.view(),
        final_small_regular: scaled_small.view(),
        ..fixture.input(mode, multipole, initial_kappa, final_kappa)
    })?;

    assert_complex_close(weighted.integral.value, expected.value);
    assert_close(
        weighted.integral.near_origin_power,
        expected.near_origin_power,
    );
    for index in [0, 3, RADINT_ACTIVE_LEN - 1] {
        assert_complex_close(
            weighted.unweighted_coupling[index],
            baseline.coupling[index],
        );
        assert_complex_close(weighted.integral.coupling[index], expected.coupling[index]);
    }
    Ok(())
}

#[test]
fn xsph_xsect_weighted_radial_integral_rejects_short_weights() {
    let fixture = radint_fixture();
    let weights = arr1(&[1.0; RADINT_ACTIVE_LEN - 1]);
    let error = xsph_xsect_weighted_radial_integral(XsphXsectWeightedRadialIntegralInput {
        mode: XsphRadialIntegralMode::RelativisticMatrixElement,
        multipole: XsphTransitionMultipole::ElectricDipole,
        initial_kappa: -1,
        final_kappa: 1,
        initial_large: fixture.initial_large.view(),
        initial_small: fixture.initial_small.view(),
        final_large_regular: fixture.final_large.view(),
        final_small_regular: fixture.final_small.view(),
        xray_bessel: fixture.bessel.view(),
        radii: fixture.radii.view(),
        log_step: 0.137,
        radial_weights: weights.view(),
        active_len: RADINT_ACTIVE_LEN,
    })
    .expect_err("fscf component must cover the active radial prefix");

    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "xsect_radial_weights",
            required: RADINT_ACTIVE_LEN,
            actual: 6
        }
    ));
}

#[test]
fn xsph_radial_cross_integral_matches_feff_radint_reference() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let cases = [
        (
            XsphRadialIntegralMode::RelativisticMatrixElement,
            XsphTransitionMultipole::ElectricDipole,
            -1,
            1,
            Complex::new(2.581_327_850_788_131_7e-6, -4.552_173_344_249_732_5e-6),
            3.0,
            4.0,
            [
                Complex::new(-1.195_804_531_816_632_7e-4, -6.562_402_291_164_173e-5),
                Complex::new(9.655_120_419_645_311e-5, -8.828_461_708_661_493e-5),
                Complex::new(2.018_055_558_250_359_2e-4, -2.310_333_399_332_404e-4),
            ],
        ),
        (
            XsphRadialIntegralMode::RelativisticMatrixElement,
            XsphTransitionMultipole::ElectricQuadrupole,
            -2,
            1,
            Complex::new(-9.615_688_713_045_417e-6, 2.281_221_711_581_777e-5),
            4.0,
            6.0,
            [
                Complex::new(9.730_546_216_957_02e-6, 5.619_345_403_590_163e-6),
                Complex::new(-7.182_214_696_449_478e-5, 1.067_786_330_861_279e-4),
                Complex::new(-8.960_351_663_910_681e-4, 3.633_517_890_039_465_5e-3),
            ],
        ),
        (
            XsphRadialIntegralMode::NonRelativisticMatrixElement,
            XsphTransitionMultipole::ElectricDipole,
            -1,
            1,
            Complex::new(-1.648_601_650_053_820_7e-8, 1.552_488_193_936_394_6e-9),
            3.0,
            4.0,
            [
                Complex::new(-5.132_471_332_985_111e-8, -2.549_325_734_676_883_5e-9),
                Complex::new(-4.076_563_072_540_999e-7, 3.806_441_721_154_816e-8),
                Complex::new(-8.222_596_606_962_085e-7, 8.650_614_027_042_337e-8),
            ],
        ),
        (
            XsphRadialIntegralMode::NonRelativisticMatrixElement,
            XsphTransitionMultipole::ElectricQuadrupole,
            -2,
            1,
            Complex::new(1.630_289_087_487_266_5e-7, -1.038_665_120_129_090_8e-8),
            4.0,
            6.0,
            [
                Complex::new(6.707_217_193_873_505e-9, 3.331_510_356_525_274e-10),
                Complex::new(8.790_876_757_248_892e-7, -5.752_818_047_410_138e-8),
                Complex::new(2.469_802_176_492_005_6e-5, -1.494_314_862_636_039e-6),
            ],
        ),
    ];

    for (
        mode,
        multipole,
        initial_kappa,
        final_kappa,
        expected,
        first_power,
        second_power,
        expected_weighted,
    ) in cases
    {
        let result = xsph_radial_cross_integral(fixture.cross_input(
            XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            mode,
            multipole,
            initial_kappa,
            final_kappa,
        ))?;
        assert_complex_close(result.value, expected);
        assert_close(result.first_near_origin_power, first_power);
        assert_close(result.second_near_origin_power, second_power);
        for (index, expected) in [0, 3, RADINT_ACTIVE_LEN - 1]
            .into_iter()
            .zip(expected_weighted)
        {
            assert_complex_close(result.weighted_irregular_coupling[index], expected);
        }
    }
    Ok(())
}

#[test]
fn xsph_xsect_weighted_radial_cross_integral_matches_scaled_final_state() -> Result<(), XsphError> {
    let fixture = radint_fixture();
    let regular_weights = arr1(&[0.9, 1.05, -0.3, 0.8, 1.2, -0.6, 0.4]);
    let irregular_weights = arr1(&[1.1, -0.2, 0.7, 1.3, -0.5, 0.6, 0.95]);
    let mode = XsphRadialIntegralMode::RelativisticMatrixElement;
    let multipole = XsphTransitionMultipole::ElectricDipole;
    let initial_kappa = -1;
    let final_kappa = 1;

    let baseline = xsph_radial_cross_integral(fixture.cross_input(
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
        mode,
        multipole,
        initial_kappa,
        final_kappa,
    ))?;
    let weighted =
        xsph_xsect_weighted_radial_cross_integral(XsphXsectWeightedRadialCrossIntegralInput {
            mode,
            branch: XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            multipole,
            initial_kappa,
            final_kappa,
            initial_large: fixture.initial_large.view(),
            initial_small: fixture.initial_small.view(),
            final_large_regular: fixture.final_large.view(),
            final_small_regular: fixture.final_small.view(),
            final_large_irregular: fixture.irregular_large.view(),
            final_small_irregular: fixture.irregular_small.view(),
            xray_bessel: fixture.bessel.view(),
            radii: fixture.radii.view(),
            log_step: 0.137,
            regular_weights: regular_weights.view(),
            irregular_weights: irregular_weights.view(),
            active_len: RADINT_ACTIVE_LEN,
        })?;

    let mut scaled_regular_large = fixture.final_large.clone();
    let mut scaled_regular_small = fixture.final_small.clone();
    let mut scaled_irregular_large = fixture.irregular_large.clone();
    let mut scaled_irregular_small = fixture.irregular_small.clone();
    for index in 0..RADINT_ACTIVE_LEN {
        scaled_regular_large[index] *= regular_weights[index];
        scaled_regular_small[index] *= regular_weights[index];
        scaled_irregular_large[index] *= irregular_weights[index];
        scaled_irregular_small[index] *= irregular_weights[index];
    }
    let expected = xsph_radial_cross_integral(XsphRadialCrossIntegralInput {
        branch: XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
        final_large_regular: scaled_regular_large.view(),
        final_small_regular: scaled_regular_small.view(),
        final_large_irregular: scaled_irregular_large.view(),
        final_small_irregular: scaled_irregular_small.view(),
        ..fixture.cross_input(
            XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            mode,
            multipole,
            initial_kappa,
            final_kappa,
        )
    })?;

    assert_complex_close(weighted.integral.value, expected.value);
    assert_close(
        weighted.integral.first_near_origin_power,
        expected.first_near_origin_power,
    );
    assert_close(
        weighted.integral.second_near_origin_power,
        expected.second_near_origin_power,
    );
    for index in [0, 3, RADINT_ACTIVE_LEN - 1] {
        assert_complex_close(
            weighted.unweighted_regular_coupling[index],
            baseline.regular_coupling[index],
        );
        assert_complex_close(
            weighted.unweighted_irregular_coupling[index],
            baseline.irregular_coupling[index],
        );
        assert_complex_close(
            weighted.integral.regular_coupling[index],
            expected.regular_coupling[index],
        );
        assert_complex_close(
            weighted.integral.irregular_coupling[index],
            expected.irregular_coupling[index],
        );
        assert_complex_close(
            weighted.integral.weighted_irregular_coupling[index],
            expected.weighted_irregular_coupling[index],
        );
    }
    Ok(())
}

#[test]
fn xsph_xsect_weighted_radial_cross_integral_supports_stored_branches() -> Result<(), XsphError> {
    let stored_fixture = radint_fixture();
    let current_fixture = radint_fixture_solution_set(2);
    let regular_weights = arr1(&[0.9, 1.05, -0.3, 0.8, 1.2, -0.6, 0.4]);
    let irregular_weights = arr1(&[1.1, -0.2, 0.7, 1.3, -0.5, 0.6, 0.95]);
    let mode = XsphRadialIntegralMode::RelativisticMatrixElement;
    let multipole = XsphTransitionMultipole::ElectricDipole;
    let initial_kappa = -1;
    let final_kappa = 1;

    let stored = xsph_radial_cross_integral(stored_fixture.cross_input(
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
        mode,
        multipole,
        initial_kappa,
        final_kappa,
    ))?;

    let stored_regular_current_irregular =
        xsph_xsect_weighted_radial_cross_integral(XsphXsectWeightedRadialCrossIntegralInput {
            mode,
            branch: XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
                stored_regular_coupling: stored.regular_coupling.view(),
            },
            multipole,
            initial_kappa,
            final_kappa,
            initial_large: current_fixture.initial_large.view(),
            initial_small: current_fixture.initial_small.view(),
            final_large_regular: current_fixture.final_large.view(),
            final_small_regular: current_fixture.final_small.view(),
            final_large_irregular: current_fixture.irregular_large.view(),
            final_small_irregular: current_fixture.irregular_small.view(),
            xray_bessel: current_fixture.bessel.view(),
            radii: current_fixture.radii.view(),
            log_step: 0.137,
            regular_weights: regular_weights.view(),
            irregular_weights: irregular_weights.view(),
            active_len: RADINT_ACTIVE_LEN,
        })?;
    let mut weighted_stored_regular = stored.regular_coupling.clone();
    let mut scaled_current_irregular_large = current_fixture.irregular_large.clone();
    let mut scaled_current_irregular_small = current_fixture.irregular_small.clone();
    for index in 0..RADINT_ACTIVE_LEN {
        weighted_stored_regular[index] *= regular_weights[index];
        scaled_current_irregular_large[index] *= irregular_weights[index];
        scaled_current_irregular_small[index] *= irregular_weights[index];
    }
    let expected_stored_regular = xsph_radial_cross_integral(XsphRadialCrossIntegralInput {
        branch: XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling: weighted_stored_regular.view(),
        },
        final_large_irregular: scaled_current_irregular_large.view(),
        final_small_irregular: scaled_current_irregular_small.view(),
        ..current_fixture.cross_input(
            XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            mode,
            multipole,
            initial_kappa,
            final_kappa,
        )
    })?;
    assert_complex_close(
        stored_regular_current_irregular.integral.value,
        expected_stored_regular.value,
    );
    for index in [0, 3, RADINT_ACTIVE_LEN - 1] {
        assert_complex_close(
            stored_regular_current_irregular.unweighted_regular_coupling[index],
            stored.regular_coupling[index],
        );
        assert_complex_close(
            stored_regular_current_irregular.integral.regular_coupling[index],
            expected_stored_regular.regular_coupling[index],
        );
        assert_complex_close(
            stored_regular_current_irregular.integral.irregular_coupling[index],
            expected_stored_regular.irregular_coupling[index],
        );
    }

    let current_regular_stored_irregular =
        xsph_xsect_weighted_radial_cross_integral(XsphXsectWeightedRadialCrossIntegralInput {
            mode,
            branch: XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
                stored_irregular_coupling: stored.irregular_coupling.view(),
            },
            multipole,
            initial_kappa,
            final_kappa,
            initial_large: current_fixture.initial_large.view(),
            initial_small: current_fixture.initial_small.view(),
            final_large_regular: current_fixture.final_large.view(),
            final_small_regular: current_fixture.final_small.view(),
            final_large_irregular: current_fixture.irregular_large.view(),
            final_small_irregular: current_fixture.irregular_small.view(),
            xray_bessel: current_fixture.bessel.view(),
            radii: current_fixture.radii.view(),
            log_step: 0.137,
            regular_weights: regular_weights.view(),
            irregular_weights: irregular_weights.view(),
            active_len: RADINT_ACTIVE_LEN,
        })?;
    let mut scaled_current_regular_large = current_fixture.final_large.clone();
    let mut scaled_current_regular_small = current_fixture.final_small.clone();
    let mut weighted_stored_irregular = stored.irregular_coupling.clone();
    for index in 0..RADINT_ACTIVE_LEN {
        scaled_current_regular_large[index] *= regular_weights[index];
        scaled_current_regular_small[index] *= regular_weights[index];
        weighted_stored_irregular[index] *= irregular_weights[index];
    }
    let expected_stored_irregular = xsph_radial_cross_integral(XsphRadialCrossIntegralInput {
        branch: XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling: weighted_stored_irregular.view(),
        },
        final_large_regular: scaled_current_regular_large.view(),
        final_small_regular: scaled_current_regular_small.view(),
        ..current_fixture.cross_input(
            XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            mode,
            multipole,
            initial_kappa,
            final_kappa,
        )
    })?;
    assert_complex_close(
        current_regular_stored_irregular.integral.value,
        expected_stored_irregular.value,
    );
    for index in [0, 3, RADINT_ACTIVE_LEN - 1] {
        assert_complex_close(
            current_regular_stored_irregular.unweighted_irregular_coupling[index],
            stored.irregular_coupling[index],
        );
        assert_complex_close(
            current_regular_stored_irregular.integral.regular_coupling[index],
            expected_stored_irregular.regular_coupling[index],
        );
        assert_complex_close(
            current_regular_stored_irregular.integral.irregular_coupling[index],
            expected_stored_irregular.irregular_coupling[index],
        );
    }

    Ok(())
}

#[test]
fn xsph_radial_cross_integral_reuses_feff_stored_couplings() -> Result<(), XsphError> {
    let stored_fixture = radint_fixture();
    let stored_regular = xsph_radial_integral(stored_fixture.input(
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?
    .coupling;
    let stored_irregular = xsph_radial_cross_integral(stored_fixture.cross_input(
        XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?
    .irregular_coupling;

    let expected_stored_irregular = [
        Complex::new(-5.401_869_350_233_13e-2, 4.014_419_969_143_441_5e-2),
        Complex::new(-2.807_192_360_017_292e-3, -2.737_336_785_994_255_4e-2),
        Complex::new(1.701_446_648_907_063_4e-2, -4.918_820_078_110_302_4e-2),
    ];
    for (index, expected) in [0, 3, RADINT_ACTIVE_LEN - 1]
        .into_iter()
        .zip(expected_stored_irregular)
    {
        assert_complex_close(stored_irregular[index], expected);
    }

    let current_fixture = radint_fixture_solution_set(2);
    let stored_regular_result = xsph_radial_cross_integral(current_fixture.cross_input(
        XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling: stored_regular.view(),
        },
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?;
    assert_complex_close(
        stored_regular_result.value,
        Complex::new(3.178_514_079_941_580_5e-6, -4.005_466_192_553_954_6e-7),
    );
    for (index, expected) in [0, 3, RADINT_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(-1.555_288_204_628_101_5e-5, 1.200_581_203_582_759_8e-4),
        Complex::new(1.382_475_976_000_127_5e-4, 1.963_369_874_523_783_8e-5),
        Complex::new(8.084_464_133_036_047e-5, -2.085_343_902_191_637_2e-4),
    ]) {
        assert_complex_close(
            stored_regular_result.weighted_irregular_coupling[index],
            expected,
        );
    }

    let stored_irregular_result = xsph_radial_cross_integral(current_fixture.cross_input(
        XsphRadialCrossIntegralBranch::CurrentRegularStoredIrregular {
            stored_irregular_coupling: stored_irregular.view(),
        },
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))?;
    assert_complex_close(
        stored_irregular_result.value,
        Complex::new(2.399_554_659_493_643e-6, -5.997_356_168_103_543e-7),
    );
    for (index, expected) in [0, 3, RADINT_ACTIVE_LEN - 1].into_iter().zip([
        Complex::new(-4.063_868_692_562_447e-5, -9.147_412_512_433e-5),
        Complex::new(8.796_016_778_519_88e-5, 4.227_931_432_992_743e-6),
        Complex::new(1.198_176_633_015_794_7e-4, 2.940_481_283_030_945_4e-5),
    ]) {
        assert_complex_close(
            stored_irregular_result.weighted_irregular_coupling[index],
            expected,
        );
    }
    Ok(())
}

#[test]
fn xsph_radial_integral_rejects_short_bessel_table() {
    let fixture = radint_fixture();
    let short_bessel = Array2::<Real>::zeros((2, RADINT_ACTIVE_LEN).f());
    let error = xsph_radial_integral(XsphRadialIntegralInput {
        xray_bessel: short_bessel.view(),
        ..fixture.input(
            XsphRadialIntegralMode::RelativisticMatrixElement,
            XsphTransitionMultipole::ElectricDipole,
            -1,
            1,
        )
    })
    .expect_err("three Bessel rows are required");

    assert!(matches!(
        error,
        XsphError::MatrixTooSmall {
            name: "xray_bessel",
            required: [3, RADINT_ACTIVE_LEN],
            actual: [2, RADINT_ACTIVE_LEN],
        }
    ));
}

#[test]
fn xsph_radial_cross_integral_rejects_short_stored_coupling() {
    let fixture = radint_fixture();
    let stored_regular = Array1::<Complex>::zeros(RADINT_ACTIVE_LEN - 1);
    let error = xsph_radial_cross_integral(fixture.cross_input(
        XsphRadialCrossIntegralBranch::StoredRegularCurrentIrregular {
            stored_regular_coupling: stored_regular.view(),
        },
        XsphRadialIntegralMode::RelativisticMatrixElement,
        XsphTransitionMultipole::ElectricDipole,
        -1,
        1,
    ))
    .expect_err("stored coupling must cover the active radial prefix");

    assert!(matches!(
        error,
        XsphError::LengthTooShort {
            name: "stored_regular_coupling",
            required: RADINT_ACTIVE_LEN,
            actual: 6,
        }
    ));
}

struct XsectDensityFixture {
    radii: Array1<Real>,
    regular_large: Array1<Complex>,
    regular_small: Array1<Complex>,
    irregular_large: Array1<Complex>,
    irregular_small: Array1<Complex>,
    atomic_large: Array1<Real>,
    atomic_small: Array1<Real>,
}

impl XsectDensityFixture {
    fn embedded_input(&self) -> XsphXsectEmbeddedDensityInput<'_> {
        XsphXsectEmbeddedDensityInput {
            final_l: 1,
            final_kappa: -2,
            wave_number: Complex::new(0.4, 0.5),
            regular_large: self.regular_large.view(),
            regular_small: self.regular_small.view(),
            irregular_large: self.irregular_large.view(),
            irregular_small: self.irregular_small.view(),
            radii: self.radii.view(),
            log_step: 0.1,
            norman_radius: 0.339_787_694_894_515_1,
            active_len: self.radii.len(),
            integration_len: 5,
        }
    }

    fn projected_input(&self) -> XsphXsectProjectedDensityInput<'_> {
        XsphXsectProjectedDensityInput {
            final_l: 1,
            final_kappa: -2,
            wave_number: Complex::new(0.4, 0.5),
            regular_large: self.regular_large.view(),
            regular_small: self.regular_small.view(),
            irregular_large: self.irregular_large.view(),
            irregular_small: self.irregular_small.view(),
            atomic_large: self.atomic_large.view(),
            atomic_small: self.atomic_small.view(),
            radii: self.radii.view(),
            log_step: 0.1,
            norman_radius: 0.339_787_694_894_515_1,
            active_len: self.radii.len(),
            integration_len: 5,
        }
    }
}

fn xsect_density_fixture() -> XsectDensityFixture {
    XsectDensityFixture {
        radii: arr1(&[0.2, 0.25, 0.32, 0.41, 0.53, 0.68]),
        regular_large: arr1(&[
            Complex::new(0.4, 0.1),
            Complex::new(0.35, 0.12),
            Complex::new(0.28, 0.16),
            Complex::new(0.2, 0.18),
            Complex::new(0.1, 0.2),
            Complex::new(0.05, 0.22),
        ]),
        regular_small: arr1(&[
            Complex::new(0.04, -0.02),
            Complex::new(0.05, -0.018),
            Complex::new(0.055, -0.015),
            Complex::new(0.06, -0.01),
            Complex::new(0.065, -0.005),
            Complex::new(0.07, 0.0),
        ]),
        irregular_large: arr1(&[
            Complex::new(0.2, 0.05),
            Complex::new(0.18, 0.08),
            Complex::new(0.15, 0.1),
            Complex::new(0.12, 0.11),
            Complex::new(0.09, 0.12),
            Complex::new(0.07, 0.13),
        ]),
        irregular_small: arr1(&[
            Complex::new(0.02, 0.01),
            Complex::new(0.022, 0.012),
            Complex::new(0.024, 0.014),
            Complex::new(0.026, 0.016),
            Complex::new(0.028, 0.018),
            Complex::new(0.03, 0.02),
        ]),
        atomic_large: arr1(&[0.9, 0.8, 0.7, 0.55, 0.4, 0.25]),
        atomic_small: arr1(&[0.05, 0.06, 0.065, 0.07, 0.075, 0.08]),
    }
}

struct RadintFixture {
    radii: Array1<Real>,
    initial_large: Array1<Real>,
    initial_small: Array1<Real>,
    final_large: Array1<Complex>,
    final_small: Array1<Complex>,
    irregular_large: Array1<Complex>,
    irregular_small: Array1<Complex>,
    bessel: Array2<Real>,
}

impl RadintFixture {
    fn input(
        &self,
        mode: XsphRadialIntegralMode,
        multipole: XsphTransitionMultipole,
        initial_kappa: i32,
        final_kappa: i32,
    ) -> XsphRadialIntegralInput<'_> {
        XsphRadialIntegralInput {
            mode,
            multipole,
            initial_kappa,
            final_kappa,
            initial_large: self.initial_large.view(),
            initial_small: self.initial_small.view(),
            final_large_regular: self.final_large.view(),
            final_small_regular: self.final_small.view(),
            xray_bessel: self.bessel.view(),
            radii: self.radii.view(),
            log_step: 0.137,
            active_len: RADINT_ACTIVE_LEN,
        }
    }

    fn cross_input<'a>(
        &'a self,
        branch: XsphRadialCrossIntegralBranch<'a>,
        mode: XsphRadialIntegralMode,
        multipole: XsphTransitionMultipole,
        initial_kappa: i32,
        final_kappa: i32,
    ) -> XsphRadialCrossIntegralInput<'a> {
        XsphRadialCrossIntegralInput {
            mode,
            branch,
            multipole,
            initial_kappa,
            final_kappa,
            initial_large: self.initial_large.view(),
            initial_small: self.initial_small.view(),
            final_large_regular: self.final_large.view(),
            final_small_regular: self.final_small.view(),
            final_large_irregular: self.irregular_large.view(),
            final_small_irregular: self.irregular_small.view(),
            xray_bessel: self.bessel.view(),
            radii: self.radii.view(),
            log_step: 0.137,
            active_len: RADINT_ACTIVE_LEN,
        }
    }
}

fn xsect_test_regular_channel(
    fixture: &RadintFixture,
    phase_shift: Complex,
) -> XsphXsectRegularChannel {
    let regular_large = xsect_test_active_complex_prefix(&fixture.final_large);
    let regular_small = xsect_test_active_complex_prefix(&fixture.final_small);
    XsphXsectRegularChannel {
        regular_solution: xsect_test_fovrg_solution(regular_large.clone(), regular_small.clone()),
        phase: XsphRegularPhase {
            phase_shift,
            phase_amplitude: Complex::new(1.0, 0.0),
            regular_large_at_muffin_tin: regular_large[RADINT_ACTIVE_LEN - 1],
            regular_small_at_muffin_tin: regular_small[RADINT_ACTIVE_LEN - 1],
            large_l: 0,
            small_l: 1,
            bessel_j_large: Complex::new(0.8, 0.1),
            neumann_large: Complex::new(-0.3, 0.05),
            bessel_j_small: Complex::new(0.25, -0.03),
            neumann_small: Complex::new(-0.6, 0.2),
        },
        normalized_solution: XsphXsectRegularSolution {
            small_component_factor: Complex::new(0.01, 0.02),
            relativistic_scale: Complex::new(1.0, -0.01),
            regular_solution_scale: Complex::new(1.0, 0.0),
            regular_large,
            regular_small,
        },
    }
}

fn xsect_test_irregular_channel(fixture: &RadintFixture) -> XsphXsectIrregularChannel {
    let irregular_large = xsect_test_active_complex_prefix(&fixture.irregular_large);
    let irregular_small = xsect_test_active_complex_prefix(&fixture.irregular_small);
    XsphXsectIrregularChannel {
        initial_condition: XsphXsectIrregularInitialCondition {
            large_component: irregular_large[RADINT_ACTIVE_LEN - 1],
            small_component: irregular_small[RADINT_ACTIVE_LEN - 1],
            small_component_factor: Complex::new(0.01, 0.02),
            relativistic_scale: Complex::new(1.0, -0.01),
        },
        irregular_solution: xsect_test_fovrg_solution(
            irregular_large.clone(),
            irregular_small.clone(),
        ),
        transformed_solution: XsphXsectIrregularTransform {
            phase_factor: Complex::new(0.9, 0.1),
            irregular_large,
            irregular_small,
        },
    }
}

fn xsect_test_active_complex_prefix(values: &Array1<Complex>) -> Array1<Complex> {
    Array1::from_iter(values.iter().take(RADINT_ACTIVE_LEN).copied())
}

fn xsect_test_fovrg_solution(
    large_component: Array1<Complex>,
    small_component: Array1<Complex>,
) -> crate::FovrgDiracSolution {
    let active_len = large_component.len();
    let empty_complex = Array1::<Complex>::zeros(0);
    crate::FovrgDiracSolution {
        large_component,
        small_component,
        large_coefficients: empty_complex.clone(),
        small_coefficients: empty_complex.clone(),
        muffin_tin_large_component: Complex::new(0.0, 0.0),
        muffin_tin_small_component: Complex::new(0.0, 0.0),
        exchange_correlation_potential: empty_complex.clone(),
        valence_exchange_correlation_potential: empty_complex.clone(),
        direct_potential: empty_complex.clone(),
        potential_coefficients: empty_complex.clone(),
        large_exchange: empty_complex.clone(),
        small_exchange: empty_complex.clone(),
        large_exchange_coefficients: empty_complex.clone(),
        small_exchange_coefficients: empty_complex.clone(),
        c3_potential: empty_complex,
        origin_powers: Array1::<Real>::zeros(0),
        normalization: Array1::<Real>::zeros(0),
        orbital_lengths: Array1::<usize>::zeros(0),
        active_len,
        retained_len: active_len,
        wkb_index: 0,
        target_last_index: active_len.saturating_sub(1),
        iteration_count: 0,
        difficult_iterations: 0,
    }
}

struct JasCorrectionFixture {
    radii: Array1<Real>,
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    q_bessel: Array3<Real>,
}

impl JasCorrectionFixture {
    fn input(&self) -> XsphJasOrthogonalityCorrectionInput<'_> {
        XsphJasOrthogonalityCorrectionInput {
            initial_j: 3,
            initial_l: 1,
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            q_bessel: self.q_bessel.view(),
            radii: self.radii.view(),
            log_step: 0.13,
            ljmax: JAS_CORRECTION_LJMAX,
            active_len: JAS_CORRECTION_ACTIVE_LEN,
        }
    }
}

struct JasRadialFixture {
    radii: Array1<Real>,
    initial_large: Array1<Real>,
    initial_small: Array1<Real>,
    same_final_large: Array1<Complex>,
    same_final_small: Array1<Complex>,
    different_final_large: Array1<Complex>,
    different_final_small: Array1<Complex>,
    same_irregular_large: Array1<Complex>,
    same_irregular_small: Array1<Complex>,
    different_irregular_large: Array1<Complex>,
    different_irregular_small: Array1<Complex>,
    needed: Array1<i32>,
    q_bessel: Array2<Real>,
    orthogonality_correction: Array1<Complex>,
}

impl JasRadialFixture {
    fn same_input(&self) -> XsphJasRadialIntegralInput<'_> {
        self.input(
            -1,
            -1,
            self.same_final_large.view(),
            self.same_final_small.view(),
        )
    }

    fn different_input(&self) -> XsphJasRadialIntegralInput<'_> {
        self.input(
            -1,
            2,
            self.different_final_large.view(),
            self.different_final_small.view(),
        )
    }

    fn same_overlap_input(
        &self,
        initial_l: usize,
        final_l: i32,
        radial_power: usize,
    ) -> XsphJasOverlapInput<'_> {
        self.overlap_input(
            initial_l,
            final_l,
            self.same_final_large.view(),
            self.same_final_small.view(),
            radial_power,
        )
    }

    fn different_overlap_input(
        &self,
        initial_l: usize,
        final_l: i32,
        radial_power: usize,
    ) -> XsphJasOverlapInput<'_> {
        self.overlap_input(
            initial_l,
            final_l,
            self.different_final_large.view(),
            self.different_final_small.view(),
            radial_power,
        )
    }

    fn same_cross_input<'a>(
        &'a self,
        regular_coupling: ArrayView2<'a, Complex>,
    ) -> XsphJasRadialCrossIntegralInput<'a> {
        self.cross_input(
            -1,
            -1,
            self.same_irregular_large.view(),
            self.same_irregular_small.view(),
            regular_coupling,
        )
    }

    fn different_cross_input<'a>(
        &'a self,
        regular_coupling: ArrayView2<'a, Complex>,
    ) -> XsphJasRadialCrossIntegralInput<'a> {
        self.cross_input(
            -1,
            2,
            self.different_irregular_large.view(),
            self.different_irregular_small.view(),
            regular_coupling,
        )
    }

    fn input<'a>(
        &'a self,
        initial_kappa: i32,
        final_kappa: i32,
        final_large_regular: ArrayView1<'a, Complex>,
        final_small_regular: ArrayView1<'a, Complex>,
    ) -> XsphJasRadialIntegralInput<'a> {
        XsphJasRadialIntegralInput {
            initial_kappa,
            final_kappa,
            initial_large: self.initial_large.view(),
            initial_small: self.initial_small.view(),
            final_large_regular,
            final_small_regular,
            needed_multipoles: self.needed.view(),
            q_bessel: self.q_bessel.view(),
            orthogonality_correction: self.orthogonality_correction.view(),
            radii: self.radii.view(),
            log_step: 0.13,
            ljmax: JAS_RADIAL_LJMAX,
            active_len: JAS_RADIAL_ACTIVE_LEN,
        }
    }

    fn overlap_input<'a>(
        &'a self,
        initial_l: usize,
        final_l: i32,
        final_large: ArrayView1<'a, Complex>,
        final_small: ArrayView1<'a, Complex>,
        radial_power: usize,
    ) -> XsphJasOverlapInput<'a> {
        XsphJasOverlapInput {
            initial_l,
            final_l,
            initial_large: self.initial_large.view(),
            initial_small: self.initial_small.view(),
            final_large,
            final_small,
            radii: self.radii.view(),
            log_step: 0.13,
            radial_power,
            active_len: JAS_RADIAL_ACTIVE_LEN,
        }
    }

    fn cross_input<'a>(
        &'a self,
        initial_kappa: i32,
        final_kappa: i32,
        final_large_irregular: ArrayView1<'a, Complex>,
        final_small_irregular: ArrayView1<'a, Complex>,
        regular_coupling: ArrayView2<'a, Complex>,
    ) -> XsphJasRadialCrossIntegralInput<'a> {
        XsphJasRadialCrossIntegralInput {
            initial_kappa,
            final_kappa,
            initial_large: self.initial_large.view(),
            initial_small: self.initial_small.view(),
            final_large_irregular,
            final_small_irregular,
            regular_coupling,
            needed_multipoles: self.needed.view(),
            q_bessel: self.q_bessel.view(),
            orthogonality_correction: self.orthogonality_correction.view(),
            radii: self.radii.view(),
            log_step: 0.13,
            ljmax: JAS_RADIAL_LJMAX,
            active_len: JAS_RADIAL_ACTIVE_LEN,
        }
    }
}

fn radint_fixture() -> RadintFixture {
    radint_fixture_solution_set(1)
}

fn jas_correction_fixture() -> JasCorrectionFixture {
    let mut radii = Array1::<Real>::zeros(JAS_CORRECTION_TOTAL_LEN);
    let mut large_component = Array1::<Real>::zeros(JAS_CORRECTION_TOTAL_LEN);
    let mut small_component = Array1::<Real>::zeros(JAS_CORRECTION_TOTAL_LEN);
    let mut q_bessel = Array3::<Real>::zeros(
        (
            JAS_CORRECTION_TOTAL_LEN,
            JAS_CORRECTION_LJMAX + 1,
            JAS_CORRECTION_Q_COUNT,
        )
            .f(),
    );
    for index in 0..JAS_CORRECTION_TOTAL_LEN {
        let i = index + 1;
        let i_real = i as Real;
        radii[index] = (-2.0 + 0.13 * index as Real).exp();
        large_component[index] = 0.42 + 0.033 * i_real + 0.001 * (i * i) as Real;
        small_component[index] = -0.07 + 0.014 * i_real - 0.0008 * (i * i) as Real;
        for q_index in 0..JAS_CORRECTION_Q_COUNT {
            let iq = q_index + 1;
            for angular_l in 0..=JAS_CORRECTION_LJMAX {
                q_bessel[(index, angular_l, q_index)] = (0.09 * (i * (angular_l + 1) * iq) as Real)
                    .cos()
                    + 0.015 * (angular_l * i) as Real
                    - 0.002 * (i * i * iq) as Real;
            }
        }
    }

    JasCorrectionFixture {
        radii,
        large_component,
        small_component,
        q_bessel,
    }
}

fn jas_radial_fixture() -> JasRadialFixture {
    let mut radii = Array1::<Real>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut initial_large = Array1::<Real>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut initial_small = Array1::<Real>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut same_final_large = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut same_final_small = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut different_final_large = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut different_final_small = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut same_irregular_large = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut same_irregular_small = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut different_irregular_large = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut different_irregular_small = Array1::<Complex>::zeros(JAS_RADIAL_TOTAL_LEN);
    let mut q_bessel = Array2::<Real>::zeros((JAS_RADIAL_TOTAL_LEN, JAS_RADIAL_LJMAX + 1).f());
    for index in 0..JAS_RADIAL_TOTAL_LEN {
        let i = index + 1;
        let i_real = i as Real;
        radii[index] = (-1.8 + 0.13 * index as Real).exp();
        initial_large[index] = 0.36 + 0.029 * i_real + 0.0012 * (i * i) as Real;
        initial_small[index] = -0.06 + 0.012 * i_real - 0.0007 * (i * i) as Real;
        same_final_large[index] = Complex::new(0.20 + 0.021 * i_real, -0.03 + 0.004 * i_real);
        same_final_small[index] = Complex::new(-0.08 + 0.017 * i_real, 0.025 + 0.003 * i_real);
        different_final_large[index] = Complex::new(0.14 + 0.025 * i_real, 0.02 - 0.006 * i_real);
        different_final_small[index] = Complex::new(-0.04 + 0.013 * i_real, -0.03 + 0.005 * i_real);
        same_irregular_large[index] = Complex::new(0.07 + 0.018 * i_real, 0.012 - 0.003 * i_real);
        same_irregular_small[index] =
            Complex::new(-0.025 + 0.010 * i_real, -0.018 + 0.004 * i_real);
        different_irregular_large[index] =
            Complex::new(0.06 + 0.016 * i_real, -0.015 + 0.002 * i_real);
        different_irregular_small[index] =
            Complex::new(-0.018 + 0.009 * i_real, 0.021 - 0.003 * i_real);
        for angular_l in 0..=JAS_RADIAL_LJMAX {
            q_bessel[(index, angular_l)] = (0.11 * (i * (angular_l + 1)) as Real).cos()
                + 0.02 * (angular_l * i) as Real
                - 0.001 * (i * i) as Real;
        }
    }
    let orthogonality_correction =
        Array1::from_iter((0..=JAS_RADIAL_LJMAX).map(|angular_l| {
            Complex::new(0.12 / (angular_l + 1) as Real, -0.015 * angular_l as Real)
        }));

    JasRadialFixture {
        radii,
        initial_large,
        initial_small,
        same_final_large,
        same_final_small,
        different_final_large,
        different_final_small,
        same_irregular_large,
        same_irregular_small,
        different_irregular_large,
        different_irregular_small,
        needed: arr1(&[1, 1, 1, 1, 0]),
        q_bessel,
        orthogonality_correction,
    }
}

fn radint_fixture_solution_set(solution_set: usize) -> RadintFixture {
    let mut radii = Array1::<Real>::zeros(RADINT_TOTAL_LEN);
    let mut initial_large = Array1::<Real>::zeros(RADINT_TOTAL_LEN);
    let mut initial_small = Array1::<Real>::zeros(RADINT_TOTAL_LEN);
    let mut final_large = Array1::<Complex>::zeros(RADINT_TOTAL_LEN);
    let mut final_small = Array1::<Complex>::zeros(RADINT_TOTAL_LEN);
    let mut irregular_large = Array1::<Complex>::zeros(RADINT_TOTAL_LEN);
    let mut irregular_small = Array1::<Complex>::zeros(RADINT_TOTAL_LEN);
    let mut bessel = Array2::<Real>::zeros((3, RADINT_TOTAL_LEN).f());
    for index in 0..RADINT_TOTAL_LEN {
        let i = index as Real + 1.0;
        radii[index] = (-3.7 + 0.137 * index as Real).exp();
        initial_large[index] = 0.58 + 0.041 * i + 0.002 * i * i;
        initial_small[index] = -0.13 + 0.017 * i - 0.0007 * i * i;
        bessel[(0, index)] = (0.19 * i).cos();
        bessel[(1, index)] = (0.17 * i).sin() + 0.04 * i;
        bessel[(2, index)] = 0.025 * i * i - 0.031 * i;
        if solution_set == 1 {
            final_large[index] = Complex::new(0.22 + 0.031 * i, -0.051 + 0.009 * i);
            final_small[index] = Complex::new(-0.11 + 0.026 * i, 0.037 + 0.006 * i);
            irregular_large[index] = Complex::new(0.075 + 0.019 * i, 0.021 - 0.004 * i);
            irregular_small[index] = Complex::new(-0.035 + 0.014 * i, -0.046 + 0.011 * i);
        } else {
            final_large[index] = Complex::new(0.16 + 0.024 * i, 0.033 - 0.007 * i);
            final_small[index] = Complex::new(-0.085 + 0.018 * i, -0.024 + 0.008 * i);
            irregular_large[index] = Complex::new(0.052 + 0.015 * i, -0.018 + 0.005 * i);
            irregular_small[index] = Complex::new(-0.028 + 0.011 * i, 0.039 - 0.006 * i);
        }
    }

    RadintFixture {
        radii,
        initial_large,
        initial_small,
        final_large,
        final_small,
        irregular_large,
        irregular_small,
        bessel,
    }
}
