use super::*;

pub(super) fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual {actual} expected {expected}"
    );
}

pub(super) fn assert_close_tol(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() < tolerance,
        "actual {actual} expected {expected}"
    );
}

pub(super) fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

pub(super) fn hole_orbital_source() -> (Array1<Real>, Array1<Real>) {
    let mut large = Array1::<Real>::zeros(251);
    let mut small = Array1::<Real>::zeros(251);
    for index in 0..15 {
        let i = index as Real + 1.0;
        large[index] = 0.1 + 0.017 * i + 0.0009 * i * i + 0.002 * (0.3 * i).sin();
        small[index] = -0.04 + 0.011 * i - 0.0004 * i * i + 0.001 * (0.25 * i).cos();
    }
    (large, small)
}

pub(super) fn phase_mesh84_input(spectroscopy: i32) -> XsphPhaseEnergyMesh84Input {
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

pub(super) struct XsphSpectrumFixture {
    pub(super) index_map: Array1<i32>,
    pub(super) final_lj: Array1<i32>,
    pub(super) radial_integrals: Array1<Complex>,
    pub(super) q_cosines: Array2<Real>,
    pub(super) transition_weights: Array3<Real>,
}

pub(super) fn xsph_spectrum_fixture() -> XsphSpectrumFixture {
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

pub(super) fn acoef_sum(coefficients: &Array5<Real>, operator: usize, lmax: usize) -> Real {
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

pub(super) fn acoef_entry(
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
