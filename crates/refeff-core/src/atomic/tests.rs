use ndarray::{Array1, Array2, Array3};

use super::*;

fn assert_close(actual: Real, expected: Real) {
    assert_close_with(actual, expected, 1.0e-12);
}

fn assert_close_with(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() < tolerance,
        "actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

fn assert_some_close(actual: Option<Real>, expected: Real, tolerance: Real) {
    match actual {
        Some(value) => assert_close_with(value, expected, tolerance),
        None => assert_eq!(actual, Some(expected)),
    }
}

mod coefficients;
mod dirac_matching;
mod dirac_state;
mod integrals;
mod orbitals;
mod radial_density;
mod tables_nuclear;
mod validation;

struct SchmidtFixture {
    kappas: Vec<i32>,
    active_lengths: Vec<usize>,
    orbital_powers: Vec<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
}

struct DsordfFixture {
    radii: Array1<Real>,
    active_lengths: Vec<usize>,
    orbital_powers: Vec<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
    derivative_large: Array1<Real>,
    derivative_small: Array1<Real>,
    derivative_large_coefficients: Array1<Real>,
    derivative_small_coefficients: Array1<Real>,
}

struct YzktegFixture {
    source: Array1<Real>,
    source_coefficients: Array1<Real>,
    radii: Array1<Real>,
}

struct VldaFixture {
    radii: Array1<Real>,
    active_lengths: Vec<usize>,
    occupations: Vec<Real>,
    valence_occupations: Vec<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    initial_potential: Array1<Real>,
    initial_development_coefficients: Array1<Real>,
    initial_energy_density: Array1<Real>,
}

struct PotrdfFixture {
    radii: Array1<Real>,
    active_lengths: Vec<usize>,
    kappas: Vec<i32>,
    orbital_powers: Vec<Real>,
    occupations: Vec<Real>,
    shell_markers: Vec<i32>,
    origin_scales: Vec<Real>,
    coulomb_coefficients: Array3<Real>,
    lagrange_parameters: Array1<Real>,
    nuclear_potential: Array1<Real>,
    nuclear_development_coefficients: Array1<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
}

struct SoldirNormFixture {
    radii: Array1<Real>,
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
}

struct SoldirSolutionNormalizationFixture {
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
}

struct SoldirSetupFixture {
    radii: Array1<Real>,
    potential: Array1<Real>,
    potential_coefficients: Array1<Real>,
    positive_origin_coefficients: Array1<Real>,
}

struct IntdirFixture {
    radii: Array1<Real>,
    potential: Array1<Real>,
    potential_coefficients: Array1<Real>,
    large_source: Array1<Real>,
    small_source: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
}

impl SoldirNormFixture {
    fn input(
        &self,
        method: i32,
        coefficient_count: usize,
        matching_small_component: Real,
        origin_power: Real,
        active_len: usize,
        matching_index_1based: usize,
    ) -> AtomicDiracNormalizationInput<'_> {
        AtomicDiracNormalizationInput {
            radii: self.radii.view(),
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            method,
            step: 0.05,
            coefficient_count,
            matching_small_component,
            origin_power,
            active_len,
            matching_index_1based,
        }
    }
}

impl SoldirSolutionNormalizationFixture {
    fn input(
        &self,
        norm: Real,
        initial_large_coefficient: Real,
        initial_small_coefficient: Real,
    ) -> AtomicDiracSolutionNormalizationInput<'_> {
        AtomicDiracSolutionNormalizationInput {
            norm,
            initial_large_coefficient,
            initial_small_coefficient,
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            coefficient_count: 4,
            active_len: 7,
        }
    }
}

impl SoldirSetupFixture {
    fn input(
        &self,
        energy: Real,
        method: i32,
        kappa: i32,
        principal_quantum_number: usize,
        negative_origin: bool,
    ) -> AtomicDiracSolverSetupInput<'_> {
        AtomicDiracSolverSetupInput {
            energy,
            origin_power: 1.25,
            initial_large_coefficient: 0.82,
            initial_small_coefficient: -0.006,
            principal_quantum_number,
            kappa,
            speed_of_light: 137.0373,
            method,
            radii: self.radii.view(),
            potential: self.potential.view(),
            potential_coefficients: if negative_origin {
                self.potential_coefficients.view()
            } else {
                self.positive_origin_coefficients.view()
            },
            active_len: 7,
        }
    }
}

impl IntdirFixture {
    fn input(
        &self,
        mode: AtomicDiracIntegrationMode,
        matching_index_1based: usize,
        max_index_1based: usize,
    ) -> AtomicDiracIntegrationInput<'_> {
        AtomicDiracIntegrationInput {
            large_source: self.large_source.view(),
            small_source: self.small_source.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            radii: self.radii.view(),
            potential: self.potential.view(),
            potential_coefficients: self.potential_coefficients.view(),
            energy: -0.08,
            origin_power: 0.999,
            initial_large_coefficient: 0.85,
            initial_small_coefficient: -0.004,
            asymptotic_large_component: 0.02,
            kappa: -1,
            speed_of_light: 137.0373,
            step: 0.05,
            matching_precision: 1.0e-7,
            coefficient_count: 6,
            active_len: 151,
            mode,
            matching_index_1based,
            max_index_1based,
        }
    }
}

impl DsordfFixture {
    fn input(
        &self,
        kind: AtomicDifferentialIntegralKind,
        power: i32,
        origin_power: Real,
    ) -> AtomicDifferentialIntegralInput<'_> {
        AtomicDifferentialIntegralInput {
            kind,
            power,
            origin_power,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            derivative_large: self.derivative_large.view(),
            derivative_small: self.derivative_small.view(),
            derivative_large_coefficients: self.derivative_large_coefficients.view(),
            derivative_small_coefficients: self.derivative_small_coefficients.view(),
        }
    }

    fn yzkrdf_input(
        &self,
        left_orbital_1based: usize,
        right_orbital_1based: usize,
        angular_momentum: usize,
        large_small: bool,
    ) -> AtomicYkZkExchangeInput<'_> {
        AtomicYkZkExchangeInput {
            left_orbital_1based,
            right_orbital_1based,
            large_small,
            angular_momentum,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }

    fn fdrirk_input<'a>(
        &'a self,
        request: AtomicRadialIntegralRequest,
        kappas: &'a [i32],
        large_small: bool,
        previous_first_factor: Option<AtomicRadialFirstFactorView<'a>>,
    ) -> AtomicRadialIntegralInput<'a> {
        AtomicRadialIntegralInput {
            request,
            large_small,
            previous_first_factor,
            kappas,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

impl YzktegFixture {
    fn input(&self) -> AtomicYkZkTransformInput<'_> {
        AtomicYkZkTransformInput {
            source: self.source.view(),
            source_coefficients: self.source_coefficients.view(),
            radii: self.radii.view(),
            initial_power: 0.65,
            step: 0.05,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 13,
        }
    }

    fn prepared_input(
        &self,
        source_len: usize,
        angular_momentum: usize,
    ) -> AtomicYkZkPreparedSourceInput<'_> {
        AtomicYkZkPreparedSourceInput {
            source: self.source.view(),
            source_coefficients: self.source_coefficients.view(),
            radii: self.radii.view(),
            step: 0.05,
            angular_momentum,
            coefficient_count: 6,
            source_len,
            active_len: 13,
        }
    }
}

impl VldaFixture {
    fn input(
        &self,
        mode: AtomicLocalDensityExchangeMode,
        accumulate_energy_density: bool,
    ) -> AtomicLocalDensityPotentialInput<'_> {
        AtomicLocalDensityPotentialInput {
            mode,
            accumulate_energy_density,
            speed_of_light: 137.035_999,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            occupations: &self.occupations,
            valence_occupations: &self.valence_occupations,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            initial_potential: self.initial_potential.view(),
            initial_development_coefficients: self.initial_development_coefficients.view(),
            initial_energy_density: self.initial_energy_density.view(),
        }
    }
}

impl PotrdfFixture {
    fn input(
        &self,
        include_exchange: bool,
        include_lagrange: bool,
    ) -> AtomicOrbitalPotentialInput<'_> {
        AtomicOrbitalPotentialInput {
            active_orbital_1based: 2,
            include_exchange,
            include_lagrange,
            self_consistent_count: 3,
            speed_of_light: 137.035_999,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            kappas: &self.kappas,
            orbital_powers: &self.orbital_powers,
            occupations: &self.occupations,
            shell_markers: &self.shell_markers,
            origin_scales: &self.origin_scales,
            coulomb_coefficients: self.coulomb_coefficients.view(),
            lagrange_parameters: self.lagrange_parameters.view(),
            nuclear_potential: self.nuclear_potential.view(),
            nuclear_development_coefficients: self.nuclear_development_coefficients.view(),
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

fn sample_dsordf_fixture() -> DsordfFixture {
    sample_atomic_radial_fixture(11)
}

fn sample_yzkrdf_fixture() -> DsordfFixture {
    sample_atomic_radial_fixture(13)
}

fn sample_soldir_norm_fixture() -> SoldirNormFixture {
    SoldirNormFixture {
        radii: Array1::from_shape_fn(251, |row| (-8.8 + 0.05 * row as Real).exp()),
        large_component: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            0.03 * index + 0.002 * (0.17 * index).sin()
        }),
        small_component: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            -0.014 * index + 0.003 * (0.11 * index).cos()
        }),
        large_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            0.021 * index - 0.0007 * index * index
        }),
        small_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            -0.013 * index + 0.0004 * index * index
        }),
    }
}

fn sample_soldir_solution_normalization_fixture(
    flip_coefficients: bool,
    flip_components: bool,
) -> SoldirSolutionNormalizationFixture {
    let mut large_coefficients = Array1::from_shape_fn(10, |row| {
        let index = (row + 1) as Real;
        0.2 * index + 0.01 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(10, |row| {
        let index = (row + 1) as Real;
        -0.11 * index + 0.003 * index * index
    });
    let mut large_component = Array1::from_shape_fn(9, |row| {
        let index = (row + 1) as Real;
        0.04 * index + 0.001 * index * index
    });
    let small_component = Array1::from_shape_fn(9, |row| {
        let index = (row + 1) as Real;
        -0.03 * index + 0.0005 * index * index
    });

    if flip_coefficients {
        large_coefficients[0] = -large_coefficients[0];
    }
    if flip_components {
        large_component[0] = -large_component[0];
    }

    SoldirSolutionNormalizationFixture {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
    }
}

fn sample_soldir_node_count_component() -> Array1<Real> {
    Array1::from_vec(vec![0.2, 0.1, -0.05, 0.0, 0.0, 0.03, -0.02, -0.01, 0.01])
}

fn sample_soldir_setup_fixture() -> SoldirSetupFixture {
    SoldirSetupFixture {
        radii: Array1::from_shape_fn(7, |row| 0.08 * (0.11 * row as Real).exp()),
        potential: Array1::from_shape_fn(7, |row| {
            let radius = 0.08 * (0.11 * row as Real).exp();
            -0.42 * (-0.30 * radius).exp() + 0.008 * row as Real
        }),
        potential_coefficients: Array1::from_vec(vec![-0.058_378_260_164_777, 0.0006, -0.0003]),
        positive_origin_coefficients: Array1::from_vec(vec![0.021, 0.0006, -0.0003]),
    }
}

fn sample_intdir_fixture() -> IntdirFixture {
    let speed_of_light = 137.0373;
    let step = 0.05;
    let nuclear_charge = 8.0;
    IntdirFixture {
        radii: Array1::from_shape_fn(251, |row| 0.03 * (step * row as Real).exp()),
        potential: Array1::from_shape_fn(251, |row| {
            let radius = 0.03 * (step * row as Real).exp();
            -0.25 * (-0.40 * radius).exp()
        }),
        potential_coefficients: Array1::from_shape_fn(10, |row| {
            if row == 0 {
                -nuclear_charge / speed_of_light
            } else {
                0.0003 * row as Real * (-1.0_f64).powi((row + 1) as i32)
            }
        }),
        large_source: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            0.001 * (0.17 * index).sin() + 0.0002 * (0.03 * index).cos()
        }),
        small_source: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            0.0007 * (0.11 * index).cos() - 0.0001 * (0.05 * index).sin()
        }),
        large_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            0.0002 * index * (-1.0_f64).powi((row + 1) as i32)
        }),
        small_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            -0.00015 * index * (-1.0_f64).powi((row + 1) as i32)
        }),
    }
}

fn sample_vlda_fixture() -> VldaFixture {
    let radial_count = 13;
    let orbital_count = 3;
    VldaFixture {
        radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
        active_lengths: vec![9, 11, 7],
        occupations: vec![2.0, 1.6, 0.7],
        valence_occupations: vec![1.0, 0.4, 0.2],
        large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        }),
        small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        }),
        initial_potential: Array1::from_shape_fn(radial_count, |row| 0.0001 * (row + 1) as Real),
        initial_development_coefficients: Array1::from_shape_fn(6, |row| 0.01 * (row + 1) as Real),
        initial_energy_density: Array1::from_shape_fn(radial_count, |row| {
            0.002 * (row + 1) as Real
        }),
    }
}

fn sample_potrdf_fixture() -> PotrdfFixture {
    let radial_count = 13;
    let orbital_count = 3;
    let coefficient_count = 6;
    PotrdfFixture {
        radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
        active_lengths: vec![9, 11, 7],
        kappas: vec![-1, 1, 1],
        orbital_powers: (1..=orbital_count)
            .map(|orbital| 0.12 + 0.09 * orbital as Real)
            .collect(),
        occupations: vec![2.0, 1.6, 0.7],
        shell_markers: vec![-1, 1, 1],
        origin_scales: vec![1.05, 0.95, 1.10],
        coulomb_coefficients: Array3::from_shape_fn(
            (orbital_count, orbital_count, 5),
            |(left, right, rank)| {
                0.015 * (left + 1) as Real + 0.011 * (right + 1) as Real + 0.003 * rank as Real
            },
        ),
        lagrange_parameters: Array1::from_shape_fn(3, |row| 0.012 * (row + 1) as Real),
        nuclear_potential: Array1::from_shape_fn(radial_count, |row| {
            -0.2 + 0.001 * (row + 1) as Real
        }),
        nuclear_development_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            -0.03 * (row + 1) as Real
        }),
        large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        }),
        small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        }),
        large_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| 0.08 * (row + 1) as Real + 0.015 * (col + 1) as Real,
        ),
        small_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| -0.02 * (row + 1) as Real + 0.01 * (col + 1) as Real,
        ),
    }
}

fn sample_atomic_radial_fixture(radial_count: usize) -> DsordfFixture {
    let orbital_count = 3;
    let coefficient_count = 6;
    DsordfFixture {
        radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
        active_lengths: vec![9, 11, 7],
        orbital_powers: (1..=orbital_count)
            .map(|orbital| 0.12 + 0.09 * orbital as Real)
            .collect(),
        large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        }),
        small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        }),
        large_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| {
                let coefficient = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                0.08 * coefficient + 0.015 * orbital
            },
        ),
        small_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| {
                let coefficient = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                -0.02 * coefficient + 0.01 * orbital
            },
        ),
        derivative_large: Array1::from_shape_fn(radial_count, |row| {
            let radial = (row + 1) as Real;
            0.015 * radial - 0.00007 * radial * radial
        }),
        derivative_small: Array1::from_shape_fn(radial_count, |row| {
            let radial = (row + 1) as Real;
            -0.004 * radial + 0.00013 * radial * radial
        }),
        derivative_large_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            let coefficient = (row + 1) as Real;
            0.05 * coefficient - 0.003
        }),
        derivative_small_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            let coefficient = (row + 1) as Real;
            -0.015 * coefficient + 0.004
        }),
    }
}

fn sample_yzkteg_fixture() -> YzktegFixture {
    let active_len = 13;
    let coefficient_count = 6;
    YzktegFixture {
        source: Array1::from_shape_fn(active_len, |row| {
            let row = (row + 1) as Real;
            0.017 * row + 0.0008 * row * row - 0.00001 * row * row * row
        }),
        source_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            let row = (row + 1) as Real;
            0.04 * row - 0.0015 * row * row
        }),
        radii: Array1::from_shape_fn(active_len, |row| (-4.2 + 0.05 * row as Real).exp()),
    }
}

impl SchmidtFixture {
    fn as_input(
        &self,
        active_orbital_1based: Option<usize>,
    ) -> AtomicSchmidtOrthogonalizationInput<'_> {
        AtomicSchmidtOrthogonalizationInput {
            active_orbital_1based,
            kappas: &self.kappas,
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

fn sample_schmidt_fixture() -> SchmidtFixture {
    SchmidtFixture {
        kappas: vec![-1, -1, 1, -1],
        active_lengths: vec![3, 4, 3, 5],
        orbital_powers: (1..=4).map(|orbital| 0.1 * orbital as Real).collect(),
        large_components: Array2::from_shape_fn((5, 4), |(row, orbital)| {
            0.07 * (row + 1) as Real + 0.11 * (orbital + 1) as Real
        }),
        small_components: Array2::from_shape_fn((5, 4), |(row, orbital)| {
            0.03 * (row + 1) as Real - 0.02 * (orbital + 1) as Real
        }),
        large_coefficients: Array2::from_shape_fn((4, 4), |(row, orbital)| {
            0.2 * (row + 1) as Real + 0.05 * (orbital + 1) as Real
        }),
        small_coefficients: Array2::from_shape_fn((4, 4), |(row, orbital)| {
            -0.03 * (row + 1) as Real + 0.04 * (orbital + 1) as Real
        }),
    }
}

fn sample_schmidt_integral(
    request: AtomicSchmidtIntegralRequest<'_>,
) -> Result<Real, AtomMathError> {
    match request {
        AtomicSchmidtIntegralRequest::Projection(request) => Ok(request
            .target_large
            .iter()
            .zip(request.reference_large.iter())
            .map(|(&target, &reference)| target * reference)
            .sum::<Real>()
            + request
                .target_small
                .iter()
                .zip(request.reference_small.iter())
                .map(|(&target, &reference)| target * reference)
                .sum::<Real>()),
        AtomicSchmidtIntegralRequest::Norm(request) => Ok(request
            .target_large
            .iter()
            .map(|&value| value * value)
            .sum::<Real>()
            + request
                .target_small
                .iter()
                .map(|&value| value * value)
                .sum::<Real>()),
    }
}

fn assert_columns_close<const ROWS: usize, const COLUMNS: usize>(
    actual: &Array2<Real>,
    expected_columns: &[[Real; ROWS]; COLUMNS],
    tolerance: Real,
) {
    assert_eq!(actual.nrows(), ROWS);
    assert_eq!(actual.ncols(), COLUMNS);
    for (column, expected_column) in expected_columns.iter().enumerate() {
        for (row, &expected) in expected_column.iter().enumerate() {
            assert_close_with(actual[(row, column)], expected, tolerance);
        }
    }
}

fn sample_s02at_overlaps() -> Array2<Real> {
    let mut overlaps =
        Array2::from_shape_fn((6, 6), |(row, column)| 0.02 * (row + column + 2) as Real);
    for index in 0..6 {
        overlaps[(index, index)] = 1.0;
    }
    overlaps[(0, 1)] = 0.91;
    overlaps[(1, 0)] = 0.91;
    overlaps[(2, 3)] = 0.82;
    overlaps[(3, 2)] = 0.82;
    overlaps
}

fn sample_atomic_radial_integral(
    request: AtomicRadialIntegralRequest,
) -> Result<Real, AtomMathError> {
    Ok(0.0001 * (request.rank + 1) as Real
        + 0.001 * request.first_left as Real
        + 0.0002 * request.first_right as Real
        + 0.00003 * request.second_left as Real
        + 0.000004 * request.second_right as Real)
}

fn sample_atomic_tabrat_integral(
    request: AtomicTabulationIntegralRequest,
) -> Result<Real, AtomMathError> {
    Ok(0.01 * (request.left + 1) as Real
        + 0.02 * (request.right + 1) as Real
        + 0.001 * request.power as Real
        + 0.1)
}
