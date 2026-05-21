use ndarray::{Array1, Array2};

use crate::{Complex, Real};

use super::{
    FovrgAngularCoefficientsInput, FovrgC3DerivativeInput, FovrgDiracSolverInput, FovrgError,
    FovrgExchangePotentialInput, FovrgFlatPotentialInput, FovrgInitialPhotoelectronInput,
    FovrgInwardSolutionInput, FovrgNuclearPotentialInput, FovrgOrbitalSetupInput,
    FovrgOrthogonalizationInput, FovrgOutgoingSolutionInput, FovrgOutwardIntegrationInput,
    FovrgOverlapIntegralInput, FovrgPotentialDevelopmentInput, FovrgYkZkExchangeInput,
    FovrgYkZkTransformInput, fovrg_angular_coefficients, fovrg_c3_derivative,
    fovrg_complex_real_product_coefficient, fovrg_dirac_solver, fovrg_exchange_potential,
    fovrg_flat_potential_propagate, fovrg_initial_photoelectron, fovrg_inward_solution,
    fovrg_nuclear_potential, fovrg_orbital_setup, fovrg_outgoing_solution, fovrg_outward_integrate,
    fovrg_overlap_integral, fovrg_potential_development, fovrg_real_product_coefficient,
    fovrg_schmidt_orthogonalize, fovrg_yk_zk_exchange, fovrg_yk_zk_transform,
};

mod derivatives;
mod overlap_exchange;
mod potentials;
mod solutions;

struct DfovrgReferenceInputs {
    exchange_cycle_count: usize,
    target_kappa: i32,
    muffin_tin_radius: Real,
    target_last_index: usize,
    energy: Complex,
    radii: Array1<Real>,
    exchange_correlation_potential: Array1<Complex>,
    valence_exchange_correlation_potential: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    valence_counts: Array1<Real>,
    kappa: Array1<i32>,
    muffin_tin_large_component: Complex,
    muffin_tin_small_component: Complex,
    irregular: bool,
    radial_match_index: usize,
    bound_orbital_count: usize,
}

impl DfovrgReferenceInputs {
    fn to_input(&self) -> FovrgDiracSolverInput<'_> {
        FovrgDiracSolverInput {
            exchange_cycle_count: self.exchange_cycle_count,
            target_kappa: self.target_kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            target_last_index: self.target_last_index,
            energy: self.energy,
            step: 0.45,
            radii: self.radii.view(),
            exchange_correlation_potential: self.exchange_correlation_potential.view(),
            valence_exchange_correlation_potential: self
                .valence_exchange_correlation_potential
                .view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            valence_counts: self.valence_counts.view(),
            kappa: self.kappa.view(),
            muffin_tin_large_component: self.muffin_tin_large_component,
            muffin_tin_small_component: self.muffin_tin_small_component,
            atomic_number: 29.0,
            irregular: self.irregular,
            c3_scale: 0,
            radial_match_index: self.radial_match_index,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn dfovrg_reference_inputs(irregular: bool) -> DfovrgReferenceInputs {
    let count = 40;
    let bound_orbitals = 3;
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (-8.8 + 0.45 * (row - 1.0)).exp()
    }));
    let exchange_correlation_potential = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(-0.16 + 0.006 * row, 0.002 * (0.31 * row).cos())
    }));
    let valence_exchange_correlation_potential = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(-0.12 + 0.004 * row, 0.001 * (0.27 * row).sin())
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.012 * orbital * (0.08 * row * orbital).sin() * (-0.010 * row).exp()
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.009 * orbital * (0.07 * row * orbital).cos() * (-0.012 * row).exp()
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.008 * row + 0.0011 * orbital * (0.19 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.005 * row + 0.0008 * orbital * (0.16 * row * orbital).sin()
    });

    if irregular {
        DfovrgReferenceInputs {
            exchange_cycle_count: 0,
            target_kappa: -1,
            muffin_tin_radius: 1.35,
            target_last_index: 16,
            energy: Complex::new(0.24, 0.035),
            radii,
            exchange_correlation_potential,
            valence_exchange_correlation_potential,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.80, 1.00, 0.70]),
            valence_counts: Array1::from_vec(vec![0.0, 0.20, 0.0]),
            kappa: Array1::from_vec(vec![-1, 1, -2]),
            muffin_tin_large_component: Complex::new(0.48, 0.06),
            muffin_tin_small_component: Complex::new(0.018, -0.009),
            irregular,
            radial_match_index: 9,
            bound_orbital_count: bound_orbitals,
        }
    } else {
        DfovrgReferenceInputs {
            exchange_cycle_count: 1,
            target_kappa: -2,
            muffin_tin_radius: 1.42,
            target_last_index: 15,
            energy: Complex::new(0.38, 0.020),
            radii,
            exchange_correlation_potential,
            valence_exchange_correlation_potential,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.80, 1.00, 0.70]),
            valence_counts: Array1::from_vec(vec![0.0, 0.20, 0.0]),
            kappa: Array1::from_vec(vec![-1, 1, -2]),
            muffin_tin_large_component: Complex::new(0.0, 0.0),
            muffin_tin_small_component: Complex::new(0.0, 0.0),
            irregular,
            radial_match_index: 9,
            bound_orbital_count: bound_orbitals,
        }
    }
}

struct SoloutReferenceInputs {
    initial_large_coefficient: Complex,
    initial_small_coefficient: Complex,
    energy: Complex,
    origin_power: Real,
    kappa: i32,
    muffin_tin_radius: Real,
    potential: Array1<Complex>,
    potential_coefficients: Array1<Complex>,
    large_exchange: Array1<Complex>,
    small_exchange: Array1<Complex>,
    large_exchange_coefficients: Array1<Complex>,
    small_exchange_coefficients: Array1<Complex>,
    c3_potential: Array1<Complex>,
    radii: Array1<Real>,
    c3_scale: i32,
    radial_match_index: usize,
    last_index: usize,
    wkb_index: usize,
    coefficient_count: usize,
    active_len: usize,
}

impl SoloutReferenceInputs {
    fn to_input(&self) -> FovrgOutgoingSolutionInput<'_> {
        FovrgOutgoingSolutionInput {
            initial_large_coefficient: self.initial_large_coefficient,
            initial_small_coefficient: self.initial_small_coefficient,
            energy: self.energy,
            origin_power: self.origin_power,
            kappa: self.kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            potential: self.potential.view(),
            potential_coefficients: self.potential_coefficients.view(),
            large_exchange: self.large_exchange.view(),
            small_exchange: self.small_exchange.view(),
            large_exchange_coefficients: self.large_exchange_coefficients.view(),
            small_exchange_coefficients: self.small_exchange_coefficients.view(),
            c3_potential: self.c3_potential.view(),
            radii: self.radii.view(),
            speed_of_light: 137.035_999_084,
            step: 0.045,
            c3_scale: self.c3_scale,
            radial_match_index: self.radial_match_index,
            last_index: self.last_index,
            wkb_index: self.wkb_index,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
        }
    }
}

fn solout_reference_inputs(case_id: usize) -> SoloutReferenceInputs {
    let active_len = 15;
    let coefficient_count = 6;
    let radii = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        0.18 * ((row - 1.0) * 0.045).exp()
    }));
    let potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
    }));
    let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
    }));
    let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));
    let large_exchange_coefficients = Array1::from_iter((1..=coefficient_count).map(|row| {
        let row = row as Real;
        Complex::new(
            0.0025 * row + 0.001 * (0.33 * row).cos(),
            -0.0015 * row + 0.0007 * (0.21 * row).sin(),
        )
    }));
    let small_exchange_coefficients = Array1::from_iter((1..=coefficient_count).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.0018 * row + 0.0008 * (0.27 * row).sin(),
            0.0012 * row + 0.0005 * (0.19 * row).cos(),
        )
    }));

    let mut potential_coefficients = Array1::<Complex>::zeros(coefficient_count);
    match case_id {
        1 => {
            potential_coefficients[0] = Complex::new(-0.21, 0.0);
            potential_coefficients[1] = Complex::new(0.013, -0.002);
            potential_coefficients[2] = Complex::new(-0.004, 0.001);
            potential_coefficients[3] = Complex::new(0.002, 0.0005);
            potential_coefficients[4] = Complex::new(-0.001, 0.0002);
            potential_coefficients[5] = Complex::new(0.0006, -0.0001);
            SoloutReferenceInputs {
                initial_large_coefficient: Complex::new(0.85, -0.13),
                initial_small_coefficient: Complex::new(-0.045, 0.018),
                energy: Complex::new(-0.42, 0.018),
                origin_power: 1.982,
                kappa: -2,
                muffin_tin_radius: 1.35,
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                large_exchange_coefficients,
                small_exchange_coefficients,
                c3_potential,
                radii,
                c3_scale: 0,
                radial_match_index: 8,
                last_index: 11,
                wkb_index: 6,
                coefficient_count,
                active_len,
            }
        }
        2 => {
            potential_coefficients[0] = Complex::new(0.11, 0.0);
            potential_coefficients[1] = Complex::new(-0.009, 0.002);
            potential_coefficients[2] = Complex::new(0.003, -0.001);
            potential_coefficients[3] = Complex::new(0.018, -0.004);
            potential_coefficients[4] = Complex::new(0.001, 0.0003);
            potential_coefficients[5] = Complex::new(-0.0004, 0.0002);
            SoloutReferenceInputs {
                initial_large_coefficient: Complex::new(-0.72, 0.21),
                initial_small_coefficient: Complex::new(0.037, -0.015),
                energy: Complex::new(0.36, -0.027),
                origin_power: 3.025,
                kappa: 3,
                muffin_tin_radius: 1.20,
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                large_exchange_coefficients,
                small_exchange_coefficients,
                c3_potential,
                radii,
                c3_scale: 1,
                radial_match_index: 9,
                last_index: 12,
                wkb_index: 7,
                coefficient_count,
                active_len,
            }
        }
        _ => {
            potential_coefficients[0] = Complex::new(-0.18, 0.0);
            potential_coefficients[1] = Complex::new(0.010, 0.001);
            potential_coefficients[2] = Complex::new(-0.003, 0.0008);
            potential_coefficients[3] = Complex::new(-0.015, 0.003);
            potential_coefficients[4] = Complex::new(0.0008, -0.0002);
            potential_coefficients[5] = Complex::new(-0.0003, 0.0001);
            SoloutReferenceInputs {
                initial_large_coefficient: Complex::new(0.64, 0.08),
                initial_small_coefficient: Complex::new(0.025, -0.011),
                energy: Complex::new(0.22, 0.041),
                origin_power: 0.965,
                kappa: -1,
                muffin_tin_radius: 1.40,
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                large_exchange_coefficients,
                small_exchange_coefficients,
                c3_potential,
                radii,
                c3_scale: 1,
                radial_match_index: 8,
                last_index: 10,
                wkb_index: 7,
                coefficient_count,
                active_len,
            }
        }
    }
}

struct SolinReferenceInputs {
    initial_large_coefficient: Complex,
    initial_small_coefficient: Complex,
    energy: Complex,
    origin_power: Real,
    kappa: i32,
    muffin_tin_radius: Real,
    potential: Array1<Complex>,
    large_exchange: Array1<Complex>,
    small_exchange: Array1<Complex>,
    c3_potential: Array1<Complex>,
    radii: Array1<Real>,
    c3_scale: i32,
    radial_match_index: usize,
    last_index: usize,
    wkb_index: usize,
    coefficient_count: usize,
    active_len: usize,
}

impl SolinReferenceInputs {
    fn to_input(&self) -> FovrgInwardSolutionInput<'_> {
        FovrgInwardSolutionInput {
            initial_large_coefficient: self.initial_large_coefficient,
            initial_small_coefficient: self.initial_small_coefficient,
            energy: self.energy,
            origin_power: self.origin_power,
            kappa: self.kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            potential: self.potential.view(),
            large_exchange: self.large_exchange.view(),
            small_exchange: self.small_exchange.view(),
            c3_potential: self.c3_potential.view(),
            radii: self.radii.view(),
            speed_of_light: 137.035_999_084,
            step: 0.045,
            c3_scale: self.c3_scale,
            radial_match_index: self.radial_match_index,
            last_index: self.last_index,
            wkb_index: self.wkb_index,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
        }
    }
}

fn solin_reference_inputs(case_id: usize) -> SolinReferenceInputs {
    let active_len = 15;
    let coefficient_count = 6;
    let radii = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        0.18 * ((row - 1.0) * 0.045).exp()
    }));
    let potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
    }));
    let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
    }));
    let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));

    match case_id {
        1 => SolinReferenceInputs {
            initial_large_coefficient: Complex::new(0.85, -0.13),
            initial_small_coefficient: Complex::new(-0.045, 0.018),
            energy: Complex::new(0.42, 0.018),
            origin_power: 1.982,
            kappa: -2,
            muffin_tin_radius: 1.35,
            potential,
            large_exchange,
            small_exchange,
            c3_potential,
            radii,
            c3_scale: 0,
            radial_match_index: 8,
            last_index: 11,
            wkb_index: 6,
            coefficient_count,
            active_len,
        },
        2 => SolinReferenceInputs {
            initial_large_coefficient: Complex::new(-0.72, 0.21),
            initial_small_coefficient: Complex::new(0.037, -0.015),
            energy: Complex::new(0.36, -0.027),
            origin_power: 3.025,
            kappa: 3,
            muffin_tin_radius: 1.20,
            potential,
            large_exchange,
            small_exchange,
            c3_potential,
            radii,
            c3_scale: 0,
            radial_match_index: 9,
            last_index: 12,
            wkb_index: 7,
            coefficient_count,
            active_len,
        },
        _ => SolinReferenceInputs {
            initial_large_coefficient: Complex::new(0.64, 0.08),
            initial_small_coefficient: Complex::new(0.025, -0.011),
            energy: Complex::new(0.22, 0.041),
            origin_power: 0.965,
            kappa: -1,
            muffin_tin_radius: 1.40,
            potential,
            large_exchange,
            small_exchange,
            c3_potential,
            radii,
            c3_scale: 0,
            radial_match_index: 8,
            last_index: 12,
            wkb_index: 7,
            coefficient_count,
            active_len,
        },
    }
}

struct WfirdcReferenceInputs {
    energy: Complex,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    orbital_lengths: Array1<usize>,
    exchange_correlation_potential: Array1<Complex>,
    c3_potential: Array1<Complex>,
    initial_large_coefficient: Complex,
    initial_small_coefficient: Complex,
    muffin_tin_radius: Real,
    c3_scale: i32,
    irregular: bool,
    radial_match_index: usize,
    wkb_index: usize,
    coefficient_count: usize,
    active_len: usize,
}

impl WfirdcReferenceInputs {
    fn to_input(&self) -> FovrgInitialPhotoelectronInput<'_> {
        FovrgInitialPhotoelectronInput {
            energy: self.energy,
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            orbital_lengths: self.orbital_lengths.view(),
            exchange_correlation_potential: self.exchange_correlation_potential.view(),
            c3_potential: self.c3_potential.view(),
            initial_large_coefficient: self.initial_large_coefficient,
            initial_small_coefficient: self.initial_small_coefficient,
            nuclear_charge: 29.0,
            muffin_tin_radius: self.muffin_tin_radius,
            step: 0.045,
            speed_of_light: 137.0373,
            c3_scale: self.c3_scale,
            irregular: self.irregular,
            radial_match_index: self.radial_match_index,
            wkb_index: self.wkb_index,
            coefficient_count: self.coefficient_count,
            orbital_count: 3,
            active_len: self.active_len,
        }
    }
}

fn wfirdc_reference_inputs(case_id: usize) -> WfirdcReferenceInputs {
    let active_len = 15;
    let bound_orbitals = 2;
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

    if case_id == 1 {
        WfirdcReferenceInputs {
            energy: Complex::new(0.42, 0.018),
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.25, 0.65]),
            kappa: Array1::from_vec(vec![-1, 1, -2]),
            orbital_lengths: Array1::from_vec(vec![0, 0, 12]),
            exchange_correlation_potential,
            c3_potential,
            initial_large_coefficient: Complex::new(0.0, 0.0),
            initial_small_coefficient: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.35,
            c3_scale: 0,
            irregular: false,
            radial_match_index: 8,
            wkb_index: 6,
            coefficient_count: 3,
            active_len,
        }
    } else {
        WfirdcReferenceInputs {
            energy: Complex::new(0.22, 0.041),
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.25, 0.65]),
            kappa: Array1::from_vec(vec![-1, 1, -1]),
            orbital_lengths: Array1::from_vec(vec![0, 0, 13]),
            exchange_correlation_potential,
            c3_potential,
            initial_large_coefficient: Complex::new(0.64, 0.08),
            initial_small_coefficient: Complex::new(0.025, -0.011),
            muffin_tin_radius: 1.40,
            c3_scale: 0,
            irregular: true,
            radial_match_index: 8,
            wkb_index: 7,
            coefficient_count: 2,
            active_len,
        }
    }
}

struct IntoutReferenceInputs {
    initial_large_component: Complex,
    initial_small_component: Complex,
    energy: Complex,
    potential: Array1<Complex>,
    potential_coefficients: Array1<Complex>,
    large_exchange: Array1<Complex>,
    small_exchange: Array1<Complex>,
    c3_potential: Array1<Complex>,
    radii: Array1<Real>,
    kappa: i32,
    c3_scale: i32,
    start_index: usize,
    last_index: usize,
    active_len: usize,
}

impl IntoutReferenceInputs {
    fn to_input(&self) -> FovrgOutwardIntegrationInput<'_> {
        FovrgOutwardIntegrationInput {
            initial_large_component: self.initial_large_component,
            initial_small_component: self.initial_small_component,
            energy: self.energy,
            potential: self.potential.view(),
            potential_coefficients: self.potential_coefficients.view(),
            large_exchange: self.large_exchange.view(),
            small_exchange: self.small_exchange.view(),
            c3_potential: self.c3_potential.view(),
            radii: self.radii.view(),
            speed_of_light: 137.035_999_084,
            step: 0.045,
            kappa: self.kappa,
            c3_scale: self.c3_scale,
            start_index: self.start_index,
            last_index: self.last_index,
            active_len: self.active_len,
        }
    }
}

fn intout_reference_inputs(case_id: usize) -> IntoutReferenceInputs {
    let active_len = 15;
    let mut potential_coefficients = Array1::<Complex>::zeros(10);
    let radii = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        0.18 * ((row - 1.0) * 0.045).exp()
    }));
    let potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
    }));
    let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
    }));
    let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));

    match case_id {
        1 => {
            potential_coefficients[0] = Complex::new(-0.21, 0.0);
            IntoutReferenceInputs {
                initial_large_component: Complex::new(0.035, -0.012),
                initial_small_component: Complex::new(-0.009, 0.004),
                energy: Complex::new(-0.42, 0.018),
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                kappa: -2,
                c3_scale: 0,
                start_index: 0,
                last_index: 11,
                active_len,
            }
        }
        2 => {
            potential_coefficients[0] = Complex::new(0.11, 0.0);
            potential_coefficients[3] = Complex::new(0.018, -0.004);
            IntoutReferenceInputs {
                initial_large_component: Complex::new(-0.017, 0.028),
                initial_small_component: Complex::new(0.011, -0.006),
                energy: Complex::new(0.36, -0.027),
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                kappa: 3,
                c3_scale: 1,
                start_index: 0,
                last_index: 12,
                active_len,
            }
        }
        _ => {
            potential_coefficients[0] = Complex::new(0.09, 0.0);
            potential_coefficients[3] = Complex::new(-0.015, 0.003);
            IntoutReferenceInputs {
                initial_large_component: Complex::new(0.026, 0.014),
                initial_small_component: Complex::new(-0.008, 0.017),
                energy: Complex::new(0.22, 0.041),
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                kappa: -1,
                c3_scale: 1,
                start_index: 3,
                last_index: 10,
                active_len,
            }
        }
    }
}

fn diff_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Real>) {
    let potential = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        Complex::new(
            (0.21 * index).sin() + 0.03 * index,
            (0.17 * index).cos() - 0.02 * index,
        )
    }));
    let radii = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        0.15 + 0.04 * index + 0.001 * index * index
    }));
    (potential, radii)
}

fn aprd_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Complex>) {
    let real_left = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.02 * row + (0.03 * row * 2.0).cos()
    }));
    let real_right = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        -0.015 * row + (0.025 * row * 3.0).sin()
    }));
    let complex_left = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    (real_left, real_right, complex_left)
}

fn muatcc_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<i32>) {
    (
        Array1::from_vec(vec![2.0, 1.5, 2.5, 1.0, 3.0]),
        Array1::from_vec(vec![0.0, 0.25, -0.10, 0.0, -0.20]),
        Array1::from_vec(vec![-1, 1, -2, 2, -3]),
    )
}

fn yzktec_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Complex>, Array1<Real>) {
    let step = 0.0725;
    let source = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        Complex::new(
            (0.19 * index).sin() + 0.02 * index,
            (0.11 * index).cos() - 0.03 * index,
        )
    }));
    let coefficients = Array1::from_iter((1..=10).map(|index| {
        let index = index as Real;
        Complex::new(
            0.04 * index + (0.13 * index).cos(),
            -0.03 * index + (0.17 * index).sin(),
        )
    }));
    let radii = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        0.018 * (step * (index - 1.0)).exp()
    }));
    (source, coefficients, radii)
}

struct YzkrdcReferenceInputs {
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
    partner_large_component: Array1<Complex>,
    partner_small_component: Array1<Complex>,
    partner_large_coefficients: Array1<Complex>,
    partner_small_coefficients: Array1<Complex>,
    radii: Array1<Real>,
    orbital_power: Real,
    partner_power: Real,
    step: Real,
    angular_momentum: usize,
    coefficient_count: usize,
    orbital_len: usize,
    source_len: usize,
    active_len: usize,
}

impl YzkrdcReferenceInputs {
    fn as_exchange_input(&self) -> FovrgYkZkExchangeInput<'_> {
        FovrgYkZkExchangeInput {
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            partner_large_component: self.partner_large_component.view(),
            partner_small_component: self.partner_small_component.view(),
            partner_large_coefficients: self.partner_large_coefficients.view(),
            partner_small_coefficients: self.partner_small_coefficients.view(),
            radii: self.radii.view(),
            orbital_power: self.orbital_power,
            partner_power: self.partner_power,
            step: self.step,
            angular_momentum: self.angular_momentum,
            coefficient_count: self.coefficient_count,
            orbital_len: self.orbital_len,
            source_len: self.source_len,
            active_len: self.active_len,
        }
    }
}

fn yzkrdc_reference_inputs(count: usize) -> YzkrdcReferenceInputs {
    let step = 0.0725;
    let orbital_column = 2.0;
    let large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.05 * row * orbital_column).sin() + 0.001 * (row + orbital_column)
    }));
    let small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.04 * row * orbital_column).cos() - 0.002 * (row - orbital_column)
    }));
    let large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        0.02 * row + (0.03 * row * orbital_column).cos()
    }));
    let small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        -0.015 * row + (0.025 * row * orbital_column).sin()
    }));
    let partner_large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.19 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let partner_small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.07 * row).cos() - 0.01 * row,
            (0.23 * row).sin() + 0.015 * row,
        )
    }));
    let partner_large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let partner_small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    YzkrdcReferenceInputs {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        partner_large_component,
        partner_small_component,
        partner_large_coefficients,
        partner_small_coefficients,
        radii,
        orbital_power: 0.65 + 0.08 * orbital_column,
        partner_power: 1.35,
        step,
        angular_momentum: 2,
        coefficient_count: 6,
        orbital_len: 9,
        source_len: 9,
        active_len: count,
    }
}

struct DsordcReferenceInputs {
    large_integrand: Array1<Complex>,
    small_integrand: Array1<Complex>,
    large_integrand_coefficients: Array1<Complex>,
    small_integrand_coefficients: Array1<Complex>,
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
    radii: Array1<Real>,
    integrand_power: Real,
    orbital_power: Real,
    step: Real,
    coefficient_count: usize,
    active_len: usize,
}

impl DsordcReferenceInputs {
    fn as_overlap_input(&self) -> FovrgOverlapIntegralInput<'_> {
        FovrgOverlapIntegralInput {
            large_integrand: self.large_integrand.view(),
            small_integrand: self.small_integrand.view(),
            large_integrand_coefficients: self.large_integrand_coefficients.view(),
            small_integrand_coefficients: self.small_integrand_coefficients.view(),
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            radii: self.radii.view(),
            integrand_power: self.integrand_power,
            orbital_power: self.orbital_power,
            step: self.step,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
        }
    }
}

fn dsordc_reference_inputs(count: usize) -> DsordcReferenceInputs {
    let step = 0.0725;
    let orbital = 3.0;
    let large_integrand = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.17 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let small_integrand = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.09 * row).cos() - 0.01 * row,
            (0.21 * row).sin() + 0.015 * row,
        )
    }));
    let large_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let small_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
    }));
    let small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
    }));
    let large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    }));
    let small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    }));
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    DsordcReferenceInputs {
        large_integrand,
        small_integrand,
        large_integrand_coefficients,
        small_integrand_coefficients,
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        radii,
        integrand_power: 1.35,
        orbital_power: 0.45 + 0.06 * orbital,
        step,
        coefficient_count: 6,
        active_len: count,
    }
}

struct OrtdacReferenceInputs {
    target_large_component: Array1<Complex>,
    target_small_component: Array1<Complex>,
    target_large_coefficients: Array1<Complex>,
    target_small_coefficients: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    orbital_powers: Array1<Real>,
    radii: Array1<Real>,
    target_power: Real,
    target_kappa: i32,
    step: Real,
    coefficient_count: usize,
    active_len: usize,
    bound_orbital_count: usize,
}

impl OrtdacReferenceInputs {
    fn as_orthogonalization_input(&self) -> FovrgOrthogonalizationInput<'_> {
        FovrgOrthogonalizationInput {
            target_large_component: self.target_large_component.view(),
            target_small_component: self.target_small_component.view(),
            target_large_coefficients: self.target_large_coefficients.view(),
            target_small_coefficients: self.target_small_coefficients.view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            orbital_powers: self.orbital_powers.view(),
            radii: self.radii.view(),
            target_power: self.target_power,
            target_kappa: self.target_kappa,
            step: self.step,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn ortdac_reference_inputs(count: usize) -> OrtdacReferenceInputs {
    let step = 0.0725;
    let bound_orbitals = 4;
    let target_large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.17 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let target_small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.09 * row).cos() - 0.01 * row,
            (0.21 * row).sin() + 0.015 * row,
        )
    }));
    let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    });
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    OrtdacReferenceInputs {
        target_large_component,
        target_small_component,
        target_large_coefficients,
        target_small_coefficients,
        bound_large_components,
        bound_small_components,
        bound_large_coefficients,
        bound_small_coefficients,
        electron_counts: Array1::from_vec(vec![1.2, 1.4, 0.0, 2.0]),
        kappa: Array1::from_vec(vec![-2, 1, -2, -2]),
        orbital_powers: Array1::from_iter((1..=bound_orbitals).map(|orbital| {
            let orbital = orbital as Real;
            0.45 + 0.06 * orbital
        })),
        radii,
        target_power: 0.45 + 0.06 * 5.0,
        target_kappa: -2,
        step,
        coefficient_count: 6,
        active_len: count,
        bound_orbital_count: bound_orbitals,
    }
}

struct PotexReferenceInputs {
    target_large_component: Array1<Complex>,
    target_small_component: Array1<Complex>,
    target_large_coefficients: Array1<Complex>,
    target_small_coefficients: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    angular_coefficients: Array2<Real>,
    orbital_powers: Array1<Real>,
    kappa: Array1<i32>,
    orbital_lengths: Array1<usize>,
    normalization: Array1<Real>,
    radii: Array1<Real>,
    target_power: Real,
    target_kappa: i32,
    target_normalization: Real,
    speed_of_light: Real,
    step: Real,
    coefficient_count: usize,
    source_len: usize,
    active_len: usize,
    radial_output_count: usize,
    bound_orbital_count: usize,
}

impl PotexReferenceInputs {
    fn as_exchange_potential_input(&self) -> FovrgExchangePotentialInput<'_> {
        FovrgExchangePotentialInput {
            target_large_component: self.target_large_component.view(),
            target_small_component: self.target_small_component.view(),
            target_large_coefficients: self.target_large_coefficients.view(),
            target_small_coefficients: self.target_small_coefficients.view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            angular_coefficients: self.angular_coefficients.view(),
            orbital_powers: self.orbital_powers.view(),
            kappa: self.kappa.view(),
            orbital_lengths: self.orbital_lengths.view(),
            normalization: self.normalization.view(),
            radii: self.radii.view(),
            target_power: self.target_power,
            target_kappa: self.target_kappa,
            target_normalization: self.target_normalization,
            speed_of_light: self.speed_of_light,
            step: self.step,
            coefficient_count: self.coefficient_count,
            source_len: self.source_len,
            active_len: self.active_len,
            radial_output_count: self.radial_output_count,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn potex_reference_inputs(count: usize) -> PotexReferenceInputs {
    let step = 0.0725;
    let bound_orbitals = 4;
    let target_large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.17 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let target_small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.09 * row).cos() - 0.01 * row,
            (0.21 * row).sin() + 0.015 * row,
        )
    }));
    let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    });
    let mut angular_coefficients = Array2::zeros((bound_orbitals, 5));
    angular_coefficients[(0, 0)] = 0.31;
    angular_coefficients[(1, 0)] = -0.18;
    angular_coefficients[(2, 0)] = 0.27;
    angular_coefficients[(2, 1)] = -0.11;
    angular_coefficients[(3, 0)] = 0.19;
    angular_coefficients[(3, 1)] = 0.07;
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    PotexReferenceInputs {
        target_large_component,
        target_small_component,
        target_large_coefficients,
        target_small_coefficients,
        bound_large_components,
        bound_small_components,
        bound_large_coefficients,
        bound_small_coefficients,
        angular_coefficients,
        orbital_powers: Array1::from_vec(vec![0.51, 0.57, 0.63, 0.69]),
        kappa: Array1::from_vec(vec![-1, 1, -2, 2]),
        orbital_lengths: Array1::from_vec(vec![9, 8, 7, 9]),
        normalization: Array1::from_vec(vec![1.01, 1.02, 1.03, 1.04]),
        radii,
        target_power: 0.75,
        target_kappa: -2,
        target_normalization: 1.08,
        speed_of_light: 137.035_999_084,
        step,
        coefficient_count: 6,
        source_len: 9,
        active_len: count,
        radial_output_count: 7,
        bound_orbital_count: bound_orbitals,
    }
}

struct PotdvpReferenceInputs {
    nuclear_coefficients: Array1<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    normalization: Array1<Real>,
    radii: Array1<Real>,
    speed_of_light: Real,
    coefficient_count: usize,
    orbital_count: usize,
}

impl PotdvpReferenceInputs {
    fn as_potential_input(&self) -> FovrgPotentialDevelopmentInput<'_> {
        FovrgPotentialDevelopmentInput {
            nuclear_coefficients: self.nuclear_coefficients.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            normalization: self.normalization.view(),
            radii: self.radii.view(),
            speed_of_light: self.speed_of_light,
            coefficient_count: self.coefficient_count,
            orbital_count: self.orbital_count,
        }
    }
}

fn potdvp_reference_inputs(count: usize) -> PotdvpReferenceInputs {
    let step = 0.0725;
    let bound_orbitals = 4;
    let large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    });
    let small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    });
    let nuclear_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        -0.35 + 0.045 * row + 0.002 * row * row
    }));
    let electron_counts = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
        let orbital = orbital as Real;
        0.45 * orbital + 0.1
    }));
    let kappa = Array1::from_vec(vec![-1, 1, -2, 3]);
    let normalization = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
        let orbital = orbital as Real;
        1.0 + 0.013 * orbital
    }));
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    PotdvpReferenceInputs {
        nuclear_coefficients,
        large_coefficients,
        small_coefficients,
        electron_counts,
        kappa,
        normalization,
        radii,
        speed_of_light: 137.035_999_084,
        coefficient_count: 8,
        orbital_count: 5,
    }
}

fn assert_complex_close(actual: Complex, expected_re: Real, expected_im: Real, tolerance: Real) {
    assert_close(actual.re, expected_re, tolerance);
    assert_close(actual.im, expected_im, tolerance);
}

fn assert_real_matrix_close<const ROWS: usize, const COLS: usize>(
    actual: &Array2<Real>,
    expected: &[[Real; COLS]; ROWS],
    tolerance: Real,
) {
    assert_eq!(actual.shape(), &[ROWS, COLS]);
    for row in 0..ROWS {
        for column in 0..COLS {
            assert_close(actual[(row, column)], expected[row][column], tolerance);
        }
    }
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}
