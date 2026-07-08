use super::*;
use crate::FEFF_HARTREE_EV;
use crate::fovrg::FovrgDiracSolverInput;
use crate::quadrature::csomm2;
use crate::rhorrp::{
    RhorrpError, RhorrpRadialSolutionAssemblyInput, RhorrpWavefunctionChannelInput,
    RhorrpWavefunctionTables, rhorrp_assemble_radial_solutions, rhorrp_wavefunction_channel,
};
use ndarray::{Array1, Array2, Array3, Array4, Axis};

#[test]
fn valence_density_update_matches_feff_ff2g_first_energy_reference() -> Result<(), DensityError> {
    let sample = sample_ff2g_state();

    let result = update_valence_density(ValenceDensityUpdateInput {
        scattering_trace: sample.scattering_trace.view(),
        potential_index: 1,
        energy_index: 1,
        last_radial_index: 5,
        scattering_ldos: sample.scattering_ldos.view(),
        embedded_ldos: sample.embedded_ldos.view(),
        previous_ldos: sample.previous_ldos.view(),
        scattering_density: sample.scattering_density.view(),
        embedded_density: sample.embedded_density.view(),
        previous_density: sample.previous_density.view(),
        valence_density: sample.valence_density.view(),
        occupancy_by_l: sample.occupancy_by_l.view(),
        current_energy: Complex::new(0.72, 0.11),
        previous_energy: Complex::new(0.61, -0.04),
        potential_multiplicity: 2.5,
        current_floor: 1,
        previous_floor: 0,
        left_sum: Complex::new(0.2, -0.1),
        right_sum: Complex::new(-0.3, 0.25),
        total_electron_count: 1.25,
        include_high_l: false,
    })?;

    assert_complex_close(
        result.embedded_ldos[(0, 1)],
        Complex::new(0.451_099_999_919_533_76, -0.215_299_999_862_909_35),
    );
    assert_complex_close(
        result.embedded_ldos[(2, 1)],
        Complex::new(0.539_700_002_484_023_6, -0.211_100_000_292_062_77),
    );
    assert_complex_close(
        result.embedded_ldos[(3, 1)],
        Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
    );
    assert_complex_close(result.previous_ldos[(2, 1)], result.embedded_ldos[(2, 1)]);
    assert_complex_close(
        result.embedded_density[0],
        Complex::new(6.406_000_025_570_392e-2, -1.129_999_976_605_176_9e-2),
    );
    assert_complex_close(
        result.embedded_density[4],
        Complex::new(2.775_000_003_725_290_3e-1, -9.609_999_980_777_503e-2),
    );
    assert_complex_close(result.previous_density[4], result.embedded_density[4]);
    assert_close(result.valence_density[0], 1.263_880_007_192_492_5e-2);
    assert_close(result.valence_density[3], 4.145_320_007_205_01e-2);
    assert_close(result.valence_density[4], 5.105_800_007_209_182e-2);
    assert_close(result.occupancy_by_l[0], -4.127_799_997_627_735e-2);
    assert_close(result.occupancy_by_l[2], -3.265_999_865_531_922_5e-3);
    assert_close(result.occupancy_by_l[3], 1.5e-2);
    assert_close(result.total_electron_count, 1.082_550_000_302_493_7);
    assert_complex_close(
        result.left_sum,
        Complex::new(7.618_000_007_234_514, -3.296_999_999_880_791),
    );
    assert_complex_close(
        result.right_sum,
        Complex::new(7.118_000_007_234_514, -2.946_999_999_880_791_4),
    );
    Ok(())
}

#[test]
fn valence_density_update_matches_feff_ff2g_high_l_reference() -> Result<(), DensityError> {
    let sample = sample_ff2g_state();

    let result = update_valence_density(ValenceDensityUpdateInput {
        scattering_trace: sample.scattering_trace.view(),
        potential_index: 1,
        energy_index: 2,
        last_radial_index: 4,
        scattering_ldos: sample.scattering_ldos.view(),
        embedded_ldos: sample.embedded_ldos.view(),
        previous_ldos: sample.previous_ldos.view(),
        scattering_density: sample.scattering_density.view(),
        embedded_density: sample.embedded_density.view(),
        previous_density: sample.previous_density.view(),
        valence_density: sample.valence_density.view(),
        occupancy_by_l: sample.occupancy_by_l.view(),
        current_energy: Complex::new(0.91, -0.08),
        previous_energy: Complex::new(0.77, 0.05),
        potential_multiplicity: 1.75,
        current_floor: 0,
        previous_floor: 1,
        left_sum: Complex::new(-0.15, 0.09),
        right_sum: Complex::new(0.05, -0.12),
        total_electron_count: -0.2,
        include_high_l: true,
    })?;

    assert_complex_close(
        result.embedded_ldos[(3, 1)],
        Complex::new(0.591_799_997_240_305, -0.209_599_997_997_283_93),
    );
    assert_complex_close(result.previous_ldos[(2, 1)], Complex::new(-4.0e-2, 4.5e-2));
    assert_complex_close(
        result.embedded_density[0],
        Complex::new(8.203_999_945_521_354e-2, -1.959_999_881_684_781e-3),
    );
    assert_complex_close(result.embedded_density[4], Complex::new(2.5e-1, -1.0e-1));
    assert_complex_close(result.previous_density[4], Complex::new(-1.5e-1, 2.0e-1));
    assert_close(result.valence_density[0], 5.560_400_087_386_373e-3);
    assert_close(result.valence_density[3], 2.428_160_011_395_812e-2);
    assert_close(result.valence_density[4], 5.0e-2);
    assert_close(result.occupancy_by_l[0], -1.041_849_999_703_466_9e-1);
    assert_close(result.occupancy_by_l[2], -9.221_500_036_381_186e-2);
    assert_close(result.occupancy_by_l[3], -8.732_799_936_085_94e-2);
    assert_close(result.total_electron_count, -8.677_334_992_048_331e-1);
    assert_complex_close(result.left_sum, Complex::new(-8.85e-1, 8.6e-1));
    assert_complex_close(
        result.right_sum,
        Complex::new(7.313_899_995_405_228, -3.091_499_992_907_047_5),
    );
    Ok(())
}

#[test]
fn pot_scf_energy_point_accumulates_feff_ff2g_over_potentials() -> Result<(), DensityError> {
    let angular_count = 4;
    let potential_count = 2;
    let radial_count = 6;
    let last_indices = Array1::from_vec(vec![5, 4]);
    let potential_multiplicities = Array1::from_vec(vec![1.5, 2.25]);
    let scattering_trace =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, potential)| {
            Complex32::new(
                (0.08 + 0.03 * angular as Real - 0.01 * potential as Real) as f32,
                (-0.04 + 0.02 * angular as Real + 0.015 * potential as Real) as f32,
            )
        });
    let scattering_ldos =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, potential)| {
            Complex::new(
                0.18 + 0.025 * angular as Real + 0.02 * potential as Real,
                -0.12 + 0.018 * angular as Real - 0.015 * potential as Real,
            )
        });
    let embedded_ldos =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, potential)| {
            Complex::new(
                0.32 + 0.04 * angular as Real + 0.03 * potential as Real,
                -0.18 + 0.02 * angular as Real - 0.012 * potential as Real,
            )
        });
    let previous_ldos =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, potential)| {
            Complex::new(
                0.21 + 0.03 * angular as Real + 0.01 * potential as Real,
                -0.13 + 0.015 * angular as Real - 0.008 * potential as Real,
            )
        });
    let scattering_density = Array3::from_shape_fn(
        (radial_count, angular_count, potential_count),
        |(r, l, p)| {
            Complex::new(
                0.006 * (r + 1) as Real + 0.012 * l as Real + 0.004 * p as Real,
                -0.004 * (r + 1) as Real + 0.009 * l as Real - 0.003 * p as Real,
            )
        },
    );
    let embedded_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            Complex::new(
                0.04 + 0.012 * radial as Real + 0.006 * potential as Real,
                -0.025 + 0.004 * radial as Real - 0.003 * potential as Real,
            )
        });
    let previous_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            Complex::new(
                0.03 + 0.009 * radial as Real + 0.004 * potential as Real,
                -0.02 + 0.003 * radial as Real - 0.002 * potential as Real,
            )
        });
    let valence_density = Array2::from_shape_fn((radial_count, potential_count), |(radial, p)| {
        0.02 + 0.01 * radial as Real + 0.015 * p as Real
    });
    let occupancy_by_l = Array2::from_shape_fn((angular_count, potential_count), |(l, p)| {
        -0.02 + 0.012 * l as Real + 0.01 * p as Real
    });
    let current_energy = Complex::new(0.62, 0.04);
    let previous_energy = Complex::new(0.55, 0.03);

    let accumulated = accumulate_pot_scf_energy_point(PotScfEnergyPointInput {
        energy_index: 2,
        current_energy,
        previous_energy,
        current_floor: 2,
        previous_floor: 1,
        highest_potential_index: potential_count - 1,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.view(),
        scattering_ldos: scattering_ldos.view(),
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: previous_ldos.view(),
        scattering_density: scattering_density.view(),
        embedded_density: embedded_density.view(),
        previous_density: previous_density.view(),
        valence_density: valence_density.view(),
        occupancy_by_l: occupancy_by_l.view(),
        include_high_l: false,
    })?;

    let mut expected_embedded_ldos = embedded_ldos.clone();
    let mut expected_previous_ldos = previous_ldos.clone();
    let mut expected_embedded_density = embedded_density.clone();
    let mut expected_previous_density = previous_density.clone();
    let mut expected_valence_density = valence_density.clone();
    let mut expected_occupancy = occupancy_by_l.clone();
    let mut expected_total = 0.0;
    let mut expected_left = Complex::new(0.0, 0.0);
    let mut expected_right = Complex::new(0.0, 0.0);
    for potential in 0..potential_count {
        let update = update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: scattering_trace.index_axis(Axis(1), potential),
            potential_index: potential,
            energy_index: 2,
            last_radial_index: last_indices[potential],
            scattering_ldos: scattering_ldos.index_axis(Axis(1), potential),
            embedded_ldos: expected_embedded_ldos.view(),
            previous_ldos: expected_previous_ldos.view(),
            scattering_density: scattering_density.index_axis(Axis(2), potential),
            embedded_density: expected_embedded_density.index_axis(Axis(1), potential),
            previous_density: expected_previous_density.index_axis(Axis(1), potential),
            valence_density: expected_valence_density.index_axis(Axis(1), potential),
            occupancy_by_l: expected_occupancy.index_axis(Axis(1), potential),
            current_energy,
            previous_energy,
            potential_multiplicity: potential_multiplicities[potential],
            current_floor: 2,
            previous_floor: 1,
            left_sum: expected_left,
            right_sum: expected_right,
            total_electron_count: expected_total,
            include_high_l: false,
        })?;
        expected_embedded_ldos = update.embedded_ldos;
        expected_previous_ldos = update.previous_ldos;
        expected_embedded_density
            .index_axis_mut(Axis(1), potential)
            .assign(&update.embedded_density);
        expected_previous_density
            .index_axis_mut(Axis(1), potential)
            .assign(&update.previous_density);
        expected_valence_density
            .index_axis_mut(Axis(1), potential)
            .assign(&update.valence_density);
        expected_occupancy
            .index_axis_mut(Axis(1), potential)
            .assign(&update.occupancy_by_l);
        expected_total = update.total_electron_count;
        expected_left = update.left_sum;
        expected_right = update.right_sum;
    }

    assert_eq!(accumulated.embedded_ldos, expected_embedded_ldos);
    assert_eq!(accumulated.previous_ldos, expected_previous_ldos);
    assert_eq!(accumulated.embedded_density, expected_embedded_density);
    assert_eq!(accumulated.previous_density, expected_previous_density);
    assert_eq!(accumulated.valence_density, expected_valence_density);
    assert_eq!(accumulated.occupancy_by_l, expected_occupancy);
    assert_close(accumulated.total_electron_count, expected_total);
    assert_complex_close(accumulated.left_sum, expected_left);
    assert_complex_close(accumulated.right_sum, expected_right);
    Ok(())
}

#[test]
fn pot_scf_contour_run_brackets_and_finishes_endpoint() -> Result<(), DensityError> {
    let angular_count = 3;
    let potential_count = 1;
    let radial_count = 3;
    let point_count = 2;
    let energy_grid = Array1::from_vec(vec![Complex::new(0.20, 0.01)]);
    let steps = Array1::from_vec(vec![0.01]);
    let source_energies = Array1::from_vec(vec![
        energy_grid[0],
        energy_grid[0] - Complex::new(steps[0], 0.0),
    ]);
    let last_indices = Array1::from_vec(vec![radial_count]);
    let potential_multiplicities = Array1::from_vec(vec![1.0]);
    let scattering_trace =
        Array3::<Complex32>::zeros((point_count, angular_count, potential_count));
    let scattering_ldos = Array3::<Complex>::zeros((point_count, angular_count, potential_count));
    let mut embedded_ldos_source =
        Array3::<Complex>::zeros((point_count, angular_count, potential_count));
    let scattering_density =
        Array4::<Complex>::zeros((point_count, radial_count, angular_count, potential_count));
    let mut embedded_density_source =
        Array3::<Complex>::zeros((point_count, radial_count, potential_count));

    for angular in 0..angular_count {
        embedded_ldos_source[(0, angular, 0)] = Complex::new(1.0 + 0.10 * angular as Real, 0.0);
        embedded_ldos_source[(1, angular, 0)] = Complex::new(2.0 + 0.20 * angular as Real, 0.0);
    }
    for radial in 0..radial_count {
        embedded_density_source[(0, radial, 0)] = Complex::new(0.10 + 0.02 * radial as Real, 0.0);
        embedded_density_source[(1, radial, 0)] = Complex::new(0.20 + 0.04 * radial as Real, 0.0);
    }

    let mut embedded_ldos = Array2::<Complex>::zeros((angular_count, potential_count));
    embedded_ldos.assign(&embedded_ldos_source.index_axis(Axis(0), 0));
    let mut embedded_density = Array2::<Complex>::zeros((radial_count, potential_count));
    embedded_density.assign(&embedded_density_source.index_axis(Axis(0), 0));
    let point1 = accumulate_pot_scf_energy_point(PotScfEnergyPointInput {
        energy_index: 1,
        current_energy: source_energies[0],
        previous_energy: Complex::new(source_energies[0].re, 0.0),
        current_floor: 1,
        previous_floor: 1,
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.index_axis(Axis(0), 0),
        scattering_ldos: scattering_ldos.index_axis(Axis(0), 0),
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: Array2::<Complex>::zeros((angular_count, potential_count)).view(),
        scattering_density: scattering_density.index_axis(Axis(0), 0),
        embedded_density: embedded_density.view(),
        previous_density: Array2::<Complex>::zeros((radial_count, potential_count)).view(),
        valence_density: Array2::<Real>::zeros((radial_count, potential_count)).view(),
        occupancy_by_l: Array2::<Real>::zeros((angular_count, potential_count)).view(),
        include_high_l: false,
    })?;

    embedded_ldos.assign(&embedded_ldos_source.index_axis(Axis(0), 1));
    embedded_density.assign(&embedded_density_source.index_axis(Axis(0), 1));
    let point2 = accumulate_pot_scf_energy_point(PotScfEnergyPointInput {
        energy_index: 2,
        current_energy: source_energies[1],
        previous_energy: source_energies[0],
        current_floor: 1,
        previous_floor: 1,
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.index_axis(Axis(0), 1),
        scattering_ldos: scattering_ldos.index_axis(Axis(0), 1),
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: point1.embedded_ldos.view(),
        scattering_density: scattering_density.index_axis(Axis(0), 1),
        embedded_density: embedded_density.view(),
        previous_density: point1.embedded_density.view(),
        valence_density: point1.valence_density.view(),
        occupancy_by_l: point1.occupancy_by_l.view(),
        include_high_l: false,
    })?;
    let electron_count_target = (point1.total_electron_count + point2.total_electron_count) / 2.0;
    assert!(point1.total_electron_count > electron_count_target);
    assert!(point2.total_electron_count < electron_count_target);

    let endpoint = finish_pot_scf_fermi_endpoint(PotScfFermiEndpointInput {
        current_energy: source_energies[1],
        previous_energy: source_energies[0],
        current_electron_delta: point2.total_electron_count - electron_count_target,
        previous_electron_delta: point1.total_electron_count - electron_count_target,
        left_sum: point2.left_sum,
        right_sum: point2.right_sum,
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        embedded_ldos: point2.embedded_ldos.view(),
        previous_ldos: point2.previous_ldos.view(),
        embedded_density: point2.embedded_density.view(),
        previous_density: point2.previous_density.view(),
        valence_density: point2.valence_density.view(),
        occupancy_by_l: point2.occupancy_by_l.view(),
        include_high_l: false,
    })?;

    let run = run_pot_scf_contour(PotScfContourRunInput {
        first_scmt_call: false,
        electron_count_target,
        active_energy_count: 1,
        floor_count: 1,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        source_energies: source_energies.view(),
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.view(),
        scattering_ldos: scattering_ldos.view(),
        embedded_ldos_source: embedded_ldos_source.view(),
        scattering_density: scattering_density.view(),
        embedded_density_source: embedded_density_source.view(),
        include_high_l: false,
    })?;

    assert_eq!(run.status, PotScfContourRunStatus::Bracketed);
    assert_eq!(run.energy_points_used, 2);
    assert_complex_close(run.current_energy, source_energies[1]);
    assert_complex_close(run.previous_energy, source_energies[0]);
    assert_eq!(run.fermi_energy, Some(endpoint.fermi_energy));
    assert_eq!(
        run.interpolation_fraction,
        Some(endpoint.interpolation_fraction)
    );
    assert_eq!(run.embedded_ldos, point2.embedded_ldos);
    assert_eq!(run.previous_ldos, point2.previous_ldos);
    assert_eq!(run.embedded_density, point2.embedded_density);
    assert_eq!(run.previous_density, point2.previous_density);
    assert_eq!(run.valence_density, endpoint.valence_density);
    assert_eq!(run.occupancy_by_l, endpoint.occupancy_by_l);
    Ok(())
}

#[test]
fn pot_scf_contour_run_requests_more_first_call_source_points() -> Result<(), DensityError> {
    let angular_count = 2;
    let radial_count = 2;
    let energy_grid = Array1::from_vec(vec![
        Complex::new(0.10, 0.04),
        Complex::new(0.18, 0.04),
        Complex::new(0.26, 0.04),
    ]);
    let steps = Array1::from_vec(vec![0.02, 0.04]);
    let source_energies = Array1::from_vec(vec![energy_grid[0]]);
    let last_indices = Array1::from_vec(vec![radial_count]);
    let potential_multiplicities = Array1::from_vec(vec![1.0]);
    let scattering_trace = Array3::<Complex32>::zeros((1, angular_count, 1));
    let scattering_ldos = Array3::<Complex>::zeros((1, angular_count, 1));
    let embedded_ldos_source = Array3::from_shape_fn((1, angular_count, 1), |(_, l, _)| {
        Complex::new(0.4 + 0.1 * l as Real, -0.02)
    });
    let scattering_density = Array4::<Complex>::zeros((1, radial_count, angular_count, 1));
    let embedded_density_source = Array3::from_shape_fn((1, radial_count, 1), |(_, r, _)| {
        Complex::new(0.05 + 0.01 * r as Real, -0.01)
    });

    let run = run_pot_scf_contour(PotScfContourRunInput {
        first_scmt_call: true,
        electron_count_target: 0.0,
        active_energy_count: energy_grid.len(),
        floor_count: steps.len(),
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        source_energies: source_energies.view(),
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.view(),
        scattering_ldos: scattering_ldos.view(),
        embedded_ldos_source: embedded_ldos_source.view(),
        scattering_density: scattering_density.view(),
        embedded_density_source: embedded_density_source.view(),
        include_high_l: false,
    })?;

    assert_eq!(run.status, PotScfContourRunStatus::NeedsMoreSourcePoints);
    assert_eq!(run.energy_points_used, 1);
    assert_complex_close(run.previous_energy, energy_grid[0]);
    assert_complex_close(run.current_energy, energy_grid[1]);
    assert_eq!(run.current_floor, steps.len());
    assert_eq!(run.previous_floor, steps.len());
    assert_eq!(run.fermi_energy, None);

    let bad_source_energies = Array1::from_vec(vec![Complex::new(0.11, 0.04)]);
    assert!(matches!(
        run_pot_scf_contour(PotScfContourRunInput {
            first_scmt_call: true,
            electron_count_target: 0.0,
            active_energy_count: energy_grid.len(),
            floor_count: steps.len(),
            energy_grid: energy_grid.view(),
            steps: steps.view(),
            source_energies: bad_source_energies.view(),
            highest_potential_index: 0,
            last_indices: last_indices.view(),
            potential_multiplicities: potential_multiplicities.view(),
            scattering_trace: scattering_trace.view(),
            scattering_ldos: scattering_ldos.view(),
            embedded_ldos_source: embedded_ldos_source.view(),
            scattering_density: scattering_density.view(),
            embedded_density_source: embedded_density_source.view(),
            include_high_l: false,
        }),
        Err(DensityError::ContourEnergyMismatch { index: 0, .. })
    ));
    Ok(())
}

#[test]
fn pot_scf_fermi_endpoint_matches_feff_scmt_formula() -> Result<(), DensityError> {
    let angular_count = 4;
    let potential_count = 2;
    let radial_count = 5;
    let current_energy = Complex::new(0.44, 0.018);
    let previous_energy = Complex::new(0.40, 0.028);
    let current_delta = -0.052;
    let previous_delta = 0.083;
    let left_sum = Complex::new(1.35, -0.42);
    let right_sum = Complex::new(1.18, 0.26);
    let last_indices = Array1::from_vec(vec![4, 3]);
    let embedded_ldos =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, pot)| {
            Complex::new(
                0.18 + 0.03 * angular as Real + 0.02 * pot as Real,
                -0.04 + 0.015 * angular as Real - 0.01 * pot as Real,
            )
        });
    let previous_ldos =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, pot)| {
            Complex::new(
                0.12 + 0.025 * angular as Real + 0.015 * pot as Real,
                -0.06 + 0.011 * angular as Real - 0.008 * pot as Real,
            )
        });
    let embedded_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, pot)| {
            Complex::new(
                0.03 + 0.009 * radial as Real + 0.004 * pot as Real,
                -0.012 + 0.003 * radial as Real - 0.002 * pot as Real,
            )
        });
    let previous_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, pot)| {
            Complex::new(
                0.025 + 0.007 * radial as Real + 0.003 * pot as Real,
                -0.015 + 0.0025 * radial as Real - 0.0015 * pot as Real,
            )
        });
    let valence_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, pot)| {
            0.05 + 0.01 * radial as Real + 0.02 * pot as Real
        });
    let occupancy_by_l =
        Array2::from_shape_fn((angular_count, potential_count), |(angular, pot)| {
            0.2 + 0.03 * angular as Real + 0.04 * pot as Real
        });

    let result = finish_pot_scf_fermi_endpoint(PotScfFermiEndpointInput {
        current_energy,
        previous_energy,
        current_electron_delta: current_delta,
        previous_electron_delta: previous_delta,
        left_sum,
        right_sum,
        highest_potential_index: potential_count - 1,
        last_indices: last_indices.view(),
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: previous_ldos.view(),
        embedded_density: embedded_density.view(),
        previous_density: previous_density.view(),
        valence_density: valence_density.view(),
        occupancy_by_l: occupancy_by_l.view(),
        include_high_l: false,
    })?;

    let fraction = expected_scmt_endpoint_fraction(
        current_energy,
        previous_energy,
        current_delta,
        previous_delta,
        right_sum,
        left_sum,
    );
    assert_close(result.interpolation_fraction, fraction);
    assert_close(
        result.fermi_energy,
        (current_energy * (1.0 - fraction) + previous_energy * fraction).re,
    );

    let occupancy_correction = expected_scmt_endpoint_correction(
        current_energy,
        previous_energy,
        embedded_ldos[(2, 1)] * 2.0,
        previous_ldos[(2, 1)] * 2.0,
        fraction,
    );
    assert_close(
        result.occupancy_by_l[(2, 1)],
        occupancy_by_l[(2, 1)] + fraction * occupancy_correction,
    );
    assert_close(result.occupancy_by_l[(3, 1)], occupancy_by_l[(3, 1)]);

    let valence_correction = expected_scmt_endpoint_correction(
        current_energy,
        previous_energy,
        embedded_density[(2, 1)] * 2.0,
        previous_density[(2, 1)] * 2.0,
        fraction,
    );
    assert_close(
        result.valence_density[(2, 1)],
        valence_density[(2, 1)] + fraction * valence_correction,
    );
    assert_close(result.valence_density[(3, 1)], valence_density[(3, 1)]);
    Ok(())
}

#[test]
fn pot_scf_fermi_endpoint_zero_current_delta_keeps_accumulators() -> Result<(), DensityError> {
    let last_indices = Array1::from_vec(vec![2]);
    let embedded_ldos = Array2::from_shape_vec(
        (2, 1),
        vec![Complex::new(0.2, -0.1), Complex::new(0.3, -0.05)],
    )
    .unwrap();
    let previous_ldos = Array2::from_shape_vec(
        (2, 1),
        vec![Complex::new(0.18, -0.12), Complex::new(0.27, -0.07)],
    )
    .unwrap();
    let embedded_density = Array2::from_shape_vec(
        (2, 1),
        vec![Complex::new(0.04, -0.01), Complex::new(0.05, -0.005)],
    )
    .unwrap();
    let previous_density = Array2::from_shape_vec(
        (2, 1),
        vec![Complex::new(0.035, -0.012), Complex::new(0.045, -0.007)],
    )
    .unwrap();
    let valence_density = Array2::from_shape_vec((2, 1), vec![0.11, 0.13]).unwrap();
    let occupancy_by_l = Array2::from_shape_vec((2, 1), vec![0.21, 0.25]).unwrap();

    let result = finish_pot_scf_fermi_endpoint(PotScfFermiEndpointInput {
        current_energy: Complex::new(0.5, 0.01),
        previous_energy: Complex::new(0.45, 0.02),
        current_electron_delta: 0.0,
        previous_electron_delta: -0.04,
        left_sum: Complex::new(0.4, -0.1),
        right_sum: Complex::new(0.5, 0.2),
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: previous_ldos.view(),
        embedded_density: embedded_density.view(),
        previous_density: previous_density.view(),
        valence_density: valence_density.view(),
        occupancy_by_l: occupancy_by_l.view(),
        include_high_l: false,
    })?;

    assert_close(result.interpolation_fraction, 0.0);
    assert_close(result.fermi_energy, 0.5);
    assert_eq!(result.valence_density, valence_density);
    assert_eq!(result.occupancy_by_l, occupancy_by_l);
    Ok(())
}

#[test]
fn pot_scf_contour_step_consumes_first_call_grid_points() -> Result<(), DensityError> {
    let (energy_grid, steps) = sample_scmt_contour_tables();
    let current_energy = energy_grid[1];

    let step = pot_scf_contour_step(PotScfContourStepInput {
        first_scmt_call: true,
        energy_index: 2,
        active_energy_count: 6,
        floor_count: 4,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        current_energy,
        previous_energy: energy_grid[0],
        current_floor: 4,
        previous_floor: 4,
        direction: 1,
        can_step_up: false,
        current_electron_delta: 0.2,
        previous_electron_delta: 0.1,
    })?;

    assert_eq!(step.status, PotScfContourStepStatus::Continue);
    assert_complex_close(step.previous_energy, current_energy);
    assert_complex_close(step.current_energy, energy_grid[2]);
    assert_eq!(step.current_floor, 4);
    assert_eq!(step.previous_floor, 4);
    Ok(())
}

#[test]
fn pot_scf_contour_step_starts_first_call_horizontal_search() -> Result<(), DensityError> {
    let (energy_grid, steps) = sample_scmt_contour_tables();
    let current_energy = energy_grid[3];

    let step = pot_scf_contour_step(PotScfContourStepInput {
        first_scmt_call: true,
        energy_index: 4,
        active_energy_count: 6,
        floor_count: 4,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        current_energy,
        previous_energy: energy_grid[2],
        current_floor: 4,
        previous_floor: 4,
        direction: 1,
        can_step_up: true,
        current_electron_delta: 0.2,
        previous_electron_delta: 0.1,
    })?;

    assert_eq!(step.status, PotScfContourStepStatus::Continue);
    assert!(!step.can_step_up);
    assert_eq!(step.direction, -1);
    assert_complex_close(step.previous_energy, current_energy);
    assert_complex_close(
        step.current_energy,
        current_energy + Complex::new(-steps[3], 0.0),
    );
    Ok(())
}

#[test]
fn pot_scf_contour_step_reports_lowest_floor_bracket() -> Result<(), DensityError> {
    let (energy_grid, steps) = sample_scmt_contour_tables();
    let current_energy = Complex::new(0.46, 0.01);
    let previous_energy = Complex::new(0.44, 0.01);

    let step = pot_scf_contour_step(PotScfContourStepInput {
        first_scmt_call: false,
        energy_index: 7,
        active_energy_count: 6,
        floor_count: 4,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        current_energy,
        previous_energy,
        current_floor: 1,
        previous_floor: 1,
        direction: 1,
        can_step_up: true,
        current_electron_delta: -0.01,
        previous_electron_delta: 0.02,
    })?;

    assert_eq!(step.status, PotScfContourStepStatus::Bracketed);
    assert_complex_close(step.previous_energy, previous_energy);
    assert_complex_close(step.current_energy, current_energy);
    assert_eq!(step.current_floor, 1);
    assert_eq!(step.previous_floor, 1);
    Ok(())
}

#[test]
fn pot_scf_contour_step_moves_down_after_horizontal_sign_change() -> Result<(), DensityError> {
    let (energy_grid, steps) = sample_scmt_contour_tables();
    let current_energy = Complex::new(0.50, 0.12);

    let step = pot_scf_contour_step(PotScfContourStepInput {
        first_scmt_call: false,
        energy_index: 8,
        active_energy_count: 6,
        floor_count: 4,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        current_energy,
        previous_energy: Complex::new(0.48, 0.12),
        current_floor: 3,
        previous_floor: 3,
        direction: 1,
        can_step_up: true,
        current_electron_delta: -0.01,
        previous_electron_delta: 0.02,
    })?;

    assert_eq!(step.status, PotScfContourStepStatus::Continue);
    assert!(!step.can_step_up);
    assert_eq!(step.previous_floor, 3);
    assert_eq!(step.current_floor, 2);
    assert_complex_close(step.previous_energy, current_energy);
    assert_complex_close(
        step.current_energy,
        Complex::new(current_energy.re, 4.0 * steps[1]),
    );
    Ok(())
}

#[test]
fn pot_scf_contour_step_moves_up_when_far_from_fermi() -> Result<(), DensityError> {
    let (energy_grid, steps) = sample_scmt_contour_tables();
    let current_energy = Complex::new(0.50, 0.08);

    let step = pot_scf_contour_step(PotScfContourStepInput {
        first_scmt_call: false,
        energy_index: 8,
        active_energy_count: 6,
        floor_count: 4,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        current_energy,
        previous_energy: Complex::new(0.48, 0.08),
        current_floor: 2,
        previous_floor: 2,
        direction: -1,
        can_step_up: true,
        current_electron_delta: 10.0,
        previous_electron_delta: 9.5,
    })?;

    assert_eq!(step.status, PotScfContourStepStatus::Continue);
    assert_eq!(step.previous_floor, 2);
    assert_eq!(step.current_floor, 3);
    assert_complex_close(step.previous_energy, current_energy);
    assert_complex_close(
        step.current_energy,
        Complex::new(current_energy.re, 4.0 * steps[2]),
    );
    Ok(())
}

#[test]
fn pot_scf_contour_step_sets_direction_after_vertical_move() -> Result<(), DensityError> {
    let (energy_grid, steps) = sample_scmt_contour_tables();
    let current_energy = Complex::new(0.50, 0.08);

    let step = pot_scf_contour_step(PotScfContourStepInput {
        first_scmt_call: false,
        energy_index: 8,
        active_energy_count: 6,
        floor_count: 4,
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        current_energy,
        previous_energy: Complex::new(0.48, 0.12),
        current_floor: 2,
        previous_floor: 3,
        direction: -1,
        can_step_up: false,
        current_electron_delta: -0.04,
        previous_electron_delta: -0.01,
    })?;

    assert_eq!(step.status, PotScfContourStepStatus::Continue);
    assert_eq!(step.direction, 1);
    assert_eq!(step.previous_floor, 2);
    assert_eq!(step.current_floor, 2);
    assert_complex_close(step.previous_energy, current_energy);
    assert_complex_close(
        step.current_energy,
        current_energy + Complex::new(steps[1], 0.0),
    );
    Ok(())
}

#[test]
fn ldos_ff2rho_tables_match_feff_non_full_potential_reference() -> Result<(), DensityError> {
    let energy = Array1::from_vec(vec![Complex::new(0.5, 0.01), Complex::new(0.75, 0.01)]);
    let embedded = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 99.0, 3.0, 4.0, 99.0, 5.0, 6.0, 99.0, 7.0, 8.0, 99.0,
        ],
    )
    .unwrap();
    let scattering = Array2::from_shape_vec(
        (4, 2),
        vec![
            Complex::new(1.5, -0.4),
            Complex::new(0.5, 0.2),
            Complex::new(-0.3, 0.7),
            Complex::new(1.1, -0.6),
            Complex::new(0.8, 0.9),
            Complex::new(-1.2, 0.4),
            Complex::new(2.0, -1.0),
            Complex::new(-0.5, 0.3),
        ],
    )
    .unwrap();
    let trace = Array2::from_shape_vec(
        (4, 2),
        vec![
            Complex::new(0.2, 0.3),
            Complex::new(-0.1, 0.4),
            Complex::new(0.5, -0.2),
            Complex::new(0.3, 0.1),
            Complex::new(-0.4, 0.6),
            Complex::new(0.7, -0.5),
            Complex::new(0.9, 0.2),
            Complex::new(-0.8, 0.25),
        ],
    )
    .unwrap();

    let tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: energy.view(),
        embedded_ldos: embedded.view(),
        scattering_ldos: scattering.view(),
        scattering_trace: trace.view(),
        angular_count: 4,
        apply_scattering: true,
    })?;

    assert_close(tables.energy_ev[0], 0.5 * FEFF_HARTREE_EV);
    assert_close(tables.energy_ev[1], 0.75 * FEFF_HARTREE_EV);
    assert_close(tables.rhoc_density[(0, 0)], 1.0);
    assert_close(tables.rhoc_density[(1, 3)], 8.0);
    assert_close(tables.ldos_density[(0, 0)], 1.37);
    assert_close(tables.ldos_density[(1, 0)], 2.18);
    assert_close(tables.ldos_density[(0, 1)], 3.41);
    assert_close(tables.ldos_density[(1, 2)], 6.88);
    assert_close(tables.ldos_density[(0, 3)], 6.5);
    Ok(())
}

#[test]
fn ldos_ff2rho_tables_can_skip_scattering_like_msapp_one() -> Result<(), DensityError> {
    let energy = Array1::from_vec(vec![Complex::new(0.5, 0.01)]);
    let embedded = Array2::from_shape_vec((2, 1), vec![1.0, 3.0]).unwrap();
    let empty_complex = Array2::<Complex>::zeros((0, 0));

    let tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: energy.view(),
        embedded_ldos: embedded.view(),
        scattering_ldos: empty_complex.view(),
        scattering_trace: empty_complex.view(),
        angular_count: 2,
        apply_scattering: false,
    })?;

    assert_eq!(tables.ldos_density, tables.rhoc_density);
    assert_close(tables.ldos_density[(0, 0)], 1.0);
    assert_close(tables.ldos_density[(0, 1)], 3.0);
    Ok(())
}

#[test]
fn ldos_fmsdos_trace_matches_feff_non_full_potential_loop() -> Result<(), DensityError> {
    let mut scattering = Array3::<Complex32>::zeros((4, 4, 2));
    scattering[(0, 0, 0)] = Complex32::new(1.0, 0.5);
    scattering[(1, 1, 0)] = Complex32::new(2.0, -0.25);
    scattering[(2, 2, 0)] = Complex32::new(3.0, 0.75);
    scattering[(3, 3, 0)] = Complex32::new(4.0, -1.25);
    scattering[(0, 0, 1)] = Complex32::new(-0.5, 0.25);
    scattering[(1, 1, 1)] = Complex32::new(0.25, 1.5);
    scattering[(2, 2, 1)] = Complex32::new(-1.25, 0.5);
    scattering[(3, 3, 1)] = Complex32::new(2.5, -0.75);

    let mut phase = Array3::<Complex32>::zeros((1, 3, 2));
    phase[(0, 1, 0)] = Complex32::new(0.10, 0.03);
    phase[(0, 2, 0)] = Complex32::new(-0.05, 0.02);
    phase[(0, 1, 1)] = Complex32::new(0.07, -0.01);
    phase[(0, 2, 1)] = Complex32::new(0.02, 0.04);

    let trace = ldos_fmsdos_trace(LdosFmsdosTraceInput {
        scattering_matrices: scattering.view(),
        phase_shifts: phase.view(),
        spin_index: 0,
        angular_count: 2,
    })?;

    assert_eq!(trace.dim(), (2, 2));
    assert_complex_close(
        trace[(0, 0)],
        fmsdos_expected(Complex::new(1.0, 0.5), Complex::new(0.10, 0.03), 0),
    );
    assert_complex_close(
        trace[(1, 0)],
        fmsdos_expected(Complex::new(9.0, -0.75), Complex::new(-0.05, 0.02), 1),
    );
    assert_complex_close(
        trace[(0, 1)],
        fmsdos_expected(Complex::new(-0.5, 0.25), Complex::new(0.07, -0.01), 0),
    );
    assert_complex_close(
        trace[(1, 1)],
        fmsdos_expected(Complex::new(1.5, 1.25), Complex::new(0.02, 0.04), 1),
    );
    Ok(())
}

#[test]
fn ldos_fmsdos_trace_rejects_short_source_tables() {
    let scattering = Array3::<Complex32>::zeros((3, 3, 1));
    let phase = Array3::<Complex32>::zeros((1, 3, 1));
    assert!(matches!(
        ldos_fmsdos_trace(LdosFmsdosTraceInput {
            scattering_matrices: scattering.view(),
            phase_shifts: phase.view(),
            spin_index: 0,
            angular_count: 2,
        }),
        Err(DensityError::CubeShapeTooSmall {
            name: "scattering_matrices",
            ..
        })
    ));

    let scattering = Array3::<Complex32>::zeros((4, 4, 1));
    let phase = Array3::<Complex32>::zeros((1, 1, 1));
    assert!(matches!(
        ldos_fmsdos_trace(LdosFmsdosTraceInput {
            scattering_matrices: scattering.view(),
            phase_shifts: phase.view(),
            spin_index: 0,
            angular_count: 2,
        }),
        Err(DensityError::CubeShapeTooSmall {
            name: "phase_shifts",
            ..
        })
    ));
}

#[test]
fn ldos_fmsdos_trace_grid_matches_gtr_binary_order() -> Result<(), DensityError> {
    let mut scattering = Array4::<Complex32>::zeros((2, 4, 4, 2));
    scattering[(0, 0, 0, 0)] = Complex32::new(1.0, 0.5);
    scattering[(0, 1, 1, 1)] = Complex32::new(0.25, 1.5);
    scattering[(0, 2, 2, 1)] = Complex32::new(-1.25, 0.5);
    scattering[(0, 3, 3, 1)] = Complex32::new(2.5, -0.75);
    scattering[(1, 0, 0, 0)] = Complex32::new(1.25, -0.5);
    scattering[(1, 1, 1, 1)] = Complex32::new(0.5, -1.0);
    scattering[(1, 2, 2, 1)] = Complex32::new(-2.0, 0.25);
    scattering[(1, 3, 3, 1)] = Complex32::new(1.25, 0.75);

    let mut phase = Array4::<Complex32>::zeros((2, 1, 3, 2));
    phase[(0, 0, 1, 0)] = Complex32::new(0.10, 0.03);
    phase[(0, 0, 2, 1)] = Complex32::new(0.02, 0.04);
    phase[(1, 0, 1, 0)] = Complex32::new(-0.08, 0.01);
    phase[(1, 0, 2, 1)] = Complex32::new(0.03, -0.02);

    let trace_grid = ldos_fmsdos_trace_grid(LdosFmsdosTraceGridInput {
        scattering_matrices: scattering.view(),
        phase_shifts: phase.view(),
        spin_index: 0,
        angular_count: 2,
    })?;

    assert_eq!(trace_grid.dim(), (2, 2, 2));
    assert_complex_close(
        trace_grid[(0, 0, 0)],
        fmsdos_expected(Complex::new(1.0, 0.5), Complex::new(0.10, 0.03), 0),
    );
    assert_complex_close(
        trace_grid[(0, 1, 1)],
        fmsdos_expected(Complex::new(1.5, 1.25), Complex::new(0.02, 0.04), 1),
    );
    assert_complex_close(
        trace_grid[(1, 0, 0)],
        fmsdos_expected(Complex::new(1.25, -0.5), Complex::new(-0.08, 0.01), 0),
    );
    assert_complex_close(
        trace_grid[(1, 1, 1)],
        fmsdos_expected(Complex::new(-0.25, 0.0), Complex::new(0.03, -0.02), 1),
    );
    Ok(())
}

#[test]
fn ldos_fmsdos_trace_grid_rejects_energy_axis_mismatch() {
    let scattering = Array4::<Complex32>::zeros((2, 4, 4, 1));
    let phase = Array4::<Complex32>::zeros((1, 1, 3, 1));

    assert_eq!(
        ldos_fmsdos_trace_grid(LdosFmsdosTraceGridInput {
            scattering_matrices: scattering.view(),
            phase_shifts: phase.view(),
            spin_index: 0,
            angular_count: 2,
        }),
        Err(DensityError::LengthMismatch {
            left_name: "scattering_matrices.energy",
            left_len: 2,
            right_name: "phase_shifts.energy",
            right_len: 1,
        })
    );
}

#[test]
fn ldos_rhol_density_matches_feff_integral_formula() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let regular_large = Array1::from_vec(vec![
        Complex::new(0.11, 0.02),
        Complex::new(0.18, -0.03),
        Complex::new(0.29, 0.04),
        Complex::new(0.37, 0.08),
        Complex::new(0.43, -0.02),
        Complex::new(0.51, 0.05),
        Complex::new(0.58, -0.01),
    ]);
    let regular_small = Array1::from_vec(vec![
        Complex::new(0.015, -0.004),
        Complex::new(0.024, 0.006),
        Complex::new(0.031, -0.005),
        Complex::new(0.038, 0.007),
        Complex::new(0.046, -0.003),
        Complex::new(0.053, 0.004),
        Complex::new(0.061, -0.002),
    ]);
    let irregular_large = Array1::from_vec(vec![
        Complex::new(-0.42, 0.31),
        Complex::new(-0.33, 0.24),
        Complex::new(-0.21, 0.19),
        Complex::new(-0.08, 0.12),
        Complex::new(0.04, 0.08),
        Complex::new(0.13, 0.03),
        Complex::new(0.20, -0.01),
    ]);
    let irregular_small = Array1::from_vec(vec![
        Complex::new(-0.035, 0.016),
        Complex::new(-0.026, 0.013),
        Complex::new(-0.018, 0.009),
        Complex::new(-0.009, 0.006),
        Complex::new(0.003, 0.004),
        Complex::new(0.011, 0.002),
        Complex::new(0.019, -0.001),
    ]);
    let wave_number = Complex::new(0.74, 0.08);
    let angular_momentum = 2;
    let radial_step = 0.05;
    let norman_radius = 0.64;

    let result = ldos_rhol_density(LdosRholDensityInput {
        radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        radial_step,
        norman_radius,
        wave_number,
        angular_momentum,
    })?;

    let alpha_wave = wave_number * (1.0 / 137.035_989_56);
    let small_component_factor = -alpha_wave
        / (Complex::new(1.0, 0.0) + (Complex::new(1.0, 0.0) + alpha_wave * alpha_wave).sqrt());
    let density_scale = ((2 * angular_momentum + 1) as Real)
        / (Complex::new(1.0, 0.0) + small_component_factor * small_component_factor)
        / std::f64::consts::PI
        * wave_number
        * (4.0 / FEFF_HARTREE_EV);

    let scattering_integrand = regular_large
        .iter()
        .zip(regular_small.iter())
        .map(|(&large, &small)| large * large + small * small)
        .collect::<Vec<_>>();
    let expected_scattering = csomm2(
        radii.as_slice().unwrap(),
        &scattering_integrand,
        radial_step,
        (2 * angular_momentum + 2) as Real,
        norman_radius,
    )? * density_scale;
    assert_complex_close(result.scattering_ldos, expected_scattering);
    assert_complex_close(
        result.scattering_ldos,
        Complex::new(1.476_982_194_257_99e-2, 4.076_350_732_075_531e-3),
    );

    let imaginary = Complex::new(0.0, 1.0);
    let embedded_integrand = irregular_large
        .iter()
        .zip(regular_large.iter())
        .zip(irregular_small.iter().zip(regular_small.iter()))
        .map(
            |((&irregular_large, &regular_large), (&irregular_small, &regular_small))| {
                irregular_large * regular_large - imaginary * regular_large * regular_large
                    + irregular_small * regular_small
                    - imaginary * regular_small * regular_small
            },
        )
        .collect::<Vec<_>>();
    let expected_embedded = -(csomm2(
        radii.as_slice().unwrap(),
        &embedded_integrand,
        radial_step,
        1.0,
        norman_radius,
    )? * density_scale)
        .im;
    assert_close(result.embedded_ldos, expected_embedded);
    assert_close(result.embedded_ldos, 1.433_537_795_285_896_9e-2);
    assert_complex_close(result.density_scale, density_scale);
    Ok(())
}

#[test]
fn pot_rholie_density_preserves_complex_feff_work_arrays() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let regular_large = Array1::from_vec(vec![
        Complex::new(0.11, 0.02),
        Complex::new(0.18, -0.03),
        Complex::new(0.29, 0.04),
        Complex::new(0.37, 0.08),
        Complex::new(0.43, -0.02),
        Complex::new(0.51, 0.05),
        Complex::new(0.58, -0.01),
    ]);
    let regular_small = Array1::from_vec(vec![
        Complex::new(0.015, -0.004),
        Complex::new(0.024, 0.006),
        Complex::new(0.031, -0.005),
        Complex::new(0.038, 0.007),
        Complex::new(0.046, -0.003),
        Complex::new(0.053, 0.004),
        Complex::new(0.061, -0.002),
    ]);
    let irregular_large = Array1::from_vec(vec![
        Complex::new(-0.42, 0.31),
        Complex::new(-0.33, 0.24),
        Complex::new(-0.21, 0.19),
        Complex::new(-0.08, 0.12),
        Complex::new(0.04, 0.08),
        Complex::new(0.13, 0.03),
        Complex::new(0.20, -0.01),
    ]);
    let irregular_small = Array1::from_vec(vec![
        Complex::new(-0.035, 0.016),
        Complex::new(-0.026, 0.013),
        Complex::new(-0.018, 0.009),
        Complex::new(-0.009, 0.006),
        Complex::new(0.003, 0.004),
        Complex::new(0.011, 0.002),
        Complex::new(0.019, -0.001),
    ]);
    let wave_number = Complex::new(0.74, 0.08);
    let angular_momentum = 2;
    let radial_step = 0.05;
    let norman_radius = 0.64;

    let pot = pot_rholie_density(PotRholieDensityInput {
        source_radii: radii.view(),
        output_radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        radial_step,
        norman_radius,
        wave_number,
        angular_momentum,
    })?;
    let ldos = ldos_rhol_density(LdosRholDensityInput {
        radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        radial_step,
        norman_radius,
        wave_number,
        angular_momentum,
    })?;

    let alpha_wave = wave_number * (1.0 / 137.035_989_56);
    let small_component_factor = -alpha_wave
        / (Complex::new(1.0, 0.0) + (Complex::new(1.0, 0.0) + alpha_wave * alpha_wave).sqrt());
    let pot_density_scale = ((2 * angular_momentum + 1) as Real)
        / (Complex::new(1.0, 0.0) + small_component_factor * small_component_factor)
        / std::f64::consts::PI
        * wave_number
        * 2.0;
    assert_complex_close(pot.density_scale, pot_density_scale);
    assert_complex_close(
        ldos.scattering_ldos * (FEFF_HARTREE_EV / 2.0),
        pot.scattering_ldos,
    );
    assert_close(
        ldos.embedded_ldos * (FEFF_HARTREE_EV / 2.0),
        pot.embedded_ldos.im,
    );

    let scattering_integrand = regular_large
        .iter()
        .zip(regular_small.iter())
        .map(|(&large, &small)| large * large + small * small)
        .collect::<Vec<_>>();
    let expected_scattering = csomm2(
        radii.as_slice().unwrap(),
        &scattering_integrand,
        radial_step,
        (2 * angular_momentum + 2) as Real,
        norman_radius,
    )? * pot_density_scale;
    assert_complex_close(pot.scattering_ldos, expected_scattering);
    assert_complex_close(
        pot.scattering_density[3],
        scattering_integrand[3] * pot_density_scale,
    );

    let imaginary = Complex::new(0.0, 1.0);
    let embedded_integrand = irregular_large
        .iter()
        .zip(regular_large.iter())
        .zip(irregular_small.iter().zip(regular_small.iter()))
        .map(
            |((&irregular_large, &regular_large), (&irregular_small, &regular_small))| {
                irregular_large * regular_large - imaginary * regular_large * regular_large
                    + irregular_small * regular_small
                    - imaginary * regular_small * regular_small
            },
        )
        .collect::<Vec<_>>();
    let expected_embedded = -(csomm2(
        radii.as_slice().unwrap(),
        &embedded_integrand,
        radial_step,
        1.0,
        norman_radius,
    )? * pot_density_scale);
    assert_complex_close(pot.embedded_ldos, expected_embedded);
    assert_complex_close(
        pot.embedded_density[3],
        -(embedded_integrand[3] * pot_density_scale),
    );
    Ok(())
}

#[test]
fn pot_rholie_density_grid_feeds_ff2g_orientation() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let base_regular_large = [
        Complex::new(0.11, 0.02),
        Complex::new(0.18, -0.03),
        Complex::new(0.29, 0.04),
        Complex::new(0.37, 0.08),
        Complex::new(0.43, -0.02),
        Complex::new(0.51, 0.05),
        Complex::new(0.58, -0.01),
    ];
    let base_regular_small = [
        Complex::new(0.015, -0.004),
        Complex::new(0.024, 0.006),
        Complex::new(0.031, -0.005),
        Complex::new(0.038, 0.007),
        Complex::new(0.046, -0.003),
        Complex::new(0.053, 0.004),
        Complex::new(0.061, -0.002),
    ];
    let base_irregular_large = [
        Complex::new(-0.42, 0.31),
        Complex::new(-0.33, 0.24),
        Complex::new(-0.21, 0.19),
        Complex::new(-0.08, 0.12),
        Complex::new(0.04, 0.08),
        Complex::new(0.13, 0.03),
        Complex::new(0.20, -0.01),
    ];
    let base_irregular_small = [
        Complex::new(-0.035, 0.016),
        Complex::new(-0.026, 0.013),
        Complex::new(-0.018, 0.009),
        Complex::new(-0.009, 0.006),
        Complex::new(0.003, 0.004),
        Complex::new(0.011, 0.002),
        Complex::new(0.019, -0.001),
    ];
    let angular_count = 2;
    let radial_count = radii.len();
    let mut regular_large = Array2::<Complex>::zeros((angular_count, radial_count));
    let mut regular_small = Array2::<Complex>::zeros((angular_count, radial_count));
    let mut irregular_large = Array2::<Complex>::zeros((angular_count, radial_count));
    let mut irregular_small = Array2::<Complex>::zeros((angular_count, radial_count));

    for angular in 0..angular_count {
        let angular_scale = angular as Real;
        for radial in 0..radial_count {
            regular_large[(angular, radial)] =
                base_regular_large[radial] * (1.0 + 0.03 * angular_scale);
            regular_small[(angular, radial)] =
                base_regular_small[radial] * (1.0 - 0.02 * angular_scale);
            irregular_large[(angular, radial)] =
                base_irregular_large[radial] * (1.0 + 0.06 * angular_scale);
            irregular_small[(angular, radial)] =
                base_irregular_small[radial] * (1.0 + 0.01 * angular_scale);
        }
    }

    let grid = pot_rholie_density_grid(PotRholieDensityGridInput {
        source_radii: radii.view(),
        output_radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        radial_step: 0.05,
        norman_radius: 0.64,
        wave_number: Complex::new(0.74, 0.08),
        angular_count,
    })?;

    assert_eq!(grid.scattering_ldos.len(), angular_count);
    assert_eq!(grid.embedded_ldos.len(), angular_count);
    assert_eq!(grid.scattering_density.dim(), (radial_count, angular_count));
    let first = pot_rholie_density(PotRholieDensityInput {
        source_radii: radii.view(),
        output_radii: radii.view(),
        regular_large: regular_large.index_axis(Axis(0), 0),
        regular_small: regular_small.index_axis(Axis(0), 0),
        irregular_large: irregular_large.index_axis(Axis(0), 0),
        irregular_small: irregular_small.index_axis(Axis(0), 0),
        radial_step: 0.05,
        norman_radius: 0.64,
        wave_number: Complex::new(0.74, 0.08),
        angular_momentum: 0,
    })?;
    let second = pot_rholie_density(PotRholieDensityInput {
        source_radii: radii.view(),
        output_radii: radii.view(),
        regular_large: regular_large.index_axis(Axis(0), 1),
        regular_small: regular_small.index_axis(Axis(0), 1),
        irregular_large: irregular_large.index_axis(Axis(0), 1),
        irregular_small: irregular_small.index_axis(Axis(0), 1),
        radial_step: 0.05,
        norman_radius: 0.64,
        wave_number: Complex::new(0.74, 0.08),
        angular_momentum: 1,
    })?;
    assert_complex_close(grid.scattering_ldos[1], second.scattering_ldos);
    assert_complex_close(grid.embedded_ldos[1], second.embedded_ldos);
    assert_complex_close(
        grid.embedded_density[3],
        first.embedded_density[3] + second.embedded_density[3],
    );

    let scattering_trace =
        Array1::from_vec(vec![Complex32::new(0.2, -0.1), Complex32::new(-0.3, 0.4)]);
    let mut embedded_ldos = Array2::<Complex>::zeros((angular_count, 1));
    for angular in 0..angular_count {
        embedded_ldos[(angular, 0)] = grid.embedded_ldos[angular];
    }
    let update = update_valence_density(ValenceDensityUpdateInput {
        scattering_trace: scattering_trace.view(),
        potential_index: 0,
        energy_index: 1,
        last_radial_index: radial_count,
        scattering_ldos: grid.scattering_ldos.view(),
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: Array2::<Complex>::zeros((angular_count, 1)).view(),
        scattering_density: grid.scattering_density.view(),
        embedded_density: grid.embedded_density.view(),
        previous_density: Array1::<Complex>::zeros(radial_count).view(),
        valence_density: Array1::<Real>::zeros(radial_count).view(),
        occupancy_by_l: Array1::<Real>::zeros(angular_count).view(),
        current_energy: Complex::new(0.72, 0.11),
        previous_energy: Complex::new(0.61, -0.04),
        potential_multiplicity: 1.0,
        current_floor: 1,
        previous_floor: 0,
        left_sum: Complex::new(0.0, 0.0),
        right_sum: Complex::new(0.0, 0.0),
        total_electron_count: 0.0,
        include_high_l: false,
    })?;
    let trace0 = Complex::new(
        scattering_trace[0].re as Real,
        scattering_trace[0].im as Real,
    );
    let trace1 = Complex::new(
        scattering_trace[1].re as Real,
        scattering_trace[1].im as Real,
    );
    assert_complex_close(
        update.embedded_ldos[(1, 0)],
        grid.embedded_ldos[1] + trace1 * grid.scattering_ldos[1],
    );
    assert_complex_close(
        update.embedded_density[3],
        grid.embedded_density[3]
            + trace0 * grid.scattering_density[(3, 0)]
            + trace1 * grid.scattering_density[(3, 1)],
    );

    let composed = pot_scf_energy_density(PotScfEnergyDensityInput {
        source_radii: radii.view(),
        output_radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        radial_step: 0.05,
        norman_radius: 0.64,
        wave_number: Complex::new(0.74, 0.08),
        angular_count,
        scattering_trace: scattering_trace.view(),
        potential_index: 0,
        energy_index: 1,
        last_radial_index: radial_count,
        embedded_ldos: embedded_ldos.view(),
        previous_ldos: Array2::<Complex>::zeros((angular_count, 1)).view(),
        previous_density: Array1::<Complex>::zeros(radial_count).view(),
        valence_density: Array1::<Real>::zeros(radial_count).view(),
        occupancy_by_l: Array1::<Real>::zeros(angular_count).view(),
        current_energy: Complex::new(0.72, 0.11),
        previous_energy: Complex::new(0.61, -0.04),
        potential_multiplicity: 1.0,
        current_floor: 1,
        previous_floor: 0,
        left_sum: Complex::new(0.0, 0.0),
        right_sum: Complex::new(0.0, 0.0),
        total_electron_count: 0.0,
        include_high_l: false,
    })?;
    assert_complex_close(composed.rholie.scattering_ldos[1], grid.scattering_ldos[1]);
    assert_complex_close(
        composed.rholie.embedded_density[3],
        grid.embedded_density[3],
    );
    assert_complex_close(
        composed.valence.embedded_ldos[(1, 0)],
        update.embedded_ldos[(1, 0)],
    );
    assert_close(
        composed.valence.valence_density[3],
        update.valence_density[3],
    );
    Ok(())
}

#[test]
fn pot_scf_contour_source_rows_lift_rholie_for_contour_driver() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let point_count = 2;
    let potential_count = 2;
    let angular_count = 2;
    let radial_count = radii.len();
    let energy_grid = Array1::from_vec(vec![
        Complex::new(0.20, 0.04),
        Complex::new(0.28, 0.04),
        Complex::new(0.36, 0.04),
    ]);
    let source_energies = Array1::from_vec(vec![energy_grid[0], energy_grid[1]]);
    let steps = Array1::from_vec(vec![0.01, 0.02, 0.03]);
    let norman_radii = Array1::from_vec(vec![0.64, 0.72]);
    let wave_numbers = Array2::from_shape_fn((point_count, potential_count), |(point, pot)| {
        Complex::new(
            0.70 + 0.05 * point as Real + 0.03 * pot as Real,
            0.06 + 0.01 * pot as Real,
        )
    });
    let scattering_trace = Array3::from_shape_fn(
        (point_count, angular_count, potential_count),
        |(point, angular, pot)| {
            Complex32::new(
                (0.04 + 0.01 * point as Real + 0.02 * angular as Real) as f32,
                (-0.03 + 0.015 * pot as Real) as f32,
            )
        },
    );
    let regular_large = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, pot, angular, radial)| {
            Complex::new(
                0.10 + 0.02 * radial as Real + 0.03 * angular as Real + 0.01 * point as Real,
                0.02 - 0.004 * pot as Real + 0.003 * radial as Real,
            )
        },
    );
    let regular_small = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, pot, angular, radial)| {
            Complex::new(
                0.015 + 0.003 * radial as Real + 0.004 * angular as Real,
                -0.004 + 0.002 * point as Real + 0.001 * pot as Real,
            )
        },
    );
    let irregular_large = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, pot, angular, radial)| {
            Complex::new(
                -0.35 + 0.04 * radial as Real + 0.02 * pot as Real,
                0.24 - 0.015 * angular as Real + 0.01 * point as Real,
            )
        },
    );
    let irregular_small = Array4::from_shape_fn(
        (point_count, potential_count, angular_count, radial_count),
        |(point, pot, angular, radial)| {
            Complex::new(
                -0.030 + 0.004 * radial as Real + 0.002 * angular as Real,
                0.014 - 0.002 * pot as Real + 0.001 * point as Real,
            )
        },
    );

    let rows = pot_scf_contour_source_rows(PotScfContourSourceRowsInput {
        source_energies: source_energies.view(),
        source_radii: radii.view(),
        output_radii: radii.view(),
        radial_step: 0.05,
        highest_potential_index: potential_count - 1,
        norman_radii: norman_radii.view(),
        wave_numbers: wave_numbers.view(),
        angular_count,
        scattering_trace: scattering_trace.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
    })?;

    let point = 1;
    let potential = 1;
    let direct = pot_rholie_density_grid(PotRholieDensityGridInput {
        source_radii: radii.view(),
        output_radii: radii.view(),
        regular_large: regular_large
            .index_axis(Axis(0), point)
            .index_axis(Axis(0), potential),
        regular_small: regular_small
            .index_axis(Axis(0), point)
            .index_axis(Axis(0), potential),
        irregular_large: irregular_large
            .index_axis(Axis(0), point)
            .index_axis(Axis(0), potential),
        irregular_small: irregular_small
            .index_axis(Axis(0), point)
            .index_axis(Axis(0), potential),
        radial_step: 0.05,
        norman_radius: norman_radii[potential],
        wave_number: wave_numbers[(point, potential)],
        angular_count,
    })?;

    assert_eq!(rows.source_energies, source_energies);
    assert_eq!(rows.scattering_trace, scattering_trace);
    assert_complex_close(
        rows.scattering_ldos[(point, 1, potential)],
        direct.scattering_ldos[1],
    );
    assert_complex_close(
        rows.embedded_ldos_source[(point, 1, potential)],
        direct.embedded_ldos[1],
    );
    assert_complex_close(
        rows.scattering_density[(point, 3, 1, potential)],
        direct.scattering_density[(3, 1)],
    );
    assert_complex_close(
        rows.embedded_density_source[(point, 3, potential)],
        direct.embedded_density[3],
    );

    let last_indices = Array1::from_vec(vec![radial_count, radial_count]);
    let potential_multiplicities = Array1::from_vec(vec![1.0, 1.5]);
    let run = run_pot_scf_contour(PotScfContourRunInput {
        first_scmt_call: true,
        electron_count_target: 0.0,
        active_energy_count: energy_grid.len(),
        floor_count: steps.len(),
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        source_energies: rows.source_energies.view(),
        highest_potential_index: potential_count - 1,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: rows.scattering_trace.view(),
        scattering_ldos: rows.scattering_ldos.view(),
        embedded_ldos_source: rows.embedded_ldos_source.view(),
        scattering_density: rows.scattering_density.view(),
        embedded_density_source: rows.embedded_density_source.view(),
        include_high_l: false,
    })?;

    assert_eq!(run.status, PotScfContourRunStatus::NeedsMoreSourcePoints);
    assert_eq!(run.energy_points_used, point_count);
    assert_complex_close(run.current_energy, energy_grid[2]);
    assert!(run.total_electron_count.is_finite());
    Ok(())
}

#[test]
fn ldos_rhol_density_grid_feeds_ff2rho_orientation() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let base_regular_large = [
        Complex::new(0.11, 0.02),
        Complex::new(0.18, -0.03),
        Complex::new(0.29, 0.04),
        Complex::new(0.37, 0.08),
        Complex::new(0.43, -0.02),
        Complex::new(0.51, 0.05),
        Complex::new(0.58, -0.01),
    ];
    let base_regular_small = [
        Complex::new(0.015, -0.004),
        Complex::new(0.024, 0.006),
        Complex::new(0.031, -0.005),
        Complex::new(0.038, 0.007),
        Complex::new(0.046, -0.003),
        Complex::new(0.053, 0.004),
        Complex::new(0.061, -0.002),
    ];
    let base_irregular_large = [
        Complex::new(-0.42, 0.31),
        Complex::new(-0.33, 0.24),
        Complex::new(-0.21, 0.19),
        Complex::new(-0.08, 0.12),
        Complex::new(0.04, 0.08),
        Complex::new(0.13, 0.03),
        Complex::new(0.20, -0.01),
    ];
    let base_irregular_small = [
        Complex::new(-0.035, 0.016),
        Complex::new(-0.026, 0.013),
        Complex::new(-0.018, 0.009),
        Complex::new(-0.009, 0.006),
        Complex::new(0.003, 0.004),
        Complex::new(0.011, 0.002),
        Complex::new(0.019, -0.001),
    ];
    let wave_numbers = Array1::from_vec(vec![Complex::new(0.74, 0.08), Complex::new(0.81, 0.03)]);
    let mut regular_large = Array3::<Complex>::zeros((2, 2, 7));
    let mut regular_small = Array3::<Complex>::zeros((2, 2, 7));
    let mut irregular_large = Array3::<Complex>::zeros((2, 2, 7));
    let mut irregular_small = Array3::<Complex>::zeros((2, 2, 7));

    for energy_index in 0..2 {
        for angular in 0..2 {
            let energy_scale = energy_index as Real;
            let angular_scale = angular as Real;
            for radial in 0..7 {
                regular_large[(energy_index, angular, radial)] =
                    base_regular_large[radial] * (1.0 + 0.07 * energy_scale + 0.03 * angular_scale);
                regular_small[(energy_index, angular, radial)] =
                    base_regular_small[radial] * (1.0 + 0.05 * energy_scale - 0.02 * angular_scale);
                irregular_large[(energy_index, angular, radial)] = base_irregular_large[radial]
                    * (1.0 - 0.04 * energy_scale + 0.06 * angular_scale);
                irregular_small[(energy_index, angular, radial)] = base_irregular_small[radial]
                    * (1.0 + 0.02 * energy_scale + 0.01 * angular_scale);
            }
        }
    }

    let grid = ldos_rhol_density_grid(LdosRholDensityGridInput {
        radii: radii.view(),
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        wave_numbers: wave_numbers.view(),
        radial_step: 0.05,
        norman_radius: 0.64,
        angular_count: 2,
    })?;

    assert_eq!(grid.scattering_ldos.dim(), (2, 2));
    assert_eq!(grid.embedded_ldos.dim(), (2, 2));
    let regular_large_energy = regular_large.index_axis(ndarray::Axis(0), 1);
    let regular_small_energy = regular_small.index_axis(ndarray::Axis(0), 1);
    let irregular_large_energy = irregular_large.index_axis(ndarray::Axis(0), 1);
    let irregular_small_energy = irregular_small.index_axis(ndarray::Axis(0), 1);
    let expected = ldos_rhol_density(LdosRholDensityInput {
        radii: radii.view(),
        regular_large: regular_large_energy.index_axis(ndarray::Axis(0), 0),
        regular_small: regular_small_energy.index_axis(ndarray::Axis(0), 0),
        irregular_large: irregular_large_energy.index_axis(ndarray::Axis(0), 0),
        irregular_small: irregular_small_energy.index_axis(ndarray::Axis(0), 0),
        radial_step: 0.05,
        norman_radius: 0.64,
        wave_number: wave_numbers[1],
        angular_momentum: 0,
    })?;
    assert_complex_close(grid.scattering_ldos[(0, 1)], expected.scattering_ldos);
    assert_close(grid.embedded_ldos[(0, 1)], expected.embedded_ldos);
    assert_complex_close(grid.density_scale[(0, 1)], expected.density_scale);

    let energy_grid = Array1::from_vec(vec![Complex::new(0.5, 0.01), Complex::new(0.75, 0.01)]);
    let trace = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex::new(0.2, 0.3),
            Complex::new(-0.1, 0.4),
            Complex::new(0.5, -0.2),
            Complex::new(0.3, 0.1),
        ],
    )
    .unwrap();
    let tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: energy_grid.view(),
        embedded_ldos: grid.embedded_ldos.view(),
        scattering_ldos: grid.scattering_ldos.view(),
        scattering_trace: trace.view(),
        angular_count: 2,
        apply_scattering: true,
    })?;

    let expected_density =
        grid.embedded_ldos[(0, 1)] + (trace[(0, 1)] * grid.scattering_ldos[(0, 1)]).im;
    assert_close(tables.rhoc_density[(1, 0)], grid.embedded_ldos[(0, 1)]);
    assert_close(tables.ldos_density[(1, 0)], expected_density);
    Ok(())
}

#[test]
fn ldos_rhol_exact_radial_tail_matches_feff_rhol_loop() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.9, 1.1, 1.4, 1.8]);
    let tail = ldos_rhol_exact_radial_tail(LdosRholExactRadialTailInput {
        radii: radii.view(),
        start_index_1based: 2,
        angular_momentum: 2,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
    })?;

    assert_eq!(tail.start_index_1based, 2);
    assert_eq!(tail.row_count(), 3);
    assert_complex_close(
        tail.regular_large[0],
        Complex::new(-2.050_540_180_418_029, -0.119_113_885_080_750_09),
    );
    assert_complex_close(
        tail.regular_small[0],
        Complex::new(0.034_421_495_198_108_36, 0.001_532_680_042_627_237),
    );
    assert_complex_close(
        tail.irregular_large[1],
        Complex::new(4.511_076_150_131_777, 3.113_694_562_704_52),
    );
    assert_complex_close(
        tail.irregular_small[2],
        Complex::new(-0.028_114_434_990_648_862, -0.019_179_563_860_727_895),
    );
    Ok(())
}

#[test]
fn ldos_rhol_exact_radial_tail_rejects_invalid_start_row() {
    let radii = Array1::from_vec(vec![0.9, 1.1, 1.4, 1.8]);

    assert_eq!(
        ldos_rhol_exact_radial_tail(LdosRholExactRadialTailInput {
            radii: radii.view(),
            start_index_1based: 5,
            angular_momentum: 2,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
        }),
        Err(DensityError::Rhorrp(
            RhorrpError::ExactRadialTailStartOutOfRange {
                start_index_1based: 5,
                radial_count: 4,
            },
        ))
    );
}

#[test]
fn ldos_rhol_radial_assembly_matches_feff_rhol_vector_steps() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.9, 1.1, 1.4, 1.8]);
    let raw_regular_large = Array1::from_vec(vec![
        Complex::new(0.3, 0.2),
        Complex::new(0.35, 0.24),
        Complex::new(0.41, 0.29),
        Complex::new(0.5, 0.33),
    ]);
    let raw_regular_small = Array1::from_vec(vec![
        Complex::new(-0.01, 0.04),
        Complex::new(-0.012, 0.05),
        Complex::new(-0.014, 0.055),
        Complex::new(-0.016, 0.06),
    ]);
    let raw_irregular_large = Array1::from_vec(vec![
        Complex::new(0.7, -0.2),
        Complex::new(0.68, -0.18),
        Complex::new(0.62, -0.16),
        Complex::new(0.58, -0.12),
    ]);
    let raw_irregular_small = Array1::from_vec(vec![
        Complex::new(0.02, 0.03),
        Complex::new(0.024, 0.035),
        Complex::new(0.027, 0.038),
        Complex::new(0.03, 0.042),
    ]);

    let assembled = ldos_rhol_assemble_radial_components(LdosRholRadialAssemblyInput {
        radii: radii.view(),
        raw_regular_large: raw_regular_large.view(),
        raw_regular_small: raw_regular_small.view(),
        raw_irregular_large: raw_irregular_large.view(),
        raw_irregular_small: raw_irregular_small.view(),
        phase_shift: Complex::new(0.2, -0.1),
        phase_amplitude: Complex::new(1.25, -0.4),
        wave_number: Complex::new(0.4, 0.5),
        angular_momentum: 2,
        match_index_1based: 2,
        exact_tail_start_index_1based: 3,
    })?;

    assert_eq!(assembled.row_count(), 4);
    assert_complex_close(
        assembled.regular_solution_scale,
        Complex::new(0.725_689_404_934_687_9, 0.232_220_609_579_100_12),
    );
    assert_complex_close(
        assembled.irregular_reciprocal_wave_scale,
        Complex::new(-0.364_154_368_523_117_8, -0.078_106_459_632_476_4),
    );
    assert_complex_close(
        assembled.regular_large[0],
        Complex::new(0.171_262_699_564_586_34, 0.214_804_063_860_667_6),
    );
    assert_complex_close(
        assembled.regular_small[0],
        Complex::new(-0.016_545_718_432_510_886, 0.026_705_370_101_596_52),
    );
    assert_complex_close(
        assembled.irregular_large[0],
        Complex::new(0.082_203_861_642_917_79, 0.210_995_197_860_541_83),
    );
    assert_complex_close(
        assembled.irregular_small[0],
        Complex::new(-0.024_096_406_051_351_62, -0.001_936_174_802_402_815),
    );
    assert_complex_close(
        assembled.regular_large[2],
        Complex::new(-1.239_107_687_370_063_5, -0.080_448_475_098_744_81),
    );
    assert_complex_close(
        assembled.irregular_large[3],
        Complex::new(2.405_099_072_873_88, 2.034_179_702_325_563),
    );
    Ok(())
}

#[test]
fn ldos_rhol_radial_assembly_keeps_s_wave_origin_unsmoothed() -> Result<(), DensityError> {
    let row_count = 120;
    let radii = Array1::from_iter((0..row_count).map(|row| 0.2 + 0.035 * row as Real));
    let raw_regular_large = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(0.20 + 0.004 * row, 0.05 + 0.002 * row)
    }));
    let raw_regular_small = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(-0.010 + 0.0002 * row, 0.020 + 0.0003 * row)
    }));
    let raw_irregular_large = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(0.70 - 0.001 * row, -0.20 + 0.002 * row)
    }));
    let raw_irregular_small = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(0.020 + 0.0001 * row, 0.030 + 0.0002 * row)
    }));

    let ldos = ldos_rhol_assemble_radial_components(LdosRholRadialAssemblyInput {
        radii: radii.view(),
        raw_regular_large: raw_regular_large.view(),
        raw_regular_small: raw_regular_small.view(),
        raw_irregular_large: raw_irregular_large.view(),
        raw_irregular_small: raw_irregular_small.view(),
        phase_shift: Complex::new(0.2, -0.1),
        phase_amplitude: Complex::new(1.25, -0.4),
        wave_number: Complex::new(0.4, 0.5),
        angular_momentum: 0,
        match_index_1based: 80,
        exact_tail_start_index_1based: 101,
    })?;
    let rhorrp = rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
        radii: radii.as_slice().expect("contiguous test radii"),
        raw_regular_large: raw_regular_large.view(),
        raw_regular_small: raw_regular_small.view(),
        raw_irregular_large: raw_irregular_large.view(),
        raw_irregular_small: raw_irregular_small.view(),
        phase_shift: Complex::new(0.2, -0.1),
        phase_amplitude: Complex::new(1.25, -0.4),
        wave_number: Complex::new(0.4, 0.5),
        angular_momentum: 0,
        match_index_1based: 80,
        exact_tail_start_index_1based: 101,
    })?;

    assert!(rhorrp.irregular_origin_smoothed);
    assert_ne!(
        ldos.irregular_large[0],
        rhorrp.irregular_large_components[0]
    );
    assert_ne!(
        ldos.irregular_small[0],
        rhorrp.irregular_small_components[0]
    );
    Ok(())
}

#[test]
fn ldos_rhol_radial_assembly_rejects_invalid_match_row() {
    let radii = Array1::from_vec(vec![0.9, 1.1, 1.4, 1.8]);
    let values = Array1::from_vec(vec![Complex::new(0.1, 0.0); 4]);

    assert_eq!(
        ldos_rhol_assemble_radial_components(LdosRholRadialAssemblyInput {
            radii: radii.view(),
            raw_regular_large: values.view(),
            raw_regular_small: values.view(),
            raw_irregular_large: values.view(),
            raw_irregular_small: values.view(),
            phase_shift: Complex::new(0.2, -0.1),
            phase_amplitude: Complex::new(1.25, -0.4),
            wave_number: Complex::new(0.4, 0.5),
            angular_momentum: 2,
            match_index_1based: 0,
            exact_tail_start_index_1based: 3,
        }),
        Err(DensityError::InvalidIndex {
            name: "ldos_rhol_match_index_1based",
            index: 0,
        })
    );
}

#[test]
fn ldos_rhol_channel_composes_feff_dfovrg_flow() -> Result<(), DensityError> {
    let reference = reference_ldos_rhol_channel_inputs();
    let wave_number = ldos_channel_wave_number_from_kinetic_energy(reference.energy);
    let ldos = ldos_rhol_channel(LdosRholChannelInput {
        solver: reference.to_input(),
        angular_momentum: 1,
        wave_number,
    })?;
    let rhorrp = rhorrp_wavefunction_channel(RhorrpWavefunctionChannelInput {
        solver: reference.to_input(),
        angular_momentum: 1,
        wave_number,
    })?;

    assert_complex_close(ldos.phase_shift, rhorrp.muffin_tin_match.phase_shift);
    assert_complex_close(
        ldos.phase_amplitude,
        rhorrp.muffin_tin_match.phase_amplitude,
    );
    assert_complex_close(
        ldos.irregular_initial_large,
        rhorrp.irregular_initial_condition.large_component,
    );
    assert_complex_close(
        ldos.irregular_initial_small,
        rhorrp.irregular_initial_condition.small_component,
    );
    assert_eq!(ldos.regular_active_len, rhorrp.regular_active_len);
    assert_eq!(ldos.irregular_active_len, rhorrp.irregular_active_len);
    assert_eq!(ldos.regular_iteration_count, rhorrp.regular_iteration_count);
    assert_eq!(
        ldos.irregular_iteration_count,
        rhorrp.irregular_iteration_count
    );
    assert_eq!(ldos.difficult_iterations, rhorrp.difficult_iterations);
    assert_eq!(
        ldos.radial_components.row_count(),
        reference.target_last_index + 1
    );
    assert_complex_close(
        ldos.radial_components.regular_large[0],
        rhorrp.radial_solutions.regular_large_components[0],
    );
    assert_complex_close(
        ldos.radial_components.irregular_small[8],
        rhorrp.radial_solutions.irregular_small_components[8],
    );
    assert_complex_close(
        ldos.radial_components.regular_large[ldos.radial_components.row_count() - 1],
        rhorrp.radial_solutions.regular_large_components[reference.target_last_index],
    );
    Ok(())
}

#[test]
fn ldos_rhol_table_driver_solves_grid_and_feeds_ff2rho() -> Result<(), DensityError> {
    let reference = reference_ldos_rhol_channel_inputs();
    let angular_count = 2;
    let energies = Array1::from_vec(vec![
        reference.energy,
        reference.energy + Complex::new(0.035, 0.004),
    ]);
    let wave_numbers = Array1::from_iter(
        energies
            .iter()
            .copied()
            .map(ldos_channel_wave_number_from_kinetic_energy),
    );
    let mut solvers = Vec::new();
    for &energy in energies.iter() {
        for angular in 0..angular_count {
            let base = reference.to_input();
            solvers.push(FovrgDiracSolverInput {
                energy,
                target_kappa: -((angular as i32) + 1),
                ..base
            });
        }
    }
    let scattering_trace =
        Array2::from_shape_fn((angular_count, energies.len()), |(angular, energy)| {
            Complex::new(
                0.11 + 0.025 * angular as Real,
                -0.04 + 0.015 * energy as Real,
            )
        });
    let norman_radius = reference.radii[reference.target_last_index - 3];

    let driver = ldos_rhol_table_driver(LdosRholTableDriverInput {
        solvers: &solvers,
        energy_grid_hartree: energies.view(),
        wave_numbers: wave_numbers.view(),
        scattering_trace: scattering_trace.view(),
        radial_step: reference.to_input().step,
        norman_radius,
        angular_count,
        apply_scattering: true,
    })?;

    let first_channel = ldos_rhol_channel(LdosRholChannelInput {
        solver: solvers[0],
        angular_momentum: 0,
        wave_number: wave_numbers[0],
    })?;
    let first_radii = Array1::from_iter(
        solvers[0]
            .radii
            .iter()
            .take(first_channel.radial_components.row_count())
            .copied(),
    );
    let first_density = ldos_rhol_density(LdosRholDensityInput {
        radii: first_radii.view(),
        regular_large: first_channel.radial_components.regular_large.view(),
        regular_small: first_channel.radial_components.regular_small.view(),
        irregular_large: first_channel.radial_components.irregular_large.view(),
        irregular_small: first_channel.radial_components.irregular_small.view(),
        radial_step: reference.to_input().step,
        norman_radius,
        wave_number: wave_numbers[0],
        angular_momentum: 0,
    })?;
    assert_complex_close(driver.phase_shifts[(0, 0)], first_channel.phase_shift);
    assert_complex_close(
        driver.phase_amplitudes[(0, 0)],
        first_channel.phase_amplitude,
    );
    assert_complex_close(
        driver.density_grid.scattering_ldos[(0, 0)],
        first_density.scattering_ldos,
    );
    assert_close(
        driver.density_grid.embedded_ldos[(0, 0)],
        first_density.embedded_ldos,
    );

    for energy in 0..energies.len() {
        assert_close(
            driver.tables.energy_ev[energy],
            energies[energy].re * FEFF_HARTREE_EV,
        );
        for angular in 0..angular_count {
            let embedded = driver.density_grid.embedded_ldos[(angular, energy)];
            let scattering = driver.density_grid.scattering_ldos[(angular, energy)];
            assert_close(driver.tables.rhoc_density[(energy, angular)], embedded);
            assert_close(
                driver.tables.ldos_density[(energy, angular)],
                embedded + (scattering_trace[(angular, energy)] * scattering).im,
            );
            assert!(driver.regular_iteration_counts[(angular, energy)] > 0);
            assert!(driver.irregular_iteration_counts[(angular, energy)] > 0);
        }
    }
    Ok(())
}

#[test]
fn ldos_rhol_table_driver_rejects_solver_grid_length_mismatch() {
    let reference = reference_ldos_rhol_channel_inputs();
    let energies = Array1::from_vec(vec![reference.energy, reference.energy]);
    let wave_numbers = Array1::from_iter(
        energies
            .iter()
            .copied()
            .map(ldos_channel_wave_number_from_kinetic_energy),
    );
    let scattering_trace = Array2::<Complex>::zeros((1, energies.len()));
    let solvers = vec![reference.to_input()];

    assert_eq!(
        ldos_rhol_table_driver(LdosRholTableDriverInput {
            solvers: &solvers,
            energy_grid_hartree: energies.view(),
            wave_numbers: wave_numbers.view(),
            scattering_trace: scattering_trace.view(),
            radial_step: reference.to_input().step,
            norman_radius: reference.radii[reference.target_last_index] * 0.75,
            angular_count: 1,
            apply_scattering: true,
        }),
        Err(DensityError::LengthMismatch {
            left_name: "ldos_rhol_table_solvers",
            left_len: 1,
            right_name: "rhol_energy*angular_count",
            right_len: 2,
        })
    );
}

#[test]
fn ldos_rhol_wavefunction_tables_selects_potential_and_feeds_ff2rho() -> Result<(), DensityError> {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
    let energy_grid = Array1::from_vec(vec![Complex::new(0.42, 0.03), Complex::new(0.56, 0.02)]);
    let wavefunctions = sample_ldos_rhorrp_wavefunction_tables();
    let scattering_trace = Array2::from_shape_fn((2, 2), |(angular, energy)| {
        Complex::new(
            0.12 + 0.015 * angular as Real,
            -0.03 + 0.02 * energy as Real,
        )
    });

    let actual = ldos_rhol_wavefunction_tables(LdosRholWavefunctionTablesInput {
        wavefunctions: &wavefunctions,
        radii: radii.view(),
        potential_index: 1,
        energy_grid_hartree: energy_grid.view(),
        scattering_trace: scattering_trace.view(),
        radial_step: 0.05,
        norman_radius: 0.64,
        angular_count: 2,
        apply_scattering: true,
    })?;

    let expected_density = ldos_rhol_density_grid(LdosRholDensityGridInput {
        radii: radii.view(),
        regular_large: wavefunctions.regular_large.index_axis(Axis(3), 1),
        regular_small: wavefunctions.regular_small.index_axis(Axis(3), 1),
        irregular_large: wavefunctions.irregular_large.index_axis(Axis(3), 1),
        irregular_small: wavefunctions.irregular_small.index_axis(Axis(3), 1),
        wave_numbers: wavefunctions.wave_numbers.index_axis(Axis(1), 1),
        radial_step: 0.05,
        norman_radius: 0.64,
        angular_count: 2,
    })?;
    let expected_tables = ldos_ff2rho_tables(LdosFf2rhoInput {
        energy_grid_hartree: energy_grid.view(),
        embedded_ldos: expected_density.embedded_ldos.view(),
        scattering_ldos: expected_density.scattering_ldos.view(),
        scattering_trace: scattering_trace.view(),
        angular_count: 2,
        apply_scattering: true,
    })?;

    assert_eq!(actual.wave_numbers, wavefunctions.wave_numbers.column(1));
    assert_eq!(actual.density_grid, expected_density);
    assert_eq!(actual.tables, expected_tables);
    Ok(())
}

#[test]
fn ldos_rhol_wavefunction_tables_rejects_bad_potential_index() {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82]);
    let energy_grid = Array1::from_vec(vec![Complex::new(0.42, 0.03), Complex::new(0.56, 0.02)]);
    let wavefunctions = sample_ldos_rhorrp_wavefunction_tables();
    let scattering_trace = Array2::<Complex>::zeros((2, 2));

    assert_eq!(
        ldos_rhol_wavefunction_tables(LdosRholWavefunctionTablesInput {
            wavefunctions: &wavefunctions,
            radii: radii.view(),
            potential_index: 2,
            energy_grid_hartree: energy_grid.view(),
            scattering_trace: scattering_trace.view(),
            radial_step: 0.05,
            norman_radius: 0.64,
            angular_count: 2,
            apply_scattering: true,
        }),
        Err(DensityError::InvalidPotentialIndex {
            name: "ldos_rhol_wavefunction_tables_potential",
            index: 2,
            available: 2,
        })
    );
}

#[test]
fn ldos_rhol_density_rejects_radial_component_length_mismatch() {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34]);
    let regular_large = Array1::from_vec(vec![Complex::new(0.1, 0.0); 3]);
    let regular_small = Array1::from_vec(vec![Complex::new(0.01, 0.0); 4]);
    let irregular_large = Array1::from_vec(vec![Complex::new(-0.1, 0.02); 4]);
    let irregular_small = Array1::from_vec(vec![Complex::new(-0.01, 0.002); 4]);

    assert_eq!(
        ldos_rhol_density(LdosRholDensityInput {
            radii: radii.view(),
            regular_large: regular_large.view(),
            regular_small: regular_small.view(),
            irregular_large: irregular_large.view(),
            irregular_small: irregular_small.view(),
            radial_step: 0.05,
            norman_radius: 0.3,
            wave_number: Complex::new(0.74, 0.08),
            angular_momentum: 1,
        }),
        Err(DensityError::LengthMismatch {
            left_name: "radii",
            left_len: 4,
            right_name: "regular_large",
            right_len: 3,
        })
    );
}

#[test]
fn ldos_rhol_density_grid_rejects_short_radial_source_table() {
    let radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82]);
    let wave_numbers = Array1::from_vec(vec![Complex::new(0.74, 0.08), Complex::new(0.81, 0.03)]);
    let regular_large = Array3::<Complex>::zeros((1, 2, 6));
    let regular_small = Array3::<Complex>::zeros((2, 2, 6));
    let irregular_large = Array3::<Complex>::zeros((2, 2, 6));
    let irregular_small = Array3::<Complex>::zeros((2, 2, 6));

    assert_eq!(
        ldos_rhol_density_grid(LdosRholDensityGridInput {
            radii: radii.view(),
            regular_large: regular_large.view(),
            regular_small: regular_small.view(),
            irregular_large: irregular_large.view(),
            irregular_small: irregular_small.view(),
            wave_numbers: wave_numbers.view(),
            radial_step: 0.05,
            norman_radius: 0.64,
            angular_count: 2,
        }),
        Err(DensityError::CubeShapeTooSmall {
            name: "regular_large",
            rows: 1,
            columns: 2,
            depth: 6,
            required_rows: 2,
            required_columns: 2,
            required_depth: 6,
        })
    );
}

#[test]
fn ldos_spin_ff2rho_tables_match_feff_spin_resolved_order() -> Result<(), DensityError> {
    let energy = Array1::from_vec(vec![Complex::new(0.5, 0.01), Complex::new(0.75, 0.01)]);
    let mut embedded = Array3::<Real>::zeros((4, 2, 2));
    for angular in 0..4 {
        for spin in 0..2 {
            for energy_index in 0..2 {
                embedded[(angular, spin, energy_index)] =
                    1.0 + angular as Real + 10.0 * spin as Real + 0.5 * energy_index as Real;
            }
        }
    }
    let mut scattering = Array3::<Complex>::zeros((4, 2, 2));
    let mut trace = Array3::<Complex>::zeros((4, 2, 2));
    scattering[(0, 0, 0)] = Complex::new(1.5, -0.4);
    trace[(0, 0, 0)] = Complex::new(0.2, 0.3);
    scattering[(0, 1, 0)] = Complex::new(-0.2, 0.6);
    trace[(0, 1, 0)] = Complex::new(0.5, -0.1);
    scattering[(3, 1, 1)] = Complex::new(0.75, 0.25);
    trace[(3, 1, 1)] = Complex::new(-0.4, 0.8);

    let tables = ldos_spin_ff2rho_tables(LdosSpinFf2rhoInput {
        energy_grid_hartree: energy.view(),
        embedded_ldos: embedded.view(),
        scattering_ldos: scattering.view(),
        scattering_trace: trace.view(),
        angular_count: 4,
        apply_scattering: true,
    })?;

    assert_close(tables.energy_ev[0], 0.5 * FEFF_HARTREE_EV);
    assert_eq!(tables.ldos_density.ncols(), 8);
    assert_close(tables.rhoc_density[(0, 0)], 1.0);
    assert_close(tables.rhoc_density[(0, 4)], 11.0);
    assert_close(tables.ldos_density[(0, 0)], 1.37);
    assert_close(tables.ldos_density[(0, 4)], 11.32);
    assert_close(tables.ldos_density[(1, 7)], 14.5 + 0.5);
    Ok(())
}

#[test]
fn ldos_hubbard_magnetic_ff2rho_tables_match_feff_step2_order() -> Result<(), DensityError> {
    let energy = Array1::from_vec(vec![Complex::new(0.5, 0.01), Complex::new(0.75, 0.01)]);
    let mut embedded = Array4::<Real>::zeros((2, 4, 2, 2));
    for angular in 0..2 {
        for magnetic in (angular * angular)..((angular + 1) * (angular + 1)) {
            for spin in 0..2 {
                for energy_index in 0..2 {
                    embedded[(angular, magnetic, spin, energy_index)] = 1.0
                        + angular as Real
                        + magnetic as Real
                        + 10.0 * spin as Real
                        + 0.5 * energy_index as Real;
                }
            }
        }
    }
    embedded[(0, 0, 0, 0)] = 2.0;
    embedded[(1, 2, 1, 0)] = 10.0;

    let mut scattering = Array4::<Complex>::zeros((2, 4, 2, 2));
    let mut trace = Array4::<Complex>::zeros((2, 4, 2, 2));
    scattering[(0, 0, 0, 0)] = Complex::new(1.5, -0.4);
    trace[(0, 0, 0, 0)] = Complex::new(0.2, 0.3);
    scattering[(1, 2, 1, 0)] = Complex::new(1.0, -0.2);
    trace[(1, 2, 1, 0)] = Complex::new(0.5, 0.25);

    let tables = ldos_hubbard_magnetic_ff2rho_tables(LdosHubbardMagneticFf2rhoInput {
        energy_grid_hartree: energy.view(),
        embedded_magnetic_ldos: embedded.view(),
        scattering_magnetic_ldos: scattering.view(),
        magnetic_scattering_trace: trace.view(),
        angular_count: 2,
    })?;

    assert_close(tables.energy_ev[0], 0.5 * FEFF_HARTREE_EV);
    assert_close(tables.energy_ev[1], 0.75 * FEFF_HARTREE_EV);
    assert_eq!(tables.lmdos_density.ncols(), 8);
    assert_eq!(tables.rhocm_density.ncols(), 8);
    assert_close(tables.rhocm_density[(0, 0)], 2.0);
    assert_close(tables.lmdos_density[(0, 0)], 2.37);
    assert_close(tables.rhocm_density[(0, 6)], 10.0);
    assert_close(tables.lmdos_density[(0, 6)], 10.0 / 3.0 + 0.15);
    assert_close(tables.rhocm_density[(1, 7)], 15.5);
    assert_close(tables.lmdos_density[(1, 7)], 15.5 / 3.0);
    Ok(())
}

#[test]
fn valence_density_update_rejects_invalid_inputs() {
    let sample = sample_ff2g_state();
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            energy_index: 0,
            ..sample.input()
        }),
        Err(DensityError::InvalidIndex {
            name: "energy_index",
            index: 0,
        })
    );
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            last_radial_index: 0,
            ..sample.input()
        }),
        Err(DensityError::InvalidIndex {
            name: "last_radial_index",
            index: 0,
        })
    );

    let short_ldos = Array1::<Complex>::zeros(2);
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            scattering_ldos: short_ldos.view(),
            ..sample.input()
        }),
        Err(DensityError::LengthMismatch {
            left_name: "scattering_trace",
            left_len: 4,
            right_name: "scattering_ldos",
            right_len: 2,
        })
    );

    let short_density = Array1::<Complex>::zeros(3);
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            embedded_density: short_density.view(),
            ..sample.input()
        }),
        Err(DensityError::LengthTooShort {
            name: "embedded_density",
            required: 5,
            actual: 3,
        })
    );

    let small_matrix = Array2::<Complex>::zeros((2, 2));
    assert_eq!(
        update_valence_density(ValenceDensityUpdateInput {
            embedded_ldos: small_matrix.view(),
            ..sample.input()
        }),
        Err(DensityError::ShapeTooSmall {
            name: "embedded_ldos",
            rows: 2,
            columns: 2,
            required_rows: 4,
            required_columns: 2,
        })
    );

    let mut bad_trace = sample.scattering_trace.clone();
    bad_trace[1] = Complex32::new(f32::NAN, 0.0);
    assert!(matches!(
        update_valence_density(ValenceDensityUpdateInput {
            scattering_trace: bad_trace.view(),
            ..sample.input()
        }),
        Err(DensityError::NonFiniteComplexValue {
            name: "scattering_trace",
            index: 1,
            ..
        })
    ));
}

#[test]
fn ldos_ff2rho_tables_reject_invalid_inputs() {
    let energy = Array1::from_vec(vec![Complex::new(0.5, 0.01)]);
    let embedded = Array2::from_shape_vec((2, 1), vec![1.0, 3.0]).unwrap();
    let short = Array2::<Complex>::zeros((1, 1));

    assert_eq!(
        ldos_ff2rho_tables(LdosFf2rhoInput {
            energy_grid_hartree: energy.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: short.view(),
            scattering_trace: short.view(),
            angular_count: 2,
            apply_scattering: true,
        }),
        Err(DensityError::ShapeTooSmall {
            name: "scattering_ldos",
            rows: 1,
            columns: 1,
            required_rows: 2,
            required_columns: 1,
        })
    );

    let bad_energy = Array1::from_vec(vec![Complex::new(f64::NAN, 0.0)]);
    assert!(matches!(
        ldos_ff2rho_tables(LdosFf2rhoInput {
            energy_grid_hartree: bad_energy.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: short.view(),
            scattering_trace: short.view(),
            angular_count: 1,
            apply_scattering: false,
        }),
        Err(DensityError::NonFiniteComplexValue {
            name: "energy_grid_hartree",
            index: 0,
            ..
        })
    ));
}

#[test]
fn broyden_density_mix_matches_feff_broydn_reference() -> Result<(), DensityError> {
    let sample = sample_broydn_state();
    let references = broydn_references();
    let mut workspace = BroydenWorkspace::zeros(4, 2);
    let mut norman_charges = sample.norman_charges.clone();

    for reference in references {
        let input_density = sample.valence_density_for_iteration(reference.iteration);
        let result = mix_broyden_density(sample.input(
            reference.iteration,
            norman_charges.view(),
            input_density.view(),
            &workspace,
        ))?;

        for potential in 0..=1 {
            assert_broydn_grid_values(
                &result.valence_density,
                potential,
                sample.last_indices[potential],
                reference.valence_density[potential],
            );
            assert_close(
                result.charge_deltas[potential],
                reference.charge_deltas[potential],
            );
            assert_close(
                result.norman_charges[potential],
                reference.norman_charges[potential],
            );
        }
        assert_close(
            result.workspace.norms[reference.iteration - 1],
            reference.norm,
        );
        for (column, expected) in reference.coefficients.into_iter().enumerate() {
            assert_close(
                result.workspace.coefficients[(reference.iteration - 1, column)],
                expected,
            );
        }
        assert_close(
            result.workspace.previous_density[(0, 0)],
            reference.previous_density_1_0,
        );

        workspace = result.workspace;
        norman_charges = result.norman_charges;
    }

    Ok(())
}

#[test]
fn broyden_density_mix_rejects_invalid_inputs() {
    let sample = sample_broydn_state();
    let workspace = BroydenWorkspace::zeros(4, 2);
    let input_density = sample.valence_density_for_iteration(1);

    assert_eq!(
        mix_broyden_density(BroydenMixInput {
            iteration: 0,
            ..sample.input(
                1,
                sample.norman_charges.view(),
                input_density.view(),
                &workspace,
            )
        }),
        Err(DensityError::InvalidIndex {
            name: "iteration",
            index: 0,
        })
    );

    let bad_last_indices = Array1::from_vec(vec![190, 0]);
    assert_eq!(
        mix_broyden_density(BroydenMixInput {
            last_indices: bad_last_indices.view(),
            ..sample.input(
                1,
                sample.norman_charges.view(),
                input_density.view(),
                &workspace,
            )
        }),
        Err(DensityError::InvalidIndex {
            name: "last_indices",
            index: 0,
        })
    );

    let zero_occupancy = Array2::<Real>::zeros((3, 2));
    assert_eq!(
        mix_broyden_density(BroydenMixInput {
            valence_occupancy: zero_occupancy.view(),
            ..sample.input(
                1,
                sample.norman_charges.view(),
                input_density.view(),
                &workspace,
            )
        }),
        Err(DensityError::ZeroScalar {
            name: "broyden_total_fermi_count",
            value: 0.0,
        })
    );

    let short_workspace = BroydenWorkspace::zeros(1, 2);
    let second_density = sample.valence_density_for_iteration(2);
    assert_eq!(
        mix_broyden_density(sample.input(
            2,
            sample.norman_charges.view(),
            second_density.view(),
            &short_workspace,
        )),
        Err(DensityError::ShapeTooSmall {
            name: "workspace.coefficients",
            rows: 1,
            columns: 1,
            required_rows: 2,
            required_columns: 2,
        })
    );
}

#[test]
fn coulomb_update_matches_feff_coulom_norman_reference() -> Result<(), DensityError> {
    let sample = sample_coulom_state();
    let result = update_coulomb_potential(CoulombPotentialUpdateInput {
        mode: CoulombUpdateMode::Norman,
        ..sample.input()
    })?;

    assert_coulom_values(
        &result.coulomb_potential,
        0,
        [
            -1.775_572_357_598_355,
            -1.771_572_355_686_939_8,
            -1.523_562_494_397_465,
            -1.201_342_285_954_037_5,
            0.0,
            0.0,
        ],
    );
    assert_coulom_values(
        &result.coulomb_potential,
        1,
        [
            -1.995_771_090_609_550_5,
            -1.991_771_087_961_443_9,
            -1.743_757_425_966_650_6,
            -1.460_139_524_464_028_5,
            0.0,
            0.0,
        ],
    );
    Ok(())
}

#[test]
fn coulomb_update_matches_feff_coulom_long_range_reference() -> Result<(), DensityError> {
    let sample = sample_coulom_state();
    let result = update_coulomb_potential(CoulombPotentialUpdateInput {
        mode: CoulombUpdateMode::LongRange,
        ..sample.input()
    })?;

    assert_coulom_values(
        &result.coulomb_potential,
        0,
        [
            -1.593_233_968_914_050_2,
            -1.589_233_967_002_635,
            -1.341_224_105_713_160_2,
            -1.019_003_897_269_732_6,
            0.0,
            0.0,
        ],
    );
    assert_coulom_values(
        &result.coulomb_potential,
        1,
        [
            -2.000_638_675_823_475_3,
            -1.996_638_673_175_368_7,
            -1.748_625_011_180_575_5,
            -1.465_007_109_677_953_3,
            0.0,
            0.0,
        ],
    );
    Ok(())
}

#[test]
fn scf_density_step_composes_broyden_and_coulomb_updates() -> Result<(), DensityError> {
    let sample = sample_coulom_state();
    let valence_occupancy = Array2::from_shape_vec((3, 2), vec![1.2, 1.5, 0.4, 0.7, 0.0, 0.2])
        .expect("valid valence occupancy shape");
    let potential_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let norman_charges = Array1::from_vec(vec![1.4, 2.1]);
    let workspace = BroydenWorkspace::zeros(3, 2);
    let accelerator = 0.0;

    let mixed = mix_broyden_density(BroydenMixInput {
        iteration: 1,
        accelerator,
        highest_potential_index: 1,
        valence_occupancy: valence_occupancy.view(),
        last_indices: sample.last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        norman_radii: sample.norman_radii.view(),
        norman_charges: norman_charges.view(),
        overlapped_valence_density: sample.overlapped_valence_density.view(),
        valence_density: sample.valence_density.view(),
        workspace: &workspace,
    })?;
    let updated = update_coulomb_potential(CoulombPotentialUpdateInput {
        mode: CoulombUpdateMode::Norman,
        highest_potential_index: 1,
        last_indices: sample.last_indices.view(),
        valence_density: mixed.valence_density.view(),
        overlapped_valence_density: sample.overlapped_valence_density.view(),
        overlapped_density: sample.overlapped_density.view(),
        atom_positions: sample.atom_positions.view(),
        representative_atoms: sample.representative_atoms.view(),
        atom_potentials: sample.atom_potentials.view(),
        norman_radii: sample.norman_radii.view(),
        charge_deltas: mixed.charge_deltas.view(),
        atomic_numbers: sample.atomic_numbers.view(),
        coulomb_potential: sample.coulomb_potential.view(),
    })?;

    let step = update_scf_density_potential(ScfDensityStepInput {
        iteration: 1,
        accelerator,
        coulomb_mode: CoulombUpdateMode::Norman,
        highest_potential_index: 1,
        valence_occupancy: valence_occupancy.view(),
        last_indices: sample.last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        norman_radii: sample.norman_radii.view(),
        norman_charges: norman_charges.view(),
        overlapped_valence_density: sample.overlapped_valence_density.view(),
        integrated_valence_density: sample.valence_density.view(),
        workspace: &workspace,
        overlapped_density: sample.overlapped_density.view(),
        atom_positions: sample.atom_positions.view(),
        representative_atoms: sample.representative_atoms.view(),
        atom_potentials: sample.atom_potentials.view(),
        atomic_numbers: sample.atomic_numbers.view(),
        coulomb_potential: sample.coulomb_potential.view(),
    })?;

    assert_eq!(step.valence_density, mixed.valence_density);
    assert_eq!(step.charge_deltas, mixed.charge_deltas);
    assert_eq!(step.norman_charges, mixed.norman_charges);
    assert_eq!(step.coulomb_potential, updated.coulomb_potential);
    assert_eq!(step.workspace, mixed.workspace);
    Ok(())
}

#[test]
fn pot_scf_iteration_composes_contour_mixing_coulomb_and_density_update() -> Result<(), DensityError>
{
    let angular_count = 3;
    let potential_count = 1;
    let radial_count = OVRLP_DENSITY_POINTS;
    let energy_grid = Array1::from_vec(vec![Complex::new(0.20, 0.01)]);
    let steps = Array1::from_vec(vec![0.01]);
    let source_energies = Array1::from_vec(vec![energy_grid[0], energy_grid[0] - steps[0]]);
    let last_indices = Array1::from_vec(vec![140]);
    let potential_multiplicities = Array1::from_vec(vec![1.0]);
    let scattering_trace = Array3::<Complex32>::zeros((2, angular_count, potential_count));
    let scattering_ldos = Array3::<Complex>::zeros((2, angular_count, potential_count));
    let mut embedded_ldos_source = Array3::<Complex>::zeros((2, angular_count, potential_count));
    let scattering_density =
        Array4::<Complex>::zeros((2, radial_count, angular_count, potential_count));
    let embedded_density_source =
        Array3::from_shape_fn((2, radial_count, potential_count), |(point, radial, _)| {
            Complex::new((0.08 + 0.0005 * radial as Real) * (point + 1) as Real, 0.0)
        });
    for angular in 0..angular_count {
        embedded_ldos_source[(0, angular, 0)] = Complex::new(1.0 + 0.20 * angular as Real, 0.0);
        embedded_ldos_source[(1, angular, 0)] = Complex::new(2.0 + 0.40 * angular as Real, 0.0);
    }

    let point1_embedded_ldos = embedded_ldos_source.index_axis(Axis(0), 0).to_owned();
    let point1_embedded_density = embedded_density_source.index_axis(Axis(0), 0).to_owned();
    let point1 = accumulate_pot_scf_energy_point(PotScfEnergyPointInput {
        energy_index: 1,
        current_energy: source_energies[0],
        previous_energy: Complex::new(source_energies[0].re, 0.0),
        current_floor: 1,
        previous_floor: 1,
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.index_axis(Axis(0), 0),
        scattering_ldos: scattering_ldos.index_axis(Axis(0), 0),
        embedded_ldos: point1_embedded_ldos.view(),
        previous_ldos: Array2::<Complex>::zeros((angular_count, potential_count)).view(),
        scattering_density: scattering_density.index_axis(Axis(0), 0),
        embedded_density: point1_embedded_density.view(),
        previous_density: Array2::<Complex>::zeros((radial_count, potential_count)).view(),
        valence_density: Array2::<Real>::zeros((radial_count, potential_count)).view(),
        occupancy_by_l: Array2::<Real>::zeros((angular_count, potential_count)).view(),
        include_high_l: false,
    })?;

    let point2_embedded_ldos = embedded_ldos_source.index_axis(Axis(0), 1).to_owned();
    let point2_embedded_density = embedded_density_source.index_axis(Axis(0), 1).to_owned();
    let point2 = accumulate_pot_scf_energy_point(PotScfEnergyPointInput {
        energy_index: 2,
        current_energy: source_energies[1],
        previous_energy: source_energies[0],
        current_floor: 1,
        previous_floor: 1,
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.index_axis(Axis(0), 1),
        scattering_ldos: scattering_ldos.index_axis(Axis(0), 1),
        embedded_ldos: point2_embedded_ldos.view(),
        previous_ldos: point1.embedded_ldos.view(),
        scattering_density: scattering_density.index_axis(Axis(0), 1),
        embedded_density: point2_embedded_density.view(),
        previous_density: point1.embedded_density.view(),
        valence_density: point1.valence_density.view(),
        occupancy_by_l: point1.occupancy_by_l.view(),
        include_high_l: false,
    })?;
    let electron_count_target = (point1.total_electron_count + point2.total_electron_count) / 2.0;

    let contour_input = PotScfContourRunInput {
        first_scmt_call: false,
        electron_count_target,
        active_energy_count: energy_grid.len(),
        floor_count: steps.len(),
        energy_grid: energy_grid.view(),
        steps: steps.view(),
        source_energies: source_energies.view(),
        highest_potential_index: 0,
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        scattering_trace: scattering_trace.view(),
        scattering_ldos: scattering_ldos.view(),
        embedded_ldos_source: embedded_ldos_source.view(),
        scattering_density: scattering_density.view(),
        embedded_density_source: embedded_density_source.view(),
        include_high_l: false,
    };
    let contour = run_pot_scf_contour(contour_input)?;
    assert_eq!(contour.status, PotScfContourRunStatus::Bracketed);

    let expected_valence_occupancy =
        Array2::from_shape_vec((angular_count, potential_count), vec![1.2, 0.4, 0.1]).unwrap();
    let norman_radii = Array1::from_vec(vec![0.65]);
    let norman_charges = Array1::from_vec(vec![1.4]);
    let workspace = BroydenWorkspace::zeros(3, potential_count);
    let overlapped_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, _)| {
            let radius = (-8.8 + 0.05 * radial as Real).exp();
            80.0 * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
        });
    let overlapped_valence_density = overlapped_density.mapv(|density| 0.42 * density);
    let coulomb_potential =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, _)| {
            -1.7 + 0.004 * (radial + 1) as Real
        });
    let atom_positions = Array2::<Real>::zeros((1, 3));
    let representative_atoms = Array1::from_vec(vec![0]);
    let atom_potentials = Array1::from_vec(vec![0]);
    let atomic_numbers = Array1::from_vec(vec![8]);

    let direct_step = update_scf_density_potential(ScfDensityStepInput {
        iteration: 1,
        accelerator: 0.0,
        coulomb_mode: CoulombUpdateMode::Norman,
        highest_potential_index: 0,
        valence_occupancy: expected_valence_occupancy.view(),
        last_indices: last_indices.view(),
        potential_multiplicities: potential_multiplicities.view(),
        norman_radii: norman_radii.view(),
        norman_charges: norman_charges.view(),
        overlapped_valence_density: overlapped_valence_density.view(),
        integrated_valence_density: contour.valence_density.view(),
        workspace: &workspace,
        overlapped_density: overlapped_density.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        atom_potentials: atom_potentials.view(),
        atomic_numbers: atomic_numbers.view(),
        coulomb_potential: coulomb_potential.view(),
    })?;

    let iteration = run_pot_scf_iteration(PotScfIterationInput {
        contour: contour_input,
        iteration: 1,
        accelerator: 0.0,
        coulomb_mode: CoulombUpdateMode::Norman,
        repeat_on_bad_counts: false,
        expected_valence_occupancy: expected_valence_occupancy.view(),
        norman_radii: norman_radii.view(),
        norman_charges: norman_charges.view(),
        overlapped_valence_density: overlapped_valence_density.view(),
        workspace: &workspace,
        overlapped_density: overlapped_density.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        atom_potentials: atom_potentials.view(),
        atomic_numbers: atomic_numbers.view(),
        coulomb_potential: coulomb_potential.view(),
    })?;

    assert_eq!(iteration.status, PotScfIterationStatus::Updated);
    assert_eq!(iteration.contour, contour);
    assert_eq!(iteration.density_step, Some(direct_step.clone()));
    assert_close(
        iteration.overlapped_density[(0, 0)],
        overlapped_density[(0, 0)] - overlapped_valence_density[(0, 0)]
            + direct_step.valence_density[(0, 0)],
    );
    assert_close(iteration.overlapped_density[(last_indices[0], 0)], 0.0);
    assert_close(
        iteration.overlapped_valence_density[(last_indices[0], 0)],
        0.0,
    );
    assert_close(
        iteration.overlapped_valence_density[(last_indices[0] - 1, 0)],
        overlapped_valence_density[(last_indices[0] - 1, 0)],
    );

    let next_outer = finish_pot_scf_outer_iteration(PotScfOuterIterationInput {
        iteration_result: &iteration,
        iteration: 1,
        max_iterations: 5,
        minimum_iterations: 3,
        previous_fermi_energy: 0.0,
        previous_norman_charges: norman_charges.view(),
        previous_occupancy_by_l: Array2::<Real>::zeros((angular_count, potential_count)).view(),
        expected_valence_occupancy: expected_valence_occupancy.view(),
        ion_charges: atomic_numbers.mapv(|value| value as Real).view(),
        previous_coulomb_potential: coulomb_potential.view(),
        fermi_tolerance: 1.0e-6,
        charge_tolerance: 1.0e-6,
        charge_sum_tolerance: 1.0e-6,
        partial_charge_tolerance: 1.0e-6,
    })?;
    assert_eq!(
        next_outer.status,
        PotScfOuterIterationStatus::NeedsNextIteration
    );
    assert_eq!(
        next_outer.overlapped_valence_density,
        direct_step.valence_density
    );
    assert_eq!(next_outer.coulomb_potential, direct_step.coulomb_potential);

    let initial_scf_state = PotScfState {
        fermi_energy: 0.0,
        norman_charges: norman_charges.clone(),
        norman_charge_reference: norman_charges.clone(),
        occupancy_by_l: Array2::<Real>::zeros((angular_count, potential_count)),
        overlapped_density: overlapped_density.clone(),
        overlapped_valence_density: overlapped_valence_density.clone(),
        coulomb_potential: coulomb_potential.clone(),
        workspace: workspace.clone(),
    };
    let ion_charges = atomic_numbers.mapv(|value| value as Real);
    let advanced = advance_pot_scf_state(PotScfStateAdvanceInput {
        contour: contour_input,
        state: &initial_scf_state,
        iteration: 1,
        max_iterations: 5,
        minimum_iterations: 3,
        accelerator: 0.0,
        coulomb_mode: CoulombUpdateMode::Norman,
        repeat_on_bad_counts: false,
        expected_valence_occupancy: expected_valence_occupancy.view(),
        norman_radii: norman_radii.view(),
        ion_charges: ion_charges.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        atom_potentials: atom_potentials.view(),
        atomic_numbers: atomic_numbers.view(),
        fermi_tolerance: 1.0e-6,
        charge_tolerance: 1.0e-6,
        charge_sum_tolerance: 1.0e-6,
        partial_charge_tolerance: 1.0e-6,
    })?;
    assert_eq!(advanced.iteration, iteration);
    assert_eq!(advanced.outer, next_outer);
    assert_eq!(advanced.state.fermi_energy, next_outer.fermi_energy);
    assert_eq!(advanced.state.norman_charges, direct_step.norman_charges);
    assert_eq!(
        advanced.state.norman_charge_reference,
        next_outer.norman_charge_reference
    );
    assert_eq!(
        advanced.state.occupancy_by_l,
        iteration.contour.occupancy_by_l
    );
    assert_eq!(
        advanced.state.overlapped_valence_density,
        direct_step.valence_density
    );
    assert_eq!(advanced.state.workspace, direct_step.workspace);

    let extended_angular_count = angular_count + 2;
    let mut extended_expected = Array2::<Real>::zeros((extended_angular_count, potential_count));
    for angular in 0..angular_count {
        extended_expected[(angular, 0)] = expected_valence_occupancy[(angular, 0)];
    }
    let extended_iteration = run_pot_scf_iteration(PotScfIterationInput {
        expected_valence_occupancy: extended_expected.view(),
        ..PotScfIterationInput {
            contour: contour_input,
            iteration: 1,
            accelerator: 0.0,
            coulomb_mode: CoulombUpdateMode::Norman,
            repeat_on_bad_counts: false,
            expected_valence_occupancy: expected_valence_occupancy.view(),
            norman_radii: norman_radii.view(),
            norman_charges: norman_charges.view(),
            overlapped_valence_density: overlapped_valence_density.view(),
            workspace: &workspace,
            overlapped_density: overlapped_density.view(),
            atom_positions: atom_positions.view(),
            representative_atoms: representative_atoms.view(),
            atom_potentials: atom_potentials.view(),
            atomic_numbers: atomic_numbers.view(),
            coulomb_potential: coulomb_potential.view(),
        }
    })?;
    assert_eq!(
        extended_iteration.contour.occupancy_by_l.dim(),
        extended_expected.dim()
    );
    for angular in 0..angular_count {
        assert_close(
            extended_iteration.contour.occupancy_by_l[(angular, 0)],
            contour.occupancy_by_l[(angular, 0)],
        );
    }
    for angular in angular_count..extended_angular_count {
        assert_eq!(extended_iteration.contour.occupancy_by_l[(angular, 0)], 0.0);
    }

    let converged_outer = finish_pot_scf_outer_iteration(PotScfOuterIterationInput {
        iteration_result: &iteration,
        iteration: 4,
        max_iterations: 5,
        minimum_iterations: 0,
        previous_fermi_energy: iteration.contour.fermi_energy.unwrap(),
        previous_norman_charges: direct_step.norman_charges.view(),
        previous_occupancy_by_l: iteration.contour.occupancy_by_l.view(),
        expected_valence_occupancy: expected_valence_occupancy.view(),
        ion_charges: ion_charges.view(),
        previous_coulomb_potential: coulomb_potential.view(),
        fermi_tolerance: 1.0,
        charge_tolerance: 1.0,
        charge_sum_tolerance: 1.0e6,
        partial_charge_tolerance: 1.0,
    })?;
    assert_eq!(
        converged_outer.status,
        PotScfOuterIterationStatus::Converged
    );
    assert_eq!(converged_outer.coulomb_potential, coulomb_potential);
    assert_close(
        converged_outer.reported_charge_transfer[0],
        -direct_step.norman_charges[0] + ion_charges[0],
    );
    assert_close(
        converged_outer.overlapped_density[(0, 0)],
        overlapped_density[(0, 0)],
    );
    assert_close(
        converged_outer.overlapped_density[(last_indices[0], 0)],
        0.0,
    );

    let bad_expected = Array2::from_elem((angular_count, potential_count), 100.0);
    let repeat = run_pot_scf_iteration(PotScfIterationInput {
        expected_valence_occupancy: bad_expected.view(),
        repeat_on_bad_counts: true,
        ..PotScfIterationInput {
            contour: contour_input,
            iteration: 1,
            accelerator: 0.0,
            coulomb_mode: CoulombUpdateMode::Norman,
            repeat_on_bad_counts: false,
            expected_valence_occupancy: expected_valence_occupancy.view(),
            norman_radii: norman_radii.view(),
            norman_charges: norman_charges.view(),
            overlapped_valence_density: overlapped_valence_density.view(),
            workspace: &workspace,
            overlapped_density: overlapped_density.view(),
            atom_positions: atom_positions.view(),
            representative_atoms: representative_atoms.view(),
            atom_potentials: atom_potentials.view(),
            atomic_numbers: atomic_numbers.view(),
            coulomb_potential: coulomb_potential.view(),
        }
    })?;
    assert_eq!(repeat.status, PotScfIterationStatus::RepeatRequired);
    assert_eq!(repeat.bad_occupation_count, angular_count);
    assert_eq!(repeat.density_step, None);
    assert_eq!(repeat.overlapped_density, overlapped_density);
    Ok(())
}

#[test]
fn coulomb_update_rejects_invalid_inputs() {
    let sample = sample_coulom_state();
    let short = Array1::<usize>::zeros(1);
    assert_eq!(
        update_coulomb_potential(CoulombPotentialUpdateInput {
            last_indices: short.view(),
            ..sample.input()
        }),
        Err(DensityError::LengthTooShort {
            name: "last_indices",
            required: 2,
            actual: 1,
        })
    );

    let bad_last = Array1::from_vec(vec![140, 252]);
    assert_eq!(
        update_coulomb_potential(CoulombPotentialUpdateInput {
            last_indices: bad_last.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidIndex {
            name: "last_indices",
            index: 252,
        })
    );

    let bad_atoms = Array1::from_vec(vec![0, 2, 1]);
    assert_eq!(
        update_coulomb_potential(CoulombPotentialUpdateInput {
            atom_potentials: bad_atoms.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidPotentialIndex {
            name: "atom_potentials",
            index: 2,
            available: 2,
        })
    );
}

#[test]
fn potential_overlap_matches_feff_ovrlp_explicit_reference() -> Result<(), DensityError> {
    let sample = sample_ovrlp_state();

    let result = overlap_potential_density(PotentialOverlapInput {
        potential_index: 1,
        explicit_overlaps: &[
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
        ],
        ..sample.input()
    })?;

    assert_overlap_grid_values(
        &result.electron_density,
        [
            1.147_581_324_726_077_3e2,
            1.159_343_448_785_123_5e2,
            1.182_847_850_423_736_6e2,
            7.780_182_969_116_142e1,
        ],
    );
    assert_overlap_grid_values(
        &result.valence_density,
        [
            9.268_692_173_721_425e1,
            9.345_625_940_677_17e1,
            9.499_826_182_402_593e1,
            6.839_302_990_519_607e1,
        ],
    );
    assert_overlap_grid_values(
        &result.coulomb_potential,
        [
            -6.116_080_385_415_219,
            -6.053_852_541_970_13,
            -5.705_837_372_088_443,
            -5.416_140_791_110_99,
        ],
    );
    assert_overlap_grid_values(
        &result.spin_density_ratio,
        [
            2.204_636_783_021_805e-4,
            2.803_310_790_607_974_4e-4,
            4.649_794_982_532_802e-4,
            1.015_400_284_461_108_2e-3,
        ],
    );
    assert_close(result.norman_radius.radius, 6.257_226_100_235_719e-1);
    Ok(())
}

#[test]
fn potential_overlap_matches_feff_ovrlp_geometry_reference() -> Result<(), DensityError> {
    let sample = sample_ovrlp_state();

    let result = overlap_potential_density(PotentialOverlapInput {
        potential_index: 0,
        explicit_overlaps: &[],
        ..sample.input()
    })?;

    assert_overlap_grid_values(
        &result.electron_density,
        [
            8.079_917_195_503_039e1,
            8.198_343_896_737_67e1,
            8.480_691_764_162_866e1,
            5.743_104_082_268_246e1,
        ],
    );
    assert_overlap_grid_values(
        &result.valence_density,
        [
            6.503_424_582_159_577e1,
            6.580_881_910_411_395e1,
            6.765_853_259_870_691e1,
            4.938_868_124_806_859e1,
        ],
    );
    assert_overlap_grid_values(
        &result.coulomb_potential,
        [
            -4.792_432_321_760_901,
            -4.716_935_029_071_595,
            -4.417_627_169_440_431,
            -4.116_381_332_024_653,
        ],
    );
    assert_overlap_grid_values(
        &result.spin_density_ratio,
        [
            2.512_401_984_923_580_4e-4,
            3.354_335_991_070_458_7e-4,
            5.895_745_463_982_858e-4,
            1.288_501_809_125_729_8e-3,
        ],
    );
    assert_close(result.norman_radius.radius, 6.302_380_902_894_656e-1);
    Ok(())
}

#[test]
fn potential_overlap_rejects_invalid_inputs() {
    let sample = sample_ovrlp_state();
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            potential_index: 8,
            ..sample.input()
        }),
        Err(DensityError::LengthTooShort {
            name: "atomic_numbers",
            required: 9,
            actual: 3,
        })
    );

    let bad_positions = Array2::<Real>::zeros((4, 2));
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            atom_positions: bad_positions.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidPositionShape {
            rows: 4,
            columns: 2,
        })
    );

    let bad_potentials = Array1::from_vec(vec![0, 4, 2, 1]);
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            atom_potentials: bad_potentials.view(),
            ..sample.input()
        }),
        Err(DensityError::InvalidPotentialIndex {
            name: "atom_potentials",
            index: 4,
            available: 3,
        })
    );

    let bad_overlap = [PotentialOverlapNeighbor {
        source_potential: 0,
        multiplicity: 1.0,
        distance: 0.0,
    }];
    assert_eq!(
        overlap_potential_density(PotentialOverlapInput {
            explicit_overlaps: &bad_overlap,
            ..sample.input()
        }),
        Err(DensityError::NonPositiveScalar {
            name: "explicit_overlaps.distance",
            value: 0.0,
        })
    );
}

#[derive(Debug, Clone)]
struct Ff2gSample {
    scattering_trace: Array1<Complex32>,
    scattering_ldos: Array1<Complex>,
    embedded_ldos: Array2<Complex>,
    previous_ldos: Array2<Complex>,
    scattering_density: Array2<Complex>,
    embedded_density: Array1<Complex>,
    previous_density: Array1<Complex>,
    valence_density: Array1<Real>,
    occupancy_by_l: Array1<Real>,
}

#[derive(Debug, Clone)]
struct CoulomSample {
    last_indices: Array1<usize>,
    valence_density: Array2<Real>,
    overlapped_valence_density: Array2<Real>,
    overlapped_density: Array2<Real>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    atom_potentials: Array1<usize>,
    norman_radii: Array1<Real>,
    charge_deltas: Array1<Real>,
    atomic_numbers: Array1<usize>,
    coulomb_potential: Array2<Real>,
}

#[derive(Debug, Clone)]
struct BroydenSample {
    last_indices: Array1<usize>,
    potential_multiplicities: Array1<Real>,
    norman_radii: Array1<Real>,
    norman_charges: Array1<Real>,
    valence_occupancy: Array2<Real>,
    overlapped_valence_density: Array2<Real>,
}

#[derive(Debug, Clone, Copy)]
struct BroydenReference {
    iteration: usize,
    valence_density: [[Real; 4]; 2],
    charge_deltas: [Real; 2],
    norman_charges: [Real; 2],
    norm: Real,
    coefficients: [Real; 3],
    previous_density_1_0: Real,
}

#[derive(Debug, Clone)]
struct OvrlpSample {
    atom_potentials: Array1<usize>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    atomic_numbers: Array1<usize>,
    electron_density: Array2<Real>,
    spin_density: Array2<Real>,
    valence_density: Array2<Real>,
    coulomb_potential: Array2<Real>,
}

impl Ff2gSample {
    fn input(&self) -> ValenceDensityUpdateInput<'_> {
        ValenceDensityUpdateInput {
            scattering_trace: self.scattering_trace.view(),
            potential_index: 1,
            energy_index: 1,
            last_radial_index: 5,
            scattering_ldos: self.scattering_ldos.view(),
            embedded_ldos: self.embedded_ldos.view(),
            previous_ldos: self.previous_ldos.view(),
            scattering_density: self.scattering_density.view(),
            embedded_density: self.embedded_density.view(),
            previous_density: self.previous_density.view(),
            valence_density: self.valence_density.view(),
            occupancy_by_l: self.occupancy_by_l.view(),
            current_energy: Complex::new(0.72, 0.11),
            previous_energy: Complex::new(0.61, -0.04),
            potential_multiplicity: 2.5,
            current_floor: 1,
            previous_floor: 0,
            left_sum: Complex::new(0.2, -0.1),
            right_sum: Complex::new(-0.3, 0.25),
            total_electron_count: 1.25,
            include_high_l: false,
        }
    }
}

impl CoulomSample {
    fn input(&self) -> CoulombPotentialUpdateInput<'_> {
        CoulombPotentialUpdateInput {
            mode: CoulombUpdateMode::Norman,
            highest_potential_index: 1,
            last_indices: self.last_indices.view(),
            valence_density: self.valence_density.view(),
            overlapped_valence_density: self.overlapped_valence_density.view(),
            overlapped_density: self.overlapped_density.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            atom_potentials: self.atom_potentials.view(),
            norman_radii: self.norman_radii.view(),
            charge_deltas: self.charge_deltas.view(),
            atomic_numbers: self.atomic_numbers.view(),
            coulomb_potential: self.coulomb_potential.view(),
        }
    }
}

impl BroydenSample {
    fn input<'a>(
        &'a self,
        iteration: usize,
        norman_charges: ndarray::ArrayView1<'a, Real>,
        valence_density: ndarray::ArrayView2<'a, Real>,
        workspace: &'a BroydenWorkspace,
    ) -> BroydenMixInput<'a> {
        BroydenMixInput {
            iteration,
            accelerator: 0.35,
            highest_potential_index: 1,
            valence_occupancy: self.valence_occupancy.view(),
            last_indices: self.last_indices.view(),
            potential_multiplicities: self.potential_multiplicities.view(),
            norman_radii: self.norman_radii.view(),
            norman_charges,
            overlapped_valence_density: self.overlapped_valence_density.view(),
            valence_density,
            workspace,
        }
    }

    fn valence_density_for_iteration(&self, iteration: usize) -> Array2<Real> {
        Array2::from_shape_fn((OVRLP_DENSITY_POINTS, 2), |(radial, potential)| {
            let radius = (-8.8 + 0.05 * radial as Real).exp();
            self.overlapped_valence_density[(radial, potential)]
                * (0.97 + 0.018 * iteration as Real + 0.004 * potential as Real)
                + (0.015 * iteration as Real + 0.003 * potential as Real) * (-0.35 * radius).exp()
        })
    }
}

impl OvrlpSample {
    fn input(&self) -> PotentialOverlapInput<'_> {
        PotentialOverlapInput {
            potential_index: 1,
            atom_potentials: self.atom_potentials.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            atomic_numbers: self.atomic_numbers.view(),
            explicit_overlaps: &[],
            electron_density: self.electron_density.view(),
            spin_density: self.spin_density.view(),
            valence_density: self.valence_density.view(),
            coulomb_potential: self.coulomb_potential.view(),
        }
    }
}

fn sample_broydn_state() -> BroydenSample {
    let last_indices = Array1::from_vec(vec![190, 196]);
    let potential_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let norman_radii = Array1::from_vec(vec![0.72, 0.88]);
    let norman_charges = Array1::from_vec(vec![1.40, 2.10]);
    let mut valence_occupancy = Array2::<Real>::zeros((3, 2));
    valence_occupancy[(0, 0)] = 1.10;
    valence_occupancy[(1, 0)] = 0.60;
    valence_occupancy[(0, 1)] = 1.45;
    valence_occupancy[(1, 1)] = 0.80;
    valence_occupancy[(2, 1)] = 0.30;

    let overlapped_valence_density =
        Array2::from_shape_fn((OVRLP_DENSITY_POINTS, 2), |(radial, potential)| {
            let radius = (-8.8 + 0.05 * radial as Real).exp();
            (45.0 + 8.0 * potential as Real) * (-0.92 * radius).exp() / (1.0 + 0.10 * radius)
        });

    BroydenSample {
        last_indices,
        potential_multiplicities,
        norman_radii,
        norman_charges,
        valence_occupancy,
        overlapped_valence_density,
    }
}

fn broydn_references() -> [BroydenReference; 3] {
    [
        BroydenReference {
            iteration: 1,
            valence_density: [
                [
                    6.850_151_802_587_897e8,
                    6.198_224_678_030_062e8,
                    1.385_302_559_470_316e7,
                    -2.110_332_566_524_774e1,
                ],
                [
                    8.100_660_694_916_214e8,
                    7.329_722_972_649_122e8,
                    1.638_192_383_754_785_5e7,
                    -1.286_749_342_987_842_8e1,
                ],
            ],
            charge_deltas: [-2.099_260_615_232_956_3, 1.049_630_307_616_478_6],
            norman_charges: [-6.992_606_152_329_563e-1, 3.149_630_307_616_478_7],
            norm: 0.0,
            coefficients: [0.0, 0.0, 0.0],
            previous_density_1_0: 4.499_308_188_880_038e1,
        },
        BroydenReference {
            iteration: 2,
            valence_density: [
                [
                    7.521_952_443_079_169e8,
                    6.806_090_282_421_587e8,
                    1.521_161_506_215_363_7e7,
                    -2.378_323_890_391_962_8e1,
                ],
                [
                    8.889_720_867_627_683e8,
                    8.043_688_549_825_951e8,
                    1.797_764_579_810_173_4e7,
                    -1.449_719_473_393_818_3e1,
                ],
            ],
            charge_deltas: [-1.764_945_825_152_590_7e-1, 8.824_729_125_762_865e-2],
            norman_charges: [-8.757_551_977_482_154e-1, 3.237_877_598_874_107_3],
            norm: 7.657_998_793_600_876,
            coefficients: [0.0, -4.286_904_563_383_834_5, 0.0],
            previous_density_1_0: 4.499_308_188_880_038e1,
        },
        BroydenReference {
            iteration: 3,
            valence_density: [
                [
                    7.521_952_443_079_171e8,
                    6.806_090_282_421_585e8,
                    1.521_161_506_215_365_4e7,
                    -2.378_323_890_391_962_8e1,
                ],
                [
                    8.889_720_867_627_683e8,
                    8.043_688_549_825_957e8,
                    1.797_764_579_810_172_7e7,
                    -1.449_719_473_393_818_5e1,
                ],
            ],
            charge_deltas: [1.776_356_839_400_250_5e-15, -1.776_356_839_400_250_5e-15],
            norman_charges: [-8.757_551_977_482_136e-1, 3.237_877_598_874_105_5],
            norm: 7.657_998_793_600_846_5,
            coefficients: [0.0, -3.286_904_563_383_833_6, -3.286_904_563_383_776_3],
            previous_density_1_0: 4.499_308_188_880_038e1,
        },
    ]
}

fn sample_ff2g_state() -> Ff2gSample {
    let l_count = 4;
    let potential_count = 3;
    let radial_count = 251;
    let scattering_trace = (0..l_count)
        .map(|angular| {
            let l = angular as Real;
            Complex32::new(
                ((0.05_f32 as Real) * l + 0.11_f32 as Real) as f32,
                ((-0.03_f32 as Real) * l + 0.07_f32 as Real) as f32,
            )
        })
        .collect::<Array1<_>>();
    let scattering_ldos = (0..l_count)
        .map(|angular| {
            let l = angular as Real;
            Complex::new(0.2 + 0.04 * l, -0.13 + 0.02 * l)
        })
        .collect::<Array1<_>>();
    let mut embedded_ldos = Array2::<Complex>::zeros((l_count, potential_count));
    let mut previous_ldos = Array2::<Complex>::zeros((l_count, potential_count));
    for angular in 0..l_count {
        let l = angular as Real;
        for potential in 0..potential_count {
            let p = potential as Real;
            embedded_ldos[(angular, potential)] =
                Complex::new(0.4 + 0.03 * l + 0.02 * p, -0.2 + 0.01 * l - 0.015 * p);
            previous_ldos[(angular, potential)] =
                Complex::new(-0.1 + 0.025 * l + 0.01 * p, 0.08 - 0.02 * l + 0.005 * p);
        }
    }
    let embedded_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as Real;
            Complex::new(0.05 * r, -0.02 * r)
        })
        .collect::<Array1<_>>();
    let previous_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as Real;
            Complex::new(-0.03 * r, 0.04 * r)
        })
        .collect::<Array1<_>>();
    let valence_density = (1..=radial_count)
        .map(|radial| 0.01 * radial as Real)
        .collect::<Array1<_>>();
    let mut scattering_density = Array2::<Complex>::zeros((radial_count, l_count));
    for radial in 0..radial_count {
        let r = (radial + 1) as Real;
        for angular in 0..l_count {
            let l = angular as Real;
            scattering_density[(radial, angular)] =
                Complex::new(0.006 * r + 0.02 * l, -0.004 * r + 0.015 * l);
        }
    }
    let occupancy_by_l = (0..l_count)
        .map(|angular| -0.03 + 0.015 * angular as Real)
        .collect::<Array1<_>>();

    Ff2gSample {
        scattering_trace,
        scattering_ldos,
        embedded_ldos,
        previous_ldos,
        scattering_density,
        embedded_density,
        previous_density,
        valence_density,
        occupancy_by_l,
    }
}

fn sample_coulom_state() -> CoulomSample {
    let last_indices = Array1::from_vec(vec![140, 132]);
    let atom_potentials = Array1::from_vec(vec![0, 1, 1]);
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let norman_radii = Array1::from_vec(vec![0.65, 0.82]);
    let charge_deltas = Array1::from_vec(vec![0.15, -0.07]);
    let atomic_numbers = Array1::from_vec(vec![8, 14]);
    let mut atom_positions = Array2::<Real>::zeros((3, 3));
    for (atom, position) in [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.1, 0.0]]
        .into_iter()
        .enumerate()
    {
        for axis in 0..3 {
            atom_positions[(atom, axis)] = position[axis];
        }
    }

    let mut valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    let mut overlapped_valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    let mut overlapped_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    let mut coulomb_potential = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 2));
    for potential in 0..=1 {
        let p = potential as Real;
        for index in 1..=OVRLP_DENSITY_POINTS {
            let radius = (-8.8 + 0.05 * (index - 1) as Real).exp();
            let density = (80.0 + 15.0 * p) * (-0.85 * radius).exp() / (1.0 + 0.12 * radius);
            overlapped_density[(index - 1, potential)] = density;
            overlapped_valence_density[(index - 1, potential)] = (0.42 + 0.03 * p) * density;
            valence_density[(index - 1, potential)] = (0.36 + 0.02 * p) * density;
            coulomb_potential[(index - 1, potential)] = -1.7 - 0.25 * p + 0.004 * index as Real;
        }
    }

    CoulomSample {
        last_indices,
        valence_density,
        overlapped_valence_density,
        overlapped_density,
        atom_positions,
        representative_atoms,
        atom_potentials,
        norman_radii,
        charge_deltas,
        atomic_numbers,
        coulomb_potential,
    }
}

fn sample_ovrlp_state() -> OvrlpSample {
    let atom_potentials = Array1::from_vec(vec![0, 1, 2, 1]);
    let mut atom_positions = Array2::<Real>::zeros((4, 3));
    for (atom, position) in [
        [0.0, 0.0, 0.0],
        [1.35, 0.2, -0.15],
        [3.10, -0.4, 0.25],
        [13.5, 0.0, 0.0],
    ]
    .into_iter()
    .enumerate()
    {
        for axis in 0..3 {
            atom_positions[(atom, axis)] = position[axis];
        }
    }
    let representative_atoms = Array1::from_vec(vec![0, 1, 2]);
    let atomic_numbers = Array1::from_vec(vec![6, 8, 14]);
    let mut electron_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    let mut spin_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    let mut valence_density = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    let mut coulomb_potential = Array2::<Real>::zeros((OVRLP_DENSITY_POINTS, 4));
    for potential in 0..4 {
        let p = potential as Real;
        for index in 1..=OVRLP_DENSITY_POINTS {
            let i = index as Real;
            let radius = legacy_loucks_radius(index);
            let density = (45.0 + 18.0 * p) * (-(1.0 + 0.08 * p) * radius).exp() + 0.05 * (i + p);
            electron_density[(index - 1, potential)] = density;
            valence_density[(index - 1, potential)] = 0.65 * density + 0.01 * p + 0.0002 * i;
            coulomb_potential[(index - 1, potential)] =
                -2.0 - 0.12 * p + 0.004 * i + 0.03 * (0.05 * i + p).cos();
            spin_density[(index - 1, potential)] = 0.02 + 0.0003 * i + 0.005 * p;
        }
    }

    OvrlpSample {
        atom_potentials,
        atom_positions,
        representative_atoms,
        atomic_numbers,
        electron_density,
        spin_density,
        valence_density,
        coulomb_potential,
    }
}

fn legacy_loucks_radius(index_1based: usize) -> Real {
    ((0.05_f32 as Real) * (index_1based as Real - 1.0) - 8.8_f32 as Real).exp()
}

fn sample_ldos_rhorrp_wavefunction_tables() -> RhorrpWavefunctionTables {
    let wave_numbers = Array2::from_shape_fn((2, 2), |(energy, potential)| {
        Complex::new(
            0.68 + 0.07 * energy as Real + 0.03 * potential as Real,
            0.05 + 0.01 * potential as Real,
        )
    });
    let phase_shifts = Array3::from_shape_fn((2, 2, 2), |(energy, angular, potential)| {
        Complex::new(
            0.03 * (energy + angular + potential) as Real,
            -0.01 * potential as Real,
        )
    });
    let regular_large = Array4::from_shape_fn((2, 2, 7, 2), |(energy, angular, row, potential)| {
        let scale = 1.0 + 0.2 * potential as Real + 0.05 * energy as Real;
        Complex::new(
            scale * (0.12 + 0.017 * row as Real + 0.024 * angular as Real),
            0.01 * energy as Real - 0.004 * row as Real,
        )
    });
    let regular_small = Array4::from_shape_fn((2, 2, 7, 2), |(energy, angular, row, potential)| {
        Complex::new(
            0.014 + 0.002 * row as Real + 0.003 * angular as Real,
            -0.006 + 0.002 * energy as Real + 0.001 * potential as Real,
        )
    });
    let irregular_large =
        Array4::from_shape_fn((2, 2, 7, 2), |(energy, angular, row, potential)| {
            Complex::new(
                -0.09 - 0.011 * row as Real + 0.02 * potential as Real,
                0.05 + 0.008 * angular as Real + 0.003 * energy as Real,
            )
        });
    let irregular_small =
        Array4::from_shape_fn((2, 2, 7, 2), |(energy, angular, row, potential)| {
            Complex::new(
                -0.012 - 0.001 * row as Real + 0.002 * potential as Real,
                0.009 + 0.0015 * angular as Real + 0.0007 * energy as Real,
            )
        });

    RhorrpWavefunctionTables {
        setups_by_potential: vec![Vec::new(), Vec::new()],
        wave_numbers,
        phase_shifts,
        regular_large,
        irregular_large,
        regular_small,
        irregular_small,
        regular_iteration_count: 0,
        irregular_iteration_count: 0,
        difficult_iterations: 0,
    }
}

struct ReferenceLdosRholChannelInputs {
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
    radial_match_index: usize,
    bound_orbital_count: usize,
}

impl ReferenceLdosRholChannelInputs {
    fn to_input(&self) -> FovrgDiracSolverInput<'_> {
        FovrgDiracSolverInput {
            exchange_cycle_count: 1,
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
            muffin_tin_large_component: Complex::new(0.0, 0.0),
            muffin_tin_small_component: Complex::new(0.0, 0.0),
            atomic_number: 29.0,
            irregular: false,
            c3_scale: 0,
            radial_match_index: self.radial_match_index,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn reference_ldos_rhol_channel_inputs() -> ReferenceLdosRholChannelInputs {
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

    ReferenceLdosRholChannelInputs {
        target_kappa: -2,
        muffin_tin_radius: 1.42,
        target_last_index: 25,
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
        radial_match_index: 9,
        bound_orbital_count: bound_orbitals,
    }
}

fn ldos_channel_wave_number_from_kinetic_energy(kinetic_energy: Complex) -> Complex {
    let alpha_kinetic = kinetic_energy / 137.035_989_56;
    (kinetic_energy * 2.0 + alpha_kinetic * alpha_kinetic).sqrt()
}

fn assert_coulom_values(values: &Array2<Real>, potential: usize, expected: [Real; 6]) {
    let indices = [
        1,
        2,
        64,
        if potential == 0 { 140 } else { 132 },
        if potential == 0 { 141 } else { 133 },
        251,
    ];
    for (index, expected_value) in indices.into_iter().zip(expected) {
        assert_close(values[(index - 1, potential)], expected_value);
    }
}

fn assert_broydn_grid_values(
    values: &Array2<Real>,
    potential: usize,
    last_index: usize,
    expected: [Real; 4],
) {
    for (index, expected_value) in [1, 2, 40, last_index].into_iter().zip(expected) {
        assert_close(values[(index - 1, potential)], expected_value);
    }
}

fn assert_overlap_grid_values(values: &Array1<Real>, expected: [Real; 4]) {
    const OVRLP_ORACLE_TOLERANCE: Real = 5.0e-7;

    for (index, expected_value) in [1, 25, 100, 180].into_iter().zip(expected) {
        assert!(
            (values[index - 1] - expected_value).abs() <= OVRLP_ORACLE_TOLERANCE,
            "{} != {}",
            values[index - 1],
            expected_value
        );
    }
}

fn expected_scmt_endpoint_fraction(
    current_energy: Complex,
    previous_energy: Complex,
    current_delta: Real,
    previous_delta: Real,
    current_sum: Complex,
    previous_sum: Complex,
) -> Real {
    if current_delta == 0.0 {
        return 0.0;
    }

    let mut fraction = current_delta / (current_delta - previous_delta);
    for _ in 0..4 {
        let correction = expected_scmt_endpoint_correction(
            current_energy,
            previous_energy,
            current_sum,
            previous_sum,
            fraction,
        );
        let residual = current_delta + fraction * correction;
        fraction -= residual / correction;
    }
    fraction
}

fn expected_scmt_endpoint_correction(
    current_energy: Complex,
    previous_energy: Complex,
    current_value: Complex,
    previous_value: Complex,
    fraction: Real,
) -> Real {
    let interpolated = previous_value * fraction + current_value * (1.0 - fraction);
    let imaginary = Complex::new(0.0, 1.0);
    ((previous_energy - current_energy) * (current_value + interpolated) / 2.0
        + imaginary * current_energy.im * (current_value - previous_value))
        .im
}

fn sample_scmt_contour_tables() -> (Array1<Complex>, Array1<Real>) {
    (
        Array1::from_vec(vec![
            Complex::new(0.10, 0.04),
            Complex::new(0.18, 0.04),
            Complex::new(0.26, 0.04),
            Complex::new(0.34, 0.04),
            Complex::new(0.42, 0.02),
            Complex::new(0.50, 0.01),
        ]),
        Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04]),
    )
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

fn fmsdos_expected(diagonal_sum: Complex, phase: Complex, angular: usize) -> Complex {
    let normalization = (2 * angular + 1) as Real;
    diagonal_sum * (Complex::new(0.0, 2.0) * phase).exp() / normalization
}

fn assert_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-8_f64.max(expected.abs() * 1.0e-12);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}
