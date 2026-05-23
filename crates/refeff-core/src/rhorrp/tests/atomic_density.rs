use super::{support::*, *};

#[test]
fn fix_irregular_origin_matches_feff_reference() -> Result<(), RhorrpError> {
    let (radii, values) = reference_irregular_solution();
    let fixed = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
        radii: &radii,
        values: values.view(),
    })?;

    assert_complex_close_tol(
        fixed[0],
        Complex::new(9.791_151_469_085_387, 3.741_459_448_683_99),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[49],
        Complex::new(-2.047_179_619_930_901_1e-1, -8.434_737_680_311_137e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[74],
        Complex::new(-6.916_158_567_064_077e-1, -8.929_639_586_361_882e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[99],
        Complex::new(8.811_645_823_831e-1, 1.866_102_289_679_183_5e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[100],
        Complex::new(9.101_077_089_878_837e-1, 2.302_339_202_367_545e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[119],
        Complex::new(1.094_598_908_088_280_5, 8.401_702_866_503_66e-1),
        1.0e-8,
    );
    Ok(())
}

#[test]
fn fix_irregular_origin_rejects_invalid_inputs() {
    let (radii, values) = reference_irregular_solution();
    assert!(matches!(
        rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii[..99],
            values: values.slice_axis(Axis(0), Slice::from(..99)),
        }),
        Err(RhorrpError::InsufficientIrregularFixPoints {
            points: 99,
            required: 100,
        })
    ));

    assert!(matches!(
        rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii[..100],
            values: values.view(),
        }),
        Err(RhorrpError::IrregularFixLengthMismatch {
            radii: 100,
            values: 120,
        })
    ));
}

#[test]
fn atomic_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference = reference_atomic_density_tables();

    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.08, 0.04, -0.03],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        9.746_265_921_948_757,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.72, -0.15, 0.18],
            orbital_index_1based: 2,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        2.182_748_347_338_233e1,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 3,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        7.107_185_239_762_148e6,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [4.2, 3.9, -2.5],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        0.0,
    );
    Ok(())
}

#[test]
fn atomic_density_rejects_invalid_inputs() {
    let reference = reference_atomic_density_tables();
    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 0,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::InvalidAtomicDensityOrbital {
            orbital: 0,
            orbital_count: 3,
        })
    ));

    let bad_potentials = [0, 1, 3, 1];
    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &bad_potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::InvalidAtomicDensityPotential {
            atom_index_1based: 3,
            potential: 3,
            max_potential: 2,
        })
    ));

    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii[..11],
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::AtomicDensityRadialLengthMismatch {
            radii: 11,
            components: 12,
        })
    ));
}

#[test]
fn integrate_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let (energies, energy_density) = reference_density_integration_inputs();

    assert_real_close(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        })?,
        -4.627_669_214_946_009e-2,
    );
    assert_real_close(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: -0.010,
            temperature_hartree: 0.000_001,
            chemical_potential_override_hartree: None,
        })?,
        -1.115_611_780_024_965e-3,
    );
    Ok(())
}

#[test]
fn integrate_density_rejects_invalid_inputs() {
    let (energies, energy_density) = reference_density_integration_inputs();

    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.slice_axis(Axis(0), Slice::from(..7)),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::DensityIntegrationLengthMismatch {
            energies: 7,
            densities: 8,
        })
    ));
    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 1,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::InvalidDensityIntegrationRealAxisCount {
            real_axis_count: 1,
            energy_count: 8,
        })
    ));

    let vertical_only = Array1::from_vec(vec![
        Complex::new(-0.03, 0.09),
        Complex::new(-0.03, 0.06),
        Complex::new(-0.03, 0.03),
        Complex::new(-0.03, 0.00),
    ]);
    let vertical_density = Array1::from_vec(vec![Complex::new(0.3, 0.1); 4]);
    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: vertical_only.view(),
            energy_density: vertical_density.view(),
            real_axis_count: 4,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::MissingDensityIntegrationCorner)
    ));
}
