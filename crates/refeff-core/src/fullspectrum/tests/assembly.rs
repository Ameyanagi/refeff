use super::{support::*, *};

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
