use super::*;
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, arr1, arr2};

#[test]
fn electron_wavelength_matches_feff_reference() -> Result<(), EelsError> {
    assert_close(
        electron_wavelength_atomic_units(1_000.0)?,
        0.732_534_340_476_640,
    );
    assert_close(
        electron_wavelength_atomic_units(100_000.0)?,
        0.069_947_069_983_283,
    );
    assert_close(
        electron_wavelength_atomic_units(300_000.0)?,
        0.037_204_017_054_112,
    );
    Ok(())
}

#[test]
fn eels_euler_rotation_matrix_matches_feff_reference() -> Result<(), EelsError> {
    assert_matrix_close(
        eels_euler_rotation_matrix(0.3, 0.4, -0.2)?.view(),
        arr2(&[
            [
                0.921_094_097_834_994,
                -0.114_815_729_042_654,
                0.372_025_551_942_260,
            ],
            [
                0.076_970_353_575_606,
                0.990_369_592_951_021,
                0.115_080_988_996_769,
            ],
            [
                -0.381_655_902_095_048,
                -0.077_365_481_465_782,
                0.921_060_994_002_885,
            ],
        ])
        .view(),
    );
    assert_matrix_close(
        eels_euler_rotation_matrix(-1.1, 0.75, 1.4)?.view(),
        arr2(&[
            [
                0.934_650_656_964_861,
                -0.175_586_157_235_345,
                0.309_188_697_759_924,
            ],
            [
                0.336_162_895_167_387,
                0.719_694_907_282_947,
                -0.607_481_479_835_946,
            ],
            [
                -0.115_856_192_531_229,
                0.671_720_732_014_663,
                0.731_688_868_873_821,
            ],
        ])
        .view(),
    );
    Ok(())
}

#[test]
fn eels_euler_rotation_matrix_uses_fortran_order_storage() -> Result<(), EelsError> {
    let matrix = eels_euler_rotation_matrix(0.3, 0.4, -0.2)?;
    let mut expected = Vec::new();
    for column in 0..3 {
        for row in 0..3 {
            expected.push(matrix[(row, column)]);
        }
    }
    assert_eq!(matrix.as_slice_memory_order(), Some(expected.as_slice()));
    Ok(())
}

#[test]
fn eels_product_matrix_vector_matches_feff_reference() -> Result<(), EelsError> {
    let first_matrix = arr2(&[[1.25, 2.0, -0.25], [-0.5, 0.125, 3.0], [0.75, -1.5, 0.5]]);
    let first_vector = arr1(&[0.2, -1.5, 4.0]);
    assert_vector_close(
        eels_product_matrix_vector(first_matrix.view(), first_vector.view())?.view(),
        arr1(&[-3.75, 11.7125, 4.4]).view(),
    );

    let second_matrix = arr2(&[[0.0, -3.5, 2.25], [1.0, 0.25, -0.75], [-2.0, 4.0, 0.5]]);
    let second_vector = arr1(&[-2.0, 0.5, 1.25]);
    assert_vector_close(
        eels_product_matrix_vector(second_matrix.view(), second_vector.view())?.view(),
        arr1(&[1.0625, -2.8125, 6.625]).view(),
    );
    Ok(())
}

#[test]
fn eels_qmesh_matches_feff_reference() -> Result<(), EelsError> {
    let theta_x = arr1(&[0.0, 0.0015, -0.002, -0.001]);
    let theta_y = arr1(&[0.0, -0.0025, 0.001, -0.003]);
    let relativistic = eels_qmesh(EelsQMeshInput {
        incident_energy_ev: 300_000.0,
        scattered_energy_ev: 299_880.0,
        beam_direction: [0.2, 0.3, 0.9],
        theta_x: theta_x.view(),
        theta_y: theta_y.view(),
        relativistic: true,
    })?;
    assert_vector_close(
        arr1(&relativistic.euler_angles).view(),
        arr1(&[0.982793723247329, 0.38103799535731686, 0.0]).view(),
    );
    assert_rect_matrix_close(
        relativistic.q_vectors.view(),
        arr2(&[
            [
                -0.003394132349274313,
                -0.4850774133237736,
                0.31093731394495,
                -0.337980548857258,
            ],
            [
                -0.00509119852391147,
                0.03334859520894076,
                0.16201990728101914,
                0.40618660665810785,
            ],
            [
                -0.015273595571734404,
                0.07864696951859113,
                -0.14100925915007487,
                -0.07837471841393498,
            ],
        ])
        .view(),
    );
    assert_vector_close(
        relativistic.q_lengths.view(),
        arr1(&[
            0.016453667022982246,
            0.49254194900916926,
            0.3779101410715296,
            0.534191919932102,
        ])
        .view(),
    );
    assert_vector_close(
        relativistic.classical_q_lengths.view(),
        arr1(&[
            0.04144385038566156,
            0.4940596914878048,
            0.37985861484381056,
            0.5356000588077447,
        ])
        .view(),
    );

    let classical = eels_qmesh(EelsQMeshInput {
        incident_energy_ev: 100_000.0,
        scattered_energy_ev: 99_800.0,
        beam_direction: [0.0, 1.0, 0.0],
        theta_x: theta_x.view(),
        theta_y: theta_y.view(),
        relativistic: false,
    })?;
    assert_vector_close(
        arr1(&classical.euler_angles).view(),
        arr1(&[
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            0.0,
        ])
        .view(),
    );
    assert_rect_matrix_close(
        classical.q_vectors.view(),
        arr2(&[
            [
                -5.992870093576868e-18,
                -0.22432428648555633,
                0.08972976693659487,
                -0.26918907648534857,
            ],
            [
                -0.09787099591081017,
                -0.09825234746796241,
                -0.09809532042162061,
                -0.09831964474550145,
            ],
            [
                -5.992870093576868e-18,
                0.13459457189133378,
                -0.1794595338731897,
                -0.08972969216178289,
            ],
        ])
        .view(),
    );
    assert_vector_close(
        classical.q_lengths.view(),
        arr1(&[
            0.09787099591081017,
            0.2794469682655916,
            0.22333796645688922,
            0.30030139709525955,
        ])
        .view(),
    );
    assert_vector_close(
        classical.q_lengths.view(),
        classical.classical_q_lengths.view(),
    );
    Ok(())
}

#[test]
fn eels_spectrum_matches_feff_reference() -> Result<(), EelsError> {
    let energy_loss = arr1(&[12.5, 28.0, 64.0]);
    let transition_tensor = Array3::from_shape_fn((3, 3, 3), |(energy, row, column)| {
        let i = (energy + 1) as Real;
        let j1 = (row + 1) as Real;
        let j2 = (column + 1) as Real;
        0.015 * i + 0.11 * j1 - 0.045 * j2 + 0.002 * i * j1 * j2
    });
    let atomic_background = arr1(&[0.092, 0.104, 0.116]);

    let spectrum = eels_spectrum(EelsSpectrumInput {
        incident_energy_ev: 200_000.0,
        beam_direction: [0.25, -0.15, 0.95],
        mesh: EelsMeshInput {
            collection_angle: 0.014,
            convergence_angle: 0.006,
            theta0: 0.0007,
            theta_x_center: 0.0012,
            theta_y_center: -0.0008,
            radial_count: 2,
            angular_count: 2,
            mode: EelsMeshMode::Uniform,
        },
        energy_loss_ev: energy_loss.view(),
        transition_tensor: transition_tensor.view(),
        atomic_background: atomic_background.view(),
        relativistic: true,
    })?;

    assert_vector_close(
        spectrum.total.view(),
        arr1(&[
            5.330409013028863e-5,
            3.468472190648792e-5,
            1.95390880411704e-5,
        ])
        .view(),
    );
    assert_vector_close(
        spectrum.background.view(),
        arr1(&[
            5.631994485295036e-4,
            2.8415578845250556e-4,
            1.385024135506364e-4,
        ])
        .view(),
    );
    assert_vector_close(
        spectrum.fine_structure.view(),
        arr1(&[
            -5.098953583992149e-4,
            -2.4947106654601764e-4,
            -1.18963325509466e-4,
        ])
        .view(),
    );
    assert_rect_matrix_close(
        spectrum.partials.view(),
        arr2(&[
            [
                3.628_362_866_235_717e-4,
                -6.954606789113099e-5,
                5.850424513429709e-6,
                -3.45947106945626e-4,
                1.839675708822848e-4,
                7.416082245471633e-5,
                -4.4755747527737183e-4,
                1.7679410353043987e-4,
                1.1274553223997504e-4,
            ],
            [
                1.9510588328310328e-4,
                -4.606423836113137e-5,
                -1.121262380465124e-5,
                -1.6916694432622382e-4,
                9.436020466361983e-5,
                4.1257244610055344e-5,
                -2.1567811671299733e-4,
                8.726352457090846e-5,
                5.881978798380475e-5,
            ],
            [
                9.938837828312541e-5,
                -2.658737781501426e-5,
                -1.1208163048553978e-5,
                -8.010742406601703e-5,
                4.6522662357093414e-5,
                2.1737585896153796e-5,
                -1.0264317739202065e-4,
                4.203472935340584e-5,
                3.040187447299786e-5,
            ],
        ])
        .view(),
    );
    assert_close(spectrum.integration_mesh.weights[0], 1.5707963267948965e-4);
    assert_close(spectrum.integration_mesh.weights[3], 5.542237284087798e-5);
    assert_close(spectrum.integration_mesh.weights[7], 5.542237284087798e-5);
    Ok(())
}

#[test]
fn eels_read_spectrum_matches_feff_readsp_reference() -> Result<(), EelsError> {
    let owned_sources = sample_readsp_sources();
    let sources = readsp_source_views(&owned_sources);

    let sensitive_cross = eels_read_spectrum(EelsReadSpectrumInput {
        sources: &sources,
        orientation_averaged: false,
        cross_terms: true,
        polarization_min: 1,
        polarization_step: 1,
        polarization_max: 9,
    })?;
    assert_eq!(sensitive_cross.effective_polarization_step, 1);
    assert_vector_close(
        sensitive_cross.energy_loss_ev.view(),
        arr1(&[10.25, 20.25, 30.25]).view(),
    );
    assert_vector_close(
        sensitive_cross.atomic_background.view(),
        arr1(&[1.23, 1.26, 1.29]).view(),
    );
    assert_readsp_tensor_rows(
        sensitive_cross.transition_tensor.view(),
        &[
            [
                0.111, 0.212, 0.313, 0.414, 0.515, 0.616, 0.717, 0.818, 0.919,
            ],
            [
                0.122, 0.224, 0.326, 0.428, 0.530, 0.632, 0.734, 0.836, 0.938,
            ],
            [
                0.133, 0.236, 0.339, 0.442, 0.545, 0.648, 0.751, 0.854, 0.957,
            ],
        ],
    );

    let sensitive_no_cross = eels_read_spectrum(EelsReadSpectrumInput {
        sources: &sources,
        orientation_averaged: false,
        cross_terms: false,
        polarization_min: 1,
        polarization_step: 1,
        polarization_max: 9,
    })?;
    assert_eq!(sensitive_no_cross.effective_polarization_step, 4);
    assert_readsp_tensor_rows(
        sensitive_no_cross.transition_tensor.view(),
        &[
            [0.111, 0.0, 0.0, 0.0, 0.515, 0.0, 0.0, 0.0, 0.919],
            [0.122, 0.0, 0.0, 0.0, 0.530, 0.0, 0.0, 0.0, 0.938],
            [0.133, 0.0, 0.0, 0.0, 0.545, 0.0, 0.0, 0.0, 0.957],
        ],
    );

    let average_from_diagonal = eels_read_spectrum(EelsReadSpectrumInput {
        sources: &sources,
        orientation_averaged: true,
        cross_terms: true,
        polarization_min: 1,
        polarization_step: 1,
        polarization_max: 9,
    })?;
    assert_eq!(average_from_diagonal.effective_polarization_step, 1);
    assert_vector_close(
        average_from_diagonal.atomic_background.view(),
        arr1(&[1.23, 1.26, 1.29]).view(),
    );
    assert_readsp_tensor_rows(
        average_from_diagonal.transition_tensor.view(),
        &[
            [0.515, 0.0, 0.0, 0.0, 0.515, 0.0, 0.0, 0.0, 0.515],
            [0.530, 0.0, 0.0, 0.0, 0.530, 0.0, 0.0, 0.0, 0.530],
            [0.545, 0.0, 0.0, 0.0, 0.545, 0.0, 0.0, 0.0, 0.545],
        ],
    );

    let average_from_ten = eels_read_spectrum(EelsReadSpectrumInput {
        sources: &sources,
        orientation_averaged: true,
        cross_terms: false,
        polarization_min: 10,
        polarization_step: 1,
        polarization_max: 10,
    })?;
    assert_vector_close(
        average_from_ten.atomic_background.view(),
        arr1(&[3.03, 3.06, 3.09]).view(),
    );
    assert_readsp_tensor_rows(
        average_from_ten.transition_tensor.view(),
        &[
            [1.02, 0.0, 0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 1.02],
            [1.04, 0.0, 0.0, 0.0, 1.04, 0.0, 0.0, 0.0, 1.04],
            [1.06, 0.0, 0.0, 0.0, 1.06, 0.0, 0.0, 0.0, 1.06],
        ],
    );

    Ok(())
}

#[test]
fn eels_gos_matches_feff_reference() -> Result<(), EelsError> {
    let energy_loss = arr1(&[12.5, 28.0, 64.0, 92.0]);
    let averaged = arr1(&[0.0045, 0.0062, 0.0087, 0.011]);
    let relativistic = eels_generalized_oscillator_strength(EelsGosInput {
        incident_energy_ev: 200_000.0,
        energy_loss_ev: energy_loss.view(),
        averaged_spectrum: averaged.view(),
        relativistic: true,
    })?;
    assert_close(relativistic.q_scale, 0.6858854501070719);
    assert_close(relativistic.q_log_step, 0.12938077038704135);
    assert_close(relativistic.edge_parameter, 100.0);
    assert_close(relativistic.energy_start_ev, 100.0);
    assert_close(relativistic.energy_step_ev, 10.0);
    assert_eq!(relativistic.q_values.len(), FEFF_EELS_GOS_Q_COUNT);
    assert_selected_q_values(&relativistic.q_values);
    assert_selected_gos_rows(
        relativistic.strengths.view(),
        &[
            [
                1.200166939655954e6,
                2.606956723601938e5,
                2.743180061971628e4,
                3.2396868649661033e3,
                1.500_423_530_532_838e2,
            ],
            [
                3.841354508768833e6,
                8.109311801422351e5,
                8.473081327354828e4,
                9.999371994697678e3,
                4.630661427836753e2,
            ],
            [
                1.510800565240755e7,
                2.7128132396096834e6,
                2.729555927210918e5,
                3.2088282622040024e4,
                1.485261481444802e3,
            ],
            [
                3.7264225704868965e7,
                5.219418883596917e6,
                4.98984358650779e5,
                5.8361116195723495e4,
                2.699590549258333e3,
            ],
        ],
    );

    let classical = eels_generalized_oscillator_strength(EelsGosInput {
        incident_energy_ev: 200_000.0,
        energy_loss_ev: energy_loss.view(),
        averaged_spectrum: averaged.view(),
        relativistic: false,
    })?;
    assert_selected_q_values(&classical.q_values);
    assert_selected_gos_rows(
        classical.strengths.view(),
        &[
            [
                1.1894589336479066e6,
                2.601859956710956e5,
                2.7426144922494695e4,
                3.239607964002951e3,
                1.5004218380799958e2,
            ],
            [
                3.6709345934449174e6,
                8.029_918_017_511_5e5,
                8.46431779296903e4,
                9.998150089793999e3,
                4.630635219389996e2,
            ],
            [
                1.1774057497869411e7,
                2.575494442482951e6,
                2.714822665394675e5,
                3.206779936634388e4,
                1.4852175634541184e3,
            ],
            [
                2.139968783736322e7,
                4.681035157673755e6,
                4.9342682065003784e5,
                5.8284146836817534e4,
                2.699425600243477e3,
            ],
        ],
    );
    Ok(())
}

#[test]
fn eels_angular_dependence_matches_feff_reference() -> Result<(), EelsError> {
    let q_vectors = arr2(&[
        [0.145, 0.310, 0.720, 1.350],
        [0.010, 0.045, 0.115, 0.210],
        [0.25, -0.40, 0.90, -1.20],
    ]);
    let weights = arr1(&[0.185, 0.295, 0.470, 0.815]);
    let partials = Array2::from_shape_fn((10, 4), |(partial, position)| {
        let l = (partial + 1) as Real;
        let k = (position + 1) as Real;
        0.003 * l.powi(2) + 0.017 * k + 0.0009 * l * k
    });

    let angular = eels_angular_dependence(EelsAngularDependenceInput {
        q_vectors_spherical: q_vectors.view(),
        weights: weights.view(),
        partial_spectra: partials.view(),
        incident_wave_number: 82.75,
    })?;

    assert_rect_matrix_close(
        angular.rows.view(),
        arr2(&[
            [
                -1.7553122764623317e-2,
                2.524324324324324e-1,
                6.502702702702704e-1,
                5.667567567567568,
                5.372972972972974e-1,
                7.897297297297298e-1,
                1.1297297297297304e-1,
                4.764864864864865,
                1.7621621621621622,
            ],
            [
                -1.6915622352268206e-1,
                2.2508474576271187e-1,
                6.020338983050848e-1,
                4.210169491525424,
                4.705084745762712e-1,
                6.955932203389831e-1,
                1.315254237288136e-1,
                3.3830508474576275,
                1.193220338983051,
            ],
            [
                -1.0071045252090503,
                1.8319148936170213e-1,
                4.997872340425533e-1,
                3.0542553191489366,
                3.7914893617021284e-1,
                5.62340425531915e-1,
                1.2063829787234043e-1,
                2.371276595744681,
                8.042553191489362e-1,
            ],
            [
                -3.4559789414201028,
                1.2981595092024542e-1,
                3.585276073619632e-1,
                1.9987730061349693,
                2.669938650306749e-1,
                3.9680981595092035e-1,
                9.15337423312883e-2,
                1.5104294478527607,
                4.957055214723926e-1,
            ],
        ])
        .view(),
    );
    Ok(())
}

#[test]
fn eels_collection_angle_dependence_matches_feff_reference() -> Result<(), EelsError> {
    let energy_loss = arr1(&[12.5, 28.0, 64.0]);
    let sigma_x = arr1(&[0.0045, 0.0062, 0.0087]);
    let sigma_y = arr1(&[0.0051, 0.0068, 0.0094]);
    let pi_spectrum = arr1(&[0.0060, 0.0077, 0.0102]);
    let base = EelsCollectionDependenceInput {
        incident_energy_ev: 200_000.0,
        beam_direction: [0.25, -0.15, 0.95],
        mesh: EelsMeshInput {
            collection_angle: 0.020,
            convergence_angle: 0.006,
            theta0: 0.001,
            theta_x_center: 0.0012,
            theta_y_center: -0.0008,
            radial_count: 5,
            angular_count: 2,
            mode: EelsMeshMode::Uniform,
        },
        magic_energy_ev: 10.0,
        energy_loss_ev: energy_loss.view(),
        sigma_x_spectrum: sigma_x.view(),
        sigma_y_spectrum: sigma_y.view(),
        pi_spectrum: pi_spectrum.view(),
        relativistic: true,
    };

    let uniform = eels_collection_angle_dependence(base)?;
    assert_eq!(uniform.magic_index, 0);
    assert_close(uniform.magic_energy_loss_ev, 12.5);
    assert_eq!(uniform.point_counts.to_vec(), vec![2, 8, 18]);
    assert_collection_rows(
        uniform.rows.view(),
        &[
            [
                5.200_000_000_000_001e-3,
                9.31245474312255e-2,
                7.40535193031613e-7,
                7.211559216646076e-6,
                7.952094409677688e-6,
            ],
            [
                1.0400000000000001e-2,
                9.08490159327226e-2,
                1.2126539050039605e-6,
                1.213535974769185e-5,
                1.334801365269581e-5,
            ],
            [
                1.5600000000000003e-2,
                8.507558779221076e-2,
                1.0342933558856195e-6,
                1.1123052631682343e-5,
                1.2157345987567962e-5,
            ],
        ],
    );

    let logarithmic = eels_collection_angle_dependence(EelsCollectionDependenceInput {
        mesh: EelsMeshInput {
            mode: EelsMeshMode::Logarithmic,
            ..base.mesh
        },
        ..base
    })?;
    assert_eq!(logarithmic.magic_index, 0);
    assert_eq!(logarithmic.point_counts.to_vec(), vec![2, 8, 18, 32]);
    assert_collection_rows(
        logarithmic.rows.view(),
        &[
            [
                1e-3,
                4.63719811734318e-2,
                6.832195667109031e-10,
                1.4050236917618574e-8,
                1.4733456484329477e-8,
            ],
            [
                2.2581008643532256e-3,
                3.177958871763603e-2,
                1.3650442476143092e-7,
                4.158844579729555e-6,
                4.295349004490987e-6,
            ],
            [
                5.099019513592785e-3,
                3.973498993079076e-2,
                2.1691625690008808e-7,
                5.242157906146296e-6,
                5.459074163046384e-6,
            ],
            [
                1.1514100370997834e-2,
                4.482992649760372e-2,
                3.500417701478247e-7,
                7.458174693614065e-6,
                7.808216463761889e-6,
            ],
        ],
    );
    Ok(())
}

#[test]
fn eels_helpers_reject_invalid_inputs() {
    assert_eq!(
        electron_wavelength_atomic_units(0.0),
        Err(EelsError::InvalidBeamEnergy { value: 0.0 })
    );
    assert!(matches!(
        electron_wavelength_atomic_units(f64::NAN),
        Err(EelsError::NonFiniteInput {
            name: "energy_ev",
            ..
        })
    ));
    assert!(matches!(
        eels_euler_rotation_matrix(0.0, f64::INFINITY, 0.0),
        Err(EelsError::NonFiniteInput { name: "beta", .. })
    ));
    assert_eq!(
        eels_product_matrix_vector(
            arr2(&[[1.0, 2.0, 3.0]]).view(),
            arr1(&[1.0, 2.0, 3.0]).view()
        ),
        Err(EelsError::InvalidMatrixShape {
            rows: 1,
            columns: 3,
        })
    );
    assert_eq!(
        eels_product_matrix_vector(
            arr2(&[[1.0, 2.0], [3.0, 4.0]]).view(),
            arr1(&[1.0, 2.0]).view()
        ),
        Err(EelsError::InvalidMatrixShape {
            rows: 2,
            columns: 2,
        })
    );
    assert_eq!(
        eels_product_matrix_vector(
            arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).view(),
            arr1(&[1.0, 2.0]).view(),
        ),
        Err(EelsError::InvalidVectorLength { length: 2 })
    );
    assert_eq!(
        eels_qmesh(EelsQMeshInput {
            incident_energy_ev: 100_000.0,
            scattered_energy_ev: 99_000.0,
            beam_direction: [0.0, 0.0, 1.0],
            theta_x: arr1(&[0.0, 0.1]).view(),
            theta_y: arr1(&[0.0]).view(),
            relativistic: true,
        }),
        Err(EelsError::QMeshLengthMismatch {
            theta_x_len: 2,
            theta_y_len: 1,
        })
    );
    assert!(matches!(
        eels_qmesh(EelsQMeshInput {
            incident_energy_ev: 100_000.0,
            scattered_energy_ev: 99_000.0,
            beam_direction: [0.0, f64::NAN, 1.0],
            theta_x: arr1(&[0.0]).view(),
            theta_y: arr1(&[0.0]).view(),
            relativistic: true,
        }),
        Err(EelsError::NonFiniteInput {
            name: "beam_direction",
            ..
        })
    ));
    let losses = arr1(&[10.0]);
    let tensor = Array3::<Real>::zeros((1, 3, 3));
    assert_eq!(
        eels_spectrum(EelsSpectrumInput {
            incident_energy_ev: 100_000.0,
            beam_direction: [0.0, 0.0, 1.0],
            mesh: EelsMeshInput {
                collection_angle: 0.01,
                convergence_angle: 0.0,
                theta0: 0.001,
                theta_x_center: 0.0,
                theta_y_center: 0.0,
                radial_count: 1,
                angular_count: 1,
                mode: EelsMeshMode::Uniform,
            },
            energy_loss_ev: losses.view(),
            transition_tensor: tensor.view(),
            atomic_background: arr1(&[0.1, 0.2]).view(),
            relativistic: true,
        }),
        Err(EelsError::SpectrumLengthMismatch {
            name: "atomic_background",
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        eels_spectrum(EelsSpectrumInput {
            incident_energy_ev: 100_000.0,
            beam_direction: [0.0, 0.0, 1.0],
            mesh: EelsMeshInput {
                collection_angle: 0.01,
                convergence_angle: 0.0,
                theta0: 0.001,
                theta_x_center: 0.0,
                theta_y_center: 0.0,
                radial_count: 1,
                angular_count: 1,
                mode: EelsMeshMode::Uniform,
            },
            energy_loss_ev: arr1(&[100_000.0]).view(),
            transition_tensor: tensor.view(),
            atomic_background: arr1(&[0.1]).view(),
            relativistic: true,
        }),
        Err(EelsError::InvalidEnergyLoss {
            index: 0,
            value: 100_000.0,
            incident_energy_ev: 100_000.0,
        })
    );
    assert_eq!(
        eels_generalized_oscillator_strength(EelsGosInput {
            incident_energy_ev: 100_000.0,
            energy_loss_ev: arr1(&[10.0, 20.0]).view(),
            averaged_spectrum: arr1(&[0.1]).view(),
            relativistic: true,
        }),
        Err(EelsError::SpectrumLengthMismatch {
            name: "averaged_spectrum",
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        eels_generalized_oscillator_strength(EelsGosInput {
            incident_energy_ev: 100_000.0,
            energy_loss_ev: arr1(&[-10.0]).view(),
            averaged_spectrum: arr1(&[0.1]).view(),
            relativistic: true,
        }),
        Err(EelsError::InvalidEnergyLoss {
            index: 0,
            value: -10.0,
            incident_energy_ev: 100_000.0,
        })
    );
    assert_eq!(
        eels_angular_dependence(EelsAngularDependenceInput {
            q_vectors_spherical: arr2(&[[0.1], [0.01], [0.0]]).view(),
            weights: arr1(&[0.0]).view(),
            partial_spectra: Array2::<Real>::zeros((10, 1)).view(),
            incident_wave_number: 10.0,
        }),
        Err(EelsError::InvalidWeight {
            index: 0,
            value: 0.0,
        })
    );
    assert_eq!(
        eels_angular_dependence(EelsAngularDependenceInput {
            q_vectors_spherical: arr2(&[[0.1], [0.01]]).view(),
            weights: arr1(&[1.0]).view(),
            partial_spectra: Array2::<Real>::zeros((10, 1)).view(),
            incident_wave_number: 10.0,
        }),
        Err(EelsError::InvalidTableShape {
            name: "q_vectors_spherical",
            rows: 2,
            columns: 1,
            expected_rows: 3,
            expected_columns: 1,
        })
    );
    assert_eq!(
        eels_collection_angle_dependence(EelsCollectionDependenceInput {
            incident_energy_ev: 100_000.0,
            beam_direction: [0.0, 0.0, 1.0],
            mesh: EelsMeshInput {
                collection_angle: 0.01,
                convergence_angle: 0.002,
                theta0: 0.001,
                theta_x_center: 0.0,
                theta_y_center: 0.0,
                radial_count: 3,
                angular_count: 2,
                mode: EelsMeshMode::Uniform,
            },
            magic_energy_ev: 5.0,
            energy_loss_ev: arr1(&[10.0, 20.0]).view(),
            sigma_x_spectrum: arr1(&[0.1]).view(),
            sigma_y_spectrum: arr1(&[0.1, 0.2]).view(),
            pi_spectrum: arr1(&[0.1, 0.2]).view(),
            relativistic: true,
        }),
        Err(EelsError::SpectrumLengthMismatch {
            name: "sigma_x_spectrum",
            expected: 2,
            actual: 1,
        })
    );
    let owned_sources = sample_readsp_sources();
    let readsp_sources = readsp_source_views(&owned_sources);
    assert_eq!(
        eels_read_spectrum(EelsReadSpectrumInput {
            sources: &readsp_sources,
            orientation_averaged: false,
            cross_terms: true,
            polarization_min: 1,
            polarization_step: 4,
            polarization_max: 9,
        }),
        Err(EelsError::InvalidPolarizationRange {
            min: 1,
            step: 4,
            max: 9,
        })
    );
    assert_eq!(
        eels_read_spectrum(EelsReadSpectrumInput {
            sources: &readsp_sources[..1],
            orientation_averaged: true,
            cross_terms: false,
            polarization_min: 10,
            polarization_step: 1,
            polarization_max: 10,
        }),
        Err(EelsError::MissingPolarizationSource { index: 10 })
    );
}

#[test]
fn eels_integration_mesh_matches_feff_uniform_reference() -> Result<(), EelsError> {
    assert_mesh_summary(
        eels_integration_mesh(EelsMeshInput {
            collection_angle: 0.015,
            convergence_angle: 0.008,
            theta0: 0.001,
            theta_x_center: 0.001,
            theta_y_center: -0.002,
            radial_count: 2,
            angular_count: 2,
            mode: EelsMeshMode::Uniform,
        })?,
        MeshSummary {
            radial_count: 2,
            angular_count: 2,
            point_count: 8,
            theta_part: 0.005_750_000_000_000,
            sum_x: 0.008_000_000_000_000,
            sum_y: -0.016_000_000_000_000,
            sum_weight: 0.000_762_080_895_545,
            weighted_x: 0.000_000_762_080_896,
            weighted_y: -0.000_001_524_161_791,
        },
        &[
            (
                1,
                -0.004_750_000_000_000,
                -0.002_000_000_000_000,
                0.000_207_737_814_219,
            ),
            (
                4,
                -0.007_625_000_000_000,
                0.012_938_938_215_282,
                0.000_057_767_544_518,
            ),
            (
                8,
                0.018_250_000_000_000,
                -0.002_000_000_000_000,
                0.000_057_767_544_518,
            ),
        ],
    );
    Ok(())
}

#[test]
fn eels_integration_mesh_matches_feff_logarithmic_reference() -> Result<(), EelsError> {
    assert_mesh_summary(
        eels_integration_mesh(EelsMeshInput {
            collection_angle: 0.015,
            convergence_angle: 0.008,
            theta0: 0.001,
            theta_x_center: -0.0015,
            theta_y_center: 0.0005,
            radial_count: 3,
            angular_count: 2,
            mode: EelsMeshMode::Logarithmic,
        })?,
        MeshSummary {
            radial_count: 3,
            angular_count: 2,
            point_count: 18,
            theta_part: 0.003_833_333_333_333,
            sum_x: -0.027_000_000_000_000,
            sum_y: 0.009_000_000_000_000,
            sum_weight: 0.000_912_791_351_009,
            weighted_x: -0.000_001_369_187_027,
            weighted_y: 0.000_000_456_395_676,
        },
        &[
            (
                1,
                -0.002_000_000_000_000,
                0.000_500_000_000_000,
                0.000_001_570_796_327,
            ),
            (
                9,
                0.009_743_650_037_571,
                0.008_668_989_922_305,
                0.000_084_053_471_998,
            ),
            (
                18,
                0.012_397_915_761_656,
                0.000_500_000_000_000,
                0.000_084_053_471_998,
            ),
        ],
    );
    Ok(())
}

#[test]
fn eels_integration_mesh_matches_feff_one_dimensional_reference() -> Result<(), EelsError> {
    assert_mesh_summary(
        eels_integration_mesh(EelsMeshInput {
            collection_angle: 0.015,
            convergence_angle: 0.008,
            theta0: 0.001,
            theta_x_center: 0.002,
            theta_y_center: 0.001,
            radial_count: 3,
            angular_count: 2,
            mode: EelsMeshMode::OneDimensional,
        })?,
        MeshSummary {
            radial_count: 3,
            angular_count: 1,
            point_count: 3,
            theta_part: 0.003_833_333_333_333,
            sum_x: 0.023_295_831_523_313,
            sum_y: 0.003_000_000_000_000,
            sum_weight: 0.004_413_160_307_671,
            weighted_x: 0.000_067_837_163_754,
            weighted_y: 0.000_004_413_160_308,
        },
        &[
            (
                1,
                0.002_500_000_000_000,
                0.001_000_000_000_000,
                0.000_003_141_592_654,
            ),
            (
                1,
                0.002_500_000_000_000,
                0.001_000_000_000_000,
                0.000_003_141_592_654,
            ),
            (
                3,
                0.015_897_915_761_656,
                0.001_000_000_000_000,
                0.004_202_673_599_880,
            ),
        ],
    );
    Ok(())
}

#[test]
fn eels_mesh_rejects_invalid_inputs() {
    let input = EelsMeshInput {
        collection_angle: 0.015,
        convergence_angle: 0.008,
        theta0: 0.001,
        theta_x_center: 0.0,
        theta_y_center: 0.0,
        radial_count: 2,
        angular_count: 2,
        mode: EelsMeshMode::Uniform,
    };
    assert_eq!(
        eels_integration_mesh(EelsMeshInput {
            radial_count: 0,
            ..input
        }),
        Err(EelsError::InvalidMeshCount {
            name: "radial_count",
            value: 0,
        })
    );
    assert!(matches!(
        eels_integration_mesh(EelsMeshInput {
            collection_angle: -0.1,
            ..input
        }),
        Err(EelsError::InvalidMeshAngle {
            name: "collection_angle",
            ..
        })
    ));
    assert!(matches!(
        eels_integration_mesh(EelsMeshInput {
            theta0: 0.0,
            mode: EelsMeshMode::Logarithmic,
            ..input
        }),
        Err(EelsError::InvalidLogMeshParameter { name: "theta0", .. })
    ));
}

fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-14,
        "actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_close(actual, expected[(row, column)]);
    }
    assert_close(determinant_3x3(actual), 1.0);
}

fn assert_vector_close(actual: ArrayView1<'_, Real>, expected: ArrayView1<'_, Real>) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_close(actual, expected);
    }
}

fn assert_rect_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_close(actual, expected[(row, column)]);
    }
}

fn assert_relative_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-12 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, diff={}, tolerance={tolerance}",
        (actual - expected).abs()
    );
}

fn assert_selected_q_values(q_values: &RealVec) {
    let indices = [0, 1, 4, 9, 19];
    let expected = [
        5.0132553634977525e-2,
        1.0718958629739758e-1,
        3.3015066023371864e-1,
        9.60612700842798e-1,
        4.463626683435738,
    ];
    for (&index, &expected) in indices.iter().zip(expected.iter()) {
        assert_close(q_values[index], expected);
    }
}

fn assert_selected_gos_rows(actual: ArrayView2<'_, Real>, expected: &[[Real; 5]; 4]) {
    assert_eq!(actual.dim(), (FEFF_EELS_GOS_Q_COUNT, 4));
    let q_indices = [0, 1, 4, 9, 19];
    for (energy_index, expected_row) in expected.iter().enumerate() {
        for (&q_index, &expected_value) in q_indices.iter().zip(expected_row.iter()) {
            assert_relative_close(actual[(q_index, energy_index)], expected_value);
        }
    }
}

fn assert_collection_rows(actual: ArrayView2<'_, Real>, expected: &[[Real; 5]]) {
    assert_eq!(
        actual.dim(),
        (expected.len(), FEFF_EELS_COLLECTION_DEPENDENCE_COLUMN_COUNT)
    );
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_value) in expected_row.iter().enumerate() {
            assert_relative_close(actual[(row, column)], expected_value);
        }
    }
}

fn assert_readsp_tensor_rows(actual: ArrayView3<'_, Real>, expected: &[[Real; 9]; 3]) {
    assert_eq!(actual.dim(), (3, 3, 3));
    for (energy_index, expected_row) in expected.iter().enumerate() {
        for (component, &expected_value) in expected_row.iter().enumerate() {
            let row = component / 3;
            let column = component % 3;
            assert_close(actual[(energy_index, row, column)], expected_value);
        }
    }
}

fn determinant_3x3(matrix: ArrayView2<'_, Real>) -> Real {
    matrix[(0, 0)] * matrix[(1, 1)] * matrix[(2, 2)]
        + matrix[(0, 1)] * matrix[(1, 2)] * matrix[(2, 0)]
        + matrix[(1, 0)] * matrix[(2, 1)] * matrix[(0, 2)]
        - matrix[(2, 0)] * matrix[(1, 1)] * matrix[(0, 2)]
        - matrix[(1, 0)] * matrix[(0, 1)] * matrix[(2, 2)]
        - matrix[(0, 0)] * matrix[(2, 1)] * matrix[(1, 2)]
}

#[derive(Debug, Clone, Copy)]
struct MeshSummary {
    radial_count: usize,
    angular_count: usize,
    point_count: usize,
    theta_part: Real,
    sum_x: Real,
    sum_y: Real,
    sum_weight: Real,
    weighted_x: Real,
    weighted_y: Real,
}

fn assert_mesh_summary(
    mesh: EelsIntegrationMesh,
    expected: MeshSummary,
    points: &[(usize, Real, Real, Real)],
) {
    assert_eq!(mesh.setup.radial_count, expected.radial_count);
    assert_eq!(mesh.setup.angular_count, expected.angular_count);
    assert_eq!(mesh.setup.point_count, expected.point_count);
    assert_eq!(mesh.theta_x.len(), expected.point_count);
    assert_eq!(mesh.theta_y.len(), expected.point_count);
    assert_eq!(mesh.weights.len(), expected.point_count);
    assert_close(mesh.setup.theta_part, expected.theta_part);
    assert_close(mesh.theta_x.sum(), expected.sum_x);
    assert_close(mesh.theta_y.sum(), expected.sum_y);
    assert_close(mesh.weights.sum(), expected.sum_weight);
    assert_close(
        mesh.theta_x
            .iter()
            .zip(mesh.weights.iter())
            .map(|(&theta, &weight)| theta * weight)
            .sum(),
        expected.weighted_x,
    );
    assert_close(
        mesh.theta_y
            .iter()
            .zip(mesh.weights.iter())
            .map(|(&theta, &weight)| theta * weight)
            .sum(),
        expected.weighted_y,
    );
    for &(index, theta_x, theta_y, weight) in points {
        let offset = index - 1;
        assert_close(mesh.theta_x[offset], theta_x);
        assert_close(mesh.theta_y[offset], theta_y);
        assert_close(mesh.weights[offset], weight);
    }
}

#[derive(Debug)]
struct OwnedReadspSource {
    energy_loss_ev: RealVec,
    selected_spectrum: RealVec,
    atomic_background: RealVec,
}

fn sample_readsp_sources() -> Vec<OwnedReadspSource> {
    (1..=10)
        .map(|polarization_index| {
            let energy_loss_ev =
                Array1::from_shape_fn(3, |energy| 10.0 * (energy + 1) as Real + 0.25);
            let selected_spectrum = Array1::from_shape_fn(3, |energy| {
                let ip = polarization_index as Real;
                let row = (energy + 1) as Real;
                0.1 * ip + 0.01 * row + 0.001 * ip * row
            });
            let atomic_background = Array1::from_shape_fn(3, |energy| {
                let ip = polarization_index as Real;
                let row = (energy + 1) as Real;
                1.0 + 0.2 * ip + 0.03 * row
            });
            OwnedReadspSource {
                energy_loss_ev,
                selected_spectrum,
                atomic_background,
            }
        })
        .collect()
}

fn readsp_source_views(sources: &[OwnedReadspSource]) -> Vec<EelsReadSpectrumSource<'_>> {
    sources
        .iter()
        .enumerate()
        .map(|(index, source)| EelsReadSpectrumSource {
            polarization_index: index + 1,
            energy_loss_ev: source.energy_loss_ev.view(),
            selected_spectrum: source.selected_spectrum.view(),
            atomic_background: source.atomic_background.view(),
        })
        .collect()
}
