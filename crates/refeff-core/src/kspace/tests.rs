use ndarray::{Array2, Array3, Array4, ArrayView4, arr1, arr2, array};

use super::*;
use crate::Complex;

mod support;

use support::*;

const BASIS: [Vector3; 3] = [[1.1, 0.2, 0.05], [-0.1, 1.3, 0.04], [0.03, 0.2, 0.9]];

#[test]
fn bravais_lattice_matches_feff_ibravais_reference() -> Result<(), KSpaceError> {
    let cases = [
        (1, 'P', BravaisLattice::TriclinicPrimitive),
        (2, 'P', BravaisLattice::TriclinicPrimitive),
        (3, 'P', BravaisLattice::MonoclinicPrimitive),
        (15, 'C', BravaisLattice::MonoclinicBaseCentered),
        (16, 'P', BravaisLattice::OrthorhombicPrimitive),
        (74, 'I', BravaisLattice::OrthorhombicBodyCentered),
        (74, 'F', BravaisLattice::OrthorhombicFaceCentered),
        (75, 'P', BravaisLattice::TetragonalPrimitive),
        (142, 'I', BravaisLattice::TetragonalBodyCentered),
        (143, 'R', BravaisLattice::TrigonalPrimitive),
        (168, 'P', BravaisLattice::HexagonalPrimitive),
        (195, 'P', BravaisLattice::CubicPrimitive),
        (225, 'F', BravaisLattice::CubicFaceCentered),
        (229, 'I', BravaisLattice::CubicBodyCentered),
    ];

    for (space_group, lattice, expected) in cases {
        let bravais = bravais_lattice(space_group, lattice)?;
        assert_eq!(bravais, expected);
        assert_eq!(
            bravais_lattice_index(space_group, lattice)?,
            expected.index()
        );
    }
    Ok(())
}

#[test]
fn kpath_segments_match_feff_reference() -> Result<(), KSpaceError> {
    let orthorhombic = define_k_path(BravaisLattice::OrthorhombicPrimitive, 7, BASIS)?;
    assert_eq!(orthorhombic.effective_kpath, 7);
    assert_eq!(orthorhombic.labels, ["X -GS-GG", "GG-GD-Y "]);
    assert_vector_close(orthorhombic.start(0), [0.55, 0.1, 0.025])?;
    assert_vector_close(orthorhombic.end(0), [0.0, 0.0, 0.0])?;
    assert_vector_close(orthorhombic.end(1), [-0.05, 0.65, 0.02])?;

    let hexagonal = define_k_path(BravaisLattice::HexagonalPrimitive, 5, BASIS)?;
    assert_eq!(hexagonal.labels, ["K -T -GG"]);
    assert_vector_close(hexagonal.start(0), [-0.433_333_333_333_333_35, 0.8, 0.01])?;
    assert_vector_close(hexagonal.end(0), [0.0, 0.0, 0.0])?;

    let cubic = define_k_path(BravaisLattice::CubicPrimitive, 5, BASIS)?;
    assert_eq!(cubic.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
    assert_vector_close(cubic.end(0), [0.5, 0.0, 0.0])?;
    assert_vector_close(cubic.end(1), [0.0, 0.5, 0.0])?;
    assert_vector_close(cubic.end(2), [0.0, 0.0, 0.5])?;

    let face_default = define_k_path(BravaisLattice::CubicFaceCentered, 0, BASIS)?;
    assert_eq!(face_default.effective_kpath, 4);
    assert_eq!(face_default.labels, ["GG-GD-X "]);
    assert_vector_close(face_default.end(0), [0.565, 0.2, 0.475])?;

    let face_axes = define_k_path(BravaisLattice::CubicFaceCentered, 6, BASIS)?;
    assert_eq!(face_axes.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
    assert_vector_close(face_axes.end(0), [1.0, 0.0, 0.0])?;
    assert_vector_close(face_axes.end(1), [0.0, 1.0, 0.0])?;
    assert_vector_close(face_axes.end(2), [0.0, 0.0, 1.0])?;

    let body_default = define_k_path(BravaisLattice::CubicBodyCentered, 0, BASIS)?;
    assert_eq!(body_default.effective_kpath, 5);
    assert_eq!(body_default.labels, ["GG-GD-H "]);
    assert_vector_close(body_default.end(0), [0.485, 0.65, -0.405])?;

    let body_axes = define_k_path(BravaisLattice::CubicBodyCentered, 6, BASIS)?;
    assert_eq!(body_axes.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
    assert_vector_close(body_axes.end(0), [1.0, 0.0, 0.0])?;
    assert_vector_close(body_axes.end(1), [0.0, 1.0, 0.0])?;
    assert_vector_close(body_axes.end(2), [0.0, 0.0, 1.0])?;
    Ok(())
}

#[test]
fn band_k_path_mesh_matches_feff_bandtot_sampling() -> Result<(), KSpaceError> {
    let path = KPath {
        bravais: BravaisLattice::CubicPrimitive,
        requested_kpath: 1,
        effective_kpath: 1,
        labels: vec!["A -B    ".to_string(), "B -C    ".to_string()],
        starts: arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ends: arr2(&[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
    };
    let mesh = band_k_path_mesh(&path, 9)?;

    assert_eq!(mesh.labels, path.labels);
    assert_eq!(mesh.segment_point_counts, [3, 6]);
    assert_eq!(mesh.segment_end_indices, [3, 9]);
    assert_eq!(mesh.point_count(), 9);
    assert_matrix_close(
        mesh.k_points.view(),
        arr2(&[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, 0.8, 0.0],
            [1.0, 1.2, 0.0],
            [1.0, 1.6, 0.0],
            [1.0, 2.0, 0.0],
        ])
        .view(),
    );
    assert_array1_close(
        mesh.path_distances.view(),
        arr1(&[0.0, 0.5, 1.0, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]).view(),
    );
    Ok(())
}

#[test]
fn band_k_path_mesh_matches_feff_default_empty_path() -> Result<(), KSpaceError> {
    let path = KPath {
        bravais: BravaisLattice::CubicPrimitive,
        requested_kpath: 0,
        effective_kpath: 0,
        labels: Vec::new(),
        starts: Array2::zeros((0, 3)),
        ends: Array2::zeros((0, 3)),
    };
    let mesh = band_k_path_mesh(&path, 4)?;

    assert_eq!(mesh.labels, ["GG-x -1 "]);
    assert_eq!(mesh.segment_point_counts, [4]);
    assert_eq!(mesh.segment_end_indices, [4]);
    assert_matrix_close(
        mesh.k_points.view(),
        arr2(&[
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        .view(),
    );
    assert_array1_close(
        mesh.path_distances.view(),
        arr1(&[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]).view(),
    );
    Ok(())
}

#[test]
fn band_k_path_mesh_rejects_invalid_inputs() {
    let valid_shape = KPath {
        bravais: BravaisLattice::CubicPrimitive,
        requested_kpath: 1,
        effective_kpath: 1,
        labels: vec!["zero    ".to_string()],
        starts: arr2(&[[0.0, 0.0, 0.0]]),
        ends: arr2(&[[0.0, 0.0, 0.0]]),
    };
    assert_eq!(
        band_k_path_mesh(&valid_shape, 1),
        Err(KSpaceError::InvalidBandKPathPointTarget { point_count: 1 })
    );
    assert_eq!(
        band_k_path_mesh(&valid_shape, 2),
        Err(KSpaceError::DegenerateBandKPathLength)
    );

    let bad_shape = KPath {
        starts: arr2(&[[0.0, 0.0]]),
        ..valid_shape
    };
    assert!(matches!(
        band_k_path_mesh(&bad_shape, 2),
        Err(KSpaceError::InvalidKPathSegmentShape { .. })
    ));
}

#[test]
fn kspace_qjltab_matches_feff_strfunqjl_reference() -> Result<(), KSpaceError> {
    let qjltab = kspace_qjltab(2)?;

    assert_eq!(qjltab.shape(), &[3, 3]);
    assert_close(qjltab[(0, 0)], (1.0 / (2.0 * PI2)).sqrt());
    assert_close(qjltab[(0, 1)], (3.0 / (2.0 * PI2)).sqrt());
    assert_close(qjltab[(1, 1)], (3.0 / (2.0 * PI2)).sqrt());
    assert_close(qjltab[(0, 2)], (5.0 / (2.0 * PI2)).sqrt());
    assert_close(qjltab[(1, 2)], (5.0 / (6.0 * PI2)).sqrt());
    assert_close(qjltab[(2, 2)], (5.0 / (24.0 * PI2)).sqrt());
    assert_eq!(qjltab[(2, 1)], 0.0);
    Ok(())
}

#[test]
fn kspace_angular_tables_match_feff_strgaunt_lmax0_reference() -> Result<(), KSpaceError> {
    let tables = kspace_angular_tables(0, 2.0)?;

    assert_eq!(tables.angular_lmax, 0);
    assert_eq!(tables.harmonic_lmax, 0);
    assert_eq!(tables.angular_state_count, 1);
    assert_matrix_close(
        tables.qjltab.view(),
        arr2(&[[(1.0 / (2.0 * PI2)).sqrt()]]).view(),
    );
    assert_eq!(tables.gaunt_counts, [1]);
    assert_eq!(tables.gaunt_indices, [0]);
    assert_close(
        tables.gaunt_values[0],
        PI2.powi(2) * (1.0 / (2.0 * PI2)).sqrt(),
    );
    assert_complex_array1_close(tables.cipwl.view(), arr1(&[Complex::new(1.0, 0.0)]).view());
    Ok(())
}

#[test]
fn kspace_angular_tables_use_feff_triangular_lm_order() -> Result<(), KSpaceError> {
    let tables = kspace_angular_tables(1, 4.0)?;

    assert_eq!(tables.angular_state_count, 4);
    assert_eq!(tables.harmonic_lmax, 2);
    assert_eq!(tables.gaunt_counts.len(), 10);
    assert_eq!(tables.gaunt_counts[0], 1);
    assert_eq!(tables.gaunt_indices[0], 0);
    assert_eq!(
        tables.gaunt_values.len(),
        tables.gaunt_counts.iter().sum::<usize>()
    );
    assert_complex_array1_close(
        tables.cipwl.view(),
        arr1(&[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(-1.0, 0.0),
        ])
        .view(),
    );
    Ok(())
}

#[test]
fn kspace_q_pair_groups_match_feff_strvecgen_order_and_tolerance() -> Result<(), KSpaceError> {
    let positions = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0005, 0.0, 0.0]]);

    let groups = kspace_q_pair_groups(positions.view(), KSPACE_Q_PAIR_TOLERANCE)?;

    assert_eq!(groups.len(), 5);
    assert_eq!(groups.counts, [3, 2, 1, 2, 1]);
    assert_matrix_close(
        groups.offsets.view(),
        arr2(&[
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0005, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0005, 0.0, 0.0],
        ])
        .view(),
    );
    assert_close(groups.max_offset_norm, 2.0005);

    assert_eq!(groups.sites.shape(), &[5, 3, 2]);
    assert_eq!(groups.sites[(0, 0, 0)], 0);
    assert_eq!(groups.sites[(0, 0, 1)], 0);
    assert_eq!(groups.sites[(0, 1, 0)], 1);
    assert_eq!(groups.sites[(0, 1, 1)], 1);
    assert_eq!(groups.sites[(0, 2, 0)], 2);
    assert_eq!(groups.sites[(0, 2, 1)], 2);
    assert_eq!(groups.sites[(1, 0, 0)], 0);
    assert_eq!(groups.sites[(1, 0, 1)], 1);
    assert_eq!(groups.sites[(1, 1, 0)], 1);
    assert_eq!(groups.sites[(1, 1, 1)], 2);
    assert_eq!(groups.sites[(2, 0, 0)], 0);
    assert_eq!(groups.sites[(2, 0, 1)], 2);
    assert_eq!(groups.sites[(3, 0, 0)], 1);
    assert_eq!(groups.sites[(3, 0, 1)], 0);
    assert_eq!(groups.sites[(3, 1, 0)], 2);
    assert_eq!(groups.sites[(3, 1, 1)], 1);
    assert_eq!(groups.sites[(4, 0, 0)], 2);
    assert_eq!(groups.sites[(4, 0, 1)], 0);
    Ok(())
}

#[test]
fn kspace_q_pair_groups_rejects_invalid_inputs() {
    let empty = Array2::<Real>::zeros((0, 3));
    assert_eq!(
        kspace_q_pair_groups(empty.view(), KSPACE_Q_PAIR_TOLERANCE),
        Err(KSpaceError::InvalidStructureFactorCount {
            name: "q_pair_positions",
            count: 0,
        })
    );

    let positions = arr2(&[[0.0, 0.0, 0.0]]);
    assert_eq!(
        kspace_q_pair_groups(positions.view(), 0.0),
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "q_pair_tolerance",
            value: 0.0,
        })
    );
}

#[test]
fn kspace_direct_lattice_setup_matches_feff_strvecgen_and_straa_reference()
-> Result<(), KSpaceError> {
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

    let setup = kspace_direct_lattice_setup(
        reciprocal_basis_for_test(),
        q_pair_offsets.view(),
        0.01,
        1.0,
    )?;

    assert_eq!(setup.index_radius, 3);
    assert_matrix_i32_eq(
        setup.direct_indices.view(),
        arr2(&[[0, 0, 0], [0, 1, 0], [1, 0, 0]]).view(),
    );
    assert_eq!(setup.direct_counts, [0, 1, 1]);
    assert_eq!(setup.direct_index_by_pair.shape(), &[1, 3]);
    assert_eq!(setup.direct_index_by_pair[(0, 1)], 2);
    assert_eq!(setup.direct_index_by_pair[(0, 2)], 1);
    Ok(())
}

#[test]
fn kspace_direct_lattice_setup_rejects_invalid_inputs() {
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    assert_eq!(
        kspace_direct_lattice_setup(reciprocal_basis_for_test(), q_pair_offsets.view(), 0.0, 0.0,),
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "rmax",
            value: 0.0,
        })
    );
}

#[test]
fn kspace_reciprocal_lattice_setup_matches_feff_strvecgen_reference() -> Result<(), KSpaceError> {
    let setup = kspace_reciprocal_lattice_setup(reciprocal_basis_for_test(), 0.1, 0.0, 0.25)?;

    assert_close(setup.gmax_squared, 0.01);
    assert_eq!(setup.index_radius, 2);
    assert_matrix_i32_eq(
        setup.reciprocal_indices.view(),
        arr2(&[
            [0, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ])
        .view(),
    );
    Ok(())
}

#[test]
fn kspace_reciprocal_lattice_setup_rejects_invalid_inputs() {
    assert_eq!(
        kspace_reciprocal_lattice_setup(reciprocal_basis_for_test(), 0.0, 0.0, 0.25),
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "gmax",
            value: 0.0,
        })
    );
    assert_eq!(
        kspace_reciprocal_lattice_setup(reciprocal_basis_for_test(), 1.0, 0.5, 0.25),
        Err(KSpaceError::InvalidStructureFactorRange {
            name: "reduced_energy_probe",
            min: 0.5,
            max: 0.25,
        })
    );
}

#[test]
fn kspace_reciprocal_pair_phases_match_feff_straa_reference() -> Result<(), KSpaceError> {
    let reciprocal_indices = arr2(&[[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.5, 0.0]]);
    let phases = kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
        direct_basis: reciprocal_basis_for_test(),
        reciprocal_basis: reciprocal_basis_for_test(),
        reciprocal_indices: reciprocal_indices.view(),
        q_pair_offsets: q_pair_offsets.view(),
        eta: 2.0,
    })?;

    let d1term1 = -2.0 * PI2 / PI2.powi(3);
    let reciprocal_factor = d1term1 * (-0.5_f64).exp();
    let expected = array![
        [
            Complex::new(d1term1, 0.0),
            Complex::new(d1term1, 0.0),
            Complex::new(d1term1, 0.0)
        ],
        [
            Complex::new(reciprocal_factor, 0.0),
            Complex::new(0.0, reciprocal_factor),
            Complex::new(reciprocal_factor, 0.0)
        ],
        [
            Complex::new(reciprocal_factor, 0.0),
            Complex::new(reciprocal_factor, 0.0),
            Complex::new(-reciprocal_factor, 0.0)
        ]
    ];

    assert_eq!(phases.max_index_abs, 1);
    assert_close(phases.d1term1, d1term1);
    assert_complex_matrix_close(phases.reciprocal_pair_phases.view(), expected.view());
    Ok(())
}

#[test]
fn kspace_reciprocal_pair_phases_rejects_invalid_inputs() {
    let reciprocal_indices = arr2(&[[0, 0, 0]]);
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);

    assert_eq!(
        kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
            direct_basis: reciprocal_basis_for_test(),
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            q_pair_offsets: q_pair_offsets.view(),
            eta: 0.0,
        }),
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: 0.0,
        })
    );

    let empty_offsets = Array2::<Real>::zeros((0, 3));
    assert_eq!(
        kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
            direct_basis: reciprocal_basis_for_test(),
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            q_pair_offsets: empty_offsets.view(),
            eta: 1.0,
        }),
        Err(KSpaceError::InvalidStructureFactorCount {
            name: "q_pair_offsets",
            count: 0,
        })
    );

    let bad_indices = arr2(&[[0, 0]]);
    assert_eq!(
        kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
            direct_basis: reciprocal_basis_for_test(),
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: bad_indices.view(),
            q_pair_offsets: q_pair_offsets.view(),
            eta: 1.0,
        }),
        Err(KSpaceError::InvalidStructureFactorShape {
            name: "reciprocal_indices",
            rows: 1,
            columns: 2,
            expected_rows: 1,
            expected_columns: 3,
        })
    );
}

#[test]
fn kspace_direct_lattice_terms_match_feff_straa_reference() -> Result<(), KSpaceError> {
    let direct_indices = arr2(&[[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    let direct_index_by_pair = arr2(&[[1, 0], [0, 1]]);
    let direct_counts = [1, 2];
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0], [0.25, -0.5, 0.125]]);
    let qjltab = arr2(&[[2.0, 3.0], [0.0, 5.0]]);
    let eta = 0.5;

    let terms = kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
        direct_basis: reciprocal_basis_for_test(),
        direct_indices: direct_indices.view(),
        direct_index_by_pair: direct_index_by_pair.view(),
        direct_counts: &direct_counts,
        q_pair_offsets: q_pair_offsets.view(),
        lmax: 1,
        j22max: 1,
        qjltab: qjltab.view(),
        eta,
    })?;

    let mut expected = Array3::<Complex>::zeros((4, 2, 2));
    fill_expected_straa_direct_terms(&mut expected, 0, 0, [1.0, 0.0, 0.0], eta);
    fill_expected_straa_direct_terms(&mut expected, 0, 1, [-0.25, 0.5, -0.125], eta);
    fill_expected_straa_direct_terms(&mut expected, 1, 1, [0.75, 0.5, -0.125], eta);

    assert_eq!(terms.direct_terms.shape(), &[4, 2, 2]);
    assert_eq!(terms.radial_terms.shape(), &[2, 2, 2, 2]);
    assert_eq!(terms.max_index_abs, 1);
    assert_close(terms.q1, -0.5 * (eta / (PI2 / 2.0)).sqrt());
    assert_complex_cube_close(terms.direct_terms.view(), expected.view());
    assert_expected_straa_radial_terms(
        terms.radial_terms.view(),
        0,
        0,
        [
            0.186_418_271_243_704_78,
            0.320_250_819_379_323_4,
            0.221_530_487_179_646_1,
            0.372_836_542_487_409_56,
        ],
    );
    assert_expected_straa_radial_terms(
        terms.radial_terms.view(),
        0,
        1,
        [
            0.505_873_779_929_486_8,
            0.723_492_006_858_024_2,
            0.773_784_677_914_680_9,
            1.011_747_559_858_973_6,
        ],
    );
    assert_expected_straa_radial_terms(
        terms.radial_terms.view(),
        1,
        1,
        [
            0.221_926_299_520_226_72,
            0.372_274_579_046_808_44,
            0.271_852_923_934_706_67,
            0.443_852_599_040_453_44,
        ],
    );
    assert_close(terms.radial_terms[(0, 0, 1, 0)], 0.0);
    Ok(())
}

#[test]
fn kspace_direct_lattice_terms_are_exact_across_one_two_four_threads() -> Result<(), KSpaceError> {
    let direct_indices = arr2(&[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0]]);
    let direct_index_by_pair = arr2(&[[1, 1, 2], [3, 4, 0]]);
    let direct_counts = [2, 2, 2];
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0], [0.25, -0.5, 0.125], [-0.2, 0.1, 0.3]]);
    let qjltab = arr2(&[[2.0, 3.0, 4.0], [0.0, 5.0, 6.0], [0.0, 0.0, 7.0]]);
    let mut outputs = Vec::new();
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test thread pool");
        outputs.push(pool.install(|| {
            kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
                direct_basis: reciprocal_basis_for_test(),
                direct_indices: direct_indices.view(),
                direct_index_by_pair: direct_index_by_pair.view(),
                direct_counts: &direct_counts,
                q_pair_offsets: q_pair_offsets.view(),
                lmax: 2,
                j22max: 3,
                qjltab: qjltab.view(),
                eta: 0.5,
            })
        })?);
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
    Ok(())
}

#[test]
fn kspace_direct_lattice_terms_rejects_invalid_inputs() {
    let direct_indices = arr2(&[[0, 0, 0]]);
    let direct_index_by_pair = arr2(&[[0]]);
    let direct_counts = [1];
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let qjltab = arr2(&[[1.0]]);

    assert_eq!(
        kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            q_pair_offsets: q_pair_offsets.view(),
            lmax: 0,
            j22max: 0,
            qjltab: qjltab.view(),
            eta: 0.0,
        }),
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: 0.0,
        })
    );

    let bad_direct_index_by_pair = arr2(&[[3]]);
    assert_eq!(
        kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: bad_direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            q_pair_offsets: q_pair_offsets.view(),
            lmax: 0,
            j22max: 0,
            qjltab: qjltab.view(),
            eta: 1.0,
        }),
        Err(KSpaceError::StructureFactorIndexOutOfRange {
            name: "direct_index_by_pair",
            index: 3,
            len: 1,
        })
    );

    let empty_direct_index_by_pair = Array2::<usize>::zeros((0, 1));
    assert_eq!(
        kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: empty_direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            q_pair_offsets: q_pair_offsets.view(),
            lmax: 0,
            j22max: 0,
            qjltab: qjltab.view(),
            eta: 1.0,
        }),
        Err(KSpaceError::StructureFactorIndexOutOfRange {
            name: "direct_counts",
            index: 1,
            len: 1,
        })
    );
}

#[test]
fn kspace_energy_dependent_terms_match_feff_strcc_reference() -> Result<(), KSpaceError> {
    let (base_direct_terms, radial_terms) = sample_strcc_tables();
    let direct_counts = [1, 2];

    let terms = kspace_energy_dependent_terms(KSpaceEnergyDependentTermsInput {
        energy: Complex::new(0.2, 0.05),
        eta: 0.75,
        lmax: 1,
        base_direct_terms: base_direct_terms.view(),
        radial_terms: radial_terms.view(),
        direct_counts: &direct_counts,
    })?;

    assert_complex_array1_close(
        terms.d1term3.view(),
        arr1(&[
            Complex::new(1.302_704_901_651_116, 0.086_975_884_800_958_18),
            Complex::new(2.871_029_202_381_297_6, -0.160_435_183_467_892_82),
        ])
        .view(),
    );
    let expected_multipliers = array![
        [
            [Complex::new(0.52, 0.005), Complex::new(-0.29, 0.0025)],
            [Complex::new(0.0, 0.0), Complex::new(0.176, -0.006)]
        ],
        [
            [
                Complex::new(0.078_370_420_036_180_82, 0.456_354_653_526_696_24),
                Complex::new(0.099_318_723_038_350_09, 0.941_968_887_605_401_2)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(-0.030_793_403_141_716_174, -0.187_048_275_069_459_1)
            ]
        ]
    ];
    assert_complex_cube_close(terms.direct_multipliers.view(), expected_multipliers.view());

    let expected_direct_terms = array![
        [
            [
                Complex::new(0.052, 0.0005),
                Complex::new(-0.0146, -0.011_475)
            ],
            [Complex::new(0.0, 0.0), Complex::new(0.044_24, 0.005_54)]
        ],
        [
            [
                Complex::new(0.001_983_444_401_435_278_3, 0.093_622_043_306_424_68),
                Complex::new(-0.051_040_013_676_625_57, 0.148_247_643_753_494_7)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.002_315_688_155_261_475_3, -0.067_622_434_494_230_82)
            ]
        ],
        [
            [
                Complex::new(-0.003_870_153_200_747_523, 0.141_608_621_260_179_73),
                Complex::new(-0.069_367_208_000_952_6, 0.245_424_094_205_185_35)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.004_847_796_093_173_631, -0.087_251_064_095_428_21)
            ]
        ],
        [
            [
                Complex::new(-0.009_723_750_802_930_33, 0.189_595_199_213_934_78),
                Complex::new(-0.087_694_402_325_279_63, 0.342_600_544_656_875_95)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.007_379_904_031_085_784, -0.106_879_693_696_625_62)
            ]
        ]
    ];
    assert_complex_cube_close(terms.direct_terms.view(), expected_direct_terms.view());
    assert_complex_close(
        terms.d300,
        Complex::new(0.099_472_029_314_711_58, -0.010_073_665_415_888_097),
    );
    assert!(!terms.ewald_terms_exceed_threshold);
    Ok(())
}

#[test]
fn kspace_energy_dependent_terms_rejects_invalid_inputs() {
    let (base_direct_terms, radial_terms) = sample_strcc_tables();
    let direct_counts = [1, 2];

    assert_eq!(
        kspace_energy_dependent_terms(KSpaceEnergyDependentTermsInput {
            energy: Complex::new(0.2, 0.05),
            eta: 0.0,
            lmax: 1,
            base_direct_terms: base_direct_terms.view(),
            radial_terms: radial_terms.view(),
            direct_counts: &direct_counts,
        }),
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: 0.0,
        })
    );

    assert_eq!(
        kspace_energy_dependent_terms(KSpaceEnergyDependentTermsInput {
            energy: Complex::new(0.0, 0.0),
            eta: 0.75,
            lmax: 1,
            base_direct_terms: base_direct_terms.view(),
            radial_terms: radial_terms.view(),
            direct_counts: &direct_counts,
        }),
        Err(KSpaceError::DegenerateStructureFactorValue {
            name: "reduced_wave_number",
            index: 0,
        })
    );

    let bad_counts = [1, 3];
    assert_eq!(
        kspace_energy_dependent_terms(KSpaceEnergyDependentTermsInput {
            energy: Complex::new(0.2, 0.05),
            eta: 0.75,
            lmax: 1,
            base_direct_terms: base_direct_terms.view(),
            radial_terms: radial_terms.view(),
            direct_counts: &bad_counts,
        }),
        Err(KSpaceError::StructureFactorIndexOutOfRange {
            name: "direct_counts",
            index: 3,
            len: 3,
        })
    );

    let bad_radial_terms = Array4::<Real>::zeros((2, 1, 2, 2));
    assert!(matches!(
        kspace_energy_dependent_terms(KSpaceEnergyDependentTermsInput {
            energy: Complex::new(0.2, 0.05),
            eta: 0.75,
            lmax: 1,
            base_direct_terms: base_direct_terms.view(),
            radial_terms: bad_radial_terms.view(),
            direct_counts: &direct_counts,
        }),
        Err(KSpaceError::InvalidStructureFactorArray4Shape { .. })
    ));
}

#[test]
fn kspace_ewald_energy_tables_applies_feff_change_eta_retry() -> Result<(), KSpaceError> {
    let reciprocal_indices = arr2(&[[0, 0, 0], [1, 0, 0]]);
    let direct_indices = arr2(&[[1, 0, 0]]);
    let direct_index_by_pair = arr2(&[[0]]);
    let direct_counts = [1];
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let qjltab = arr2(&[[1.0, 1.0], [0.0, 1.0]]);

    let input = || KSpaceEwaldEnergyTablesInput {
        energy: Complex::new(30.0, 0.0),
        initial_eta: 0.75,
        lmax: 1,
        j22max: 1,
        direct_basis: reciprocal_basis_for_test(),
        reciprocal_basis: reciprocal_basis_for_test(),
        reciprocal_indices: reciprocal_indices.view(),
        direct_indices: direct_indices.view(),
        direct_index_by_pair: direct_index_by_pair.view(),
        direct_counts: &direct_counts,
        q_pair_offsets: q_pair_offsets.view(),
        qjltab: qjltab.view(),
    };
    let tables = kspace_ewald_energy_tables(input())?;
    let initial_tables = KSpaceInitialEwaldTables {
        eta: 0.75,
        reciprocal_pair_phases: kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
            direct_basis: reciprocal_basis_for_test(),
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            q_pair_offsets: q_pair_offsets.view(),
            eta: 0.75,
        })?,
        direct_lattice_terms: kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            q_pair_offsets: q_pair_offsets.view(),
            lmax: 1,
            j22max: 1,
            qjltab: qjltab.view(),
            eta: 0.75,
        })?,
    };
    let cached = kspace_ewald_energy_tables_from_initial(input(), &initial_tables)?;

    assert_eq!(tables.retry_count, 1);
    assert_eq!(cached, tables);
    assert_close(tables.eta, 1.05);
    assert!(!tables.energy_dependent_terms.ewald_terms_exceed_threshold);
    assert_close(
        tables.direct_lattice_terms.q1,
        -0.5 * (1.05 / (PI2 / 2.0)).sqrt(),
    );
    assert_complex_close(
        tables.energy_dependent_terms.d1term3[1],
        Complex::new(467_577_001_216.874_1, 0.0),
    );
    assert_complex_close(
        tables.energy_dependent_terms.d300,
        Complex::new(-7_731_376_744.381_291, 0.0),
    );
    assert_close(
        tables.reciprocal_pair_phases.d1term1,
        -2.0 * PI2 / PI2.powi(3),
    );
    Ok(())
}

#[test]
fn kspace_ewald_energy_tables_keeps_stable_eta_without_retry() -> Result<(), KSpaceError> {
    let reciprocal_indices = arr2(&[[0, 0, 0]]);
    let direct_indices = arr2(&[[1, 0, 0]]);
    let direct_index_by_pair = arr2(&[[0]]);
    let direct_counts = [1];
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let qjltab = arr2(&[[1.0]]);

    let input = || KSpaceEwaldEnergyTablesInput {
        energy: Complex::new(0.2, 0.05),
        initial_eta: 0.75,
        lmax: 0,
        j22max: 0,
        direct_basis: reciprocal_basis_for_test(),
        reciprocal_basis: reciprocal_basis_for_test(),
        reciprocal_indices: reciprocal_indices.view(),
        direct_indices: direct_indices.view(),
        direct_index_by_pair: direct_index_by_pair.view(),
        direct_counts: &direct_counts,
        q_pair_offsets: q_pair_offsets.view(),
        qjltab: qjltab.view(),
    };
    let tables = kspace_ewald_energy_tables(input())?;
    let initial_tables = KSpaceInitialEwaldTables {
        eta: 0.75,
        reciprocal_pair_phases: kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
            direct_basis: reciprocal_basis_for_test(),
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            q_pair_offsets: q_pair_offsets.view(),
            eta: 0.75,
        })?,
        direct_lattice_terms: kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            q_pair_offsets: q_pair_offsets.view(),
            lmax: 0,
            j22max: 0,
            qjltab: qjltab.view(),
            eta: 0.75,
        })?,
    };
    let cached = kspace_ewald_energy_tables_from_initial(input(), &initial_tables)?;

    assert_eq!(tables.retry_count, 0);
    assert_eq!(cached, tables);
    assert_close(tables.eta, 0.75);
    assert!(!tables.energy_dependent_terms.ewald_terms_exceed_threshold);
    Ok(())
}

#[test]
fn kspace_ewald_energy_tables_rejects_eta_above_feff_maximum() {
    let reciprocal_indices = arr2(&[[0, 0, 0]]);
    let direct_indices = arr2(&[[1, 0, 0]]);
    let direct_index_by_pair = arr2(&[[0]]);
    let direct_counts = [1];
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let qjltab = arr2(&[[1.0]]);

    assert_eq!(
        kspace_ewald_energy_tables(KSpaceEwaldEnergyTablesInput {
            energy: Complex::new(0.2, 0.05),
            initial_eta: 3.1,
            lmax: 0,
            j22max: 0,
            direct_basis: reciprocal_basis_for_test(),
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            q_pair_offsets: q_pair_offsets.view(),
            qjltab: qjltab.view(),
        }),
        Err(KSpaceError::EwaldEtaExceeded { eta: 3.1, max: 3.0 })
    );
}

#[test]
fn kspace_harmonic_polynomials_match_feff_strharpol_reference() -> Result<(), KSpaceError> {
    let qjltab = arr2(&[[2.0, 3.0, 5.0], [0.0, 7.0, 11.0], [0.0, 0.0, 13.0]]);
    let hp = kspace_harmonic_polynomials(KSpaceHarmonicPolynomialsInput {
        vector: [2.0, 3.0, 4.0],
        lmax: 2,
        qjltab: qjltab.view(),
    })?;
    assert_array1_close(
        hp.view(),
        arr1(&[2.0, 21.0, 12.0, 14.0, 468.0, 396.0, 47.5, 264.0, -195.0]).view(),
    );

    let zero = kspace_harmonic_polynomials(KSpaceHarmonicPolynomialsInput {
        vector: [1.0e-10, 0.0, 0.0],
        lmax: 2,
        qjltab: qjltab.view(),
    })?;
    assert_array1_close(
        zero.view(),
        arr1(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(),
    );
    Ok(())
}

#[test]
fn kspace_strbbdd_lattice_sum_matches_feff_reciprocal_direct_reference() -> Result<(), KSpaceError>
{
    let k = [0.1, -0.2, 0.3];
    let lmax = 1;
    let eta = 2.0;
    let energy = Complex::new(0.5, 0.25);
    let gmax_squared = 100.0;
    let reciprocal_basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let direct_basis = [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.75]];
    let reciprocal_indices = arr2(&[[1, -1, 0], [0, 0, 1]]);
    let reciprocal_pair_phases = array![
        [Complex::new(1.0, 0.2), Complex::new(0.5, -0.25)],
        [Complex::new(-0.2, 0.4), Complex::new(0.75, 0.1)]
    ];
    let qjltab = arr2(&[[0.5, 2.0], [0.0, -1.0]]);
    let d1term3 = arr1(&[Complex::new(2.0, 0.0), Complex::new(-1.0, 0.5)]);
    let q_pair_offsets = arr2(&[[99.0, 99.0, 99.0], [0.2, 0.0, -0.1]]);
    let direct_indices = arr2(&[[1, 0, 0], [-1, 1, 0], [0, 0, 1]]);
    let direct_index_by_pair = arr2(&[[0, 1], [0, 2]]);
    let direct_counts = [1, 2];
    let mut direct_terms = Array3::<Complex>::zeros((4, 2, 2));
    for mml in 0..4 {
        for direct_term in 0..2 {
            for q_pair in 0..2 {
                direct_terms[(mml, direct_term, q_pair)] = Complex::new(
                    0.05 * (mml + 1) as Real + 0.03 * direct_term as Real - 0.02 * q_pair as Real,
                    0.02 * mml as Real - 0.01 * direct_term as Real + 0.04 * q_pair as Real,
                );
            }
        }
    }
    let d300 = Complex::new(0.123, -0.456);

    let actual = kspace_strbbdd_lattice_sum(KSpaceStrbbddInput {
        k,
        lmax,
        eta,
        energy,
        gmax_squared,
        reciprocal_basis,
        reciprocal_indices: reciprocal_indices.view(),
        reciprocal_pair_phases: reciprocal_pair_phases.view(),
        d1term3: d1term3.view(),
        qjltab: qjltab.view(),
        q_pair_offsets: q_pair_offsets.view(),
        direct_basis,
        direct_indices: direct_indices.view(),
        direct_index_by_pair: direct_index_by_pair.view(),
        direct_counts: &direct_counts,
        direct_terms: direct_terms.view(),
        d300,
    })?;

    let mut expected = Array2::<Complex>::zeros((4, 2));
    let reciprocal_power_base = [
        (-2.0 * dot_for_test(reciprocal_basis[0], k) / eta).exp(),
        (-2.0 * dot_for_test(reciprocal_basis[1], k) / eta).exp(),
        (-2.0 * dot_for_test(reciprocal_basis[2], k) / eta).exp(),
    ];
    let f0 = (-dot_for_test(k, k) / eta).exp();
    for reciprocal in 0..2 {
        let g = [
            reciprocal_indices[(reciprocal, 0)],
            reciprocal_indices[(reciprocal, 1)],
            reciprocal_indices[(reciprocal, 2)],
        ];
        let kn = [
            k[0] + Real::from(g[0]) * reciprocal_basis[0][0]
                + Real::from(g[1]) * reciprocal_basis[1][0]
                + Real::from(g[2]) * reciprocal_basis[2][0],
            k[1] + Real::from(g[0]) * reciprocal_basis[0][1]
                + Real::from(g[1]) * reciprocal_basis[1][1]
                + Real::from(g[2]) * reciprocal_basis[2][1],
            k[2] + Real::from(g[0]) * reciprocal_basis[0][2]
                + Real::from(g[1]) * reciprocal_basis[1][2]
                + Real::from(g[2]) * reciprocal_basis[2][2],
        ];
        let hp = match reciprocal {
            0 => [0.5, 1.2, 0.6, -1.1],
            _ => [0.5, 0.2, 2.6, -0.1],
        };
        let denominator = Complex::new(dot_for_test(kn, kn), 0.0) - energy;
        let f1 = Complex::new(
            f0 * reciprocal_power_base[0].powi(g[0])
                * reciprocal_power_base[1].powi(g[1])
                * reciprocal_power_base[2].powi(g[2]),
            0.0,
        ) / denominator;
        for q_pair in 0..2 {
            for mml in 0..4 {
                let angular_momentum = if mml == 0 { 0 } else { 1 };
                expected[(mml, q_pair)] += f1
                    * reciprocal_pair_phases[(reciprocal, q_pair)]
                    * d1term3[angular_momentum]
                    * hp[mml];
            }
        }
    }

    let q_pair_phase = Complex::new(0.0, PI2 * dot_for_test(k, [0.2, 0.0, -0.1])).exp();
    for mml in 0..4 {
        expected[(mml, 1)] *= q_pair_phase;
    }

    let direct_power_base = [
        Complex::new(0.0, PI2 * dot_for_test(direct_basis[0], k)).exp(),
        Complex::new(0.0, PI2 * dot_for_test(direct_basis[1], k)).exp(),
        Complex::new(0.0, PI2 * dot_for_test(direct_basis[2], k)).exp(),
    ];
    let direct_phase_0 = direct_power_base[0];
    let direct_phase_1 = direct_power_base[0].powi(-1) * direct_power_base[1];
    let direct_phase_2 = direct_power_base[2];
    for mml in 0..4 {
        expected[(mml, 0)] += direct_phase_0 * direct_terms[(mml, 0, 0)];
        expected[(mml, 1)] +=
            direct_phase_1 * direct_terms[(mml, 0, 1)] + direct_phase_2 * direct_terms[(mml, 1, 1)];
    }
    expected[(0, 0)] += d300;

    assert_complex_matrix_close(actual.view(), expected.view());
    Ok(())
}

#[test]
fn kspace_strbbdd_lattice_sum_rejects_invalid_inputs() {
    let reciprocal_indices = Array2::<i32>::zeros((0, 3));
    let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
    let d1term3 = arr1(&[] as &[Complex]);
    let qjltab = arr2(&[[1.0]]);
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let direct_indices = arr2(&[[0, 0, 0]]);
    let direct_index_by_pair = arr2(&[[3]]);
    let direct_counts = [1];
    let direct_terms = Array3::<Complex>::zeros((1, 1, 1));

    let result = kspace_strbbdd_lattice_sum(KSpaceStrbbddInput {
        k: [0.0, 0.0, 0.0],
        lmax: 0,
        eta: 1.0,
        energy: Complex::new(0.0, 0.0),
        gmax_squared: 1.0,
        reciprocal_basis: reciprocal_basis_for_test(),
        reciprocal_indices: reciprocal_indices.view(),
        reciprocal_pair_phases: reciprocal_pair_phases.view(),
        d1term3: d1term3.view(),
        qjltab: qjltab.view(),
        q_pair_offsets: q_pair_offsets.view(),
        direct_basis: reciprocal_basis_for_test(),
        direct_indices: direct_indices.view(),
        direct_index_by_pair: direct_index_by_pair.view(),
        direct_counts: &direct_counts,
        direct_terms: direct_terms.view(),
        d300: Complex::new(0.0, 0.0),
    });
    let error = result.unwrap_err();
    assert!(
        matches!(
            error,
            KSpaceError::StructureFactorIndexOutOfRange {
                name: "direct_index_by_pair",
                index: 3,
                len: 1
            }
        ),
        "{error:?}"
    );

    let result = kspace_strbbdd_lattice_sum(KSpaceStrbbddInput {
        k: [0.0, 0.0, 0.0],
        lmax: 0,
        eta: 0.0,
        energy: Complex::new(0.0, 0.0),
        gmax_squared: 1.0,
        reciprocal_basis: reciprocal_basis_for_test(),
        reciprocal_indices: reciprocal_indices.view(),
        reciprocal_pair_phases: reciprocal_pair_phases.view(),
        d1term3: d1term3.view(),
        qjltab: qjltab.view(),
        q_pair_offsets: q_pair_offsets.view(),
        direct_basis: reciprocal_basis_for_test(),
        direct_indices: direct_indices.view(),
        direct_index_by_pair: direct_index_by_pair.view(),
        direct_counts: &direct_counts,
        direct_terms: direct_terms.view(),
        d300: Complex::new(0.0, 0.0),
    });
    assert!(matches!(
        result,
        Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: 0.0
        })
    ));
}

#[test]
fn kspace_strset_non_rel_from_lattice_sum_composes_strbbdd_and_strset() -> Result<(), KSpaceError> {
    let reciprocal_indices = Array2::<i32>::zeros((0, 3));
    let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
    let d1term3 = arr1(&[Complex::new(1.0, 0.0)]);
    let qjltab = arr2(&[[1.0]]);
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let direct_indices = Array2::<i32>::zeros((0, 3));
    let direct_index_by_pair = Array2::<usize>::zeros((0, 1));
    let direct_counts = [0_usize];
    let direct_terms = Array3::<Complex>::zeros((1, 0, 1));
    let q_pair_sites = array![[[0_usize, 0]]];
    let q_pair_counts = [1_usize];
    let site_offsets = [0_usize];
    let site_state_counts = [1_usize];
    let gaunt_counts = [1_usize];
    let gaunt_indices = [0_usize];
    let gaunt_values = [2.0];
    let cipwl = arr1(&[Complex::new(1.0, 0.0)]);

    let matrices = kspace_strset_non_rel_from_lattice_sum(KSpaceStrsetNonRelFromLatticeSumInput {
        lattice_sum: KSpaceStrbbddInput {
            k: [0.0, 0.0, 0.0],
            lmax: 0,
            eta: 1.0,
            energy: Complex::new(0.0, 0.0),
            gmax_squared: 1.0,
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            reciprocal_pair_phases: reciprocal_pair_phases.view(),
            d1term3: d1term3.view(),
            qjltab: qjltab.view(),
            q_pair_offsets: q_pair_offsets.view(),
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            direct_terms: direct_terms.view(),
            d300: Complex::new(1.0, 0.25),
        },
        angular_state_count: 1,
        q_pair_sites: q_pair_sites.view(),
        q_pair_counts: &q_pair_counts,
        site_offsets: &site_offsets,
        site_state_counts: &site_state_counts,
        gaunt_counts: &gaunt_counts,
        gaunt_indices: &gaunt_indices,
        gaunt_values: &gaunt_values,
        cipwl: cipwl.view(),
        wave_number: Complex::new(0.5, 0.25),
    })?;

    assert_complex_matrix_close(
        matrices.dllmmke.view(),
        arr2(&[[Complex::new(1.0, 0.25)]]).view(),
    );
    assert_complex_matrix_close(
        matrices.taukinv.view(),
        arr2(&[[Complex::new(-1.75, -1.0)]]).view(),
    );
    Ok(())
}

#[test]
fn kspace_strset_rel_from_lattice_sum_composes_strbbdd_and_strset() -> Result<(), KSpaceError> {
    let reciprocal_indices = Array2::<i32>::zeros((0, 3));
    let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
    let d1term3 = arr1(&[Complex::new(1.0, 0.0)]);
    let qjltab = arr2(&[[1.0]]);
    let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
    let direct_indices = Array2::<i32>::zeros((0, 3));
    let direct_index_by_pair = Array2::<usize>::zeros((0, 1));
    let direct_counts = [0_usize];
    let direct_terms = Array3::<Complex>::zeros((1, 0, 1));
    let q_pair_sites = array![[[0_usize, 0]]];
    let q_pair_counts = [1_usize];
    let site_offsets = [0_usize];
    let site_state_counts = [1_usize];
    let gaunt_counts = [1_usize];
    let gaunt_indices = [0_usize];
    let gaunt_values = [3.0];
    let cipwl = arr1(&[Complex::new(1.0, 0.0)]);
    let rel_component_counts = arr2(&[[1_usize], [0]]);
    let rel_component_indices = array![[[0_usize], [0]]];
    let rel_component_coefficients = array![[[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]]];

    let matrices = kspace_strset_rel_from_lattice_sum(KSpaceStrsetRelFromLatticeSumInput {
        lattice_sum: KSpaceStrbbddInput {
            k: [0.0, 0.0, 0.0],
            lmax: 0,
            eta: 1.0,
            energy: Complex::new(0.0, 0.0),
            gmax_squared: 1.0,
            reciprocal_basis: reciprocal_basis_for_test(),
            reciprocal_indices: reciprocal_indices.view(),
            reciprocal_pair_phases: reciprocal_pair_phases.view(),
            d1term3: d1term3.view(),
            qjltab: qjltab.view(),
            q_pair_offsets: q_pair_offsets.view(),
            direct_basis: reciprocal_basis_for_test(),
            direct_indices: direct_indices.view(),
            direct_index_by_pair: direct_index_by_pair.view(),
            direct_counts: &direct_counts,
            direct_terms: direct_terms.view(),
            d300: Complex::new(1.0, 0.0),
        },
        angular_state_count: 1,
        q_pair_sites: q_pair_sites.view(),
        q_pair_counts: &q_pair_counts,
        site_offsets: &site_offsets,
        site_state_counts: &site_state_counts,
        gaunt_counts: &gaunt_counts,
        gaunt_indices: &gaunt_indices,
        gaunt_values: &gaunt_values,
        cipwl: cipwl.view(),
        rel_component_counts: rel_component_counts.view(),
        rel_component_indices: rel_component_indices.view(),
        rel_component_coefficients: rel_component_coefficients.view(),
        wave_number: Complex::new(0.5, 0.0),
    })?;

    assert_complex_matrix_close(
        matrices.dllmmke.view(),
        arr2(&[[Complex::new(1.0, 0.0)]]).view(),
    );
    assert_complex_matrix_close(
        matrices.taukinv.view(),
        arr2(&[[Complex::new(-3.0, -0.5)]]).view(),
    );
    Ok(())
}

#[test]
fn kspace_strset_non_relativistic_matches_feff_strset_reference() -> Result<(), KSpaceError> {
    let dllmmke = array![
        [Complex::new(1.0, 0.5), Complex::new(0.2, -0.1)],
        [Complex::new(-0.25, 0.1), Complex::new(0.3, 0.4)],
        [Complex::new(0.4, -0.2), Complex::new(-0.5, 0.25)],
        [Complex::new(-0.3, 0.7), Complex::new(0.1, -0.6)]
    ];
    let mut q_pair_sites = Array3::<usize>::zeros((2, 2, 2));
    q_pair_sites[(0, 0, 0)] = 0;
    q_pair_sites[(0, 0, 1)] = 0;
    q_pair_sites[(0, 1, 0)] = 1;
    q_pair_sites[(0, 1, 1)] = 1;
    q_pair_sites[(1, 0, 0)] = 2;
    q_pair_sites[(1, 0, 1)] = 0;
    let q_pair_counts = [2, 1];
    let site_offsets = [0, 3, 6];
    let site_state_counts = [3, 3, 3];
    let gaunt_counts = [1, 2, 0, 1, 1, 1];
    let gaunt_indices = [0, 1, 2, 3, 0, 2];
    let gaunt_values = [2.0, -1.0, 0.5, 1.25, -0.75, 3.0];
    let cipwl = arr1(&[
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
    ]);

    let taukinv = kspace_strset_non_relativistic(KSpaceStrsetNonRelInput {
        angular_state_count: 3,
        dllmmke: dllmmke.view(),
        q_pair_sites: q_pair_sites.view(),
        q_pair_counts: &q_pair_counts,
        site_offsets: &site_offsets,
        site_state_counts: &site_state_counts,
        gaunt_counts: &gaunt_counts,
        gaunt_indices: &gaunt_indices,
        gaunt_values: &gaunt_values,
        cipwl: cipwl.view(),
        wave_number: Complex::new(2.0, 0.5),
    })?;

    let mut expected = Array2::<Complex>::zeros((9, 9));
    expected[(0, 0)] = Complex::new(-1.5, -3.0);
    expected[(1, 0)] = Complex::new(-0.2, -0.45);
    expected[(0, 1)] = Complex::new(0.2, 0.45);
    expected[(1, 1)] = Complex::new(0.5, -2.0);
    expected[(2, 0)] = Complex::new(-0.375, 0.875);
    expected[(0, 2)] = Complex::new(-0.375, 0.875);
    expected[(2, 1)] = Complex::new(-0.375, 0.75);
    expected[(1, 2)] = Complex::new(0.375, -0.75);
    expected[(2, 2)] = Complex::new(-0.7, -1.4);

    for row in 0..3 {
        for column in 0..3 {
            expected[(row + 3, column + 3)] = expected[(row, column)];
        }
    }

    expected[(6, 0)] = Complex::new(-0.4, 0.2);
    expected[(7, 0)] = Complex::new(-0.275, 0.55);
    expected[(6, 1)] = Complex::new(0.275, -0.55);
    expected[(8, 0)] = Complex::new(0.125, -0.75);
    expected[(6, 2)] = Complex::new(0.125, -0.75);
    expected[(8, 1)] = Complex::new(0.075, 0.15);
    expected[(7, 2)] = Complex::new(-0.075, -0.15);
    expected[(8, 2)] = Complex::new(1.5, -0.75);

    assert_complex_matrix_close(taukinv.view(), expected.view());
    Ok(())
}

#[test]
fn kspace_strset_non_relativistic_rejects_invalid_inputs() {
    let dllmmke = array![[Complex::new(1.0, 0.0)]];
    let q_pair_sites = array![[[2, 0]]];
    let q_pair_counts = [1];
    let site_offsets = [0];
    let site_state_counts = [1];
    let gaunt_counts = [1];
    let gaunt_indices = [0];
    let gaunt_values = [1.0];
    let cipwl = arr1(&[Complex::new(1.0, 0.0)]);

    let result = kspace_strset_non_relativistic(KSpaceStrsetNonRelInput {
        angular_state_count: 1,
        dllmmke: dllmmke.view(),
        q_pair_sites: q_pair_sites.view(),
        q_pair_counts: &q_pair_counts,
        site_offsets: &site_offsets,
        site_state_counts: &site_state_counts,
        gaunt_counts: &gaunt_counts,
        gaunt_indices: &gaunt_indices,
        gaunt_values: &gaunt_values,
        cipwl: cipwl.view(),
        wave_number: Complex::new(1.0, 0.0),
    });

    assert!(matches!(
        result,
        Err(KSpaceError::StructureFactorIndexOutOfRange {
            name: "q_pair_sites",
            index: 2,
            len: 1
        })
    ));
}

#[test]
fn kspace_strset_relativistic_matches_feff_strset_reference() -> Result<(), KSpaceError> {
    let dllmmke = array![
        [Complex::new(1.0, 0.5), Complex::new(-0.5, 0.2)],
        [Complex::new(0.2, -0.1), Complex::new(0.1, 0.3)],
        [Complex::new(-0.3, 0.4), Complex::new(0.4, -0.2)]
    ];
    let mut q_pair_sites = Array3::<usize>::zeros((2, 2, 2));
    q_pair_sites[(0, 0, 0)] = 0;
    q_pair_sites[(0, 0, 1)] = 0;
    q_pair_sites[(0, 1, 0)] = 1;
    q_pair_sites[(0, 1, 1)] = 1;
    q_pair_sites[(1, 0, 0)] = 2;
    q_pair_sites[(1, 0, 1)] = 0;
    let q_pair_counts = [2, 1];
    let site_offsets = [0, 2, 4];
    let site_state_counts = [2, 2, 2];
    let gaunt_counts = [1, 1, 1];
    let gaunt_indices = [0, 1, 2];
    let gaunt_values = [1.0, 1.0, 1.0];
    let cipwl = arr1(&[Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]);
    let rel_component_counts = arr2(&[[1_usize, 1], [1, 1]]);
    let mut rel_component_indices = Array3::<usize>::zeros((1, 2, 2));
    rel_component_indices[(0, 0, 0)] = 0;
    rel_component_indices[(0, 1, 0)] = 1;
    rel_component_indices[(0, 0, 1)] = 1;
    rel_component_indices[(0, 1, 1)] = 0;
    let mut rel_component_coefficients = Array3::<Complex>::zeros((1, 2, 2));
    rel_component_coefficients[(0, 0, 0)] = Complex::new(1.0, 0.0);
    rel_component_coefficients[(0, 1, 0)] = Complex::new(0.5, 0.25);
    rel_component_coefficients[(0, 0, 1)] = Complex::new(-0.25, 0.5);
    rel_component_coefficients[(0, 1, 1)] = Complex::new(0.75, 0.0);

    let taukinv = kspace_strset_relativistic(KSpaceStrsetRelInput {
        angular_state_count: 2,
        dllmmke: dllmmke.view(),
        q_pair_sites: q_pair_sites.view(),
        q_pair_counts: &q_pair_counts,
        site_offsets: &site_offsets,
        site_state_counts: &site_state_counts,
        gaunt_counts: &gaunt_counts,
        gaunt_indices: &gaunt_indices,
        gaunt_values: &gaunt_values,
        cipwl: cipwl.view(),
        rel_component_counts: rel_component_counts.view(),
        rel_component_indices: rel_component_indices.view(),
        rel_component_coefficients: rel_component_coefficients.view(),
        wave_number: Complex::new(0.6, 0.2),
    })?;

    let mut expected = Array2::<Complex>::zeros((6, 6));
    expected[(0, 0)] = Complex::new(-0.643_75, -1.412_5);
    expected[(0, 1)] = Complex::new(-0.2, -0.056_25);
    expected[(1, 0)] = Complex::new(-0.075, 0.193_75);
    expected[(1, 1)] = Complex::new(-0.293_75, -0.931_25);
    for row in 0..2 {
        for column in 0..2 {
            expected[(row + 2, column + 2)] = expected[(row, column)];
        }
    }
    expected[(4, 0)] = Complex::new(0.375, -0.137_5);
    expected[(4, 1)] = Complex::new(0.118_75, -0.268_75);
    expected[(5, 0)] = Complex::new(-0.256_25, -0.143_75);
    expected[(5, 1)] = Complex::new(0.156_25, -0.05);

    assert_complex_matrix_close(taukinv.view(), expected.view());
    Ok(())
}

#[test]
fn kspace_strset_relativistic_rejects_invalid_transform_indices() {
    let dllmmke = array![[Complex::new(1.0, 0.0)]];
    let q_pair_sites = array![[[0, 0]]];
    let q_pair_counts = [1];
    let site_offsets = [0];
    let site_state_counts = [1];
    let gaunt_counts = [1];
    let gaunt_indices = [0];
    let gaunt_values = [1.0];
    let cipwl = arr1(&[Complex::new(1.0, 0.0)]);
    let rel_component_counts = arr2(&[[1_usize], [0]]);
    let rel_component_indices = array![[[1_usize], [0]]];
    let rel_component_coefficients = array![[[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]]];

    let result = kspace_strset_relativistic(KSpaceStrsetRelInput {
        angular_state_count: 1,
        dllmmke: dllmmke.view(),
        q_pair_sites: q_pair_sites.view(),
        q_pair_counts: &q_pair_counts,
        site_offsets: &site_offsets,
        site_state_counts: &site_state_counts,
        gaunt_counts: &gaunt_counts,
        gaunt_indices: &gaunt_indices,
        gaunt_values: &gaunt_values,
        cipwl: cipwl.view(),
        rel_component_counts: rel_component_counts.view(),
        rel_component_indices: rel_component_indices.view(),
        rel_component_coefficients: rel_component_coefficients.view(),
        wave_number: Complex::new(1.0, 0.0),
    });

    assert!(matches!(
        result,
        Err(KSpaceError::StructureFactorIndexOutOfRange {
            name: "rel_component_indices",
            index: 1,
            len: 1
        })
    ));
}

#[test]
fn reciprocal_coordinate_helpers_match_feff_reference() -> Result<(), KSpaceError> {
    let direct = arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
    let reciprocal = arr2(&[
        [PI2 / 2.0, 0.0, 0.0],
        [0.0, PI2 / 3.0, 0.0],
        [0.0, 0.0, PI2 / 4.0],
    ]);
    let vector = [3.2, -1.55, 8.2];

    let subtracted = subtract_lattice_translation(reciprocal.view(), vector)?;
    assert_eq!(subtracted.translation_count, 5);
    assert_close(subtracted.vector[0], -0.4);
    assert_close(subtracted.vector[1], 0.483_333_333_333_333_4);
    assert_close(subtracted.vector[2], 0.05);

    let reduced = reduce_to_lattice_cell(direct.view(), reciprocal.view(), vector)?;
    assert_eq!(reduced.translation_count, 5);
    assert_close(reduced.vector[0], 1.2);
    assert_close(reduced.vector[1], 1.45);
    assert_close(reduced.vector[2], 0.2);

    let operation = array![[1, -2, 0], [3, 0, 1], [-1, 2, 1]];
    let changed = change_cartesian_basis(reciprocal.view(), direct.view(), operation.view())?;
    assert_close(changed[(0, 0)], PI2);
    assert_close(changed[(0, 1)], -3.0 * PI2);
    assert_close(changed[(0, 2)], 0.0);
    assert_close(changed[(1, 0)], 2.0 * PI2);
    assert_close(changed[(1, 1)], 0.0);
    assert_close(changed[(1, 2)], 4.0 * PI2 / 3.0);
    assert_close(changed[(2, 0)], -std::f64::consts::PI);
    assert_close(changed[(2, 1)], 3.0 * std::f64::consts::PI);
    assert_close(changed[(2, 2)], PI2);
    Ok(())
}

#[test]
fn reciprocal_lattice_vectors_match_feff_gbass_reference() -> Result<(), KSpaceError> {
    let direct = arr2(&[[2.0, 0.3, -0.2], [0.1, 3.0, 0.5], [0.2, 0.4, 4.0]]);
    let reciprocal = reciprocal_lattice_vectors(direct.view())?;
    let expected = skew_reciprocal_basis();
    assert_matrix_close(reciprocal.view(), expected.view());

    let roundtrip = reciprocal_lattice_vectors(reciprocal.view())?;
    assert_matrix_close(roundtrip.view(), direct.view());
    Ok(())
}

#[test]
fn ldos_weyl_kmesh_matches_feff_changeklist_reference() -> Result<(), KSpaceError> {
    let mesh = ldos_weyl_kmesh(7)?;
    let expected = arr2(&[
        [
            1.218_062_399_172_268_3e-1,
            3.027_484_039_224_245e-1,
            8.878_581_578_314_426e-2,
        ],
        [
            -2.563_875_201_655_463_3e-1,
            1.054_968_078_448_490_5e-1,
            -3.224_283_684_337_115e-1,
        ],
        [
            3.654_187_197_516_805e-1,
            -9.175_478_823_272_654e-2,
            2.663_574_473_494_332_3e-1,
        ],
        [
            -1.277_504_033_109_266_9e-2,
            -2.890_063_843_103_019e-1,
            -1.448_567_368_674_229_5e-1,
        ],
        [
            -3.909_688_004_138_658_4e-1,
            -4.862_579_803_878_777e-1,
            4.439_290_789_157_208_7e-1,
        ],
        [
            2.308_374_395_033_61e-1,
            3.164_904_235_345_469e-1,
            3.271_489_469_886_646e-2,
        ],
        [
            -1.473_563_205_794_121_7e-1,
            1.192_388_274_569_715_6e-1,
            -3.784_992_895_179_897e-1,
        ],
    ]);
    assert_matrix_close(mesh.k_points.view(), expected.view());
    assert_array1_close(mesh.weights.view(), arr1(&[1.0 / 7.0; 7]).view());
    assert_eq!(
        ldos_weyl_kmesh(0),
        Err(KSpaceError::InvalidKMeshPointTarget { mesh_points: 0 })
    );
    Ok(())
}

#[test]
fn kmesh_bravais_basis_matches_feff_bravais_reference() -> Result<(), KSpaceError> {
    let right_angles = [BRAVAIS_RIGHT_ANGLE; 3];
    let triclinic_angles = [1.2, 1.3, 1.1];
    let monoclinic_angles = [BRAVAIS_RIGHT_ANGLE, BRAVAIS_RIGHT_ANGLE, 1.2];
    let cases = vec![
        (
            "H  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [true, false, false],
            false,
            17.901_484_003_701_512,
            arr2(&[
                [1.732_050_776_481_628_4, -1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 4.0],
            ]),
        ),
        (
            "F  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [1.0, 1.5, 2.0],
            0.5,
            [true, true, true],
            true,
            41.341_705_691_712_875,
            arr2(&[[0.0, 1.5, 2.0], [1.0, 0.0, 2.0], [1.0, 1.5, 0.0]]),
        ),
        (
            "B  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [1.0, 1.5, 2.0],
            0.5,
            [true, true, true],
            true,
            20.670_852_845_856_437,
            arr2(&[[-1.0, 1.5, 2.0], [1.0, -1.5, 2.0], [1.0, 1.5, -2.0]]),
        ),
        (
            "P  ",
            [2.0, 3.0, 4.0],
            triclinic_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [false, false, false],
            false,
            12.539_759_914_879_173,
            arr2(&[
                [
                    1.768_622_103_620_578_5,
                    0.765_345_256_288_045_5,
                    0.534_997_657_249_174_7,
                ],
                [0.0, 2.796_117_257_901_679, 1.087_073_263_430_020_9],
                [0.0, 0.0, 4.0],
            ]),
        ),
        (
            "C  ",
            [2.0, 3.0, 4.0],
            monoclinic_angles,
            [0.932_039_085_967_226_3, 3.0, 2.0],
            1.0,
            [false, true, false],
            false,
            22.178_096_559_550_61,
            arr2(&[
                [0.932_039_085_967_226_3, 0.362_357_754_476_673_6, -2.0],
                [0.0, 3.0, 0.0],
                [0.932_039_085_967_226_3, 0.362_357_754_476_673_6, 2.0],
            ]),
        ),
        (
            "P  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [false, false, false],
            true,
            10.335_426_422_928_219,
            arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
        ),
        (
            "CXZ",
            [2.0, 3.0, 4.0],
            right_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [false, true, false],
            true,
            20.670_852_845_856_437,
            arr2(&[[1.0, 0.0, -2.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]]),
        ),
        (
            "CYZ",
            [2.0, 3.0, 4.0],
            right_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [false, false, true],
            true,
            20.670_852_845_856_437,
            arr2(&[[2.0, 0.0, 0.0], [0.0, 1.5, -2.0], [0.0, 1.5, 2.0]]),
        ),
        (
            "C  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [true, false, false],
            true,
            20.670_852_845_856_437,
            arr2(&[[1.0, -1.5, 0.0], [1.0, 1.5, 0.0], [0.0, 0.0, 4.0]]),
        ),
        (
            "M  ",
            [2.0, 3.0, 4.0],
            monoclinic_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [false, false, false],
            false,
            11.089_048_279_775_305,
            arr2(&[
                [1.864_078_171_934_452_6, 0.724_715_508_953_347_2, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]),
        ),
        (
            "R  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [2.0, 3.0, 4.0],
            1.0,
            [true, true, true],
            false,
            53.704_451_047_204_61,
            arr2(&[
                [0.577_350_269_189_625_8, -1.0, 1.333_333_333_333_333_3],
                [0.577_350_269_189_625_8, 1.0, 1.333_333_333_333_333_3],
                [-1.154_700_538_379_251_7, 0.0, 1.333_333_333_333_333_3],
            ]),
        ),
        (
            "I  ",
            [2.0, 3.0, 4.0],
            right_angles,
            [1.0, 3.0, 4.0],
            0.5,
            [true, true, true],
            true,
            62.012_558_537_569_31,
            arr2(&[[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
        ),
    ];

    for (
        lattice,
        lengths,
        angles,
        adjusted_lengths,
        afact,
        dependencies,
        orthogonal,
        brillouin_zone_volume,
        direct_vectors,
    ) in cases
    {
        let basis = kmesh_bravais_basis(lattice, lengths, angles)?;
        assert_vector_values_close(basis.adjusted_lengths, adjusted_lengths);
        assert_close(basis.afact, afact);
        assert_eq!(basis.dependencies, dependencies);
        assert_eq!(basis.orthogonal, orthogonal);
        assert_close(basis.brillouin_zone_volume, brillouin_zone_volume);
        assert_matrix_close(basis.direct_vectors.view(), direct_vectors.view());
    }

    let hexagonal = kmesh_bravais_basis("H  ", [2.0, 3.0, 4.0], right_angles)?;
    assert_matrix_close(
        hexagonal.reciprocal_vectors.view(),
        arr2(&[
            [3.627_598_894_524_551_6, 1.813_799_447_262_275_8, 0.0],
            [0.0, 3.141_592_741_012_573_2, 0.0],
            [0.0, 0.0, 1.570_796_370_506_286_6],
        ])
        .view(),
    );

    let body = kmesh_bravais_basis("I  ", [2.0, 3.0, 4.0], right_angles)?;
    assert_matrix_close(
        body.reciprocal_vectors.view(),
        arr2(&[
            [0.0, 3.141_592_741_012_573_2, 3.141_592_741_012_573_2],
            [3.141_592_741_012_573_2, 0.0, 3.141_592_741_012_573_2],
            [3.141_592_741_012_573_2, 3.141_592_741_012_573_2, 0.0],
        ])
        .view(),
    );
    Ok(())
}

#[test]
fn kmesh_basis_divisions_match_feff_basdiv_reference() -> Result<(), KSpaceError> {
    let reciprocal = skew_reciprocal_basis();
    let cases = [
        ([false, false, false], 120, [6, 4, 3], 140),
        ([true, false, false], 120, [5, 5, 3], 144),
        ([false, true, false], 120, [4, 4, 4], 125),
        ([false, false, true], 120, [6, 4, 4], 175),
        ([true, true, false], 120, [4, 4, 4], 125),
        ([false, false, false], 4, [2, 1, 1], 12),
    ];

    for (dependencies, requested, divisions, mesh_points) in cases {
        assert_eq!(
            kmesh_basis_divisions(reciprocal.view(), requested, dependencies)?,
            KMeshDivisions {
                divisions,
                mesh_points,
            }
        );
    }
    Ok(())
}

#[test]
fn kmesh_tetrahedron_division_matches_feff_tetdiv_reference() -> Result<(), KSpaceError> {
    let branch_one = array![
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],
    ];
    let branch_two = array![
        [[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0]],
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]],
    ];
    let branch_three = array![
        [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1]],
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]],
    ];
    let branch_four = array![
        [[1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
        [[1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
    ];

    let cases = [
        (
            [1, 1, 1],
            arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            branch_one.clone(),
        ),
        (
            [1, 1, 1],
            arr2(&[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            branch_two.clone(),
        ),
        (
            [1, 1, 1],
            arr2(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            branch_three.clone(),
        ),
        (
            [1, 1, 1],
            arr2(&[[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            branch_four,
        ),
        (
            [2, 3, 4],
            arr2(&[[2.0, 0.5, 0.0], [0.0, 3.0, 0.25], [0.1, 0.0, 4.0]]),
            branch_three,
        ),
    ];

    for (divisions, reciprocal, expected) in cases {
        assert_eq!(
            kmesh_tetrahedron_division(divisions, reciprocal.view())?,
            expected
        );
    }
    Ok(())
}

#[test]
fn kmesh_tetrahedron_records_match_feff_tetcnt_reference() -> Result<(), KSpaceError> {
    let reciprocal = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let offsets = kmesh_tetrahedron_division([1, 1, 1], reciprocal.view())?;

    let identity =
        kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1, 2, 3, 4, 5, 6, 7, 8], 8)?;
    assert_eq!(identity.irreducible_point_count, 8);
    assert_eq!(identity.tetrahedron_count, 6);
    assert_eq!(identity.unique_tetrahedron_count, 6);
    assert_close(identity.tetrahedron_weight, 1.0 / 6.0);
    assert_eq!(
        identity.write_chunk_size,
        KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE
    );
    assert_eq!(identity.record_count, 1);
    assert_eq!(
        identity.records,
        array![
            [1_usize, 1, 2, 4, 8],
            [1, 1, 2, 6, 8],
            [1, 1, 3, 4, 8],
            [1, 1, 3, 7, 8],
            [1, 1, 5, 6, 8],
            [1, 1, 5, 7, 8],
        ]
    );

    let collapsed =
        kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1, 2, 2, 3, 2, 3, 3, 4], 4)?;
    assert_eq!(collapsed.irreducible_point_count, 4);
    assert_eq!(collapsed.tetrahedron_count, 6);
    assert_eq!(collapsed.unique_tetrahedron_count, 1);
    assert_close(collapsed.tetrahedron_weight, 1.0 / 6.0);
    assert_eq!(collapsed.record_count, 1);
    assert_eq!(collapsed.records, array![[6_usize, 1, 2, 3, 4]]);

    let stretched_offsets = kmesh_tetrahedron_division([2, 1, 1], reciprocal.view())?;
    let stretched_links = (1..=12).collect::<Vec<_>>();
    let stretched =
        kmesh_tetrahedron_records(stretched_offsets.view(), [2, 1, 1], &stretched_links, 12)?;
    assert_eq!(stretched.irreducible_point_count, 12);
    assert_eq!(stretched.tetrahedron_count, 12);
    assert_eq!(stretched.unique_tetrahedron_count, 12);
    assert_close(stretched.tetrahedron_weight, 1.0 / 12.0);
    assert_eq!(stretched.record_count, 1);
    assert_eq!(
        stretched.records,
        array![
            [1_usize, 1, 2, 4, 8],
            [1, 1, 2, 6, 8],
            [1, 1, 3, 4, 8],
            [1, 1, 3, 7, 8],
            [1, 1, 5, 6, 8],
            [1, 1, 5, 7, 8],
            [1, 5, 6, 8, 12],
            [1, 5, 6, 10, 12],
            [1, 5, 7, 8, 12],
            [1, 5, 7, 11, 12],
            [1, 5, 9, 10, 12],
            [1, 5, 9, 11, 12],
        ]
    );
    Ok(())
}

#[test]
fn reduce_kmesh_irreducible_points_matches_feff_reduz_reference() -> Result<(), KSpaceError> {
    let reciprocal = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let identity_operations = array![[[1, 0, 0], [0, 1, 0], [0, 0, 1]]];
    let identity =
        reduce_kmesh_irreducible_points([1, 1, 1], identity_operations.view(), reciprocal.view())?;
    assert_eq!(identity.shift, [1, 1, 1]);
    assert_close(identity.total_weight, 1.0);
    assert_eq!(identity.work_links, vec![1; 8]);
    assert_eq!(identity.work_symmetry, vec![1; 8]);
    assert_eq!(identity.full_links, vec![1]);
    assert_eq!(identity.full_symmetry, vec![1]);
    assert_eq!(
        identity.work_grid,
        array![
            [0_usize, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    );
    assert_array1_close(
        identity.work_weights.view(),
        array![0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125].view(),
    );
    assert_array1_close(identity.full_weights.view(), array![1.0].view());
    assert_array1_close(identity.irreducible_weights.view(), array![1.0].view());
    assert_matrix_close(
        identity.work_vectors.view(),
        arr2(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.5],
            [0.5, 1.5, 0.5],
            [0.5, 1.5, 1.5],
            [1.5, 0.5, 0.5],
            [1.5, 0.5, 1.5],
            [1.5, 1.5, 0.5],
            [1.5, 1.5, 1.5],
        ])
        .view(),
    );
    assert_matrix_close(
        identity.full_vectors.view(),
        arr2(&[[0.5, 0.5, 0.5]]).view(),
    );
    assert_matrix_close(
        identity.irreducible_fractional_vectors.view(),
        arr2(&[[0.5, 0.5, 0.5]]).view(),
    );

    let sign = reduce_kmesh_irreducible_points(
        [2, 1, 1],
        sign_flip_symmetry_operations().view(),
        reciprocal.view(),
    )?;
    assert_eq!(sign.shift, [1, 1, 1]);
    assert_close(sign.total_weight, 2.0);
    assert_eq!(sign.work_links, vec![1; 12]);
    assert_eq!(sign.work_symmetry, vec![1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1]);
    assert_eq!(sign.full_links, vec![1, 1]);
    assert_eq!(sign.full_symmetry, vec![1, 3]);
    assert_array1_close(
        sign.work_weights.view(),
        array![
            0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625,
            0.0625
        ]
        .view(),
    );
    assert_array1_close(sign.full_weights.view(), array![0.5, 0.5].view());
    assert_array1_close(sign.irreducible_weights.view(), array![1.0].view());
    assert_matrix_close(
        sign.full_vectors.view(),
        arr2(&[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]).view(),
    );
    assert_matrix_close(
        sign.irreducible_fractional_vectors.view(),
        arr2(&[[0.25, 0.5, 0.5]]).view(),
    );

    let shear_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    ];
    let skew_reciprocal = arr2(&[[2.0, 0.5, 0.0], [0.0, 3.0, 0.25], [0.1, 0.0, 4.0]]);
    let shear = reduce_kmesh_irreducible_points(
        [2, 2, 1],
        shear_operations.view(),
        skew_reciprocal.view(),
    )?;
    assert_eq!(shear.shift, [0, 0, 0]);
    assert_close(shear.total_weight, 4.0);
    assert_eq!(
        shear.work_links,
        vec![1, 1, 2, 2, 1, 1, 3, 3, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1]
    );
    assert_eq!(
        shear.work_symmetry,
        vec![1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    assert_eq!(shear.full_links, vec![1, 2, 3, 2]);
    assert_eq!(shear.full_symmetry, vec![1, 1, 1, 2]);
    assert_array1_close(
        shear.full_weights.view(),
        array![0.25, 0.25, 0.25, 0.25].view(),
    );
    assert_array1_close(
        shear.irreducible_weights.view(),
        array![0.25, 0.5, 0.25].view(),
    );
    assert_matrix_close(
        shear.full_vectors.view(),
        arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 1.5, 0.125],
            [1.0, 0.25, 0.0],
            [1.0, 1.75, 0.125],
        ])
        .view(),
    );
    assert_matrix_close(
        shear.irreducible_vectors.view(),
        arr2(&[[0.0, 0.0, 0.0], [0.0, 1.5, 0.125], [1.0, 0.25, 0.0]]).view(),
    );
    assert_matrix_close(
        shear.irreducible_fractional_vectors.view(),
        arr2(&[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]]).view(),
    );
    Ok(())
}

#[test]
fn kmesh_arbitrary_mesh_matches_feff_arbmsh_flow_reference() -> Result<(), KSpaceError> {
    let reciprocal = skew_reciprocal_basis();
    let mesh = kmesh_arbitrary_mesh(
        reciprocal.view(),
        sign_flip_symmetry_operations().view(),
        4,
        [false, false, false],
        true,
    )?;

    assert_eq!(mesh.requested_point_count, 4);
    assert_eq!(mesh.divisions, [2, 1, 1]);
    assert_eq!(mesh.work_point_count, 12);
    assert_eq!(mesh.full_point_count, 2);
    assert_eq!(mesh.irreducible_point_count, 1);
    assert_close(mesh.total_weight, 2.0);
    assert_eq!(mesh.reduction.shift, [1, 1, 1]);
    assert_eq!(mesh.reduction.work_links, vec![1; 12]);
    assert_eq!(
        mesh.reduction.work_symmetry,
        vec![1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1]
    );
    assert_eq!(mesh.reduction.full_links, vec![1, 1]);
    assert_eq!(mesh.reduction.full_symmetry, vec![1, 3]);
    assert_array1_close(
        mesh.reduction.work_weights.view(),
        array![
            0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625,
            0.0625
        ]
        .view(),
    );
    assert_array1_close(mesh.reduction.full_weights.view(), array![0.5, 0.5].view());
    assert_array1_close(
        mesh.reduction.irreducible_weights.view(),
        array![1.0].view(),
    );
    assert_matrix_close(
        mesh.reduction.irreducible_fractional_vectors.view(),
        arr2(&[[0.25, 0.5, 0.5]]).view(),
    );

    let tetrahedra = mesh
        .tetrahedra
        .as_ref()
        .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)?;
    assert_eq!(tetrahedra.irreducible_point_count, 1);
    assert_eq!(tetrahedra.tetrahedron_count, 12);
    assert_eq!(tetrahedra.unique_tetrahedron_count, 1);
    assert_close(tetrahedra.tetrahedron_weight, 1.0 / 12.0);
    assert_eq!(tetrahedra.record_count, 1);
    assert_eq!(tetrahedra.records, array![[12_usize, 1, 1, 1, 1]]);

    let mesh_without_tetrahedra = kmesh_arbitrary_mesh(
        reciprocal.view(),
        sign_flip_symmetry_operations().view(),
        4,
        [false, false, false],
        false,
    )?;
    assert!(mesh_without_tetrahedra.tetrahedra.is_none());
    Ok(())
}

#[test]
fn reduce_kmesh_common_divisor_matches_feff_divisi_reference() -> Result<(), KSpaceError> {
    let cases = [
        (
            arr2(&[[3, 6, 9], [12, 15, 18]]),
            9,
            arr2(&[[3, 6, 9], [12, 15, 18]]),
            9,
            1,
        ),
        (
            arr2(&[[6, 12, 18], [24, 30, 36]]),
            12,
            arr2(&[[3, 6, 9], [12, 15, 18]]),
            6,
            2,
        ),
        (
            arr2(&[[8, 12, 16], [20, 24, 28]]),
            8,
            arr2(&[[2, 3, 4], [5, 6, 7]]),
            2,
            4,
        ),
        (
            arr2(&[[2, 4, 6], [4, 8, 12], [6, 12, 18]]),
            3,
            arr2(&[[1, 2, 3], [2, 4, 6], [3, 6, 9]]),
            1,
            2,
        ),
    ];

    for (k_list, division, expected_k_list, expected_division, common_divisor) in cases {
        assert_eq!(
            reduce_kmesh_common_divisor(k_list.view(), division)?,
            KMeshDivisionReduction {
                k_list: expected_k_list,
                division: expected_division,
                common_divisor,
            }
        );
    }
    Ok(())
}

#[test]
fn redefine_lattice_symmetry_operations_matches_feff_sdef_reference() -> Result<(), KSpaceError> {
    let operations = sample_sdef_operations();
    let cxz_expected = array![
        [[111, 113, 112], [131, 133, 132], [121, 123, 122]],
        [[211, 213, 212], [231, 233, 232], [221, 223, 222]]
    ];
    for lattice in ["CXZ", "BO ", "bo"] {
        assert_eq!(
            redefine_lattice_symmetry_operations(operations.view(), lattice)?,
            cxz_expected
        );
    }

    let cyz_expected = array![
        [[133, 132, 131], [123, 122, 121], [113, 112, 111]],
        [[233, 232, 231], [223, 222, 221], [213, 212, 211]]
    ];
    for lattice in ["CYZ", "AO ", "ao"] {
        assert_eq!(
            redefine_lattice_symmetry_operations(operations.view(), lattice)?,
            cyz_expected
        );
    }

    assert_eq!(
        redefine_lattice_symmetry_operations(operations.view(), "P  ")?,
        operations
    );
    Ok(())
}

#[test]
fn transform_lapw_symmetry_operations_matches_feff_sdefl_reference() -> Result<(), KSpaceError> {
    let operations = sample_sdefl_operations();
    let shear_direct = arr2(&[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let shear_reciprocal = reciprocal_lattice_vectors(shear_direct.view())?;
    let transformed_expected = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 2, 0], [0, 1, 0], [0, 0, 1]]
    ];

    assert_eq!(
        transform_lapw_symmetry_operations(
            shear_direct.view(),
            shear_reciprocal.view(),
            operations.view(),
            "P  ",
            true,
        )?,
        transformed_expected
    );
    assert_eq!(
        transform_lapw_symmetry_operations(
            shear_direct.view(),
            shear_reciprocal.view(),
            operations.view(),
            "P  ",
            false,
        )?,
        operations
    );
    assert_eq!(
        transform_lapw_symmetry_operations(
            shear_direct.view(),
            shear_reciprocal.view(),
            operations.view(),
            "CXZ",
            false,
        )?,
        transformed_expected
    );

    let diagonal_direct = arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
    let diagonal_reciprocal = reciprocal_lattice_vectors(diagonal_direct.view())?;
    assert_eq!(
        transform_lapw_symmetry_operations(
            diagonal_direct.view(),
            diagonal_reciprocal.view(),
            operations.view(),
            "P  ",
            true,
        )?,
        operations
    );
    Ok(())
}

#[test]
fn point_group_operations_match_feff_reference() -> Result<(), KSpaceError> {
    let cubic = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let cubic_metric = reciprocal_metric(cubic.view())?;
    let cubic_group = point_group_operations(cubic.view(), cubic_metric.view(), 64)?;
    assert_eq!(cubic_group.len(), 48);
    assert_operation_close(
        cubic_group.operation(0),
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    )?;
    assert_operation_close(
        cubic_group.operation(8),
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    )?;
    assert_operation_close(
        cubic_group.operation(47),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )?;

    let orthorhombic = arr2(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let orthorhombic_metric = reciprocal_metric(orthorhombic.view())?;
    let orthorhombic_group =
        point_group_operations(orthorhombic.view(), orthorhombic_metric.view(), 64)?;
    assert_eq!(orthorhombic_group.len(), 8);
    assert_operation_close(
        orthorhombic_group.operation(0),
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    )?;
    assert_operation_close(
        orthorhombic_group.operation(7),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )?;
    Ok(())
}

#[test]
fn symmetry_check_matches_feff_reference() -> Result<(), KSpaceError> {
    let operations = sign_flip_symmetry_operations();
    let translations = Array2::<Real>::zeros((4, 3));
    let checked = symmetry_check(operations.view(), translations.view())?;
    assert_eq!(checked.ierr, 0);
    assert_eq!(checked.invalid_operation_index(), None);
    assert_eq!(
        checked.multiplication,
        arr2(&[[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]])
    );

    let mut bad_translations = translations;
    bad_translations[(1, 0)] = PI2 / 2.0;
    let checked = symmetry_check(operations.view(), bad_translations.view())?;
    assert_eq!(checked.ierr, 2);
    assert_eq!(checked.invalid_operation_index(), Some(1));
    assert_eq!(
        checked.multiplication,
        arr2(&[[1, 2, 3, 4], [2, 1, -1, -1], [3, -1, 1, -1], [4, -1, -1, 1]])
    );
    Ok(())
}

#[test]
fn kspace_helpers_reject_invalid_inputs() {
    assert_eq!(
        bravais_lattice(0, 'P'),
        Err(KSpaceError::InvalidSpaceGroup { space_group: 0 })
    );
    assert_eq!(
        bravais_lattice(225, 'C'),
        Err(KSpaceError::InvalidBravaisResult {
            space_group: 225,
            lattice: 'C',
        })
    );
    assert_eq!(
        define_k_path(BravaisLattice::TetragonalPrimitive, 1, BASIS),
        Err(KSpaceError::UnsupportedBravais { bravais: 8 })
    );
    assert_eq!(
        define_k_path(BravaisLattice::CubicPrimitive, 99, BASIS),
        Err(KSpaceError::InvalidKPath {
            bravais: 12,
            kpath: 99,
        })
    );

    let bad_matrix = Array2::<Real>::zeros((2, 3));
    assert_eq!(
        subtract_lattice_translation(bad_matrix.view(), [0.0; 3]),
        Err(KSpaceError::InvalidMatrixShape {
            name: "reciprocal_vectors",
            rows: 2,
            columns: 3,
        })
    );
    let matrix = Array2::<Real>::zeros((3, 3));
    assert_eq!(
        reciprocal_lattice_vectors(matrix.view()),
        Err(KSpaceError::DegenerateLatticeVolume { determinant: 0.0 })
    );
    assert_eq!(
        kmesh_bravais_basis("P  ", [0.0, 3.0, 4.0], [BRAVAIS_RIGHT_ANGLE; 3]),
        Err(KSpaceError::DegenerateLatticeVolume { determinant: 0.0 })
    );
    assert!(matches!(
        kmesh_bravais_basis(
            "P  ",
            [Real::NAN, 3.0, 4.0],
            [BRAVAIS_RIGHT_ANGLE; 3],
        ),
        Err(KSpaceError::NonFiniteValue {
            name: "lattice_lengths",
            index: 0,
            value,
        }) if value.is_nan()
    ));
    assert_eq!(
        kmesh_basis_divisions(matrix.view(), 0, [false; 3]),
        Err(KSpaceError::InvalidKMeshPointTarget { mesh_points: 0 })
    );
    assert_eq!(
        kmesh_arbitrary_mesh(
            matrix.view(),
            sign_flip_symmetry_operations().view(),
            0,
            [false; 3],
            false,
        ),
        Err(KSpaceError::InvalidKMeshPointTarget { mesh_points: 0 })
    );
    assert_eq!(
        kmesh_basis_divisions(matrix.view(), 16, [false; 3]),
        Err(KSpaceError::DegenerateReciprocalVector {
            index: 0,
            length: 0.0,
        })
    );
    assert_eq!(
        kmesh_tetrahedron_division([0, 1, 1], matrix.view()),
        Err(KSpaceError::InvalidKMeshDivision {
            component: 0,
            value: 0,
        })
    );
    assert_eq!(
        kmesh_tetrahedron_division([1, 1, 1], bad_matrix.view()),
        Err(KSpaceError::InvalidMatrixShape {
            name: "reciprocal_vectors",
            rows: 2,
            columns: 3,
        })
    );
    let offsets = Array3::<i32>::zeros((6, 4, 3));
    assert_eq!(
        kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1; 8], 0),
        Err(KSpaceError::InvalidIrreducibleKPointCount { count: 0 })
    );
    assert_eq!(
        kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1; 7], 1),
        Err(KSpaceError::InvalidWorkMeshLinkCount {
            expected: 8,
            actual: 7,
        })
    );
    assert_eq!(
        kmesh_tetrahedron_records(offsets.view(), [1, 1, 1], &[1, 2, 0, 1, 1, 1, 1, 1], 2),
        Err(KSpaceError::InvalidWorkMeshLink {
            index: 2,
            value: 0,
            irreducible_point_count: 2,
        })
    );
    let bad_offsets = Array3::<i32>::zeros((6, 4, 2));
    assert_eq!(
        kmesh_tetrahedron_records(bad_offsets.view(), [1, 1, 1], &[1; 8], 1),
        Err(KSpaceError::InvalidTetrahedronOffsetShape {
            tetrahedra: 6,
            corners: 4,
            coordinates: 2,
        })
    );
    let mut bad_offsets = Array3::<i32>::zeros((6, 4, 3));
    bad_offsets[(0, 0, 0)] = 2;
    assert_eq!(
        kmesh_tetrahedron_records(bad_offsets.view(), [1, 1, 1], &[1; 8], 1),
        Err(KSpaceError::InvalidTetrahedronOffset {
            tetrahedron: 0,
            corner: 0,
            axis: 0,
            value: 2,
        })
    );
    let bad_klist = Array2::<i32>::zeros((2, 2));
    assert_eq!(
        reduce_kmesh_common_divisor(bad_klist.view(), 12),
        Err(KSpaceError::InvalidKMeshListShape {
            rows: 2,
            columns: 2,
        })
    );
    let bad_operations = Array3::<i32>::zeros((2, 2, 3));
    assert_eq!(
        redefine_lattice_symmetry_operations(bad_operations.view(), "CXZ"),
        Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: 2,
            rows: 2,
            columns: 3,
        })
    );
    assert_eq!(
        transform_lapw_symmetry_operations(
            bad_matrix.view(),
            matrix.view(),
            sign_flip_symmetry_operations().view(),
            "P  ",
            true,
        ),
        Err(KSpaceError::InvalidMatrixShape {
            name: "direct_vectors",
            rows: 2,
            columns: 3,
        })
    );
    assert_eq!(
        reduce_kmesh_irreducible_points([1, 1, 1], bad_operations.view(), matrix.view(),),
        Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: 2,
            rows: 2,
            columns: 3,
        })
    );
    let reciprocal = skew_reciprocal_basis();
    assert_eq!(
        kmesh_arbitrary_mesh(
            reciprocal.view(),
            bad_operations.view(),
            4,
            [false; 3],
            false,
        ),
        Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: 2,
            rows: 2,
            columns: 3,
        })
    );
    let no_operations = Array3::<i32>::zeros((0, 3, 3));
    assert_eq!(
        reduce_kmesh_irreducible_points([1, 1, 1], no_operations.view(), matrix.view()),
        Err(KSpaceError::NoSymmetryOperations)
    );
    assert_eq!(
        reduce_kmesh_irreducible_points(
            [1, 1, 1],
            sign_flip_symmetry_operations().view(),
            bad_matrix.view(),
        ),
        Err(KSpaceError::InvalidMatrixShape {
            name: "reciprocal_vectors",
            rows: 2,
            columns: 3,
        })
    );
    assert!(matches!(
        reduce_to_lattice_cell(matrix.view(), matrix.view(), [Real::NAN, 0.0, 0.0]),
        Err(KSpaceError::NonFiniteValue {
            name: "vector",
            index: 0,
            value,
        }) if value.is_nan()
    ));
    assert_eq!(
        point_group_operations(matrix.view(), matrix.view(), 0),
        Err(KSpaceError::InvalidPointGroupCapacity { capacity: 0 })
    );
    let identity = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_eq!(
        point_group_operations(identity.view(), matrix.view(), 4),
        Err(KSpaceError::DegenerateMetricDiagonal {
            index: 0,
            value: 0.0,
        })
    );

    let no_translations = Array2::<Real>::zeros((0, 3));
    assert_eq!(
        symmetry_check(no_operations.view(), no_translations.view()),
        Err(KSpaceError::NoSymmetryOperations)
    );
    let translations = Array2::<Real>::zeros((2, 3));
    assert_eq!(
        symmetry_check(bad_operations.view(), translations.view()),
        Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: 2,
            rows: 2,
            columns: 3,
        })
    );
    let operations = sign_flip_symmetry_operations();
    let bad_translations = Array2::<Real>::zeros((3, 3));
    assert_eq!(
        symmetry_check(operations.view(), bad_translations.view()),
        Err(KSpaceError::InvalidSymmetryTranslationShape {
            operations: 4,
            rows: 3,
            columns: 3,
        })
    );
    let rotating_operation = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    ];
    let translations = Array2::<Real>::zeros((2, 3));
    assert_eq!(
        symmetry_check(rotating_operation.view(), translations.view()),
        Err(KSpaceError::SymmetryProductMissing { left: 2, right: 2 })
    );
}

fn dot_for_test(left: Vector3, right: Vector3) -> Real {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn fill_expected_straa_direct_terms(
    terms: &mut Array3<Complex>,
    direct_term: usize,
    q_pair: usize,
    delta: Vector3,
    eta: Real,
) {
    let scaled = [PI2 * delta[0], PI2 * delta[1], PI2 * delta[2]];
    let rsquad = dot_for_test(scaled, scaled);
    let q1 = -0.5 * (eta / (PI2 / 2.0)).sqrt();
    let q3 = (-eta * rsquad / 4.0).exp();
    let l0_factor = q1 * q3;
    let l1_factor = q1 * (-eta / 2.0) * q3;

    terms[(0, direct_term, q_pair)] = Complex::new(l0_factor * 2.0, 0.0);
    terms[(1, direct_term, q_pair)] = Complex::new(l1_factor * 5.0 * scaled[1], 0.0);
    terms[(2, direct_term, q_pair)] = Complex::new(l1_factor * 3.0 * scaled[2], 0.0);
    terms[(3, direct_term, q_pair)] = Complex::new(l1_factor * 5.0 * scaled[0], 0.0);
}

fn assert_expected_straa_radial_terms(
    radial_terms: ArrayView4<'_, Real>,
    direct_term: usize,
    q_pair: usize,
    expected: [Real; 4],
) {
    assert_close(radial_terms[(0, 0, direct_term, q_pair)], expected[0]);
    assert_close(radial_terms[(1, 0, direct_term, q_pair)], expected[1]);
    assert_close(radial_terms[(0, 1, direct_term, q_pair)], expected[2]);
    assert_close(radial_terms[(1, 1, direct_term, q_pair)], expected[3]);
}

fn sample_strcc_tables() -> (Array3<Complex>, Array4<Real>) {
    let mut base_direct_terms = Array3::<Complex>::zeros((4, 2, 2));
    for q_pair in 0..2 {
        for direct_term in 0..2 {
            for mml in 0..4 {
                base_direct_terms[(mml, direct_term, q_pair)] = Complex::new(
                    0.1 * (mml + 1) as Real + 0.2 * direct_term as Real - 0.05 * q_pair as Real,
                    0.03 * mml as Real + 0.04 * q_pair as Real,
                );
            }
        }
    }

    let mut radial_terms = Array4::<Real>::zeros((2, 2, 2, 2));
    radial_terms[(0, 0, 0, 0)] = 0.5;
    radial_terms[(1, 0, 0, 0)] = 0.1;
    radial_terms[(0, 1, 0, 0)] = 0.25;
    radial_terms[(1, 1, 0, 0)] = -0.2;
    radial_terms[(0, 0, 0, 1)] = -0.3;
    radial_terms[(1, 0, 0, 1)] = 0.05;
    radial_terms[(0, 1, 0, 1)] = 0.4;
    radial_terms[(1, 1, 0, 1)] = 0.15;
    radial_terms[(0, 0, 1, 1)] = 0.2;
    radial_terms[(1, 0, 1, 1)] = -0.12;
    radial_terms[(0, 1, 1, 1)] = -0.1;
    radial_terms[(1, 1, 1, 1)] = 0.07;
    radial_terms[(0, 0, 1, 0)] = 9.0;
    radial_terms[(1, 0, 1, 0)] = 9.0;
    radial_terms[(0, 1, 1, 0)] = 9.0;
    radial_terms[(1, 1, 1, 0)] = 9.0;

    (base_direct_terms, radial_terms)
}

fn reciprocal_basis_for_test() -> [Vector3; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
