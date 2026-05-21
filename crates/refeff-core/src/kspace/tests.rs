use ndarray::{ArrayView1, arr2, array};

use super::*;

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

fn sample_sdef_operations() -> Array3<i32> {
    array![
        [[111, 112, 113], [121, 122, 123], [131, 132, 133]],
        [[211, 212, 213], [221, 222, 223], [231, 232, 233]]
    ]
}

fn sample_sdefl_operations() -> Array3<i32> {
    array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ]
}

fn sign_flip_symmetry_operations() -> Array3<i32> {
    array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
}

fn skew_reciprocal_basis() -> RealMat {
    arr2(&[
        [
            3.138_666_777_779_998_4,
            -7.979_661_299_440_674e-2,
            -1.489_536_775_895_592_7e-1,
        ],
        [
            -3.404_655_487_761_354_4e-1,
            2.138_549_228_250_1,
            -1.968_316_453_862_033e-1,
        ],
        [
            1.994_915_324_860_168_6e-1,
            -2.713_084_841_809_829e-1,
            1.587_952_598_588_694,
        ],
    ])
}

fn assert_vector_close(actual: Option<Vector3>, expected: Vector3) -> Result<(), KSpaceError> {
    let Some(actual) = actual else {
        return Err(KSpaceError::KPathDefinitionIncomplete {
            bravais: 0,
            kpath: 0,
            available: 0,
            required: 1,
        });
    };
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected);
    }
    Ok(())
}

fn assert_vector_values_close(actual: Vector3, expected: Vector3) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected);
    }
}

fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_close(actual, expected[(row, column)]);
    }
}

fn assert_array1_close(actual: ArrayView1<'_, Real>, expected: ArrayView1<'_, Real>) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(*actual, *expected);
    }
}

fn assert_operation_close(
    actual: Option<[[Real; 3]; 3]>,
    expected: [[Real; 3]; 3],
) -> Result<(), KSpaceError> {
    let Some(actual) = actual else {
        return Err(KSpaceError::NoPointGroupOperations);
    };
    for row in 0..3 {
        for col in 0..3 {
            assert_close(actual[row][col], expected[row][col]);
        }
    }
    Ok(())
}

fn assert_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:?}, got {actual:?}"
    );
}
