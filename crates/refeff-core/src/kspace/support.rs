use super::*;

pub(super) fn row_norm(matrix: ArrayView2<'_, Real>, row: usize) -> Real {
    (matrix[(row, 0)].powi(2) + matrix[(row, 1)].powi(2) + matrix[(row, 2)].powi(2)).sqrt()
}

pub(super) fn mesh_division_from_real(component: usize, value: Real) -> Result<usize, KSpaceError> {
    if !value.is_finite() || value < 0.0 || value >= usize::MAX as Real {
        return Err(KSpaceError::KMeshDivisionOverflow { component, value });
    }
    Ok((value as usize).max(1))
}

pub(super) fn kmesh_point_count(divisions: [usize; 3]) -> Result<usize, KSpaceError> {
    divisions
        .into_iter()
        .map(|division| division.checked_add(1))
        .try_fold(1usize, |product, value| {
            let value = value.ok_or(KSpaceError::KMeshPointCountOverflow)?;
            product
                .checked_mul(value)
                .ok_or(KSpaceError::KMeshPointCountOverflow)
        })
}

pub(super) fn kmesh_cell_count(divisions: [usize; 3]) -> Result<usize, KSpaceError> {
    divisions.into_iter().try_fold(1usize, |product, division| {
        product
            .checked_mul(division)
            .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)
    })
}

pub(super) fn work_mesh_coordinate(
    point: usize,
    row_stride: usize,
    plane_stride: usize,
    axis: usize,
) -> usize {
    let first = point / plane_stride;
    let second = (point - first * plane_stride) / row_stride;
    let third = point - first * plane_stride - second * row_stride;
    match axis {
        0 => first,
        1 => second,
        _ => third,
    }
}

pub(super) fn kmesh_submesh_shift(operations: ArrayView3<'_, i32>) -> [usize; 3] {
    for operation in 0..operations.shape()[0] {
        for vertex in 1usize..=8 {
            let first = (vertex - 1) / 4;
            let second = (vertex - first * 4 - 1) / 2;
            let third = vertex - first * 4 - second * 2 - 1;
            let doubled = [
                2 * (first as i128) + 1,
                2 * (second as i128) + 1,
                2 * (third as i128) + 1,
            ];
            if (0..3)
                .any(|axis| (operation_row_dot(operations, operation, axis, doubled) - 1) % 2 != 0)
            {
                return [0, 0, 0];
            }
        }
    }
    [1, 1, 1]
}

pub(super) fn mapped_work_mesh_index(
    operations: ArrayView3<'_, i32>,
    operation: usize,
    coordinates: [usize; 3],
    divisions: [usize; 3],
    shift: [usize; 3],
    row_stride: usize,
    plane_stride: usize,
) -> Result<usize, KSpaceError> {
    let doubled = [
        2 * i128::from(coordinates[0] as i64) + i128::from(shift[0] as i64),
        2 * i128::from(coordinates[1] as i64) + i128::from(shift[1] as i64),
        2 * i128::from(coordinates[2] as i64) + i128::from(shift[2] as i64),
    ];
    let mut mapped = [0usize; 3];
    for axis in 0..3 {
        let modulus = 2 * i128::from(divisions[axis] as i64);
        let mut value = operation_row_dot(operations, operation, axis, doubled) % modulus;
        if value < 0 {
            value += modulus;
        }
        value = (value - i128::from(shift[axis] as i64)) / 2;
        mapped[axis] = usize::try_from(value).map_err(|_| KSpaceError::KMeshPointCountOverflow)?;
    }
    mapped[0]
        .checked_mul(plane_stride)
        .and_then(|first| {
            mapped[1]
                .checked_mul(row_stride)
                .and_then(|second| first.checked_add(second))
        })
        .and_then(|value| value.checked_add(mapped[2]))
        .ok_or(KSpaceError::KMeshPointCountOverflow)
}

pub(super) fn operation_row_dot(
    operations: ArrayView3<'_, i32>,
    operation: usize,
    row: usize,
    vector: [i128; 3],
) -> i128 {
    (0..3)
        .map(|column| i128::from(operations[(operation, row, column)]) * vector[column])
        .sum()
}

pub(super) fn boundary_weight(coordinates: [usize; 3], divisions: [usize; 3]) -> Real {
    (0..3).fold(1.0, |weight, axis| {
        if coordinates[axis].is_multiple_of(divisions[axis]) {
            weight / 2.0
        } else {
            weight
        }
    })
}

pub(super) fn checked_full_work_link(
    link: usize,
    index: usize,
    full_point_count: usize,
) -> Result<usize, KSpaceError> {
    if link == 0 || link > full_point_count {
        Err(KSpaceError::InvalidFullMeshLink {
            index,
            value: link,
            full_point_count,
        })
    } else {
        Ok(link)
    }
}

pub(super) fn fractional_coordinate(coordinate: usize, shift: usize, division: usize) -> Real {
    (coordinate as Real + (shift as Real) / 2.0) / (division as Real)
}

pub(super) fn kmesh_vector(
    reciprocal_vectors: ArrayView2<'_, Real>,
    fractional: Vector3,
) -> Vector3 {
    let mut vector = [0.0; 3];
    for axis in 0..3 {
        vector[axis] = reciprocal_vectors[(0, axis)] * fractional[0]
            + reciprocal_vectors[(1, axis)] * fractional[1]
            + reciprocal_vectors[(2, axis)] * fractional[2];
    }
    vector
}

pub(super) fn validate_full_mesh_link_symmetry(
    full_links: &[usize],
    full_symmetry: &[usize],
) -> Result<(), KSpaceError> {
    for first in 0..full_links.len() {
        for second in first + 1..full_links.len() {
            if full_links[first] == full_links[second]
                && full_symmetry[first] == full_symmetry[second]
            {
                return Err(KSpaceError::DuplicateFullMeshLinkSymmetry {
                    first,
                    second,
                    link: full_links[first],
                    symmetry: full_symmetry[first],
                });
            }
        }
    }
    Ok(())
}

pub(super) fn validate_kmesh_divisions(divisions: [usize; 3]) -> Result<(), KSpaceError> {
    for (component, value) in divisions.into_iter().enumerate() {
        if value == 0 {
            return Err(KSpaceError::InvalidKMeshDivision { component, value });
        }
    }
    Ok(())
}

pub(super) fn validate_tetrahedron_offsets(
    tetrahedron_offsets: ArrayView3<'_, i32>,
) -> Result<(), KSpaceError> {
    let shape = tetrahedron_offsets.shape();
    if shape[0] != 6 || shape[1] != 4 || shape[2] != 3 {
        return Err(KSpaceError::InvalidTetrahedronOffsetShape {
            tetrahedra: shape[0],
            corners: shape[1],
            coordinates: shape[2],
        });
    }
    for ((tetrahedron, corner, axis), &value) in tetrahedron_offsets.indexed_iter() {
        if value != 0 && value != 1 {
            return Err(KSpaceError::InvalidTetrahedronOffset {
                tetrahedron,
                corner,
                axis,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_work_mesh_links(
    point_links: &[usize],
    expected_point_count: usize,
    irreducible_point_count: usize,
) -> Result<(), KSpaceError> {
    if irreducible_point_count == 0 {
        return Err(KSpaceError::InvalidIrreducibleKPointCount {
            count: irreducible_point_count,
        });
    }
    if point_links.len() != expected_point_count {
        return Err(KSpaceError::InvalidWorkMeshLinkCount {
            expected: expected_point_count,
            actual: point_links.len(),
        });
    }
    for (index, &value) in point_links.iter().enumerate() {
        if value == 0 || value > irreducible_point_count {
            return Err(KSpaceError::InvalidWorkMeshLink {
                index,
                value,
                irreducible_point_count,
            });
        }
    }
    Ok(())
}

pub(super) fn tetrahedron_corner_index(
    tetrahedron_offsets: ArrayView3<'_, i32>,
    tetrahedron: usize,
    corner: usize,
) -> Result<[usize; 3], KSpaceError> {
    let mut offset = [0usize; 3];
    for axis in 0..3 {
        offset[axis] = match tetrahedron_offsets[(tetrahedron, corner, axis)] {
            0 => 0,
            1 => 1,
            value => {
                return Err(KSpaceError::InvalidTetrahedronOffset {
                    tetrahedron,
                    corner,
                    axis,
                    value,
                });
            }
        };
    }
    Ok(offset)
}

pub(super) fn tetrahedron_work_mesh_offsets(
    tetrahedron_offsets: ArrayView3<'_, i32>,
    row_stride: usize,
    plane_stride: usize,
) -> Result<[[usize; 4]; 6], KSpaceError> {
    let mut offsets = [[0usize; 4]; 6];
    for (tetrahedron, row_offsets) in offsets.iter_mut().enumerate() {
        for (corner, corner_offset) in row_offsets.iter_mut().enumerate() {
            *corner_offset = offset_work_mesh_index(
                0,
                row_stride,
                plane_stride,
                tetrahedron_corner_index(tetrahedron_offsets, tetrahedron, corner)?,
            )?;
        }
    }
    Ok(offsets)
}

pub(super) fn offset_work_mesh_index(
    base: usize,
    row_stride: usize,
    plane_stride: usize,
    offset: [usize; 3],
) -> Result<usize, KSpaceError> {
    base.checked_add(
        offset[0]
            .checked_mul(plane_stride)
            .and_then(|first| {
                offset[1]
                    .checked_mul(row_stride)
                    .and_then(|second| first.checked_add(second))
            })
            .and_then(|value| value.checked_add(offset[2]))
            .ok_or(KSpaceError::KMeshPointCountOverflow)?,
    )
    .ok_or(KSpaceError::KMeshPointCountOverflow)
}

pub(super) fn tetrahedron_vertex_order(shortest_diagonal: usize) -> [usize; 8] {
    match shortest_diagonal {
        1 => [2, 1, 4, 3, 6, 5, 8, 7],
        2 => [3, 4, 1, 2, 7, 8, 5, 6],
        3 => [5, 6, 7, 8, 1, 2, 3, 4],
        _ => [1, 2, 3, 4, 5, 6, 7, 8],
    }
}

pub(super) fn tetrahedron_vertices(vertex_order: [usize; 8]) -> [[usize; 4]; 6] {
    [
        [
            vertex_order[0],
            vertex_order[1],
            vertex_order[3],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[3],
            vertex_order[2],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[2],
            vertex_order[6],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[6],
            vertex_order[4],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[4],
            vertex_order[5],
            vertex_order[7],
        ],
        [
            vertex_order[0],
            vertex_order[5],
            vertex_order[1],
            vertex_order[7],
        ],
    ]
}

pub(super) fn vertex_coordinate(vertex: usize, axis: usize) -> i32 {
    let first = (vertex - 1) / 4;
    let second = (vertex - first * 4 - 1) / 2;
    let third = vertex - first * 4 - second * 2 - 1;
    match axis {
        0 => first as i32,
        1 => second as i32,
        _ => third as i32,
    }
}

pub(super) fn lattice_code3(lattice: &str) -> [u8; 3] {
    let mut code = [b' '; 3];
    for (index, byte) in lattice.bytes().take(3).enumerate() {
        code[index] = byte.to_ascii_uppercase();
    }
    code
}

pub(super) fn has_non_right_angle(angles: Vector3) -> bool {
    angles.into_iter().any(|angle| !is_feff_right_angle(angle))
}

pub(super) fn is_feff_right_angle(angle: Real) -> bool {
    (angle - BRAVAIS_RIGHT_ANGLE).abs() <= BRAVAIS_ANGLE_EPSILON
}

pub(super) fn reciprocal_coordinates(
    reciprocal_vectors: ArrayView2<'_, Real>,
    vector: Vector3,
) -> Vector3 {
    let mut reduced = [0.0; 3];
    for row in 0..3 {
        for col in 0..3 {
            reduced[row] += reciprocal_vectors[(row, col)] * vector[col];
        }
        reduced[row] /= PI2;
    }
    reduced
}

pub(super) fn shift_reduced_coordinates(
    reduced: &mut Vector3,
    wrap_negative: bool,
) -> Result<usize, KSpaceError> {
    let mut translation_count = 0usize;
    for (component, value) in reduced.iter_mut().enumerate() {
        let shift = nearest_integer(*value, component)?;
        let magnitude = usize::try_from(shift.unsigned_abs())
            .map_err(|_| KSpaceError::TranslationCountOverflow)?;
        translation_count = translation_count
            .checked_add(magnitude)
            .ok_or(KSpaceError::TranslationCountOverflow)?;
        *value -= shift as Real;
        if wrap_negative {
            if *value < 0.0 && *value > REDUCE_NEGATIVE_EPSILON {
                *value = 0.0;
            } else if *value < 0.0 {
                *value += 1.0;
            }
        }
    }
    Ok(translation_count)
}

pub(super) fn nearest_integer(value: Real, component: usize) -> Result<i64, KSpaceError> {
    let rounded = value.round();
    if rounded < i64::MIN as Real || rounded > i64::MAX as Real {
        return Err(KSpaceError::TranslationOverflow { component, value });
    }
    Ok(rounded as i64)
}

pub(super) fn validate_basis(basis: [Vector3; 3]) -> Result<(), KSpaceError> {
    for (vector_index, vector) in basis.into_iter().enumerate() {
        validate_vector_component("reciprocal_basis", vector_index * 3, vector[0])?;
        validate_vector_component("reciprocal_basis", vector_index * 3 + 1, vector[1])?;
        validate_vector_component("reciprocal_basis", vector_index * 3 + 2, vector[2])?;
    }
    Ok(())
}

pub(super) fn validate_vector(name: &'static str, vector: Vector3) -> Result<(), KSpaceError> {
    for (index, value) in vector.into_iter().enumerate() {
        validate_vector_component(name, index, value)?;
    }
    Ok(())
}

pub(super) fn validate_matrix(
    name: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), KSpaceError> {
    if matrix.nrows() != 3 || matrix.ncols() != 3 {
        return Err(KSpaceError::InvalidMatrixShape {
            name,
            rows: matrix.nrows(),
            columns: matrix.ncols(),
        });
    }
    for ((row, column), &value) in matrix.indexed_iter() {
        validate_vector_component(name, row * 3 + column, value)?;
    }
    Ok(())
}

pub(super) fn validate_symmetry_inputs(
    operations: ArrayView3<'_, i32>,
    translations: ArrayView2<'_, Real>,
) -> Result<(), KSpaceError> {
    validate_symmetry_operation_shape(operations)?;
    let operation_count = operations.shape()[0];
    if operation_count == 0 {
        return Err(KSpaceError::NoSymmetryOperations);
    }
    if translations.nrows() != operation_count || translations.ncols() != 3 {
        return Err(KSpaceError::InvalidSymmetryTranslationShape {
            operations: operation_count,
            rows: translations.nrows(),
            columns: translations.ncols(),
        });
    }
    for ((row, column), &value) in translations.indexed_iter() {
        validate_vector_component("translations", row * 3 + column, value)?;
    }
    Ok(())
}

pub(super) fn validate_symmetry_operation_shape(
    operations: ArrayView3<'_, i32>,
) -> Result<(), KSpaceError> {
    let shape = operations.shape();
    if shape[1] != 3 || shape[2] != 3 {
        return Err(KSpaceError::InvalidSymmetryOperationShape {
            operations: shape[0],
            rows: shape[1],
            columns: shape[2],
        });
    }
    Ok(())
}

pub(super) fn validate_vector_component(
    name: &'static str,
    index: usize,
    value: Real,
) -> Result<(), KSpaceError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(KSpaceError::NonFiniteValue { name, index, value })
    }
}
