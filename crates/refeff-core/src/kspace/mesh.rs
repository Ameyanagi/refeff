use super::support::*;
use super::*;

/// Port of FEFF `KSPACE/kmesh.f90` `basdiv`.
///
/// `reciprocal_vectors` stores reciprocal-lattice vectors as rows, matching
/// FEFF `gbas(i,*)`. `dependencies` is FEFF `iarb`: `[0]` couples divisions
/// 1 and 2, `[1]` couples 1 and 3, and `[2]` couples 2 and 3. The returned
/// point count is FEFF's adjusted `(n1 + 1) * (n2 + 1) * (n3 + 1)` value.
pub fn kmesh_basis_divisions(
    reciprocal_vectors: ArrayView2<'_, Real>,
    requested_mesh_points: usize,
    dependencies: [bool; 3],
) -> Result<KMeshDivisions, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    if requested_mesh_points == 0 {
        return Err(KSpaceError::InvalidKMeshPointTarget {
            mesh_points: requested_mesh_points,
        });
    }

    let lengths = [
        row_norm(reciprocal_vectors, 0),
        row_norm(reciprocal_vectors, 1),
        row_norm(reciprocal_vectors, 2),
    ];
    for (index, length) in lengths.into_iter().enumerate() {
        if length <= LATTICE_VOLUME_EPSILON {
            return Err(KSpaceError::DegenerateReciprocalVector { index, length });
        }
    }

    let length_product = lengths[0] * lengths[1] * lengths[2];
    if !length_product.is_finite() || length_product <= LATTICE_VOLUME_EPSILON {
        return Err(KSpaceError::InvalidKMeshScale {
            mesh_points: requested_mesh_points,
            length_product,
        });
    }

    let scale = ((requested_mesh_points as Real) / length_product).powf(1.0 / 3.0);
    if !scale.is_finite() {
        return Err(KSpaceError::InvalidKMeshScale {
            mesh_points: requested_mesh_points,
            length_product,
        });
    }
    let rn = [lengths[0] * scale, lengths[1] * scale, lengths[2] * scale];

    let divisions = if dependencies[0] && dependencies[1] {
        let value = (rn[0] * rn[1] * rn[2]).powf(1.0 / 3.0) + BASDIV_OFFSET;
        let division = mesh_division_from_real(0, value)?;
        [division; 3]
    } else if dependencies[0] {
        let division = mesh_division_from_real(0, (rn[0] * rn[1]).sqrt() + BASDIV_OFFSET)?;
        [division, division, mesh_division_from_real(2, rn[2])?]
    } else if dependencies[1] {
        let division = mesh_division_from_real(0, (rn[0] * rn[2]).sqrt() + BASDIV_OFFSET)?;
        [division, mesh_division_from_real(1, rn[1])?, division]
    } else if dependencies[2] {
        let division = mesh_division_from_real(1, (rn[1] * rn[2]).sqrt() + BASDIV_OFFSET)?;
        [mesh_division_from_real(0, rn[0])?, division, division]
    } else {
        [
            mesh_division_from_real(0, rn[0] + BASDIV_OFFSET)?,
            mesh_division_from_real(1, rn[1] + BASDIV_OFFSET)?,
            mesh_division_from_real(2, rn[2] + BASDIV_OFFSET)?,
        ]
    };

    Ok(KMeshDivisions {
        divisions,
        mesh_points: kmesh_point_count(divisions)?,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `TETDIV`.
///
/// Splits one reciprocal-lattice parallelepiped cell into six tetrahedra using
/// FEFF's shortest body-diagonal rule. `divisions` is FEFF `n`, and
/// `reciprocal_vectors` stores `gbas(i,*)` row vectors. The returned array has
/// shape `(tetrahedron, corner, xyz)` and stores the zero/one corner offsets
/// that FEFF returns as `TET0(:, corner, tetrahedron)`.
pub fn kmesh_tetrahedron_division(
    divisions: [usize; 3],
    reciprocal_vectors: ArrayView2<'_, Real>,
) -> Result<Array3<i32>, KSpaceError> {
    validate_kmesh_divisions(divisions)?;
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;

    let mut points = [[0.0; 3]; 8];
    for first in 0..=1 {
        for second in 0..=1 {
            for third in 0..=1 {
                let index = 4 * first + 2 * second + third;
                for axis in 0..3 {
                    points[index][axis] = reciprocal_vectors[(0, axis)] * (first as Real)
                        / (divisions[0] as Real)
                        + reciprocal_vectors[(1, axis)] * (second as Real) / (divisions[1] as Real)
                        + reciprocal_vectors[(2, axis)] * (third as Real) / (divisions[2] as Real);
                }
            }
        }
    }

    let mut diagonal_lengths = [0.0; 4];
    for diagonal in 0..4 {
        diagonal_lengths[diagonal] = (0..3)
            .map(|axis| {
                let delta = points[diagonal][axis] - points[7 - diagonal][axis];
                delta * delta
            })
            .sum();
    }
    let shortest = (1..4).fold(0, |best, diagonal| {
        if diagonal_lengths[diagonal] < diagonal_lengths[best] {
            diagonal
        } else {
            best
        }
    });

    let vertex_order = tetrahedron_vertex_order(shortest);
    let tetrahedra = tetrahedron_vertices(vertex_order);
    Ok(Array3::from_shape_fn(
        (6, 4, 3),
        |(tetrahedron, corner, axis)| vertex_coordinate(tetrahedra[tetrahedron][corner], axis),
    ))
}

/// Port of FEFF `KSPACE/kmesh.f90` `TETCNT`.
///
/// `tetrahedron_offsets` is the `(6, 4, 3)` table returned by
/// [`kmesh_tetrahedron_division`]. `point_links` is FEFF `linkw` after
/// irreducible-zone reduction: one 1-based irreducible k-point id for each
/// work-mesh point in `(n1 + 1) * (n2 + 1) * (n3 + 1)` order. The returned
/// `records` table mirrors FEFF `ITTFL`, with each row storing multiplicity
/// followed by four sorted 1-based tetrahedron corners.
pub fn kmesh_tetrahedron_records(
    tetrahedron_offsets: ArrayView3<'_, i32>,
    divisions: [usize; 3],
    point_links: &[usize],
    irreducible_point_count: usize,
) -> Result<KMeshTetrahedronRecords, KSpaceError> {
    validate_kmesh_divisions(divisions)?;
    validate_tetrahedron_offsets(tetrahedron_offsets)?;
    validate_work_mesh_links(
        point_links,
        kmesh_point_count(divisions)?,
        irreducible_point_count,
    )?;

    let cell_count = kmesh_cell_count(divisions)?;
    let tetrahedron_count = cell_count
        .checked_mul(6)
        .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)?;
    let row_stride = divisions[2]
        .checked_add(1)
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let plane_stride = divisions[1]
        .checked_add(1)
        .and_then(|value| value.checked_mul(row_stride))
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let corner_offsets =
        tetrahedron_work_mesh_offsets(tetrahedron_offsets, row_stride, plane_stride)?;

    let mut tetrahedra = Vec::with_capacity(tetrahedron_count);
    for first in 0..divisions[0] {
        for second in 0..divisions[1] {
            for third in 0..divisions[2] {
                let base = first
                    .checked_mul(plane_stride)
                    .and_then(|value| {
                        second
                            .checked_mul(row_stride)
                            .and_then(|offset| value.checked_add(offset))
                    })
                    .and_then(|value| value.checked_add(third))
                    .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                for tetrahedron_offsets in &corner_offsets {
                    let mut corners = [0usize; 4];
                    for (corner, &corner_offset) in tetrahedron_offsets.iter().enumerate() {
                        let point_index = base
                            .checked_add(corner_offset)
                            .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                        corners[corner] = *point_links.get(point_index).ok_or(
                            KSpaceError::InvalidWorkMeshLinkCount {
                                expected: point_index + 1,
                                actual: point_links.len(),
                            },
                        )?;
                    }
                    corners.sort_unstable();
                    tetrahedra.push(corners);
                }
            }
        }
    }
    tetrahedra.sort_unstable();

    let mut rows = Vec::new();
    let mut index = 0;
    while index < tetrahedra.len() {
        let current = tetrahedra[index];
        let mut multiplicity = 1usize;
        index += 1;
        while index < tetrahedra.len() && tetrahedra[index] == current {
            multiplicity = multiplicity
                .checked_add(1)
                .ok_or(KSpaceError::KMeshTetrahedronCountOverflow)?;
            index += 1;
        }
        rows.extend([multiplicity, current[0], current[1], current[2], current[3]]);
    }

    let unique_tetrahedron_count = rows.len() / 5;
    let record_count = unique_tetrahedron_count.div_ceil(KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE);
    let records = Array2::from_shape_vec((unique_tetrahedron_count, 5), rows)
        .map_err(|_| KSpaceError::KMeshTetrahedronCountOverflow)?;

    Ok(KMeshTetrahedronRecords {
        irreducible_point_count,
        tetrahedron_count,
        unique_tetrahedron_count,
        tetrahedron_weight: 1.0 / (tetrahedron_count as Real),
        write_chunk_size: KSPACE_TETRAHEDRON_WRITE_CHUNK_SIZE,
        record_count,
        records,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `ARBMSH`.
///
/// FEFF `ARBMSH` composes `basdiv`, `REDUZ`, `TETDIV`, and `TETCNT`: it chooses
/// reciprocal-lattice divisions for the requested mesh size, reduces the work
/// mesh to irreducible k-points, and optionally creates FEFF tetrahedron
/// integration records. The returned fields preserve FEFF's `nka`, `nkw`,
/// `nkf`, `nki`, and one-based link semantics.
pub fn kmesh_arbitrary_mesh(
    reciprocal_vectors: ArrayView2<'_, Real>,
    operations: ArrayView3<'_, i32>,
    requested_mesh_points: usize,
    dependencies: [bool; 3],
    include_tetrahedra: bool,
) -> Result<KMeshArbitraryMesh, KSpaceError> {
    let division_result =
        kmesh_basis_divisions(reciprocal_vectors, requested_mesh_points, dependencies)?;
    let divisions = division_result.divisions;
    let full_point_count = kmesh_cell_count(divisions)?;
    let reduction = reduce_kmesh_irreducible_points(divisions, operations, reciprocal_vectors)?;
    let irreducible_point_count = reduction.irreducible_weights.len();
    let total_weight = reduction.total_weight;

    let tetrahedra = if include_tetrahedra {
        let tetrahedron_offsets = kmesh_tetrahedron_division(divisions, reciprocal_vectors)?;
        Some(kmesh_tetrahedron_records(
            tetrahedron_offsets.view(),
            divisions,
            &reduction.work_links,
            irreducible_point_count,
        )?)
    } else {
        None
    };

    Ok(KMeshArbitraryMesh {
        requested_point_count: requested_mesh_points,
        divisions,
        work_point_count: division_result.mesh_points,
        full_point_count,
        irreducible_point_count,
        total_weight,
        reduction,
        tetrahedra,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `REDUZ`.
///
/// Builds FEFF's work, full, and irreducible k-mesh arrays from integer
/// symmetry operations in reciprocal-lattice coordinates. Link and symmetry
/// outputs keep FEFF's 1-based numbering so they can be fed directly into
/// [`kmesh_tetrahedron_records`] and FEFF-compatible text/binary handoff code.
pub fn reduce_kmesh_irreducible_points(
    divisions: [usize; 3],
    operations: ArrayView3<'_, i32>,
    reciprocal_vectors: ArrayView2<'_, Real>,
) -> Result<KMeshReduction, KSpaceError> {
    validate_kmesh_divisions(divisions)?;
    validate_symmetry_operation_shape(operations)?;
    if operations.shape()[0] == 0 {
        return Err(KSpaceError::NoSymmetryOperations);
    }
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;

    let work_point_count = kmesh_point_count(divisions)?;
    let full_point_count = kmesh_cell_count(divisions)?;
    let row_stride = divisions[2]
        .checked_add(1)
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let plane_stride = divisions[1]
        .checked_add(1)
        .and_then(|value| value.checked_mul(row_stride))
        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
    let work_grid = Array2::from_shape_fn((work_point_count, 3), |(point, axis)| {
        work_mesh_coordinate(point, row_stride, plane_stride, axis)
    });
    let shift = kmesh_submesh_shift(operations);

    let mut work_links = (1..=work_point_count).collect::<Vec<_>>();
    let mut work_symmetry = vec![0usize; work_point_count];
    let mut full_work_links = vec![0usize; work_point_count];
    let mut full_link_count = 0usize;

    for operation in 0..operations.shape()[0] {
        for point in 0..work_point_count {
            let mapped = mapped_work_mesh_index(
                operations,
                operation,
                [
                    work_grid[(point, 0)],
                    work_grid[(point, 1)],
                    work_grid[(point, 2)],
                ],
                divisions,
                shift,
                row_stride,
                plane_stride,
            )?;
            let mapped_link = mapped + 1;
            if mapped_link < work_links[point]
                || (mapped_link == work_links[point] && work_symmetry[point] == 0)
            {
                work_symmetry[point] = operation + 1;
                work_links[point] = work_links[point].min(mapped_link);
            }

            if operation == 0 {
                if mapped == point {
                    full_link_count = full_link_count
                        .checked_add(1)
                        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                    full_work_links[point] = full_link_count;
                } else if mapped < point {
                    full_work_links[point] = full_work_links[mapped].min(mapped_link);
                } else {
                    full_link_count = full_link_count
                        .checked_add(1)
                        .ok_or(KSpaceError::KMeshPointCountOverflow)?;
                    full_work_links[point] = full_link_count;
                }
            }
        }
    }

    let irreducible_point_count = work_links
        .iter()
        .enumerate()
        .filter(|(point, link)| **link == point + 1)
        .count();
    let mut work_weights = Array1::<Real>::zeros(work_point_count);
    let mut full_weights = Array1::<Real>::zeros(full_point_count);
    let mut irreducible_weight_sums = vec![0.0; work_point_count];
    let mut total_weight = 0.0;
    for point in 0..work_point_count {
        let weight = boundary_weight(
            [
                work_grid[(point, 0)],
                work_grid[(point, 1)],
                work_grid[(point, 2)],
            ],
            divisions,
        );
        work_weights[point] = weight;
        let full_link = checked_full_work_link(full_work_links[point], point, full_point_count)?;
        full_weights[full_link - 1] += weight;
        irreducible_weight_sums[work_links[point] - 1] += weight;
        total_weight += weight;
    }

    let mut irreducible_weights = Array1::<Real>::zeros(irreducible_point_count);
    let mut irreducible_index = 0usize;
    for (point, &link) in work_links.iter().enumerate() {
        if link == point + 1 {
            irreducible_weights[irreducible_index] = irreducible_weight_sums[link - 1];
            irreducible_index += 1;
        }
    }

    work_weights.mapv_inplace(|weight| weight / total_weight);
    full_weights.mapv_inplace(|weight| weight / total_weight);
    irreducible_weights.mapv_inplace(|weight| weight / total_weight);

    let mut work_vectors = Array2::<Real>::zeros((work_point_count, 3));
    let mut full_vectors = Array2::<Real>::zeros((full_point_count, 3));
    let mut irreducible_vectors = Array2::<Real>::zeros((irreducible_point_count, 3));
    let mut irreducible_fractional_vectors = Array2::<Real>::zeros((irreducible_point_count, 3));
    let mut full_links = vec![0usize; full_point_count];
    let mut full_symmetry = vec![0usize; full_point_count];
    let mut final_irreducible_count = 0usize;
    let mut final_full_count = 0usize;

    for point in 0..work_point_count {
        let fractional = [
            fractional_coordinate(work_grid[(point, 0)], shift[0], divisions[0]),
            fractional_coordinate(work_grid[(point, 1)], shift[1], divisions[1]),
            fractional_coordinate(work_grid[(point, 2)], shift[2], divisions[2]),
        ];
        let vector = kmesh_vector(reciprocal_vectors, fractional);
        for axis in 0..3 {
            work_vectors[(point, axis)] = vector[axis];
        }

        if work_links[point] == point + 1 {
            final_irreducible_count += 1;
            work_links[point] = final_irreducible_count;
            for axis in 0..3 {
                irreducible_vectors[(final_irreducible_count - 1, axis)] = vector[axis];
                irreducible_fractional_vectors[(final_irreducible_count - 1, axis)] =
                    fractional[axis];
            }
        } else {
            work_links[point] = work_links[work_links[point] - 1];
        }

        if full_work_links[point] == final_full_count + 1 {
            final_full_count += 1;
            full_links[final_full_count - 1] = work_links[point];
            full_symmetry[final_full_count - 1] = work_symmetry[point];
            for axis in 0..3 {
                full_vectors[(final_full_count - 1, axis)] = vector[axis];
            }
        }
    }

    if final_irreducible_count != irreducible_point_count {
        return Err(KSpaceError::KMeshReductionIrreducibleCountMismatch {
            expected: irreducible_point_count,
            actual: final_irreducible_count,
        });
    }
    validate_full_mesh_link_symmetry(&full_links, &full_symmetry)?;

    Ok(KMeshReduction {
        shift,
        total_weight,
        work_grid,
        work_vectors,
        full_vectors,
        irreducible_vectors,
        irreducible_fractional_vectors,
        work_weights,
        full_weights,
        irreducible_weights,
        work_links,
        full_links,
        work_symmetry,
        full_symmetry,
    })
}

/// Port of FEFF `KSPACE/kmesh.f90` `divisi`.
///
/// FEFF removes repeated common prime factors from every k-list component and
/// divides `idiv` by the same factor. Its control flow exits completely on the
/// first failed prime divisibility test, so this intentionally does not compute
/// a full greatest common divisor.
pub fn reduce_kmesh_common_divisor(
    k_list: ArrayView2<'_, i32>,
    division: usize,
) -> Result<KMeshDivisionReduction, KSpaceError> {
    if k_list.nrows() == 0 || k_list.ncols() != 3 {
        return Err(KSpaceError::InvalidKMeshListShape {
            rows: k_list.nrows(),
            columns: k_list.ncols(),
        });
    }

    let mut reduced = k_list.to_owned();
    let mut common_divisor = 1usize;
    'prime_search: for prime in DIVISI_PRIMES {
        for _ in 0..DIVISI_ITERATIONS {
            if reduced.iter().any(|value| value % prime != 0) {
                break 'prime_search;
            }
            common_divisor = common_divisor
                .checked_mul(prime as usize)
                .ok_or(KSpaceError::KMeshCommonDivisorOverflow)?;
            reduced.mapv_inplace(|value| value / prime);
        }
    }

    Ok(KMeshDivisionReduction {
        k_list: reduced,
        division: (division / common_divisor).max(1),
        common_divisor,
    })
}
