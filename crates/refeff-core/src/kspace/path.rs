use ndarray::{Array1, Array2};

use super::support::*;
use super::*;

const BAND_K_PATH_LENGTH_SCALE: Real = 1_000_000.0;

#[derive(Debug, Clone, Copy)]
struct Segment {
    label: &'static str,
    start: Vector3,
    end: Vector3,
}

/// Port of FEFF `BAND/kpath.f90` for supported Bravais lattices.
///
/// `reciprocal_basis` stores the three reciprocal basis vectors as
/// `[g1, g2, g3]`. FEFF leaves some lattices as hard `STOP` cases; those return
/// [`KSpaceError::UnsupportedBravais`]. Segment starts and ends are returned as
/// `ndarray` matrices with shape `(segment, xyz)`.
pub fn define_k_path(
    bravais: BravaisLattice,
    kpath: i32,
    reciprocal_basis: [Vector3; 3],
) -> Result<KPath, KSpaceError> {
    validate_basis(reciprocal_basis)?;

    let mut effective_kpath = kpath;
    let mut segments = Vec::with_capacity(12);
    let take = match bravais {
        BravaisLattice::OrthorhombicPrimitive => {
            orthorhombic_primitive_segments(kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::HexagonalPrimitive => {
            hexagonal_primitive_segments(kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::CubicPrimitive => {
            cubic_primitive_segments(kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::CubicFaceCentered => {
            if effective_kpath == 0 {
                effective_kpath = 4;
            }
            cubic_face_centered_segments(effective_kpath, reciprocal_basis, &mut segments)?
        }
        BravaisLattice::CubicBodyCentered => {
            if effective_kpath == 0 {
                effective_kpath = 5;
            }
            cubic_body_centered_segments(effective_kpath, reciprocal_basis, &mut segments)?
        }
        _ => {
            return Err(KSpaceError::UnsupportedBravais {
                bravais: bravais.index(),
            });
        }
    };

    if segments.len() < take {
        return Err(KSpaceError::KPathDefinitionIncomplete {
            bravais: bravais.index(),
            kpath: effective_kpath,
            available: segments.len(),
            required: take,
        });
    }

    let mut labels = Vec::with_capacity(take);
    let mut starts = Array2::<Real>::zeros((take, 3));
    let mut ends = Array2::<Real>::zeros((take, 3));
    for (row, segment) in segments.into_iter().take(take).enumerate() {
        labels.push(segment.label.to_string());
        for axis in 0..3 {
            starts[(row, axis)] = segment.start[axis];
            ends[(row, axis)] = segment.end[axis];
        }
    }

    Ok(KPath {
        bravais,
        requested_kpath: kpath,
        effective_kpath,
        labels,
        starts,
        ends,
    })
}

/// Build the sampled BAND k-path used by FEFF `BAND/bandtot.f90`.
///
/// FEFF first obtains high-symmetry path segments from `define_kpath`, then
/// distributes the requested `nkp` count across segments in proportion to
/// segment length using integer arithmetic and a minimum of two points per
/// segment. Junction points are duplicated with unchanged cumulative path
/// distance, matching `bandtot.f90`'s `bk(:,ik)` and `KP(ik)` construction.
pub fn band_k_path_mesh(
    path: &KPath,
    requested_point_count: usize,
) -> Result<BandKPathMesh, KSpaceError> {
    if requested_point_count < 2 {
        return Err(KSpaceError::InvalidBandKPathPointTarget {
            point_count: requested_point_count,
        });
    }
    let segment_count = validate_k_path_segment_shape(path)?;
    if segment_count == 0 {
        return default_band_k_path_mesh(requested_point_count);
    }

    let segment_lengths = (0..segment_count)
        .map(|segment| k_path_segment_length(path, segment))
        .collect::<Result<Vec<_>, _>>()?;
    let total_length = segment_lengths.iter().copied().sum::<Real>();
    if !total_length.is_finite() || total_length <= 0.0 {
        return Err(KSpaceError::DegenerateBandKPathLength);
    }
    let denominator = rounded_scaled_band_k_path_length(total_length)?;

    let mut segment_point_counts = Vec::with_capacity(segment_count);
    let mut segment_end_indices = Vec::with_capacity(segment_count);
    let mut point_count = 0_usize;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let scaled_length = truncated_scaled_band_k_path_length(length, segment)?;
        let proportional = requested_point_count
            .checked_mul(scaled_length)
            .ok_or(KSpaceError::BandKPathPointCountOverflow)?
            / denominator;
        let segment_points = proportional.max(2);
        point_count = point_count
            .checked_add(segment_points)
            .ok_or(KSpaceError::BandKPathPointCountOverflow)?;
        segment_point_counts.push(segment_points);
        segment_end_indices.push(point_count);
    }

    let mut k_points = Array2::<Real>::zeros((point_count, 3));
    let mut path_distances = Array1::<Real>::zeros(point_count);
    let mut point = 0_usize;
    for (segment, &segment_points) in segment_point_counts.iter().enumerate().take(segment_count) {
        let step = [
            (path.ends[(segment, 0)] - path.starts[(segment, 0)]) / ((segment_points - 1) as Real),
            (path.ends[(segment, 1)] - path.starts[(segment, 1)]) / ((segment_points - 1) as Real),
            (path.ends[(segment, 2)] - path.starts[(segment, 2)]) / ((segment_points - 1) as Real),
        ];
        let step_distance = vector_norm(step);
        for segment_point in 0..segment_points {
            for axis in 0..3 {
                k_points[(point, axis)] =
                    path.starts[(segment, axis)] + step[axis] * (segment_point as Real);
            }
            if point == 0 {
                path_distances[point] = 0.0;
            } else if segment_point == 0 {
                path_distances[point] = path_distances[point - 1];
            } else {
                path_distances[point] = path_distances[point - 1] + step_distance;
            }
            point += 1;
        }
    }

    Ok(BandKPathMesh {
        labels: path.labels.clone(),
        segment_point_counts,
        segment_end_indices,
        k_points,
        path_distances,
    })
}

fn default_band_k_path_mesh(requested_point_count: usize) -> Result<BandKPathMesh, KSpaceError> {
    let mut k_points = Array2::<Real>::zeros((requested_point_count, 3));
    let mut path_distances = Array1::<Real>::zeros(requested_point_count);
    let denominator = (requested_point_count - 1) as Real;
    for point in 0..requested_point_count {
        let distance = (point as Real) / denominator;
        k_points[(point, 0)] = distance;
        path_distances[point] = distance;
    }
    Ok(BandKPathMesh {
        labels: vec!["GG-x -1 ".to_string()],
        segment_point_counts: vec![requested_point_count],
        segment_end_indices: vec![requested_point_count],
        k_points,
        path_distances,
    })
}

fn validate_k_path_segment_shape(path: &KPath) -> Result<usize, KSpaceError> {
    let labels = path.labels.len();
    if path.starts.nrows() != labels
        || path.starts.ncols() != 3
        || path.ends.nrows() != labels
        || path.ends.ncols() != 3
    {
        return Err(KSpaceError::InvalidKPathSegmentShape {
            labels,
            start_rows: path.starts.nrows(),
            start_columns: path.starts.ncols(),
            end_rows: path.ends.nrows(),
            end_columns: path.ends.ncols(),
        });
    }
    Ok(labels)
}

fn k_path_segment_length(path: &KPath, segment: usize) -> Result<Real, KSpaceError> {
    let delta = [
        path.ends[(segment, 0)] - path.starts[(segment, 0)],
        path.ends[(segment, 1)] - path.starts[(segment, 1)],
        path.ends[(segment, 2)] - path.starts[(segment, 2)],
    ];
    for (axis, &value) in delta.iter().enumerate() {
        if !value.is_finite() {
            return Err(KSpaceError::NonFiniteValue {
                name: "kpath_segment_delta",
                index: segment * 3 + axis,
                value,
            });
        }
    }
    Ok(vector_norm(delta))
}

fn truncated_scaled_band_k_path_length(length: Real, segment: usize) -> Result<usize, KSpaceError> {
    let scaled = (length * BAND_K_PATH_LENGTH_SCALE).trunc();
    if !scaled.is_finite() {
        return Err(KSpaceError::NonFiniteValue {
            name: "kpath_segment_length",
            index: segment,
            value: length,
        });
    }
    if scaled > (usize::MAX as Real) {
        return Err(KSpaceError::BandKPathPointCountOverflow);
    }
    Ok(scaled as usize)
}

fn rounded_scaled_band_k_path_length(length: Real) -> Result<usize, KSpaceError> {
    let rounded = (length * BAND_K_PATH_LENGTH_SCALE).round();
    if !rounded.is_finite() || rounded <= 0.0 {
        return Err(KSpaceError::DegenerateBandKPathLength);
    }
    if rounded > (usize::MAX as Real) {
        return Err(KSpaceError::BandKPathPointCountOverflow);
    }
    Ok(rounded as usize)
}

fn vector_norm(vector: Vector3) -> Real {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

fn orthorhombic_primitive_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let y = lc3(0.0, 0.5, 0.0, basis);
    let x = lc3(0.5, 0.0, 0.0, basis);
    let z = lc3(0.0, 0.0, 0.5, basis);
    let u = lc3(0.5, 0.0, 0.5, basis);
    let t = lc3(0.0, 0.5, 0.5, basis);
    let s = lc3(0.5, 0.5, 0.0, basis);
    let r = lc3(0.5, 0.5, 0.5, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 12,
        2 => 7,
        3 => 4,
        4 => 3,
        5 => 1,
        6 => 1,
        7 => 2,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::OrthorhombicPrimitive.index(),
                kpath,
            });
        }
    };

    if !matches!(kpath, 4 | 6 | 7) {
        push_segment(segments, "GG-GS-X ", gamma, x);
        push_segment(segments, "X -G -U ", x, u);
        push_segment(segments, "U -A -Z ", u, z);
        push_segment(segments, "Z -GL-GG", z, gamma);
    }
    if kpath == 7 {
        push_segment(segments, "X -GS-GG", x, gamma);
    }
    push_segment(segments, "GG-GD-Y ", gamma, y);
    push_segment(segments, "Y -H -T ", y, t);
    push_segment(segments, "T -B -Z ", t, z);
    push_segment(segments, "X -D -S ", x, s);
    push_segment(segments, "S -C -Y ", s, y);
    push_segment(segments, "U -P -R ", u, r);
    push_segment(segments, "R -E -T ", r, t);
    push_segment(segments, "S -Q -T ", s, t);
    Ok(take)
}

fn hexagonal_primitive_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let m = lc3(0.0, 0.5, 0.0, basis);
    let a = lc3(0.0, 0.0, 0.5, basis);
    let l = lc3(0.0, 0.5, 0.5, basis);
    let k = lc3(-1.0 / 3.0, 2.0 / 3.0, 0.0, basis);
    let h = lc3(-1.0 / 3.0, 2.0 / 3.0, 0.5, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 9,
        2 => 7,
        3 => 4,
        4 => 1,
        5 => 1,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::HexagonalPrimitive.index(),
                kpath,
            });
        }
    };

    if kpath != 5 {
        push_segment(segments, "GG-GS-M ", gamma, m);
        push_segment(segments, "M -T'-K ", m, k);
    }
    push_segment(segments, "K -T -GG", k, gamma);
    push_segment(segments, "GG-GD-A ", gamma, a);
    push_segment(segments, "A -R -L ", a, l);
    push_segment(segments, "L -S'-H ", l, h);
    push_segment(segments, "H -S -A ", h, a);
    push_segment(segments, "M -U -L ", m, l);
    push_segment(segments, "K -P -H ", k, h);
    Ok(take)
}

fn cubic_primitive_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let x = lc3(0.0, 0.5, 0.0, basis);
    let m = lc3(0.5, 0.5, 0.0, basis);
    let r = lc3(0.5, 0.5, 0.5, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        0 => 1,
        1 => 5,
        2 => 4,
        3 => 3,
        4 => 2,
        5 => 3,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::CubicPrimitive.index(),
                kpath,
            });
        }
    };

    if kpath == 5 {
        push_segment(segments, "GG-GD-X ", gamma, [0.5, 0.0, 0.0]);
        push_segment(segments, "GG-GD-Y ", gamma, [0.0, 0.5, 0.0]);
        push_segment(segments, "GG-GD-Z ", gamma, [0.0, 0.0, 0.5]);
    }
    push_segment(segments, "GG-GD-X ", gamma, x);
    push_segment(segments, "X -Y -M ", x, m);
    push_segment(segments, "M -V -R ", m, r);
    push_segment(segments, "R -GL-GG", r, gamma);
    push_segment(segments, "GG-GS-M ", gamma, m);
    Ok(take)
}

fn cubic_face_centered_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let x = lc3(0.5, 0.0, 0.5, basis);
    let l = lc3(0.5, 0.5, 0.5, basis);
    let w = lc3(0.75, 0.25, 0.5, basis);
    let k = lc3(0.75, 0.375, 0.375, basis);
    let u = lc3(0.625, 0.25, 0.625, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 9,
        2 => 5,
        3 => 2,
        4 => 1,
        5 => 1,
        6 => 3,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::CubicFaceCentered.index(),
                kpath,
            });
        }
    };

    if kpath == 4 {
        push_segment(segments, "GG-GD-X ", gamma, x);
    }
    if kpath == 6 {
        push_segment(segments, "GG-GD-X ", gamma, [1.0, 0.0, 0.0]);
        push_segment(segments, "GG-GD-Y ", gamma, [0.0, 1.0, 0.0]);
        push_segment(segments, "GG-GD-Z ", gamma, [0.0, 0.0, 1.0]);
    }
    if kpath != 5 {
        push_segment(segments, "X -GD-GG", x, gamma);
    }
    push_segment(segments, "GG-GL-L ", gamma, l);
    push_segment(segments, "L -Q -W ", l, w);
    push_segment(segments, "W -N -K ", w, k);
    push_segment(segments, "K -GS-GG", k, gamma);
    push_segment(segments, "L -M -U ", l, u);
    push_segment(segments, "U -S -X ", u, x);
    push_segment(segments, "X -Z -W ", x, w);
    push_segment(segments, "W -B -U ", w, u);
    Ok(take)
}

fn cubic_body_centered_segments(
    kpath: i32,
    basis: [Vector3; 3],
    segments: &mut Vec<Segment>,
) -> Result<usize, KSpaceError> {
    let h = lc3(0.5, 0.5, -0.5, basis);
    let p = lc3(0.25, 0.25, 0.25, basis);
    let n = lc3(0.5, 0.0, 0.0, basis);
    let gamma = [0.0; 3];

    let take = match kpath {
        1 => 6,
        2 => 5,
        3 => 4,
        4 => 3,
        5 => 1,
        6 => 3,
        _ => {
            return Err(KSpaceError::InvalidKPath {
                bravais: BravaisLattice::CubicBodyCentered.index(),
                kpath,
            });
        }
    };

    if kpath == 6 {
        push_segment(segments, "GG-GD-X ", gamma, [1.0, 0.0, 0.0]);
        push_segment(segments, "GG-GD-Y ", gamma, [0.0, 1.0, 0.0]);
        push_segment(segments, "GG-GD-Z ", gamma, [0.0, 0.0, 1.0]);
    }
    push_segment(segments, "GG-GD-H ", gamma, h);
    push_segment(segments, "H -G -N ", h, n);
    push_segment(segments, "N -GS-GG", n, gamma);
    push_segment(segments, "GG-GL-P ", gamma, p);
    push_segment(segments, "P -F -H ", p, h);
    push_segment(segments, "N -D -P ", n, p);
    Ok(take)
}

fn push_segment(segments: &mut Vec<Segment>, label: &'static str, start: Vector3, end: Vector3) {
    segments.push(Segment { label, start, end });
}

fn lc3(c1: Real, c2: Real, c3: Real, basis: [Vector3; 3]) -> Vector3 {
    [
        c1 * basis[0][0] + c2 * basis[1][0] + c3 * basis[2][0],
        c1 * basis[0][1] + c2 * basis[1][1] + c3 * basis[2][1],
        c1 * basis[0][2] + c2 * basis[1][2] + c3 * basis[2][2],
    ]
}
