use ndarray::Array2;

use super::support::*;
use super::*;

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
