use ndarray::{Array1, Array2};

use super::*;

const LDOS_WEYL_A: Real = 0.359;

/// Port of FEFF `LDOS/reldos.f90` `changeklist`.
///
/// FEFF uses this as a replacement Weyl k-mesh for LDOS diagnostics. The
/// original routine hardcodes a simple-cubic direct basis and constructs
/// uniformly weighted k-points in reciprocal-coordinate units.
pub fn ldos_weyl_kmesh(point_count: usize) -> Result<LdosWeylKMesh, KSpaceError> {
    if point_count == 0 {
        return Err(KSpaceError::InvalidKMeshPointTarget {
            mesh_points: point_count,
        });
    }

    let zetax = 3.0_f64.sqrt() * LDOS_WEYL_A;
    let zetay = 5.0_f64.sqrt() * LDOS_WEYL_A;
    let zetaz = 2.0 * 13.0_f64.sqrt() * LDOS_WEYL_A;
    let mut k_points = Array2::zeros((point_count, 3));
    for point in 0..point_count {
        let feff_index = (point + 1) as Real;
        k_points[(point, 0)] = weyl_fraction(feff_index, zetax);
        k_points[(point, 1)] = weyl_fraction(feff_index, zetay);
        k_points[(point, 2)] = weyl_fraction(feff_index, zetaz);
    }

    Ok(LdosWeylKMesh {
        k_points,
        weights: Array1::from_elem(point_count, 1.0 / point_count as Real),
    })
}

fn weyl_fraction(index: Real, zeta: Real) -> Real {
    let scaled = index * zeta;
    scaled - scaled.trunc() - 0.5
}
