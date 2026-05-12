//! Full multiple-scattering helpers.
//!
//! FEFF's FMS routines use Rehr-Albers polynomial tables when building
//! multiple-scattering propagators. The helpers here keep the legacy table
//! layout explicit while returning Rust-owned `ndarray` storage.

use ndarray::{Array2, ArrayView2, ArrayView4, ShapeBuilder};
use num_complex::Complex32;
use thiserror::Error;

use crate::{Real, state::StateKet};

/// Atom record used by FEFF FMS cluster preparation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsAtom {
    /// Cartesian position in FEFF FMS single-precision arithmetic.
    pub position: [f32; 3],
    /// FEFF potential index for this atom.
    pub potential: i32,
}

/// Error returned by FEFF FMS helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FmsError {
    /// FEFF angular limits must fit the allocated `clm(lx+2, 2*lx+3)` table.
    #[error("{name}={value} is invalid for lx={lx}")]
    InvalidAngularLimit {
        name: &'static str,
        value: usize,
        lx: usize,
    },
    /// FEFF state-ket atom indices are one-based.
    #[error("state atom index must be one-based, got {atom}")]
    InvalidStateAtom { atom: usize },
    /// A zero-based Rust atom index was outside the supplied cluster table.
    #[error("atom index {index} is outside cluster length {len}")]
    AtomIndexOutOfRange { index: usize, len: usize },
    /// FMS cluster coordinates must be finite.
    #[error("atom {atom} coordinate axis {axis} must be finite")]
    NonFiniteCoordinate { atom: usize, axis: usize },
    /// FMS potential indices must fit the caller-provided potential range.
    #[error("potential {potential} is outside 0..={max_potential}")]
    PotentialOutOfRange {
        potential: i32,
        max_potential: usize,
    },
    /// FEFF `sortat` requires the first atom to be the central potential.
    #[error("first atom potential {actual} does not match central potential {expected}")]
    CentralAtomMismatch { expected: i32, actual: i32 },
    /// FEFF `xgllm` is called with `mu <= l1`.
    #[error("mu={mu} is invalid for angular momentum l={angular_momentum}")]
    MuOutOfRange { mu: usize, angular_momentum: usize },
    /// An input table is too small for a required FEFF index.
    #[error("{table} table is too small for {axis} index {index}")]
    TableIndexOutOfRange {
        table: &'static str,
        axis: &'static str,
        index: usize,
    },
    /// FEFF `xnlm(mu,l)` must be finite and nonzero when used as a divisor.
    #[error("xnlm({mu},{angular_momentum}) must be finite and nonzero")]
    InvalidNormalization { mu: usize, angular_momentum: usize },
    /// `rho` appears in the denominator of FEFF `xclmz`.
    #[error("rho must be nonzero")]
    ZeroRho,
    /// `rho` must contain finite real and imaginary parts.
    #[error("rho must be finite")]
    NonFiniteRho,
}

/// Port of FEFF `xclmz`: Rehr-Albers Hankel-like polynomial table.
///
/// The returned matrix has FEFF's work shape `clm(lx+2, 2*lx+3)` and
/// Fortran-order strides. Rust indices are zero-based, so FEFF `clm(il, im)`
/// is `table[(il - 1, im - 1)]`.
pub fn rehr_albers_polynomials(
    lx: usize,
    lmaxp1: usize,
    mmaxp1: usize,
    rho: Complex32,
) -> Result<Array2<Complex32>, FmsError> {
    let max_lmaxp1 = lx.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lx,
        lx,
    })?;
    if lmaxp1 == 0 || lmaxp1 > max_lmaxp1 {
        return Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: lmaxp1,
            lx,
        });
    }
    if mmaxp1 == 0 {
        return Err(FmsError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
            lx,
        });
    }
    if !(rho.re.is_finite() && rho.im.is_finite()) {
        return Err(FmsError::NonFiniteRho);
    }
    if rho == Complex32::new(0.0, 0.0) {
        return Err(FmsError::ZeroRho);
    }

    let rows = lx.checked_add(2).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lx,
        lx,
    })?;
    let cols = lx
        .checked_mul(2)
        .and_then(|value| value.checked_add(3))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    let mut clm = Array2::zeros((rows, cols).f());

    let one = Complex32::new(1.0, 0.0);
    let z = Complex32::new(0.0, -1.0) / rho;
    clm[(0, 0)] = one;
    clm[(1, 0)] = one - z;

    let lmax = lmaxp1 - 1;
    for il in 2..=lmax {
        let factor = odd_factor(il, lx)? * z;
        clm[(il, 0)] = clm[(il - 2, 0)] - factor * clm[(il - 1, 0)];
    }

    let mut cmm = one;
    let mmxp1 = lmaxp1.min(mmaxp1);
    for im in 2..=mmxp1 {
        let m = im - 1;
        let cmm_factor = odd_factor(m, lx)? * z;
        cmm = -cmm * cmm_factor;
        clm[(im - 1, im - 1)] = cmm;
        clm[(im, im - 1)] = cmm * odd_factor(im, lx)? * (one - Complex32::new(im as f32, 0.0) * z);

        for il in (im + 1)..=lmax {
            let factor = odd_factor(il, lx)? * z;
            clm[(il, im - 1)] =
                clm[(il - 2, im - 1)] - factor * (clm[(il - 1, im - 1)] + clm[(il - 1, im - 2)]);
        }
    }

    Ok(clm)
}

/// Port of FEFF `athep`: sort atoms by radius from the central atom.
///
/// The sort key is `x^2 + y^2 + z^2 + (input_index + 1) * 1e-6`, matching the
/// FEFF tie-breaker that preserves the old order for equidistant atoms. The
/// returned vector contains the sorted FEFF `ra` keys.
pub fn sort_atoms_by_radius(atoms: &mut [FmsAtom]) -> Result<Vec<f64>, FmsError> {
    let mut keyed_atoms = atoms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, atom)| sort_radius_key(index, atom).map(|key| (key, atom)))
        .collect::<Result<Vec<_>, _>>()?;

    keyed_atoms.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut keys = Vec::with_capacity(keyed_atoms.len());
    for (slot, (key, atom)) in atoms.iter_mut().zip(keyed_atoms) {
        *slot = atom;
        keys.push(key);
    }
    Ok(keys)
}

/// Port of FEFF `sortat`: move representative atoms into the FMS prefix.
///
/// The input atoms must already be sorted by radial distance. `max_potential`
/// is FEFF's inclusive `npot` loop bound; potential indices `0..=npot` are
/// considered. The returned vector maps each potential to its representative
/// zero-based atom index when that potential is present.
pub fn sort_representative_atoms(
    central_potential: i32,
    max_potential: usize,
    atoms: &mut [FmsAtom],
) -> Result<Vec<Option<usize>>, FmsError> {
    let central = checked_potential(central_potential, max_potential)?;
    let first = atoms
        .first()
        .ok_or(FmsError::AtomIndexOutOfRange { index: 0, len: 0 })?;
    if first.potential != central_potential {
        return Err(FmsError::CentralAtomMismatch {
            expected: central_potential,
            actual: first.potential,
        });
    }

    for (index, atom) in atoms.iter().enumerate() {
        ensure_finite_position(index, atom.position)?;
        checked_potential(atom.potential, max_potential)?;
    }

    let mut representative = vec![None; max_potential + 1];
    representative[central] = Some(0);
    for (potential, slot) in representative.iter_mut().enumerate() {
        if potential == central {
            continue;
        }
        *slot = atoms
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, atom)| atom.potential == potential as i32)
            .map(|(index, _)| index);
    }

    for potential in 0..=max_potential {
        let Some(point) = representative[potential] else {
            continue;
        };
        if point <= potential {
            continue;
        }

        atoms.swap(potential, point);
        for slot in representative
            .iter_mut()
            .take(max_potential + 1)
            .skip(potential + 1)
        {
            if *slot == Some(potential) {
                *slot = Some(point);
            }
        }
        representative[potential] = Some(potential);
    }

    let prefix_len = atoms.len().min(max_potential + 1);
    for (potential, representative_slot) in representative.iter_mut().enumerate() {
        let Some(point) = *representative_slot else {
            continue;
        };
        let last_in_prefix = atoms
            .iter()
            .take(prefix_len)
            .enumerate()
            .filter(|(_, atom)| atom.potential == potential as i32)
            .map(|(index, _)| index)
            .next_back();

        if let Some(last_in_prefix) = last_in_prefix
            && last_in_prefix != point
        {
            let position = atoms[last_in_prefix].position;
            atoms[last_in_prefix].position = atoms[point].position;
            atoms[point].position = position;
            *representative_slot = Some(last_in_prefix);
        }
    }

    Ok(representative)
}

/// Port of FEFF `getang`: polar angles for the vector `positions[i] - positions[j]`.
///
/// Rust indices are zero-based. The returned values are `(theta, phi)` in
/// radians using FEFF's single-precision thresholds.
pub fn pair_polar_angles(
    positions: &[[f32; 3]],
    i: usize,
    j: usize,
) -> Result<(f32, f32), FmsError> {
    let left = checked_position(positions, i)?;
    let right = checked_position(positions, j)?;
    if i == j {
        return Ok((0.0, 0.0));
    }

    let x = left[0] - right[0];
    let y = left[1] - right[1];
    let z = left[2] - right[2];
    let r = (x * x + y * y + z * z).sqrt();

    const TINY: f32 = 1.0e-7;
    let phi = if x.abs() < TINY {
        if y.abs() < TINY {
            0.0
        } else if y > TINY {
            std::f32::consts::FRAC_PI_2
        } else {
            -std::f32::consts::FRAC_PI_2
        }
    } else {
        y.atan2(x)
    };

    let theta = if r <= TINY {
        0.0
    } else if z <= -r {
        std::f32::consts::PI
    } else if z < r {
        (z / r).acos()
    } else {
        0.0
    };

    Ok((theta, phi))
}

/// Port of FEFF `xgllm`: z-axis Rehr-Albers propagator term.
///
/// `xclm` is indexed as `xclm(m, l, atom2 - 1, atom1 - 1)` and `xnlm` as
/// `xnlm(mu, l)`, matching FEFF's zero-based angular axes and one-based atom
/// labels. The state atoms in [`StateKet`] are therefore interpreted as FEFF
/// one-based atom indices.
pub fn rehr_albers_z_axis_propagator(
    mu: usize,
    first: StateKet,
    second: StateKet,
    xclm: ArrayView4<'_, Complex32>,
    xnlm: ArrayView2<'_, Real>,
) -> Result<Complex32, FmsError> {
    let iat1 = checked_atom_index(first.atom)?;
    let iat2 = checked_atom_index(second.atom)?;
    let l1 = first.angular_momentum;
    let l2 = second.angular_momentum;

    if mu > l1 {
        return Err(FmsError::MuOutOfRange {
            mu,
            angular_momentum: l1,
        });
    }

    ensure_axis_len("xclm", "m", xclm.shape()[0], l1.max(l2))?;
    ensure_axis_len("xclm", "l", xclm.shape()[1], l1.max(l2))?;
    ensure_axis_len("xclm", "atom2", xclm.shape()[2], iat2)?;
    ensure_axis_len("xclm", "atom1", xclm.shape()[3], iat1)?;
    ensure_axis_len("xnlm", "mu", xnlm.shape()[0], mu)?;
    ensure_axis_len("xnlm", "l", xnlm.shape()[1], l1.max(l2))?;

    if mu > l2 {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let norm_l1 = normalization_value(xnlm, mu, l1)?;
    let norm_l2 = normalization_value(xnlm, mu, l2)?;
    let angular_weight = angular_weight(l1)?;
    let sign = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };
    let numax = l1.min(l2 - mu);

    let sum = (0..=numax).try_fold(Complex32::new(0.0, 0.0), |sum, nu| {
        let mn = mu.checked_add(nu).ok_or(FmsError::InvalidAngularLimit {
            name: "mu",
            value: mu,
            lx: l2,
        })?;
        let gamtl = angular_weight * xclm[(nu, l1, iat2, iat1)] / norm_l1;
        let gam = xclm[(mn, l2, iat2, iat1)] * (sign * norm_l2);
        Ok(sum + gamtl * gam)
    })?;

    Ok(sum)
}

fn sort_radius_key(index: usize, atom: FmsAtom) -> Result<f64, FmsError> {
    ensure_finite_position(index, atom.position)?;
    Ok(f64::from(atom.position[0]) * f64::from(atom.position[0])
        + f64::from(atom.position[1]) * f64::from(atom.position[1])
        + f64::from(atom.position[2]) * f64::from(atom.position[2])
        + (index as f64 + 1.0) * 1.0e-6)
}

fn checked_potential(potential: i32, max_potential: usize) -> Result<usize, FmsError> {
    let Ok(potential_index) = usize::try_from(potential) else {
        return Err(FmsError::PotentialOutOfRange {
            potential,
            max_potential,
        });
    };
    if potential_index <= max_potential {
        Ok(potential_index)
    } else {
        Err(FmsError::PotentialOutOfRange {
            potential,
            max_potential,
        })
    }
}

fn checked_position(positions: &[[f32; 3]], index: usize) -> Result<[f32; 3], FmsError> {
    let position = positions
        .get(index)
        .copied()
        .ok_or(FmsError::AtomIndexOutOfRange {
            index,
            len: positions.len(),
        })?;
    ensure_finite_position(index, position)?;
    Ok(position)
}

fn ensure_finite_position(atom: usize, position: [f32; 3]) -> Result<(), FmsError> {
    for (axis, value) in position.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(FmsError::NonFiniteCoordinate { atom, axis });
        }
    }
    Ok(())
}

fn checked_atom_index(atom: usize) -> Result<usize, FmsError> {
    atom.checked_sub(1)
        .ok_or(FmsError::InvalidStateAtom { atom })
}

fn ensure_axis_len(
    table: &'static str,
    axis: &'static str,
    len: usize,
    index: usize,
) -> Result<(), FmsError> {
    if index < len {
        Ok(())
    } else {
        Err(FmsError::TableIndexOutOfRange { table, axis, index })
    }
}

fn normalization_value(
    xnlm: ArrayView2<'_, Real>,
    mu: usize,
    angular_momentum: usize,
) -> Result<f32, FmsError> {
    let value = xnlm[(mu, angular_momentum)] as f32;
    if value.is_finite() && value != 0.0 {
        Ok(value)
    } else {
        Err(FmsError::InvalidNormalization {
            mu,
            angular_momentum,
        })
    }
}

fn angular_weight(angular_momentum: usize) -> Result<Complex32, FmsError> {
    let value = angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "angular_momentum",
            value: angular_momentum,
            lx: angular_momentum,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}

fn odd_factor(index: usize, lx: usize) -> Result<Complex32, FmsError> {
    let value = index
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}

#[cfg(test)]
mod tests {
    use super::{FmsAtom, pair_polar_angles, sort_atoms_by_radius, sort_representative_atoms};
    use super::{FmsError, rehr_albers_polynomials, rehr_albers_z_axis_propagator};
    use crate::{Real, angular::legendre_normalization_table, state::StateKet};
    use ndarray::{Array2, Array4, ArrayView2, ShapeBuilder};
    use num_complex::Complex32;
    use std::error::Error;

    #[test]
    fn xclmz_matches_feff_reference_lx3() -> Result<(), FmsError> {
        let table = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;

        assert_eq!(table.shape(), &[5, 9]);
        assert_eq!(table.strides(), &[1, 5]);
        assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(table[(1, 0)], Complex32::new(1.2322206, 0.725_689_4));
        assert_complex32_close(table[(3, 0)], Complex32::new(-10.012509, 5.438_266));
        assert_complex32_close(table[(2, 1)], Complex32::new(-2.1395304, 4.1993084));
        assert_complex32_close(table[(3, 2)], Complex32::new(-23.036537, -6.8588142));
        assert_complex32_close(table[(4, 3)], Complex32::new(8.928_719, -161.62775));
        assert_complex32_close(
            matrix_sum(table.view()),
            Complex32::new(-58.983994, -154.61885),
        );
        assert_eq!(nonzero_count(table.view()), 11);
        Ok(())
    }

    #[test]
    fn xclmz_matches_feff_reference_with_limited_m() -> Result<(), FmsError> {
        let table = rehr_albers_polynomials(4, 3, 2, Complex32::new(-0.8, 1.1))?;

        assert_eq!(table.shape(), &[6, 11]);
        assert_eq!(table.strides(), &[1, 6]);
        assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(table[(1, 0)], Complex32::new(1.5945946, -0.432_432_4));
        assert_complex32_close(table[(2, 0)], Complex32::new(3.2834187, -2.840029));
        assert_complex32_close(table[(1, 1)], Complex32::new(0.5945946, -0.432_432_4));
        assert_complex32_close(table[(2, 1)], Complex32::new(2.7830534, -4.382761));
        assert_complex32_close(
            matrix_sum(table.view()),
            Complex32::new(9.255661, -8.087655),
        );
        assert_eq!(nonzero_count(table.view()), 5);
        Ok(())
    }

    #[test]
    fn xclmz_rejects_invalid_inputs() {
        assert_eq!(
            rehr_albers_polynomials(3, 0, 1, Complex32::new(1.0, 0.0)),
            Err(FmsError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
                lx: 3,
            })
        );
        assert_eq!(
            rehr_albers_polynomials(3, 5, 1, Complex32::new(1.0, 0.0)),
            Err(FmsError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 5,
                lx: 3,
            })
        );
        assert_eq!(
            rehr_albers_polynomials(3, 1, 1, Complex32::new(0.0, 0.0)),
            Err(FmsError::ZeroRho)
        );
        assert_eq!(
            rehr_albers_polynomials(3, 1, 1, Complex32::new(f32::NAN, 0.0)),
            Err(FmsError::NonFiniteRho)
        );
    }

    #[test]
    fn atheap_matches_feff_reference_sort_order() -> Result<(), FmsError> {
        let mut atoms = vec![
            FmsAtom {
                position: [2.0, 0.0, 0.0],
                potential: 1,
            },
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [-1.0, 0.0, 0.0],
                potential: 2,
            },
            FmsAtom {
                position: [1.0, 0.0, 0.0],
                potential: 3,
            },
            FmsAtom {
                position: [0.0, 2.0, 0.0],
                potential: 4,
            },
        ];

        let keys = sort_atoms_by_radius(&mut atoms)?;

        assert_eq!(
            atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
            vec![0, 2, 3, 1, 4]
        );
        assert_eq!(atoms[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(atoms[1].position, [-1.0, 0.0, 0.0]);
        assert_close_f64(keys[0], 2.0e-6);
        assert_close_f64(keys[1], 1.000_003);
        assert_close_f64(keys[2], 1.000_004);
        assert_close_f64(keys[3], 4.000_001);
        assert_close_f64(keys[4], 4.000_005);
        Ok(())
    }

    #[test]
    fn getang_matches_feff_reference_angles() -> Result<(), FmsError> {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [0.0, 5.0e-8, 2.0e-7],
            [0.0, 2.0e-7, 0.0],
        ];

        let (theta, phi) = pair_polar_angles(&positions, 1, 0)?;
        assert_close_f32(theta, 0.841_068_6);
        assert_close_f32(phi, 1.107_148_8);

        let (theta, phi) = pair_polar_angles(&positions, 3, 2)?;
        assert_close_f32(theta, 2.498_091_5);
        assert_close_f32(phi, 1.570_796_4);

        assert_eq!(pair_polar_angles(&positions, 0, 0)?, (0.0, 0.0));
        Ok(())
    }

    #[test]
    fn sortat_matches_feff_reference_representative_order() -> Result<(), FmsError> {
        let mut atoms = vec![
            FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            },
            FmsAtom {
                position: [1.0, 0.0, 0.0],
                potential: 2,
            },
            FmsAtom {
                position: [2.0, 0.0, 0.0],
                potential: 1,
            },
            FmsAtom {
                position: [3.0, 0.0, 0.0],
                potential: 3,
            },
            FmsAtom {
                position: [4.0, 0.0, 0.0],
                potential: 2,
            },
            FmsAtom {
                position: [5.0, 0.0, 0.0],
                potential: 1,
            },
        ];

        let representatives = sort_representative_atoms(0, 3, &mut atoms)?;

        assert_eq!(representatives, vec![Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(
            atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 2, 1]
        );
        assert_eq!(atoms[1].position, [2.0, 0.0, 0.0]);
        assert_eq!(atoms[2].position, [1.0, 0.0, 0.0]);
        assert_eq!(atoms[3].position, [3.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn fms_cluster_helpers_reject_invalid_inputs() {
        let positions = [[0.0, 0.0, 0.0]];
        assert_eq!(
            pair_polar_angles(&positions, 1, 0),
            Err(FmsError::AtomIndexOutOfRange { index: 1, len: 1 })
        );

        let mut atoms = [FmsAtom {
            position: [f32::NAN, 0.0, 0.0],
            potential: 0,
        }];
        assert_eq!(
            sort_atoms_by_radius(&mut atoms),
            Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 0 })
        );

        let mut atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        assert_eq!(
            sort_representative_atoms(0, 1, &mut atoms),
            Err(FmsError::CentralAtomMismatch {
                expected: 0,
                actual: 1,
            })
        );
        assert_eq!(
            sort_representative_atoms(-1, 1, &mut atoms),
            Err(FmsError::PotentialOutOfRange {
                potential: -1,
                max_potential: 1,
            })
        );
    }

    #[test]
    fn xgllm_matches_feff_reference() -> Result<(), Box<dyn Error>> {
        let (xclm, xnlm) = reference_xgllm_tables()?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 3,
            magnetic: 0,
            spin: 1,
        };

        assert_complex32_close(
            rehr_albers_z_axis_propagator(0, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(415.546_9, -1006.2809),
        );
        assert_complex32_close(
            rehr_albers_z_axis_propagator(1, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(-307.497_3, 722.469_5),
        );
        assert_complex32_close(
            rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(115.08963, -235.94589),
        );
        Ok(())
    }

    #[test]
    fn xgllm_matches_feff_empty_sum_case() -> Result<(), Box<dyn Error>> {
        let (xclm, xnlm) = reference_xgllm_tables()?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 3,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        };

        assert_complex32_close(
            rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
            Complex32::new(0.0, 0.0),
        );
        Ok(())
    }

    #[test]
    fn xgllm_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
        let (xclm, xnlm) = reference_xgllm_tables()?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 0,
            spin: 1,
        };
        let second = StateKet {
            atom: 2,
            angular_momentum: 3,
            magnetic: 0,
            spin: 1,
        };

        assert_eq!(
            rehr_albers_z_axis_propagator(3, first, second, xclm.view(), xnlm.view()),
            Err(FmsError::MuOutOfRange {
                mu: 3,
                angular_momentum: 2,
            })
        );
        assert_eq!(
            rehr_albers_z_axis_propagator(
                0,
                StateKet { atom: 0, ..first },
                second,
                xclm.view(),
                xnlm.view(),
            ),
            Err(FmsError::InvalidStateAtom { atom: 0 })
        );

        let mut bad_xnlm = xnlm.clone();
        bad_xnlm[(0, 2)] = 0.0;
        assert_eq!(
            rehr_albers_z_axis_propagator(0, first, second, xclm.view(), bad_xnlm.view()),
            Err(FmsError::InvalidNormalization {
                mu: 0,
                angular_momentum: 2,
            })
        );
        Ok(())
    }

    fn reference_xgllm_tables() -> Result<(Array4<Complex32>, Array2<Real>), Box<dyn Error>> {
        let clm = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;
        let mut xclm = Array4::zeros((4, 4, 2, 2).f());
        for l in 0..=3 {
            for m in 0..=3 {
                xclm[(m, l, 1, 0)] = clm[(l, m)];
                xclm[(m, l, 0, 1)] = clm[(l, m)];
            }
        }
        Ok((xclm, legendre_normalization_table(3)?))
    }

    fn matrix_sum(matrix: ArrayView2<'_, Complex32>) -> Complex32 {
        matrix
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn nonzero_count(matrix: ArrayView2<'_, Complex32>) -> usize {
        matrix
            .iter()
            .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
            .count()
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32) {
        assert!(
            (actual - expected).norm() < 2.0e-4,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_close_f32(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 2.0e-6,
            "actual={actual} expected={expected}"
        );
    }

    fn assert_close_f64(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
