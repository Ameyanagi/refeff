//! Small XSPH planning helpers ported from FEFF.
//!
//! The full XSPH phase-shift driver is still being ported incrementally. This
//! module contains self-contained helper kernels from `XSPH/mincalc.f90` and
//! `XSPH/ljneeded0.f90` that decide which final-state calculations can be
//! shared and which angular channels are needed for each shared calculation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ShapeBuilder};
use thiserror::Error;

use crate::{
    AngularError, BesselError, Complex, Real, legendre_polynomials_into, spherical_bessel_j_y,
    wigner_3j,
};

const QBESSEL_MAX_LJ: usize = 39;
const QBESSEL_ZERO_CUTOFF: Real = 1.0e8;
const CWIG3J_MAX_DOUBLED_ARGUMENT: i32 = 116;

/// Shared final-state calculation plan returned by [`xsph_minimize_calculations`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphCalculationPlan {
    /// Maximum `lj` encountered in the active final-state index list, FEFF `ljj`.
    pub max_lj: i32,
    /// Rows `[kind, max_lj_for_kind, representative_l]`, FEFF `indcalc`.
    pub calculations: Array2<i32>,
    /// Per-final-state map to a calculation row, FEFF `indmap`.
    ///
    /// Positive values mark the first occurrence of a final-state `kind`.
    /// Negative values reuse the absolute calculation index from an earlier
    /// occurrence, matching FEFF's convention.
    pub index_map: Array1<i32>,
}

/// FEFF `XSPH/xmult.f90` relativistic multipole prefactors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphRelativisticMultipoleFactors {
    /// FEFF `xm1`, multiplying the radial `P_k * Q_k'` contribution.
    pub p_q_prime: Complex,
    /// FEFF `xm2`, multiplying the radial `Q_k * P_k'` contribution.
    pub q_p_prime: Complex,
}

/// FEFF `specupdlg` branch for regular or irregular radial contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphSpectrumUpdateMode {
    /// FEFF `imode = 1`, regular radial-integral branch.
    Regular,
    /// FEFF `imode = 2`, irregular radial-integral branch.
    Irregular,
}

/// Inputs for FEFF `XSPH/specupdlg.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphLgSpectrumUpdateInput<'a> {
    /// One-based shared calculation index, FEFF `icalc`.
    pub calculation_index: i32,
    /// Spin component, FEFF `isp` in the range `0..=1`.
    pub spin_index: usize,
    /// Per-final-state calculation map, FEFF `indmap(1:indmax)`.
    pub index_map: ArrayView1<'a, i32>,
    /// Angular-decomposition output index, FEFF `lind(1:indmax)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Radial-integral/Legendre index, FEFF `ljind(1:indmax)`.
    pub final_lj: ArrayView1<'a, i32>,
    /// Doubled initial-state angular momentum, FEFF `jinit`.
    pub initial_j2: i32,
    /// Transition weights with compact magnetic index, FEFF `hbmat(0:1,ii,mjinit)`.
    ///
    /// Shape must be at least `(2, active_len, initial_j2 + 1)`, where the
    /// compact magnetic column is `(mjinit + initial_j2) / 2`.
    pub transition_weights: ArrayView3<'a, Real>,
    /// Radial integrals `xirflj(0:ljmax)`.
    pub radial_integrals: ArrayView1<'a, Complex>,
    /// MDFF q-vector weights, FEFF `qw(1:nq)`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// Cosines between q-vector pairs, FEFF `cosmdff(1:nq,1:nq)`.
    pub q_cosines: ArrayView2<'a, Real>,
    /// Whether FEFF `mixdff` is enabled.
    pub mix_dff: bool,
    /// FEFF `imdff` selector when `mix_dff` is enabled.
    pub mdff_mode: i32,
    /// Largest active `lj`/`lg` index.
    pub ljmax: usize,
    /// Number of active final states, FEFF `indmax`.
    pub active_len: usize,
    /// FEFF regular/irregular update mode.
    pub mode: XsphSpectrumUpdateMode,
}

/// Error returned by XSPH planning helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum XsphError {
    /// FEFF `mincalc` expects at least one active final-state index.
    #[error("XSPH calculation planning requires at least one active index")]
    EmptyIndexSet,
    /// A supplied index row is shorter than the requested active prefix.
    #[error("{name} length {actual} is shorter than active length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// Angular momentum indices used as output slots must be non-negative.
    #[error("{name} entry {index} must be non-negative, got {value}")]
    NegativeAngularMomentum {
        name: &'static str,
        index: usize,
        value: i32,
    },
    /// FEFF `ljneeded0` would stop when an `lj` index exceeds `ljmax`.
    #[error("XSPH angular momentum {angular_momentum} exceeds ljmax {ljmax}")]
    AngularMomentumOutOfRange {
        angular_momentum: usize,
        ljmax: usize,
    },
    /// Shared calculation indices are one-based in FEFF.
    #[error("XSPH calculation index must be positive, got {calculation_index}")]
    NonPositiveCalculationIndex { calculation_index: i32 },
    /// The FEFF map convention cannot represent `abs(i32::MIN)`.
    #[error("XSPH index map entry {index} cannot be negated: {value}")]
    IndexMapOverflow { index: usize, value: i32 },
    /// Requested output size overflows `usize`.
    #[error("XSPH ljmax {ljmax} cannot be represented as an output vector length")]
    AngularMomentumCapacityOverflow { ljmax: usize },
    /// XSPH scalar inputs must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// XSPH complex inputs must have finite real and imaginary parts.
    #[error("{name} entry {index} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        index: usize,
        real: Real,
        imaginary: Real,
    },
    /// Spherical Bessel evaluation failed.
    #[error(transparent)]
    Bessel(#[from] BesselError),
    /// Wigner-symbol evaluation failed.
    #[error(transparent)]
    Angular(#[from] AngularError),
    /// Relativistic kappa values must be nonzero.
    #[error("XSPH relativistic kappa must be nonzero")]
    ZeroKappa,
    /// Integer angular inputs must stay in the supported FEFF range.
    #[error("{name} value {value} is outside the supported XSPH integer range")]
    IntegerOutOfRange { name: &'static str, value: i32 },
    /// Rust-sized inputs must fit the FEFF integer helper range.
    #[error("{name} size {value} is outside the supported XSPH integer range")]
    SizeOutOfRange { name: &'static str, value: usize },
    /// FEFF `bcoefjas` generated too few final-state rows for `indmax`.
    #[error("XSPH generated {generated} NRIXS final states, fewer than active length {required}")]
    InsufficientGeneratedStates { required: usize, generated: usize },
    /// FEFF `specupd*` spin indices are limited to two spin components.
    #[error("XSPH spin index must be 0 or 1, got {spin_index}")]
    InvalidSpinIndex { spin_index: usize },
    /// FEFF `specupd*` received an unsupported MDFF selector.
    #[error("XSPH unsupported MDFF mode {mdff_mode}")]
    InvalidMdffMode { mdff_mode: i32 },
    /// Multidimensional arrays must have enough rows for the FEFF active shape.
    #[error("{name} shape {actual:?} is smaller than required shape {required:?}")]
    ShapeTooSmall {
        name: &'static str,
        required: [usize; 3],
        actual: [usize; 3],
    },
    /// Two-dimensional q-pair tables must cover every active q weight.
    #[error("{name} shape {actual:?} is smaller than required shape {required:?}")]
    MatrixTooSmall {
        name: &'static str,
        required: [usize; 2],
        actual: [usize; 2],
    },
}

/// Port of FEFF `XSPH/mincalc.f90`.
///
/// The input arrays may contain extra capacity, mirroring FEFF's `kfinmax`;
/// `active_len` is FEFF's `indmax` and selects the active prefix. The returned
/// `calculations` table contains one row for each distinct `kind`, with the
/// maximum `ljind` observed for that kind folded into column 1.
pub fn xsph_minimize_calculations(
    kind: ArrayView1<'_, i32>,
    orbital_l: ArrayView1<'_, i32>,
    final_lj: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<XsphCalculationPlan, XsphError> {
    validate_active_len("kind", kind.len(), active_len)?;
    validate_active_len("orbital_l", orbital_l.len(), active_len)?;
    validate_active_len("final_lj", final_lj.len(), active_len)?;
    validate_final_lj(final_lj, active_len)?;

    let mut calculations = Array2::<i32>::zeros((active_len, 3));
    let mut index_map = Array1::<i32>::zeros(active_len);

    let mut calculation_count = 1_usize;
    calculations[(0, 0)] = kind[0];
    calculations[(0, 1)] = final_lj[0];
    calculations[(0, 2)] = orbital_l[0];
    index_map[0] = 1;
    let mut max_lj = final_lj[0];

    for index in 1..active_len {
        let current_kind = kind[index];
        let current_lj = final_lj[index];
        max_lj = max_lj.max(current_lj);

        let existing = (0..calculation_count)
            .find(|&row| current_kind == calculations[(row, 0)])
            .map(|row| row + 1);

        if let Some(one_based_row) = existing {
            index_map[index] =
                -i32::try_from(one_based_row).map_err(|_| XsphError::IndexMapOverflow {
                    index,
                    value: i32::MIN,
                })?;
            let row = one_based_row - 1;
            calculations[(row, 1)] = calculations[(row, 1)].max(current_lj);
        } else {
            let row = calculation_count;
            calculation_count += 1;
            index_map[index] =
                i32::try_from(calculation_count).map_err(|_| XsphError::IndexMapOverflow {
                    index,
                    value: i32::MIN,
                })?;
            calculations[(row, 0)] = current_kind;
            calculations[(row, 1)] = current_lj;
            calculations[(row, 2)] = orbital_l[index];
        }
    }

    let compact_calculations = Array2::from_shape_fn((calculation_count, 3), |(row, column)| {
        calculations[(row, column)]
    });

    Ok(XsphCalculationPlan {
        max_lj,
        calculations: compact_calculations,
        index_map,
    })
}

/// Port of FEFF `XSPH/ljneeded0.f90`.
///
/// Returns FEFF's integer flags for angular channels `0..=ljmax` that are used
/// by the one-based shared calculation `calculation_index`.
pub fn xsph_lj_needed_flags(
    ljmax: usize,
    final_lj: ArrayView1<'_, i32>,
    index_map: ArrayView1<'_, i32>,
    active_len: usize,
    calculation_index: i32,
) -> Result<Array1<i32>, XsphError> {
    validate_active_len("final_lj", final_lj.len(), active_len)?;
    validate_active_len("index_map", index_map.len(), active_len)?;
    validate_final_lj(final_lj, active_len)?;
    if calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex { calculation_index });
    }

    let output_len = ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax })?;
    let mut needed = Array1::<i32>::zeros(output_len);
    for index in 0..active_len {
        let mapped = index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: index_map[index],
            })?;
        if mapped == calculation_index {
            let angular_momentum = usize::try_from(final_lj[index]).map_err(|_| {
                XsphError::NegativeAngularMomentum {
                    name: "final_lj",
                    index,
                    value: final_lj[index],
                }
            })?;
            if angular_momentum > ljmax {
                return Err(XsphError::AngularMomentumOutOfRange {
                    angular_momentum,
                    ljmax,
                });
            }
            needed[angular_momentum] = 1;
        }
    }
    Ok(needed)
}

/// Port of FEFF `XSPH/qbesselget.f90`.
///
/// Builds a Fortran-order table `j_l(qtrans * r)` with rows over radii and
/// columns over `l = 0..=ljmax`. FEFF skips Bessel evaluation and stores zeros
/// when `qtrans * r >= 1e8`; this adapter keeps the same cutoff.
pub fn xsph_q_bessel_table(
    qtrans: Real,
    radii: ArrayView1<'_, Real>,
    ljmax: usize,
) -> Result<Array2<Real>, XsphError> {
    validate_finite_real("qtrans", qtrans)?;
    if ljmax > QBESSEL_MAX_LJ {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: ljmax,
            ljmax: QBESSEL_MAX_LJ,
        });
    }

    let column_count = ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax })?;
    let mut table = Array2::<Real>::zeros((radii.len(), column_count).f());
    for (radius_index, &radius) in radii.iter().enumerate() {
        validate_finite_real("radius", radius)?;
        let argument = qtrans * radius;
        validate_finite_real("qtrans * radius", argument)?;
        if argument < QBESSEL_ZERO_CUTOFF {
            let values = spherical_bessel_j_y(Complex::new(argument, 0.0), ljmax)?;
            for angular_momentum in 0..=ljmax {
                table[(radius_index, angular_momentum)] = values.j[angular_momentum].re;
            }
        }
    }
    Ok(table)
}

/// Port of FEFF `XSPH/xmultjas.f90`.
///
/// Returns the longitudinal multipole prefactor for
/// `<k|exp(i*q*z)|k'>`. FEFF declares `xm` as `complex*16`, but this helper is
/// real-valued because `xmultjas` removes the `i**ls` phase and applies it in
/// the caller.
pub fn xsph_longitudinal_multipole_factor(
    kappa: i32,
    kappa_prime: i32,
    multipole_l: i32,
) -> Result<Complex, XsphError> {
    if kappa == 0 || kappa_prime == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if multipole_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            index: 0,
            value: multipole_l,
        });
    }

    let j2 = doubled_j_from_kappa("kappa", kappa)?;
    let parity = if kappa > 0 { -1 } else { 1 };
    let j2_prime = doubled_j_from_kappa("kappa_prime", kappa_prime)?;
    let parity_prime = if kappa_prime > 0 { -1 } else { 1 };
    let doubled_multipole = multipole_l
        .checked_mul(2)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole_l",
            value: multipole_l,
        })?;

    let parity_check =
        i64::from(j2) + i64::from(j2_prime) + i64::from(doubled_multipole) + i64::from(parity)
            - i64::from(parity_prime);
    let doubled_difference = (i64::from(j2) - i64::from(j2_prime)).abs();
    let doubled_sum = i64::from(j2) + i64::from(j2_prime);
    if parity_check.rem_euclid(4) == 0
        || i64::from(doubled_multipole) < doubled_difference
        || i64::from(doubled_multipole) > doubled_sum
    {
        return Ok(Complex::new(0.0, 0.0));
    }

    validate_cwig3j_doubled_argument("kappa", kappa, j2)?;
    validate_cwig3j_doubled_argument("kappa_prime", kappa_prime, j2_prime)?;
    validate_cwig3j_doubled_argument("multipole_l", multipole_l, doubled_multipole)?;
    let angular_weight = ((f64::from(j2) + 1.0) * (f64::from(j2_prime) + 1.0)).sqrt()
        * (f64::from(doubled_multipole) + 1.0);
    let value = angular_weight * wigner_3j(j2, doubled_multipole, j2_prime, 1, 0, 2)?;
    Ok(Complex::new(value, 0.0))
}

/// Port of FEFF `XSPH/xmult.f90`.
///
/// Returns FEFF's `xm1` and `xm2` angular prefactors for Grant equation 6.30,
/// used by `radint.f90` before the relativistic radial integrals. `bessel_l`
/// is FEFF `ls`, the spherical-Bessel order, and `multipole_l` is FEFF `lb`,
/// the vector multipole order.
pub fn xsph_relativistic_multipole_factors(
    kappa: i32,
    kappa_prime: i32,
    bessel_l: i32,
    multipole_l: i32,
) -> Result<XsphRelativisticMultipoleFactors, XsphError> {
    if kappa == 0 || kappa_prime == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_nonnegative_angular_momentum("bessel_l", bessel_l)?;
    validate_nonnegative_angular_momentum("multipole_l", multipole_l)?;

    let Some(ls_lb_factor) = xsph_ls_lb_factor(bessel_l, multipole_l)? else {
        return Ok(XsphRelativisticMultipoleFactors {
            p_q_prime: Complex::new(0.0, 0.0),
            q_p_prime: Complex::new(0.0, 0.0),
        });
    };

    let j2 = doubled_j_from_kappa("kappa", kappa)?;
    validate_cwig3j_doubled_argument("kappa", kappa, j2)?;
    let parity = if kappa > 0 { -1 } else { 1 };
    let j2_prime = doubled_j_from_kappa("kappa_prime", kappa_prime)?;
    validate_cwig3j_doubled_argument("kappa_prime", kappa_prime, j2_prime)?;
    let parity_prime = if kappa_prime > 0 { -1 } else { 1 };

    let p_q_prime = xsph_relativistic_multipole_component(
        ls_lb_factor,
        (j2 - parity) / 2,
        (j2_prime + parity_prime) / 2,
        bessel_l,
        multipole_l,
        j2,
        j2_prime,
        1.0,
    )?;
    let q_p_prime = xsph_relativistic_multipole_component(
        ls_lb_factor,
        (j2 + parity) / 2,
        (j2_prime - parity_prime) / 2,
        bessel_l,
        multipole_l,
        j2,
        j2_prime,
        -1.0,
    )?;

    Ok(XsphRelativisticMultipoleFactors {
        p_q_prime,
        q_p_prime,
    })
}

/// Port of FEFF `XSPH/bcoefjas.f90`.
///
/// Builds the two spin-component NRIXS transition weights `hbmat(0:1, 1:indmax)`
/// for a single doubled initial magnetic quantum number. The returned array is
/// Fortran-order with shape `(2, active_len)`, matching FEFF's spin-first
/// storage.
#[allow(clippy::too_many_arguments)]
pub fn xsph_nrixs_transition_weights(
    initial_kappa: i32,
    initial_mj2: i32,
    lmax: usize,
    jmax: i32,
    ljmax: i32,
    lgind: ArrayView1<'_, i32>,
    ljind: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<Array2<Real>, XsphError> {
    if initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len("lgind", lgind.len(), active_len)?;
    validate_active_len("ljind", ljind.len(), active_len)?;
    if jmax < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "jmax",
            index: 0,
            value: jmax,
        });
    }
    validate_cwig3j_doubled_argument("jmax", jmax, jmax)?;

    let lmax_i32 = usize_to_i32("lmax", lmax)?;
    let doubled_lmax = lmax_i32.checked_mul(2).ok_or(XsphError::SizeOutOfRange {
        name: "lmax",
        value: lmax,
    })?;
    let abs_ljmax = ljmax.checked_abs().ok_or(XsphError::IntegerOutOfRange {
        name: "ljmax",
        value: ljmax,
    })?;
    validate_cwig3j_integer_argument("ljmax", abs_ljmax)?;
    let jinit = doubled_j_from_kappa("initial_kappa", initial_kappa)?;
    validate_cwig3j_doubled_argument("initial_kappa", initial_kappa, jinit)?;
    let abs_initial_mj2 = initial_mj2
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_mj2",
            value: initial_mj2,
        })?;
    let initial_parity = if initial_kappa > 0 { -1 } else { 1 };

    let mut final_j2 = Vec::new();
    for lj in 0..=abs_ljmax {
        let lower = (2 * lj - jinit).abs().max(1);
        let upper = (2 * lj + jinit).min(jmax);
        let mut jfin = lower;
        while jfin <= upper {
            let final_parity = if (jinit + jfin + 2 * lj).rem_euclid(4) == 0 {
                -initial_parity
            } else {
                initial_parity
            };
            let final_l2 = if final_parity > 0 { jfin - 1 } else { jfin + 1 };
            if final_l2 <= doubled_lmax {
                final_j2.push(jfin);
            }
            jfin += 2;
        }
    }
    if final_j2.len() < active_len {
        return Err(XsphError::InsufficientGeneratedStates {
            required: active_len,
            generated: final_j2.len(),
        });
    }

    let mut weights = Array2::<Real>::zeros((2, active_len).f());
    for index in 0..active_len {
        let jfin = final_j2[index];
        let lj = validate_indexed_angular_momentum("ljind", index, ljind[index])?;
        let lg = validate_indexed_angular_momentum("lgind", index, lgind[index])?
            .checked_mul(2)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "lgind",
                value: lgind[index],
            })?;
        validate_cwig3j_doubled_argument("jfin", jfin, jfin)?;
        validate_cwig3j_integer_argument("ljind", lj)?;
        validate_cwig3j_doubled_argument("lgind", lgind[index], lg)?;

        let mut simple_3j = if abs_initial_mj2 <= jfin {
            wigner_3j(jinit, 2 * lj, jfin, -initial_mj2, 0, 2)?
        } else {
            0.0
        };
        if (i64::from(initial_mj2) + 1).rem_euclid(4) != 0 {
            simple_3j = -simple_3j;
        }

        for spin_index in 0..=1 {
            let mut ls_to_j = 0.0;
            if abs_initial_mj2 <= jfin && abs_initial_mj2 - 1 <= doubled_lmax {
                let spin_mj2 = 2 * usize_to_i32("spin_index", spin_index)? - 1;
                let magnetic_l2 =
                    initial_mj2
                        .checked_sub(spin_mj2)
                        .ok_or(XsphError::IntegerOutOfRange {
                            name: "initial_mj2",
                            value: initial_mj2,
                        })?;
                ls_to_j = wigner_3j(lg, 1, jfin, magnetic_l2, spin_mj2, 2)?;
                if (i64::from(lg) - 1 + i64::from(initial_mj2)).rem_euclid(4) != 0 {
                    ls_to_j = -ls_to_j;
                }
                ls_to_j *= (f64::from(jfin) + 1.0).sqrt();
            }
            weights[(spin_index, index)] = ls_to_j * simple_3j;
        }
    }

    Ok(weights)
}

/// Port of FEFF `XSPH/specupdlg.f90`.
///
/// Updates the NRIXS angular-decomposition spectrum buckets `xseclg(0:ljmax)`
/// in place for one shared calculation and spin component. The transition
/// weights use compact magnetic columns: `mjinit = -jinit, -jinit+2, ..., jinit`
/// maps to `(mjinit + jinit) / 2`.
pub fn xsph_update_nrixs_lg_spectrum(
    input: XsphLgSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
) -> Result<(), XsphError> {
    if input.calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex {
            calculation_index: input.calculation_index,
        });
    }
    if input.spin_index > 1 {
        return Err(XsphError::InvalidSpinIndex {
            spin_index: input.spin_index,
        });
    }
    if input.initial_j2 < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "initial_j2",
            index: 0,
            value: input.initial_j2,
        });
    }
    validate_cwig3j_doubled_argument("initial_j2", input.initial_j2, input.initial_j2)?;
    validate_active_len("index_map", input.index_map.len(), input.active_len)?;
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    validate_active_len("final_lj", input.final_lj.len(), input.active_len)?;
    let channel_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "radial_integrals",
        input.radial_integrals.len(),
        channel_count,
    )?;
    validate_active_len("spectrum", spectrum.len(), channel_count)?;

    let magnetic_count = usize::try_from(input.initial_j2)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?;
    let required_weights = [2, input.active_len, magnetic_count];
    let weight_shape = input.transition_weights.shape();
    let actual_weights = [weight_shape[0], weight_shape[1], weight_shape[2]];
    if actual_weights
        .iter()
        .zip(required_weights.iter())
        .any(|(actual, required)| actual < required)
    {
        return Err(XsphError::ShapeTooSmall {
            name: "transition_weights",
            required: required_weights,
            actual: actual_weights,
        });
    }

    let q_count = input.q_weights.len();
    validate_q_inputs(input.q_weights, input.q_cosines, q_count)?;
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_count = channel_count;
    let mut legendre_by_pair = vec![0.0; q_count * q_count * legendre_count];
    for iq in 0..q_count {
        for iqq in 0..q_count {
            let cosine = input.q_cosines[(iq, iqq)];
            validate_finite_real("q_cosines", cosine)?;
            let offset = (iq * q_count + iqq) * legendre_count;
            legendre_polynomials_into(
                cosine,
                &mut legendre_by_pair[offset..offset + legendre_count],
            );
        }
    }

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_indexed_angular_momentum("final_lj", index, input.final_lj[index])?;
        let final_lj = usize::try_from(final_lj).map_err(|_| XsphError::IntegerOutOfRange {
            name: "final_lj",
            value: input.final_lj[index],
        })?;
        if final_lj > input.ljmax {
            return Err(XsphError::AngularMomentumOutOfRange {
                angular_momentum: final_lj,
                ljmax: input.ljmax,
            });
        }
        let orbital_l =
            validate_indexed_angular_momentum("orbital_l", index, input.orbital_l[index])?;
        let orbital_l = usize::try_from(orbital_l).map_err(|_| XsphError::IntegerOutOfRange {
            name: "orbital_l",
            value: input.orbital_l[index],
        })?;
        if orbital_l > input.ljmax {
            continue;
        }

        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;
        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * legendre_count + final_lj];
            let radial = input.radial_integrals[final_lj];
            let amplitude = match input.mode {
                XsphSpectrumUpdateMode::Regular => {
                    -Complex::new(0.0, 1.0) * radial * radial * legendre
                }
                XsphSpectrumUpdateMode::Irregular => radial * legendre,
            };
            spectrum[orbital_l] -= amplitude * trace * q_weights[iq] * q_weights[iqq];
        }
    }

    Ok(())
}

fn validate_active_len(
    name: &'static str,
    actual: usize,
    active_len: usize,
) -> Result<(), XsphError> {
    if active_len == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    if actual < active_len {
        return Err(XsphError::LengthTooShort {
            name,
            required: active_len,
            actual,
        });
    }
    Ok(())
}

fn validate_final_lj(final_lj: ArrayView1<'_, i32>, active_len: usize) -> Result<(), XsphError> {
    for index in 0..active_len {
        let value = final_lj[index];
        if value < 0 {
            return Err(XsphError::NegativeAngularMomentum {
                name: "final_lj",
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_finite_real(name: &'static str, value: Real) -> Result<(), XsphError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(XsphError::NonFiniteScalar { name, value })
    }
}

fn validate_nonnegative_angular_momentum(name: &'static str, value: i32) -> Result<(), XsphError> {
    if value < 0 {
        Err(XsphError::NegativeAngularMomentum {
            name,
            index: 0,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_indexed_angular_momentum(
    name: &'static str,
    index: usize,
    value: i32,
) -> Result<i32, XsphError> {
    if value < 0 {
        Err(XsphError::NegativeAngularMomentum { name, index, value })
    } else {
        Ok(value)
    }
}

fn usize_to_i32(name: &'static str, value: usize) -> Result<i32, XsphError> {
    i32::try_from(value).map_err(|_| XsphError::SizeOutOfRange { name, value })
}

fn validate_q_inputs(
    q_weights: ArrayView1<'_, Complex>,
    q_cosines: ArrayView2<'_, Real>,
    q_count: usize,
) -> Result<(), XsphError> {
    let shape = q_cosines.shape();
    let actual = [shape[0], shape[1]];
    let required = [q_count, q_count];
    if actual[0] < required[0] || actual[1] < required[1] {
        return Err(XsphError::MatrixTooSmall {
            name: "q_cosines",
            required,
            actual,
        });
    }
    for (index, &weight) in q_weights.iter().enumerate() {
        validate_finite_complex("q_weights", index, weight)?;
    }
    Ok(())
}

fn validate_finite_complex(
    name: &'static str,
    index: usize,
    value: Complex,
) -> Result<(), XsphError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(XsphError::NonFiniteComplex {
            name,
            index,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn xsph_effective_q_weights(
    q_weights: ArrayView1<'_, Complex>,
    mix_dff: bool,
) -> Result<Vec<Complex>, XsphError> {
    q_weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| {
            let effective = if mix_dff { weight } else { weight.sqrt() };
            validate_finite_complex("effective_q_weight", index, effective)?;
            Ok(effective)
        })
        .collect()
}

fn xsph_q_pairs(
    mix_dff: bool,
    mdff_mode: i32,
    q_count: usize,
) -> Result<Vec<(usize, usize)>, XsphError> {
    if !mix_dff {
        return Ok((0..q_count).map(|index| (index, index)).collect());
    }
    match mdff_mode {
        1 => Ok((0..q_count)
            .flat_map(|iq| (0..q_count).map(move |iqq| (iq, iqq)))
            .collect()),
        2 if q_count >= 2 => Ok(vec![(0, 1)]),
        2 => Err(XsphError::MatrixTooSmall {
            name: "q_weights",
            required: [2, 1],
            actual: [q_count, 1],
        }),
        _ => Err(XsphError::InvalidMdffMode { mdff_mode }),
    }
}

fn xsph_transition_trace(
    transition_weights: ArrayView3<'_, Real>,
    spin_index: usize,
    state_index: usize,
    initial_j2: i32,
) -> Result<Real, XsphError> {
    let mut trace = 0.0;
    let mut magnetic_j2 = -initial_j2;
    while magnetic_j2 <= initial_j2 {
        let magnetic_index = usize::try_from((magnetic_j2 + initial_j2) / 2).map_err(|_| {
            XsphError::IntegerOutOfRange {
                name: "initial_j2",
                value: initial_j2,
            }
        })?;
        let value = transition_weights[(spin_index, state_index, magnetic_index)];
        validate_finite_real("transition_weights", value)?;
        trace += value * value;
        magnetic_j2 += 2;
    }
    Ok(trace)
}

fn doubled_j_from_kappa(name: &'static str, kappa: i32) -> Result<i32, XsphError> {
    let abs_kappa = kappa
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange { name, value: kappa })?;
    abs_kappa
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(XsphError::IntegerOutOfRange { name, value: kappa })
}

fn validate_cwig3j_doubled_argument(
    name: &'static str,
    original_value: i32,
    doubled_value: i32,
) -> Result<(), XsphError> {
    if doubled_value <= CWIG3J_MAX_DOUBLED_ARGUMENT {
        Ok(())
    } else {
        Err(XsphError::IntegerOutOfRange {
            name,
            value: original_value,
        })
    }
}

fn validate_cwig3j_integer_argument(name: &'static str, value: i32) -> Result<(), XsphError> {
    if value <= CWIG3J_MAX_DOUBLED_ARGUMENT / 2 {
        Ok(())
    } else {
        Err(XsphError::IntegerOutOfRange { name, value })
    }
}

fn xsph_ls_lb_factor(bessel_l: i32, multipole_l: i32) -> Result<Option<Complex>, XsphError> {
    let real_factor = if multipole_l > 0 && bessel_l == multipole_l - 1 {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        let value = f64::from(2 * multipole_l - 1) * f64::from(multipole_l + 1) / 2.0;
        value.sqrt()
    } else if bessel_l > 0 && bessel_l - 1 == multipole_l {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        let value = f64::from(2 * multipole_l + 3) * f64::from(multipole_l) / 2.0;
        value.sqrt()
    } else if bessel_l == multipole_l {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        f64::from(2 * multipole_l + 1) / 2.0_f64.sqrt()
    } else {
        return Ok(None);
    };
    Ok(Some(imaginary_unit_power(bessel_l) * real_factor))
}

#[allow(clippy::too_many_arguments)]
fn xsph_relativistic_multipole_component(
    ls_lb_factor: Complex,
    lambda: i32,
    lambda_prime: i32,
    bessel_l: i32,
    multipole_l: i32,
    j2: i32,
    j2_prime: i32,
    imaginary_sign: Real,
) -> Result<Complex, XsphError> {
    let nine_j = xsph_nine_j(lambda, lambda_prime, bessel_l, j2, j2_prime, multipole_l);
    let three_j = wigner_3j(lambda, bessel_l, lambda_prime, 0, 0, 1)?;
    let angular_weight = (6.0
        * (f64::from(j2) + 1.0)
        * (f64::from(j2_prime) + 1.0)
        * f64::from(2 * multipole_l + 1)
        * f64::from(2 * lambda + 1)
        * f64::from(2 * lambda_prime + 1))
    .sqrt();
    Ok(ls_lb_factor
        * (nine_j * three_j * alternating_sign(lambda) * angular_weight)
        * Complex::new(0.0, imaginary_sign))
}

fn xsph_nine_j(
    lambda: i32,
    lambda_prime: i32,
    bessel_l: i32,
    j2: i32,
    j2_prime: i32,
    multipole_l: i32,
) -> Real {
    if bessel_l > multipole_l {
        -f64::from(bessel_l + multipole_l + 1)
            * xsph_six_j(1, 2, 2 * multipole_l, bessel_l + multipole_l, 2 * bessel_l)
            * xsph_six_j(
                2 * multipole_l,
                bessel_l + multipole_l,
                2 * lambda_prime,
                j2_prime,
                j2,
            )
            * xsph_six_j(
                bessel_l + multipole_l,
                2 * bessel_l,
                2 * lambda,
                j2,
                2 * lambda_prime,
            )
    } else if bessel_l < multipole_l {
        -f64::from(bessel_l + multipole_l + 1)
            * xsph_six_j(1, 2, 2 * multipole_l, bessel_l + multipole_l, 2 * bessel_l)
            * xsph_six_j(
                bessel_l + multipole_l,
                2 * multipole_l,
                j2_prime,
                2 * lambda_prime,
                j2,
            )
            * xsph_six_j(
                2 * bessel_l,
                bessel_l + multipole_l,
                j2,
                2 * lambda,
                2 * lambda_prime,
            )
    } else {
        let first_term = -f64::from(2 * bessel_l + 2)
            * xsph_six_j(1, 2, 2 * multipole_l, 2 * multipole_l + 1, 2 * multipole_l)
            * xsph_six_j(
                2 * multipole_l,
                2 * multipole_l + 1,
                2 * lambda_prime,
                j2_prime,
                j2,
            )
            * xsph_six_j(
                2 * multipole_l,
                2 * multipole_l + 1,
                j2,
                2 * lambda,
                2 * lambda_prime,
            );
        if bessel_l == 0 {
            first_term
        } else {
            first_term
                - f64::from(2 * bessel_l)
                    * xsph_six_j(1, 2, 2 * multipole_l, 2 * multipole_l - 1, 2 * multipole_l)
                    * xsph_six_j(
                        2 * multipole_l - 1,
                        2 * multipole_l,
                        j2_prime,
                        2 * lambda_prime,
                        j2,
                    )
                    * xsph_six_j(
                        2 * multipole_l - 1,
                        2 * multipole_l,
                        2 * lambda,
                        j2,
                        2 * lambda_prime,
                    )
        }
    }
}

fn xsph_six_j(j1: i32, j2: i32, j3: i32, j4: i32, j5: i32) -> Real {
    if j2 != j1 + 1 {
        return 0.0;
    }
    if j4 == j3 + 1 {
        let g2 = j5 - 1;
        if g2 < (j1 - j3).abs() || g2 > j1 + j3 {
            return 0.0;
        }
        let value = (1.0 + f64::from(g2 + j1 - j3) / 2.0) * (1.0 + f64::from(g2 - j1 + j3) / 2.0)
            / f64::from(j1 + 1)
            / f64::from(j1 + 2)
            / f64::from(j3 + 1)
            / f64::from(j3 + 2);
        value.sqrt() * alternating_sign(nint(1.0 + f64::from(g2 + j1 + j3) / 2.0))
    } else if j3 == j4 + 1 {
        let g2 = j5;
        if g2 < (j1 - j4).abs() || g2 > j1 + j4 {
            return 0.0;
        }
        let value = (1.0 - f64::from(g2 - j1 - j4) / 2.0) * (2.0 + f64::from(g2 + j1 + j4) / 2.0)
            / f64::from(j1 + 1)
            / f64::from(j1 + 2)
            / f64::from(j4 + 1)
            / f64::from(j4 + 2);
        value.sqrt() * alternating_sign(nint(1.0 + f64::from(g2 + j1 + j4) / 2.0))
    } else {
        0.0
    }
}

fn nint(value: Real) -> i32 {
    value.round() as i32
}

fn alternating_sign(exponent: i32) -> Real {
    if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array3, ShapeBuilder, arr1, arr2};

    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    #[test]
    fn xsph_minimize_calculations_matches_feff_reference() -> Result<(), XsphError> {
        let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
        let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
        let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

        let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 8)?;

        assert_eq!(plan.max_lj, 6);
        assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2, 4, -3, -1]));
        assert_eq!(
            plan.calculations,
            arr2(&[[2, 5, 1], [4, 4, 2], [-3, 6, 1], [5, 0, 0]])
        );
        Ok(())
    }

    #[test]
    fn xsph_minimize_calculations_honors_active_prefix() -> Result<(), XsphError> {
        let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
        let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
        let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

        let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 5)?;

        assert_eq!(plan.max_lj, 5);
        assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2]));
        assert_eq!(plan.calculations, arr2(&[[2, 5, 1], [4, 4, 2], [-3, 3, 1]]));
        Ok(())
    }

    #[test]
    fn xsph_lj_needed_flags_match_feff_reference() -> Result<(), XsphError> {
        let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);
        let index_map = arr1(&[1, 2, -1, 3, -2, 4, -3, -1]);

        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 1)?,
            arr1(&[0, 1, 1, 0, 0, 1, 0])
        );
        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 2)?,
            arr1(&[0, 1, 0, 0, 1, 0, 0])
        );
        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 3)?,
            arr1(&[0, 0, 0, 1, 0, 0, 1])
        );
        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 4)?,
            arr1(&[1, 0, 0, 0, 0, 0, 0])
        );
        Ok(())
    }

    #[test]
    fn xsph_q_bessel_table_matches_feff_reference() -> Result<(), XsphError> {
        let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
        let table = xsph_q_bessel_table(0.35, radii.view(), 4)?;

        assert_eq!(table.shape(), &[4, 5]);
        assert_eq!(table.strides(), &[1, 4]);
        let expected = arr2(&[
            [
                9.997_958_458_381_769e-1,
                1.166_523_756_252_462e-2,
                8.165_952_107_648_562e-5,
                4.083_055_447_551_5e-7,
                1.587_874_544_380_937_5e-9,
            ],
            [
                9.797_080_213_012_896e-1,
                1.152_437_384_397_447_3e-1,
                8.095_451_039_379_387e-3,
                4.055_621_228_179_726_3e-4,
                1.579_141_698_006_595_3e-5,
            ],
            [
                8.261_173_577_085_878e-1,
                3.129_012_474_446_291e-1,
                6.788_620_641_892_411e-2,
                1.036_640_216_929_531_6e-2,
                1.223_141_376_378_009_1e-3,
            ],
            [
                9.385_522_838_839_835e-2,
                -9.429_243_227_927_261e-2,
                -1.342_662_707_938_009e-1,
                -1.612_046_859_156_612_8e-3,
                1.326_542_239_346_443e-1,
            ],
        ]);
        for ((row, column), &expected_value) in expected.indexed_iter() {
            assert_close(table[(row, column)], expected_value);
        }
        Ok(())
    }

    #[test]
    fn xsph_q_bessel_table_applies_feff_large_argument_cutoff() -> Result<(), XsphError> {
        let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
        let table = xsph_q_bessel_table(1.0e8, radii.view(), 4)?;

        let expected_first_row = [
            4.205_477_931_907_825e-8,
            9.072_704_282_365_188e-8,
            -4.205_475_210_096_54e-8,
            -9.072_706_385_102_794e-8,
            4.205_468_859_202_071e-8,
        ];
        for (column, &expected_value) in expected_first_row.iter().enumerate() {
            assert_close(table[(0, column)], expected_value);
        }
        for row in 1..4 {
            for column in 0..5 {
                assert_close(table[(row, column)], 0.0);
            }
        }
        Ok(())
    }

    #[test]
    fn xsph_longitudinal_multipole_factor_matches_feff_reference() -> Result<(), XsphError> {
        let cases = [
            (-1, -1, 0, -std::f64::consts::SQRT_2),
            (-1, 1, 1, 2.449_489_742_783_178),
            (1, -1, 1, 2.449_489_742_783_178),
            (-2, 1, 1, 0.0),
            (2, -1, 2, -4.472_135_954_999_58),
            (-3, 2, 3, 0.0),
            (3, -2, 2, 2.927_700_218_845_598),
            (-2, -2, 5, 0.0),
        ];

        for (kappa, kappa_prime, multipole_l, expected) in cases {
            let value = xsph_longitudinal_multipole_factor(kappa, kappa_prime, multipole_l)?;
            assert_close(value.re, expected);
            assert_close(value.im, 0.0);
        }
        Ok(())
    }

    #[test]
    fn xsph_relativistic_multipole_factors_match_feff_reference() -> Result<(), XsphError> {
        let cases = [
            (-1, -1, 0, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            (
                1,
                -1,
                0,
                1,
                Complex::new(0.0, -8.164_965_809_277_261e-1),
                Complex::new(0.0, -2.449_489_742_783_178),
            ),
            (
                -2,
                -1,
                0,
                1,
                Complex::new(0.0, -2.309_401_076_758_503_4),
                Complex::new(0.0, 0.0),
            ),
            (2, -1, 2, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            (
                -2,
                1,
                1,
                2,
                Complex::new(-3.872_983_346_207_417_5, 0.0),
                Complex::new(-7.745_966_692_414_837e-1, 0.0),
            ),
            (
                3,
                -2,
                1,
                1,
                Complex::new(2.323_790_007_724_448_4, 0.0),
                Complex::new(2.323_790_007_724_45, 0.0),
            ),
            (
                -3,
                2,
                1,
                2,
                Complex::new(-3.549_647_869_859_77, 0.0),
                Complex::new(-1.521_277_658_511_329_2, 0.0),
            ),
            (2, -3, 3, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            (
                1,
                1,
                1,
                1,
                Complex::new(-2.449_489_742_783_178, 0.0),
                Complex::new(-2.449_489_742_783_178, 0.0),
            ),
            (
                -2,
                -2,
                1,
                1,
                Complex::new(3.098_386_676_965_933_6, 0.0),
                Complex::new(3.098_386_676_965_934, 0.0),
            ),
        ];

        for (kappa, kappa_prime, bessel_l, multipole_l, expected_pq, expected_qp) in cases {
            let factors =
                xsph_relativistic_multipole_factors(kappa, kappa_prime, bessel_l, multipole_l)?;
            assert_close(factors.p_q_prime.re, expected_pq.re);
            assert_close(factors.p_q_prime.im, expected_pq.im);
            assert_close(factors.q_p_prime.re, expected_qp.re);
            assert_close(factors.q_p_prime.im, expected_qp.im);
        }
        Ok(())
    }

    #[test]
    fn xsph_relativistic_multipole_factors_return_zero_for_unmatched_orders()
    -> Result<(), XsphError> {
        let factors = xsph_relativistic_multipole_factors(-1, 1, 4, 1)?;

        assert_close(factors.p_q_prime.re, 0.0);
        assert_close(factors.p_q_prime.im, 0.0);
        assert_close(factors.q_p_prime.re, 0.0);
        assert_close(factors.q_p_prime.im, 0.0);
        Ok(())
    }

    #[test]
    fn xsph_nrixs_transition_weights_match_feff_reference() -> Result<(), XsphError> {
        let lgind = arr1(&[0, 1, 2, 1, 3, 2, 4]);
        let ljind = arr1(&[0, 1, 1, 2, 2, 3, 3]);
        let weights = xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, lgind.view(), ljind.view(), 7)?;
        assert_eq!(weights.shape(), &[2, 7]);
        assert_eq!(weights.strides(), &[1, 2]);
        let expected = arr2(&[
            [
                0.0,
                -3.333_333_333_333_333_7e-1,
                3.162_277_660_168_380_5e-1,
                1.825_741_858_350_554_4e-1,
                -2.390_457_218_668_785e-1,
                -1.690_308_509_457_032e-1,
                1.992_047_682_223_989_4e-1,
            ],
            [
                -7.071_067_811_865_477e-1,
                2.357_022_603_955_158_7e-1,
                -2.581_988_897_471_612_6e-1,
                2.581_988_897_471_612_6e-1,
                2.070_196_678_027_061_4e-1,
                -2.070_196_678_027_061_4e-1,
                -1.781_741_612_749_495_3e-1,
            ],
        ]);
        for ((spin, channel), &expected_value) in expected.indexed_iter() {
            assert_close(weights[(spin, channel)], expected_value);
        }

        let lgind = arr1(&[1, 2, 1, 3, 2, 4, 3, 4]);
        let ljind = arr1(&[0, 1, 1, 2, 2, 3, 3, 4]);
        let weights =
            xsph_nrixs_transition_weights(2, -1, 4, 11, 4, lgind.view(), ljind.view(), 8)?;
        let expected = arr2(&[
            [
                4.082_482_904_638_632_4e-1,
                0.0,
                -1.054_092_553_389_460_6e-1,
                7.824_607_964_359_512e-2,
                0.0,
                0.0,
                1.106_566_670_344_975_2e-1,
                -9.390_602_830_316_835e-2,
            ],
            [
                2.886_751_345_948_13e-1,
                0.0,
                -7.453_559_924_999_303e-2,
                -9.035_079_029_052_508e-2,
                0.0,
                0.0,
                -1.277_753_129_999_878_7e-1,
                1.049_901_313_914_518_7e-1,
            ],
        ]);
        for ((spin, channel), &expected_value) in expected.indexed_iter() {
            assert_close(weights[(spin, channel)], expected_value);
        }

        let lgind = arr1(&[0, 1, 2, 2, 3]);
        let ljind = arr1(&[0, 1, 2, 2, 3]);
        let weights = xsph_nrixs_transition_weights(-2, 3, 4, 9, 3, lgind.view(), ljind.view(), 5)?;
        let expected = arr2(&[
            [0.0, 0.0, 2.0e-1, -1.309_307_341_415_953e-1, 0.0],
            [
                0.0,
                0.0,
                -1.000_000_000_000_000_2e-1,
                -2.618_614_682_831_905e-1,
                0.0,
            ],
        ]);
        for ((spin, channel), &expected_value) in expected.indexed_iter() {
            assert_close(weights[(spin, channel)], expected_value);
        }
        Ok(())
    }

    #[test]
    fn xsph_update_nrixs_lg_spectrum_matches_feff_reference() -> Result<(), XsphError> {
        let index_map = arr1(&[1, -1, 2, 1, -2]);
        let orbital_l = arr1(&[0, 1, 2, 3, 4]);
        let final_lj = arr1(&[0, 1, 2, 3, 1]);
        let radial_integrals = arr1(&[
            Complex::new(0.12, -0.03),
            Complex::new(-0.08, 0.19),
            Complex::new(0.31, 0.07),
            Complex::new(-0.22, -0.11),
        ]);
        let q_cosines = arr2(&[[0.25, -0.35], [0.60, -0.40]]);
        let mut transition_weights = Array3::<Real>::zeros((2, 5, 4).f());
        for state in 0..5 {
            let state_feff = state as Real + 1.0;
            for spin in 0..2 {
                let spin_feff = spin as Real;
                for (magnetic_index, magnetic_j2) in [-3, -1, 1, 3].iter().enumerate() {
                    let magnetic = Real::from(*magnetic_j2);
                    transition_weights[(spin, state, magnetic_index)] = 0.05 * state_feff
                        + 0.11 * spin_feff
                        + 0.017 * magnetic
                        + 0.003 * state_feff * magnetic;
                }
            }
        }

        let q_weights = arr1(&[Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)]);
        let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
        xsph_update_nrixs_lg_spectrum(
            XsphLgSpectrumUpdateInput {
                calculation_index: 1,
                spin_index: 1,
                index_map: index_map.view(),
                orbital_l: orbital_l.view(),
                final_lj: final_lj.view(),
                initial_j2: 3,
                transition_weights: transition_weights.view(),
                radial_integrals: radial_integrals.view(),
                q_weights: q_weights.view(),
                q_cosines: q_cosines.view(),
                mix_dff: false,
                mdff_mode: 0,
                ljmax: 3,
                active_len: 5,
                mode: XsphSpectrumUpdateMode::Regular,
            },
            spectrum.view_mut(),
        )?;
        let expected = [
            Complex::new(1.100_552_32e-2, -1.768_391_84e-2),
            Complex::new(1.004_038_768e-2, -2.057_271_974e-2),
            Complex::new(1.0e-2, -2.0e-2),
            Complex::new(1.156_784_538_79e-2, -2.277_795_550_092_5e-2),
        ];
        for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }

        let q_weights = arr1(&[Complex::new(0.7, -0.2), Complex::new(-0.15, 0.45)]);
        let mut spectrum = Array1::from_elem(4, Complex::new(-0.03, 0.04));
        xsph_update_nrixs_lg_spectrum(
            XsphLgSpectrumUpdateInput {
                calculation_index: 1,
                spin_index: 0,
                index_map: index_map.view(),
                orbital_l: orbital_l.view(),
                final_lj: final_lj.view(),
                initial_j2: 3,
                transition_weights: transition_weights.view(),
                radial_integrals: radial_integrals.view(),
                q_weights: q_weights.view(),
                q_cosines: q_cosines.view(),
                mix_dff: true,
                mdff_mode: 1,
                ljmax: 3,
                active_len: 5,
                mode: XsphSpectrumUpdateMode::Irregular,
            },
            spectrum.view_mut(),
        )?;
        let expected = [
            Complex::new(-3.066_69e-2, 3.953_56e-2),
            Complex::new(-2.859_349_665e-2, 3.854_721_595e-2),
            Complex::new(-3.0e-2, 4.0e-2),
            Complex::new(-4.005_742_490_156_249e-2, 3.762_661_973_593_751e-2),
        ];
        for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }

        let q_weights = arr1(&[Complex::new(0.25, 0.1), Complex::new(0.3, -0.35)]);
        let mut spectrum = Array1::from_elem(4, Complex::new(0.02, 0.01));
        xsph_update_nrixs_lg_spectrum(
            XsphLgSpectrumUpdateInput {
                calculation_index: 2,
                spin_index: 1,
                index_map: index_map.view(),
                orbital_l: orbital_l.view(),
                final_lj: final_lj.view(),
                initial_j2: 3,
                transition_weights: transition_weights.view(),
                radial_integrals: radial_integrals.view(),
                q_weights: q_weights.view(),
                q_cosines: q_cosines.view(),
                mix_dff: true,
                mdff_mode: 2,
                ljmax: 3,
                active_len: 5,
                mode: XsphSpectrumUpdateMode::Regular,
            },
            spectrum.view_mut(),
        )?;
        let expected = [
            Complex::new(2.0e-2, 1.0e-2),
            Complex::new(2.0e-2, 1.0e-2),
            Complex::new(1.995_779_884_1e-2, 8.875_159_533_25e-3),
            Complex::new(2.0e-2, 1.0e-2),
        ];
        for (&actual, &expected) in spectrum.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn xsph_planning_helpers_reject_invalid_inputs() {
        let kind = arr1(&[2]);
        let orbital_l = arr1(&[1]);
        let final_lj = arr1(&[2]);

        assert!(matches!(
            xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 0),
            Err(XsphError::EmptyIndexSet)
        ));
        assert!(matches!(
            xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 2),
            Err(XsphError::LengthTooShort { name: "kind", .. })
        ));

        let bad_lj = arr1(&[-1]);
        assert!(matches!(
            xsph_minimize_calculations(kind.view(), orbital_l.view(), bad_lj.view(), 1),
            Err(XsphError::NegativeAngularMomentum { .. })
        ));

        let index_map = arr1(&[1]);
        assert!(matches!(
            xsph_lj_needed_flags(1, final_lj.view(), index_map.view(), 1, 1),
            Err(XsphError::AngularMomentumOutOfRange { .. })
        ));
        assert!(matches!(
            xsph_lj_needed_flags(2, final_lj.view(), index_map.view(), 1, 0),
            Err(XsphError::NonPositiveCalculationIndex { .. })
        ));

        let overflow_map = arr1(&[i32::MIN]);
        assert!(matches!(
            xsph_lj_needed_flags(2, final_lj.view(), overflow_map.view(), 1, 1),
            Err(XsphError::IndexMapOverflow { .. })
        ));

        let radii = arr1(&[1.0]);
        assert!(matches!(
            xsph_q_bessel_table(Real::NAN, radii.view(), 4),
            Err(XsphError::NonFiniteScalar { name: "qtrans", .. })
        ));
        let bad_radii = arr1(&[Real::INFINITY]);
        assert!(matches!(
            xsph_q_bessel_table(1.0, bad_radii.view(), 4),
            Err(XsphError::NonFiniteScalar { name: "radius", .. })
        ));
        assert!(matches!(
            xsph_q_bessel_table(1.0, radii.view(), 40),
            Err(XsphError::AngularMomentumOutOfRange {
                angular_momentum: 40,
                ljmax: 39,
            })
        ));
        assert!(matches!(
            xsph_q_bessel_table(0.0, radii.view(), 4),
            Err(XsphError::Bessel(
                BesselError::NonPositiveRealArgument { .. }
            ))
        ));

        assert!(matches!(
            xsph_longitudinal_multipole_factor(0, 1, 1),
            Err(XsphError::ZeroKappa)
        ));
        assert!(matches!(
            xsph_longitudinal_multipole_factor(1, 1, -1),
            Err(XsphError::NegativeAngularMomentum {
                name: "multipole_l",
                ..
            })
        ));
        assert!(matches!(
            xsph_longitudinal_multipole_factor(i32::MIN, 1, 1),
            Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
        ));
        assert!(matches!(
            xsph_longitudinal_multipole_factor(60, -60, 1),
            Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
        ));
        assert!(matches!(
            xsph_relativistic_multipole_factors(0, 1, 0, 1),
            Err(XsphError::ZeroKappa)
        ));
        assert!(matches!(
            xsph_relativistic_multipole_factors(1, 1, -1, 1),
            Err(XsphError::NegativeAngularMomentum {
                name: "bessel_l",
                ..
            })
        ));
        assert!(matches!(
            xsph_relativistic_multipole_factors(1, 1, 0, -1),
            Err(XsphError::NegativeAngularMomentum {
                name: "multipole_l",
                ..
            })
        ));
        assert!(matches!(
            xsph_relativistic_multipole_factors(1, 1, 59, 59),
            Err(XsphError::IntegerOutOfRange {
                name: "bessel_l",
                ..
            })
        ));

        let lgind = arr1(&[0]);
        let ljind = arr1(&[0]);
        assert!(matches!(
            xsph_nrixs_transition_weights(0, 1, 4, 9, 3, lgind.view(), ljind.view(), 1),
            Err(XsphError::ZeroKappa)
        ));
        assert!(matches!(
            xsph_nrixs_transition_weights(-1, 1, 4, -1, 3, lgind.view(), ljind.view(), 1),
            Err(XsphError::NegativeAngularMomentum { name: "jmax", .. })
        ));
        assert!(matches!(
            xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, lgind.view(), ljind.view(), 2),
            Err(XsphError::LengthTooShort { name: "lgind", .. })
        ));
        let bad_lgind = arr1(&[-1]);
        assert!(matches!(
            xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, bad_lgind.view(), ljind.view(), 1),
            Err(XsphError::NegativeAngularMomentum { name: "lgind", .. })
        ));
        let two_lgind = arr1(&[0, 0]);
        let two_ljind = arr1(&[0, 0]);
        assert!(matches!(
            xsph_nrixs_transition_weights(-1, 1, 0, 1, 0, two_lgind.view(), two_ljind.view(), 2),
            Err(XsphError::InsufficientGeneratedStates { .. })
        ));
    }
}
