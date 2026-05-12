//! Angular-momentum normalization helpers.
//!
//! FEFF stores associated-Legendre normalization factors in `xnlm`; FMS uses
//! `xnlm(m,l)` while GENFMT carries the same values in a one-based table. The
//! helpers here compute the shared value
//! `sqrt((2l+1) * (l-m)! / (l+m)!)`.

use ndarray::{Array2, Array3, Array6, ShapeBuilder};

use crate::{Complex, ComplexVec, Real, RealVec};

// FEFF `ylm.f90` stores an unsuffixed PI literal in a double variable, so
// gfortran rounds it as single precision before widening.
#[allow(clippy::approx_constant)]
const YLM_PI: Real = 3.141_592_7_f32 as Real;

/// Error returned by angular normalization helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AngularError {
    /// Integer indices must fit in `u32` before conversion to `f64`.
    #[error("angular index {value} is too large for stable floating-point conversion")]
    IndexTooLarge { value: usize },
    /// FEFF angular helpers accept only integer (`1`) and half-integer (`2`) scales.
    #[error("invalid angular momentum scale {scale}; expected 1 or 2")]
    InvalidWignerScale { scale: i32 },
    /// A Wigner 3j argument did not divide evenly by the selected scale.
    #[error("Wigner 3j argument {argument} is not divisible by scale {scale}")]
    InvalidWignerParity { argument: i32, scale: i32 },
    /// FEFF's common `cwig3j` table is limited to factorial arguments up to 58.
    #[error("Wigner 3j factorial argument {argument} exceeds FEFF limit {limit}")]
    WignerFactorialOutOfRange { argument: i32, limit: i32 },
    /// The requested magnetic index does not fit the allocated table.
    #[error("magnetic index {magnetic} is outside table range for lmax {lmax}")]
    MagneticIndexOutOfRange { magnetic: isize, lmax: usize },
    /// FEFF Wigner rotations require a finite angle.
    #[error("Wigner rotation angle must be finite")]
    NonFiniteRotationAngle,
    /// FEFF `ylm` requires finite Cartesian vector components.
    #[error("spherical-harmonic vector components must be finite")]
    NonFiniteVector,
    /// FEFF transition matrices require finite polarization tensor entries.
    #[error("polarization tensor entry ({row}, {column}) must be finite")]
    NonFinitePolarizationTensor { row: isize, column: isize },
    /// FEFF spin-folding expects at least one compiled spin channel.
    #[error("invalid FEFF spin channel count {value}; expected at least 1")]
    InvalidSpinChannelCount { value: usize },
}

/// Spin-orbit Clebsch-Gordon tables used by FEFF's FMS and POT paths.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinOrbitCouplingTables {
    /// `j = l + 1/2` coefficients, indexed as `[l, m + m_offset, spin - 1]`.
    pub plus: Array3<Real>,
    /// `j = l - 1/2` coefficients, indexed as `[l, m + m_offset, spin - 1]`.
    pub minus: Array3<Real>,
    /// Offset added to signed `m` before indexing the second axis.
    pub m_offset: usize,
}

/// Inputs for FEFF `bcoef`, the transition B-matrix builder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionBMatrixInput {
    /// Maximum orbital momentum `lx`, used for the signed magnetic-index axes.
    pub lmax: usize,
    /// Initial-state relativistic kappa.
    pub initial_kappa: i32,
    /// FEFF polarization selector. `0` uses the orientational average branch;
    /// nonzero values use the full tensor branch.
    pub polarization: i32,
    /// Polarization tensor indexed as `[p + 1][p_prime + 1]` for `p=-1..=1`.
    pub polarization_tensor: [[Complex; 3]; 3],
    /// FEFF `le2` multipole selector.
    pub multipole: i32,
    /// Whether to trace the resulting matrix over orbital `m_l`.
    pub trace_orbital: bool,
    /// FEFF spin selector `ispin`.
    pub spin: i32,
    /// Compiled number of spin channels, FEFF `nspu`.
    pub spin_channels: usize,
    /// Angle between x-ray k-vector and spin vector.
    pub spin_vector_angle: Real,
}

/// FEFF `bcoef` output in Rust-owned storage.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionBMatrix {
    /// Final-state kappa indices for FEFF transition slots `1..=8`.
    pub kappa_indices: [i32; 8],
    /// Orbital angular momenta for FEFF transition slots `1..=8`.
    pub orbital_momenta: [i32; 8],
    /// `bmat(ml2, ms2, k2, ml1, ms1, k1)` in FEFF axis order.
    pub matrix: Array6<Complex>,
    /// Offset added to signed `m_l` before indexing `matrix` axes 0 and 3.
    pub l_offset: usize,
}

impl TransitionBMatrix {
    /// Return a matrix element using FEFF's signed magnetic indices and
    /// one-based transition slots.
    #[must_use]
    pub fn value(
        &self,
        ml2: isize,
        ms2: usize,
        transition2: usize,
        ml1: isize,
        ms1: usize,
        transition1: usize,
    ) -> Option<Complex> {
        if ms1 > 1 || ms2 > 1 || !(1..=8).contains(&transition1) || !(1..=8).contains(&transition2)
        {
            return None;
        }
        let ml2 = self.magnetic_index(ml2)?;
        let ml1 = self.magnetic_index(ml1)?;
        self.matrix
            .get([ml2, ms2, transition2 - 1, ml1, ms1, transition1 - 1])
            .copied()
    }

    fn magnetic_index(&self, magnetic: isize) -> Option<usize> {
        let offset = isize::try_from(self.l_offset).ok()?;
        let index = magnetic + offset;
        let len = offset.checked_mul(2)?.checked_add(1)?;
        if index < 0 || index >= len {
            None
        } else {
            usize::try_from(index).ok()
        }
    }
}

/// Compute ordinary Legendre polynomials `P_l(x)` for `l = 0..=lmax`.
///
/// This ports FEFF `cpl0`, which fills `pl0(l + 1)` by the three-term
/// recurrence. The returned vector uses zero-based Rust indexing, so `out[l]`
/// contains `P_l(x)`.
#[must_use]
pub fn legendre_polynomials(x: Real, lmax: usize) -> RealVec {
    let mut values = RealVec::zeros(lmax + 1);
    if let Some(slice) = values.as_slice_mut() {
        legendre_polynomials_into(x, slice);
    }
    values
}

/// Fill a slice with ordinary Legendre polynomials `P_l(x)`.
///
/// `values[0]` receives `P_0(x)`, `values[1]` receives `P_1(x)`, and so on.
/// An empty slice is accepted and left unchanged, matching FEFF's defensive
/// bounds checks around short `pl0` arrays.
pub fn legendre_polynomials_into(x: Real, values: &mut [Real]) {
    if values.is_empty() {
        return;
    }
    values[0] = 1.0;
    if values.len() < 2 {
        return;
    }
    values[1] = x;
    if values.len() < 3 {
        return;
    }

    for ell in 1..(values.len() - 1) {
        let ell_real = ell as Real;
        values[ell + 1] = ((2.0 * ell_real + 1.0) * x * values[ell] - ell_real * values[ell - 1])
            / (ell_real + 1.0);
    }
}

/// Return FEFF's associated-Legendre normalization factor for nonnegative `m`.
///
/// Values with `m > l` are zero, matching FEFF's explicitly zeroed `xnlm`
/// table outside the physically valid triangular region.
pub fn legendre_normalization(l: usize, m: usize) -> Result<Real, AngularError> {
    if m > l {
        return Ok(0.0);
    }

    let numerator = usize_to_real(2 * l + 1)?;
    let denominator = ((l - m + 1)..=(l + m))
        .map(usize_to_real)
        .try_fold(1.0, |accumulator, value| {
            value.map(|value| accumulator * value)
        })?;

    Ok((numerator / denominator).sqrt())
}

/// Build FEFF's `xnlm(m,l)` normalization table in Fortran order.
pub fn legendre_normalization_table(lmax: usize) -> Result<Array2<Real>, AngularError> {
    let mut table = Array2::zeros((lmax + 1, lmax + 1).f());
    for l in 0..=lmax {
        for m in 0..=l {
            table[[m, l]] = legendre_normalization(l, m)?;
        }
    }
    Ok(table)
}

/// Port of FEFF `ylm`: complex spherical harmonics for a Cartesian vector.
///
/// Values are returned in FEFF order:
/// `Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...`.
/// The input vector is normalized internally; the zero vector follows FEFF's
/// convention of using `theta = 0` and `phi = 0`.
pub fn spherical_harmonics(vector: [Real; 3], lmax: usize) -> Result<ComplexVec, AngularError> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(AngularError::NonFiniteVector);
    }

    let count = lmax
        .checked_add(1)
        .and_then(|size| size.checked_mul(size))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let mut values = vec![Complex::new(0.0, 0.0); count];
    let y00 = 1.0 / (4.0 * YLM_PI).sqrt();
    values[0] = Complex::new(y00, 0.0);
    if lmax == 0 {
        return Ok(ComplexVec::from_vec(values));
    }

    let abmax = vector[0].abs().max(vector[1].abs());
    let (cos_phi, sin_phi) = if abmax > 0.0 {
        let a = vector[0] / abmax;
        let b = vector[1] / abmax;
        let ab = (a * a + b * b).sqrt();
        (a / ab, b / ab)
    } else {
        (1.0, 0.0)
    };

    let abcmax = abmax.max(vector[2].abs());
    let (cos_theta, sin_theta) = if abcmax > 0.0 {
        let a = vector[0] / abcmax;
        let b = vector[1] / abcmax;
        let c = vector[2] / abcmax;
        let ab = a * a + b * b;
        let abc = (ab + c * c).sqrt();
        (c / abc, ab.sqrt() / abc)
    } else {
        (1.0, 0.0)
    };

    values[spherical_harmonic_index(1, 0)] = Complex::new(3.0_f64.sqrt() * y00 * cos_theta, 0.0);
    let temp = -(1.5_f64).sqrt() * y00 * sin_theta;
    values[spherical_harmonic_index(1, 1)] = Complex::new(temp * cos_phi, temp * sin_phi);
    values[spherical_harmonic_index(1, -1)] = -values[spherical_harmonic_index(1, 1)].conj();

    for l in 2..=lmax {
        let l_real = l as Real;
        let sign_l = alternating_sign_usize(l);
        let previous_diagonal = values[spherical_harmonic_index(l - 1, (l - 1) as isize)];

        let diagonal_factor = -((2.0 * l_real + 1.0) / (2.0 * l_real)).sqrt() * sin_theta;
        let diagonal = Complex::new(
            diagonal_factor * (cos_phi * previous_diagonal.re - sin_phi * previous_diagonal.im),
            diagonal_factor * (cos_phi * previous_diagonal.im + sin_phi * previous_diagonal.re),
        );
        values[spherical_harmonic_index(l, l as isize)] = diagonal;
        values[spherical_harmonic_index(l, -(l as isize))] = diagonal.conj() * sign_l;

        let near_diagonal = previous_diagonal * ((2.0 * l_real + 1.0).sqrt() * cos_theta);
        values[spherical_harmonic_index(l, (l - 1) as isize)] = near_diagonal;
        values[spherical_harmonic_index(l, -((l - 1) as isize))] =
            near_diagonal.conj() * alternating_sign_usize(l - 1);

        let first_factor = cos_theta * (4.0 * l_real * l_real - 1.0).sqrt();
        let second_factor_base = -((2.0 * l_real + 1.0) / (2.0 * l_real - 3.0)).sqrt();
        for m in (0..=(l - 2)).rev() {
            let m_real = m as Real;
            let denominator = ((l_real + m_real) * (l_real - m_real)).sqrt();
            let first_factor = first_factor / denominator;
            let second_factor = second_factor_base
                * ((l_real + m_real - 1.0) * (l_real - m_real - 1.0)).sqrt()
                / denominator;
            let harmonic = values[spherical_harmonic_index(l - 1, m as isize)] * first_factor
                + values[spherical_harmonic_index(l - 2, m as isize)] * second_factor;
            values[spherical_harmonic_index(l, m as isize)] = harmonic;
            values[spherical_harmonic_index(l, -(m as isize))] =
                harmonic.conj() * alternating_sign_usize(m);
        }
    }

    Ok(ComplexVec::from_vec(values))
}

/// Port of FEFF `bcoef`: build the energy-independent transition B matrix.
///
/// The returned matrix keeps FEFF's axis order
/// `bmat(ml2, ms2, k2, ml1, ms1, k1)`, with signed magnetic quantum numbers
/// shifted by `l_offset`. Transition slots are zero-based in the raw ndarray
/// but correspond to FEFF's slots `1..=8`.
pub fn transition_b_matrix(
    input: TransitionBMatrixInput,
) -> Result<TransitionBMatrix, AngularError> {
    if input.spin_channels == 0 {
        return Err(AngularError::InvalidSpinChannelCount {
            value: input.spin_channels,
        });
    }
    ensure_finite_polarization_tensor(&input.polarization_tensor)?;

    let lmax_i32 = usize_to_i32(input.lmax)?;
    let magnetic_len = input
        .lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AngularError::IndexTooLarge { value: input.lmax })?;
    let extended_magnetic_len = magnetic_len
        .checked_add(1)
        .ok_or(AngularError::IndexTooLarge { value: input.lmax })?;

    let mut jind = [0_i32; 8];
    let mut lind = [0_i32; 8];
    let mut kiind = [0_i32; 8];
    fill_transition_indices(
        input.initial_kappa,
        input.multipole,
        lmax_i32,
        &mut jind,
        &mut lind,
        &mut kiind,
    );

    let mut matrix = Array6::zeros((magnetic_len, 2, 8, magnetic_len, 2, 8).f());
    if input.polarization == 0 {
        fill_averaged_transition_matrix(input.multipole, lmax_i32, &lind, &mut matrix);
    } else {
        fill_polarized_transition_matrix(
            input,
            lmax_i32,
            magnetic_len,
            extended_magnetic_len,
            &jind,
            &lind,
            &mut matrix,
        )?;
    }

    if input.trace_orbital {
        trace_transition_matrix(lmax_i32, &lind, &mut matrix);
    }
    fold_transition_spin(
        input.spin,
        input.spin_channels,
        lmax_i32,
        &lind,
        &mut matrix,
    );

    Ok(TransitionBMatrix {
        kappa_indices: kiind,
        orbital_momenta: lind,
        matrix,
        l_offset: input.lmax,
    })
}

/// Port of FEFF `cwig3j`.
///
/// Inputs are scaled by `scale`: use `scale = 1` for integer angular momenta
/// and `scale = 2` for half-integers represented as doubled integers.
pub fn wigner_3j(
    j1: i32,
    j2: i32,
    j3: i32,
    m1: i32,
    m2: i32,
    scale: i32,
) -> Result<Real, AngularError> {
    const FACTORIAL_LIMIT: i32 = 58;

    if scale != 1 && scale != 2 {
        return Err(AngularError::InvalidWignerScale { scale });
    }

    let double_scale = scale + scale;
    let m3 = -m1 - m2;
    if m1.abs() + m2.abs() == 0 && (j1 + j2 + j3) % double_scale != 0 {
        return Ok(0.0);
    }

    let mut values = [
        j1 + j2 - j3,
        j2 + j3 - j1,
        j3 + j1 - j2,
        j1 + m1,
        j1 - m1,
        j2 + m2,
        j2 - m2,
        j3 + m3,
        j3 - m3,
        j1 + j2 + j3 + scale,
        j2 - j3 - m1,
        j1 - j3 + m2,
    ];

    for (index, value) in values.iter_mut().enumerate() {
        if index < 10 && *value < 0 {
            return Ok(0.0);
        }
        if *value % scale != 0 {
            return Err(AngularError::InvalidWignerParity {
                argument: *value,
                scale,
            });
        }
        *value /= scale;
        if *value > FACTORIAL_LIMIT {
            return Err(AngularError::WignerFactorialOutOfRange {
                argument: *value,
                limit: FACTORIAL_LIMIT,
            });
        }
    }

    let log_factorial = log_factorials(FACTORIAL_LIMIT)?;
    let k_min = values[10].max(values[11]).max(0);
    let k_max = values[0].min(values[4]).min(values[5]);
    if k_min > k_max {
        return Ok(0.0);
    }

    let mut sign = if k_min % 2 == 0 { 1.0 } else { -1.0 };
    let c = values[..9].iter().try_fold(
        -log_factorial_value(&log_factorial, values[9], FACTORIAL_LIMIT)?,
        |accumulator, value| {
            Ok::<_, AngularError>(
                accumulator + log_factorial_value(&log_factorial, *value, FACTORIAL_LIMIT)?,
            )
        },
    )? / 2.0;

    let mut coefficient = 0.0;
    for k in k_min..=k_max {
        let b = log_factorial_value(&log_factorial, k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, values[0] - k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, values[4] - k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, values[5] - k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, k - values[10], FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, k - values[11], FACTORIAL_LIMIT)?;
        coefficient += sign * (c - b).exp();
        sign = -sign;
    }

    if (j1 - j2 - m3) % double_scale != 0 {
        coefficient = -coefficient;
    }
    Ok(coefficient)
}

/// Port of FEFF `rotwig`: Wigner small-d rotation matrix element.
///
/// `jj`, `m1`, and `m2` are scaled by `scale`, matching FEFF's `ient`: use
/// `scale = 1` for integer angular momenta and `scale = 2` for half-integers
/// represented as doubled integers.
pub fn wigner_rotation(
    beta: Real,
    jj: i32,
    m1: i32,
    m2: i32,
    scale: i32,
) -> Result<Real, AngularError> {
    const FACTORIAL_LIMIT: i32 = 58;

    if scale != 1 && scale != 2 {
        return Err(AngularError::InvalidWignerScale { scale });
    }
    if !beta.is_finite() {
        return Err(AngularError::NonFiniteRotationAngle);
    }

    let (m1p, m2p, beta, sign) = if m1 >= 0 && m1.abs() >= m2.abs() {
        (m1, m2, beta, 1.0)
    } else if m2 >= 0 && m2.abs() >= m1.abs() {
        (m2, m1, -beta, 1.0)
    } else if m1 <= 0 && m1.abs() >= m2.abs() {
        (
            -m1,
            -m2,
            beta,
            alternating_sign(checked_scaled_argument(m1 - m2, scale)?),
        )
    } else {
        (
            -m2,
            -m1,
            -beta,
            alternating_sign(checked_scaled_argument(m2 - m1, scale)?),
        )
    };

    let log_factorial = log_factorials(FACTORIAL_LIMIT)?;
    let zeta = (beta / 2.0).cos();
    let eta = (beta / 2.0).sin();
    let mut total = 0.0;
    let mut term_index = m1p - m2p;
    let last = jj - m2p;
    while term_index <= last {
        let factorial_arguments = [
            checked_scaled_argument(jj + m1p, scale)?,
            checked_scaled_argument(jj - m1p, scale)?,
            checked_scaled_argument(jj + m2p, scale)?,
            checked_scaled_argument(jj - m2p, scale)?,
            checked_scaled_argument(jj + m1p - term_index, scale)?,
            checked_scaled_argument(jj - m2p - term_index, scale)?,
            checked_scaled_argument(term_index, scale)?,
            checked_scaled_argument(m2p - m1p + term_index, scale)?,
        ];
        let zeta_power = checked_scaled_argument(2 * jj + m1p - m2p - 2 * term_index, scale)?;
        let eta_power = checked_scaled_argument(2 * term_index - m1p + m2p, scale)?;
        if zeta_power < 0 || eta_power < 0 {
            return Err(AngularError::WignerFactorialOutOfRange {
                argument: zeta_power.min(eta_power),
                limit: FACTORIAL_LIMIT,
            });
        }

        let mut factor = 0.0;
        for &argument in &factorial_arguments[..4] {
            factor += log_factorial_value(&log_factorial, argument, FACTORIAL_LIMIT)? / 2.0;
        }
        for &argument in &factorial_arguments[4..] {
            factor -= log_factorial_value(&log_factorial, argument, FACTORIAL_LIMIT)?;
        }

        let coefficient =
            alternating_sign(checked_scaled_argument(term_index, scale)?) * factor.exp();
        let term = match (zeta_power, eta_power) {
            (0, 0) => coefficient,
            (_, 0) => coefficient * zeta.powi(zeta_power),
            (0, _) => coefficient * eta.powi(eta_power),
            _ => coefficient * zeta.powi(zeta_power) * eta.powi(eta_power),
        };
        total += term;
        term_index += scale;
    }

    Ok(sign * total)
}

/// Build FEFF `t3jp` and `t3jm` spin-orbit coupling tables.
pub fn spin_orbit_coupling_tables(lmax: usize) -> Result<SpinOrbitCouplingTables, AngularError> {
    let mut plus = Array3::zeros((lmax + 1, 2 * lmax + 1, 2).f());
    let mut minus = Array3::zeros((lmax + 1, 2 * lmax + 1, 2).f());
    let lmax_isize =
        isize::try_from(lmax).map_err(|_| AngularError::IndexTooLarge { value: lmax })?;

    for l in 0..=lmax {
        let l_i32 = usize_to_i32(l)?;
        let l_isize = isize::try_from(l).map_err(|_| AngularError::IndexTooLarge { value: l })?;
        for magnetic in -l_isize..=l_isize {
            let magnetic_i32 = isize_to_i32(magnetic)?;
            let magnetic_index = magnetic_table_index(magnetic, lmax)?;
            for spin_index in 0..2 {
                let spin_i32 = usize_to_i32(spin_index + 1)?;
                let j1 = 2 * l_i32;
                let j2 = 1;
                let j3p = j1 + 1;
                let j3m = j1 - 1;
                let m1 = 2 * magnetic_i32;
                let m2 = 2 * spin_i32 - 3;
                let sign = feff_spin_coupling_sign(j2, j1, m1, m2);

                plus[[l, magnetic_index, spin_index]] =
                    sign * f64::from(j3p + 1).sqrt() * wigner_3j(j1, j2, j3p, m1, m2, 2)?;
                minus[[l, magnetic_index, spin_index]] =
                    sign * f64::from(j3m + 1).sqrt() * wigner_3j(j1, j2, j3m, m1, m2, 2)?;
            }
        }
    }

    Ok(SpinOrbitCouplingTables {
        plus,
        minus,
        m_offset: usize::try_from(lmax_isize)
            .map_err(|_| AngularError::IndexTooLarge { value: lmax })?,
    })
}

fn ensure_finite_polarization_tensor(tensor: &[[Complex; 3]; 3]) -> Result<(), AngularError> {
    for (row, values) in tensor.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(AngularError::NonFinitePolarizationTensor {
                    row: row as isize - 1,
                    column: column as isize - 1,
                });
            }
        }
    }
    Ok(())
}

fn fill_transition_indices(
    initial_kappa: i32,
    multipole: i32,
    lmax: i32,
    jind: &mut [i32; 8],
    lind: &mut [i32; 8],
    kiind: &mut [i32; 8],
) {
    for k in -1_i32..=1 {
        let mut kappa = initial_kappa + k;
        if k == 0 {
            kappa = -kappa;
        }
        let mut jkap = kappa.abs();
        let mut lkap = if kappa <= 0 { kappa.abs() - 1 } else { kappa };
        if lkap > lmax {
            jkap = 0;
            lkap = -1;
            kappa = 0;
        }
        let index = (k + 1) as usize;
        jind[index] = jkap;
        lind[index] = lkap;
        kiind[index] = kappa;
    }

    for k in -2_i32..=2 {
        let mut jkap = initial_kappa.abs() + k;
        if jkap <= 0 {
            jkap = 0;
        }
        let mut kappa = jkap;
        if initial_kappa < 0 && k.abs() != 1 {
            kappa = -jkap;
        }
        if initial_kappa > 0 && k.abs() == 1 {
            kappa = -jkap;
        }
        let mut lkap = if kappa <= 0 { -kappa - 1 } else { kappa };
        if lkap > lmax || multipole == 0 || (multipole == 1 && k.abs() == 2) {
            jkap = 0;
            lkap = -1;
            kappa = 0;
        }
        let index = (k + 5) as usize;
        jind[index] = jkap;
        lind[index] = lkap;
        kiind[index] = kappa;
    }
}

fn fill_averaged_transition_matrix(
    multipole: i32,
    lmax: i32,
    lind: &[i32; 8],
    matrix: &mut Array6<Complex>,
) {
    for (transition, &l) in lind.iter().enumerate() {
        if l < 0 {
            continue;
        }
        let multipole_degeneracy = if multipole == 2 && transition >= 3 {
            5.0
        } else {
            3.0
        };
        let mut value = 0.5 / (2.0 * f64::from(l) + 1.0) / multipole_degeneracy;
        if transition < 3 {
            value = -value;
        }
        for spin in 0..=1 {
            for magnetic in -l..=l {
                let index = magnetic_index_i32(magnetic, lmax);
                matrix[[index, spin, transition, index, spin, transition]] =
                    Complex::new(value, 0.0);
            }
        }
    }
}

fn fill_polarized_transition_matrix(
    input: TransitionBMatrixInput,
    lmax: i32,
    magnetic_len: usize,
    extended_magnetic_len: usize,
    jind: &[i32; 8],
    lind: &[i32; 8],
    matrix: &mut Array6<Complex>,
) -> Result<(), AngularError> {
    let mut t3j = vec![0.0; 8 * 2 * extended_magnetic_len];
    let mut x3j = vec![0.0; 8 * 3 * extended_magnetic_len];
    let mut qmat = vec![0.0; extended_magnetic_len * magnetic_len * 2 * 8];
    let mut pmat =
        vec![Complex::new(0.0, 0.0); extended_magnetic_len * 8 * extended_magnetic_len * 8];
    let mut tmat = vec![Complex::new(0.0, 0.0); extended_magnetic_len * 8 * magnetic_len * 2 * 8];

    for transition in 0..8 {
        let j = jind[transition];
        if j <= 0 {
            continue;
        }
        for mp in -j + 1..=j {
            for spin in 0..=1 {
                let j1 = 2 * lind[transition];
                let j2 = 1;
                let j3 = 2 * j - 1;
                let m1 = 2 * (mp - spin as i32);
                let m2 = 2 * spin as i32 - 1;
                let mut value = f64::from(j3 + 1).sqrt() * wigner_3j(j1, j2, j3, m1, m2, 2)?;
                if ((j2 - j1 - m1 - m2) / 2) % 2 != 0 {
                    value = -value;
                }
                let index = t3j_index(transition, spin, mp, lmax, extended_magnetic_len);
                t3j[index] = value;
            }

            for polarization in -1..=1 {
                let j1 = 2 * j - 1;
                let j2 = if transition >= 3 && input.multipole == 2 {
                    4
                } else {
                    2
                };
                let j3 = 2 * input.initial_kappa.abs() - 1;
                let m1 = -2 * mp + 1;
                let m2 = 2 * polarization;
                let index = x3j_index(transition, polarization, mp, lmax, extended_magnetic_len);
                x3j[index] = wigner_3j(j1, j2, j3, m1, m2, 2)?;
            }
        }
    }

    for transition in 0..8 {
        let l = lind[transition];
        let j = jind[transition];
        if l < 0 || j <= 0 {
            continue;
        }
        for spin in 0..=1 {
            for ml in -l..=l {
                for mj in -j + 1..=j {
                    let mp = ml + spin as i32;
                    let jj = 2 * j - 1;
                    let mmj = 2 * mj - 1;
                    let mmp = 2 * mp - 1;
                    let rotation = wigner_rotation(input.spin_vector_angle, jj, mmj, mmp, 2)?;
                    let t3j_value =
                        t3j[t3j_index(transition, spin, mp, lmax, extended_magnetic_len)];
                    let index = qmat_index(mj, ml, spin, transition, lmax, magnetic_len);
                    qmat[index] = rotation * t3j_value;
                }
            }
        }
    }

    for transition2 in 0..8 {
        let j2 = jind[transition2];
        if j2 <= 0 {
            continue;
        }
        for m2 in -j2 + 1..=j2 {
            for transition1 in 0..8 {
                let j1 = jind[transition1];
                if j1 <= 0 {
                    continue;
                }
                for m1 in -j1 + 1..=j1 {
                    if (m2 - m1).abs() <= 2 {
                        let mut value = Complex::new(0.0, 0.0);
                        for p_prime in -1..=1 {
                            for p in -1..=1 {
                                if m1 - p == m2 - p_prime {
                                    let mut sign = 1.0;
                                    if input.multipole == 1 && p > 0 && transition1 >= 3 {
                                        sign = -sign;
                                    }
                                    if input.multipole == 1 && p_prime > 0 && transition2 >= 3 {
                                        sign = -sign;
                                    }
                                    value += input.polarization_tensor[(p + 1) as usize]
                                        [(p_prime + 1) as usize]
                                        * (sign
                                            * x3j[x3j_index(
                                                transition1,
                                                p,
                                                m1,
                                                lmax,
                                                extended_magnetic_len,
                                            )]
                                            * x3j[x3j_index(
                                                transition2,
                                                p_prime,
                                                m2,
                                                lmax,
                                                extended_magnetic_len,
                                            )]);
                                }
                            }
                        }

                        let mut sign = 1.0;
                        if (jind[transition1] - jind[transition2]) % 2 != 0 {
                            sign = -sign;
                        }
                        if transition2 < 3 {
                            sign = -sign;
                        }
                        value *= sign * imaginary_unit_power(lind[transition2] - lind[transition1]);
                        let index = pmat_index(
                            m1,
                            transition1,
                            m2,
                            transition2,
                            lmax,
                            extended_magnetic_len,
                        );
                        pmat[index] = value;
                    }
                }
            }
        }
    }

    for transition1 in 0..8 {
        let l1 = lind[transition1];
        let j1 = jind[transition1];
        if l1 < 0 || j1 <= 0 {
            continue;
        }
        for spin in 0..=1 {
            for ml in -l1..=l1 {
                for transition2 in 0..8 {
                    let j2 = jind[transition2];
                    if j2 <= 0 {
                        continue;
                    }
                    for mj in -j2 + 1..=j2 {
                        let mut value = Complex::new(0.0, 0.0);
                        for mp in -j1 + 1..=j1 {
                            value += pmat[pmat_index(
                                mj,
                                transition2,
                                mp,
                                transition1,
                                lmax,
                                extended_magnetic_len,
                            )] * qmat
                                [qmat_index(mp, ml, spin, transition1, lmax, magnetic_len)];
                        }
                        let index =
                            tmat_index(mj, transition2, ml, spin, transition1, lmax, magnetic_len);
                        tmat[index] = value;
                    }
                }
            }
        }
    }

    for transition1 in 0..8 {
        let l1 = lind[transition1];
        if l1 < 0 {
            continue;
        }
        for spin1 in 0..=1 {
            for ml1 in -l1..=l1 {
                for transition2 in 0..8 {
                    let l2 = lind[transition2];
                    let j2 = jind[transition2];
                    if l2 < 0 || j2 <= 0 {
                        continue;
                    }
                    for spin2 in 0..=1 {
                        for ml2 in -l2..=l2 {
                            let mut value = Complex::new(0.0, 0.0);
                            for mj in -j2 + 1..=j2 {
                                value += qmat
                                    [qmat_index(mj, ml2, spin2, transition2, lmax, magnetic_len)]
                                    * tmat[tmat_index(
                                        mj,
                                        transition2,
                                        ml1,
                                        spin1,
                                        transition1,
                                        lmax,
                                        magnetic_len,
                                    )];
                            }
                            let ml2_index = magnetic_index_i32(ml2, lmax);
                            let ml1_index = magnetic_index_i32(ml1, lmax);
                            matrix
                                [[ml2_index, spin2, transition2, ml1_index, spin1, transition1]] =
                                value;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn trace_transition_matrix(lmax: i32, lind: &[i32; 8], matrix: &mut Array6<Complex>) {
    let zero_index = magnetic_index_i32(0, lmax);
    for transition1 in 0..8 {
        for spin1 in 0..=1 {
            for transition2 in 0..8 {
                for spin2 in 0..=1 {
                    if lind[transition1] != lind[transition2] || spin1 != spin2 {
                        matrix[[
                            zero_index,
                            spin2,
                            transition2,
                            zero_index,
                            spin1,
                            transition1,
                        ]] = Complex::new(0.0, 0.0);
                    } else {
                        let l = lind[transition1];
                        for magnetic in 1..=l {
                            let negative = magnetic_index_i32(-magnetic, lmax);
                            let positive = magnetic_index_i32(magnetic, lmax);
                            let addition = matrix
                                [[negative, spin1, transition2, negative, spin1, transition1]]
                                + matrix
                                    [[positive, spin1, transition2, positive, spin1, transition1]];
                            matrix[[
                                zero_index,
                                spin1,
                                transition2,
                                zero_index,
                                spin1,
                                transition1,
                            ]] += addition;
                        }
                    }
                }
            }
        }
    }
}

fn fold_transition_spin(
    spin: i32,
    spin_channels: usize,
    lmax: i32,
    lind: &[i32; 8],
    matrix: &mut Array6<Complex>,
) {
    if spin == 0 {
        for transition1 in 0..8 {
            let l1 = lind[transition1];
            if l1 < 0 {
                continue;
            }
            for transition2 in 0..8 {
                let l2 = lind[transition2];
                if l2 < 0 {
                    continue;
                }
                for ml1 in -l1..=l1 {
                    for ml2 in -l2..=l2 {
                        let ml1_index = magnetic_index_i32(ml1, lmax);
                        let ml2_index = magnetic_index_i32(ml2, lmax);
                        let addition =
                            matrix[[ml2_index, 1, transition2, ml1_index, 1, transition1]];
                        matrix[[ml2_index, 0, transition2, ml1_index, 0, transition1]] += addition;
                    }
                }
            }
        }
    } else if spin == 2 || (spin == 1 && spin_channels == 1) {
        for transition1 in 0..8 {
            let l1 = lind[transition1];
            if l1 < 0 {
                continue;
            }
            for transition2 in 0..8 {
                let l2 = lind[transition2];
                if l2 < 0 {
                    continue;
                }
                for ml1 in -l1..=l1 {
                    for ml2 in -l2..=l2 {
                        let ml1_index = magnetic_index_i32(ml1, lmax);
                        let ml2_index = magnetic_index_i32(ml2, lmax);
                        matrix[[ml2_index, 0, transition2, ml1_index, 0, transition1]] =
                            matrix[[ml2_index, 1, transition2, ml1_index, 1, transition1]];
                    }
                }
            }
        }
    }
}

fn t3j_index(
    transition: usize,
    spin: usize,
    magnetic: i32,
    lmax: i32,
    extended_magnetic_len: usize,
) -> usize {
    (transition * 2 + spin) * extended_magnetic_len + extended_magnetic_index_i32(magnetic, lmax)
}

fn x3j_index(
    transition: usize,
    polarization: i32,
    magnetic: i32,
    lmax: i32,
    extended_magnetic_len: usize,
) -> usize {
    (transition * 3 + (polarization + 1) as usize) * extended_magnetic_len
        + extended_magnetic_index_i32(magnetic, lmax)
}

fn qmat_index(
    mj: i32,
    ml: i32,
    spin: usize,
    transition: usize,
    lmax: i32,
    magnetic_len: usize,
) -> usize {
    (((extended_magnetic_index_i32(mj, lmax) * magnetic_len + magnetic_index_i32(ml, lmax)) * 2
        + spin)
        * 8)
        + transition
}

fn pmat_index(
    m1: i32,
    transition1: usize,
    m2: i32,
    transition2: usize,
    lmax: i32,
    extended_magnetic_len: usize,
) -> usize {
    (((extended_magnetic_index_i32(m1, lmax) * 8 + transition1) * extended_magnetic_len
        + extended_magnetic_index_i32(m2, lmax))
        * 8)
        + transition2
}

fn tmat_index(
    mj: i32,
    transition2: usize,
    ml: i32,
    spin: usize,
    transition1: usize,
    lmax: i32,
    magnetic_len: usize,
) -> usize {
    ((((extended_magnetic_index_i32(mj, lmax) * 8 + transition2) * magnetic_len
        + magnetic_index_i32(ml, lmax))
        * 2
        + spin)
        * 8)
        + transition1
}

fn magnetic_index_i32(magnetic: i32, lmax: i32) -> usize {
    (magnetic + lmax) as usize
}

fn extended_magnetic_index_i32(magnetic: i32, lmax: i32) -> usize {
    (magnetic + lmax) as usize
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

fn usize_to_real(value: usize) -> Result<Real, AngularError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| AngularError::IndexTooLarge { value })
}

fn usize_to_i32(value: usize) -> Result<i32, AngularError> {
    i32::try_from(value).map_err(|_| AngularError::IndexTooLarge { value })
}

fn isize_to_i32(value: isize) -> Result<i32, AngularError> {
    i32::try_from(value).map_err(|_| AngularError::MagneticIndexOutOfRange {
        magnetic: value,
        lmax: usize::MAX,
    })
}

fn magnetic_table_index(magnetic: isize, lmax: usize) -> Result<usize, AngularError> {
    let lmax_isize =
        isize::try_from(lmax).map_err(|_| AngularError::IndexTooLarge { value: lmax })?;
    let shifted = magnetic + lmax_isize;
    usize::try_from(shifted).map_err(|_| AngularError::MagneticIndexOutOfRange { magnetic, lmax })
}

fn feff_spin_coupling_sign(j2: i32, j1: i32, m1: i32, m2: i32) -> Real {
    let phase = (j2 - j1 - m1 - m2) / 2;
    if phase % 2 == 0 { 1.0 } else { -1.0 }
}

fn checked_scaled_argument(argument: i32, scale: i32) -> Result<i32, AngularError> {
    if argument % scale != 0 {
        return Err(AngularError::InvalidWignerParity { argument, scale });
    }
    Ok(argument / scale)
}

fn alternating_sign(exponent: i32) -> Real {
    if exponent % 2 == 0 { 1.0 } else { -1.0 }
}

fn alternating_sign_usize(exponent: usize) -> Real {
    if exponent.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

fn spherical_harmonic_index(l: usize, m: isize) -> usize {
    let base = l * l + l;
    if m < 0 {
        base - m.unsigned_abs()
    } else {
        base + m as usize
    }
}

fn log_factorials(limit: i32) -> Result<Vec<Real>, AngularError> {
    let limit = usize::try_from(limit).map_err(|_| AngularError::WignerFactorialOutOfRange {
        argument: limit,
        limit,
    })?;
    let mut values = Vec::with_capacity(limit + 1);
    let mut previous = 0.0;
    values.push(previous);
    for index in 1..=limit {
        let index = usize_to_real(index)?;
        previous += index.ln();
        values.push(previous);
    }
    Ok(values)
}

fn log_factorial_value(
    log_factorials: &[Real],
    argument: i32,
    limit: i32,
) -> Result<Real, AngularError> {
    if argument < 0 || argument > limit {
        return Err(AngularError::WignerFactorialOutOfRange { argument, limit });
    }
    let index = usize::try_from(argument)
        .map_err(|_| AngularError::WignerFactorialOutOfRange { argument, limit })?;
    log_factorials
        .get(index)
        .copied()
        .ok_or(AngularError::WignerFactorialOutOfRange { argument, limit })
}

#[cfg(test)]
mod tests {
    use super::{
        AngularError, TransitionBMatrixInput, legendre_normalization, legendre_normalization_table,
        legendre_polynomials, spherical_harmonics, spin_orbit_coupling_tables, transition_b_matrix,
        wigner_3j, wigner_rotation,
    };
    use crate::Complex;

    #[test]
    fn computes_snlm_values() -> Result<(), AngularError> {
        assert_close(legendre_normalization(0, 0)?, 1.0);
        assert_close(legendre_normalization(1, 0)?, 3.0_f64.sqrt());
        assert_close(legendre_normalization(1, 1)?, (3.0_f64 / 2.0).sqrt());
        assert_close(legendre_normalization(2, 2)?, (5.0_f64 / 24.0).sqrt());
        assert_eq!(legendre_normalization(1, 2)?, 0.0);
        Ok(())
    }

    #[test]
    fn computes_cpl0_legendre_polynomials() {
        let values = legendre_polynomials(0.25, 4);
        let expected = [1.0, 0.25, -0.40625, -0.3359375, 0.15771484375];

        for (&actual, expected) in values.iter().zip(expected) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn builds_fortran_order_xnlm_table() -> Result<(), AngularError> {
        let table = legendre_normalization_table(3)?;

        assert_eq!(table.shape(), &[4, 4]);
        assert_eq!(table.strides(), &[1, 4]);
        assert_close(table[[0, 2]], 5.0_f64.sqrt());
        assert_close(table[[2, 2]], (5.0_f64 / 24.0).sqrt());
        assert_eq!(table[[3, 2]], 0.0);
        Ok(())
    }

    #[test]
    fn computes_integer_wigner_3j_coefficients() -> Result<(), AngularError> {
        assert_close(wigner_3j(1, 1, 0, 0, 0, 1)?, -1.0 / 3.0_f64.sqrt());
        assert_eq!(wigner_3j(1, 1, 3, 0, 0, 1)?, 0.0);
        Ok(())
    }

    #[test]
    fn computes_half_integer_wigner_3j_coefficients() -> Result<(), AngularError> {
        assert_close(wigner_3j(0, 1, 1, 0, -1, 2)?, -1.0 / 2.0_f64.sqrt());
        assert_eq!(
            wigner_3j(2, 2, 2, 1, 0, 2),
            Err(AngularError::InvalidWignerParity {
                argument: 3,
                scale: 2,
            })
        );
        Ok(())
    }

    #[test]
    fn computes_wigner_rotation_elements() -> Result<(), AngularError> {
        assert_close(wigner_rotation(0.7, 2, 1, -1, 1)?, 0.2974375221921237);
        assert_close(wigner_rotation(1.1, 3, -2, 1, 1)?, 0.4544222701103565);
        assert_close(wigner_rotation(0.7, 3, 1, -1, 2)?, -0.5648429673316498);
        assert_close(wigner_rotation(-0.9, 5, -3, 1, 2)?, 0.494867123375203);
        assert_close(wigner_rotation(0.4, 4, -2, -4, 2)?, -0.3740481938792059);
        Ok(())
    }

    #[test]
    fn rejects_invalid_wigner_rotation_inputs() {
        assert_eq!(
            wigner_rotation(0.1, 1, 0, 0, 3),
            Err(AngularError::InvalidWignerScale { scale: 3 })
        );
        assert_eq!(
            wigner_rotation(f64::NAN, 1, 0, 0, 1),
            Err(AngularError::NonFiniteRotationAngle)
        );
        assert!(matches!(
            wigner_rotation(0.1, 3, 1, 0, 2),
            Err(AngularError::InvalidWignerParity { .. })
        ));
    }

    #[test]
    fn computes_feff_spherical_harmonics() -> Result<(), AngularError> {
        let values = spherical_harmonics([1.0, 2.0, 3.0], 3)?;
        let expected = [
            Complex::new(0.28209478784887687, 0.0),
            Complex::new(0.09233719417642765, -0.1846743883528553),
            Complex::new(0.3917535369473459, 0.0),
            Complex::new(-0.09233719417642765, -0.1846743883528553),
            Complex::new(-0.08277304213899862, -0.1103640561853315),
            Complex::new(0.16554608427799725, -0.3310921685559945),
            Complex::new(0.29286359223107555, 0.0),
            Complex::new(-0.16554608427799725, -0.3310921685559945),
            Complex::new(-0.08277304213899862, 0.1103640561853315),
            Complex::new(-0.08761323662775768, 0.015929679386865035),
            Complex::new(-0.17558813818777735, -0.23411751758370314),
            Complex::new(0.1912556872248307, -0.3825113744496614),
            Complex::new(0.0641157227438533, 0.0),
            Complex::new(-0.1912556872248307, -0.3825113744496614),
            Complex::new(-0.17558813818777735, 0.23411751758370314),
            Complex::new(0.08761323662775768, 0.015929679386865035),
        ];

        assert_complex_iter_close(values.iter().copied(), &expected);
        Ok(())
    }

    #[test]
    fn spherical_harmonics_match_feff_axis_and_zero_vector_cases() -> Result<(), AngularError> {
        let xz = spherical_harmonics([-0.5, 0.0, 2.0], 2)?;
        assert_complex_iter_close(
            xz.iter().copied(),
            &[
                Complex::new(0.28209478784887687, 0.0),
                Complex::new(-0.08379463832252748, 0.0),
                Complex::new(0.47401405587946666, 0.0),
                Complex::new(0.08379463832252748, 0.0),
                Complex::new(0.022722011567568246, 0.0),
                Complex::new(-0.18177609254054597, 0.0),
                Complex::new(0.5751257874583111, 0.0),
                Complex::new(0.18177609254054597, 0.0),
                Complex::new(0.022722011567568246, 0.0),
            ],
        );

        let zero = spherical_harmonics([0.0, 0.0, 0.0], 2)?;
        assert_complex_iter_close(
            zero.iter().copied(),
            &[
                Complex::new(0.28209478784887687, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.4886025051046183, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.6307831217284703, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        );
        Ok(())
    }

    #[test]
    fn rejects_non_finite_spherical_harmonic_vector() {
        assert_eq!(
            spherical_harmonics([f64::NAN, 0.0, 1.0], 2),
            Err(AngularError::NonFiniteVector)
        );
    }

    #[test]
    fn builds_feff_averaged_transition_b_matrix() -> Result<(), AngularError> {
        let result = transition_b_matrix(TransitionBMatrixInput {
            polarization: 0,
            ..sample_transition_b_matrix_input()
        })?;

        assert_eq!(result.kappa_indices, [-2, 1, 0, 0, 0, -1, 2, -3]);
        assert_eq!(result.orbital_momenta, [1, 1, -1, -1, -1, 0, 2, 2]);
        assert_transition_values(
            &result,
            &[
                Complex::new(-0.05555555555555555, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.02, 0.0),
                Complex::new(0.02, 0.0),
            ],
        );
        assert_complex_close(
            matrix_sum(&result.matrix),
            Complex::new(-0.06666666666666678, 0.0),
        );
        assert_eq!(nonzero_count(&result.matrix), 34);
        Ok(())
    }

    #[test]
    fn builds_feff_polarized_transition_b_matrix() -> Result<(), AngularError> {
        let result = transition_b_matrix(sample_transition_b_matrix_input())?;

        assert_eq!(result.kappa_indices, [-2, 1, 0, 0, 0, -1, 2, -3]);
        assert_eq!(result.orbital_momenta, [1, 1, -1, -1, -1, 0, 2, 2]);
        assert_transition_values(
            &result,
            &[
                Complex::new(-0.06273441744900005, -0.002816883591316175),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.028021517059414094, 0.0012286136649168643),
                Complex::new(0.032783586221432195, 0.001621399287040505),
            ],
        );
        assert_complex_close(
            matrix_sum(&result.matrix),
            Complex::new(-0.1498383428670292, 1.0054312926918387),
        );
        assert_eq!(nonzero_count(&result.matrix), 776);
        Ok(())
    }

    #[test]
    fn applies_feff_trace_and_spin_folding_to_transition_b_matrix() -> Result<(), AngularError> {
        let traced = transition_b_matrix(TransitionBMatrixInput {
            trace_orbital: true,
            ..sample_transition_b_matrix_input()
        })?;
        assert_transition_values(
            &traced,
            &[
                Complex::new(-0.13927806177774438, -0.006120603534129609),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.08851128918371495, -0.0038584833449001833),
                Complex::new(0.08464910571422107, 0.0008909740920669523),
            ],
        );
        assert_complex_close(
            matrix_sum(&traced.matrix),
            Complex::new(-0.31619310063452616, 0.8078421709741431),
        );
        assert_eq!(nonzero_count(&traced.matrix), 728);

        let spin_folded = transition_b_matrix(TransitionBMatrixInput {
            spin: 0,
            ..sample_transition_b_matrix_input()
        })?;
        assert_transition_values(
            &spin_folded,
            &[
                Complex::new(-0.12775761018690238, -0.0018521924356471728),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.05865146715590615, -0.0005346566843031169),
                Complex::new(0.06730612780091635, 0.0012482092313231155),
            ],
        );
        assert_complex_close(
            matrix_sum(&spin_folded.matrix),
            Complex::new(-0.2170418650429801, 1.3409109998393136),
        );
        assert_eq!(nonzero_count(&spin_folded.matrix), 828);
        Ok(())
    }

    #[test]
    fn builds_feff_magnetic_dipole_transition_b_matrix() -> Result<(), AngularError> {
        let result = transition_b_matrix(TransitionBMatrixInput {
            initial_kappa: 2,
            multipole: 1,
            spin: 2,
            spin_vector_angle: -0.4,
            ..sample_transition_b_matrix_input()
        })?;

        assert_eq!(result.kappa_indices, [1, -2, 3, 0, -1, 2, -3, 0]);
        assert_eq!(result.orbital_momenta, [1, 1, 3, -1, 0, 2, 2, -1]);
        assert_transition_values(
            &result,
            &[
                Complex::new(-0.04864195024406099, 0.003463869265078493),
                Complex::new(0.0017631363458188638, -0.0002788047964405202),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.035984450619728485, 0.0004976156588231403),
                Complex::new(0.0, 0.0),
            ],
        );
        assert_complex_close(
            matrix_sum(&result.matrix),
            Complex::new(0.21773917179567237, 1.463795753151296),
        );
        assert_eq!(nonzero_count(&result.matrix), 1432);
        Ok(())
    }

    #[test]
    fn rejects_invalid_transition_b_matrix_inputs() {
        assert_eq!(
            transition_b_matrix(TransitionBMatrixInput {
                spin_channels: 0,
                ..sample_transition_b_matrix_input()
            }),
            Err(AngularError::InvalidSpinChannelCount { value: 0 })
        );

        let mut input = sample_transition_b_matrix_input();
        input.polarization_tensor[0][2] = Complex::new(f64::NAN, 0.0);
        assert_eq!(
            transition_b_matrix(input),
            Err(AngularError::NonFinitePolarizationTensor { row: -1, column: 1 })
        );
    }

    #[test]
    fn builds_spin_orbit_coupling_tables() -> Result<(), AngularError> {
        let tables = spin_orbit_coupling_tables(1)?;

        assert_eq!(tables.plus.shape(), &[2, 3, 2]);
        assert_eq!(tables.plus.strides(), &[1, 2, 6]);
        assert_eq!(tables.m_offset, 1);
        assert_close(tables.plus[[0, 1, 0]], 1.0);
        assert_close(tables.plus[[0, 1, 1]], 1.0);
        assert_close(tables.minus[[0, 1, 0]], 0.0);
        Ok(())
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_complex_iter_close(
        actual: impl ExactSizeIterator<Item = Complex>,
        expected: &[Complex],
    ) {
        assert_eq!(actual.len(), expected.len());
        for (actual, &expected) in actual.zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    fn sample_transition_b_matrix_input() -> TransitionBMatrixInput {
        TransitionBMatrixInput {
            lmax: 3,
            initial_kappa: -1,
            polarization: 1,
            polarization_tensor: [
                [
                    Complex::new(0.20, -0.05),
                    Complex::new(-0.10, 0.04),
                    Complex::new(0.03, 0.02),
                ],
                [
                    Complex::new(0.11, -0.07),
                    Complex::new(0.50, 0.00),
                    Complex::new(-0.08, 0.09),
                ],
                [
                    Complex::new(0.06, 0.01),
                    Complex::new(0.13, -0.02),
                    Complex::new(0.17, 0.03),
                ],
            ],
            multipole: 2,
            trace_orbital: false,
            spin: 1,
            spin_channels: 1,
            spin_vector_angle: 0.3,
        }
    }

    fn assert_transition_values(result: &super::TransitionBMatrix, expected: &[Complex; 6]) {
        let indices = [
            (0, 0, 1, 0, 0, 1),
            (1, 0, 2, -1, 0, 2),
            (0, 1, 4, 0, 1, 4),
            (-1, 0, 5, 1, 1, 6),
            (0, 0, 7, 0, 0, 7),
            (0, 0, 8, 0, 0, 8),
        ];
        for (index, &expected) in indices.iter().zip(expected.iter()) {
            let actual = result.value(index.0, index.1, index.2, index.3, index.4, index.5);
            assert!(actual.is_some(), "missing transition matrix value");
            if let Some(actual) = actual {
                assert_complex_close(actual, expected);
            }
        }
    }

    fn matrix_sum(matrix: &ndarray::Array6<Complex>) -> Complex {
        matrix
            .iter()
            .copied()
            .fold(Complex::new(0.0, 0.0), |accumulator, value| {
                accumulator + value
            })
    }

    fn nonzero_count(matrix: &ndarray::Array6<Complex>) -> usize {
        matrix.iter().filter(|value| value.norm() > 1.0e-14).count()
    }
}
