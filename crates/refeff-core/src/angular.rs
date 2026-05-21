//! Angular-momentum normalization helpers.
//!
//! FEFF stores associated-Legendre normalization factors in `xnlm`; FMS uses
//! `xnlm(m,l)` while GENFMT carries the same values in a one-based table. The
//! helpers here compute the shared value
//! `sqrt((2l+1) * (l-m)! / (l+m)!)`.

use ndarray::{Array2, Array3, Array4, Array6, ArrayView2, ShapeBuilder};
use refeff_linalg::complex_matmul;

use crate::{Complex, ComplexMat, ComplexVec, Real, RealMat, RealVec};

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
    /// FEFF relativistic state indexing uses nonzero kappa values.
    #[error("invalid relativistic kappa {kappa}; expected nonzero finite i32 range")]
    InvalidRelativisticKappa { kappa: i32 },
    /// FEFF `MUEM05` must lie in `-abs(kappa)..abs(kappa)-1`.
    #[error("relativistic MUEM05 {mu_minus_half} is outside kappa {kappa} range")]
    RelativisticMagneticIndexOutOfRange { kappa: i32, mu_minus_half: i32 },
    /// The requested magnetic index does not fit the allocated table.
    #[error("magnetic index {magnetic} is outside table range for lmax {lmax}")]
    MagneticIndexOutOfRange { magnetic: isize, lmax: usize },
    /// A FEFF angular helper received an inconsistent output table dimension.
    #[error("angular table dimension {name} must be at least {minimum}, got {value}")]
    InvalidAngularTableDimension {
        name: &'static str,
        value: usize,
        minimum: usize,
    },
    /// FEFF Wigner rotations require a finite angle.
    #[error("Wigner rotation angle must be finite")]
    NonFiniteRotationAngle,
    /// FEFF `ylm` requires finite Cartesian vector components.
    #[error("spherical-harmonic vector components must be finite")]
    NonFiniteVector,
    /// FEFF transition matrices require finite polarization tensor entries.
    #[error("polarization tensor entry ({row}, {column}) must be finite")]
    NonFinitePolarizationTensor { row: isize, column: isize },
    /// FEFF basis transformations consume square matrices of the compiled order.
    #[error("basis-transform matrix {name} must be {expected}x{expected}, got {rows}x{columns}")]
    InvalidBasisTransformShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected: usize,
    },
    /// FEFF `iniptz` accepts tensor selectors `1..=10`.
    #[error("invalid polarization tensor selector {index}; expected 1..=10")]
    InvalidPolarizationTensorIndex { index: usize },
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

/// FEFF `CALCCGC` relativistic Clebsch-Gordan coefficient table.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativisticClebschGordanCoefficients {
    /// FEFF `CGC(IKM, IS)` coefficients as `(state, spin_component)`.
    pub coefficients: RealMat,
    /// FEFF `LTAB` branch orbital momentum values.
    pub orbital_momentum: Vec<usize>,
    /// FEFF `KAPTAB` branch relativistic kappa values.
    pub kappa: Vec<i32>,
    /// FEFF `NMUETAB` branch state counts.
    pub spin_multiplicity: Vec<usize>,
}

/// FEFF `BASTRMAT` basis-transformation matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisTransformMatrices {
    /// Maximum orbital momentum used to construct the matrices.
    pub lmax: usize,
    /// Matrix order, FEFF `NKM = 2 * (LMAX + 1)^2`.
    pub order: usize,
    /// FEFF `RC`: transforms real spherical harmonics to complex harmonics.
    pub real_to_complex: ComplexMat,
    /// FEFF `CREL`: transforms complex harmonics to relativistic `(kappa,mue)`.
    pub complex_to_relativistic: ComplexMat,
    /// FEFF `RREL`: transforms real harmonics to relativistic `(kappa,mue)`.
    pub real_to_relativistic: ComplexMat,
}

/// FEFF `CHANGEREP` representation-conversion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisTransformMode {
    /// FEFF `REL>RLM`.
    RelativisticToReal,
    /// FEFF `RLM>REL`.
    RealToRelativistic,
    /// FEFF `REL>CLM`.
    RelativisticToComplex,
    /// FEFF `CLM>REL`.
    ComplexToRelativistic,
    /// FEFF `CLM>RLM`.
    ComplexToReal,
    /// FEFF `RLM>CLM`.
    RealToComplex,
}

/// Coordinate system used by FEFF `iniptz` when constructing `ptz`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarizationTensorMode {
    /// Direct spherical tensor basis, with selector `1..=9` choosing one entry.
    Spherical,
    /// Cartesian products rewritten into FEFF's spherical-index tensor basis.
    Cartesian,
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

/// Port of FEFF `iniptz`: construct a polarization tensor.
///
/// The returned `3x3` matrix is indexed in signed spherical order
/// `[-1, 0, 1]` on both axes. Selector `10` is the orientational average
/// `diag(1/3, 1/3, 1/3)` in both modes.
pub fn polarization_tensor(
    selector: usize,
    mode: PolarizationTensorMode,
) -> Result<Array2<Complex>, AngularError> {
    if !(1..=10).contains(&selector) {
        return Err(AngularError::InvalidPolarizationTensorIndex { index: selector });
    }

    let mut tensor = Array2::zeros((3, 3).f());
    if selector == 10 {
        for magnetic in -1..=1 {
            tensor[(polarization_index(magnetic), polarization_index(magnetic))] =
                Complex::new(1.0 / 3.0, 0.0);
        }
        return Ok(tensor);
    }

    match mode {
        PolarizationTensorMode::Spherical => {
            let zero_based = selector - 1;
            tensor[(zero_based / 3, zero_based % 3)] = Complex::new(1.0, 0.0);
        }
        PolarizationTensorMode::Cartesian => {
            fill_cartesian_polarization_tensor(&mut tensor, selector);
        }
    }
    Ok(tensor)
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

/// Port of FEFF `KSPACE/calccgc.f90`.
///
/// Builds FEFF's linear `CGC(IKM, IS)` table for relativistic
/// Clebsch-Gordan coefficients. The branch tables mirror FEFF's `LTAB`,
/// `KAPTAB`, and `NMUETAB` initialization used by KSPACE basis transforms.
pub fn relativistic_clebsch_gordan_coefficients(
    lmax: usize,
) -> Result<RelativisticClebschGordanCoefficients, AngularError> {
    let branch_count = lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let coefficient_count = lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .and_then(|value| value.checked_mul(2))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;

    let mut orbital_momentum = Vec::with_capacity(branch_count);
    let mut kappa = Vec::with_capacity(branch_count);
    let mut spin_multiplicity = Vec::with_capacity(branch_count);
    let mut coefficients = Array2::zeros((coefficient_count, 2).f());
    let mut state = 0usize;

    for branch in 1..=branch_count {
        let orbital = branch / 2;
        let branch_kappa = if branch.is_multiple_of(2) {
            usize_to_i32(orbital)?
        } else {
            -usize_to_i32(
                orbital
                    .checked_add(1)
                    .ok_or(AngularError::IndexTooLarge { value: orbital })?,
            )?
        };
        let multiplicity = usize::try_from(branch_kappa.unsigned_abs())
            .map_err(|_| AngularError::IndexTooLarge { value: branch })?
            .checked_mul(2)
            .ok_or(AngularError::IndexTooLarge { value: branch })?;

        orbital_momentum.push(orbital);
        kappa.push(branch_kappa);
        spin_multiplicity.push(multiplicity);

        let orbital_real = usize_to_real(orbital)?;
        let mut mue = -f64::from(branch_kappa.unsigned_abs()) - 0.5;
        let two_l_plus_one = 2.0 * orbital_real + 1.0;
        for _ in 0..multiplicity {
            mue += 1.0;
            if branch_kappa < 0 {
                coefficients[(state, 0)] = ((orbital_real - mue + 0.5) / two_l_plus_one).sqrt();
                coefficients[(state, 1)] = ((orbital_real + mue + 0.5) / two_l_plus_one).sqrt();
            } else {
                coefficients[(state, 0)] = ((orbital_real + mue + 0.5) / two_l_plus_one).sqrt();
                coefficients[(state, 1)] = -((orbital_real - mue + 0.5) / two_l_plus_one).sqrt();
            }
            state += 1;
        }
    }

    Ok(RelativisticClebschGordanCoefficients {
        coefficients,
        orbital_momentum,
        kappa,
        spin_multiplicity,
    })
}

/// Port of FEFF `MKGTR/calclbcoef.f90`.
///
/// The returned `clbcoef(im, ii, is, ll)` table keeps FEFF's axis order as
/// `(mj_lmax, j_lmax, 2, lmax + 1)` and uses Fortran-order storage. Rust
/// indices are zero-based, while `ii` still represents FEFF's one-based
/// half-integer final-state angular momentum slot.
pub fn mkgtr_clebsch_gordan_coefficients(
    lmax: usize,
    j_lmax: usize,
    mj_lmax: usize,
) -> Result<Array4<Real>, AngularError> {
    if j_lmax == 0 {
        return Err(AngularError::InvalidAngularTableDimension {
            name: "j_lmax",
            value: j_lmax,
            minimum: 1,
        });
    }

    let l_count = lmax
        .checked_add(1)
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let active_j_lmax = j_lmax.min(l_count);
    let required_mj_lmax =
        checked_double_usize(active_j_lmax).ok_or(AngularError::IndexTooLarge {
            value: active_j_lmax,
        })?;
    if mj_lmax < required_mj_lmax {
        return Err(AngularError::InvalidAngularTableDimension {
            name: "mj_lmax",
            value: mj_lmax,
            minimum: required_mj_lmax,
        });
    }

    let mut coefficients = Array4::zeros((mj_lmax, j_lmax, 2, l_count).f());
    for ll in 0..l_count {
        let lnow = checked_double_usize_to_i32(ll)?;
        let active_j = j_lmax.min(
            ll.checked_add(1)
                .ok_or(AngularError::IndexTooLarge { value: ll })?,
        );
        for is in 0..=1 {
            let ms = checked_double_usize_to_i32(is)?
                .checked_sub(1)
                .ok_or(AngularError::IndexTooLarge { value: is })?;
            for ii in 1..=active_j {
                let jnow = checked_double_usize_to_i32(ii)?
                    .checked_sub(1)
                    .ok_or(AngularError::IndexTooLarge { value: ii })?;
                for im in
                    1..=checked_double_usize(ii).ok_or(AngularError::IndexTooLarge { value: ii })?
                {
                    let im_i32 = usize_to_i32(im)?;
                    let mj = checked_double_i32(
                        im_i32
                            .checked_sub(1)
                            .ok_or(AngularError::IndexTooLarge { value: im })?,
                    )?
                    .checked_sub(jnow)
                    .ok_or(AngularError::IndexTooLarge { value: im })?;
                    let neg_mj = mj
                        .checked_neg()
                        .ok_or(AngularError::IndexTooLarge { value: im })?;
                    let mut coefficient = wigner_3j(1, jnow, lnow, ms, neg_mj, 2)?;
                    let sign_argument = lnow
                        .checked_add(mj)
                        .and_then(|value| value.checked_sub(1))
                        .ok_or(AngularError::IndexTooLarge { value: ll })?;
                    if (sign_argument / 2) % 2 != 0 {
                        coefficient = -coefficient;
                    }
                    coefficients[(im - 1, ii - 1, is, ll)] = coefficient;
                }
            }
        }
    }
    Ok(coefficients)
}

/// Port of FEFF `BAND/ikapmue.f90`: one-based `(kappa, MUEM05)` state index.
///
/// `mu_minus_half` is FEFF's integer `MUEM05`, the relativistic magnetic
/// quantum number shifted by `-1/2`. FEFF stores states in kappa-branch order
/// and uses `I = 2*L*(J+1/2) + J + MUEM05 + 1`; this helper returns that same
/// one-based index with explicit validation.
pub fn relativistic_state_index_1based(
    kappa: i32,
    mu_minus_half: i32,
) -> Result<usize, AngularError> {
    if kappa == 0 || kappa == i32::MIN {
        return Err(AngularError::InvalidRelativisticKappa { kappa });
    }

    let jp05 = kappa.abs();
    if mu_minus_half < -jp05 || mu_minus_half >= jp05 {
        return Err(AngularError::RelativisticMagneticIndexOutOfRange {
            kappa,
            mu_minus_half,
        });
    }

    let orbital = if kappa < 0 { -kappa - 1 } else { kappa };
    let index = 2_i128 * i128::from(orbital) * i128::from(jp05)
        + i128::from(jp05)
        + i128::from(mu_minus_half)
        + 1;
    usize::try_from(index).map_err(|_| AngularError::IndexTooLarge { value: usize::MAX })
}

/// Port of FEFF `KSPACE/bastrans.f90` `BASTRMAT`.
///
/// The matrices convert matrix representations between real spherical
/// harmonics (`RLM`), complex spherical harmonics (`CLM`), and relativistic
/// `(kappa, mue)` states (`REL`). Storage is `ndarray` Fortran order to match
/// FEFF's column-major matrix layout.
pub fn basis_transform_matrices(lmax: usize) -> Result<BasisTransformMatrices, AngularError> {
    let (orbital_count, order) = basis_transform_dimensions(lmax)?;
    let branch_count = lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(1))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let cgc = relativistic_clebsch_gordan_coefficients(lmax)?;

    let mut complex_to_relativistic = Array2::zeros((order, order).f());
    for lnr in 0..=lmax {
        let lnr_isize =
            isize::try_from(lnr).map_err(|_| AngularError::IndexTooLarge { value: lnr })?;
        for magnetic in -lnr_isize..=lnr_isize {
            let lm = spherical_spin_index(lnr, magnetic)?;
            let mut state = 0usize;
            for branch in 1..=branch_count {
                let orbital = branch / 2;
                let jp05 = if 2 * orbital == branch {
                    orbital
                } else {
                    orbital
                        .checked_add(1)
                        .ok_or(AngularError::IndexTooLarge { value: orbital })?
                };
                let jp05_isize = isize::try_from(jp05)
                    .map_err(|_| AngularError::IndexTooLarge { value: jp05 })?;
                for muem05 in -jp05_isize..jp05_isize {
                    let muep05 = muem05 + 1;
                    if orbital == lnr {
                        if muep05 == magnetic {
                            complex_to_relativistic[(lm, state)] =
                                Complex::new(cgc.coefficients[(state, 0)], 0.0);
                        }
                        if muem05 == magnetic {
                            complex_to_relativistic[(lm + orbital_count, state)] =
                                Complex::new(cgc.coefficients[(state, 1)], 0.0);
                        }
                    }
                    state = state
                        .checked_add(1)
                        .ok_or(AngularError::IndexTooLarge { value: branch })?;
                }
            }
        }
    }

    let mut real_to_complex = Array2::zeros((order, order).f());
    let weight = 1.0 / 2.0_f64.sqrt();
    for orbital in 0..=lmax {
        let orbital_isize =
            isize::try_from(orbital).map_err(|_| AngularError::IndexTooLarge { value: orbital })?;
        for magnetic in -orbital_isize..=orbital_isize {
            let index = spherical_spin_index(orbital, magnetic)?;
            let mirror = spherical_spin_index(orbital, -magnetic)?;
            if magnetic < 0 {
                real_to_complex[(index, index)] = Complex::new(0.0, -weight);
                real_to_complex[(mirror, index)] = Complex::new(weight, 0.0);
                real_to_complex[(index + orbital_count, index + orbital_count)] =
                    Complex::new(0.0, -weight);
                real_to_complex[(mirror + orbital_count, index + orbital_count)] =
                    Complex::new(weight, 0.0);
            } else if magnetic == 0 {
                real_to_complex[(index, index)] = Complex::new(1.0, 0.0);
                real_to_complex[(index + orbital_count, index + orbital_count)] =
                    Complex::new(1.0, 0.0);
            } else {
                let sign = if magnetic % 2 == 0 { 1.0 } else { -1.0 };
                real_to_complex[(index, index)] = Complex::new(weight * sign, 0.0);
                real_to_complex[(mirror, index)] = Complex::new(0.0, weight * sign);
                real_to_complex[(index + orbital_count, index + orbital_count)] =
                    Complex::new(weight * sign, 0.0);
                real_to_complex[(mirror + orbital_count, index + orbital_count)] =
                    Complex::new(0.0, weight * sign);
            }
        }
    }

    let real_to_relativistic = fortran_complex_matrix(
        complex_matmul(real_to_complex.view(), complex_to_relativistic.view()).view(),
    );

    Ok(BasisTransformMatrices {
        lmax,
        order,
        real_to_complex,
        complex_to_relativistic,
        real_to_relativistic,
    })
}

/// Port of FEFF `KSPACE/bastrans.f90` `CHANGEREP`.
///
/// Converts a square matrix between FEFF's `RLM`, `CLM`, and `REL`
/// representations using the transform bundle returned by
/// [`basis_transform_matrices`]. The output uses Fortran-order `ndarray`
/// storage like FEFF matrix work arrays.
pub fn change_basis_representation(
    input: ArrayView2<'_, Complex>,
    mode: BasisTransformMode,
    transforms: &BasisTransformMatrices,
) -> Result<ComplexMat, AngularError> {
    validate_basis_matrix_shape("input", input, transforms.order)?;
    validate_basis_matrix_shape(
        "real_to_complex",
        transforms.real_to_complex.view(),
        transforms.order,
    )?;
    validate_basis_matrix_shape(
        "complex_to_relativistic",
        transforms.complex_to_relativistic.view(),
        transforms.order,
    )?;
    validate_basis_matrix_shape(
        "real_to_relativistic",
        transforms.real_to_relativistic.view(),
        transforms.order,
    )?;

    let (left, right, left_conjugate, right_conjugate) = match mode {
        BasisTransformMode::RelativisticToReal => (
            transforms.real_to_relativistic.view(),
            transforms.real_to_relativistic.view(),
            false,
            true,
        ),
        BasisTransformMode::RealToRelativistic => (
            transforms.real_to_relativistic.view(),
            transforms.real_to_relativistic.view(),
            true,
            false,
        ),
        BasisTransformMode::RelativisticToComplex => (
            transforms.complex_to_relativistic.view(),
            transforms.complex_to_relativistic.view(),
            false,
            true,
        ),
        BasisTransformMode::ComplexToRelativistic => (
            transforms.complex_to_relativistic.view(),
            transforms.complex_to_relativistic.view(),
            true,
            false,
        ),
        BasisTransformMode::ComplexToReal => (
            transforms.real_to_complex.view(),
            transforms.real_to_complex.view(),
            false,
            true,
        ),
        BasisTransformMode::RealToComplex => (
            transforms.real_to_complex.view(),
            transforms.real_to_complex.view(),
            true,
            false,
        ),
    };

    let left_matrix = if left_conjugate {
        conjugate_transpose(left)
    } else {
        left.to_owned()
    };
    let right_matrix = if right_conjugate {
        conjugate_transpose(right)
    } else {
        right.to_owned()
    };
    let work = complex_matmul(left_matrix.view(), input);
    Ok(fortran_complex_matrix(
        complex_matmul(work.view(), right_matrix.view()).view(),
    ))
}

fn basis_transform_dimensions(lmax: usize) -> Result<(usize, usize), AngularError> {
    let orbital_count = lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let order = orbital_count
        .checked_mul(2)
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    Ok((orbital_count, order))
}

fn spherical_spin_index(orbital: usize, magnetic: isize) -> Result<usize, AngularError> {
    let orbital_isize =
        isize::try_from(orbital).map_err(|_| AngularError::IndexTooLarge { value: orbital })?;
    if magnetic < -orbital_isize || magnetic > orbital_isize {
        return Err(AngularError::MagneticIndexOutOfRange {
            magnetic,
            lmax: orbital,
        });
    }
    let base = orbital
        .checked_mul(
            orbital
                .checked_add(1)
                .ok_or(AngularError::IndexTooLarge { value: orbital })?,
        )
        .ok_or(AngularError::IndexTooLarge { value: orbital })?;
    let shifted = isize::try_from(base)
        .map_err(|_| AngularError::IndexTooLarge { value: orbital })?
        + magnetic;
    usize::try_from(shifted).map_err(|_| AngularError::MagneticIndexOutOfRange {
        magnetic,
        lmax: orbital,
    })
}

fn validate_basis_matrix_shape(
    name: &'static str,
    matrix: ArrayView2<'_, Complex>,
    expected: usize,
) -> Result<(), AngularError> {
    if matrix.shape() != [expected, expected] {
        return Err(AngularError::InvalidBasisTransformShape {
            name,
            rows: matrix.nrows(),
            columns: matrix.ncols(),
            expected,
        });
    }
    Ok(())
}

fn conjugate_transpose(matrix: ArrayView2<'_, Complex>) -> ComplexMat {
    Array2::from_shape_fn((matrix.ncols(), matrix.nrows()).f(), |(row, column)| {
        matrix[(column, row)].conj()
    })
}

fn fortran_complex_matrix(matrix: ArrayView2<'_, Complex>) -> ComplexMat {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()).f(), |(row, column)| {
        matrix[(row, column)]
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

fn checked_double_usize(value: usize) -> Option<usize> {
    value.checked_mul(2)
}

fn checked_double_usize_to_i32(value: usize) -> Result<i32, AngularError> {
    usize_to_i32(checked_double_usize(value).ok_or(AngularError::IndexTooLarge { value })?)
}

fn checked_double_i32(value: i32) -> Result<i32, AngularError> {
    value
        .checked_mul(2)
        .ok_or(AngularError::IndexTooLarge { value: usize::MAX })
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

fn fill_cartesian_polarization_tensor(tensor: &mut Array2<Complex>, selector: usize) {
    let one = Complex::new(1.0, 0.0);
    let imaginary = Complex::new(0.0, 1.0);
    let half = 0.5;
    let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();

    match selector {
        1 => {
            set_polarization(tensor, 1, 1, one * half);
            set_polarization(tensor, -1, -1, one * half);
            set_polarization(tensor, -1, 1, -one * half);
            set_polarization(tensor, 1, -1, -one * half);
        }
        2 => {
            set_polarization(tensor, 1, 1, imaginary * half);
            set_polarization(tensor, -1, -1, -imaginary * half);
            set_polarization(tensor, -1, 1, -imaginary * half);
            set_polarization(tensor, 1, -1, imaginary * half);
        }
        3 => {
            set_polarization(tensor, -1, 0, one * inv_sqrt_two);
            set_polarization(tensor, 1, 0, -one * inv_sqrt_two);
        }
        4 => {
            set_polarization(tensor, 1, 1, -imaginary * half);
            set_polarization(tensor, -1, -1, imaginary * half);
            set_polarization(tensor, -1, 1, -imaginary * half);
            set_polarization(tensor, 1, -1, imaginary * half);
        }
        5 => {
            set_polarization(tensor, 1, 1, one * half);
            set_polarization(tensor, -1, -1, one * half);
            set_polarization(tensor, -1, 1, one * half);
            set_polarization(tensor, 1, -1, one * half);
        }
        6 => {
            set_polarization(tensor, -1, 0, -imaginary * inv_sqrt_two);
            set_polarization(tensor, 1, 0, -imaginary * inv_sqrt_two);
        }
        7 => {
            set_polarization(tensor, 0, -1, one * inv_sqrt_two);
            set_polarization(tensor, 0, 1, -one * inv_sqrt_two);
        }
        8 => {
            set_polarization(tensor, 0, -1, imaginary * inv_sqrt_two);
            set_polarization(tensor, 0, 1, imaginary * inv_sqrt_two);
        }
        9 => set_polarization(tensor, 0, 0, one),
        _ => {}
    }
}

fn set_polarization(tensor: &mut Array2<Complex>, row: isize, column: isize, value: Complex) {
    tensor[(polarization_index(row), polarization_index(column))] = value;
}

fn polarization_index(magnetic: isize) -> usize {
    match magnetic {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 0,
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
mod tests;
