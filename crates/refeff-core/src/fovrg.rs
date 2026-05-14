//! FEFF FOVRG numerical helpers.
//!
//! These routines cover small pieces of the relativistic radial solver that can
//! be validated independently of the full `dfovrg` integration path.

use ndarray::{Array1, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Complex, ComplexVec, Real, RealVec};

// `diff.f90` uses unsuffixed Fortran real literals in these stencils. Preserve
// their default-real rounding before widening to the Rust `Real` type.
const F77_REAL_HALF: Real = 0.5_f32 as Real;
const F77_REAL_ONE_POINT_TWO: Real = 1.2_f32 as Real;
const F77_REAL_ONE_POINT_FIVE: Real = 1.5_f32 as Real;
const F77_REAL_TWO: Real = 2.0_f32 as Real;
const F77_REAL_TWO_POINT_FOUR_FIVE: Real = 2.45_f32 as Real;
const F77_REAL_THREE_POINT_SEVEN_FIVE: Real = 3.75_f32 as Real;
const F77_REAL_SIX: Real = 6.0_f32 as Real;
const F77_REAL_SIX_AND_TWO_THIRDS: Real = 6.666_666_5_f32 as Real;
const F77_REAL_SEVEN_POINT_FIVE: Real = 7.5_f32 as Real;
const F77_REAL_EIGHT: Real = 8.0_f32 as Real;
const F77_REAL_TWELVE: Real = 12.0_f32 as Real;
const F77_REAL_ONE_SIXTH: Real = 0.166_666_67_f32 as Real;

/// Inputs for FEFF `FOVRG/diff.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgC3DerivativeInput<'a> {
    /// Complex potential values `v`.
    pub potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic grid step `dx`.
    pub delta: Real,
    /// Number of active radial rows `n`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/yzktec.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgYkZkTransformInput<'a> {
    /// Tabulated source function `f`.
    pub source: ArrayView1<'a, Complex>,
    /// Origin development coefficients `af` for [`FovrgYkZkTransformInput::source`].
    pub source_coefficients: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Initial origin power `ap`; FEFF uses only its real part before overwriting it.
    pub initial_power: Complex,
    /// Logarithmic radial step `h`.
    pub step: Real,
    /// Multipole order `k`.
    pub angular_momentum: usize,
    /// Number of active origin coefficients `nd`.
    pub coefficient_count: usize,
    /// Number of active source samples `np`; FEFF clamps this to `idim - 1`.
    pub source_len: usize,
    /// Active radial capacity `idim`.
    pub active_len: usize,
    /// Optional tail correction `dyzk`.
    pub tail_correction: Complex,
}

/// Inputs for FEFF `FOVRG/yzkrdc.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgYkZkExchangeInput<'a> {
    /// Bound-orbital large radial component `cg(:, i)`.
    pub large_component: ArrayView1<'a, Real>,
    /// Bound-orbital small radial component `cp(:, i)`.
    pub small_component: ArrayView1<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(:, i)`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(:, i)`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Partner large radial component `ps`.
    pub partner_large_component: ArrayView1<'a, Complex>,
    /// Partner small radial component `qs`.
    pub partner_small_component: ArrayView1<'a, Complex>,
    /// Partner large origin coefficients `aps`.
    pub partner_large_coefficients: ArrayView1<'a, Complex>,
    /// Partner small origin coefficients `aqs`.
    pub partner_small_coefficients: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Bound-orbital origin power `fl(i)`.
    pub orbital_power: Real,
    /// Partner origin power `flps`.
    pub partner_power: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Multipole order `k`.
    pub angular_momentum: usize,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Bound-orbital maximum tabulated row `nmax(i)`.
    pub orbital_len: usize,
    /// Global active source row count `np`.
    pub source_len: usize,
    /// Active radial capacity `idim`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/dsordc.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOverlapIntegralInput<'a> {
    /// Radial large-component integrand `dg`.
    pub large_integrand: ArrayView1<'a, Complex>,
    /// Radial small-component integrand `dp`.
    pub small_integrand: ArrayView1<'a, Complex>,
    /// Origin coefficients `ag` for [`FovrgOverlapIntegralInput::large_integrand`].
    pub large_integrand_coefficients: ArrayView1<'a, Complex>,
    /// Origin coefficients `ap` for [`FovrgOverlapIntegralInput::small_integrand`].
    pub small_integrand_coefficients: ArrayView1<'a, Complex>,
    /// Bound-orbital large radial component `cg(:, j)`.
    pub large_component: ArrayView1<'a, Real>,
    /// Bound-orbital small radial component `cp(:, j)`.
    pub small_component: ArrayView1<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(:, j)`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(:, j)`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Origin power `a` of the incoming integrand.
    pub integrand_power: Real,
    /// Bound-orbital origin power `fl(j)`.
    pub orbital_power: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Number of active radial rows `idim`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/ortdac.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOrthogonalizationInput<'a> {
    /// Target large radial component `ps`.
    pub target_large_component: ArrayView1<'a, Complex>,
    /// Target small radial component `qs`.
    pub target_small_component: ArrayView1<'a, Complex>,
    /// Target large origin coefficients `aps`.
    pub target_large_coefficients: ArrayView1<'a, Complex>,
    /// Target small origin coefficients `aqs`.
    pub target_small_coefficients: ArrayView1<'a, Complex>,
    /// Bound-orbital large radial components `cg(row, orbital)`.
    pub bound_large_components: ArrayView2<'a, Real>,
    /// Bound-orbital small radial components `cp(row, orbital)`.
    pub bound_small_components: ArrayView2<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(coefficient, orbital)`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(coefficient, orbital)`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// Bound-orbital origin powers `fl`.
    pub orbital_powers: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Target origin power `fl(norb)`.
    pub target_power: Real,
    /// Target relativistic kappa `ikap`.
    pub target_kappa: i32,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Number of active radial rows `idim`.
    pub active_len: usize,
    /// Number of bound orbitals, equivalent to FEFF `norb - 1`.
    pub bound_orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/potex.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgExchangePotentialInput<'a> {
    /// Target large radial component `ps`.
    pub target_large_component: ArrayView1<'a, Complex>,
    /// Target small radial component `qs`.
    pub target_small_component: ArrayView1<'a, Complex>,
    /// Target large origin coefficients `aps`.
    pub target_large_coefficients: ArrayView1<'a, Complex>,
    /// Target small origin coefficients `aqs`.
    pub target_small_coefficients: ArrayView1<'a, Complex>,
    /// Bound-orbital large radial components `cg(row, orbital)`.
    pub bound_large_components: ArrayView2<'a, Real>,
    /// Bound-orbital small radial components `cp(row, orbital)`.
    pub bound_small_components: ArrayView2<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(coefficient, orbital)`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(coefficient, orbital)`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// FEFF angular coefficients `afgkc(kap(target), orbital, index)`.
    pub angular_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital origin powers `fl`.
    pub orbital_powers: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// Bound-orbital maximum tabulated rows `nmax`.
    pub orbital_lengths: ArrayView1<'a, usize>,
    /// Bound-orbital normalization factors `fix`.
    pub normalization: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Target origin power `fl(norb)`.
    pub target_power: Real,
    /// Target relativistic kappa `kap(norb)`.
    pub target_kappa: i32,
    /// Target normalization factor `fix(norb)`.
    pub target_normalization: Real,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `np`, the source grid limit passed through `yzkrdc`.
    pub source_len: usize,
    /// Number of active radial rows `idim`.
    pub active_len: usize,
    /// FEFF `jri`, rows retained in the output potentials.
    pub radial_output_count: usize,
    /// Number of bound orbitals, equivalent to FEFF `norb - 1`.
    pub bound_orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/potdvp.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgPotentialDevelopmentInput<'a> {
    /// Nuclear potential development coefficients `anoy`.
    pub nuclear_coefficients: ArrayView1<'a, Real>,
    /// Bound-orbital large-component coefficients `bg(coefficient, orbital)`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small-component coefficients `bp(coefficient, orbital)`.
    pub small_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// FEFF normalization factors `fix`.
    pub normalization: ArrayView1<'a, Real>,
    /// Radial grid `dr`; only the first point enters this kernel.
    pub radii: ArrayView1<'a, Real>,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `norb`; bound orbitals `1..norb-1` contribute.
    pub orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/nucdec.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgNuclearPotentialInput {
    /// Nuclear charge `dz`.
    pub nuclear_charge: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// FEFF input/output `dr1`, the first tabulation radius multiplied by `dz`.
    pub first_radius_times_charge: Real,
    /// Number of radial tabulation points `np`.
    pub radial_count: usize,
    /// Number of origin development coefficients `ndor`.
    pub coefficient_count: usize,
}

/// Output from FEFF `FOVRG/yzktec.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgYkZkTransform {
    /// Transformed `yk` values, zero-filled after [`FovrgYkZkTransform::computed_len`].
    pub yk: ComplexVec,
    /// Intermediate `zk` values, zero-filled after [`FovrgYkZkTransform::computed_len`].
    pub zk: ComplexVec,
    /// Mutated `af` development coefficients for `yk`.
    pub yk_coefficients: ComplexVec,
    /// Development coefficients `ag` for `zk`.
    pub zk_coefficients: ComplexVec,
    /// FEFF output scalar `ap`, the leading origin constant for `yk`.
    pub origin_constant: Complex,
    /// Number of meaningful radial rows, equivalent to clamped `np + 1`.
    pub computed_len: usize,
}

/// Output from FEFF `FOVRG/nucdec.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgNuclearPotential {
    /// Origin development coefficients `av`.
    pub development_coefficients: RealVec,
    /// Radial grid `dr`.
    pub radii: RealVec,
    /// Nuclear potential `dv`.
    pub potential: RealVec,
    /// FEFF 1-based nuclear-radius index `nuc`.
    pub nucleus_index: usize,
    /// FEFF output `dr1`.
    pub first_radius_times_charge: Real,
}

/// Output from FEFF `FOVRG/ortdac.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgOrthogonalization {
    /// Orthogonalized target large radial component `ps`.
    pub large_component: ComplexVec,
    /// Orthogonalized target small radial component `qs`.
    pub small_component: ComplexVec,
    /// Orthogonalized target large origin coefficients `aps`.
    pub large_coefficients: ComplexVec,
    /// Orthogonalized target small origin coefficients `aqs`.
    pub small_coefficients: ComplexVec,
    /// Per-bound-orbital overlap coefficients used for subtraction.
    pub overlaps: ComplexVec,
}

/// Output from FEFF `FOVRG/potex.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgExchangePotential {
    /// Large-component exchange potential `eg`.
    pub large_potential: ComplexVec,
    /// Small-component exchange potential `ep`.
    pub small_potential: ComplexVec,
    /// Large-component origin coefficients `ceg`.
    pub large_coefficients: ComplexVec,
    /// Small-component origin coefficients `cep`.
    pub small_coefficients: ComplexVec,
}

/// Output from FEFF `FOVRG/potdvp.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgPotentialDevelopment {
    /// Potential development coefficients `av` after FEFF's division by `cl`.
    pub potential_coefficients: ComplexVec,
    /// Transformed density coefficients `ag` before division by `cl`.
    pub density_coefficients: RealVec,
    /// FEFF output `ap(1)` before division by `cl`.
    pub origin_correction: Real,
}

/// Error returned by FOVRG helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FovrgError {
    /// FEFF `diff` uses rows 1..=8 in the first two one-sided stencils.
    #[error("FOVRG {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    /// Active rows must fit in every input array.
    #[error("FOVRG active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Counts that are later converted to FEFF integer exponents must fit.
    #[error("FOVRG {name} count {actual} exceeds maximum {maximum}")]
    CountTooLarge {
        name: &'static str,
        actual: usize,
        maximum: usize,
    },
    /// Simpson integration in FEFF `dsordc` advances by two rows.
    #[error("FOVRG {name} count {actual} must be odd")]
    CountMustBeOdd { name: &'static str, actual: usize },
    /// Scalar inputs must be finite.
    #[error("FOVRG {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Positive scalar inputs must be finite and greater than zero.
    #[error("FOVRG {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Divisor-like scalar inputs must be nonzero.
    #[error("FOVRG {name} must be nonzero")]
    ZeroInput { name: &'static str },
    /// FEFF formulas with a zero denominator are reported instead of evaluated.
    #[error("FOVRG denominator {name} is zero")]
    ZeroDenominator { name: &'static str },
    /// Radii must be positive.
    #[error("FOVRG radius row {row} must be positive, got {value}")]
    NonPositiveRadius { row: usize, value: Real },
    /// Kappa values are nonzero quantum numbers in FEFF radial kernels.
    #[error("FOVRG {name} row {row} has invalid quantum number {value}")]
    InvalidQuantumNumber {
        name: &'static str,
        row: usize,
        value: i32,
    },
    /// Complex inputs must be finite.
    #[error("FOVRG {name} row {row} must be finite, got {value}")]
    NonFiniteComplexInput {
        name: &'static str,
        row: usize,
        value: Complex,
    },
    /// Real vector inputs must be finite.
    #[error("FOVRG {name} row {row} must be finite, got {value}")]
    NonFiniteRealInput {
        name: &'static str,
        row: usize,
        value: Real,
    },
    /// Complex potential values must be finite.
    #[error("FOVRG potential row {row} must be finite, got {value}")]
    NonFinitePotential { row: usize, value: Complex },
    /// Output values must remain finite.
    #[error("FOVRG derivative row {row} must be finite, got {value}")]
    NonFiniteResult { row: usize, value: Complex },
}

/// Port of `FOVRG/diff.f90`: C3 radial derivative term.
///
/// FEFF first differentiates `v(r) * r^2` with one-sided boundary stencils and
/// a centered fourth-order interior stencil, then returns
/// `(d(v*r^2)/dx - 2*v*r^2) / r * (kap+1) / cl`.
pub fn fovrg_c3_derivative(input: FovrgC3DerivativeInput<'_>) -> Result<ComplexVec, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 8)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_nonzero_finite("delta", input.delta)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;

    for row in 0..input.active_len {
        validate_radius(row, input.radii[row])?;
        validate_potential(row, input.potential[row])?;
    }

    let vt = Array1::from_iter(
        (0..input.active_len).map(|row| input.potential[row] * input.radii[row].powi(2)),
    );
    let mut derivative = Array1::<Complex>::zeros(input.active_len);

    derivative[0] = ((F77_REAL_SIX * vt[1]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[3]
        + F77_REAL_ONE_POINT_TWO * vt[5])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[0]
            + F77_REAL_SEVEN_POINT_FIVE * vt[2]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[4]
            + F77_REAL_ONE_SIXTH * vt[6]))
        / input.delta;
    derivative[1] = ((F77_REAL_SIX * vt[2]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[4]
        + F77_REAL_ONE_POINT_TWO * vt[6])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[1]
            + F77_REAL_SEVEN_POINT_FIVE * vt[3]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[5]
            + F77_REAL_ONE_SIXTH * vt[7]))
        / input.delta;

    for row in 2..input.active_len - 2 {
        derivative[row] = ((vt[row - 2] + F77_REAL_EIGHT * vt[row + 1])
            - (F77_REAL_EIGHT * vt[row - 1] + vt[row + 2]))
            / F77_REAL_TWELVE
            / input.delta;
    }

    let last = input.active_len - 1;
    derivative[last - 1] = (vt[last] - vt[last - 2]) / (F77_REAL_TWO * input.delta);
    derivative[last] = (F77_REAL_HALF * vt[last - 2] - F77_REAL_TWO * vt[last - 1]
        + F77_REAL_ONE_POINT_FIVE * vt[last])
        / input.delta;

    let scale = ((input.kappa as f32 + 1.0_f32) as Real) / input.speed_of_light;
    let mut output = Array1::<Complex>::zeros(input.active_len);
    for row in 0..input.active_len {
        let value = (derivative[row] - F77_REAL_TWO * vt[row]) / input.radii[row] * scale;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(FovrgError::NonFiniteResult { row, value });
        }
        output[row] = value;
    }
    Ok(output)
}

/// Port of `FOVRG/yzktec.f90`: build the radial `yk` and `zk` exchange kernels.
///
/// FEFF evaluates
/// `zk(r) = r^-k * integral(0..r, f(u) * u^k du)` and then
/// `yk(r) = zk(r) + r^(k+1) * integral(r..infinity, f(u) * u^(-k-1) du)`.
/// The first integration runs outward on the logarithmic radial mesh and the
/// second runs backward from FEFF's clamped `np + 1` endpoint.
pub fn fovrg_yk_zk_transform(
    input: FovrgYkZkTransformInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len("source", input.active_len, input.source.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_complex_input("initial_power", 0, input.initial_power)?;
    validate_complex_input("tail_correction", 0, input.tail_correction)?;
    if input.angular_momentum > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "angular_momentum",
            actual: input.angular_momentum,
            maximum: i32::MAX as usize - 1,
        });
    }

    let source_len = input.source_len.min(input.active_len - 1);
    let computed_len = source_len + 1;
    for row in 0..source_len {
        validate_complex_input("source", row, input.source[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "source_coefficients",
            coefficient,
            input.source_coefficients[coefficient],
        )?;
    }

    let k = input.angular_momentum;
    let k_real = k as Real;
    let k_i32 = k as i32;
    let k_plus_one_i32 = (k + 1) as i32;
    let singular_tolerance = 1.0e-5_f32 as Real;
    let mut yk = Array1::<Complex>::zeros(input.active_len);
    let mut zk = Array1::<Complex>::zeros(input.active_len);
    let mut yk_coefficients = Array1::<Complex>::from_iter(
        (0..input.coefficient_count).map(|row| input.source_coefficients[row]),
    );
    let mut zk_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    for row in 0..source_len {
        yk[row] = input.source[row];
    }

    let mut power = input.initial_power.re;
    let mut origin_constant = Complex::new(0.0, 0.0);
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_nonzero_denominator("yk_zk_origin_zk", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != Complex::new(0.0, 0.0) {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;

            let yk_denominator = power - k_real - 1.0;
            if yk_denominator.abs() <= singular_tolerance {
                yk_coefficients[coefficient] = Complex::new(0.0, 0.0);
                power -= 1.0;
            } else {
                yk_coefficients[coefficient] =
                    ((k + k + 1) as Real) * zk_coefficients[coefficient] / yk_denominator;
            }
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] *= input.radii[row];
    }

    let hk = input.step * k_real;
    let e = (-input.step).exp();
    let ehk = e.powi(k_i32);
    let b1 = if k == 0 {
        input.step / 2.0
    } else {
        (ehk - 1.0 + hk) / (hk * k_real)
    };
    let b0 = input.step - (1.0 + hk) * b1;
    for row in 0..source_len {
        zk[row + 1] = zk[row] * ehk + b0 * yk[row] + b1 * yk[row + 1];
    }

    yk[source_len] = zk[source_len] + input.tail_correction;
    let backward_ehk = ehk * e;
    let backward_hk = hk + input.step;
    let backward_order = (k + k + 1) as Real;
    let backward_b1 =
        backward_order * (backward_ehk - 1.0 + backward_hk) / (backward_hk * (k_real + 1.0));
    let backward_b0 = backward_order * input.step - (1.0 + backward_hk) * backward_b1;
    for row in (0..source_len).rev() {
        yk[row] = yk[row + 1] * backward_ehk + backward_b0 * zk[row + 1] + backward_b1 * zk[row];
    }

    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);
    validate_complex_result("origin_constant", 0, origin_constant)?;
    for row in 0..computed_len {
        validate_complex_result("yk", row, yk[row])?;
        validate_complex_result("zk", row, zk[row])?;
    }
    for row in 0..input.coefficient_count {
        validate_complex_result("yk_coefficients", row, yk_coefficients[row])?;
        validate_complex_result("zk_coefficients", row, zk_coefficients[row])?;
    }

    Ok(FovrgYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_len,
    })
}

/// Port of `FOVRG/yzkrdc.f90`: construct exchange source terms and `yk/zk`.
///
/// FEFF forms `f = cg_i * ps + cp_i * qs`, builds origin coefficients from the
/// products of the large/small development polynomials, and then delegates the
/// radial integrations to `yzktec`.
pub fn fovrg_yk_zk_exchange(
    input: FovrgYkZkExchangeInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("orbital_len", input.orbital_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len(
        "partner_large_component",
        input.active_len,
        input.partner_large_component.len(),
    )?;
    validate_active_len(
        "partner_small_component",
        input.active_len,
        input.partner_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_active_len(
        "partner_large_coefficients",
        input.coefficient_count,
        input.partner_large_coefficients.len(),
    )?;
    validate_active_len(
        "partner_small_coefficients",
        input.coefficient_count,
        input.partner_small_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_finite("partner_power", input.partner_power)?;

    let source_len = input
        .orbital_len
        .min(input.source_len)
        .min(input.active_len - 1);
    for row in 0..source_len {
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_complex_input(
            "partner_large_component",
            row,
            input.partner_large_component[row],
        )?;
        validate_complex_input(
            "partner_small_component",
            row,
            input.partner_small_component[row],
        )?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_large_coefficients",
            coefficient,
            input.partner_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_small_coefficients",
            coefficient,
            input.partner_small_coefficients[coefficient],
        )?;
    }

    let source = Array1::from_iter((0..input.active_len).map(|row| {
        if row < source_len {
            input.large_component[row] * input.partner_large_component[row]
                + input.small_component[row] * input.partner_small_component[row]
        } else {
            Complex::new(0.0, 0.0)
        }
    }));
    let source_coefficients = Array1::from_iter((1..=input.coefficient_count).map(|count| {
        complex_real_product_coefficient(
            input.partner_large_coefficients,
            input.large_coefficients,
            count,
        ) + complex_real_product_coefficient(
            input.partner_small_coefficients,
            input.small_coefficients,
            count,
        )
    }));

    fovrg_yk_zk_transform(FovrgYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: Complex::new(input.orbital_power + input.partner_power, 0.0),
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len,
        active_len: input.active_len,
        tail_correction: Complex::new(0.0, 0.0),
    })
}

/// Port of `FOVRG/dsordc.f90`: complex radial overlap integral.
///
/// FEFF forms `hg = dg * cg_j + dp * cp_j`, integrates `hg(r) * r` over the
/// logarithmic radial mesh with its Simpson stencil, and adds the analytic
/// origin contribution from the product of the large/small development
/// coefficients.
pub fn fovrg_overlap_integral(input: FovrgOverlapIntegralInput<'_>) -> Result<Complex, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "large_integrand",
        input.active_len,
        input.large_integrand.len(),
    )?;
    validate_active_len(
        "small_integrand",
        input.active_len,
        input.small_integrand.len(),
    )?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_integrand_coefficients",
        input.coefficient_count,
        input.large_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "small_integrand_coefficients",
        input.coefficient_count,
        input.small_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite("integrand_power", input.integrand_power)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input("large_integrand", row, input.large_integrand[row])?;
        validate_complex_input("small_integrand", row, input.small_integrand[row])?;
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "large_integrand_coefficients",
            coefficient,
            input.large_integrand_coefficients[coefficient],
        )?;
        validate_complex_input(
            "small_integrand_coefficients",
            coefficient,
            input.small_integrand_coefficients[coefficient],
        )?;
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
    }

    let mixed_integrand = Array1::from_iter((0..input.active_len).map(|row| {
        (input.large_integrand[row] * input.large_component[row]
            + input.small_integrand[row] * input.small_component[row])
            * input.radii[row]
    }));

    let simpson_sum = (1..input.active_len - 1)
        .step_by(2)
        .fold(Complex::new(0.0, 0.0), |sum, row| {
            sum + mixed_integrand[row] + mixed_integrand[row] + mixed_integrand[row + 1]
        });
    let mut integral = input.step
        * (simpson_sum + simpson_sum + mixed_integrand[0] - mixed_integrand[input.active_len - 1])
        / 3.0;

    let mut origin_power = input.integrand_power + input.orbital_power;
    for coefficient in 1..=input.coefficient_count {
        origin_power += 1.0;
        validate_nonzero_denominator("overlap_origin_power", origin_power)?;
        let origin_coefficient = complex_real_product_coefficient(
            input.large_integrand_coefficients,
            input.large_coefficients,
            coefficient,
        ) + complex_real_product_coefficient(
            input.small_integrand_coefficients,
            input.small_coefficients,
            coefficient,
        );
        integral += origin_coefficient * input.radii[0].powf(origin_power) / origin_power;
    }
    validate_complex_result("overlap_integral", 0, integral)?;
    Ok(integral)
}

/// Port of `FOVRG/ortdac.f90`: Schmidt orthogonalization against bound orbitals.
///
/// FEFF walks the bound orbitals in order, skips orbitals whose kappa differs
/// from `ikap` or whose occupation is not positive, computes the current
/// overlap with `dsordc`, and subtracts that overlap from both the radial
/// target arrays and their origin development coefficients.
pub fn fovrg_schmidt_orthogonalize(
    input: FovrgOrthogonalizationInput<'_>,
) -> Result<FovrgOrthogonalization, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
    }

    let mut large_component = input.target_large_component.to_owned();
    let mut small_component = input.target_small_component.to_owned();
    let mut large_coefficients = input.target_large_coefficients.to_owned();
    let mut small_coefficients = input.target_small_coefficients.to_owned();
    let mut overlaps = Array1::<Complex>::zeros(input.bound_orbital_count);

    for orbital in 0..input.bound_orbital_count {
        if input.kappa[orbital] != input.target_kappa || input.electron_counts[orbital] <= 0.0 {
            continue;
        }

        let overlap = fovrg_overlap_integral(FovrgOverlapIntegralInput {
            large_integrand: large_component.view(),
            small_integrand: small_component.view(),
            large_integrand_coefficients: large_coefficients.view(),
            small_integrand_coefficients: small_coefficients.view(),
            large_component: input.bound_large_components.column(orbital),
            small_component: input.bound_small_components.column(orbital),
            large_coefficients: input.bound_large_coefficients.column(orbital),
            small_coefficients: input.bound_small_coefficients.column(orbital),
            radii: input.radii,
            integrand_power: input.target_power,
            orbital_power: input.orbital_powers[orbital],
            step: input.step,
            coefficient_count: input.coefficient_count,
            active_len: input.active_len,
        })?;
        overlaps[orbital] = overlap;

        for row in 0..input.active_len {
            large_component[row] -= overlap * input.bound_large_components[(row, orbital)];
            small_component[row] -= overlap * input.bound_small_components[(row, orbital)];
        }
        for coefficient in 0..input.coefficient_count {
            large_coefficients[coefficient] -=
                overlap * input.bound_large_coefficients[(coefficient, orbital)];
            small_coefficients[coefficient] -=
                overlap * input.bound_small_coefficients[(coefficient, orbital)];
        }
    }

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for orbital in 0..input.bound_orbital_count {
        validate_complex_result("overlaps", orbital, overlaps[orbital])?;
    }

    Ok(FovrgOrthogonalization {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        overlaps,
    })
}

/// Port of `FOVRG/potex.f90`: exchange-potential accumulation.
///
/// FEFF loops over bound orbitals and allowed multipoles, obtains the `yk`
/// exchange kernel from `yzkrdc`, accumulates the radial exchange potentials
/// `eg/ep`, updates their origin development coefficients `ceg/cep`, and
/// finally divides retained rows and coefficients by `cl`.
pub fn fovrg_exchange_potential(
    input: FovrgExchangePotentialInput<'_>,
) -> Result<FovrgExchangePotential, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_matrix_rows(
        "angular_coefficients",
        input.bound_orbital_count,
        input.angular_coefficients.shape()[0],
    )?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_lengths",
        input.bound_orbital_count,
        input.orbital_lengths.len(),
    )?;
    validate_active_len(
        "normalization",
        input.bound_orbital_count,
        input.normalization.len(),
    )?;
    validate_active_len(
        "radial_output_count",
        input.radial_output_count,
        input.active_len,
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_nonzero_finite("target_normalization", input.target_normalization)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_count_at_least("orbital_length", input.orbital_lengths[orbital], 1)?;
        for index in 0..input.angular_coefficients.shape()[1] {
            validate_real_input(
                "angular_coefficients",
                orbital,
                input.angular_coefficients[(orbital, index)],
            )?;
        }
    }

    let target_j = target_j_value(input.target_kappa);
    let mut large_potential = Array1::<Complex>::zeros(input.active_len);
    let mut small_potential = Array1::<Complex>::zeros(input.active_len);
    let mut large_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut small_coefficients = Array1::<Complex>::zeros(input.coefficient_count);

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (bound_j + target_j) / 2;
        let mut multipole = bound_j.abs_diff(max_multipole);
        if (input.kappa[orbital] < 0) != (input.target_kappa < 0) {
            multipole += 1;
        }
        let min_multipole = multipole;

        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            validate_matrix_cols(
                "angular_coefficients",
                angular_index + 1,
                input.angular_coefficients.shape()[1],
            )?;
            let angular_coefficient = input.angular_coefficients[(orbital, angular_index)];
            if angular_coefficient != 0.0 {
                let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                    large_component: input.bound_large_components.column(orbital),
                    small_component: input.bound_small_components.column(orbital),
                    large_coefficients: input.bound_large_coefficients.column(orbital),
                    small_coefficients: input.bound_small_coefficients.column(orbital),
                    partner_large_component: input.target_large_component,
                    partner_small_component: input.target_small_component,
                    partner_large_coefficients: input.target_large_coefficients,
                    partner_small_coefficients: input.target_small_coefficients,
                    radii: input.radii,
                    orbital_power: input.orbital_powers[orbital],
                    partner_power: input.target_power,
                    step: input.step,
                    angular_momentum: multipole,
                    coefficient_count: input.coefficient_count,
                    orbital_len: input.orbital_lengths[orbital],
                    source_len: input.source_len,
                    active_len: input.active_len,
                })?;

                for row in 0..input.active_len {
                    large_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_large_components[(row, orbital)];
                    small_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_small_components[(row, orbital)];
                }

                if let Some(coefficient_start) = exchange_coefficient_start(
                    multipole,
                    input.kappa[orbital],
                    input.target_kappa,
                    input.target_power,
                )
                .filter(|&start| start <= input.coefficient_count)
                {
                    for coefficient in coefficient_start..=input.coefficient_count {
                        let target_row = coefficient - 1;
                        let bound_row = coefficient - coefficient_start;
                        let scale = angular_coefficient
                            * transform.origin_constant
                            * input.normalization[orbital]
                            / input.target_normalization;
                        large_coefficients[target_row] +=
                            input.bound_large_coefficients[(bound_row, orbital)] * scale;
                        small_coefficients[target_row] +=
                            input.bound_small_coefficients[(bound_row, orbital)] * scale;
                    }
                }

                let product_start = 2 * input.kappa[orbital].unsigned_abs() as usize + 1;
                if product_start <= input.coefficient_count {
                    let scale = angular_coefficient * input.normalization[orbital].powi(2);
                    for coefficient in product_start..=input.coefficient_count {
                        let product_count = coefficient + 1 - product_start;
                        large_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_large_coefficients.column(orbital),
                                product_count,
                            );
                        small_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_small_coefficients.column(orbital),
                                product_count,
                            );
                    }
                }
            }
            multipole += 2;
        }
    }

    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] /= input.speed_of_light;
        small_coefficients[coefficient] /= input.speed_of_light;
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for row in 0..input.active_len {
        if row < input.radial_output_count {
            large_potential[row] /= input.speed_of_light;
            small_potential[row] /= input.speed_of_light;
        } else {
            large_potential[row] = Complex::new(0.0, 0.0);
            small_potential[row] = Complex::new(0.0, 0.0);
        }
        validate_complex_result("large_potential", row, large_potential[row])?;
        validate_complex_result("small_potential", row, small_potential[row])?;
    }

    Ok(FovrgExchangePotential {
        large_potential,
        small_potential,
        large_coefficients,
        small_coefficients,
    })
}

/// Port of `FOVRG/nucdec.f90`: point-nucleus radial grid and potential.
///
/// FEFF10 currently resets the nuclear mass to zero inside `nucdec`, so the
/// active branch is the point-nucleus Coulomb potential:
/// `dr(i) = dr1 / dz * exp(hx * (i - 1))`, `dv(i) = -dz / dr(i)`, and
/// `av(1) = -dz` with all remaining development coefficients zero.
pub fn fovrg_nuclear_potential(
    input: FovrgNuclearPotentialInput,
) -> Result<FovrgNuclearPotential, FovrgError> {
    validate_positive_finite("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite("step", input.step)?;
    validate_positive_finite("first_radius_times_charge", input.first_radius_times_charge)?;
    validate_count_at_least("radial_count", input.radial_count, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 5)?;

    let first_radius = input.first_radius_times_charge / input.nuclear_charge;
    let mut radii = Array1::<Real>::zeros(input.radial_count);
    let mut potential = Array1::<Real>::zeros(input.radial_count);
    for row in 0..input.radial_count {
        radii[row] = first_radius * (input.step * row as Real).exp();
        validate_radius(row, radii[row])?;

        potential[row] = -input.nuclear_charge / radii[row];
        validate_real_result("nuclear_potential", row, potential[row])?;
    }

    let mut development_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    development_coefficients[0] = -input.nuclear_charge;
    validate_real_result("development_coefficients", 0, development_coefficients[0])?;

    Ok(FovrgNuclearPotential {
        development_coefficients,
        radii,
        potential,
        nucleus_index: 1,
        first_radius_times_charge: input.first_radius_times_charge,
    })
}

/// Port of `FOVRG/potdvp.f90`: potential development coefficients.
///
/// FEFF accumulates bound-orbital density development coefficients from
/// occupied large/small radial polynomials, integrates those coefficients into
/// a local potential expansion, adds the nuclear development, and divides the
/// resulting `av` coefficients by `cl`.
pub fn fovrg_potential_development(
    input: FovrgPotentialDevelopmentInput<'_>,
) -> Result<FovrgPotentialDevelopment, FovrgError> {
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_count_at_least("orbital_count", input.orbital_count, 1)?;
    validate_count_at_least("nuclear_coefficients", input.nuclear_coefficients.len(), 2)?;
    validate_count_at_least("radii", input.radii.len(), 1)?;
    validate_active_len(
        "nuclear_coefficients",
        input.coefficient_count,
        input.nuclear_coefficients.len(),
    )?;
    validate_matrix_rows(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.shape()[0],
    )?;
    let bound_orbitals = input.orbital_count - 1;
    validate_matrix_cols(
        "large_coefficients",
        bound_orbitals,
        input.large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "small_coefficients",
        bound_orbitals,
        input.small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        bound_orbitals,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", bound_orbitals, input.kappa.len())?;
    validate_active_len("normalization", bound_orbitals, input.normalization.len())?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_radius(0, input.radii[0])?;
    if input.coefficient_count > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "coefficient_count",
            actual: input.coefficient_count,
            maximum: i32::MAX as usize - 1,
        });
    }

    for coefficient in 0..input.nuclear_coefficients.len() {
        validate_real_input(
            "nuclear_coefficients",
            coefficient,
            input.nuclear_coefficients[coefficient],
        )?;
    }
    for orbital in 0..bound_orbitals {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        for coefficient in 0..input.coefficient_count {
            validate_real_input(
                "large_coefficients",
                coefficient,
                input.large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "small_coefficients",
                coefficient,
                input.small_coefficients[(coefficient, orbital)],
            )?;
        }
    }

    let mut density_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    for orbital in 0..bound_orbitals {
        let kappa_abs = input.kappa[orbital].unsigned_abs() as usize;
        let leading_power = kappa_abs.saturating_mul(2);
        let product_count = input.coefficient_count + 2;
        if leading_power >= product_count {
            continue;
        }
        let max_product_order = product_count - leading_power;
        for product_order in 1..=max_product_order {
            let density_row = leading_power - 2 + product_order;
            density_coefficients[density_row - 1] += input.electron_counts[orbital]
                * (real_product_coefficient(
                    input.large_coefficients.column(orbital),
                    input.large_coefficients.column(orbital),
                    product_order,
                ) + real_product_coefficient(
                    input.small_coefficients.column(orbital),
                    input.small_coefficients.column(orbital),
                    product_order,
                ))
                * input.normalization[orbital].powi(2);
        }
    }

    let mut origin_correction = 0.0;
    for coefficient in 1..=input.coefficient_count {
        let row = coefficient - 1;
        density_coefficients[row] /= (coefficient + 2) as Real * (coefficient + 1) as Real;
        origin_correction +=
            density_coefficients[row] * input.radii[0].powi(coefficient as i32 + 1);
    }

    let mut potential_coefficients = Array1::from_iter(
        input
            .nuclear_coefficients
            .iter()
            .copied()
            .map(|value| Complex::new(value, 0.0)),
    );
    for coefficient in 1..=input.coefficient_count {
        let potential_row = coefficient + 3;
        if potential_row <= input.coefficient_count {
            potential_coefficients[potential_row - 1] -= density_coefficients[coefficient - 1];
        }
    }
    potential_coefficients[1] += origin_correction;
    for row in 0..potential_coefficients.len() {
        potential_coefficients[row] /= input.speed_of_light;
        validate_complex_result("potential_coefficients", row, potential_coefficients[row])?;
    }
    for row in 0..density_coefficients.len() {
        validate_real_result("density_coefficients", row, density_coefficients[row])?;
    }
    validate_real_result("origin_correction", 0, origin_correction)?;

    Ok(FovrgPotentialDevelopment {
        potential_coefficients,
        density_coefficients,
        origin_correction,
    })
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), FovrgError> {
    if actual < minimum {
        Err(FovrgError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn target_j_value(kappa: i32) -> usize {
    2 * kappa.unsigned_abs() as usize - 1
}

fn exchange_coefficient_start(
    multipole: usize,
    bound_kappa: i32,
    target_kappa: i32,
    target_power: Real,
) -> Option<usize> {
    let bound_abs = i64::from(bound_kappa.unsigned_abs());
    let target_abs = i64::from(target_kappa.unsigned_abs());
    let multipole = multipole as i64;
    let start = if target_power < 0.0 {
        multipole + 1 + bound_abs + target_abs
    } else {
        multipole + 1 + bound_abs - target_abs
    };
    (start >= 1).then_some(start as usize)
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), FovrgError> {
    if active_len > len {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_matrix_rows(
    field: &'static str,
    required: usize,
    available: usize,
) -> Result<(), FovrgError> {
    if required > available {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len: required,
            len: available,
        })
    } else {
        Ok(())
    }
}

fn validate_matrix_cols(
    field: &'static str,
    required: usize,
    available: usize,
) -> Result<(), FovrgError> {
    if required > available {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len: required,
            len: available,
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteInput { name, value })
    }
}

fn validate_nonzero_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value == 0.0 {
        Err(FovrgError::ZeroInput { name })
    } else {
        Ok(())
    }
}

fn validate_positive_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

fn validate_nonzero_denominator(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if value == 0.0 {
        Err(FovrgError::ZeroDenominator { name })
    } else {
        Ok(())
    }
}

fn validate_radius(row: usize, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput {
            name: "radius",
            value,
        })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveRadius { row, value })
    } else {
        Ok(())
    }
}

fn validate_nonzero_kappa(name: &'static str, row: usize, value: i32) -> Result<(), FovrgError> {
    if value == 0 {
        Err(FovrgError::InvalidQuantumNumber { name, row, value })
    } else {
        Ok(())
    }
}

fn validate_real_input(name: &'static str, row: usize, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteRealInput { name, row, value })
    }
}

fn validate_real_result(name: &'static str, row: usize, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteRealInput { name, row, value })
    }
}

fn validate_complex_input(
    name: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteComplexInput { name, row, value })
    }
}

fn validate_potential(row: usize, value: Complex) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFinitePotential { row, value })
    }
}

fn complex_real_product_coefficient(
    complex_coefficients: ArrayView1<'_, Complex>,
    real_coefficients: ArrayView1<'_, Real>,
    count: usize,
) -> Complex {
    (0..count).fold(Complex::new(0.0, 0.0), |sum, index| {
        sum + complex_coefficients[index] * real_coefficients[count - 1 - index]
    })
}

fn real_product_coefficient(
    left_coefficients: ArrayView1<'_, Real>,
    right_coefficients: ArrayView1<'_, Real>,
    count: usize,
) -> Real {
    (0..count).fold(0.0, |sum, index| {
        sum + left_coefficients[index] * right_coefficients[count - 1 - index]
    })
}

fn validate_complex_result(
    name: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteComplexInput { name, row, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::{Complex, Real};

    use super::{
        FovrgC3DerivativeInput, FovrgError, FovrgExchangePotentialInput,
        FovrgNuclearPotentialInput, FovrgOrthogonalizationInput, FovrgOverlapIntegralInput,
        FovrgPotentialDevelopmentInput, FovrgYkZkExchangeInput, FovrgYkZkTransformInput,
        fovrg_c3_derivative, fovrg_exchange_potential, fovrg_nuclear_potential,
        fovrg_overlap_integral, fovrg_potential_development, fovrg_schmidt_orthogonalize,
        fovrg_yk_zk_exchange, fovrg_yk_zk_transform,
    };

    #[test]
    fn c3_derivative_matches_feff_diff_reference() -> Result<(), FovrgError> {
        let (potential, radii) = diff_reference_inputs(10);

        let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 10,
        })?;

        let expected = [
            (-0.011_975_827_006_405_27, -0.011_279_195_671_167_455),
            (-0.016_505_394_195_758_99, -0.008_884_114_730_822_418),
            (-0.020_242_542_448_345_43, -0.005_647_908_958_998_54),
            (-0.022_839_291_155_546_27, -0.001_659_964_058_354_706_8),
            (-0.024_047_315_082_090_202, 0.002_950_607_669_371_263_3),
            (-0.023_683_648_659_231_31, 0.008_014_885_042_325_136),
            (-0.021_663_526_338_827_583, 0.013_330_188_602_550_464),
            (-0.018_012_853_921_219_218, 0.018_667_556_473_840_063),
            (-0.012_457_714_462_626_513, 0.023_984_332_127_499_31),
            (-0.007_300_598_102_380_937, 0.028_056_048_903_698_883),
        ];
        for (actual, (expected_re, expected_im)) in derivative.iter().zip(expected) {
            assert_complex_close(*actual, expected_re, expected_im, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn c3_derivative_rejects_invalid_inputs() {
        let (potential, radii) = diff_reference_inputs(8);

        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 7,
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 9,
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "potential",
                ..
            })
        ));
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0,
                active_len: 8,
            }),
            Err(FovrgError::ZeroInput { name: "delta" })
        ));

        let mut bad_radii = radii.clone();
        bad_radii[3] = 0.0;
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: bad_radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 8,
            }),
            Err(FovrgError::NonPositiveRadius { row: 3, .. })
        ));

        let mut bad_potential = potential.clone();
        bad_potential[2] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: bad_potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 8,
            }),
            Err(FovrgError::NonFinitePotential { row: 2, .. })
        ));
    }

    #[test]
    fn nuclear_potential_matches_feff_nucdec_point_reference() -> Result<(), FovrgError> {
        let potential = fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 29.0,
            step: 0.0725,
            first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
            radial_count: 8,
            coefficient_count: 6,
        })?;

        assert_eq!(potential.nucleus_index, 1);
        assert_close(
            potential.first_radius_times_charge,
            0.004_371_259_177_768_818_5,
            1.0e-15,
        );
        let expected_coefficients = [-29.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (actual, expected) in potential
            .development_coefficients
            .iter()
            .zip(expected_coefficients)
        {
            assert_close(*actual, expected, 1.0e-13);
        }

        let expected_rows = [
            (0.000_150_733_075_095_476_5, -192_393.076_182_058_78),
            (0.000_162_067_117_982_503_44, -178_938.210_051_534_35),
            (0.000_174_253_399_358_552_06, -166_424.299_937_634_06),
            (0.000_187_356_001_427_070_04, -154_785.540_783_909_73),
            (0.000_201_443_824_912_202_5, -143_960.729_561_402),
            (0.000_216_590_951_376_884_9, -133_892.943_429_283_77),
            (0.000_232_877_032_784_649_17, -124_529.240_403_099_25),
            (0.000_250_387_710_353_676, -115_820.380_956_545_8),
        ];
        for (row, (expected_radius, expected_potential)) in expected_rows.into_iter().enumerate() {
            assert_close(potential.radii[row], expected_radius, 1.0e-13);
            assert_close(potential.potential[row], expected_potential, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn nuclear_potential_rejects_invalid_inputs() {
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0725,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 0,
                coefficient_count: 6,
            }),
            Err(FovrgError::CountTooSmall {
                name: "radial_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0725,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 8,
                coefficient_count: 4,
            }),
            Err(FovrgError::CountTooSmall {
                name: "coefficient_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 0.0,
                step: 0.0725,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 8,
                coefficient_count: 6,
            }),
            Err(FovrgError::NonPositiveInput {
                name: "nuclear_charge",
                ..
            })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 8,
                coefficient_count: 6,
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0725,
                first_radius_times_charge: Real::NAN,
                radial_count: 8,
                coefficient_count: 6,
            }),
            Err(FovrgError::NonFiniteInput {
                name: "first_radius_times_charge",
                ..
            })
        ));
    }

    #[test]
    fn yk_zk_transform_matches_feff_yzktec_reference() -> Result<(), FovrgError> {
        let (source, coefficients, radii) = yzktec_reference_inputs(12);

        let transform = fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, -0.25),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.011, -0.006),
        })?;

        assert_eq!(transform.computed_len, 10);
        assert_complex_close(
            transform.origin_constant,
            1_069.293_326_934_643,
            639.337_203_837_502_8,
            1.0e-12,
        );

        let expected_rows = [
            (
                0.006_376_970_423_953_328,
                0.003_747_109_936_537_645_4,
                0.000_019_115_876_398_023_115,
                0.000_002_615_603_860_575_636,
            ),
            (
                0.007_841_326_927_116_237,
                0.004_425_503_213_295_339,
                0.000_415_175_421_810_819_7,
                0.001_186_221_311_123_577_3,
            ),
            (
                0.009_454_062_278_996_728,
                0.004_817_609_696_528_203,
                0.001_052_233_690_642_138_8,
                0.002_225_420_005_754_274_7,
            ),
            (
                0.011_156_498_748_891_856,
                0.004_912_703_002_968_925,
                0.001_915_624_422_266_479,
                0.003_118_393_016_633_964_7,
            ),
            (
                0.012_883_154_525_001_68,
                0.004_698_896_965_378_377,
                0.002_982_924_829_137_327,
                0.003_859_683_726_441_837_7,
            ),
            (
                0.014_563_357_943_144_598,
                0.004_164_285_902_400_606,
                0.004_223_307_649_668_978,
                0.004_440_459_445_284_031,
            ),
            (
                0.016_123_447_951_845_19,
                0.003_298_256_791_962_156,
                0.005_597_243_449_987_172_5,
                0.004_848_768_666_445_236,
            ),
            (
                0.017_489_549_856_229_16,
                0.002_093_015_402_338_084,
                0.007_056_612_425_756_153,
                0.005_069_801_813_782_371,
            ),
            (
                0.018_590_890_204_374_55,
                0.000_545_375_511_912_808,
                0.008_545_277_387_035_53,
                0.005_086_162_856_970_115_5,
            ),
            (
                0.019_630_800_153_888_66,
                -0.001_305_325_902_639_564_2,
                0.008_630_800_153_888_66,
                0.004_694_674_097_360_436,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
            assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
        }

        let expected_coefficients = [
            (
                -1.824_158_963_244_542,
                -0.246_122_633_186_553_6,
                0.237_140_665_221_790_4,
                0.031_995_942_314_251_964,
            ),
            (
                2.794_098_740_012_050_3,
                0.730_272_609_187_755_4,
                0.195_586_911_800_843_6,
                0.051_119_082_643_142_88,
            ),
            (
                0.609_454_103_153_871_8,
                0.232_241_030_552_876_95,
                0.164_552_607_851_545_35,
                0.062_705_078_249_276_76,
            ),
            (
                0.297_530_519_518_787_1,
                0.147_284_129_112_308_2,
                0.139_839_344_173_829_93,
                0.069_223_540_682_784_84,
            ),
            (
                0.178_046_974_447_949_95,
                0.107_477_058_743_461_04,
                0.119_291_472_880_126_45,
                0.072_009_629_358_118_89,
            ),
            (
                0.116_898_830_661_045_85,
                0.082_624_380_349_051_94,
                0.101_701_982_675_109_88,
                0.071_883_210_903_675_18,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_coefficients.into_iter().enumerate() {
            assert_complex_close(transform.yk_coefficients[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk_coefficients[row], zk_re, zk_im, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn yk_zk_transform_rejects_invalid_inputs() {
        let (source, coefficients, radii) = yzktec_reference_inputs(12);

        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 1,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 11,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "source_coefficients",
                ..
            })
        ));

        let mut bad_source = source.clone();
        bad_source[3] = Complex::new(0.0, Real::NAN);
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: bad_source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonFiniteComplexInput {
                name: "source",
                row: 3,
                ..
            })
        ));

        let mut bad_radii = radii.clone();
        bad_radii[0] = -1.0;
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: bad_radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonPositiveRadius { row: 0, .. })
        ));
    }

    #[test]
    fn yk_zk_exchange_matches_feff_yzkrdc_reference() -> Result<(), FovrgError> {
        let input = yzkrdc_reference_inputs(12);

        let transform = fovrg_yk_zk_exchange(input.as_exchange_input())?;

        assert_eq!(transform.computed_len, 10);
        assert_complex_close(
            transform.origin_constant,
            1_321.269_761_542_853_5,
            1_058.551_269_340_285_2,
            1.0e-12,
        );

        let expected_rows = [
            (
                0.007_686_009_135_817_749,
                0.006_170_157_063_400_744,
                0.000_000_645_317_783_462_879_7,
                0.000_000_110_270_749_084_274_43,
            ),
            (
                0.009_300_746_624_727_518,
                0.007_544_419_441_270_886,
                0.001_294_275_945_600_778,
                0.000_639_802_281_166_626_1,
            ),
            (
                0.010_786_770_527_864_456,
                0.008_925_139_869_295_514,
                0.002_573_522_373_652_341,
                0.001_630_025_738_506_313,
            ),
            (
                0.012_109_032_230_448_815,
                0.010_184_928_348_947_297,
                0.003_887_582_939_633_221_6,
                0.002_904_232_874_818_797,
            ),
            (
                0.013_206_275_901_284_993,
                0.011_197_011_268_772_228,
                0.005_274_223_639_134_339,
                0.004_372_201_622_443_519_5,
            ),
            (
                0.013_990_089_034_721_83,
                0.011_844_365_308_609_77,
                0.006_755_737_128_168_105,
                0.005_923_123_611_939_633,
            ),
            (
                0.014_345_254_715_897_94,
                0.012_029_374_779_196_415,
                0.008_335_434_490_732_629,
                0.007_430_974_286_925_581,
            ),
            (
                0.014_131_414_050_294_111,
                0.011_683_264_170_573_946,
                0.009_995_141_713_724_128,
                0.008_761_915_452_378_507,
            ),
            (
                0.013_185_862_903_069_551,
                0.010_774_522_262_808_485,
                0.011_694_148_402_953_802,
                0.009_783_248_603_735_934,
            ),
            (
                0.011_660_651_859_152_821,
                0.009_488_268_367_479_21,
                0.011_660_651_859_152_821,
                0.009_488_268_367_479_21,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
            assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
        }

        let expected_coefficients = [
            (6.375_854_958_562_043, 1.073_871_387_646_292_4),
            (1.497_833_540_772_686, 0.370_169_086_848_655_57),
            (1.049_320_795_997_538_8, 0.338_218_568_996_506_2),
            (0.843_625_047_557_286_8, 0.332_360_660_760_794_2),
            (0.713_658_559_831_859_2, 0.329_689_349_293_459_3),
            (0.619_406_204_717_043, 0.325_898_372_715_123_5),
        ];
        for (row, (expected_re, expected_im)) in expected_coefficients.into_iter().enumerate() {
            assert_complex_close(
                transform.yk_coefficients[row],
                expected_re,
                expected_im,
                1.0e-13,
            );
        }
        Ok(())
    }

    #[test]
    fn yk_zk_exchange_rejects_invalid_inputs() {
        let mut input = yzkrdc_reference_inputs(12);
        input.large_component[2] = Real::NAN;

        assert!(matches!(
            fovrg_yk_zk_exchange(input.as_exchange_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "large_component",
                row: 2,
                ..
            })
        ));

        let mut input = yzkrdc_reference_inputs(12);
        input.partner_small_coefficients[1] = Complex::new(0.0, Real::INFINITY);
        assert!(matches!(
            fovrg_yk_zk_exchange(input.as_exchange_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "partner_small_coefficients",
                row: 1,
                ..
            })
        ));

        let input = yzkrdc_reference_inputs(4);
        assert!(matches!(
            fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                active_len: 5,
                ..input.as_exchange_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "large_component",
                ..
            })
        ));
    }

    #[test]
    fn overlap_integral_matches_feff_dsordc_reference() -> Result<(), FovrgError> {
        let input = dsordc_reference_inputs(9);

        let integral = fovrg_overlap_integral(input.as_overlap_input())?;

        assert_complex_close(
            integral,
            0.018_257_373_605_649_284,
            0.014_647_428_406_545_006,
            1.0e-13,
        );
        Ok(())
    }

    #[test]
    fn overlap_integral_rejects_invalid_inputs() {
        let input = dsordc_reference_inputs(9);

        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                active_len: 8,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::CountMustBeOdd {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                active_len: 2,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                active_len: 11,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "large_integrand",
                ..
            })
        ));
        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                step: 0.0,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));

        let mut input = dsordc_reference_inputs(9);
        input.radii[2] = 0.0;
        assert!(matches!(
            fovrg_overlap_integral(input.as_overlap_input()),
            Err(FovrgError::NonPositiveRadius { row: 2, .. })
        ));

        let mut input = dsordc_reference_inputs(9);
        input.large_integrand_coefficients[3] = Complex::new(Real::NAN, 0.0);
        assert!(matches!(
            fovrg_overlap_integral(input.as_overlap_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "large_integrand_coefficients",
                row: 3,
                ..
            })
        ));
    }

    #[test]
    fn schmidt_orthogonalization_matches_feff_ortdac_reference() -> Result<(), FovrgError> {
        let input = ortdac_reference_inputs(9);

        let orthogonalized = fovrg_schmidt_orthogonalize(input.as_orthogonalization_input())?;

        assert_ne!(orthogonalized.overlaps[0], Complex::new(0.0, 0.0));
        assert_eq!(orthogonalized.overlaps[1], Complex::new(0.0, 0.0));
        assert_eq!(orthogonalized.overlaps[2], Complex::new(0.0, 0.0));
        assert_ne!(orthogonalized.overlaps[3], Complex::new(0.0, 0.0));

        let expected_rows = [
            (
                0.184_796_621_476_688_8,
                0.960_525_659_674_847_8,
                0.953_489_591_844_743_2,
                0.196_175_227_984_495_05,
            ),
            (
                0.364_943_848_030_108_26,
                0.909_210_209_413_431_4,
                0.932_155_457_421_994,
                0.411_067_158_250_690_3,
            ),
            (
                0.535_755_652_121_730_3,
                0.846_307_238_142_853_7,
                0.903_311_497_295_576_5,
                0.608_386_237_505_285_2,
            ),
            (
                0.692_898_032_849_261_3,
                0.772_271_807_926_033_1,
                0.867_091_885_664_384_1,
                0.780_141_644_613_100_9,
            ),
            (
                0.832_426_823_226_043_9,
                0.687_685_325_115_306_8,
                0.823_683_636_514_673_7,
                0.919_472_245_213_980_9,
            ),
            (
                0.950_900_258_284_096_4,
                0.593_246_291_500_718_2,
                0.773_325_760_496_256_5,
                1.020_947_524_294_445_2,
            ),
            (
                1.045_477_281_469_491_5,
                0.489_760_058_904_124,
                0.716_308_162_735_437_1,
                1.080_805_531_074_559_2,
            ),
            (
                1.113_998_793_619_971_6,
                0.378_127_783_523_524,
                0.652_970_269_213_538_8,
                1.097_117_398_093_875_3,
            ),
            (
                1.155_049_530_148_776_2,
                0.259_334_766_409_572_26,
                0.583_699_367_941_233_6,
                1.069_871_225_404_380_5,
            ),
        ];
        for (row, (large_re, large_im, small_re, small_im)) in expected_rows.into_iter().enumerate()
        {
            assert_complex_close(
                orthogonalized.large_component[row],
                large_re,
                large_im,
                1.0e-13,
            );
            assert_complex_close(
                orthogonalized.small_component[row],
                small_re,
                small_im,
                1.0e-13,
            );
        }

        let expected_coefficients = [
            (
                0.998_449_079_711_476_6,
                0.111_350_550_939_607_72,
                0.068_224_857_711_253_53,
                1.016_544_722_930_134,
            ),
            (
                1.013_026_259_606_995_7,
                0.245_410_754_538_065_13,
                0.135_740_121_285_759_1,
                1.018_823_567_734_752_8,
            ),
            (
                1.011_555_658_693_028_5,
                0.370_053_923_878_711_44,
                0.201_841_908_576_728_68,
                1.007_158_627_823_294_7,
            ),
            (
                0.994_728_243_334_449_6,
                0.480_813_496_862_218_4,
                0.265_837_715_515_805,
                0.982_072_667_678_714_5,
            ),
            (
                0.963_494_952_966_650_6,
                0.573_626_109_357_632,
                0.327_051_990_766_753_06,
                0.944_281_663_334_709_4,
            ),
            (
                0.919_050_620_050_682_9,
                0.644_948_608_319_962_8,
                0.384_831_574_144_536_1,
                0.894_684_562_223_998_2,
            ),
        ];
        for (coefficient, (large_re, large_im, small_re, small_im)) in
            expected_coefficients.into_iter().enumerate()
        {
            assert_complex_close(
                orthogonalized.large_coefficients[coefficient],
                large_re,
                large_im,
                1.0e-13,
            );
            assert_complex_close(
                orthogonalized.small_coefficients[coefficient],
                small_re,
                small_im,
                1.0e-13,
            );
        }
        Ok(())
    }

    #[test]
    fn schmidt_orthogonalization_rejects_invalid_inputs() {
        let input = ortdac_reference_inputs(9);

        assert!(matches!(
            fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
                target_kappa: 0,
                ..input.as_orthogonalization_input()
            }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "target_kappa",
                value: 0,
                ..
            })
        ));
        assert!(matches!(
            fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
                active_len: 8,
                ..input.as_orthogonalization_input()
            }),
            Err(FovrgError::CountMustBeOdd {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
                bound_orbital_count: 5,
                ..input.as_orthogonalization_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "bound_large_components",
                ..
            })
        ));

        let mut input = ortdac_reference_inputs(9);
        input.electron_counts[0] = Real::NAN;
        assert!(matches!(
            fovrg_schmidt_orthogonalize(input.as_orthogonalization_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "electron_counts",
                row: 0,
                ..
            })
        ));

        let mut input = ortdac_reference_inputs(9);
        input.target_large_component[1] = Complex::new(0.0, Real::INFINITY);
        assert!(matches!(
            fovrg_schmidt_orthogonalize(input.as_orthogonalization_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "target_large_component",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn exchange_potential_matches_feff_potex_reference() -> Result<(), FovrgError> {
        let input = potex_reference_inputs(9);

        let potential = fovrg_exchange_potential(input.as_exchange_potential_input())?;

        let expected_rows = [
            (
                0.000_005_554_864_571_582_592,
                0.000_004_589_245_040_261_105,
                0.000_039_609_278_623_293_83,
                0.000_033_434_207_770_074_9,
            ),
            (
                0.000_011_841_826_673_104_183,
                0.000_009_794_866_583_804_325,
                0.000_042_053_026_297_685_21,
                0.000_035_634_329_767_400_91,
            ),
            (
                0.000_018_578_019_491_824_635,
                0.000_015_404_560_990_245_302,
                0.000_043_309_970_634_588_22,
                0.000_036_974_047_816_590_13,
            ),
            (
                0.000_025_293_374_225_649_448,
                0.000_020_974_401_383_246_206,
                0.000_043_220_277_722_069_02,
                0.000_037_209_351_789_692_02,
            ),
            (
                0.000_031_463_027_867_695_25,
                0.000_026_005_262_272_416_62,
                0.000_041_711_066_672_227_27,
                0.000_036_233_682_295_299_64,
            ),
            (
                0.000_036_572_642_292_251_735,
                0.000_030_066_055_448_048_974,
                0.000_038_831_760_336_099_916,
                0.000_034_098_776_520_822_48,
            ),
            (
                0.000_040_212_198_293_964_37,
                0.000_032_909_100_990_340_72,
                0.000_034_779_450_472_774_91,
                0.000_030_991_961_623_472_31,
            ),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ];
        for (row, (large_re, large_im, small_re, small_im)) in expected_rows.into_iter().enumerate()
        {
            assert_complex_close(potential.large_potential[row], large_re, large_im, 1.0e-13);
            assert_complex_close(potential.small_potential[row], small_re, small_im, 1.0e-13);
        }

        let expected_coefficients = [
            (
                0.056_004_531_605_744_41,
                0.046_663_043_007_772_96,
                0.000_603_453_997_835_984_7,
                0.000_503_278_831_814_845_3,
            ),
            (
                -1.349_603_830_126_038,
                -1.128_877_944_124_393_2,
                -0.045_179_344_996_730_146,
                -0.037_885_484_599_471_386,
            ),
            (
                -2.231_032_417_788_144_4,
                -1.854_555_246_757_578,
                -0.141_157_217_260_626_58,
                -0.117_915_434_665_414_93,
            ),
            (
                24.781_460_626_354_06,
                19.753_480_329_995_963,
                2.027_705_895_254_902,
                1.613_953_852_227_001_6,
            ),
            (
                24.726_993_882_200_276,
                19.712_367_835_956_03,
                4.170_773_455_085_641,
                3.325_408_244_119_067_5,
            ),
            (
                24.319_899_966_071_53,
                19.384_360_683_910_17,
                6.262_813_264_194_341,
                4.995_924_412_893_355,
            ),
        ];
        for (coefficient, (large_re, large_im, small_re, small_im)) in
            expected_coefficients.into_iter().enumerate()
        {
            assert_complex_close(
                potential.large_coefficients[coefficient],
                large_re,
                large_im,
                1.0e-13,
            );
            assert_complex_close(
                potential.small_coefficients[coefficient],
                small_re,
                small_im,
                1.0e-13,
            );
        }
        Ok(())
    }

    #[test]
    fn exchange_potential_rejects_invalid_inputs() {
        let input = potex_reference_inputs(9);

        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                target_kappa: 0,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "target_kappa",
                value: 0,
                ..
            })
        ));
        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                radial_output_count: 10,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "radial_output_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                speed_of_light: 0.0,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::ZeroInput {
                name: "speed_of_light"
            })
        ));
        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                bound_orbital_count: 5,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "bound_large_components",
                ..
            })
        ));

        let mut input = potex_reference_inputs(9);
        input.orbital_lengths[2] = 0;
        assert!(matches!(
            fovrg_exchange_potential(input.as_exchange_potential_input()),
            Err(FovrgError::CountTooSmall {
                name: "orbital_length",
                ..
            })
        ));

        let mut input = potex_reference_inputs(9);
        input.angular_coefficients[(1, 0)] = Real::NAN;
        assert!(matches!(
            fovrg_exchange_potential(input.as_exchange_potential_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "angular_coefficients",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn potential_development_matches_feff_potdvp_reference() -> Result<(), FovrgError> {
        let input = potdvp_reference_inputs(12);

        let development = fovrg_potential_development(input.as_potential_input())?;

        assert_close(
            development.origin_correction,
            0.000_092_381_409_682_418_76,
            1.0e-13,
        );
        let expected_potential = [
            -0.002_211_097_828_492_991_6,
            -0.001_838_258_707_742_217_9,
            -0.001_437_578_456_148_908_5,
            -0.003_049_520_002_144_625,
            -0.002_623_511_736_279_590_5,
            -0.002_546_330_557_249_715,
            -0.002_045_957_521_005_020_5,
            -0.001_773_999_888_200_908_3,
            0.001_583_525_507_534_584_8,
            0.002_189_205_770_785_14,
        ];
        for (actual, expected) in development
            .potential_coefficients
            .iter()
            .zip(expected_potential)
        {
            assert_complex_close(*actual, expected, 0.0, 1.0e-13);
        }

        let expected_density = [
            0.279_894_020_220_530_5,
            0.284_515_551_889_673_2,
            0.340_938_951_910_833_2,
            0.343_369_832_974_347,
            0.381_101_847_054_515_8,
            0.388_553_939_183_866_9,
            0.381_768_833_467_862,
            0.368_012_415_945_436_16,
        ];
        for (actual, expected) in development
            .density_coefficients
            .iter()
            .zip(expected_density)
        {
            assert_close(*actual, expected, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn potential_development_rejects_invalid_inputs() {
        let mut input = potdvp_reference_inputs(12);
        input.nuclear_coefficients[0] = Real::NAN;
        assert!(matches!(
            fovrg_potential_development(input.as_potential_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "nuclear_coefficients",
                row: 0,
                ..
            })
        ));

        let mut input = potdvp_reference_inputs(12);
        input.kappa[1] = 0;
        assert!(matches!(
            fovrg_potential_development(input.as_potential_input()),
            Err(FovrgError::InvalidQuantumNumber {
                name: "kappa",
                row: 1,
                value: 0,
            })
        ));

        let mut input = potdvp_reference_inputs(12);
        input.large_coefficients = Array2::zeros((7, 4));
        assert!(matches!(
            fovrg_potential_development(input.as_potential_input()),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "large_coefficients",
                ..
            })
        ));

        let input = potdvp_reference_inputs(12);
        assert!(matches!(
            fovrg_potential_development(FovrgPotentialDevelopmentInput {
                speed_of_light: 0.0,
                ..input.as_potential_input()
            }),
            Err(FovrgError::ZeroInput {
                name: "speed_of_light"
            })
        ));
    }

    fn diff_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Real>) {
        let potential = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            Complex::new(
                (0.21 * index).sin() + 0.03 * index,
                (0.17 * index).cos() - 0.02 * index,
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            0.15 + 0.04 * index + 0.001 * index * index
        }));
        (potential, radii)
    }

    fn yzktec_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Complex>, Array1<Real>) {
        let step = 0.0725;
        let source = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            Complex::new(
                (0.19 * index).sin() + 0.02 * index,
                (0.11 * index).cos() - 0.03 * index,
            )
        }));
        let coefficients = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(
                0.04 * index + (0.13 * index).cos(),
                -0.03 * index + (0.17 * index).sin(),
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            0.018 * (step * (index - 1.0)).exp()
        }));
        (source, coefficients, radii)
    }

    struct YzkrdcReferenceInputs {
        large_component: Array1<Real>,
        small_component: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
        partner_large_component: Array1<Complex>,
        partner_small_component: Array1<Complex>,
        partner_large_coefficients: Array1<Complex>,
        partner_small_coefficients: Array1<Complex>,
        radii: Array1<Real>,
        orbital_power: Real,
        partner_power: Real,
        step: Real,
        angular_momentum: usize,
        coefficient_count: usize,
        orbital_len: usize,
        source_len: usize,
        active_len: usize,
    }

    impl YzkrdcReferenceInputs {
        fn as_exchange_input(&self) -> FovrgYkZkExchangeInput<'_> {
            FovrgYkZkExchangeInput {
                large_component: self.large_component.view(),
                small_component: self.small_component.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                partner_large_component: self.partner_large_component.view(),
                partner_small_component: self.partner_small_component.view(),
                partner_large_coefficients: self.partner_large_coefficients.view(),
                partner_small_coefficients: self.partner_small_coefficients.view(),
                radii: self.radii.view(),
                orbital_power: self.orbital_power,
                partner_power: self.partner_power,
                step: self.step,
                angular_momentum: self.angular_momentum,
                coefficient_count: self.coefficient_count,
                orbital_len: self.orbital_len,
                source_len: self.source_len,
                active_len: self.active_len,
            }
        }
    }

    fn yzkrdc_reference_inputs(count: usize) -> YzkrdcReferenceInputs {
        let step = 0.0725;
        let orbital_column = 2.0;
        let large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.05 * row * orbital_column).sin() + 0.001 * (row + orbital_column)
        }));
        let small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.04 * row * orbital_column).cos() - 0.002 * (row - orbital_column)
        }));
        let large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            0.02 * row + (0.03 * row * orbital_column).cos()
        }));
        let small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            -0.015 * row + (0.025 * row * orbital_column).sin()
        }));
        let partner_large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.19 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let partner_small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.07 * row).cos() - 0.01 * row,
                (0.23 * row).sin() + 0.015 * row,
            )
        }));
        let partner_large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let partner_small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        YzkrdcReferenceInputs {
            large_component,
            small_component,
            large_coefficients,
            small_coefficients,
            partner_large_component,
            partner_small_component,
            partner_large_coefficients,
            partner_small_coefficients,
            radii,
            orbital_power: 0.65 + 0.08 * orbital_column,
            partner_power: 1.35,
            step,
            angular_momentum: 2,
            coefficient_count: 6,
            orbital_len: 9,
            source_len: 9,
            active_len: count,
        }
    }

    struct DsordcReferenceInputs {
        large_integrand: Array1<Complex>,
        small_integrand: Array1<Complex>,
        large_integrand_coefficients: Array1<Complex>,
        small_integrand_coefficients: Array1<Complex>,
        large_component: Array1<Real>,
        small_component: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
        radii: Array1<Real>,
        integrand_power: Real,
        orbital_power: Real,
        step: Real,
        coefficient_count: usize,
        active_len: usize,
    }

    impl DsordcReferenceInputs {
        fn as_overlap_input(&self) -> FovrgOverlapIntegralInput<'_> {
            FovrgOverlapIntegralInput {
                large_integrand: self.large_integrand.view(),
                small_integrand: self.small_integrand.view(),
                large_integrand_coefficients: self.large_integrand_coefficients.view(),
                small_integrand_coefficients: self.small_integrand_coefficients.view(),
                large_component: self.large_component.view(),
                small_component: self.small_component.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                radii: self.radii.view(),
                integrand_power: self.integrand_power,
                orbital_power: self.orbital_power,
                step: self.step,
                coefficient_count: self.coefficient_count,
                active_len: self.active_len,
            }
        }
    }

    fn dsordc_reference_inputs(count: usize) -> DsordcReferenceInputs {
        let step = 0.0725;
        let orbital = 3.0;
        let large_integrand = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.17 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let small_integrand = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.09 * row).cos() - 0.01 * row,
                (0.21 * row).sin() + 0.015 * row,
            )
        }));
        let large_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let small_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
        }));
        let small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
        }));
        let large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            0.02 * row + (0.03 * row * orbital).cos()
        }));
        let small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            -0.015 * row + (0.025 * row * orbital).sin()
        }));
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        DsordcReferenceInputs {
            large_integrand,
            small_integrand,
            large_integrand_coefficients,
            small_integrand_coefficients,
            large_component,
            small_component,
            large_coefficients,
            small_coefficients,
            radii,
            integrand_power: 1.35,
            orbital_power: 0.45 + 0.06 * orbital,
            step,
            coefficient_count: 6,
            active_len: count,
        }
    }

    struct OrtdacReferenceInputs {
        target_large_component: Array1<Complex>,
        target_small_component: Array1<Complex>,
        target_large_coefficients: Array1<Complex>,
        target_small_coefficients: Array1<Complex>,
        bound_large_components: Array2<Real>,
        bound_small_components: Array2<Real>,
        bound_large_coefficients: Array2<Real>,
        bound_small_coefficients: Array2<Real>,
        electron_counts: Array1<Real>,
        kappa: Array1<i32>,
        orbital_powers: Array1<Real>,
        radii: Array1<Real>,
        target_power: Real,
        target_kappa: i32,
        step: Real,
        coefficient_count: usize,
        active_len: usize,
        bound_orbital_count: usize,
    }

    impl OrtdacReferenceInputs {
        fn as_orthogonalization_input(&self) -> FovrgOrthogonalizationInput<'_> {
            FovrgOrthogonalizationInput {
                target_large_component: self.target_large_component.view(),
                target_small_component: self.target_small_component.view(),
                target_large_coefficients: self.target_large_coefficients.view(),
                target_small_coefficients: self.target_small_coefficients.view(),
                bound_large_components: self.bound_large_components.view(),
                bound_small_components: self.bound_small_components.view(),
                bound_large_coefficients: self.bound_large_coefficients.view(),
                bound_small_coefficients: self.bound_small_coefficients.view(),
                electron_counts: self.electron_counts.view(),
                kappa: self.kappa.view(),
                orbital_powers: self.orbital_powers.view(),
                radii: self.radii.view(),
                target_power: self.target_power,
                target_kappa: self.target_kappa,
                step: self.step,
                coefficient_count: self.coefficient_count,
                active_len: self.active_len,
                bound_orbital_count: self.bound_orbital_count,
            }
        }
    }

    fn ortdac_reference_inputs(count: usize) -> OrtdacReferenceInputs {
        let step = 0.0725;
        let bound_orbitals = 4;
        let target_large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.17 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let target_small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.09 * row).cos() - 0.01 * row,
                (0.21 * row).sin() + 0.015 * row,
            )
        }));
        let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let bound_large_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
            });
        let bound_small_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
            });
        let bound_large_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                0.02 * row + (0.03 * row * orbital).cos()
            });
        let bound_small_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                -0.015 * row + (0.025 * row * orbital).sin()
            });
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        OrtdacReferenceInputs {
            target_large_component,
            target_small_component,
            target_large_coefficients,
            target_small_coefficients,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.2, 1.4, 0.0, 2.0]),
            kappa: Array1::from_vec(vec![-2, 1, -2, -2]),
            orbital_powers: Array1::from_iter((1..=bound_orbitals).map(|orbital| {
                let orbital = orbital as Real;
                0.45 + 0.06 * orbital
            })),
            radii,
            target_power: 0.45 + 0.06 * 5.0,
            target_kappa: -2,
            step,
            coefficient_count: 6,
            active_len: count,
            bound_orbital_count: bound_orbitals,
        }
    }

    struct PotexReferenceInputs {
        target_large_component: Array1<Complex>,
        target_small_component: Array1<Complex>,
        target_large_coefficients: Array1<Complex>,
        target_small_coefficients: Array1<Complex>,
        bound_large_components: Array2<Real>,
        bound_small_components: Array2<Real>,
        bound_large_coefficients: Array2<Real>,
        bound_small_coefficients: Array2<Real>,
        angular_coefficients: Array2<Real>,
        orbital_powers: Array1<Real>,
        kappa: Array1<i32>,
        orbital_lengths: Array1<usize>,
        normalization: Array1<Real>,
        radii: Array1<Real>,
        target_power: Real,
        target_kappa: i32,
        target_normalization: Real,
        speed_of_light: Real,
        step: Real,
        coefficient_count: usize,
        source_len: usize,
        active_len: usize,
        radial_output_count: usize,
        bound_orbital_count: usize,
    }

    impl PotexReferenceInputs {
        fn as_exchange_potential_input(&self) -> FovrgExchangePotentialInput<'_> {
            FovrgExchangePotentialInput {
                target_large_component: self.target_large_component.view(),
                target_small_component: self.target_small_component.view(),
                target_large_coefficients: self.target_large_coefficients.view(),
                target_small_coefficients: self.target_small_coefficients.view(),
                bound_large_components: self.bound_large_components.view(),
                bound_small_components: self.bound_small_components.view(),
                bound_large_coefficients: self.bound_large_coefficients.view(),
                bound_small_coefficients: self.bound_small_coefficients.view(),
                angular_coefficients: self.angular_coefficients.view(),
                orbital_powers: self.orbital_powers.view(),
                kappa: self.kappa.view(),
                orbital_lengths: self.orbital_lengths.view(),
                normalization: self.normalization.view(),
                radii: self.radii.view(),
                target_power: self.target_power,
                target_kappa: self.target_kappa,
                target_normalization: self.target_normalization,
                speed_of_light: self.speed_of_light,
                step: self.step,
                coefficient_count: self.coefficient_count,
                source_len: self.source_len,
                active_len: self.active_len,
                radial_output_count: self.radial_output_count,
                bound_orbital_count: self.bound_orbital_count,
            }
        }
    }

    fn potex_reference_inputs(count: usize) -> PotexReferenceInputs {
        let step = 0.0725;
        let bound_orbitals = 4;
        let target_large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.17 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let target_small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.09 * row).cos() - 0.01 * row,
                (0.21 * row).sin() + 0.015 * row,
            )
        }));
        let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let bound_large_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
            });
        let bound_small_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
            });
        let bound_large_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                0.02 * row + (0.03 * row * orbital).cos()
            });
        let bound_small_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                -0.015 * row + (0.025 * row * orbital).sin()
            });
        let mut angular_coefficients = Array2::zeros((bound_orbitals, 5));
        angular_coefficients[(0, 0)] = 0.31;
        angular_coefficients[(1, 0)] = -0.18;
        angular_coefficients[(2, 0)] = 0.27;
        angular_coefficients[(2, 1)] = -0.11;
        angular_coefficients[(3, 0)] = 0.19;
        angular_coefficients[(3, 1)] = 0.07;
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        PotexReferenceInputs {
            target_large_component,
            target_small_component,
            target_large_coefficients,
            target_small_coefficients,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            angular_coefficients,
            orbital_powers: Array1::from_vec(vec![0.51, 0.57, 0.63, 0.69]),
            kappa: Array1::from_vec(vec![-1, 1, -2, 2]),
            orbital_lengths: Array1::from_vec(vec![9, 8, 7, 9]),
            normalization: Array1::from_vec(vec![1.01, 1.02, 1.03, 1.04]),
            radii,
            target_power: 0.75,
            target_kappa: -2,
            target_normalization: 1.08,
            speed_of_light: 137.035_999_084,
            step,
            coefficient_count: 6,
            source_len: 9,
            active_len: count,
            radial_output_count: 7,
            bound_orbital_count: bound_orbitals,
        }
    }

    struct PotdvpReferenceInputs {
        nuclear_coefficients: Array1<Real>,
        large_coefficients: Array2<Real>,
        small_coefficients: Array2<Real>,
        electron_counts: Array1<Real>,
        kappa: Array1<i32>,
        normalization: Array1<Real>,
        radii: Array1<Real>,
        speed_of_light: Real,
        coefficient_count: usize,
        orbital_count: usize,
    }

    impl PotdvpReferenceInputs {
        fn as_potential_input(&self) -> FovrgPotentialDevelopmentInput<'_> {
            FovrgPotentialDevelopmentInput {
                nuclear_coefficients: self.nuclear_coefficients.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                electron_counts: self.electron_counts.view(),
                kappa: self.kappa.view(),
                normalization: self.normalization.view(),
                radii: self.radii.view(),
                speed_of_light: self.speed_of_light,
                coefficient_count: self.coefficient_count,
                orbital_count: self.orbital_count,
            }
        }
    }

    fn potdvp_reference_inputs(count: usize) -> PotdvpReferenceInputs {
        let step = 0.0725;
        let bound_orbitals = 4;
        let large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.02 * row + (0.03 * row * orbital).cos()
        });
        let small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            -0.015 * row + (0.025 * row * orbital).sin()
        });
        let nuclear_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            -0.35 + 0.045 * row + 0.002 * row * row
        }));
        let electron_counts = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
            let orbital = orbital as Real;
            0.45 * orbital + 0.1
        }));
        let kappa = Array1::from_vec(vec![-1, 1, -2, 3]);
        let normalization = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
            let orbital = orbital as Real;
            1.0 + 0.013 * orbital
        }));
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        PotdvpReferenceInputs {
            nuclear_coefficients,
            large_coefficients,
            small_coefficients,
            electron_counts,
            kappa,
            normalization,
            radii,
            speed_of_light: 137.035_999_084,
            coefficient_count: 8,
            orbital_count: 5,
        }
    }

    fn assert_complex_close(
        actual: Complex,
        expected_re: Real,
        expected_im: Real,
        tolerance: Real,
    ) {
        assert_close(actual.re, expected_re, tolerance);
        assert_close(actual.im, expected_im, tolerance);
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
