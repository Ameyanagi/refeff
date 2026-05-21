//! Public data structures for FEFF `GENFMT` helper routines.

use ndarray::{Array1, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView6};
use thiserror::Error;

use crate::{Complex, Real};

/// Inputs for FEFF `GENFMT/rot3i.f90` initial-state rotation matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialStateRotationInput {
    /// FEFF `lxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mxp1`, equal to `mmax + 1`.
    pub mmaxp1: usize,
    /// FEFF `beta(ileg)` scattering angle in radians.
    pub beta_angle: Real,
}

/// Inputs for FEFF `GENFMT/rdpath.f90` path angle construction.
#[derive(Debug, Clone, Copy)]
pub struct PathRotationInput<'a> {
    /// Path atom coordinates as `(nleg, 3)`.
    ///
    /// Row `0` is FEFF `rat(:,1)`, and row `nleg - 1` is the absorber row
    /// used as FEFF `rat(:,nleg)`. Coordinates are used as supplied; callers
    /// should perform any Angstrom/Bohr conversion before calling this helper.
    pub positions: ArrayView2<'a, Real>,
    /// Whether to include FEFF's extra z-axis polarization pseudo-leg.
    pub polarized: bool,
}

/// Inputs for FEFF `GENFMT/setlam.f90` lambda-index selection.
#[derive(Debug, Clone, Copy)]
pub struct LambdaIndexInput<'a> {
    /// FEFF `icalc` selector: `0..=9` for exact order, `10` for the cute
    /// heuristic, or a negative encoded `(nmax, mmax, iord)` request.
    pub calculation: i32,
    /// FEFF one-based energy index `ie`; the cute heuristic raises `nmax` for
    /// `ie >= 42`.
    pub energy_index: usize,
    /// FEFF `nsc`, used to detect single-scattering paths.
    pub scattering_count: usize,
    /// FEFF `ilinit`, the initial-state angular momentum.
    pub initial_l: usize,
    /// FEFF `beta(1:nleg)` path scattering angles in radians.
    pub beta_angles: &'a [Real],
    /// FEFF `lamtot`, the capacity of `mlam` and `nlam`.
    pub lambda_capacity: usize,
    /// FEFF `mtot`, the maximum magnetic index dimension.
    pub max_m: usize,
    /// FEFF `ntot`, the maximum order index dimension.
    pub max_n: usize,
}

/// Inputs for FEFF `GENFMT/xstar.f90` central-atom plane-wave factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XStarInput {
    /// FEFF `eps1`: primary polarization vector.
    pub primary_polarization: [Real; 3],
    /// FEFF `eps2`: secondary polarization vector for elliptic polarization.
    pub secondary_polarization: [Real; 3],
    /// FEFF `vec1`: direction to the first atom in the path.
    pub first_leg: [Real; 3],
    /// FEFF `vec2`: direction to the last atom in the path.
    pub last_leg: [Real; 3],
    /// FEFF `ndeg`, the path degeneracy used for this approximation.
    pub degeneracy: Real,
    /// FEFF `ilinit`, supported by the embedded Legendre table for `1..=4`.
    pub initial_l: usize,
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
}

/// Inputs for FEFF `GENFMT/sclmz.f90` curved-wave polynomial tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvedWavePolynomialInput {
    /// FEFF `lmaxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mmaxp1`; columns above `lmaxp1` are retained as zeroes.
    pub mmaxp1: usize,
    /// FEFF complex path length `rho(ileg)`.
    pub rho: Complex,
}

/// Inputs for FEFF `GENFMT/snlm.f90` Legendre-normalization tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtLegendreNormalizationInput {
    /// FEFF `lmaxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub mmaxp1: usize,
}

/// Inputs for FEFF `GENFMT/fmtrxi.f90` scattering-amplitude F matrices.
#[derive(Debug, Clone, Copy)]
pub struct ScatteringAmplitudeMatrixInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lam1x`, the active row lambda count.
    pub left_lambda_count: usize,
    /// FEFF `lam2x`, the active column lambda count.
    pub right_lambda_count: usize,
    /// FEFF signed phase vector for one energy and potential.
    ///
    /// The vector length must be odd. Rust index `phase_offset + l` stores
    /// FEFF `ph(ie,l,ipot)`, and `phase_offset - l` stores `ph(ie,-l,ipot)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
    /// FEFF `lmax(ie,ipot)`, inclusive.
    pub angular_limit: usize,
    /// FEFF `clmi(:,:,ileg)` table for the first leg.
    pub first_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `clmi(:,:,ilegp)` table for the following leg.
    pub second_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `dri(:,:,:,ilegp)` rotation matrix.
    ///
    /// Rust indices are `(l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotation: ArrayView3<'a, Real>,
    /// Magnetic-index offset for the second and third axes of `rotation`.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(ileg)` phase factor.
    pub eta: Real,
}

/// Inputs for FEFF `GENFMT/mmtrxi.f90` polarized scattering-amplitude matrices.
#[derive(Debug, Clone, Copy)]
pub struct PolarizedScatteringAmplitudeInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lam1x`, the active square lambda dimension.
    pub lambda_count: usize,
    /// FEFF transition angular momenta `lind(1:8)`.
    ///
    /// Negative entries are ignored, matching FEFF transition slots that are
    /// not active for the selected edge and polarization.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF radial transition factors `rkk(ie,1:8)` for one energy.
    pub radial_factors: ArrayView1<'a, Complex>,
    /// FEFF `bmati(-mtot:mtot,1:8,-mtot:mtot,1:8)`.
    ///
    /// Rust indices are `(m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrix: ArrayView4<'a, Complex>,
    /// Magnetic-index offset for the first and third axes of `transition_matrix`.
    pub transition_magnetic_offset: usize,
    /// FEFF `clmi(:,:,ileg)` table for the first leg.
    pub first_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `clmi(:,:,ilegp)` table for the following leg.
    pub second_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(ileg)` phase factor.
    pub eta: Real,
}

/// Rotation inputs for FEFF `GENFMT/mmtr.f90` matrix assembly.
#[derive(Debug, Clone, Copy)]
pub enum TransitionRotationInput<'a> {
    /// FEFF `ipol != 0`: use separate rotations from polarization to first
    /// leg and last leg to polarization, plus the two azimuthal phase factors.
    Polarized {
        /// FEFF `dri(:,:,:,nsc+2)`, angle between z and first leg.
        first_rotation: ArrayView3<'a, Real>,
        /// FEFF `dri(:,:,:,nleg)`, angle between last leg and z.
        last_rotation: ArrayView3<'a, Real>,
        /// FEFF `eta(0)`, gamma between polarization and first leg.
        first_eta: Real,
        /// FEFF `eta(nsc+2)`, alpha between last leg and polarization.
        last_eta: Real,
    },
    /// FEFF `ipol == 0`: use the precombined first-to-last-leg rotation.
    Unpolarized {
        /// FEFF `dri(:,:,:,nsc+1)`, angle between last leg and first leg.
        combined_rotation: ArrayView3<'a, Real>,
    },
}

/// Inputs for FEFF `GENFMT/mmtr.f90` energy-independent transition matrix.
#[derive(Debug, Clone, Copy)]
pub struct EnergyIndependentMatrixInput<'a> {
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `bmat(-lx:lx,0:1,1:8,-lx:lx,0:1,1:8)`.
    ///
    /// Rust indices are `(m1 + transition_magnetic_offset, spin1, k1,
    /// m2 + transition_magnetic_offset, spin2, k2)`.
    pub transition_b_matrix: ArrayView6<'a, Complex>,
    /// Magnetic-index offset for the first and fourth `transition_b_matrix`
    /// axes.
    pub transition_magnetic_offset: usize,
    /// FEFF selected spin index `is`.
    pub spin_index: usize,
    /// FEFF `ilinit`, the initial orbital angular-momentum limit.
    pub initial_l: usize,
    /// FEFF `mtot`, the output magnetic-index limit.
    pub magnetic_limit: usize,
    /// Magnetic-index offset for all rotation matrices.
    pub rotation_magnetic_offset: usize,
    /// Polarized or unpolarized FEFF rotation branch.
    pub rotations: TransitionRotationInput<'a>,
}

/// Compact FEFF `rot3i` rotation table for one path leg.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialStateRotation {
    /// FEFF `dri(il,m1+mtot+1,m2+mtot+1,ileg)` without unused global padding.
    ///
    /// Rust indices are `(il - 1, m1 + magnetic_offset, m2 + magnetic_offset)`.
    pub matrix: Array3<Real>,
    /// Offset added to signed magnetic indices before indexing `matrix`.
    pub magnetic_offset: usize,
}

/// FEFF `rdpath` angle tables for one path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathRotationAngles {
    /// FEFF `beta(1:nangle)` scattering angles in radians.
    pub beta_angles: Array1<Real>,
    /// FEFF `eta(0:nleg+1)` azimuthal phase factors.
    ///
    /// Rust index `j` intentionally maps to FEFF `eta(j)` so the polarized
    /// endpoints remain directly addressable as `eta_values[0]` and
    /// `eta_values[nleg + 1]`.
    pub eta_values: Array1<Real>,
    /// FEFF `ri(1:nleg)` leg lengths in the same units as the input positions.
    pub leg_lengths: Array1<Real>,
}

/// FEFF lambda index arrays and associated `setlam` metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaIndexSet {
    /// FEFF `mlam(1:lamx)` magnetic indices.
    pub m_indices: Array1<i32>,
    /// FEFF `nlam(1:lamx)` order indices.
    pub n_indices: Array1<i32>,
    /// FEFF `laml0x`: prefix count whose entries are within `ilinit`.
    pub initial_l_prefix_len: usize,
    /// FEFF `mmaxp1`, computed after capacity truncation and ordering.
    pub max_m_plus_one: usize,
    /// FEFF final `nmax`, computed after capacity truncation and ordering.
    pub max_n: usize,
    /// FEFF `iord`, the requested Rehr-Albers order.
    pub order: i32,
    /// Requested `nmax` before lambda-capacity truncation.
    pub requested_n_max: usize,
    /// Requested `mmax` before lambda-capacity truncation.
    pub requested_m_max: usize,
    /// Whether FEFF would have logged `Lambda array filled, some order lost`.
    pub truncated: bool,
}

/// Error returned by FEFF `GENFMT` helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GenfmtError {
    /// FEFF only defines nonnegative `icalc` values through `10`.
    #[error("undefined FEFF lambda calculation {calculation}")]
    UndefinedLambdaCalculation { calculation: i32 },
    /// A negative `icalc` could not be decoded safely.
    #[error("lambda calculation code {calculation} cannot be decoded safely")]
    LambdaCodeOverflow { calculation: i32 },
    /// The cute heuristic needs finite beta angles.
    #[error("beta angle at index {index} must be finite, got {value}")]
    NonFiniteBetaAngle { index: usize, value: Real },
    /// A generated FEFF integer field would overflow.
    #[error("lambda field {field}={value} does not fit in i32")]
    IntegerOverflow { field: &'static str, value: usize },
    /// GENFMT angular limits must be positive and fit index calculations.
    #[error("invalid GENFMT angular limit {name}={value}")]
    InvalidAngularLimit { name: &'static str, value: usize },
    /// FEFF `rot3i` requires a finite beta angle.
    #[error("rotation beta angle must be finite")]
    NonFiniteRotationAngle,
    /// FEFF path angle construction needs at least one path row.
    #[error("path positions must contain at least one leg")]
    EmptyPath,
    /// FEFF path angle construction uses three Cartesian coordinates per row.
    #[error("path positions must have exactly 3 coordinate columns, got {columns}")]
    InvalidPathCoordinateColumns { columns: usize },
    /// FEFF path coordinates must be finite.
    #[error(
        "path position leg index {leg_index} component {component} must be finite, got {value}"
    )]
    NonFinitePathCoordinate {
        leg_index: usize,
        component: usize,
        value: Real,
    },
    /// FEFF `sclmz` needs a finite complex path length.
    #[error("{field} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        field: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// FEFF `sclmz` divides by the complex path length.
    #[error("{field} must be nonzero")]
    ZeroComplex { field: &'static str },
    /// FEFF `xstar` only tabulates Legendre coefficients through `ilinit=4`.
    #[error("initial angular momentum {initial_l} is outside GENFMT xstar table range 1..=4")]
    InvalidInitialAngularMomentum { initial_l: usize },
    /// Scalar GENFMT inputs must be finite.
    #[error("{field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Vector GENFMT inputs must have finite components.
    #[error("{field}[{index}] must be finite, got {value}")]
    NonFiniteVector {
        field: &'static str,
        index: usize,
        value: Real,
    },
    /// FEFF `xxcos` is undefined for zero-length vectors.
    #[error("{field} must have nonzero length")]
    ZeroVector { field: &'static str },
    /// Generated lambda indices exceed the caller's FEFF dimensions.
    #[error(
        "lambda selection exceeded dimensions: mmaxp1={max_m_plus_one}, nmax={max_n}, mtot={max_m}, ntot={max_n_limit}"
    )]
    DimensionExceeded {
        max_m_plus_one: usize,
        max_n: usize,
        max_m: usize,
        max_n_limit: usize,
    },
    /// A lambda count exceeds the supplied lambda-index arrays.
    #[error("{name}={requested} exceeds lambda array length {available}")]
    LambdaCountOutOfRange {
        name: &'static str,
        requested: usize,
        available: usize,
    },
    /// FEFF signed phase vectors must cover `-lmax..=lmax`.
    #[error("signed phase vector length {length} must be odd and nonzero")]
    InvalidSignedPhaseShape { length: usize },
    /// A FEFF lambda index cannot be represented safely.
    #[error("lambda {field} at index {index} has invalid value {value}")]
    InvalidLambdaIndex {
        index: usize,
        field: &'static str,
        value: i32,
    },
    /// An ndarray axis is too short for FEFF-compatible indexing.
    #[error("{table} axis {axis} length {length} is smaller than required {required}")]
    TableAxisTooShort {
        table: &'static str,
        axis: &'static str,
        length: usize,
        required: usize,
    },
    /// A complex table entry must be finite.
    #[error("{table}({row},{column}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTableComplex {
        table: &'static str,
        row: usize,
        column: usize,
        real: Real,
        imaginary: Real,
    },
    /// A complex tensor entry must be finite.
    #[error("{table}({i0},{i1},{i2},{i3}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTensorComplex {
        table: &'static str,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        real: Real,
        imaginary: Real,
    },
    /// A six-dimensional complex tensor entry must be finite.
    #[error("{table}({i0},{i1},{i2},{i3},{i4},{i5}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTensor6Complex {
        table: &'static str,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
        real: Real,
        imaginary: Real,
    },
    /// A real table entry must be finite.
    #[error("{table}({row},{column}) must be finite, got {value}")]
    NonFiniteTableScalar {
        table: &'static str,
        row: usize,
        column: usize,
        value: Real,
    },
    /// FEFF divides by `xnlm(m,l)` in `fmtrxi`.
    #[error("xnlm({magnetic},{angular_momentum}) must be nonzero")]
    ZeroLegendreNormalization {
        angular_momentum: usize,
        magnetic: usize,
    },
}
