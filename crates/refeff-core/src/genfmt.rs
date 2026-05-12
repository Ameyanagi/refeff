//! FEFF `GENFMT` helper routines.
//!
//! This module ports small, self-contained setup routines used by FEFF's
//! curved-wave multiple-scattering formatter. `lambda_indices` is the Rust
//! equivalent of `GENFMT/setlam.f90`: it builds the Rehr-Albers lambda index
//! arrays `(m, n)` from FEFF's `icalc` mode, path order, and dimension limits.

use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView6,
    ShapeBuilder,
};
use thiserror::Error;

use crate::{Complex, Real};

const ONE_DEGREE_RADIANS: Real = 0.017_453_292_52;
const RDPATH_EPSILON: Real = 1.0e-6;

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

/// Build FEFF `rot3i` real rotation matrices for a single path leg.
///
/// The recursion is the Edmonds small-`d` rotation used by FEFF before GENFMT
/// matrix assembly. FEFF writes into a globally padded `dri` array; this helper
/// returns only the active magnetic range `-(mxp1-1)..=(mxp1-1)` for each
/// `il`, with zeroes retained where FEFF would not fill entries.
pub fn initial_state_rotation(
    input: InitialStateRotationInput,
) -> Result<InitialStateRotation, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;
    if !input.beta_angle.is_finite() {
        return Err(GenfmtError::NonFiniteRotationAngle);
    }

    let magnetic_offset = input.mmaxp1 - 1;
    let m_dim = checked_double_plus_one("mmaxp1", magnetic_offset)?;
    let mut matrix = Array3::<Real>::zeros((input.lmaxp1, m_dim, m_dim).f());

    let work_l = input.lmaxp1.max(2);
    let ndm = input
        .lmaxp1
        .checked_add(input.mmaxp1)
        .and_then(|value| value.checked_sub(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?;
    let work_m = checked_double_plus_one("lmaxp1", work_l)?
        .checked_sub(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?
        .max(ndm)
        .max(3);
    let mut work = Array3::<Real>::zeros((work_l + 1, work_m + 1, work_m + 1).f());
    fill_initial_state_rotation_work(input.lmaxp1, input.mmaxp1, input.beta_angle, &mut work);
    copy_initial_state_rotation(
        input.lmaxp1,
        input.mmaxp1,
        magnetic_offset,
        &work,
        &mut matrix,
    )?;

    Ok(InitialStateRotation {
        matrix,
        magnetic_offset,
    })
}

/// Compute FEFF `rdpath` path rotations, azimuths, and leg lengths.
///
/// FEFF uses these `beta`, `eta`, and `ri` tables to choose lambda indices and
/// rotate GENFMT scattering amplitudes into each local path frame. This helper
/// ports only the deterministic geometry calculation from `rdpath.f90`; it
/// does not read path files, mutate global module state, or convert units.
pub fn path_rotation_angles(
    input: PathRotationInput<'_>,
) -> Result<PathRotationAngles, GenfmtError> {
    let nleg = input.positions.shape()[0];
    let coordinate_columns = input.positions.shape()[1];
    if nleg == 0 {
        return Err(GenfmtError::EmptyPath);
    }
    if coordinate_columns != 3 {
        return Err(GenfmtError::InvalidPathCoordinateColumns {
            columns: coordinate_columns,
        });
    }

    let padded_len = nleg
        .checked_add(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "nleg",
            value: nleg,
        })?;
    let mut rat = vec![[0.0; 3]; padded_len];
    for leg_index in 0..nleg {
        for (component, coordinate) in rat[leg_index + 1].iter_mut().enumerate() {
            let value = input.positions[(leg_index, component)];
            if !value.is_finite() {
                return Err(GenfmtError::NonFinitePathCoordinate {
                    leg_index,
                    component,
                    value,
                });
            }
            *coordinate = value;
        }
    }
    rat[0] = rat[nleg];

    if input.polarized {
        rat[nleg + 1] = rat[nleg];
        rat[nleg + 1][2] += 1.0;
        let value = rat[nleg + 1][2];
        if !value.is_finite() {
            return Err(GenfmtError::NonFinitePathCoordinate {
                leg_index: nleg,
                component: 2,
                value,
            });
        }
    }

    let nangle =
        nleg.checked_add(usize::from(input.polarized))
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "nleg",
                value: nleg,
            })?;
    let mut beta_angles = Array1::<Real>::zeros(nangle);
    let mut eta_values = Array1::<Real>::zeros(padded_len);
    let mut leg_lengths = Array1::<Real>::zeros(nleg);
    let work_len = nangle
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "nangle",
            value: nangle,
        })?;
    let mut alpha = vec![0.0; work_len];
    let mut gamma = vec![0.0; work_len];
    let nsc = nleg - 1;

    for j in 1..=nangle {
        let (i, ip1, im1, fixed_previous) = if j == nsc + 1 {
            (0, if input.polarized { nleg + 1 } else { 1 }, nsc, false)
        } else if j == nsc + 2 {
            (0, 1, nleg + 1, true)
        } else {
            (j, j + 1, j - 1, false)
        };

        let forward = rdpath_trig(vector_difference(rat[ip1], rat[i]));
        let previous = if fixed_previous {
            rdpath_trig([0.0, 0.0, 1.0])
        } else {
            rdpath_trig(vector_difference(rat[i], rat[im1]))
        };

        let cppp = previous.cp * forward.cp + previous.sp * forward.sp;
        let sppp = forward.sp * previous.cp - forward.cp * previous.sp;
        let phi = previous.sp.atan2(previous.cp);
        let phip = forward.sp.atan2(forward.cp);
        let alph = Complex::new(
            -(previous.st * forward.ct - previous.ct * forward.st * cppp),
            forward.st * sppp,
        );
        let gamm = Complex::new(
            -(previous.st * forward.ct * cppp - previous.ct * forward.st),
            -previous.st * sppp,
        );
        let beta_cosine =
            bounded_beta_cosine(previous.ct * forward.ct + previous.st * forward.st * cppp)?;
        let alpha_angle = rdpath_arg(alph, phip - phi);
        let gamma_angle = rdpath_arg(gamm, 0.0);

        beta_angles[j - 1] = beta_cosine.acos();
        alpha[j] = std::f64::consts::PI - gamma_angle;
        gamma[j] = std::f64::consts::PI - alpha_angle;

        if j <= nleg {
            leg_lengths[j - 1] = point_distance(rat[i], rat[im1]);
        }
    }

    alpha[0] = alpha[nangle];
    for j in 1..=nleg {
        eta_values[j] = alpha[j - 1] + gamma[j];
    }
    if input.polarized {
        eta_values[0] = gamma[nleg + 1];
        eta_values[nleg + 1] = alpha[nleg];
    }

    Ok(PathRotationAngles {
        beta_angles,
        eta_values,
        leg_lengths,
    })
}

/// Compute FEFF `xstar`, the central-atom plane-wave polarization factor.
///
/// FEFF evaluates the orientationally averaged `ystar` expression for the
/// primary polarization and, when `ellipticity != 0`, adds the secondary
/// polarization weighted by `ellipticity^2`. The vector cosines match
/// `xxcos` from `xstar.f90`, but zero-length and non-finite inputs are reported
/// as errors instead of allowing division by zero.
pub fn xstar(input: XStarInput) -> Result<Real, GenfmtError> {
    if !(1..=4).contains(&input.initial_l) {
        return Err(GenfmtError::InvalidInitialAngularMomentum {
            initial_l: input.initial_l,
        });
    }
    validate_finite_scalar("degeneracy", input.degeneracy)?;
    validate_finite_scalar("ellipticity", input.ellipticity)?;

    let x = normalized_dot("first_leg", input.first_leg, "last_leg", input.last_leg)?;
    let primary_y = normalized_dot(
        "primary_polarization",
        input.primary_polarization,
        "first_leg",
        input.first_leg,
    )?;
    let primary_z = normalized_dot(
        "primary_polarization",
        input.primary_polarization,
        "last_leg",
        input.last_leg,
    )?;
    let mut value = ystar(input.initial_l, x, primary_y, primary_z);

    if input.ellipticity != 0.0 {
        let secondary_y = normalized_dot(
            "secondary_polarization",
            input.secondary_polarization,
            "first_leg",
            input.first_leg,
        )?;
        let secondary_z = normalized_dot(
            "secondary_polarization",
            input.secondary_polarization,
            "last_leg",
            input.last_leg,
        )?;
        value += input.ellipticity
            * input.ellipticity
            * ystar(input.initial_l, x, secondary_y, secondary_z);
    }

    Ok(input.degeneracy * value / (1.0 + input.ellipticity * input.ellipticity))
}

/// Build FEFF `sclmz` curved-wave Rehr-Albers polynomial factors.
///
/// FEFF stores the result in `clmi(il, im, ileg)`. This Rust helper returns the
/// active two-dimensional leg table in Fortran-order ndarray storage, with
/// FEFF one-based indices mapped to Rust `(il - 1, im - 1)`. The row dimension
/// is `lmaxp1 + 1` because FEFF fills the `im + 1` row for diagonal magnetic
/// recurrences; the column dimension is the requested `mmaxp1`, with columns
/// above `lmaxp1` left at zero.
pub fn curved_wave_polynomials(
    input: CurvedWavePolynomialInput,
) -> Result<Array2<Complex>, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;
    validate_finite_complex("rho", input.rho)?;
    if input.rho == Complex::new(0.0, 0.0) {
        return Err(GenfmtError::ZeroComplex { field: "rho" });
    }

    let rows = input
        .lmaxp1
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: input.lmaxp1,
        })?;
    let mut table = Array2::zeros((rows, input.mmaxp1).f());
    let one = Complex::new(1.0, 0.0);
    let z = -Complex::new(0.0, 1.0) / input.rho;

    table[(0, 0)] = one;
    table[(1, 0)] = table[(0, 0)] - z;

    let lmax = input.lmaxp1 - 1;
    for il in 2..=lmax {
        table[(il, 0)] = table[(il - 2, 0)]
            - z * checked_odd_factor(il, "lmaxp1", input.lmaxp1)? * table[(il - 1, 0)];
    }

    let mut cmm = one;
    let mmxp1 = input.mmaxp1.min(input.lmaxp1);
    for im in 2..=mmxp1 {
        let m = im - 1;
        cmm = -cmm * checked_odd_factor(m, "mmaxp1", input.mmaxp1)? * z;
        table[(im - 1, im - 1)] = cmm;
        table[(im, im - 1)] =
            cmm * checked_odd_factor(im, "mmaxp1", input.mmaxp1)? * (one - (im as Real) * z);

        for il in (im + 1)..=lmax {
            let l = il - 1;
            table[(il, im - 1)] = table[(l - 1, im - 1)]
                - checked_odd_factor(il, "lmaxp1", input.lmaxp1)?
                    * z
                    * (table[(il - 1, im - 1)] + table[(il - 1, m - 1)]);
        }
    }

    Ok(table)
}

/// Build FEFF `fmtrxi` scattering-amplitude F matrix for one energy and leg pair.
///
/// The output is equivalent to FEFF `fmati(1:lam1x,1:lam2x,ilegp)` and uses
/// Fortran-order ndarray storage. The implementation keeps FEFF's j-averaged
/// phase-shift branch,
/// `(exp(2i ph(-l))-1)/(2i) * (l+1) + (exp(2i ph(l))-1)/(2i) * l`,
/// while reporting invalid shapes and non-finite inputs as Rust errors instead
/// of relying on common-block dimensions.
pub fn scattering_amplitude_matrix(
    input: ScatteringAmplitudeMatrixInput<'_>,
) -> Result<Array2<Complex>, GenfmtError> {
    let phase_offset = validate_scattering_amplitude_input(input)?;
    let angular_count = checked_count("angular_limit", input.angular_limit)?;
    let max_lambda_count = input.left_lambda_count.max(input.right_lambda_count);
    let max_m = input.angular_limit;
    let max_n = lambda_n_limit(input.n_indices, max_lambda_count)?;
    let max_m_count = checked_count("angular_limit", max_m)?;
    let max_n_count = checked_count("nlam", max_n)?;
    let mut gam = Array3::<Complex>::zeros((angular_count, max_m_count, max_n_count).f());
    let mut gamtl = Array3::<Complex>::zeros((angular_count, max_m_count, max_n_count).f());

    for l in 0..=input.angular_limit {
        let t_matrix = averaged_t_matrix(input.phase_shifts, phase_offset, l)?;
        for lambda in 0..max_lambda_count {
            let magnetic = lambda_abs_magnetic(input.m_indices[lambda], lambda)?;
            if magnetic > l {
                continue;
            }
            let order = lambda_order(input.n_indices[lambda], lambda)?;
            if order > max_n {
                continue;
            }

            if lambda < input.left_lambda_count {
                let combined_mn =
                    magnetic
                        .checked_add(order)
                        .ok_or(GenfmtError::InvalidLambdaIndex {
                            index: lambda,
                            field: "nlam",
                            value: input.n_indices[lambda],
                        })?;
                let normalization = xnlm_entry(input.xnlm, magnetic, l)?;
                gam[(l, magnetic, order)] = if combined_mn <= l {
                    let sign = alternating_sign(magnetic);
                    normalization
                        * sign
                        * complex_entry(
                            input.first_leg_polynomials,
                            "first_leg_polynomials",
                            l,
                            combined_mn,
                        )?
                } else {
                    Complex::new(0.0, 0.0)
                };
            }

            if lambda < input.right_lambda_count {
                let normalization = xnlm_entry(input.xnlm, magnetic, l)?;
                gamtl[(l, magnetic, order)] = t_matrix / normalization
                    * complex_entry(
                        input.second_leg_polynomials,
                        "second_leg_polynomials",
                        l,
                        order,
                    )?;
            }
        }
    }

    let mut matrix =
        Array2::<Complex>::zeros((input.left_lambda_count, input.right_lambda_count).f());
    for left in 0..input.left_lambda_count {
        let m1 = input.m_indices[left];
        let n1 = lambda_order(input.n_indices[left], left)?;
        let abs_m1 = lambda_abs_magnetic(m1, left)?;
        for right in 0..input.right_lambda_count {
            let m2 = input.m_indices[right];
            let n2 = lambda_order(input.n_indices[right], right)?;
            let abs_m2 = lambda_abs_magnetic(m2, right)?;
            let combined_mn = abs_m1
                .checked_add(n1)
                .ok_or(GenfmtError::InvalidLambdaIndex {
                    index: left,
                    field: "nlam",
                    value: input.n_indices[left],
                })?;
            let l_min = abs_m1.max(abs_m2).max(combined_mn).max(n2);
            let mut value = Complex::new(0.0, 0.0);

            for l in l_min..=input.angular_limit {
                if abs_m1 > l || abs_m2 > l {
                    continue;
                }
                let rotation =
                    rotation_entry(input.rotation, input.rotation_magnetic_offset, l, m1, m2)?;
                value += gam[(l, abs_m1, n1)] * gamtl[(l, abs_m2, n2)] * rotation;
            }

            if input.eta != 0.0 {
                value *= (-Complex::new(0.0, 1.0) * input.eta * (m1 as Real)).exp();
            }
            matrix[(left, right)] = value;
        }
    }

    Ok(matrix)
}

/// Build FEFF `mmtrxi` polarized scattering-amplitude matrix.
///
/// This is the polarization branch that contracts FEFF's energy-independent
/// transition matrix `bmati`, radial transition factors `rkk`, curved-wave
/// polynomial tables, and lambda indices into `fmati(1:lam1x,1:lam1x,ilegp)`.
/// The output uses Fortran-order ndarray storage and preserves FEFF's
/// transition loop order.
pub fn polarized_scattering_amplitude_matrix(
    input: PolarizedScatteringAmplitudeInput<'_>,
) -> Result<Array2<Complex>, GenfmtError> {
    validate_polarized_scattering_amplitude_input(input)?;
    let mut matrix = Array2::<Complex>::zeros((input.lambda_count, input.lambda_count).f());
    if input.lambda_count == 0 {
        return Ok(matrix);
    }

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let Some((min_l, max_l)) = active_transition_limits(&transition_l) else {
        return Ok(matrix);
    };
    let angular_count = checked_count("lind", max_l)?;
    let max_n = lambda_n_limit(input.n_indices, input.lambda_count)?;
    let max_n_count = checked_count("nlam", max_n)?;
    let mut gam = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());
    let mut gamtl = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());

    for l in min_l..=max_l {
        let t_matrix = (2 * l + 1) as Real;
        for lambda in 0..input.lambda_count {
            let signed_magnetic = input.m_indices[lambda];
            if signed_magnetic < 0 {
                continue;
            }
            let magnetic = lambda_abs_magnetic(signed_magnetic, lambda)?;
            if magnetic > l {
                continue;
            }
            let order = lambda_order(input.n_indices[lambda], lambda)?;
            let combined_mn =
                magnetic
                    .checked_add(order)
                    .ok_or(GenfmtError::InvalidLambdaIndex {
                        index: lambda,
                        field: "nlam",
                        value: input.n_indices[lambda],
                    })?;
            let normalization = xnlm_entry(input.xnlm, magnetic, l)?;
            gam[(l, magnetic, order)] = if combined_mn <= l {
                let sign = alternating_sign(magnetic);
                normalization
                    * sign
                    * complex_entry(
                        input.first_leg_polynomials,
                        "first_leg_polynomials",
                        l,
                        combined_mn,
                    )?
            } else {
                Complex::new(0.0, 0.0)
            };
            gamtl[(l, magnetic, order)] = t_matrix / normalization
                * complex_entry(
                    input.second_leg_polynomials,
                    "second_leg_polynomials",
                    l,
                    order,
                )?;
        }
    }

    for left in 0..input.lambda_count {
        let m1 = input.m_indices[left];
        let n1 = lambda_order(input.n_indices[left], left)?;
        let abs_m1 = lambda_abs_magnetic(m1, left)?;
        for right in 0..input.lambda_count {
            let m2 = input.m_indices[right];
            let n2 = lambda_order(input.n_indices[right], right)?;
            let abs_m2 = lambda_abs_magnetic(m2, right)?;
            let mut value = Complex::new(0.0, 0.0);

            for (k1, &maybe_l1) in transition_l.iter().enumerate() {
                let Some(l1) = maybe_l1 else {
                    continue;
                };
                if abs_m1 > l1 {
                    continue;
                }
                for (k2, &maybe_l2) in transition_l.iter().enumerate() {
                    let Some(l2) = maybe_l2 else {
                        continue;
                    };
                    if abs_m2 > l2 {
                        continue;
                    }
                    value += transition_matrix_entry(
                        input.transition_matrix,
                        input.transition_magnetic_offset,
                        m1,
                        k1,
                        m2,
                        k2,
                    )? * complex_vector_entry(input.radial_factors, "radial_factors", k1)?
                        * complex_vector_entry(input.radial_factors, "radial_factors", k2)?
                        * gam[(l1, abs_m1, n1)]
                        * gamtl[(l2, abs_m2, n2)];
                }
            }

            matrix[(left, right)] =
                value * (-Complex::new(0.0, 1.0) * input.eta * (m1 as Real)).exp();
        }
    }

    Ok(matrix)
}

/// Build FEFF `mmtr` energy-independent transition matrix.
///
/// FEFF calls `bcoef` before this step; this helper starts from the resulting
/// `bmat` tensor and applies the `mmtr.f90` rotation and phase rules. The
/// returned ndarray has FEFF `bmati(mu1,k1,mu2,k2)` axis order with signed
/// magnetic indices shifted by `magnetic_limit`.
pub fn energy_independent_transition_matrix(
    input: EnergyIndependentMatrixInput<'_>,
) -> Result<Array4<Complex>, GenfmtError> {
    validate_energy_independent_matrix_input(input)?;
    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let transition_count = transition_l.len();
    let magnetic_dim = checked_double_plus_one("magnetic_limit", input.magnetic_limit)?;
    let mut matrix = Array4::<Complex>::zeros(
        (
            magnetic_dim,
            transition_count,
            magnetic_dim,
            transition_count,
        )
            .f(),
    );
    if transition_count == 0 {
        return Ok(matrix);
    }

    let active_limit = input.magnetic_limit.min(input.initial_l);
    let active_limit_i32 = checked_i32("initial_l", active_limit)?;
    for mu1 in -active_limit_i32..=active_limit_i32 {
        let mu1_index = signed_magnetic_index(
            mu1,
            input.magnetic_limit,
            "magnetic_limit",
            "bmati",
            "mu1",
            magnetic_dim,
        )?;
        for mu2 in -active_limit_i32..=active_limit_i32 {
            let mu2_index = signed_magnetic_index(
                mu2,
                input.magnetic_limit,
                "magnetic_limit",
                "bmati",
                "mu2",
                magnetic_dim,
            )?;

            match input.rotations {
                TransitionRotationInput::Polarized {
                    first_rotation,
                    last_rotation,
                    first_eta,
                    last_eta,
                } => {
                    for (k1, &maybe_l1) in transition_l.iter().enumerate() {
                        let Some(l1) = maybe_l1 else {
                            continue;
                        };
                        let l1_i32 = checked_i32("lind", l1)?;
                        for (k2, &maybe_l2) in transition_l.iter().enumerate() {
                            let Some(l2) = maybe_l2 else {
                                continue;
                            };
                            let l2_i32 = checked_i32("lind", l2)?;
                            for m1 in -l1_i32..=l1_i32 {
                                for m2 in -l2_i32..=l2_i32 {
                                    let phase = (-Complex::new(0.0, 1.0)
                                        * (last_eta * (m2 as Real) + first_eta * (m1 as Real)))
                                        .exp();
                                    let first = rotation_entry(
                                        first_rotation,
                                        input.rotation_magnetic_offset,
                                        l1,
                                        mu1,
                                        m1,
                                    )?;
                                    let last = rotation_entry(
                                        last_rotation,
                                        input.rotation_magnetic_offset,
                                        l2,
                                        m2,
                                        mu2,
                                    )?;
                                    matrix[(mu1_index, k1, mu2_index, k2)] +=
                                        transition_b_matrix_entry(
                                            input.transition_b_matrix,
                                            input.transition_magnetic_offset,
                                            m1,
                                            input.spin_index,
                                            k1,
                                            m2,
                                            k2,
                                        )? * phase
                                            * first
                                            * last;
                                }
                            }
                        }
                    }
                }
                TransitionRotationInput::Unpolarized { combined_rotation } => {
                    for (k1, &maybe_l1) in transition_l.iter().enumerate() {
                        let Some(l1) = maybe_l1 else {
                            continue;
                        };
                        matrix[(mu1_index, k1, mu2_index, k1)] += transition_b_matrix_entry(
                            input.transition_b_matrix,
                            input.transition_magnetic_offset,
                            0,
                            input.spin_index,
                            k1,
                            0,
                            k1,
                        )? * rotation_entry(
                            combined_rotation,
                            input.rotation_magnetic_offset,
                            l1,
                            mu1,
                            mu2,
                        )?;
                    }
                }
            }
        }
    }

    Ok(matrix)
}

/// Build FEFF `mlam` and `nlam` arrays from `GENFMT/setlam.f90` rules.
///
/// The returned arrays preserve FEFF's insertion order, including `-m` before
/// `+m`, and then apply FEFF's second pass that moves entries satisfying
/// `n <= ilinit && abs(m) <= ilinit` to the front to minimize `laml0x`.
/// Capacity handling also follows FEFF: if `lamtot` fills, the result is
/// truncated and flagged instead of failing.
pub fn lambda_indices(input: LambdaIndexInput<'_>) -> Result<LambdaIndexSet, GenfmtError> {
    let request = lambda_request(input)?;
    let mut raw = Vec::with_capacity(input.lambda_capacity.min(128));
    let mut truncated = false;

    if request.order >= 0 {
        let order = usize::try_from(request.order).map_err(|_| GenfmtError::IntegerOverflow {
            field: "iord",
            value: request.order.unsigned_abs() as usize,
        })?;
        let valid_n_max = request.n_max.min(order / 2);

        'outer: for n in 0..=valid_n_max {
            let valid_m_max = request.m_max.min(order - 2 * n);
            for m in 0..=valid_m_max {
                if raw.len() >= input.lambda_capacity {
                    truncated = true;
                    break 'outer;
                }
                raw.push((-checked_i32("m", m)?, checked_i32("n", n)?));

                if m != 0 {
                    if raw.len() >= input.lambda_capacity {
                        truncated = true;
                        break 'outer;
                    }
                    raw.push((checked_i32("m", m)?, checked_i32("n", n)?));
                }
            }
        }
    }

    let mut pairs = Vec::with_capacity(raw.len());
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| within_initial_l(m, n, input.initial_l)),
    );
    let initial_l_prefix_len = pairs.len();
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| !within_initial_l(m, n, input.initial_l)),
    );

    let max_m_plus_one = max_lambda_m_plus_one(&pairs)?;
    let max_n = max_lambda_n(&pairs)?;

    if max_n > input.max_n || max_m_plus_one > input.max_m.saturating_add(1) {
        return Err(GenfmtError::DimensionExceeded {
            max_m_plus_one,
            max_n,
            max_m: input.max_m,
            max_n_limit: input.max_n,
        });
    }

    let (m_values, n_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok(LambdaIndexSet {
        m_indices: Array1::from_vec(m_values),
        n_indices: Array1::from_vec(n_values),
        initial_l_prefix_len,
        max_m_plus_one,
        max_n,
        order: request.order,
        requested_n_max: request.n_max,
        requested_m_max: request.m_max,
        truncated,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LambdaRequest {
    order: i32,
    n_max: usize,
    m_max: usize,
}

fn lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    if input.calculation < 0 {
        return decode_lambda_request(input.calculation);
    }

    if input.scattering_count == 1 {
        return Ok(LambdaRequest {
            order: checked_order(input.initial_l, input.initial_l)?,
            n_max: input.initial_l,
            m_max: input.initial_l,
        });
    }

    if input.calculation < 10 {
        let order = input.calculation;
        return Ok(LambdaRequest {
            order,
            n_max: usize::try_from(order / 2).map_err(|_| GenfmtError::IntegerOverflow {
                field: "nmax",
                value: order.unsigned_abs() as usize,
            })?,
            m_max: usize::try_from(order).map_err(|_| GenfmtError::IntegerOverflow {
                field: "mmax",
                value: order.unsigned_abs() as usize,
            })?,
        });
    }

    if input.calculation == 10 {
        return cute_lambda_request(input);
    }

    Err(GenfmtError::UndefinedLambdaCalculation {
        calculation: input.calculation,
    })
}

fn decode_lambda_request(calculation: i32) -> Result<LambdaRequest, GenfmtError> {
    let code = calculation
        .checked_neg()
        .ok_or(GenfmtError::LambdaCodeOverflow { calculation })?;
    let order = (code / 10_000) - 1;
    Ok(LambdaRequest {
        order,
        n_max: usize::try_from(code % 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
        m_max: usize::try_from((code % 10_000) / 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
    })
}

fn cute_lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    let mut m_max = input.initial_l;
    for (index, &angle) in input.beta_angles.iter().enumerate() {
        if !angle.is_finite() {
            return Err(GenfmtError::NonFiniteBetaAngle {
                index,
                value: angle,
            });
        }
        let magnitude = angle.abs();
        let pi_distance = (magnitude - std::f64::consts::PI).abs();
        if magnitude > ONE_DEGREE_RADIANS && pi_distance > ONE_DEGREE_RADIANS {
            m_max = 3;
        }
    }

    let n_max = if input.energy_index >= 42 {
        9
    } else {
        input.initial_l
    };

    Ok(LambdaRequest {
        order: checked_order(n_max, m_max)?,
        n_max,
        m_max,
    })
}

fn checked_order(n_max: usize, m_max: usize) -> Result<i32, GenfmtError> {
    let order = n_max
        .checked_mul(2)
        .and_then(|value| value.checked_add(m_max))
        .ok_or(GenfmtError::IntegerOverflow {
            field: "iord",
            value: n_max,
        })?;
    checked_i32("iord", order)
}

fn checked_i32(field: &'static str, value: usize) -> Result<i32, GenfmtError> {
    i32::try_from(value).map_err(|_| GenfmtError::IntegerOverflow { field, value })
}

fn within_initial_l(m: i32, n: i32, initial_l: usize) -> bool {
    let abs_m = m.unsigned_abs() as usize;
    let Ok(n) = usize::try_from(n) else {
        return false;
    };
    n <= initial_l && abs_m <= initial_l
}

fn max_lambda_m_plus_one(pairs: &[(i32, i32)]) -> Result<usize, GenfmtError> {
    pairs.iter().try_fold(0, |maximum, &(m, _)| {
        if m < 0 {
            return Ok(maximum);
        }
        let plus_one = m.checked_add(1).ok_or(GenfmtError::IntegerOverflow {
            field: "mmaxp1",
            value: m.unsigned_abs() as usize,
        })?;
        let value = usize::try_from(plus_one).map_err(|_| GenfmtError::IntegerOverflow {
            field: "mmaxp1",
            value: m.unsigned_abs() as usize,
        })?;
        Ok(maximum.max(value))
    })
}

fn max_lambda_n(pairs: &[(i32, i32)]) -> Result<usize, GenfmtError> {
    pairs.iter().try_fold(0, |maximum, &(_, n)| {
        if n < 0 {
            return Ok(maximum);
        }
        let value = usize::try_from(n).map_err(|_| GenfmtError::IntegerOverflow {
            field: "nmax",
            value: n.unsigned_abs() as usize,
        })?;
        Ok(maximum.max(value))
    })
}

fn validate_positive_limit(name: &'static str, value: usize) -> Result<(), GenfmtError> {
    if value == 0 || isize::try_from(value).is_err() {
        Err(GenfmtError::InvalidAngularLimit { name, value })
    } else {
        Ok(())
    }
}

fn checked_double_plus_one(name: &'static str, value: usize) -> Result<usize, GenfmtError> {
    value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit { name, value })
}

fn checked_odd_factor(value: usize, name: &'static str, limit: usize) -> Result<Real, GenfmtError> {
    let factor = value.checked_mul(2).and_then(|value| value.checked_sub(1));
    factor
        .map(|value| value as Real)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value: limit })
}

fn checked_count(name: &'static str, value: usize) -> Result<usize, GenfmtError> {
    value
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value })
}

fn validate_scattering_amplitude_input(
    input: ScatteringAmplitudeMatrixInput<'_>,
) -> Result<usize, GenfmtError> {
    let angular_count = checked_count("angular_limit", input.angular_limit)?;
    validate_positive_limit("angular_limit", angular_count)?;
    validate_finite_scalar("eta", input.eta)?;

    let lambda_len = input.m_indices.len().min(input.n_indices.len());
    validate_lambda_count("left_lambda_count", input.left_lambda_count, lambda_len)?;
    validate_lambda_count("right_lambda_count", input.right_lambda_count, lambda_len)?;

    let phase_len = input.phase_shifts.len();
    if phase_len == 0 || phase_len.is_multiple_of(2) {
        return Err(GenfmtError::InvalidSignedPhaseShape { length: phase_len });
    }
    let phase_offset = phase_len / 2;
    let phase_required = phase_offset
        .checked_add(input.angular_limit)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "angular_limit",
            value: input.angular_limit,
        })?;
    ensure_axis_len("phase_shifts", "signed_l", phase_len, phase_required)?;

    ensure_axis_len("xnlm", "m", input.xnlm.shape()[0], angular_count)?;
    ensure_axis_len("xnlm", "l", input.xnlm.shape()[1], angular_count)?;
    ensure_axis_len(
        "first_leg_polynomials",
        "l",
        input.first_leg_polynomials.shape()[0],
        angular_count,
    )?;
    ensure_axis_len(
        "second_leg_polynomials",
        "l",
        input.second_leg_polynomials.shape()[0],
        angular_count,
    )?;
    ensure_axis_len("rotation", "l", input.rotation.shape()[0], angular_count)?;

    let rotation_required = input
        .rotation_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: input.rotation_magnetic_offset,
        })?;
    ensure_axis_len(
        "rotation",
        "m1",
        input.rotation.shape()[1],
        rotation_required,
    )?;
    ensure_axis_len(
        "rotation",
        "m2",
        input.rotation.shape()[2],
        rotation_required,
    )?;

    Ok(phase_offset)
}

fn validate_polarized_scattering_amplitude_input(
    input: PolarizedScatteringAmplitudeInput<'_>,
) -> Result<(), GenfmtError> {
    validate_finite_scalar("eta", input.eta)?;
    let lambda_len = input.m_indices.len().min(input.n_indices.len());
    validate_lambda_count("lambda_count", input.lambda_count, lambda_len)?;

    let transition_count = input.transition_angular_momenta.len();
    ensure_axis_len(
        "radial_factors",
        "transition",
        input.radial_factors.len(),
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "transition1",
        input.transition_matrix.shape()[1],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "transition2",
        input.transition_matrix.shape()[3],
        transition_count,
    )?;

    let magnetic_required = input
        .transition_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "transition_magnetic_offset",
            value: input.transition_magnetic_offset,
        })?;
    ensure_axis_len(
        "transition_matrix",
        "m1",
        input.transition_matrix.shape()[0],
        magnetic_required,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "m2",
        input.transition_matrix.shape()[2],
        magnetic_required,
    )?;

    if let Some((_, max_l)) = active_transition_limits(&transition_angular_momenta(
        input.transition_angular_momenta,
    )?) {
        let angular_count = checked_count("lind", max_l)?;
        ensure_axis_len("xnlm", "m", input.xnlm.shape()[0], angular_count)?;
        ensure_axis_len("xnlm", "l", input.xnlm.shape()[1], angular_count)?;
        ensure_axis_len(
            "first_leg_polynomials",
            "l",
            input.first_leg_polynomials.shape()[0],
            angular_count,
        )?;
        ensure_axis_len(
            "second_leg_polynomials",
            "l",
            input.second_leg_polynomials.shape()[0],
            angular_count,
        )?;
    }

    Ok(())
}

fn validate_energy_independent_matrix_input(
    input: EnergyIndependentMatrixInput<'_>,
) -> Result<(), GenfmtError> {
    validate_positive_limit(
        "magnetic_limit",
        checked_count("magnetic_limit", input.magnetic_limit)?,
    )?;
    validate_positive_limit(
        "rotation_magnetic_offset",
        checked_count("rotation_magnetic_offset", input.rotation_magnetic_offset)?,
    )?;

    let transition_count = input.transition_angular_momenta.len();
    ensure_axis_len(
        "transition_b_matrix",
        "transition1",
        input.transition_b_matrix.shape()[2],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "transition2",
        input.transition_b_matrix.shape()[5],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "spin1",
        input.transition_b_matrix.shape()[1],
        input.spin_index + 1,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "spin2",
        input.transition_b_matrix.shape()[4],
        input.spin_index + 1,
    )?;
    let transition_magnetic_required = input
        .transition_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "transition_magnetic_offset",
            value: input.transition_magnetic_offset,
        })?;
    ensure_axis_len(
        "transition_b_matrix",
        "m1",
        input.transition_b_matrix.shape()[0],
        transition_magnetic_required,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "m2",
        input.transition_b_matrix.shape()[3],
        transition_magnetic_required,
    )?;

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    if let Some((_, max_l)) = active_transition_limits(&transition_l) {
        let angular_count = checked_count("lind", max_l)?;
        match input.rotations {
            TransitionRotationInput::Polarized {
                first_rotation,
                last_rotation,
                first_eta,
                last_eta,
            } => {
                validate_finite_scalar("first_eta", first_eta)?;
                validate_finite_scalar("last_eta", last_eta)?;
                validate_rotation_table(
                    "first_rotation",
                    first_rotation,
                    input.rotation_magnetic_offset,
                    angular_count,
                )?;
                validate_rotation_table(
                    "last_rotation",
                    last_rotation,
                    input.rotation_magnetic_offset,
                    angular_count,
                )?;
            }
            TransitionRotationInput::Unpolarized { combined_rotation } => {
                validate_rotation_table(
                    "combined_rotation",
                    combined_rotation,
                    input.rotation_magnetic_offset,
                    angular_count,
                )?;
            }
        }
    }

    Ok(())
}

fn validate_rotation_table(
    name: &'static str,
    rotation: ArrayView3<'_, Real>,
    offset: usize,
    angular_count: usize,
) -> Result<(), GenfmtError> {
    ensure_axis_len(name, "l", rotation.shape()[0], angular_count)?;
    let magnetic_required = offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: offset,
        })?;
    ensure_axis_len(name, "m1", rotation.shape()[1], magnetic_required)?;
    ensure_axis_len(name, "m2", rotation.shape()[2], magnetic_required)?;
    Ok(())
}

fn validate_lambda_count(
    name: &'static str,
    requested: usize,
    available: usize,
) -> Result<(), GenfmtError> {
    if requested <= available {
        Ok(())
    } else {
        Err(GenfmtError::LambdaCountOutOfRange {
            name,
            requested,
            available,
        })
    }
}

fn ensure_axis_len(
    table: &'static str,
    axis: &'static str,
    length: usize,
    required: usize,
) -> Result<(), GenfmtError> {
    if length >= required {
        Ok(())
    } else {
        Err(GenfmtError::TableAxisTooShort {
            table,
            axis,
            length,
            required,
        })
    }
}

fn lambda_n_limit(n_indices: ArrayView1<'_, i32>, count: usize) -> Result<usize, GenfmtError> {
    let mut max_n = 0;
    for index in 0..count {
        max_n = max_n.max(lambda_order(n_indices[index], index)?);
    }
    Ok(max_n)
}

fn transition_angular_momenta(
    transition_l: ArrayView1<'_, i32>,
) -> Result<Vec<Option<usize>>, GenfmtError> {
    transition_l
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value < 0 {
                Ok(None)
            } else {
                let angular_momentum =
                    usize::try_from(value).map_err(|_| GenfmtError::InvalidLambdaIndex {
                        index,
                        field: "lind",
                        value,
                    })?;
                Ok(Some(angular_momentum))
            }
        })
        .collect()
}

fn active_transition_limits(transition_l: &[Option<usize>]) -> Option<(usize, usize)> {
    let mut limits: Option<(usize, usize)> = None;
    for &angular_momentum in transition_l.iter().flatten() {
        limits = Some(match limits {
            Some((minimum, maximum)) => {
                (minimum.min(angular_momentum), maximum.max(angular_momentum))
            }
            None => (angular_momentum, angular_momentum),
        });
    }
    limits
}

fn lambda_order(value: i32, index: usize) -> Result<usize, GenfmtError> {
    usize::try_from(value).map_err(|_| GenfmtError::InvalidLambdaIndex {
        index,
        field: "nlam",
        value,
    })
}

fn lambda_abs_magnetic(value: i32, index: usize) -> Result<usize, GenfmtError> {
    usize::try_from(value.unsigned_abs()).map_err(|_| GenfmtError::InvalidLambdaIndex {
        index,
        field: "mlam",
        value,
    })
}

fn averaged_t_matrix(
    phase_shifts: ArrayView1<'_, Complex>,
    phase_offset: usize,
    angular_momentum: usize,
) -> Result<Complex, GenfmtError> {
    let negative = complex_vector_entry(
        phase_shifts,
        "phase_shifts",
        phase_offset - angular_momentum,
    )?;
    let positive = complex_vector_entry(
        phase_shifts,
        "phase_shifts",
        phase_offset + angular_momentum,
    )?;
    let imaginary = Complex::new(0.0, 1.0);
    let negative_t =
        ((2.0 * imaginary * negative).exp() - Complex::new(1.0, 0.0)) / (2.0 * imaginary);
    let positive_t =
        ((2.0 * imaginary * positive).exp() - Complex::new(1.0, 0.0)) / (2.0 * imaginary);
    Ok(negative_t * (angular_momentum as Real + 1.0) + positive_t * angular_momentum as Real)
}

fn xnlm_entry(
    xnlm: ArrayView2<'_, Real>,
    magnetic: usize,
    angular_momentum: usize,
) -> Result<Real, GenfmtError> {
    let value = real_entry(xnlm, "xnlm", magnetic, angular_momentum)?;
    if value == 0.0 {
        Err(GenfmtError::ZeroLegendreNormalization {
            angular_momentum,
            magnetic,
        })
    } else {
        Ok(value)
    }
}

fn rotation_entry(
    rotation: ArrayView3<'_, Real>,
    offset: usize,
    angular_momentum: usize,
    first_magnetic: i32,
    second_magnetic: i32,
) -> Result<Real, GenfmtError> {
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "rotation_magnetic_offset",
        "rotation",
        "m1",
        rotation.shape()[1],
    )?;
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "rotation_magnetic_offset",
        "rotation",
        "m2",
        rotation.shape()[2],
    )?;
    let value = rotation[(angular_momentum, first, second)];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableScalar {
            table: "rotation",
            row: angular_momentum,
            column: first,
            value,
        })
    }
}

fn transition_matrix_entry(
    transition_matrix: ArrayView4<'_, Complex>,
    offset: usize,
    first_magnetic: i32,
    first_transition: usize,
    second_magnetic: i32,
    second_transition: usize,
) -> Result<Complex, GenfmtError> {
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_matrix",
        "m1",
        transition_matrix.shape()[0],
    )?;
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_matrix",
        "m2",
        transition_matrix.shape()[2],
    )?;
    complex4_entry(
        transition_matrix,
        "transition_matrix",
        first,
        first_transition,
        second,
        second_transition,
    )
}

fn transition_b_matrix_entry(
    transition_b_matrix: ArrayView6<'_, Complex>,
    offset: usize,
    first_magnetic: i32,
    spin_index: usize,
    first_transition: usize,
    second_magnetic: i32,
    second_transition: usize,
) -> Result<Complex, GenfmtError> {
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_b_matrix",
        "m1",
        transition_b_matrix.shape()[0],
    )?;
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_b_matrix",
        "m2",
        transition_b_matrix.shape()[3],
    )?;
    complex6_entry(
        transition_b_matrix,
        "transition_b_matrix",
        [
            first,
            spin_index,
            first_transition,
            second,
            spin_index,
            second_transition,
        ],
    )
}

fn signed_magnetic_index(
    value: i32,
    offset: usize,
    offset_name: &'static str,
    table: &'static str,
    axis: &'static str,
    length: usize,
) -> Result<usize, GenfmtError> {
    let magnitude =
        usize::try_from(value.unsigned_abs()).map_err(|_| GenfmtError::InvalidLambdaIndex {
            index: 0,
            field: "mlam",
            value,
        })?;
    let required = offset
        .checked_add(magnitude)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: offset_name,
            value: offset,
        })?;
    let index = if value < 0 {
        offset.checked_sub(magnitude)
    } else {
        offset.checked_add(magnitude)
    }
    .ok_or(GenfmtError::TableAxisTooShort {
        table,
        axis,
        length,
        required,
    })?;
    ensure_axis_len(table, axis, length, index + 1)?;
    Ok(index)
}

fn complex_vector_entry(
    vector: ArrayView1<'_, Complex>,
    table: &'static str,
    index: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(table, "index", vector.len(), index + 1)?;
    let value = vector[index];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table,
            row: index,
            column: 0,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex_entry(
    table: ArrayView2<'_, Complex>,
    name: &'static str,
    row: usize,
    column: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(name, "row", table.shape()[0], row + 1)?;
    ensure_axis_len(name, "column", table.shape()[1], column + 1)?;
    let value = table[(row, column)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table: name,
            row,
            column,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex4_entry(
    table: ArrayView4<'_, Complex>,
    name: &'static str,
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(name, "axis0", table.shape()[0], i0 + 1)?;
    ensure_axis_len(name, "axis1", table.shape()[1], i1 + 1)?;
    ensure_axis_len(name, "axis2", table.shape()[2], i2 + 1)?;
    ensure_axis_len(name, "axis3", table.shape()[3], i3 + 1)?;
    let value = table[(i0, i1, i2, i3)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensorComplex {
            table: name,
            i0,
            i1,
            i2,
            i3,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex6_entry(
    table: ArrayView6<'_, Complex>,
    name: &'static str,
    index: [usize; 6],
) -> Result<Complex, GenfmtError> {
    let [i0, i1, i2, i3, i4, i5] = index;
    ensure_axis_len(name, "axis0", table.shape()[0], i0 + 1)?;
    ensure_axis_len(name, "axis1", table.shape()[1], i1 + 1)?;
    ensure_axis_len(name, "axis2", table.shape()[2], i2 + 1)?;
    ensure_axis_len(name, "axis3", table.shape()[3], i3 + 1)?;
    ensure_axis_len(name, "axis4", table.shape()[4], i4 + 1)?;
    ensure_axis_len(name, "axis5", table.shape()[5], i5 + 1)?;
    let value = table[(i0, i1, i2, i3, i4, i5)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensor6Complex {
            table: name,
            i0,
            i1,
            i2,
            i3,
            i4,
            i5,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn real_entry(
    table: ArrayView2<'_, Real>,
    name: &'static str,
    row: usize,
    column: usize,
) -> Result<Real, GenfmtError> {
    ensure_axis_len(name, "row", table.shape()[0], row + 1)?;
    ensure_axis_len(name, "column", table.shape()[1], column + 1)?;
    let value = table[(row, column)];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableScalar {
            table: name,
            row,
            column,
            value,
        })
    }
}

fn fill_initial_state_rotation_work(
    lmaxp1: usize,
    mmaxp1: usize,
    beta: Real,
    work: &mut Array3<Real>,
) {
    let ndm = lmaxp1 + mmaxp1 - 1;
    let half_beta = beta / 2.0;
    let xc = half_beta.cos();
    let xs = half_beta.sin();
    let s = beta.sin();

    work[(1, 1, 1)] = 1.0;
    work[(2, 1, 1)] = xc * xc;
    work[(2, 1, 2)] = s / 2.0_f64.sqrt();
    work[(2, 1, 3)] = xs * xs;
    work[(2, 2, 1)] = -work[(2, 1, 2)];
    work[(2, 2, 2)] = beta.cos();
    work[(2, 2, 3)] = work[(2, 1, 2)];
    work[(2, 3, 1)] = work[(2, 1, 3)];
    work[(2, 3, 2)] = -work[(2, 2, 3)];
    work[(2, 3, 3)] = work[(2, 1, 1)];

    for l in 3..=lmaxp1 {
        let ln = (2 * l - 1).min(ndm);
        let lm = (2 * l - 3).min(ndm);
        for n in 1..=ln {
            for m in 1..=lm {
                let l_signed = l as isize;
                let n_signed = n as isize;
                let m_signed = m as isize;
                let t1 = ((2 * l_signed - 1 - n_signed) * (2 * l_signed - 2 - n_signed)) as Real;
                let t = ((2 * l_signed - 1 - m_signed) * (2 * l_signed - 2 - m_signed)) as Real;
                let f1 = (t1 / t).sqrt();
                let f2 = (((2 * l_signed - 1 - n_signed) * (n_signed - 1)) as Real / t).sqrt();
                let f3 = if n > 2 {
                    (((n - 2) * (n - 1)) as Real / t).sqrt()
                } else {
                    0.0
                };

                let mut dlnm = f1 * xc * xc * work[(l - 1, n, m)];
                if n > 1 {
                    dlnm -= f2 * s * work[(l - 1, n - 1, m)];
                }
                if n > 2 {
                    dlnm += f3 * xs * xs * work[(l - 1, n - 2, m)];
                }
                work[(l, n, m)] = dlnm;

                if n > 2 * l - 3 {
                    work[(l, m, n)] = alternating_sign(n - m) * dlnm;
                }
            }

            if n > 2 * l - 3 {
                work[(l, 2 * l - 2, 2 * l - 2)] = work[(l, 2, 2)];
                work[(l, 2 * l - 1, 2 * l - 2)] = -work[(l, 1, 2)];
                work[(l, 2 * l - 2, 2 * l - 1)] = -work[(l, 2, 1)];
                work[(l, 2 * l - 1, 2 * l - 1)] = work[(l, 1, 1)];
            }
        }
    }
}

fn copy_initial_state_rotation(
    lmaxp1: usize,
    mmaxp1: usize,
    magnetic_offset: usize,
    work: &Array3<Real>,
    matrix: &mut Array3<Real>,
) -> Result<(), GenfmtError> {
    let magnetic_offset =
        isize::try_from(magnetic_offset).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
        })?;

    for il in 1..=lmaxp1 {
        let mx = (il - 1).min(mmaxp1 - 1);
        let mx_signed = isize::try_from(mx).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
        })?;
        let il_signed = isize::try_from(il).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: lmaxp1,
        })?;

        for m1_slot in 0..=(2 * mx) {
            let m1 = isize::try_from(m1_slot).map_err(|_| GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: mmaxp1,
            })? - mx_signed;
            for m2_slot in 0..=(2 * mx) {
                let m2 =
                    isize::try_from(m2_slot).map_err(|_| GenfmtError::InvalidAngularLimit {
                        name: "mmaxp1",
                        value: mmaxp1,
                    })? - mx_signed;
                let row = shifted_index(m1, magnetic_offset, "mmaxp1", mmaxp1)?;
                let column = shifted_index(m2, magnetic_offset, "mmaxp1", mmaxp1)?;
                let work_row = shifted_index(m1, il_signed, "lmaxp1", lmaxp1)?;
                let work_column = shifted_index(m2, il_signed, "lmaxp1", lmaxp1)?;
                matrix[(il - 1, row, column)] = work[(il, work_row, work_column)];
            }
        }
    }
    Ok(())
}

fn shifted_index(
    value: isize,
    offset: isize,
    name: &'static str,
    limit: usize,
) -> Result<usize, GenfmtError> {
    let index = value
        .checked_add(offset)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value: limit })?;
    usize::try_from(index).map_err(|_| GenfmtError::InvalidAngularLimit { name, value: limit })
}

#[derive(Debug, Clone, Copy)]
struct RdpathTrig {
    ct: Real,
    st: Real,
    cp: Real,
    sp: Real,
}

fn rdpath_trig(vector: [Real; 3]) -> RdpathTrig {
    let [x, y, z] = vector;
    let rxy = x.hypot(y);
    let r = rxy.hypot(z);
    let (ct, st) = if r < RDPATH_EPSILON {
        (1.0, 0.0)
    } else {
        (z / r, rxy / r)
    };
    let (cp, sp) = if rxy < RDPATH_EPSILON {
        (if ct < 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        (x / rxy, y / rxy)
    };

    RdpathTrig { ct, st, cp, sp }
}

fn rdpath_arg(value: Complex, fallback: Real) -> Real {
    let real = if value.re.abs() < RDPATH_EPSILON {
        0.0
    } else {
        value.re
    };
    let imaginary = if value.im.abs() < RDPATH_EPSILON {
        0.0
    } else {
        value.im
    };

    if real == 0.0 && imaginary == 0.0 {
        fallback
    } else {
        imaginary.atan2(real)
    }
}

fn bounded_beta_cosine(value: Real) -> Result<Real, GenfmtError> {
    if !value.is_finite() {
        return Err(GenfmtError::NonFiniteScalar {
            field: "beta_cosine",
            value,
        });
    }
    if value < -1.0 {
        Ok(-1.0)
    } else if value > 1.0 {
        Ok(1.0)
    } else {
        Ok(value)
    }
}

fn vector_difference(end: [Real; 3], start: [Real; 3]) -> [Real; 3] {
    [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
}

fn point_distance(left: [Real; 3], right: [Real; 3]) -> Real {
    (left[0] - right[0])
        .hypot(left[1] - right[1])
        .hypot(left[2] - right[2])
}

fn alternating_sign(power: usize) -> Real {
    if power.is_multiple_of(2) { 1.0 } else { -1.0 }
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), GenfmtError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GenfmtError::NonFiniteScalar { field, value })
    }
}

fn validate_finite_complex(field: &'static str, value: Complex) -> Result<(), GenfmtError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(GenfmtError::NonFiniteComplex {
            field,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn normalized_dot(
    left_field: &'static str,
    left: [Real; 3],
    right_field: &'static str,
    right: [Real; 3],
) -> Result<Real, GenfmtError> {
    validate_vector(left_field, left)?;
    validate_vector(right_field, right)?;

    let dot = left.iter().zip(right).map(|(&a, b)| a * b).sum::<Real>();
    let left_norm = left.iter().map(|value| value * value).sum::<Real>();
    let right_norm = right.iter().map(|value| value * value).sum::<Real>();

    if left_norm == 0.0 {
        return Err(GenfmtError::ZeroVector { field: left_field });
    }
    if right_norm == 0.0 {
        return Err(GenfmtError::ZeroVector { field: right_field });
    }

    Ok(dot / (left_norm * right_norm).sqrt())
}

fn validate_vector(field: &'static str, vector: [Real; 3]) -> Result<(), GenfmtError> {
    for (index, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn ystar(initial_l: usize, x: Real, y: Real, z: Real) -> Real {
    const LEGENDRE: [[Real; 5]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.5, 0.0, 1.5, 0.0, 0.0],
        [0.0, -1.5, 0.0, 2.5, 0.0],
        [0.375, 0.0, -3.75, 0.0, 4.375],
    ];
    let coefficients = LEGENDRE[initial_l];
    let l = initial_l as Real;

    let pln0 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .map(|(power, coefficient)| coefficient * x.powi(power as i32))
        .sum::<Real>();
    let pln1 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .skip(1)
        .map(|(power, coefficient)| {
            let power_real = power as Real;
            coefficient * power_real * x.powi(power as i32 - 1)
        })
        .sum::<Real>();
    let pln2 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .skip(2)
        .map(|(power, coefficient)| {
            let power_real = power as Real;
            coefficient * power_real * (power_real - 1.0) * x.powi(power as i32 - 2)
        })
        .sum::<Real>();

    let ytemp = -l * pln0 + pln1 * (x + y * z) - pln2 * (y * y + z * z - 2.0 * x * y * z);
    ytemp * 3.0 / l / (4.0 * l * l - 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        CurvedWavePolynomialInput, EnergyIndependentMatrixInput, GenfmtError, InitialStateRotation,
        InitialStateRotationInput, LambdaIndexInput, PathRotationInput,
        PolarizedScatteringAmplitudeInput, ScatteringAmplitudeMatrixInput, TransitionRotationInput,
        XStarInput, curved_wave_polynomials, energy_independent_transition_matrix,
        initial_state_rotation, lambda_indices, path_rotation_angles,
        polarized_scattering_amplitude_matrix, scattering_amplitude_matrix, xstar,
    };
    use crate::{Complex, Real, legendre_normalization_table};
    use ndarray::{Array1, Array2, Array3, Array4, Array6, ShapeBuilder, arr2};

    fn input<'a>(
        calculation: i32,
        energy_index: usize,
        scattering_count: usize,
        initial_l: usize,
        beta_angles: &'a [f64],
        lambda_capacity: usize,
    ) -> LambdaIndexInput<'a> {
        LambdaIndexInput {
            calculation,
            energy_index,
            scattering_count,
            initial_l,
            beta_angles,
            lambda_capacity,
            max_m: 10,
            max_n: 10,
        }
    }

    #[test]
    fn exact_order_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [0.0, std::f64::consts::PI, 0.5, 2.8];
        let lambda = lambda_indices(input(2, 10, 2, 3, &beta, 40))?;

        assert_eq!(lambda.order, 2);
        assert_eq!(lambda.requested_n_max, 1);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 6);
        assert_eq!(lambda.max_n, 1);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert!(!lambda.truncated);
        assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2, 0]);
        assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0, 1]);
        Ok(())
    }

    #[test]
    fn single_scattering_uses_initial_l_exact_reference() -> Result<(), GenfmtError> {
        let beta = [0.3, 1.2];
        let lambda = lambda_indices(input(10, 8, 1, 2, &beta, 40))?;

        assert_eq!(lambda.order, 6);
        assert_eq!(lambda.requested_n_max, 2);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 15);
        assert_eq!(lambda.max_n, 2);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(
            lambda.m_indices.to_vec(),
            vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2]
        );
        assert_eq!(
            lambda.n_indices.to_vec(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        );
        Ok(())
    }

    #[test]
    fn cute_linear_low_energy_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [
            0.0,
            std::f64::consts::PI,
            0.010,
            std::f64::consts::PI - 0.010,
        ];
        let lambda = lambda_indices(input(10, 41, 2, 4, &beta, 80))?;

        assert_eq!(lambda.order, 12);
        assert_eq!(lambda.requested_n_max, 4);
        assert_eq!(lambda.requested_m_max, 4);
        assert_eq!(lambda.initial_l_prefix_len, 45);
        assert_eq!(lambda.max_n, 4);
        assert_eq!(lambda.max_m_plus_one, 5);
        assert_eq!(lambda.m_indices.len(), 45);
        assert_eq!(
            &lambda.m_indices.to_vec()[..9],
            &[0, -1, 1, -2, 2, -3, 3, -4, 4]
        );
        assert_eq!(
            &lambda.n_indices.to_vec()[36..],
            &[4, 4, 4, 4, 4, 4, 4, 4, 4]
        );
        Ok(())
    }

    #[test]
    fn cute_nonlinear_high_energy_sorts_initial_l_prefix() -> Result<(), GenfmtError> {
        let beta = [0.0, 0.25, std::f64::consts::PI];
        let lambda = lambda_indices(input(10, 42, 2, 4, &beta, 80))?;

        assert_eq!(lambda.order, 21);
        assert_eq!(lambda.requested_n_max, 9);
        assert_eq!(lambda.requested_m_max, 3);
        assert_eq!(lambda.m_indices.len(), 70);
        assert_eq!(lambda.initial_l_prefix_len, 35);
        assert_eq!(lambda.max_n, 9);
        assert_eq!(lambda.max_m_plus_one, 4);
        assert_eq!(&lambda.n_indices.to_vec()[..7], &[0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&lambda.n_indices.to_vec()[28..35], &[4, 4, 4, 4, 4, 4, 4]);
        assert_eq!(&lambda.n_indices.to_vec()[35..42], &[5, 5, 5, 5, 5, 5, 5]);
        assert_eq!(&lambda.n_indices.to_vec()[63..], &[9, 9, 9, 9, 9, 9, 9]);
        Ok(())
    }

    #[test]
    fn negative_calculation_decodes_requested_limits() -> Result<(), GenfmtError> {
        let beta = [0.0, 0.5];
        let lambda = lambda_indices(input(-80_205, 12, 2, 2, &beta, 80))?;

        assert_eq!(lambda.order, 7);
        assert_eq!(lambda.requested_n_max, 5);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 15);
        assert_eq!(lambda.max_n, 3);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(
            lambda.m_indices.to_vec(),
            vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1]
        );
        assert_eq!(
            lambda.n_indices.to_vec(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
        );
        Ok(())
    }

    #[test]
    fn capacity_truncation_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [0.0, 1.0];
        let lambda = lambda_indices(input(4, 10, 2, 1, &beta, 5))?;

        assert!(lambda.truncated);
        assert_eq!(lambda.order, 4);
        assert_eq!(lambda.requested_n_max, 2);
        assert_eq!(lambda.requested_m_max, 4);
        assert_eq!(lambda.initial_l_prefix_len, 3);
        assert_eq!(lambda.max_n, 0);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2]);
        assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn cute_calculation_rejects_nonfinite_beta() {
        let beta = [f64::NAN];

        assert!(matches!(
            lambda_indices(input(10, 42, 2, 4, &beta, 80)),
            Err(GenfmtError::NonFiniteBetaAngle { index: 0, .. })
        ));
    }

    #[test]
    fn undefined_calculation_is_an_error_for_multiple_scattering() {
        assert_eq!(
            lambda_indices(input(11, 1, 2, 0, &[], 10)),
            Err(GenfmtError::UndefinedLambdaCalculation { calculation: 11 })
        );
    }

    #[test]
    fn dimension_overflow_is_reported() {
        let mut bad = input(10, 42, 2, 4, &[0.25], 80);
        bad.max_n = 8;

        assert!(matches!(
            lambda_indices(bad),
            Err(GenfmtError::DimensionExceeded {
                max_n: 9,
                max_n_limit: 8,
                ..
            })
        ));
    }

    #[test]
    fn initial_state_rotation_matches_feff_full_reference() -> Result<(), GenfmtError> {
        let rotation = initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 4,
            mmaxp1: 4,
            beta_angle: 0.7,
        })?;

        assert_eq!(rotation.matrix.shape(), &[4, 7, 7]);
        assert_eq!(rotation.matrix.strides(), &[1, 4, 28]);
        assert_eq!(rotation.magnetic_offset, 3);
        assert_close(rotation_sum(&rotation), 14.508_147_433_950_487);
        assert_eq!(rotation_nonzero_count(&rotation), 84);
        assert_close(rotation_value(&rotation, 1, 0, 0), 1.0);
        assert_close(
            rotation_value(&rotation, 2, -1, -1),
            0.882_421_093_642_244_2,
        );
        assert_close(
            rotation_value(&rotation, 2, -1, 0),
            0.455_530_695_206_085_63,
        );
        assert_close(rotation_value(&rotation, 2, 0, 1), 0.455_530_695_206_085_63);
        assert_close(
            rotation_value(&rotation, 3, -2, 1),
            0.075_746_411_121_730_47,
        );
        assert_close(
            rotation_value(&rotation, 4, -3, 3),
            0.001_625_504_772_936_771_3,
        );
        assert_close(
            rotation_value(&rotation, 4, 0, 0),
            -0.028_712_995_143_227_615,
        );
        Ok(())
    }

    #[test]
    fn initial_state_rotation_matches_feff_limited_m_reference() -> Result<(), GenfmtError> {
        let rotation = initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 5,
            mmaxp1: 2,
            beta_angle: -0.4,
        })?;

        assert_eq!(rotation.matrix.shape(), &[5, 3, 3]);
        assert_eq!(rotation.matrix.strides(), &[1, 5, 15]);
        assert_eq!(rotation.magnetic_offset, 1);
        assert_close(rotation_sum(&rotation), 10.424_101_881_334_796);
        assert_eq!(rotation_nonzero_count(&rotation), 37);
        assert_close(rotation_value(&rotation, 1, 0, 0), 1.0);
        assert_close(
            rotation_value(&rotation, 2, -1, -1),
            0.960_530_497_001_442_6,
        );
        assert_close(rotation_value(&rotation, 2, -1, 0), -0.275_360_350_564_871);
        assert_close(rotation_value(&rotation, 2, 0, 1), -0.275_360_350_564_871);
        assert_close(
            rotation_value(&rotation, 3, -1, 1),
            0.112_177_142_327_859_86,
        );
        assert_close(rotation_value(&rotation, 5, -1, 1), 0.307_544_785_027_699_8);
        assert_close(rotation_value(&rotation, 5, 0, 0), 0.342_377_357_912_471_87);
        Ok(())
    }

    #[test]
    fn initial_state_rotation_rejects_invalid_inputs() {
        assert_eq!(
            initial_state_rotation(InitialStateRotationInput {
                lmaxp1: 0,
                mmaxp1: 1,
                beta_angle: 0.0,
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            initial_state_rotation(InitialStateRotationInput {
                lmaxp1: 1,
                mmaxp1: 0,
                beta_angle: 0.0,
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            initial_state_rotation(InitialStateRotationInput {
                lmaxp1: 1,
                mmaxp1: 1,
                beta_angle: f64::NAN,
            }),
            Err(GenfmtError::NonFiniteRotationAngle)
        );
    }

    #[test]
    fn path_rotation_angles_match_polarized_rdpath_reference() -> Result<(), GenfmtError> {
        let positions = arr2(&[
            [1.2, -0.4, 0.7],
            [-0.3, 1.1, 1.5],
            [0.5, 0.2, -0.6],
            [0.0, 0.0, 0.0],
        ]);
        let angles = path_rotation_angles(PathRotationInput {
            positions: positions.view(),
            polarized: true,
        })?;

        assert_array_close(
            &angles.beta_angles,
            &[
                2.166_858_401_769_925_3,
                2.450_803_939_009_357,
                2.431_538_373_717_806,
                0.731_447_381_254_918_5,
                1.065_347_578_436_332_9,
            ],
        );
        assert_array_close(
            &angles.eta_values,
            &[
                3.463_343_207_986_435_3,
                3.671_719_781_241_285,
                6.729_824_761_627_887,
                11.178_806_101_438_672,
                0.800_671_291_800_303_8,
                3.522_099_030_702_158,
            ],
        );
        assert_array_close(
            &angles.leg_lengths,
            &[
                1.445_683_229_480_096,
                2.267_156_809_750_926_7,
                2.420_743_687_382_041,
                0.806_225_774_829_855,
            ],
        );
        Ok(())
    }

    #[test]
    fn path_rotation_angles_match_unpolarized_rdpath_reference() -> Result<(), GenfmtError> {
        let positions = arr2(&[[-0.2, 0.8, -1.0], [1.4, -0.5, 0.3], [0.0, 0.0, 0.0]]);
        let angles = path_rotation_angles(PathRotationInput {
            positions: positions.view(),
            polarized: false,
        })?;

        assert_array_close(
            &angles.beta_angles,
            &[
                2.571_854_110_984_37,
                2.662_458_542_799_463,
                1.048_872_653_395_752_4,
            ],
        );
        assert_array_close(
            &angles.eta_values,
            &[
                0.0,
                std::f64::consts::TAU,
                6.283_185_307_179_585,
                std::f64::consts::TAU,
                0.0,
            ],
        );
        assert_array_close(
            &angles.leg_lengths,
            &[
                1.296_148_139_681_572_2,
                2.437_211_521_390_788_3,
                1.516_575_088_810_31,
            ],
        );
        Ok(())
    }

    #[test]
    fn path_rotation_angles_rejects_invalid_inputs() {
        let empty = Array2::<Real>::zeros((0, 3));
        assert_eq!(
            path_rotation_angles(PathRotationInput {
                positions: empty.view(),
                polarized: false,
            }),
            Err(GenfmtError::EmptyPath)
        );

        let bad_columns = Array2::<Real>::zeros((1, 2));
        assert_eq!(
            path_rotation_angles(PathRotationInput {
                positions: bad_columns.view(),
                polarized: false,
            }),
            Err(GenfmtError::InvalidPathCoordinateColumns { columns: 2 })
        );

        let nonfinite = arr2(&[[0.0, f64::NAN, 0.0]]);
        assert!(matches!(
            path_rotation_angles(PathRotationInput {
                positions: nonfinite.view(),
                polarized: false,
            }),
            Err(GenfmtError::NonFinitePathCoordinate {
                leg_index: 0,
                component: 1,
                ..
            })
        ));
    }

    #[test]
    fn curved_wave_polynomials_match_feff_sclmz_reference() -> Result<(), GenfmtError> {
        let table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 4,
            mmaxp1: 4,
            rho: Complex::new(1.25, 0.4),
        })?;

        assert_eq!(table.shape(), &[5, 4]);
        assert_eq!(table.strides(), &[1, 5]);
        assert_eq!(complex_nonzero_count(&table), 11);
        assert_complex_close(table[(0, 0)], Complex::new(1.0, 0.0));
        assert_complex_close(
            table[(1, 0)],
            Complex::new(1.232_220_609_579_100_2, 0.725_689_404_934_687_9),
        );
        assert_complex_close(
            table[(2, 0)],
            Complex::new(0.278_565_725_973_782_6, 3.188_188_430_678_23),
        );
        assert_complex_close(
            table[(3, 1)],
            Complex::new(-28.733_692_908_170_283, 2.550_923_127_350_68),
        );
        assert_complex_close(table[(4, 2)], Complex::new(0.0, 0.0));
        assert_complex_close(
            complex_sum(&table),
            Complex::new(-58.983_990_231_020_26, -154.618_863_530_600_9),
        );
        Ok(())
    }

    #[test]
    fn curved_wave_polynomials_match_limited_m_reference() -> Result<(), GenfmtError> {
        let table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 5,
            mmaxp1: 3,
            rho: Complex::new(-0.8, 1.1),
        })?;

        assert_eq!(table.shape(), &[6, 3]);
        assert_eq!(table.strides(), &[1, 6]);
        assert_eq!(complex_nonzero_count(&table), 12);
        assert_complex_close(
            table[(1, 0)],
            Complex::new(1.594_594_594_594_594_5, -0.432_432_432_432_432_35),
        );
        assert_complex_close(
            table[(2, 0)],
            Complex::new(3.283_418_553_688_824, -2.840_029_218_407_596),
        );
        assert_complex_close(
            table[(3, 1)],
            Complex::new(3.013_207_509_920_446_7, -35.022_288_906_876_184),
        );
        assert_complex_close(
            table[(4, 2)],
            Complex::new(-180.487_514_146_329_86, -250.055_955_704_979_3),
        );
        assert_complex_close(
            complex_sum(&table),
            Complex::new(-306.259_756_232_255_1, -662.066_424_389_366_5),
        );
        Ok(())
    }

    #[test]
    fn curved_wave_polynomials_retain_requested_zero_columns() -> Result<(), GenfmtError> {
        let table = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 2,
            mmaxp1: 4,
            rho: Complex::new(1.0, 0.25),
        })?;

        assert_eq!(table.shape(), &[3, 4]);
        assert!(
            table
                .column(2)
                .iter()
                .all(|&value| value == Complex::new(0.0, 0.0))
        );
        assert!(
            table
                .column(3)
                .iter()
                .all(|&value| value == Complex::new(0.0, 0.0))
        );
        Ok(())
    }

    #[test]
    fn curved_wave_polynomials_reject_invalid_inputs() {
        assert_eq!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 0,
                mmaxp1: 1,
                rho: Complex::new(1.0, 0.0),
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 1,
                mmaxp1: 0,
                rho: Complex::new(1.0, 0.0),
            }),
            Err(GenfmtError::InvalidAngularLimit {
                name: "mmaxp1",
                value: 0,
            })
        );
        assert_eq!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 1,
                mmaxp1: 1,
                rho: Complex::new(0.0, 0.0),
            }),
            Err(GenfmtError::ZeroComplex { field: "rho" })
        );
        assert!(matches!(
            curved_wave_polynomials(CurvedWavePolynomialInput {
                lmaxp1: 1,
                mmaxp1: 1,
                rho: Complex::new(f64::NAN, 0.0),
            }),
            Err(GenfmtError::NonFiniteComplex { field: "rho", .. })
        ));
    }

    #[test]
    fn scattering_amplitude_matrix_matches_feff_fmtrxi_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        let data = fmtrxi_reference_data()?;
        let matrix = scattering_amplitude_matrix(data.input())?;

        assert_eq!(matrix.shape(), &[6, 5]);
        assert_eq!(matrix.strides(), &[1, 6]);
        assert_complex_close(
            matrix[(0, 0)],
            Complex::new(-38.563_289_559_671_01, 28.084_721_411_987_896),
        );
        assert_complex_close(
            matrix[(0, 1)],
            Complex::new(-129.565_304_116_042_23, 92.125_635_892_089_4),
        );
        assert_complex_close(
            matrix[(1, 2)],
            Complex::new(122.713_265_094_310_16, 21.039_927_424_360_677),
        );
        assert_complex_close(
            matrix[(3, 4)],
            Complex::new(-63.332_044_984_118_596, -84.365_936_676_961_67),
        );
        assert_complex_close(
            matrix[(5, 4)],
            Complex::new(-1_309.182_568_320_504, 255.082_893_344_668_2),
        );
        assert_complex_close(
            complex_sum(&matrix),
            Complex::new(-3_078.729_163_920_782_4, 1_027.554_784_760_136),
        );
        Ok(())
    }

    #[test]
    fn scattering_amplitude_matrix_rejects_invalid_inputs() -> Result<(), Box<dyn std::error::Error>>
    {
        let data = fmtrxi_reference_data()?;
        assert!(matches!(
            scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
                left_lambda_count: 9,
                ..data.input()
            }),
            Err(GenfmtError::LambdaCountOutOfRange {
                name: "left_lambda_count",
                requested: 9,
                available: 8,
            })
        ));

        let bad_phase = Array1::from_vec(vec![Complex::new(0.0, 0.0); 4]);
        assert_eq!(
            scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
                phase_shifts: bad_phase.view(),
                ..data.input()
            }),
            Err(GenfmtError::InvalidSignedPhaseShape { length: 4 })
        );

        let mut nonfinite_phase = data.phase_shifts.clone();
        nonfinite_phase[4] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
                phase_shifts: nonfinite_phase.view(),
                ..data.input()
            }),
            Err(GenfmtError::NonFiniteTableComplex {
                table: "phase_shifts",
                row: 4,
                ..
            })
        ));

        let mut zero_xnlm = data.xnlm.clone();
        zero_xnlm[(1, 1)] = 0.0;
        assert_eq!(
            scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
                xnlm: zero_xnlm.view(),
                ..data.input()
            }),
            Err(GenfmtError::ZeroLegendreNormalization {
                angular_momentum: 1,
                magnetic: 1,
            })
        );

        let short_polynomials = Array2::zeros((4, 1).f());
        assert!(matches!(
            scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
                first_leg_polynomials: short_polynomials.view(),
                ..data.input()
            }),
            Err(GenfmtError::TableAxisTooShort {
                table: "first_leg_polynomials",
                axis: "column",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn polarized_scattering_amplitude_matrix_matches_feff_mmtrxi_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        let data = mmtrxi_reference_data()?;
        let matrix = polarized_scattering_amplitude_matrix(data.input())?;

        assert_eq!(matrix.shape(), &[6, 6]);
        assert_eq!(matrix.strides(), &[1, 6]);
        assert_complex_close(
            matrix[(0, 0)],
            Complex::new(-2_845.112_371_916_357, 2_888.147_341_052_974),
        );
        assert_complex_close(
            matrix[(0, 1)],
            Complex::new(-10_079.776_065_551_37, 9_413.994_100_845_948),
        );
        assert_complex_close(
            matrix[(1, 2)],
            Complex::new(8_697.313_993_828_167, -374.375_986_576_882_5),
        );
        assert_complex_close(
            matrix[(3, 4)],
            Complex::new(-4_714.438_045_315_254, -3_615.819_287_952_961_5),
        );
        assert_complex_close(
            matrix[(5, 5)],
            Complex::new(-16_490.015_276_258_873, 9_905.708_935_168_93),
        );
        assert_complex_close(
            complex_sum(&matrix),
            Complex::new(-235_884.893_264_593_76, 120_845.446_342_197_36),
        );
        Ok(())
    }

    #[test]
    fn polarized_scattering_amplitude_matrix_rejects_invalid_inputs()
    -> Result<(), Box<dyn std::error::Error>> {
        let data = mmtrxi_reference_data()?;
        assert!(matches!(
            polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
                lambda_count: 9,
                ..data.input()
            }),
            Err(GenfmtError::LambdaCountOutOfRange {
                name: "lambda_count",
                requested: 9,
                available: 8,
            })
        ));

        let mut bad_radial = data.radial_factors.clone();
        bad_radial[1] = Complex::new(0.0, f64::NAN);
        assert!(matches!(
            polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
                radial_factors: bad_radial.view(),
                ..data.input()
            }),
            Err(GenfmtError::NonFiniteTableComplex {
                table: "radial_factors",
                row: 1,
                ..
            })
        ));

        let mut bad_transition = data.transition_matrix.clone();
        bad_transition[(4, 0, 4, 0)] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
                transition_matrix: bad_transition.view(),
                ..data.input()
            }),
            Err(GenfmtError::NonFiniteTensorComplex {
                table: "transition_matrix",
                i0: 4,
                i1: 0,
                i2: 4,
                i3: 0,
                ..
            })
        ));

        let short_transition_matrix = Array4::zeros((8, 8, 9, 8).f());
        assert!(matches!(
            polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
                transition_matrix: short_transition_matrix.view(),
                ..data.input()
            }),
            Err(GenfmtError::TableAxisTooShort {
                table: "transition_matrix",
                axis: "m1",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn energy_independent_transition_matrix_matches_feff_mmtr_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        let data = mmtr_reference_data();
        let polarized = energy_independent_transition_matrix(data.polarized_input())?;

        assert_eq!(polarized.shape(), &[7, 8, 7, 8]);
        assert_eq!(polarized.strides(), &[1, 7, 56, 392]);
        assert_complex_close(
            polarized[(3, 0, 3, 0)],
            Complex::new(0.021_453_694_254_769_01, 0.071_512_314_182_563_39),
        );
        assert_complex_close(
            polarized[(2, 1, 4, 2)],
            Complex::new(0.002_111_873_685_701_496, 1.236_234_538_760_950_8),
        );
        assert_complex_close(
            polarized[(5, 3, 3, 4)],
            Complex::new(0.628_672_134_559_167_4, 1.917_320_183_093_828_2),
        );
        assert_complex_close(
            polarized[(1, 5, 1, 5)],
            Complex::new(0.581_425_567_184_014_2, 3.044_675_502_642_624),
        );
        assert_complex_close(
            active_bmati_sum(&polarized),
            Complex::new(286.229_896_462_046_5, 1_632.094_116_299_501_8),
        );

        let averaged = energy_independent_transition_matrix(data.unpolarized_input())?;
        assert_complex_close(
            averaged[(3, 0, 3, 0)],
            Complex::new(0.014_330_047_336_884_089, 0.047_766_824_456_280_305),
        );
        assert_complex_close(
            averaged[(2, 1, 4, 1)],
            Complex::new(0.028_570_007_096_571_4, 0.095_233_356_988_571_34),
        );
        assert_complex_close(
            averaged[(5, 3, 3, 3)],
            Complex::new(0.040_492_545_604_276_02, 0.134_975_152_014_253_42),
        );
        assert_complex_close(
            averaged[(1, 5, 1, 5)],
            Complex::new(0.078_103_726_170_988_49, 0.260_345_753_903_294_95),
        );
        assert_complex_close(
            active_bmati_sum(&averaged),
            Complex::new(7.154_567_773_293_091, 23.848_559_244_310_298),
        );
        Ok(())
    }

    #[test]
    fn energy_independent_transition_matrix_rejects_invalid_inputs() {
        let data = mmtr_reference_data();
        assert!(matches!(
            energy_independent_transition_matrix(EnergyIndependentMatrixInput {
                spin_index: 2,
                ..data.polarized_input()
            }),
            Err(GenfmtError::TableAxisTooShort {
                table: "transition_b_matrix",
                axis: "spin1",
                ..
            })
        ));

        let mut bad_bmat = data.transition_b_matrix.clone();
        bad_bmat[(3, 1, 0, 3, 1, 0)] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            energy_independent_transition_matrix(EnergyIndependentMatrixInput {
                transition_b_matrix: bad_bmat.view(),
                ..data.polarized_input()
            }),
            Err(GenfmtError::NonFiniteTensor6Complex {
                table: "transition_b_matrix",
                i0: 3,
                i1: 1,
                i2: 0,
                i3: 3,
                i4: 1,
                i5: 0,
                ..
            })
        ));

        let short_rotation = Array3::zeros((3, 7, 7).f());
        assert!(matches!(
            energy_independent_transition_matrix(EnergyIndependentMatrixInput {
                rotations: TransitionRotationInput::Unpolarized {
                    combined_rotation: short_rotation.view(),
                },
                ..data.unpolarized_input()
            }),
            Err(GenfmtError::TableAxisTooShort {
                table: "combined_rotation",
                axis: "l",
                ..
            })
        ));
    }

    #[test]
    fn xstar_matches_feff_linear_references() -> Result<(), GenfmtError> {
        assert_close(
            xstar(XStarInput {
                primary_polarization: [1.0, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [2.0, 0.0, 0.0],
                last_leg: [0.0, 3.0, 0.0],
                degeneracy: 3.5,
                initial_l: 1,
                ellipticity: 0.0,
            })?,
            0.0,
        );
        assert_close(
            xstar(XStarInput {
                primary_polarization: [0.2, 0.9, 0.4],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [1.0, 0.5, -0.25],
                last_leg: [0.4, -0.3, 1.2],
                degeneracy: 1.75,
                initial_l: 1,
                ellipticity: 0.0,
            })?,
            0.185_559_995_771_885_34,
        );
        Ok(())
    }

    #[test]
    fn xstar_matches_feff_elliptic_references() -> Result<(), GenfmtError> {
        assert_close(
            xstar(XStarInput {
                primary_polarization: [0.3, 1.0, -0.2],
                secondary_polarization: [-0.4, 0.2, 1.5],
                first_leg: [1.2, -0.5, 0.8],
                last_leg: [-0.7, 1.4, 0.6],
                degeneracy: 2.25,
                initial_l: 2,
                ellipticity: 0.7,
            })?,
            -0.014_836_343_260_557_886,
        );
        assert_close(
            xstar(XStarInput {
                primary_polarization: [1.0, 2.0, 3.0],
                secondary_polarization: [2.0, -1.0, 0.5],
                first_leg: [-0.25, 0.75, 1.50],
                last_leg: [1.1, -0.9, 0.4],
                degeneracy: 5.0,
                initial_l: 4,
                ellipticity: -0.35,
            })?,
            0.254_890_323_398_489_77,
        );
        Ok(())
    }

    #[test]
    fn xstar_rejects_invalid_inputs() {
        assert_eq!(
            xstar(XStarInput {
                primary_polarization: [1.0, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [1.0, 0.0, 0.0],
                last_leg: [0.0, 1.0, 0.0],
                degeneracy: 1.0,
                initial_l: 5,
                ellipticity: 0.0,
            }),
            Err(GenfmtError::InvalidInitialAngularMomentum { initial_l: 5 })
        );
        assert!(matches!(
            xstar(XStarInput {
                primary_polarization: [f64::NAN, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [1.0, 0.0, 0.0],
                last_leg: [0.0, 1.0, 0.0],
                degeneracy: 1.0,
                initial_l: 1,
                ellipticity: 0.0,
            }),
            Err(GenfmtError::NonFiniteVector {
                field: "primary_polarization",
                index: 0,
                ..
            })
        ));
        assert_eq!(
            xstar(XStarInput {
                primary_polarization: [1.0, 0.0, 0.0],
                secondary_polarization: [0.0, 1.0, 0.0],
                first_leg: [0.0, 0.0, 0.0],
                last_leg: [0.0, 1.0, 0.0],
                degeneracy: 1.0,
                initial_l: 1,
                ellipticity: 0.0,
            }),
            Err(GenfmtError::ZeroVector { field: "first_leg" })
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    fn assert_array_close(actual: &Array1<Real>, expected: &[Real]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: {actual} != {expected}"
            );
        }
    }

    struct FmtrxiReferenceData {
        m_indices: Array1<i32>,
        n_indices: Array1<i32>,
        phase_shifts: Array1<Complex>,
        first_polynomials: Array2<Complex>,
        second_polynomials: Array2<Complex>,
        rotation: Array3<Real>,
        xnlm: Array2<Real>,
    }

    impl FmtrxiReferenceData {
        fn input(&self) -> ScatteringAmplitudeMatrixInput<'_> {
            ScatteringAmplitudeMatrixInput {
                m_indices: self.m_indices.view(),
                n_indices: self.n_indices.view(),
                left_lambda_count: 6,
                right_lambda_count: 5,
                phase_shifts: self.phase_shifts.view(),
                angular_limit: 3,
                first_leg_polynomials: self.first_polynomials.view(),
                second_leg_polynomials: self.second_polynomials.view(),
                rotation: self.rotation.view(),
                rotation_magnetic_offset: 4,
                xnlm: self.xnlm.view(),
                eta: 0.37,
            }
        }
    }

    fn fmtrxi_reference_data() -> Result<FmtrxiReferenceData, Box<dyn std::error::Error>> {
        let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
        let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
        let phase_shifts = Array1::from_iter((-4..=4).map(|l| {
            let l = l as Real;
            Complex::new(0.015 * l + 0.02, -0.01 * l + 0.03)
        }));
        let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 4,
            mmaxp1: 9,
            rho: Complex::new(1.25, 0.4),
        })?;
        let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 4,
            mmaxp1: 9,
            rho: Complex::new(-0.8, 1.1),
        })?;
        let mut rotation = Array3::zeros((5, 9, 9).f());
        for l in 0..=4 {
            let il = (l + 1) as Real;
            for m1 in -4..=4 {
                for m2 in -4..=4 {
                    if i32_abs_usize(m1) <= l && i32_abs_usize(m2) <= l {
                        let row = (m1 + 4) as usize;
                        let column = (m2 + 4) as usize;
                        rotation[(l, row, column)] =
                            (0.11 * il + 0.07 * (m1 as Real) - 0.05 * (m2 as Real)).cos();
                    }
                }
            }
        }
        let xnlm = legendre_normalization_table(4)?;

        Ok(FmtrxiReferenceData {
            m_indices,
            n_indices,
            phase_shifts,
            first_polynomials,
            second_polynomials,
            rotation,
            xnlm,
        })
    }

    struct MmtrxiReferenceData {
        m_indices: Array1<i32>,
        n_indices: Array1<i32>,
        transition_angular_momenta: Array1<i32>,
        radial_factors: Array1<Complex>,
        transition_matrix: Array4<Complex>,
        first_polynomials: Array2<Complex>,
        second_polynomials: Array2<Complex>,
        xnlm: Array2<Real>,
    }

    impl MmtrxiReferenceData {
        fn input(&self) -> PolarizedScatteringAmplitudeInput<'_> {
            PolarizedScatteringAmplitudeInput {
                m_indices: self.m_indices.view(),
                n_indices: self.n_indices.view(),
                lambda_count: 6,
                transition_angular_momenta: self.transition_angular_momenta.view(),
                radial_factors: self.radial_factors.view(),
                transition_matrix: self.transition_matrix.view(),
                transition_magnetic_offset: 4,
                first_leg_polynomials: self.first_polynomials.view(),
                second_leg_polynomials: self.second_polynomials.view(),
                xnlm: self.xnlm.view(),
                eta: 0.37,
            }
        }
    }

    fn mmtrxi_reference_data() -> Result<MmtrxiReferenceData, Box<dyn std::error::Error>> {
        let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
        let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
        let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
        let radial_factors = Array1::from_iter((1..=8).map(|k| {
            let k = k as Real;
            Complex::new(0.9 + 0.07 * k, -0.02 * k)
        }));
        let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 4,
            mmaxp1: 9,
            rho: Complex::new(1.25, 0.4),
        })?;
        let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 4,
            mmaxp1: 9,
            rho: Complex::new(-0.8, 1.1),
        })?;
        let mut transition_matrix = Array4::zeros((9, 8, 9, 8).f());
        for k2 in 1..=8 {
            for m2 in -4_i32..=4 {
                for k1 in 1..=8 {
                    for m1 in -4_i32..=4 {
                        let first_m = (m1 + 4) as usize;
                        let second_m = (m2 + 4) as usize;
                        transition_matrix[(first_m, k1 - 1, second_m, k2 - 1)] = Complex::new(
                            0.01 * (m1 as Real) + 0.02 * (m2 as Real) + 0.03 * (k1 as Real)
                                - 0.015 * (k2 as Real),
                            0.02 * ((m1 - m2) as Real) + 0.01 * (k1 as Real) + 0.04 * (k2 as Real),
                        );
                    }
                }
            }
        }
        let xnlm = legendre_normalization_table(4)?;

        Ok(MmtrxiReferenceData {
            m_indices,
            n_indices,
            transition_angular_momenta,
            radial_factors,
            transition_matrix,
            first_polynomials,
            second_polynomials,
            xnlm,
        })
    }

    struct MmtrReferenceData {
        transition_angular_momenta: Array1<i32>,
        transition_b_matrix: Array6<Complex>,
        combined_rotation: Array3<Real>,
        first_rotation: Array3<Real>,
        last_rotation: Array3<Real>,
    }

    impl MmtrReferenceData {
        fn polarized_input(&self) -> EnergyIndependentMatrixInput<'_> {
            EnergyIndependentMatrixInput {
                transition_angular_momenta: self.transition_angular_momenta.view(),
                transition_b_matrix: self.transition_b_matrix.view(),
                transition_magnetic_offset: 3,
                spin_index: 1,
                initial_l: 2,
                magnetic_limit: 3,
                rotation_magnetic_offset: 3,
                rotations: TransitionRotationInput::Polarized {
                    first_rotation: self.first_rotation.view(),
                    last_rotation: self.last_rotation.view(),
                    first_eta: 0.23,
                    last_eta: 0.41,
                },
            }
        }

        fn unpolarized_input(&self) -> EnergyIndependentMatrixInput<'_> {
            EnergyIndependentMatrixInput {
                transition_angular_momenta: self.transition_angular_momenta.view(),
                transition_b_matrix: self.transition_b_matrix.view(),
                transition_magnetic_offset: 3,
                spin_index: 0,
                initial_l: 2,
                magnetic_limit: 3,
                rotation_magnetic_offset: 3,
                rotations: TransitionRotationInput::Unpolarized {
                    combined_rotation: self.combined_rotation.view(),
                },
            }
        }
    }

    fn mmtr_reference_data() -> MmtrReferenceData {
        let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
        let mut transition_b_matrix = Array6::zeros((7, 2, 8, 7, 2, 8).f());
        for k2 in 1..=8 {
            for s2 in 0..=1 {
                for m2 in -3_i32..=3 {
                    for k1 in 1..=8 {
                        for s1 in 0..=1 {
                            for m1 in -3_i32..=3 {
                                let first_m = (m1 + 3) as usize;
                                let second_m = (m2 + 3) as usize;
                                transition_b_matrix[(first_m, s1, k1 - 1, second_m, s2, k2 - 1)] =
                                    Complex::new(
                                        0.01 * (m1 as Real)
                                            + 0.02 * (m2 as Real)
                                            + 0.03 * (k1 as Real)
                                            - 0.015 * (k2 as Real)
                                            + 0.04 * (s1 as Real)
                                            - 0.025 * (s2 as Real),
                                        0.02 * ((m1 - m2) as Real)
                                            + 0.01 * (k1 as Real)
                                            + 0.04 * (k2 as Real)
                                            + 0.03 * (s1 as Real)
                                            + 0.02 * (s2 as Real),
                                    );
                            }
                        }
                    }
                }
            }
        }

        let combined_rotation = mmtr_rotation_table(1);
        let first_rotation = mmtr_rotation_table(2);
        let last_rotation = mmtr_rotation_table(3);
        MmtrReferenceData {
            transition_angular_momenta,
            transition_b_matrix,
            combined_rotation,
            first_rotation,
            last_rotation,
        }
    }

    fn mmtr_rotation_table(leg: usize) -> Array3<Real> {
        let mut rotation = Array3::zeros((4, 7, 7).f());
        for l in 0..=3 {
            let il = (l + 1) as Real;
            for m1 in -3_i32..=3 {
                for m2 in -3_i32..=3 {
                    if i32_abs_usize(m1) <= l && i32_abs_usize(m2) <= l {
                        let row = (m1 + 3) as usize;
                        let column = (m2 + 3) as usize;
                        rotation[(l, row, column)] = (0.13 * il + 0.07 * (m1 as Real)
                            - 0.05 * (m2 as Real)
                            + 0.17 * (leg as Real))
                            .cos();
                    }
                }
            }
        }
        rotation
    }

    fn i32_abs_usize(value: i32) -> usize {
        value.unsigned_abs() as usize
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    fn complex_sum(table: &ndarray::Array2<Complex>) -> Complex {
        table
            .iter()
            .copied()
            .fold(Complex::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn active_bmati_sum(table: &Array4<Complex>) -> Complex {
        let mut sum = Complex::new(0.0, 0.0);
        for mu1 in 1..=5 {
            for k1 in 0..8 {
                for mu2 in 1..=5 {
                    for k2 in 0..8 {
                        sum += table[(mu1, k1, mu2, k2)];
                    }
                }
            }
        }
        sum
    }

    fn complex_nonzero_count(table: &ndarray::Array2<Complex>) -> usize {
        table
            .iter()
            .filter(|&&value| value.re.abs() > 1.0e-14 || value.im.abs() > 1.0e-14)
            .count()
    }

    fn rotation_value(rotation: &InitialStateRotation, il: usize, m1: isize, m2: isize) -> f64 {
        let row = (m1 + rotation.magnetic_offset as isize) as usize;
        let column = (m2 + rotation.magnetic_offset as isize) as usize;
        rotation.matrix[(il - 1, row, column)]
    }

    fn rotation_sum(rotation: &InitialStateRotation) -> f64 {
        rotation.matrix.iter().sum()
    }

    fn rotation_nonzero_count(rotation: &InitialStateRotation) -> usize {
        rotation
            .matrix
            .iter()
            .filter(|&&value| value.abs() > 1.0e-14)
            .count()
    }
}
