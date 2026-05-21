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

use crate::{Complex, Real};

mod types;

pub use types::*;

const ONE_DEGREE_RADIANS: Real = 0.017_453_292_52;
const RDPATH_EPSILON: Real = 1.0e-6;
const SNLM_AFAC: Real = 1.0 / 64.0;
const SNLM_FACTORIAL_LIMIT: usize = 210;
const SNLM_FACTORIAL_COUNT: usize = SNLM_FACTORIAL_LIMIT + 1;

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

/// Build FEFF `snlm` associated-Legendre normalization factors.
///
/// The returned table uses FEFF's GENFMT layout `xnlm(il, im)` with Rust
/// indices `(l, m)`. Entries where `m > l` are retained as zeroes. The
/// implementation follows FEFF's scaled-factorial calculation so the values
/// match `snlm.f90` while keeping invalid dimensions as explicit errors.
pub fn genfmt_legendre_normalization_table(
    input: GenfmtLegendreNormalizationInput,
) -> Result<Array2<Real>, GenfmtError> {
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;

    let l_max = input.lmaxp1 - 1;
    let m_max = input.mmaxp1 - 1;
    let active_m_max = l_max.min(m_max);
    let factorial_limit =
        l_max
            .checked_add(active_m_max)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "lmaxp1+mmaxp1",
                value: input.lmaxp1,
            })?;
    if factorial_limit > SNLM_FACTORIAL_LIMIT {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1+mmaxp1",
            value: factorial_limit,
        });
    }

    let scaled_factorials = snlm_scaled_factorials();
    let mut table = Array2::<Real>::zeros((input.lmaxp1, input.mmaxp1).f());
    for il in 1..=input.lmaxp1 {
        let active_m = input.mmaxp1.min(il);
        for im in 1..=active_m {
            let l = il - 1;
            let m = im - 1;
            let odd_factor = checked_double_plus_one("lmaxp1", l)? as Real;
            let value = (odd_factor * scaled_factorials[l - m] / scaled_factorials[l + m]).sqrt()
                * SNLM_AFAC.powi(checked_i32("mmaxp1", m)?);
            if !value.is_finite() {
                return Err(GenfmtError::NonFiniteScalar {
                    field: "xnlm",
                    value,
                });
            }
            table[(l, m)] = value;
        }
    }

    Ok(table)
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

fn snlm_scaled_factorials() -> [Real; SNLM_FACTORIAL_COUNT] {
    let mut factors = [0.0; SNLM_FACTORIAL_COUNT];
    factors[0] = 1.0;
    factors[1] = SNLM_AFAC;
    for i in 2..=SNLM_FACTORIAL_LIMIT {
        factors[i] = factors[i - 1] * (i as Real) * SNLM_AFAC;
    }
    factors
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
mod tests;
