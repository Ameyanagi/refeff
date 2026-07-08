use super::validation::{checked_count, validate_finite_scalar, validate_positive_limit};
use super::*;

/// Compose ordinary FEFF GENFMT path setup before the spin and energy loops.
///
/// This ports the driver block that runs `rdpath`, computes
/// `reff=sum(ri)/2`, calls `setlam(icalc, 1)`, then fills `dri` with `rot3i`.
/// In the polarized ordinary branch FEFF calls the extra pseudo-leg rotation
/// with `(ilinit + 1, ilinit + 1)`.
pub fn genfmt_ordinary_path_setup(
    input: GenfmtOrdinaryPathSetupInput<'_>,
) -> Result<GenfmtPathSetup, GenfmtError> {
    genfmt_path_setup(GenfmtPathSetupParts {
        positions: input.positions,
        polarized: input.polarized,
        calculation: input.calculation,
        initial_l: input.initial_l,
        lambda_capacity: input.lambda_capacity,
        max_m: input.max_m,
        max_n: input.max_n,
        lmaxp1: input.lmaxp1,
        polarized_extra_mode: GenfmtPolarizedExtraRotation::OrdinaryInitialState,
    })
}

/// Compose FEFF GENFMTJAS path setup before the energy loop.
///
/// GENFMTJAS uses the same `rdpath`, `setlam(icalc, 1)`, and real-leg `rot3i`
/// setup as ordinary GENFMT, but its polarized pseudo-leg currently uses
/// `(lmaxp1, mmaxp1)`.
pub fn genfmt_jas_path_setup(
    input: GenfmtJasPathSetupInput<'_>,
) -> Result<GenfmtPathSetup, GenfmtError> {
    genfmt_path_setup(GenfmtPathSetupParts {
        positions: input.positions,
        polarized: input.polarized,
        calculation: input.calculation,
        initial_l: input.initial_l,
        lambda_capacity: input.lambda_capacity,
        max_m: input.max_m,
        max_n: input.max_n,
        lmaxp1: input.lmaxp1,
        polarized_extra_mode: GenfmtPolarizedExtraRotation::FullPath,
    })
}

#[derive(Debug, Clone, Copy)]
struct GenfmtPathSetupParts<'a> {
    positions: ArrayView2<'a, Real>,
    polarized: bool,
    calculation: i32,
    initial_l: usize,
    lambda_capacity: usize,
    max_m: usize,
    max_n: usize,
    lmaxp1: usize,
    polarized_extra_mode: GenfmtPolarizedExtraRotation,
}

#[derive(Debug, Clone, Copy)]
enum GenfmtPolarizedExtraRotation {
    OrdinaryInitialState,
    FullPath,
}

fn genfmt_path_setup(input: GenfmtPathSetupParts<'_>) -> Result<GenfmtPathSetup, GenfmtError> {
    let angles = path_rotation_angles(PathRotationInput {
        positions: input.positions,
        polarized: input.polarized,
    })?;
    let leg_count = angles.leg_lengths.len();
    validate_positive_limit("leg_count", leg_count)?;
    let scattering_count = leg_count
        .checked_sub(1)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "leg_count",
            value: leg_count,
        })?;
    let beta_angles = angles
        .beta_angles
        .as_slice()
        .ok_or(GenfmtError::TableAxisTooShort {
            table: "beta_angles",
            axis: "storage",
            length: 0,
            required: angles.beta_angles.len(),
        })?;
    let lambda = lambda_indices(LambdaIndexInput {
        calculation: input.calculation,
        energy_index: 1,
        scattering_count,
        initial_l: input.initial_l,
        beta_angles: &beta_angles[..leg_count],
        lambda_capacity: input.lambda_capacity,
        max_m: input.max_m,
        max_n: input.max_n,
    })?;
    let mmaxp1 = lambda.max_m_plus_one;
    let polarized_extra = if input.polarized {
        match input.polarized_extra_mode {
            GenfmtPolarizedExtraRotation::OrdinaryInitialState => {
                let initial_limit = checked_count("initial_l", input.initial_l)?;
                Some((initial_limit, initial_limit))
            }
            GenfmtPolarizedExtraRotation::FullPath => Some((input.lmaxp1, mmaxp1)),
        }
    } else {
        None
    };
    let rotations = genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
        beta_angles: angles.beta_angles.view(),
        leg_count,
        lmaxp1: input.lmaxp1,
        mmaxp1,
        polarized_extra,
    })?;
    let effective_half_path_length =
        genfmt_path_setup_effective_half_path_length(angles.leg_lengths.view())?;

    Ok(GenfmtPathSetup {
        angles,
        effective_half_path_length,
        lambda,
        rotations,
    })
}

fn genfmt_path_setup_effective_half_path_length(
    leg_lengths: ArrayView1<'_, Real>,
) -> Result<Real, GenfmtError> {
    validate_positive_limit("leg_lengths", leg_lengths.len())?;
    let mut path_length = 0.0;
    for (index, &leg_length) in leg_lengths.iter().enumerate() {
        if !leg_length.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field: "leg_lengths",
                index,
                value: leg_length,
            });
        }
        if leg_length < 0.0 {
            return Err(GenfmtError::NegativeScalar {
                field: "leg_lengths",
                value: leg_length,
            });
        }
        path_length += leg_length;
    }
    let effective_half_path_length = path_length / 2.0;
    validate_finite_scalar("effective_half_path_length", effective_half_path_length)?;
    Ok(effective_half_path_length)
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
