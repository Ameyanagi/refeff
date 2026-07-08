use super::polynomial::alternating_sign;
use super::validation::*;
use super::*;
use ndarray::{Axis, Slice};

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

/// Build FEFF `rot3i` rotation tables for the GENFMT path setup loop.
///
/// FEFF calls `rot3i` once for each real path leg and, when polarization is
/// active, once more for the extra `nleg + 1` pseudo-leg produced by `rdpath`.
pub fn genfmt_path_rotation_tables(
    input: GenfmtPathRotationTablesInput<'_>,
) -> Result<GenfmtPathRotationTables, GenfmtError> {
    validate_positive_limit("leg_count", input.leg_count)?;
    validate_positive_limit("lmaxp1", input.lmaxp1)?;
    validate_positive_limit("mmaxp1", input.mmaxp1)?;
    if let Some((lmaxp1, mmaxp1)) = input.polarized_extra {
        validate_positive_limit("polarized_lmaxp1", lmaxp1)?;
        validate_positive_limit("polarized_mmaxp1", mmaxp1)?;
    }

    let rotation_count = input.leg_count + usize::from(input.polarized_extra.is_some());
    ensure_rotation_axis_len(
        "beta_angles",
        "leg",
        input.beta_angles.len(),
        rotation_count,
    )?;

    let extra_lmaxp1 = input
        .polarized_extra
        .map(|(lmaxp1, _)| lmaxp1)
        .unwrap_or(input.lmaxp1);
    let extra_mmaxp1 = input
        .polarized_extra
        .map(|(_, mmaxp1)| mmaxp1)
        .unwrap_or(input.mmaxp1);
    let rotation_lmaxp1 = input.lmaxp1.max(extra_lmaxp1);
    let rotation_mmaxp1 = input.mmaxp1.max(extra_mmaxp1);
    let rotation_magnetic_offset = rotation_mmaxp1 - 1;
    let magnetic_dim = checked_double_plus_one("rotation_mmaxp1", rotation_magnetic_offset)?;
    let mut rotations =
        Array4::<Real>::zeros((rotation_count, rotation_lmaxp1, magnetic_dim, magnetic_dim).f());

    for leg_index in 0..input.leg_count {
        let rotation = initial_state_rotation(InitialStateRotationInput {
            lmaxp1: input.lmaxp1,
            mmaxp1: input.mmaxp1,
            beta_angle: input.beta_angles[leg_index],
        })?;
        copy_rotation_to_common_table(
            &rotation,
            rotation_magnetic_offset,
            leg_index,
            &mut rotations,
        )?;
    }

    if let Some((lmaxp1, mmaxp1)) = input.polarized_extra {
        let rotation = initial_state_rotation(InitialStateRotationInput {
            lmaxp1,
            mmaxp1,
            beta_angle: input.beta_angles[input.leg_count],
        })?;
        copy_rotation_to_common_table(
            &rotation,
            rotation_magnetic_offset,
            input.leg_count,
            &mut rotations,
        )?;
    }

    Ok(GenfmtPathRotationTables {
        rotations,
        real_leg_count: input.leg_count,
        rotation_magnetic_offset,
    })
}

impl GenfmtPathRotationTables {
    /// Total number of FEFF `rot3i` tables stored in call order.
    pub fn len(&self) -> usize {
        self.rotations.shape()[0]
    }

    /// Whether no FEFF `rot3i` tables are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rotation tables for FEFF's real path legs, excluding the pseudo-leg.
    pub fn real_leg_rotations(&self) -> Result<ArrayView4<'_, Real>, GenfmtError> {
        validate_positive_limit("real_leg_count", self.real_leg_count)?;
        ensure_rotation_axis_len(
            "rotations",
            "leg",
            self.rotations.shape()[0],
            self.real_leg_count,
        )?;
        Ok(self
            .rotations
            .slice_axis(Axis(0), Slice::from(..self.real_leg_count)))
    }

    /// Optional polarized pseudo-leg rotation at FEFF `nleg + 1`.
    pub fn polarized_extra_rotation(&self) -> Result<Option<ArrayView3<'_, Real>>, GenfmtError> {
        validate_positive_limit("real_leg_count", self.real_leg_count)?;
        let extra_index = self.real_leg_count;
        if self.rotations.shape()[0] <= extra_index {
            return Ok(None);
        }
        Ok(Some(self.rotations.index_axis(Axis(0), extra_index)))
    }

    /// Select the FEFF rotations consumed by `mmtr`, `mmtrjas`, and `mmtrjas0`.
    ///
    /// Unpolarized paths use FEFF `dri(:,:,:,nleg)`. Polarized paths use the
    /// pseudo-leg `dri(:,:,:,nleg+1)` for the first side and the real final-leg
    /// `dri(:,:,:,nleg)` for the second side, plus `eta(0)` and `eta(nleg+1)`.
    pub fn transition_rotations<'a>(
        &'a self,
        eta_values: ArrayView1<'a, Real>,
        polarized: bool,
    ) -> Result<TransitionRotationInput<'a>, GenfmtError> {
        validate_positive_limit("real_leg_count", self.real_leg_count)?;
        ensure_rotation_axis_len(
            "rotations",
            "leg",
            self.rotations.shape()[0],
            self.real_leg_count,
        )?;

        let last_leg_rotation = self.rotations.index_axis(Axis(0), self.real_leg_count - 1);
        if !polarized {
            return Ok(TransitionRotationInput::Unpolarized {
                combined_rotation: last_leg_rotation,
            });
        }

        let eta_last_index =
            self.real_leg_count
                .checked_add(1)
                .ok_or(GenfmtError::InvalidAngularLimit {
                    name: "real_leg_count",
                    value: self.real_leg_count,
                })?;
        ensure_rotation_axis_len("eta_values", "leg", eta_values.len(), eta_last_index + 1)?;
        let first_eta = finite_eta_value(eta_values, 0)?;
        let last_eta = finite_eta_value(eta_values, eta_last_index)?;
        let Some(first_rotation) = self.polarized_extra_rotation()? else {
            return Err(GenfmtError::TableAxisTooShort {
                table: "rotations",
                axis: "leg",
                length: self.rotations.shape()[0],
                required: self.real_leg_count + 1,
            });
        };

        Ok(TransitionRotationInput::Polarized {
            first_rotation,
            last_rotation: last_leg_rotation,
            first_eta,
            last_eta,
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

fn copy_rotation_to_common_table(
    rotation: &InitialStateRotation,
    target_magnetic_offset: usize,
    rotation_index: usize,
    table: &mut Array4<Real>,
) -> Result<(), GenfmtError> {
    ensure_rotation_axis_len("rotations", "leg", table.shape()[0], rotation_index + 1)?;
    ensure_rotation_axis_len(
        "rotations",
        "l",
        table.shape()[1],
        rotation.matrix.shape()[0],
    )?;
    let target_offset =
        isize::try_from(target_magnetic_offset).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: target_magnetic_offset,
        })?;
    let source_offset = isize::try_from(rotation.magnetic_offset).map_err(|_| {
        GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: rotation.magnetic_offset,
        }
    })?;

    for l in 0..rotation.matrix.shape()[0] {
        for source_row in 0..rotation.matrix.shape()[1] {
            let m1 = isize::try_from(source_row).map_err(|_| GenfmtError::InvalidAngularLimit {
                name: "rotation_magnetic_offset",
                value: rotation.magnetic_offset,
            })? - source_offset;
            let target_row = shifted_index(
                m1,
                target_offset,
                "rotation_magnetic_offset",
                target_magnetic_offset,
            )?;
            ensure_rotation_axis_len("rotations", "m1", table.shape()[2], target_row + 1)?;

            for source_column in 0..rotation.matrix.shape()[2] {
                let m2 = isize::try_from(source_column).map_err(|_| {
                    GenfmtError::InvalidAngularLimit {
                        name: "rotation_magnetic_offset",
                        value: rotation.magnetic_offset,
                    }
                })? - source_offset;
                let target_column = shifted_index(
                    m2,
                    target_offset,
                    "rotation_magnetic_offset",
                    target_magnetic_offset,
                )?;
                ensure_rotation_axis_len("rotations", "m2", table.shape()[3], target_column + 1)?;
                table[(rotation_index, l, target_row, target_column)] =
                    rotation.matrix[(l, source_row, source_column)];
            }
        }
    }

    Ok(())
}

fn finite_eta_value(eta_values: ArrayView1<'_, Real>, index: usize) -> Result<Real, GenfmtError> {
    let value = eta_values[index];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteVector {
            field: "eta_values",
            index,
            value,
        })
    }
}

fn ensure_rotation_axis_len(
    table: &'static str,
    axis: &'static str,
    length: usize,
    required: usize,
) -> Result<(), GenfmtError> {
    if length < required {
        Err(GenfmtError::TableAxisTooShort {
            table,
            axis,
            length,
            required,
        })
    } else {
        Ok(())
    }
}
