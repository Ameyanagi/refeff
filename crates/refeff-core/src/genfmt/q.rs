use super::validation::*;
use super::*;

/// Prepare q-vector phases and beta angles for FEFF `GENFMT/genfmtjas.f90`.
///
/// This ports the small `angread` replacement block in `genfmtjas`: regular
/// NRIXS paths convert `qtrig(:,1:4)` into `pha=conjg(cmplx(qcsf,qsnf))` and
/// `beta=atan2(qsnt,qcst)`, while q-averaged paths use one unrotated q entry
/// with unit weight.
pub fn jas_q_angles(input: JasQAngleInput<'_>) -> Result<JasQAngles, GenfmtError> {
    if input.qaverage {
        return Ok(JasQAngles {
            phases: Array1::from_vec(vec![Complex::new(1.0, 0.0)]),
            beta_angles: Array1::from_vec(vec![0.0]),
            weights: Array1::from_vec(vec![Complex::new(1.0, 0.0)]),
        });
    }

    let q_count = input.q_weights.len();
    validate_positive_limit("q_count", q_count)?;
    ensure_axis_len("q_trig", "q", input.q_trig.shape()[0], q_count)?;
    ensure_axis_len("q_trig", "component", input.q_trig.shape()[1], 4)?;

    let mut phases = Array1::<Complex>::zeros(q_count);
    let mut beta_angles = Array1::<Real>::zeros(q_count);
    let mut weights = Array1::<Complex>::zeros(q_count);
    for q in 0..q_count {
        let cos_theta = q_trig_entry(input.q_trig, q, 0)?;
        let sin_theta = q_trig_entry(input.q_trig, q, 1)?;
        let cos_phi = q_trig_entry(input.q_trig, q, 2)?;
        let sin_phi = q_trig_entry(input.q_trig, q, 3)?;
        let weight = q_weight_entry(input.q_weights, q)?;
        phases[q] = Complex::new(cos_phi, -sin_phi);
        beta_angles[q] = sin_theta.atan2(cos_theta);
        weights[q] = weight;
    }

    Ok(JasQAngles {
        phases,
        beta_angles,
        weights,
    })
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

fn q_trig_entry(
    q_trig: ArrayView2<'_, Real>,
    q: usize,
    component: usize,
) -> Result<Real, GenfmtError> {
    let value = q_trig[(q, component)];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableScalar {
            table: "q_trig",
            row: q,
            column: component,
            value,
        })
    }
}

fn q_weight_entry(q_weights: ArrayView1<'_, Complex>, q: usize) -> Result<Complex, GenfmtError> {
    let value = q_weights[q];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table: "q_weights",
            row: q,
            column: 0,
            real: value.re,
            imaginary: value.im,
        })
    }
}
