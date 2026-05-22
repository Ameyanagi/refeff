use super::*;

/// Port of FEFF `xgllm`: z-axis Rehr-Albers propagator term.
///
/// `xclm` is indexed as `xclm(m, l, atom2 - 1, atom1 - 1)` and `xnlm` as
/// `xnlm(mu, l)`, matching FEFF's zero-based angular axes and one-based atom
/// labels. The state atoms in [`StateKet`] are therefore interpreted as FEFF
/// one-based atom indices.
pub fn rehr_albers_z_axis_propagator(
    mu: usize,
    first: StateKet,
    second: StateKet,
    xclm: ArrayView4<'_, Complex32>,
    xnlm: ArrayView2<'_, Real>,
) -> Result<Complex32, FmsError> {
    let iat1 = checked_atom_index(first.atom)?;
    let iat2 = checked_atom_index(second.atom)?;
    let l1 = first.angular_momentum;
    let l2 = second.angular_momentum;

    if mu > l1 {
        return Err(FmsError::MuOutOfRange {
            mu,
            angular_momentum: l1,
        });
    }

    ensure_axis_len("xclm", "m", xclm.shape()[0], l1.max(l2))?;
    ensure_axis_len("xclm", "l", xclm.shape()[1], l1.max(l2))?;
    ensure_axis_len("xclm", "atom2", xclm.shape()[2], iat2)?;
    ensure_axis_len("xclm", "atom1", xclm.shape()[3], iat1)?;
    ensure_axis_len("xnlm", "mu", xnlm.shape()[0], mu)?;
    ensure_axis_len("xnlm", "l", xnlm.shape()[1], l1.max(l2))?;

    if mu > l2 {
        return Ok(Complex32::new(0.0, 0.0));
    }

    let norm_l1 = normalization_value(xnlm, mu, l1)?;
    let norm_l2 = normalization_value(xnlm, mu, l2)?;
    let angular_weight = angular_weight(l1)?;
    let sign = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };
    let numax = l1.min(l2 - mu);

    let sum = (0..=numax).try_fold(Complex32::new(0.0, 0.0), |sum, nu| {
        let mn = mu.checked_add(nu).ok_or(FmsError::InvalidAngularLimit {
            name: "mu",
            value: mu,
            lx: l2,
        })?;
        let gamtl = angular_weight * xclm[(nu, l1, iat2, iat1)] / norm_l1;
        let gam = xclm[(mn, l2, iat2, iat1)] * (sign * norm_l2);
        Ok::<Complex32, FmsError>(sum + gamtl * gam)
    })?;

    Ok(sum)
}
