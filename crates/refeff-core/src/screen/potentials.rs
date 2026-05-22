//! SCREEN potential and CRPA density helper kernels.

use ndarray::{Array1, Array2, ShapeBuilder};

use crate::{Real, RealMat, RealVec};

use super::types::*;
use super::validation::{
    validate_active_count, validate_count_at_least, validate_finite, validate_increasing,
    validate_positive, validate_positive_result, validate_result_finite,
};

/// Port of SCREEN `ldafxc`: local-density exchange-correlation kernel.
///
/// FEFF evaluates only the first `active_count` rows, sets non-positive
/// electron-density rows to zero, and uses a pure-exchange branch when
/// `exchange_selector == 2`.
pub fn screen_lda_exchange_correlation_kernel(
    radii: &[Real],
    electron_density: &[Real],
    exchange_selector: i32,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, electron_density.len())?;

    let mut output = Array1::zeros(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let density = electron_density[index];
        validate_positive("radius", radius)?;
        validate_finite("electron_density", density)?;
        if density <= 0.0 {
            continue;
        }

        let rs = (density / 3.0).powf(-1.0 / 3.0);
        let exchange = -1.222 / rs;
        let correlation = if exchange_selector == 2 {
            0.0
        } else {
            -0.75924 / (11.4 + rs)
        };
        output[index] = rs.powi(3) / radius.powi(2) / 6.0 * (exchange + correlation);
    }
    Ok(output)
}

/// Port the SCREEN/CRPA radial Coulomb response kernel setup.
///
/// FEFF fills the upper triangle as `K(m,n) = 4*pi/r(n)`, mirrors it into the
/// lower triangle, and optionally adds `4*pi*fxc(i)` to the diagonal for TDLDA
/// runs. Because the FEFF radial grid is monotonically increasing, the
/// symmetric result is `4*pi/max(r_i, r_j)` plus the optional diagonal local
/// exchange-correlation term. The returned matrix uses Fortran-order
/// [`ndarray::Array2`] storage so downstream solver code can preserve FEFF's
/// column-major traversal.
pub fn screen_coulomb_kernel_matrix(
    radii: &[Real],
    active_count: usize,
    local_kernel: Option<&[Real]>,
) -> Result<RealMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    if let Some(local_kernel) = local_kernel {
        validate_active_count(active_count, local_kernel.len())?;
    }

    for &radius in radii.iter().take(active_count) {
        validate_positive("radius", radius)?;
    }

    let scale = 4.0 * std::f64::consts::PI;
    let mut matrix = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in row..active_count {
            let value = scale / radii[column];
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
    if let Some(local_kernel) = local_kernel {
        for index in 0..active_count {
            let value = local_kernel[index];
            validate_finite("local_kernel", value)?;
            matrix[(index, index)] += scale * value;
        }
    }
    Ok(matrix)
}

/// Port the SCREEN bare core-hole potential setup.
///
/// FEFF first forms shell weights
/// `(dgc0(i)^2 + dpc0(i)^2) * dx * r(i)`, then evaluates the radial Coulomb
/// potential `int rho(r') / max(r, r') dr'`. This helper returns FEFF's final
/// `vch = wscrn` vector. The implementation uses prefix and suffix reductions
/// instead of the original nested loops, preserving the same mathematical
/// expression with linear complexity.
pub fn screen_bare_core_hole_potential(
    radii: &[Real],
    large_component: &[Real],
    small_component: &[Real],
    dx: Real,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, large_component.len())?;
    validate_active_count(active_count, small_component.len())?;

    let mut shell_weight = Vec::with_capacity(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let large = large_component[index];
        let small = small_component[index];
        validate_positive("radius", radius)?;
        validate_finite("large_component", large)?;
        validate_finite("small_component", small)?;
        let radial_density = large.mul_add(large, small * small);
        let shell = radial_density * dx * radius;
        validate_result_finite("core_hole_shell_weight", shell)?;
        shell_weight.push(shell);
    }

    screen_radial_coulomb_potential(radii, &shell_weight, active_count)
}

/// Evaluate FEFF's radial Coulomb potential from shell weights.
///
/// Both `SCREEN/screensub.f90` and `CRPA/chi_crpa.f90` form radial shell
/// weights first and then evaluate `sum_j weight(j) / max(r_i, r_j)`. This
/// helper keeps that common loop available for core-hole and CRPA density
/// sources. Prefix and suffix reductions preserve the FEFF expression while
/// avoiding the original nested-loop cost.
pub fn screen_radial_coulomb_potential(
    radii: &[Real],
    shell_weights: &[Real],
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, shell_weights.len())?;

    let mut outer_weight = Vec::with_capacity(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let shell = shell_weights[index];
        validate_positive("radius", radius)?;
        validate_finite("shell_weight", shell)?;
        outer_weight.push(shell / radius);
    }

    let mut tail = vec![0.0; active_count + 1];
    for index in (0..active_count).rev() {
        tail[index] = tail[index + 1] + outer_weight[index];
        validate_result_finite("radial_coulomb_tail_weight", tail[index])?;
    }

    let mut prefix = 0.0;
    let mut output = Array1::zeros(active_count);
    for index in 0..active_count {
        prefix += shell_weights[index];
        validate_result_finite("radial_coulomb_prefix_weight", prefix)?;
        let value = prefix / radii[index] + tail[index + 1];
        validate_result_finite("radial_coulomb_potential", value)?;
        output[index] = value;
    }
    Ok(output)
}

/// Port the CRPA total-density projection and normalization setup.
///
/// `CRPA/chi_crpa.f90` optionally damps the total density by a
/// `cos(...)^4` radial window, normalizes `sum rho(r_i) * r_i * dx` to one,
/// and forms shell weights for the following Coulomb-potential loop. FEFF then
/// zeros `vch(jnrm+1:)`; pass `norman_count = jnrm` to preserve that active
/// prefix.
pub fn screen_crpa_density_weights(
    radii: &[Real],
    total_density: &[Real],
    dx: Real,
    active_count: usize,
    norman_count: usize,
    projection_window: Option<ScreenCrpaProjectionWindow>,
) -> Result<ScreenCrpaDensityWeights, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_count_at_least("norman_count", norman_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, total_density.len())?;
    if norman_count > active_count {
        return Err(ScreenError::ActiveCountOutOfRange {
            active_count: norman_count,
            len: active_count,
        });
    }
    if let Some(window) = projection_window {
        validate_finite("projection_inner_radius", window.inner_radius)?;
        validate_finite("projection_outer_radius", window.outer_radius)?;
        validate_increasing(
            "projection_inner_radius",
            window.inner_radius,
            "projection_outer_radius",
            window.outer_radius,
        )?;
    }

    let mut projected_density = Vec::with_capacity(active_count);
    let mut normalization = 0.0;
    for index in 0..active_count {
        let radius = radii[index];
        let mut density = total_density[index];
        validate_positive("radius", radius)?;
        validate_finite("total_density", density)?;
        if let Some(window) = projection_window {
            let clamped_radius = radius.max(window.inner_radius).min(window.outer_radius);
            let scaled = (clamped_radius - window.inner_radius)
                / (window.outer_radius - window.inner_radius);
            density *= (scaled * std::f64::consts::FRAC_PI_2).cos().powi(4);
            validate_result_finite("projected_crpa_density", density)?;
        }
        normalization += density * radius * dx;
        validate_result_finite("crpa_density_normalization", normalization)?;
        projected_density.push(density);
    }
    validate_positive_result("crpa_density_normalization", normalization)?;

    let mut normalized_density = Array1::zeros(active_count);
    let mut shell_weights = Array1::zeros(active_count);
    for index in 0..active_count {
        let density = projected_density[index] / normalization;
        validate_result_finite("normalized_crpa_density", density)?;
        normalized_density[index] = density;
        if index < norman_count {
            let shell = density * dx * radii[index];
            validate_result_finite("crpa_shell_weight", shell)?;
            shell_weights[index] = shell;
        }
    }

    Ok(ScreenCrpaDensityWeights {
        normalized_density,
        shell_weights,
        normalization,
    })
}

/// Port the CRPA Hubbard-parameter accumulation loop.
///
/// After solving the screened response equation, FEFF stores
/// `vch(i) = wscrn(i) * den_CRPA(i,ie)` and accumulates screened and bare
/// Hubbard interactions with the normalized total CRPA density:
/// `sum potential(i) * totden_CRPA(i) * dx * ri(i)`. The scalar outputs are the
/// values FEFF writes to `crpa.dat`; no Hartree-to-eV conversion is applied.
pub fn screen_crpa_hubbard_summary(
    radii: &[Real],
    screened_potential: &[Real],
    bare_potential: &[Real],
    total_density: &[Real],
    orbital_density: &[Real],
    dx: Real,
    active_count: usize,
) -> Result<ScreenCrpaHubbardSummary, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, screened_potential.len())?;
    validate_active_count(active_count, bare_potential.len())?;
    validate_active_count(active_count, total_density.len())?;
    validate_active_count(active_count, orbital_density.len())?;

    let mut screened_density_potential = Array1::zeros(active_count);
    let mut hubbard_u = 0.0;
    let mut bare_u = 0.0;
    let mut occupation = 0.0;
    for index in 0..active_count {
        let radius = radii[index];
        let screened = screened_potential[index];
        let bare = bare_potential[index];
        let total = total_density[index];
        let orbital = orbital_density[index];
        validate_positive("radius", radius)?;
        validate_finite("screened_potential", screened)?;
        validate_finite("bare_potential", bare)?;
        validate_finite("total_density", total)?;
        validate_finite("orbital_density", orbital)?;

        let density_potential = screened * orbital;
        validate_result_finite("crpa_screened_density_potential", density_potential)?;
        screened_density_potential[index] = density_potential;

        let weight = total * dx * radius;
        validate_result_finite("crpa_hubbard_weight", weight)?;
        hubbard_u += screened * weight;
        bare_u += bare * weight;
        occupation += weight;
        validate_result_finite("crpa_hubbard_u", hubbard_u)?;
        validate_result_finite("crpa_bare_u", bare_u)?;
        validate_result_finite("crpa_occupation", occupation)?;
    }

    Ok(ScreenCrpaHubbardSummary {
        screened_density_potential,
        hubbard_u,
        occupation,
        bare_u,
    })
}
