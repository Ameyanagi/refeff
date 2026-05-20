//! FEFF SCREEN helper kernels.
//!
//! These routines cover small, self-contained pieces from `SCREEN/frgrid.f90`,
//! `SCREEN/fegrid.f90`, `SCREEN/fxc.f90`, and the response setup blocks in
//! `SCREEN/screensub.f90` and `CRPA/chi_crpa.f90`, plus the compact CRPA radial
//! density setup block. The full SCREEN/CRPA drivers also depend on phase,
//! potential, and FMS handoff state; keeping these kernels separate makes them
//! usable and testable while those drivers are ported incrementally.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use refeff_linalg::{real_lu_factor, real_lu_solve_vector};
use thiserror::Error;

use crate::{Complex, ComplexMat, ComplexVec, Real, RealMat, RealVec};

/// Error returned by FEFF SCREEN helper kernels.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ScreenError {
    #[error("SCREEN input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    #[error("SCREEN complex input {name} must be finite, got {real}+{imaginary}i")]
    NonFiniteComplexInput {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    #[error("SCREEN input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    #[error("SCREEN radial count must be positive")]
    EmptyRadialGrid,
    #[error("SCREEN active radial count {active_count} exceeds input length {len}")]
    ActiveCountOutOfRange { active_count: usize, len: usize },
    #[error("SCREEN radial index is outside isize range: {value}")]
    RadialIndexOutOfRange { value: Real },
    #[error("SCREEN {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    #[error("SCREEN input {upper_name} must exceed {lower_name}: {upper} <= {lower}")]
    NonIncreasingInput {
        lower_name: &'static str,
        upper_name: &'static str,
        lower: Real,
        upper: Real,
    },
    #[error("SCREEN energy grid requires {required} points but capacity is {available}")]
    EnergyGridTooLong { required: usize, available: usize },
    #[error("SCREEN energy grid size overflow for {name}")]
    EnergyGridSizeOverflow { name: &'static str },
    #[error("SCREEN energy grid unexpectedly has no points")]
    EmptyEnergyGrid,
    #[error("SCREEN energy index {index} is out of range for {len} energies")]
    EnergyIndexOutOfRange { index: usize, len: usize },
    #[error("SCREEN result {name} must be finite, got {value}")]
    NonFiniteResult { name: &'static str, value: Real },
    #[error("SCREEN complex result {name} must be finite, got {real}+{imaginary}i")]
    NonFiniteComplexResult {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    #[error("SCREEN result {name} must be positive, got {value}")]
    NonPositiveResult { name: &'static str, value: Real },
    #[error(
        "SCREEN matrix {name} must be at least {active_count}x{active_count}, got {rows}x{columns}"
    )]
    MatrixTooSmall {
        name: &'static str,
        rows: usize,
        columns: usize,
        active_count: usize,
    },
    #[error("SCREEN matrix {name}({row},{column}) must be finite, got {value}")]
    NonFiniteMatrixInput {
        name: &'static str,
        row: usize,
        column: usize,
        value: Real,
    },
    #[error("SCREEN complex matrix {name}({row},{column}) must be finite, got {real}+{imaginary}i")]
    NonFiniteComplexMatrixInput {
        name: &'static str,
        row: usize,
        column: usize,
        real: Real,
        imaginary: Real,
    },
    #[error("SCREEN linear solve failed: {0}")]
    Linalg(#[from] refeff_linalg::LinalgError),
}

/// Inputs for SCREEN `setegi`: rectangular complex-energy contour setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenContourEnergyGridInput {
    /// Lower real-axis energy `emin`.
    pub min_real_energy: Real,
    /// Upper real-axis energy `emax`.
    pub max_real_energy: Real,
    /// Maximum imaginary-axis energy `eimax`.
    pub max_imaginary_energy: Real,
    /// Minimum imaginary-axis offset `ermin`; FEFF clamps non-positive values to 0.05.
    pub min_imaginary_energy: Real,
    /// Number of real-axis divisions `ner`.
    pub real_points: usize,
    /// Number of imaginary-axis divisions `nei`.
    pub imaginary_points: usize,
    /// Capacity of the output energy table, equivalent to FEFF `nex`.
    pub max_points: usize,
}

/// SCREEN complex-energy contour with FEFF's active-length convention.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenContourEnergyGrid {
    /// Complex contour energies `em`, zero-filled after [`ScreenContourEnergyGrid::active_len`].
    pub energies: ComplexVec,
    /// Number of active contour points returned as FEFF `ne`.
    pub active_len: usize,
    /// Effective `ermin` after FEFF's non-positive clamp.
    pub effective_min_imaginary_energy: Real,
}

/// CRPA radial projection window from `chi_crpa.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenCrpaProjectionWindow {
    /// Lower clamp radius. FEFF uses `rcut0 = rcut - 1`.
    pub inner_radius: Real,
    /// Upper clamp radius. FEFF uses `rcut = rnrm * rcutin`.
    pub outer_radius: Real,
}

/// Normalized CRPA radial density and shell weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenCrpaDensityWeights {
    /// Density after optional projection and FEFF normalization.
    pub normalized_density: RealVec,
    /// FEFF `vch(i) = normalized_density(i) * dx * ri(i)` weights, with the
    /// tail after `jnrm` zeroed.
    pub shell_weights: RealVec,
    /// Pre-normalization integral `sum rho(i) * ri(i) * dx`.
    pub normalization: Real,
}

/// CRPA Hubbard-parameter accumulation result from `chi_crpa.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenCrpaHubbardSummary {
    /// FEFF final `vch(i) = wscrn(i) * den_CRPA(i,ie)` radial table.
    pub screened_density_potential: RealVec,
    /// Screened Hubbard interaction `U_Hub`, in the same Hartree units written
    /// to `crpa.dat`.
    pub hubbard_u: Real,
    /// FEFF occupation integral `n_occ`.
    pub occupation: Real,
    /// Bare Hubbard interaction `U_Bare`, in the same Hartree units written to
    /// `crpa.dat`.
    pub bare_u: Real,
}

/// Inputs for [`screen_fms_response_slice`].
#[derive(Debug, Clone)]
pub struct ScreenFmsResponseSliceInput<'a> {
    /// FEFF radial grid `ri`.
    pub radii: &'a [Real],
    /// Regular radial solution `pr(:,l)`.
    pub regular_solution: ArrayView1<'a, Complex>,
    /// Irregular radial solution `pn(:,l)`.
    pub irregular_solution: ArrayView1<'a, Complex>,
    /// FEFF cluster Green's function `gtrl(l,ie)`.
    pub cluster_green: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Loucks-grid spacing `dx`.
    pub dx: Real,
    /// Angular momentum `l`.
    pub angular_momentum: usize,
    /// Active radial count, FEFF `ilast`.
    pub active_count: usize,
    /// FMS correction prefix, FEFF `jnrm`.
    pub fms_count: usize,
}

/// Inputs for [`screen_crpa_response_slice`].
#[derive(Debug, Clone)]
pub struct ScreenCrpaResponseSliceInput<'a> {
    /// FEFF radial grid `ri`.
    pub radii: &'a [Real],
    /// Regular radial solution `pr(:,l)`.
    pub regular_solution: ArrayView1<'a, Complex>,
    /// Irregular radial solution `pn(:,l)`.
    pub irregular_solution: ArrayView1<'a, Complex>,
    /// Diagonal CRPA/FMS cluster Green's function `gtrl(l,l,ie)`.
    pub cluster_green: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Loucks-grid spacing `dx`.
    pub dx: Real,
    /// Angular momentum `l` for this response channel.
    pub angular_momentum: usize,
    /// Selected constrained-RPA channel `ll_CRPA`.
    pub crpa_angular_momentum: usize,
    /// Optional CRPA projection window. FEFF's default CRPA path applies this
    /// only when `angular_momentum == crpa_angular_momentum`.
    pub projection_window: Option<ScreenCrpaProjectionWindow>,
    /// Active radial count, FEFF `ilast`.
    pub active_count: usize,
}

/// Port of SCREEN `setri`: build the logarithmic radial grid.
///
/// FEFF stores radial samples as `ri(i) = exp(-x0 + (i-1)*dx)` using 1-based
/// loop bounds. This helper returns the same values in Rust's zero-based
/// [`ndarray::Array1`] layout.
pub fn screen_radial_grid(dx: Real, x0: Real, count: usize) -> Result<RealVec, ScreenError> {
    validate_positive("dx", dx)?;
    validate_finite("x0", x0)?;
    if count == 0 {
        return Err(ScreenError::EmptyRadialGrid);
    }

    Ok(Array1::from_iter(
        (0..count).map(|index| (-x0 + index as Real * dx).exp()),
    ))
}

/// Port of SCREEN `SetEGrid`: exponential grid on the imaginary axis.
///
/// FEFF fills `em(ie) = i * (exp((ne-ie)*dx)-1)`, where
/// `dx = log(emax+1)/(ne-1)`. The resulting table runs from `i*emax`
/// down to zero, matching the reference routine's storage order.
pub fn screen_exponential_energy_grid(
    max_imaginary_energy: Real,
    count: usize,
) -> Result<ComplexVec, ScreenError> {
    validate_positive("max_imaginary_energy", max_imaginary_energy)?;
    validate_count_at_least("energy", count, 2)?;

    let denominator = (count - 1) as Real;
    let dx = (max_imaginary_energy + 1.0).ln() / denominator;
    Ok(Array1::from_iter((1..=count).map(|index_1based| {
        let scaled = (count - index_1based) as Real * dx;
        Complex::new(0.0, scaled.exp() - 1.0)
    })))
}

/// Port of SCREEN `setegi`: rectangular complex-energy contour.
///
/// FEFF starts at `emax + i*ermin`, climbs the imaginary branch to `eimax`,
/// steps across the top edge toward `emin`, descends back to `ermin`, and then
/// reverses the table. Non-positive `ermin` is clamped to `0.05` before any
/// step sizes are computed.
pub fn screen_contour_energy_grid(
    input: ScreenContourEnergyGridInput,
) -> Result<ScreenContourEnergyGrid, ScreenError> {
    validate_finite("min_real_energy", input.min_real_energy)?;
    validate_finite("max_real_energy", input.max_real_energy)?;
    validate_finite("max_imaginary_energy", input.max_imaginary_energy)?;
    validate_finite("min_imaginary_energy", input.min_imaginary_energy)?;
    validate_count_at_least("real_points", input.real_points, 2)?;
    validate_count_at_least("imaginary_points", input.imaginary_points, 2)?;
    validate_count_at_least("max_points", input.max_points, 1)?;
    validate_increasing(
        "min_real_energy",
        input.min_real_energy,
        "max_real_energy",
        input.max_real_energy,
    )?;

    let effective_min_imaginary_energy = if input.min_imaginary_energy <= 0.0 {
        0.05
    } else {
        input.min_imaginary_energy
    };
    validate_increasing(
        "min_imaginary_energy",
        effective_min_imaginary_energy,
        "max_imaginary_energy",
        input.max_imaginary_energy,
    )?;

    let max_iterations = input
        .max_points
        .checked_mul(input.max_points)
        .ok_or(ScreenError::EnergyGridSizeOverflow { name: "max_points" })?;
    let real_step =
        (input.max_real_energy - input.min_real_energy) / (input.real_points - 1) as Real;
    let imaginary_step = Complex::new(
        0.0,
        (input.max_imaginary_energy - effective_min_imaginary_energy)
            / (input.imaginary_points - 1) as Real,
    );

    let mut points = Vec::with_capacity(input.max_points.min(max_iterations));
    points.push(Complex::new(
        input.max_real_energy,
        effective_min_imaginary_energy,
    ));
    let mut accumulated_imaginary = effective_min_imaginary_energy;
    let mut delta = imaginary_step;

    for index_1based in 2..=max_iterations {
        let previous = points.last().copied().ok_or(ScreenError::EmptyEnergyGrid)?;
        if previous.re < input.min_real_energy {
            delta = -imaginary_step;
            if previous.im <= effective_min_imaginary_energy {
                let active_len = if previous.im <= 0.0 {
                    index_1based - 2
                } else {
                    index_1based - 1
                };
                points.truncate(active_len);
                break;
            }
        } else if accumulated_imaginary.abs() >= input.max_imaginary_energy {
            delta = Complex::new(-real_step, 0.0);
            accumulated_imaginary = 0.0;
        }

        accumulated_imaginary += delta.im.abs();
        points.push(previous + delta);
    }

    if points.len() > input.max_points {
        return Err(ScreenError::EnergyGridTooLong {
            required: points.len(),
            available: input.max_points,
        });
    }

    let active_len = points.len();
    let mut energies = Array1::<Complex>::zeros(input.max_points);
    for (index, energy) in points.into_iter().rev().enumerate() {
        energies[index] = energy;
    }

    Ok(ScreenContourEnergyGrid {
        energies,
        active_len,
        effective_min_imaginary_energy,
    })
}

/// Port of SCREEN `getiat`: map a radius to FEFF's 1-based radial index.
///
/// Fortran assigns the floating-point expression to an integer, which truncates
/// toward zero. Returning an `isize` preserves that behavior for callers that
/// need to handle out-of-grid locations explicitly. Values reconstructed from
/// the same logarithmic grid are snapped back to exact integer boundaries when
/// roundoff alone would move them just below the FEFF index.
pub fn screen_radial_index_1based(x0: Real, dx: Real, radius: Real) -> Result<isize, ScreenError> {
    validate_finite("x0", x0)?;
    validate_positive("dx", dx)?;
    validate_positive("radius", radius)?;

    let value = (radius.ln() + x0) / dx + 1.0;
    if value < isize::MIN as Real || value > isize::MAX as Real {
        return Err(ScreenError::RadialIndexOutOfRange { value });
    }
    Ok(feff_truncated_index(value))
}

fn feff_truncated_index(value: Real) -> isize {
    let nearest = value.round();
    let tolerance = 1.0e-12 * nearest.abs().max(1.0);
    if value >= 0.0 && (value - nearest).abs() <= tolerance {
        nearest as isize
    } else {
        value.trunc() as isize
    }
}

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

/// Port the SCREEN/CRPA contour trapezoid energy step.
///
/// `screensub.f90` and `chi_crpa.f90` integrate each `chi0re(:,:,ie)` slice
/// with endpoint half-steps and centered interior steps:
/// `(em(ie+1) - em(ie-1)) / 2`. The `energy_index` argument is zero-based and
/// maps to FEFF's one-based `ie`.
pub fn screen_energy_integration_delta(
    energies: ArrayView1<'_, Complex>,
    energy_index: usize,
) -> Result<Complex, ScreenError> {
    validate_count_at_least("energies", energies.len(), 2)?;
    if energy_index >= energies.len() {
        return Err(ScreenError::EnergyIndexOutOfRange {
            index: energy_index,
            len: energies.len(),
        });
    }
    for &energy in energies {
        validate_finite_complex_input("energy", energy)?;
    }

    let delta = if energy_index == 0 {
        (energies[1] - energies[0]) / 2.0
    } else if energy_index + 1 == energies.len() {
        (energies[energy_index] - energies[energy_index - 1]) / 2.0
    } else {
        (energies[energy_index + 1] - energies[energy_index - 1]) / 2.0
    };
    validate_result_finite_complex("energy_integration_delta", delta)?;
    Ok(delta)
}

/// Accumulate one SCREEN/CRPA response slice into the contour integral.
///
/// FEFF stores only the active upper triangle during the energy loop:
/// `chi0r(ir1,i) += chi0re(ir1,i) * de` for `ir1 <= i`. This helper preserves
/// that convention and leaves the lower triangle from `accumulated` unchanged;
/// use [`screen_symmetrize_response_upper`] before building the response system.
pub fn screen_integrate_response_step(
    accumulated: ArrayView2<'_, Complex>,
    response_at_energy: ArrayView2<'_, Complex>,
    energy_delta: Complex,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape(
        "accumulated_response",
        accumulated.nrows(),
        accumulated.ncols(),
        active_count,
    )?;
    validate_active_matrix_shape(
        "response_at_energy",
        response_at_energy.nrows(),
        response_at_energy.ncols(),
        active_count,
    )?;
    validate_finite_complex_input("energy_delta", energy_delta)?;

    let mut output = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in 0..active_count {
            let value = accumulated[(row, column)];
            validate_finite_complex_matrix("accumulated_response", row, column, value)?;
            output[(row, column)] = value;
        }
    }
    for row in 0..active_count {
        for column in row..active_count {
            let response = response_at_energy[(row, column)];
            validate_finite_complex_matrix("response_at_energy", row, column, response)?;
            let value = output[(row, column)] + response * energy_delta;
            validate_result_finite_complex("integrated_response", value)?;
            output[(row, column)] = value;
        }
    }
    Ok(output)
}

/// Mirror FEFF's stored upper-triangle response matrix.
///
/// The original SCREEN/CRPA routines fill `chi0r(ir1,i)` only for `ir1 <= i`
/// during energy integration, then copy `chi0r(i,ir1)` into the lower triangle
/// before solving. This is a plain symmetric copy, not a Hermitian conjugate.
pub fn screen_symmetrize_response_upper(
    response_upper: ArrayView2<'_, Complex>,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape(
        "response_upper",
        response_upper.nrows(),
        response_upper.ncols(),
        active_count,
    )?;

    let mut output = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in row..active_count {
            let value = response_upper[(row, column)];
            validate_finite_complex_matrix("response_upper", row, column, value)?;
            output[(row, column)] = value;
            output[(column, row)] = value;
        }
    }
    Ok(output)
}

/// Port the CRPA angular-channel density row.
///
/// `chi_crpa.f90` stores
/// `DIMAG((pr*pn + pr**2*gtrl)*ck*4) * (2*l + 1) / pi` in `den_CRPA(:,ie)`
/// for the selected CRPA angular momentum. The regular and irregular radial
/// solutions are passed as `ndarray` views over FEFF's active radial prefix.
pub fn screen_crpa_orbital_density(
    regular_solution: ArrayView1<'_, Complex>,
    irregular_solution: ArrayView1<'_, Complex>,
    cluster_green: Complex,
    wave_number: Complex,
    angular_momentum: usize,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_scale = (2.0 * angular_momentum as Real + 1.0) / std::f64::consts::PI;
    let mut density = Array1::zeros(active_count);
    for index in 0..active_count {
        let regular = regular_solution[index];
        let irregular = irregular_solution[index];
        validate_finite_complex_input("regular_solution", regular)?;
        validate_finite_complex_input("irregular_solution", irregular)?;
        let response =
            (regular * irregular + regular * regular * cluster_green) * wave_number * 4.0;
        validate_result_finite_complex("crpa_orbital_density_response", response)?;
        let value = response.im * angular_scale;
        validate_result_finite("crpa_orbital_density", value)?;
        density[index] = value;
    }
    Ok(density)
}

/// Build one SCREEN atomic response slice.
///
/// In `screensub.f90`, each angular channel adds an upper-triangle contribution
/// `factor * r(m) * r(n) * pr(m)^2 * pn(n)^2`, where
/// `factor = -((2*l + 1) * (2*ck)^2 * dx^2) / (2*pi^2)`. The returned matrix
/// stores the active upper triangle in Fortran order; lower-triangle entries
/// remain zero until [`screen_symmetrize_response_upper`] is applied after
/// energy integration.
pub fn screen_atomic_response_slice(
    radii: &[Real],
    regular_solution: ArrayView1<'_, Complex>,
    irregular_solution: ArrayView1<'_, Complex>,
    wave_number: Complex,
    dx: Real,
    angular_momentum: usize,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor = doubled_wave
        * doubled_wave
        * (-(angular_weight * dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("atomic_response_prefactor", prefactor)?;

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        let row_radius = radii[row];
        let regular = regular_solution[row];
        validate_positive("radius", row_radius)?;
        validate_finite_complex_input("regular_solution", regular)?;
        for column in row..active_count {
            let column_radius = radii[column];
            let irregular = irregular_solution[column];
            validate_positive("radius", column_radius)?;
            validate_finite_complex_input("irregular_solution", irregular)?;
            let value =
                prefactor * row_radius * column_radius * regular * regular * irregular * irregular;
            validate_result_finite_complex("atomic_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Build one SCREEN FMS cluster response correction slice.
///
/// When the FMS cluster contains more than the absorber, `screensub.f90` adds a
/// `1:jnrm` upper-triangle correction to the atomic response slice:
/// `factor*r(m)*r(n)*(2*gtrl*pr(m)^2*pr(n)*pn(n) + gtrl^2*pr(m)^2*pr(n)^2)`.
/// `fms_count` is FEFF `jnrm`; entries outside that prefix remain zero.
pub fn screen_fms_response_slice(
    input: ScreenFmsResponseSliceInput<'_>,
) -> Result<ComplexMat, ScreenError> {
    let ScreenFmsResponseSliceInput {
        radii,
        regular_solution,
        irregular_solution,
        cluster_green,
        wave_number,
        dx,
        angular_momentum,
        active_count,
        fms_count,
    } = input;

    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_count_at_least("fms_count", fms_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    if fms_count > active_count {
        return Err(ScreenError::ActiveCountOutOfRange {
            active_count: fms_count,
            len: active_count,
        });
    }
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor = doubled_wave
        * doubled_wave
        * (-(angular_weight * dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("fms_response_prefactor", prefactor)?;
    let cluster_green_squared = cluster_green * cluster_green;

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..fms_count {
        let row_radius = radii[row];
        let regular_row = regular_solution[row];
        validate_positive("radius", row_radius)?;
        validate_finite_complex_input("regular_solution", regular_row)?;
        let regular_row_squared = regular_row * regular_row;
        for column in row..fms_count {
            let column_radius = radii[column];
            let regular_column = regular_solution[column];
            let irregular_column = irregular_solution[column];
            validate_positive("radius", column_radius)?;
            validate_finite_complex_input("regular_solution", regular_column)?;
            validate_finite_complex_input("irregular_solution", irregular_column)?;
            let cluster_term =
                2.0 * cluster_green * regular_row_squared * regular_column * irregular_column
                    + cluster_green_squared * regular_row_squared * regular_column * regular_column;
            let value = prefactor * row_radius * column_radius * cluster_term;
            validate_result_finite_complex("fms_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Build one CRPA response slice from `chi_crpa.f90`.
///
/// CRPA stores the same upper-triangle `chi0re(m,n)` workspace as SCREEN, but
/// separates the angular prefactor from the base factor and applies a
/// `sin(...)^4` radial projection to the selected constrained channel. Passing
/// `cluster_green = 0` yields the atomic part. A nonzero `cluster_green` adds
/// the diagonal FMS terms used by the CRPA driver.
pub fn screen_crpa_response_slice(
    input: ScreenCrpaResponseSliceInput<'_>,
) -> Result<ComplexMat, ScreenError> {
    let ScreenCrpaResponseSliceInput {
        radii,
        regular_solution,
        irregular_solution,
        cluster_green,
        wave_number,
        dx,
        angular_momentum,
        crpa_angular_momentum,
        projection_window,
        active_count,
    } = input;

    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;
    let projection_window = projection_window.filter(|_| angular_momentum == crpa_angular_momentum);
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

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor =
        doubled_wave * doubled_wave * (-(dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("crpa_response_prefactor", prefactor)?;
    let cluster_green_squared = cluster_green * cluster_green;

    let mut projection_weights = Vec::with_capacity(active_count);
    for &radius in radii.iter().take(active_count) {
        validate_positive("radius", radius)?;
        let weight = match projection_window {
            Some(window) => crpa_response_projection_weight(radius, window)?,
            None => 1.0,
        };
        projection_weights.push(weight);
    }

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        let row_radius = radii[row];
        let regular_row = regular_solution[row];
        validate_finite_complex_input("regular_solution", regular_row)?;
        let row_factor = row_radius * projection_weights[row] * regular_row * regular_row;
        for column in row..active_count {
            let column_radius = radii[column];
            let regular_column = regular_solution[column];
            let irregular_column = irregular_solution[column];
            validate_finite_complex_input("regular_solution", regular_column)?;
            validate_finite_complex_input("irregular_solution", irregular_column)?;
            let response_column = irregular_column * irregular_column
                + 2.0 * cluster_green * regular_column * irregular_column
                + cluster_green_squared * regular_column * regular_column;
            let value = prefactor
                * angular_weight
                * row_factor
                * column_radius
                * projection_weights[column]
                * response_column;
            validate_result_finite_complex("crpa_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

fn crpa_response_projection_weight(
    radius: Real,
    window: ScreenCrpaProjectionWindow,
) -> Result<Real, ScreenError> {
    let clamped = radius.max(window.inner_radius).min(window.outer_radius);
    let scaled = (clamped - window.inner_radius) / (window.outer_radius - window.inner_radius);
    let weight = (scaled * std::f64::consts::FRAC_PI_2).sin().powi(4);
    validate_result_finite("crpa_response_projection_weight", weight)?;
    Ok(weight)
}

/// Port the SCREEN/CRPA response-system matrix setup.
///
/// FEFF builds the real system matrix as `A = I - K * imag(chi0)`, then passes
/// that matrix to LAPACK `dgetrf`/`dgetrs`. The inputs are `ndarray` views so
/// callers can pass full FEFF work arrays and select the active `ilast` prefix.
/// The returned matrix uses Fortran-order storage to preserve the layout that
/// downstream FEFF-compatible linear algebra expects.
pub fn screen_response_system_matrix(
    kernel: ArrayView2<'_, Real>,
    susceptibility: ArrayView2<'_, Complex>,
    active_count: usize,
) -> Result<RealMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape("kernel", kernel.nrows(), kernel.ncols(), active_count)?;
    validate_active_matrix_shape(
        "susceptibility",
        susceptibility.nrows(),
        susceptibility.ncols(),
        active_count,
    )?;

    for row in 0..active_count {
        for column in 0..active_count {
            validate_finite_matrix("kernel", row, column, kernel[(row, column)])?;
            validate_finite_complex_matrix(
                "susceptibility",
                row,
                column,
                susceptibility[(row, column)],
            )?;
        }
    }

    let mut system = Array2::zeros((active_count, active_count).f());
    for index in 0..active_count {
        system[(index, index)] = 1.0;
    }
    for column in 0..active_count {
        for index in 0..active_count {
            let susceptibility_imaginary = susceptibility[(index, column)].im;
            if susceptibility_imaginary == 0.0 {
                continue;
            }
            for row in 0..active_count {
                system[(row, column)] -= kernel[(row, index)] * susceptibility_imaginary;
            }
        }
        for row in 0..active_count {
            validate_result_finite("response_system_matrix", system[(row, column)])?;
        }
    }
    Ok(system)
}

/// Solve FEFF's screened-core-hole response equation.
///
/// This is the matrix-inversion block shared by `SCREEN/screensub.f90` and
/// `CRPA/chi_crpa.f90`: build `A = I - K * imag(chi0)` and solve
/// `A * wscrn = v_ch` with FEFF-compatible real LU factorization. The result is
/// the screened potential vector that FEFF stores back into `wscrn`.
pub fn screen_solve_response_potential(
    kernel: ArrayView2<'_, Real>,
    susceptibility: ArrayView2<'_, Complex>,
    bare_potential: ArrayView1<'_, Real>,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, bare_potential.len())?;
    for &value in bare_potential.iter().take(active_count) {
        validate_finite("bare_potential", value)?;
    }

    let system = screen_response_system_matrix(kernel, susceptibility, active_count)?;
    let rhs = Array1::from_iter(bare_potential.iter().take(active_count).copied());
    let lu = real_lu_factor(system.view())?;
    let solution = real_lu_solve_vector(&lu, rhs.view())?;
    for &value in &solution {
        validate_result_finite("screened_response_potential", value)?;
    }
    Ok(solution)
}

fn validate_active_count(active_count: usize, len: usize) -> Result<(), ScreenError> {
    if active_count > len {
        Err(ScreenError::ActiveCountOutOfRange { active_count, len })
    } else {
        Ok(())
    }
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), ScreenError> {
    if actual < minimum {
        Err(ScreenError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteInput { name, value })
    }
}

fn validate_finite_complex_input(name: &'static str, value: Complex) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexInput {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveInput { name, value })
    }
}

fn validate_result_finite_complex(name: &'static str, value: Complex) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexResult {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_increasing(
    lower_name: &'static str,
    lower: Real,
    upper_name: &'static str,
    upper: Real,
) -> Result<(), ScreenError> {
    if upper > lower {
        Ok(())
    } else {
        Err(ScreenError::NonIncreasingInput {
            lower_name,
            upper_name,
            lower,
            upper,
        })
    }
}

fn validate_result_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteResult { name, value })
    }
}

fn validate_positive_result(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_result_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveResult { name, value })
    }
}

fn validate_active_matrix_shape(
    name: &'static str,
    rows: usize,
    columns: usize,
    active_count: usize,
) -> Result<(), ScreenError> {
    if rows < active_count || columns < active_count {
        Err(ScreenError::MatrixTooSmall {
            name,
            rows,
            columns,
            active_count,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Real,
) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteMatrixInput {
            name,
            row,
            column,
            value,
        })
    }
}

fn validate_finite_complex_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Complex,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name,
            row,
            column,
            real: value.re,
            imaginary: value.im,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ScreenContourEnergyGridInput, ScreenCrpaProjectionWindow, ScreenCrpaResponseSliceInput,
        ScreenError, ScreenFmsResponseSliceInput, screen_atomic_response_slice,
        screen_bare_core_hole_potential, screen_contour_energy_grid, screen_coulomb_kernel_matrix,
        screen_crpa_density_weights, screen_crpa_hubbard_summary, screen_crpa_orbital_density,
        screen_crpa_response_slice, screen_energy_integration_delta,
        screen_exponential_energy_grid, screen_fms_response_slice, screen_integrate_response_step,
        screen_lda_exchange_correlation_kernel, screen_radial_coulomb_potential,
        screen_radial_grid, screen_radial_index_1based, screen_response_system_matrix,
        screen_solve_response_potential, screen_symmetrize_response_upper,
    };
    use ndarray::array;
    use refeff_linalg::LinalgError;

    use crate::Complex;

    #[test]
    fn exponential_energy_grid_matches_feff_setegrid_reference() -> Result<(), ScreenError> {
        let grid = screen_exponential_energy_grid(8.0, 5)?;

        assert_complex_close(grid[0], 0.0, 8.000_000_000_000_002, 1.0e-14);
        assert_complex_close(grid[1], 0.0, 4.196_152_422_706_632, 1.0e-14);
        assert_complex_close(grid[2], 0.0, 2.000_000_000_000_000_4, 1.0e-14);
        assert_complex_close(grid[3], 0.0, 0.732_050_807_568_877_4, 1.0e-14);
        assert_complex_close(grid[4], 0.0, 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn contour_energy_grid_matches_feff_setegi_reference() -> Result<(), ScreenError> {
        let grid = screen_contour_energy_grid(ScreenContourEnergyGridInput {
            min_real_energy: -0.2,
            max_real_energy: 0.4,
            max_imaginary_energy: 0.5,
            min_imaginary_energy: 0.0,
            real_points: 4,
            imaginary_points: 4,
            max_points: 20,
        })?;

        assert_eq!(grid.active_len, 10);
        assert_close(grid.effective_min_imaginary_energy, 0.05, 1.0e-15);
        assert_complex_close(grid.energies[0], -0.2, 0.05, 1.0e-14);
        assert_complex_close(grid.energies[1], -0.2, 0.2, 1.0e-14);
        assert_complex_close(grid.energies[2], -0.2, 0.35, 1.0e-14);
        assert_complex_close(grid.energies[3], -0.2, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[4], -5.551_115_123_125_783e-17, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[5], 0.2, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[6], 0.4, 0.5, 1.0e-14);
        assert_complex_close(grid.energies[7], 0.4, 0.35, 1.0e-14);
        assert_complex_close(grid.energies[8], 0.4, 0.2, 1.0e-14);
        assert_complex_close(grid.energies[9], 0.4, 0.05, 1.0e-14);
        assert_complex_close(grid.energies[10], 0.0, 0.0, 1.0e-15);
        Ok(())
    }

    #[test]
    fn radial_grid_matches_feff_setri_reference() -> Result<(), ScreenError> {
        let grid = screen_radial_grid(0.05, 8.8, 5)?;

        assert_close(grid[0], 0.000_150_733_075_095_476_5, 1.0e-15);
        assert_close(grid[1], 0.000_158_461_325_115_751_26, 1.0e-15);
        assert_close(grid[2], 0.000_166_585_810_987_633_24, 1.0e-15);
        assert_close(grid[3], 0.000_175_126_848_157_658_42, 1.0e-15);
        assert_close(grid[4], 0.000_184_105_793_667_578_87, 1.0e-15);
        assert_eq!(screen_radial_index_1based(8.8, 0.05, grid[2])?, 3);
        assert_eq!(screen_radial_index_1based(8.8, 0.05, 1.0)?, 177);
        assert_eq!(screen_radial_index_1based(0.0, 1.0, 0.01)?, -3);
        Ok(())
    }

    #[test]
    fn lda_exchange_correlation_kernel_matches_feff_ldafxc_reference() -> Result<(), ScreenError> {
        let radii = [0.5, 0.75, 1.0, 1.5, 2.0];
        let density = [0.04, 0.10, 0.0, -1.0, 0.25];

        let full = screen_lda_exchange_correlation_kernel(&radii, &density, 0, radii.len())?;
        assert_close(full[0], -16.919_199_214_545_813, 1.0e-13);
        assert_close(full[1], -3.960_989_192_391_738_6, 1.0e-13);
        assert_close(full[2], 0.0, 1.0e-15);
        assert_close(full[3], 0.0, 1.0e-15);
        assert_close(full[4], -0.294_609_719_384_913, 1.0e-13);

        let exchange_only =
            screen_lda_exchange_correlation_kernel(&radii, &density, 2, radii.len())?;
        assert_close(exchange_only[0], -14.488_412_060_289_518, 1.0e-13);
        assert_close(exchange_only[1], -3.495_786_749_594_309_6, 1.0e-13);
        assert_close(exchange_only[4], -0.266_878_831_976_939_35, 1.0e-13);
        Ok(())
    }

    #[test]
    fn coulomb_kernel_matrix_matches_feff_response_setup_reference() -> Result<(), ScreenError> {
        let radii = [0.5, 1.0, 2.0];
        let local_kernel = [0.1, -0.2, 0.0];
        let matrix = screen_coulomb_kernel_matrix(&radii, radii.len(), Some(&local_kernel))?;
        let pi = std::f64::consts::PI;

        assert_close(matrix[(0, 0)], 8.4 * pi, 1.0e-14);
        assert_close(matrix[(0, 1)], 4.0 * pi, 1.0e-14);
        assert_close(matrix[(1, 0)], 4.0 * pi, 1.0e-14);
        assert_close(matrix[(0, 2)], 2.0 * pi, 1.0e-14);
        assert_close(matrix[(2, 0)], 2.0 * pi, 1.0e-14);
        assert_close(matrix[(1, 1)], 3.2 * pi, 1.0e-14);
        assert_close(matrix[(1, 2)], 2.0 * pi, 1.0e-14);
        assert_close(matrix[(2, 1)], 2.0 * pi, 1.0e-14);
        assert_close(matrix[(2, 2)], 2.0 * pi, 1.0e-14);
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                assert_close(matrix[(row, column)], matrix[(column, row)], 1.0e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn bare_core_hole_potential_matches_feff_loop_reference() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0, 4.0];
        let large = [1.0, 0.5, 0.25];
        let small = [0.0, 0.25, 0.0];
        let potential = screen_bare_core_hole_potential(&radii, &large, &small, 0.1, radii.len())?;

        assert_close(potential[0], 0.1375, 1.0e-14);
        assert_close(potential[1], 0.0875, 1.0e-14);
        assert_close(potential[2], 0.046875, 1.0e-14);
        Ok(())
    }

    #[test]
    fn radial_coulomb_potential_matches_feff_shell_weight_loop() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0, 3.0];
        let shell_weights = [0.5, 0.5, 0.0];
        let potential = screen_radial_coulomb_potential(&radii, &shell_weights, radii.len())?;

        assert_close(potential[0], 0.75, 1.0e-14);
        assert_close(potential[1], 0.5, 1.0e-14);
        assert_close(potential[2], 1.0 / 3.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn crpa_density_weights_match_feff_normalization_reference() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0, 3.0];
        let density = [2.0, 4.0, 6.0];
        let weights = screen_crpa_density_weights(&radii, &density, 0.1, radii.len(), 2, None)?;

        assert_close(weights.normalization, 2.8, 1.0e-14);
        assert_close(weights.normalized_density[0], 5.0 / 7.0, 1.0e-14);
        assert_close(weights.normalized_density[1], 10.0 / 7.0, 1.0e-14);
        assert_close(weights.normalized_density[2], 15.0 / 7.0, 1.0e-14);
        assert_close(weights.shell_weights[0], 1.0 / 14.0, 1.0e-14);
        assert_close(weights.shell_weights[1], 2.0 / 7.0, 1.0e-14);
        assert_close(weights.shell_weights[2], 0.0, 1.0e-14);

        let projected = screen_crpa_density_weights(
            &radii,
            &density,
            0.1,
            radii.len(),
            radii.len(),
            Some(ScreenCrpaProjectionWindow {
                inner_radius: 1.0,
                outer_radius: 3.0,
            }),
        )?;
        assert_close(projected.normalization, 0.4, 1.0e-14);
        assert_close(projected.normalized_density[0], 5.0, 1.0e-14);
        assert_close(projected.normalized_density[1], 2.5, 1.0e-14);
        assert_close(projected.normalized_density[2], 0.0, 1.0e-14);
        assert_close(projected.shell_weights[0], 0.5, 1.0e-14);
        assert_close(projected.shell_weights[1], 0.5, 1.0e-14);
        assert_close(projected.shell_weights[2], 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn crpa_hubbard_summary_matches_feff_accumulation_reference() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0, 3.0];
        let screened = [0.5, 1.0, 1.5];
        let bare = [2.0, 1.0, 0.5];
        let total_density = [5.0 / 7.0, 10.0 / 7.0, 15.0 / 7.0];
        let orbital_density = [0.2, 0.3, 0.4];

        let summary = screen_crpa_hubbard_summary(
            &radii,
            &screened,
            &bare,
            &total_density,
            &orbital_density,
            0.1,
            radii.len(),
        )?;

        assert_close(summary.screened_density_potential[0], 0.1, 1.0e-14);
        assert_close(summary.screened_density_potential[1], 0.3, 1.0e-14);
        assert_close(summary.screened_density_potential[2], 0.6, 1.0e-14);
        assert_close(summary.hubbard_u, 9.0 / 7.0, 1.0e-14);
        assert_close(summary.occupation, 1.0, 1.0e-14);
        assert_close(summary.bare_u, 0.75, 1.0e-14);
        Ok(())
    }

    #[test]
    fn energy_integration_delta_matches_feff_trapezoid_reference() -> Result<(), ScreenError> {
        let energies = array![
            Complex::new(0.0, 0.1),
            Complex::new(1.0, 0.2),
            Complex::new(3.0, 0.5),
            Complex::new(6.0, 1.1)
        ];

        assert_complex_close(
            screen_energy_integration_delta(energies.view(), 0)?,
            0.5,
            0.05,
            1.0e-14,
        );
        assert_complex_close(
            screen_energy_integration_delta(energies.view(), 1)?,
            1.5,
            0.2,
            1.0e-14,
        );
        assert_complex_close(
            screen_energy_integration_delta(energies.view(), 2)?,
            2.5,
            0.45,
            1.0e-14,
        );
        assert_complex_close(
            screen_energy_integration_delta(energies.view(), 3)?,
            1.5,
            0.3,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn response_integration_and_symmetry_match_feff_upper_triangle() -> Result<(), ScreenError> {
        let accumulated = array![
            [Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)],
            [Complex::new(9.0, 0.0), Complex::new(4.0, 1.0)]
        ];
        let response_at_energy = array![
            [Complex::new(0.5, 1.0), Complex::new(-1.0, 0.5)],
            [Complex::new(3.0, 3.0), Complex::new(2.0, -1.0)]
        ];
        let integrated = screen_integrate_response_step(
            accumulated.view(),
            response_at_energy.view(),
            Complex::new(0.2, 0.1),
            2,
        )?;

        assert_eq!(integrated.strides(), &[1, 2]);
        assert_complex_close(integrated[(0, 0)], 1.0, 1.25, 1.0e-14);
        assert_complex_close(integrated[(0, 1)], 1.75, 0.0, 1.0e-14);
        assert_complex_close(integrated[(1, 0)], 9.0, 0.0, 1.0e-14);
        assert_complex_close(integrated[(1, 1)], 4.5, 1.0, 1.0e-14);

        let symmetric = screen_symmetrize_response_upper(integrated.view(), 2)?;
        assert_complex_close(symmetric[(0, 1)], 1.75, 0.0, 1.0e-14);
        assert_complex_close(symmetric[(1, 0)], 1.75, 0.0, 1.0e-14);
        assert_complex_close(symmetric[(1, 1)], 4.5, 1.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn crpa_orbital_density_matches_feff_density_row_reference() -> Result<(), ScreenError> {
        let regular = array![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.25),
            Complex::new(0.0, 1.0)
        ];
        let irregular = array![
            Complex::new(0.2, 0.1),
            Complex::new(0.4, -0.2),
            Complex::new(-0.3, 0.2)
        ];

        let density = screen_crpa_orbital_density(
            regular.view(),
            irregular.view(),
            Complex::new(0.1, 0.2),
            Complex::new(0.7, 0.3),
            2,
            regular.len(),
        )?;

        assert_close(density[0], 1.909_859_317_102_744_5, 1.0e-14);
        assert_close(density[1], 0.696_302_876_027_042_2, 1.0e-14);
        assert_close(density[2], -2.801_126_998_417_358, 1.0e-14);
        Ok(())
    }

    #[test]
    fn atomic_response_slice_matches_feff_upper_triangle_reference() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0];
        let regular = array![Complex::new(1.0, 0.0), Complex::new(0.5, 0.25)];
        let irregular = array![Complex::new(0.2, 0.1), Complex::new(-0.3, 0.2)];

        let response = screen_atomic_response_slice(
            &radii,
            regular.view(),
            irregular.view(),
            Complex::new(0.7, 0.3),
            0.1,
            1,
            radii.len(),
        )?;

        assert_eq!(response.strides(), &[1, 2]);
        assert_complex_close(
            response[(0, 0)],
            2.918_050_088_899_328_5e-5,
            -0.000_173_867_151_130_251_67,
            1.0e-14,
        );
        assert_complex_close(
            response[(0, 1)],
            -0.000_855_961_359_410_469_6,
            0.000_328_280_635_001_174_44,
            1.0e-14,
        );
        assert_complex_close(response[(1, 0)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(
            response[(1, 1)],
            -0.000_485_125_827_279_513_3,
            -0.000_304_875_441_579_794_35,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn fms_response_slice_matches_feff_cluster_reference() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0, 3.0];
        let regular = array![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.25),
            Complex::new(0.25, -0.1)
        ];
        let irregular = array![
            Complex::new(0.2, 0.1),
            Complex::new(-0.3, 0.2),
            Complex::new(0.4, 0.05)
        ];

        let response = screen_fms_response_slice(ScreenFmsResponseSliceInput {
            radii: &radii,
            regular_solution: regular.view(),
            irregular_solution: irregular.view(),
            cluster_green: Complex::new(0.1, 0.2),
            wave_number: Complex::new(0.7, 0.3),
            dx: 0.1,
            angular_momentum: 1,
            active_count: radii.len(),
            fms_count: 2,
        })?;

        assert_eq!(response.strides(), &[1, 3]);
        assert_complex_close(
            response[(0, 0)],
            0.000_430_412_388_112_651,
            -0.000_263_840_362_204_647_56,
            1.0e-14,
        );
        assert_complex_close(
            response[(0, 1)],
            -0.000_063_832_345_694_672_8,
            0.000_699_876_076_009_448_3,
            1.0e-14,
        );
        assert_complex_close(response[(0, 2)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(response[(1, 0)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(
            response[(1, 1)],
            -0.000_373_875_167_640_226_43,
            0.000_230_537_355_656_206_72,
            1.0e-14,
        );
        assert_complex_close(response[(2, 2)], 0.0, 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn crpa_response_slice_matches_feff_projected_reference() -> Result<(), ScreenError> {
        let radii = [1.0, 2.0, 3.0];
        let regular = array![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.25),
            Complex::new(0.25, -0.1)
        ];
        let irregular = array![
            Complex::new(0.2, 0.1),
            Complex::new(-0.3, 0.2),
            Complex::new(0.4, 0.05)
        ];

        let response = screen_crpa_response_slice(ScreenCrpaResponseSliceInput {
            radii: &radii,
            regular_solution: regular.view(),
            irregular_solution: irregular.view(),
            cluster_green: Complex::new(0.1, 0.2),
            wave_number: Complex::new(0.7, 0.3),
            dx: 0.1,
            angular_momentum: 1,
            crpa_angular_momentum: 1,
            projection_window: Some(ScreenCrpaProjectionWindow {
                inner_radius: 1.0,
                outer_radius: 3.0,
            }),
            active_count: radii.len(),
        })?;

        assert_eq!(response.strides(), &[1, 3]);
        assert_complex_close(response[(0, 0)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(response[(0, 1)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(response[(0, 2)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(response[(1, 0)], 0.0, 0.0, 1.0e-14);
        assert_complex_close(
            response[(1, 1)],
            -0.000_053_687_562_182_483_7,
            -0.000_004_646_130_370_224_231,
            1.0e-14,
        );
        assert_complex_close(
            response[(1, 2)],
            0.000_182_520_613_470_705_1,
            -0.000_287_665_880_223_793_86,
            1.0e-14,
        );
        assert_complex_close(
            response[(2, 2)],
            -0.000_427_456_676_939_791_8,
            -0.000_205_369_424_409_379_63,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn response_system_matrix_matches_feff_inversion_setup_reference() -> Result<(), ScreenError> {
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex::new(1.0, 0.1), Complex::new(2.0, 0.2)],
            [Complex::new(3.0, 0.3), Complex::new(4.0, 0.05)]
        ];

        let system = screen_response_system_matrix(kernel.view(), susceptibility.view(), 2)?;

        assert_eq!(system.strides(), &[1, 2]);
        assert_close(system[(0, 0)], 0.65, 1.0e-14);
        assert_close(system[(0, 1)], -0.425, 1.0e-14);
        assert_close(system[(1, 0)], -0.35, 1.0e-14);
        assert_close(system[(1, 1)], 0.85, 1.0e-14);
        Ok(())
    }

    #[test]
    fn screened_response_potential_matches_feff_dgetrs_reference() -> Result<(), ScreenError> {
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex::new(1.0, 0.1), Complex::new(2.0, 0.2)],
            [Complex::new(3.0, 0.3), Complex::new(4.0, 0.05)]
        ];
        let bare = array![0.8, 0.2];

        let screened =
            screen_solve_response_potential(kernel.view(), susceptibility.view(), bare.view(), 2)?;

        assert_close(screened[0], 612.0 / 323.0, 1.0e-14);
        assert_close(screened[1], 328.0 / 323.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn screen_helpers_reject_invalid_inputs() {
        assert!(matches!(
            screen_radial_grid(0.0, 8.8, 5),
            Err(ScreenError::NonPositiveInput { name: "dx", .. })
        ));
        assert!(matches!(
            screen_radial_grid(0.05, 8.8, 0),
            Err(ScreenError::EmptyRadialGrid)
        ));
        assert!(matches!(
            screen_exponential_energy_grid(8.0, 1),
            Err(ScreenError::CountTooSmall { name: "energy", .. })
        ));
        assert!(matches!(
            screen_radial_index_1based(8.8, 0.05, -1.0),
            Err(ScreenError::NonPositiveInput { name: "radius", .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[1.0], &[0.1], 0, 2),
            Err(ScreenError::ActiveCountOutOfRange { .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[0.0], &[0.1], 0, 1),
            Err(ScreenError::NonPositiveInput { name: "radius", .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[1.0], &[f64::NAN], 0, 1),
            Err(ScreenError::NonFiniteInput {
                name: "electron_density",
                ..
            })
        ));
        assert!(matches!(
            screen_coulomb_kernel_matrix(&[1.0], 2, None),
            Err(ScreenError::ActiveCountOutOfRange { .. })
        ));
        assert!(matches!(
            screen_coulomb_kernel_matrix(&[1.0], 1, Some(&[f64::NAN])),
            Err(ScreenError::NonFiniteInput {
                name: "local_kernel",
                ..
            })
        ));
        assert!(matches!(
            screen_bare_core_hole_potential(&[1.0], &[1.0], &[0.0], 0.0, 1),
            Err(ScreenError::NonPositiveInput { name: "dx", .. })
        ));
        assert!(matches!(
            screen_bare_core_hole_potential(&[1.0], &[f64::INFINITY], &[0.0], 0.1, 1),
            Err(ScreenError::NonFiniteInput {
                name: "large_component",
                ..
            })
        ));
        assert!(matches!(
            screen_radial_coulomb_potential(&[1.0], &[f64::NAN], 1),
            Err(ScreenError::NonFiniteInput {
                name: "shell_weight",
                ..
            })
        ));
        assert!(matches!(
            screen_crpa_density_weights(&[1.0], &[0.0], 0.1, 1, 1, None),
            Err(ScreenError::NonPositiveResult {
                name: "crpa_density_normalization",
                ..
            })
        ));
        assert!(matches!(
            screen_crpa_hubbard_summary(&[1.0], &[1.0], &[1.0], &[1.0], &[f64::NAN], 0.1, 1,),
            Err(ScreenError::NonFiniteInput {
                name: "orbital_density",
                ..
            })
        ));
        assert!(matches!(
            screen_atomic_response_slice(
                &[1.0],
                array![Complex::new(1.0, 0.0)].view(),
                array![Complex::new(1.0, 0.0)].view(),
                Complex::new(f64::NAN, 0.0),
                0.1,
                0,
                1,
            ),
            Err(ScreenError::NonFiniteComplexInput {
                name: "wave_number",
                ..
            })
        ));
        assert!(matches!(
            screen_fms_response_slice(ScreenFmsResponseSliceInput {
                radii: &[1.0],
                regular_solution: array![Complex::new(1.0, 0.0)].view(),
                irregular_solution: array![Complex::new(1.0, 0.0)].view(),
                cluster_green: Complex::new(0.0, 0.0),
                wave_number: Complex::new(1.0, 0.0),
                dx: 0.1,
                angular_momentum: 0,
                active_count: 1,
                fms_count: 2,
            }),
            Err(ScreenError::ActiveCountOutOfRange {
                active_count: 2,
                len: 1
            })
        ));
        assert!(matches!(
            screen_crpa_response_slice(ScreenCrpaResponseSliceInput {
                radii: &[1.0],
                regular_solution: array![Complex::new(1.0, 0.0)].view(),
                irregular_solution: array![Complex::new(1.0, 0.0)].view(),
                cluster_green: Complex::new(0.0, 0.0),
                wave_number: Complex::new(1.0, 0.0),
                dx: 0.1,
                angular_momentum: 0,
                crpa_angular_momentum: 0,
                projection_window: Some(ScreenCrpaProjectionWindow {
                    inner_radius: 2.0,
                    outer_radius: 1.0,
                }),
                active_count: 1,
            }),
            Err(ScreenError::NonIncreasingInput {
                upper_name: "projection_outer_radius",
                ..
            })
        ));
        let two_energies = array![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        assert!(matches!(
            screen_energy_integration_delta(two_energies.view(), 2),
            Err(ScreenError::EnergyIndexOutOfRange { index: 2, len: 2 })
        ));
        let regular = array![Complex::new(f64::NAN, 0.0)];
        let irregular = array![Complex::new(1.0, 0.0)];
        assert!(matches!(
            screen_crpa_orbital_density(
                regular.view(),
                irregular.view(),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                0,
                1,
            ),
            Err(ScreenError::NonFiniteComplexInput {
                name: "regular_solution",
                ..
            })
        ));
        assert!(matches!(
            screen_crpa_density_weights(
                &[1.0],
                &[1.0],
                0.1,
                1,
                1,
                Some(ScreenCrpaProjectionWindow {
                    inner_radius: 2.0,
                    outer_radius: 1.0,
                }),
            ),
            Err(ScreenError::NonIncreasingInput {
                upper_name: "projection_outer_radius",
                ..
            })
        ));
        let kernel = array![[1.0]];
        let susceptibility = array![[Complex::new(0.0, 0.0)]];
        assert!(matches!(
            screen_response_system_matrix(kernel.view(), susceptibility.view(), 2),
            Err(ScreenError::MatrixTooSmall { name: "kernel", .. })
        ));
        let bad_susceptibility = array![[Complex::new(f64::NAN, 0.0)]];
        assert!(matches!(
            screen_response_system_matrix(kernel.view(), bad_susceptibility.view(), 1),
            Err(ScreenError::NonFiniteComplexMatrixInput {
                name: "susceptibility",
                row: 0,
                column: 0,
                ..
            })
        ));
        let bare = array![f64::NAN];
        assert!(matches!(
            screen_solve_response_potential(kernel.view(), susceptibility.view(), bare.view(), 1),
            Err(ScreenError::NonFiniteInput {
                name: "bare_potential",
                ..
            })
        ));
        let singular_susceptibility = array![
            [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]
        ];
        let identity_kernel = array![[1.0, 0.0], [0.0, 1.0]];
        let singular_rhs = array![1.0, 1.0];
        assert!(matches!(
            screen_solve_response_potential(
                identity_kernel.view(),
                singular_susceptibility.view(),
                singular_rhs.view(),
                2
            ),
            Err(ScreenError::Linalg(LinalgError::SingularMatrix {
                pivot: 0
            }))
        ));
        assert!(matches!(
            screen_contour_energy_grid(ScreenContourEnergyGridInput {
                min_real_energy: 0.4,
                max_real_energy: 0.4,
                max_imaginary_energy: 0.5,
                min_imaginary_energy: 0.05,
                real_points: 4,
                imaginary_points: 4,
                max_points: 20,
            }),
            Err(ScreenError::NonIncreasingInput {
                upper_name: "max_real_energy",
                ..
            })
        ));
        assert!(matches!(
            screen_contour_energy_grid(ScreenContourEnergyGridInput {
                min_real_energy: -0.2,
                max_real_energy: 0.4,
                max_imaginary_energy: 0.04,
                min_imaginary_energy: 0.0,
                real_points: 4,
                imaginary_points: 4,
                max_points: 20,
            }),
            Err(ScreenError::NonIncreasingInput {
                upper_name: "max_imaginary_energy",
                ..
            })
        ));
        assert!(matches!(
            screen_contour_energy_grid(ScreenContourEnergyGridInput {
                min_real_energy: -0.2,
                max_real_energy: 0.4,
                max_imaginary_energy: 0.5,
                min_imaginary_energy: 0.0,
                real_points: 4,
                imaginary_points: 4,
                max_points: 8,
            }),
            Err(ScreenError::EnergyGridTooLong {
                required: 10,
                available: 8
            })
        ));
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn assert_complex_close(
        actual: crate::Complex,
        expected_re: f64,
        expected_im: f64,
        tolerance: f64,
    ) {
        assert_close(actual.re, expected_re, tolerance);
        assert_close(actual.im, expected_im, tolerance);
    }
}
