//! FEFF common energy and radial-grid helpers.
//!
//! These functions port the small common routines `getxk.f90`, `xx.f90`,
//! `m_ifuns.f90`, radial resampling helpers from `COMMON/`, and the ATOM
//! `FixAtomicQuantities` resampling helper from `ATOM/scfdat.f90`. FEFF uses
//! a 1-based logarithmic radial grid with `x = -8.8 + (j - 1) * delta` and
//! `r = exp(x)`.

use std::f64::consts::PI;

use crate::interpolation::terp;
use crate::quadrature::somm2;
use crate::vector::distance_between;
use crate::{Complex, Real};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex32;
use refeff_linalg::{Complex32Lu, LinalgError, complex32_lu_factor};

/// Default FEFF logarithmic radial-grid spacing.
pub const LOUCKS_DELTA: Real = 0.05;

/// Offset used by FEFF's Loucks radial grid.
pub const LOUCKS_X_OFFSET: Real = 8.8;

/// FEFF Hartree constant in eV, from `COMMON/m_constants.f90`.
pub const FEFF_HARTREE_EV: Real = 27.211_396;

/// FEFF Fermi-momentum factor `(9*pi/4)^(1/3)`, from `COMMON/m_constants.f90`.
pub const FEFF_FERMI_MOMENTUM_FACTOR: Real = 1.919_158_292_677_512_8;

mod resample;
mod types;

pub use resample::*;
pub use types::*;

const SPINOR_ZERO_THRESHOLD: Real = 1.0e-11;
const SUMAX_WIGNER_SEITZ_RADIUS: Real = 15.0;
const SUMAX_LITERAL_DELTA: Real = 0.05_f32 as Real;
const SUMAX_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const SIDX_DENSITY_CUTOFF: Real = 1.0e-5;
const FRNRM_DENSITY_POINTS: usize = 251;
const FRNRM_NRPTX: usize = 1251;
const FRNRM_LITERAL_DELTA: Real = 0.05_f32 as Real;
const FRNRM_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const FRNRM_CORRECTION_THRESHOLD: Real = 0.0001_f32 as Real;
const MOVRLP_NOVP: usize = 50;

/// Convert energy in Hartrees to FEFF's signed photoelectron wave number.
///
/// This ports `getxk`: `sqrt(2E)` above the edge and `-sqrt(-2E)` below it.
#[must_use]
pub fn wave_number_from_hartree(energy: Real) -> Real {
    let magnitude = (2.0 * energy).abs().sqrt();
    if energy < 0.0 { -magnitude } else { magnitude }
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a 1-based index.
#[must_use]
pub fn loucks_x(index_1based: usize) -> Real {
    radial_x(index_1based, LOUCKS_DELTA)
}

/// Return the radial coordinate for a 1-based Loucks grid index.
#[must_use]
pub fn loucks_radius(index_1based: usize) -> Real {
    loucks_x(index_1based).exp()
}

/// Return the 1-based Loucks grid index immediately below `radius`.
pub fn loucks_index_below(radius: Real) -> Result<usize, GridError> {
    radial_index_below(radius, LOUCKS_DELTA)
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a custom spacing.
#[must_use]
pub fn radial_x(index_1based: usize, delta: Real) -> Real {
    -LOUCKS_X_OFFSET + (index_1based as Real - 1.0) * delta
}

/// Return the radial coordinate for a custom logarithmic spacing.
#[must_use]
pub fn radial_radius(index_1based: usize, delta: Real) -> Real {
    radial_x(index_1based, delta).exp()
}

/// Return the 1-based grid index immediately below `radius` for a custom spacing.
pub fn radial_index_below(radius: Real, delta: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(GridError::InvalidDelta { delta });
    }
    let index = ((radius.ln() + LOUCKS_X_OFFSET) / delta + 1.0).trunc();
    Ok(index as usize)
}

/// Integrate a radial density into a Coulomb potential using FEFF `potslw`.
///
/// This ports `ATOM/potslw.f90`, a four-point integration stencil used by the
/// potential module's Coulomb update. FEFF only defines values through `np`; the
/// Rust result preserves the caller's grid length and zero-fills the inactive
/// tail.
pub fn coulomb_potential_slw(
    input: CoulombPotentialSlwInput<'_>,
) -> Result<CoulombPotentialSlw, GridError> {
    validate_delta(input.delta)?;

    let density_len = input.density.len();
    let radii_len = input.radii.len();
    if density_len != radii_len {
        return Err(GridError::CoulombLengthMismatch {
            density_len,
            radii_len,
        });
    }
    validate_positive_grid_length("density", density_len)?;
    validate_source_len_at_least("active", input.active_len, 3)?;
    ensure_source_length("density", input.active_len, density_len)?;
    validate_component_prefix_values("density", input.density, input.active_len)?;
    validate_positive_radii(input.radii, input.active_len)?;

    let mut potential = Array1::<Real>::zeros(density_len);
    let mut work = Array1::<Real>::zeros(density_len);
    let scale = input.delta / 24.0;
    for index in 0..input.active_len {
        potential[index] = input.density[index] * input.radii[index];
    }

    let grid_ratio = input.delta.exp();
    let grid_ratio2 = grid_ratio * grid_ratio;
    work[1] = input.radii[0] * (input.density[1] - input.density[0] * grid_ratio2)
        / (12.0 * (grid_ratio - 1.0));
    work[0] = potential[0] / 3.0 - work[1] / grid_ratio2;
    work[1] = potential[1] / 3.0 - work[1] * grid_ratio2;

    let last_inner = input.active_len - 2;
    for index in 2..=last_inner {
        work[index] = work[index - 1]
            + scale
                * (13.0 * (potential[index] + potential[index - 1])
                    - (potential[index - 2] + potential[index + 1]));
    }

    work[input.active_len - 1] = work[last_inner];
    potential[last_inner] = work[last_inner];
    potential[input.active_len - 1] = work[last_inner];
    for fortran_i in 3..=last_inner + 1 {
        let index = input.active_len - fortran_i;
        potential[index] = potential[index + 1] / grid_ratio
            + scale
                * (13.0 * (work[index + 1] / grid_ratio + work[index])
                    - (work[index + 2] / grid_ratio2 + work[index - 1] * grid_ratio));
    }
    potential[0] = potential[2] / grid_ratio2
        + input.delta * (work[0] + 4.0 * work[1] / grid_ratio + work[2] / grid_ratio2) / 3.0;

    potential
        .iter_mut()
        .zip(input.radii.iter())
        .take(input.active_len)
        .for_each(|(potential, radius)| *potential /= radius);

    Ok(CoulombPotentialSlw {
        potential,
        active_len: input.active_len,
    })
}

/// Build FEFF's SCMT complex-energy contour from `ecv` to `xmu`.
///
/// This ports `POT/grids.f90`. FEFF first creates a short vertical line above
/// `ecv`, then a real-axis bridge that retains the initial imaginary part, and
/// finally a descending set of points above `xmu`. The Rust version preserves
/// FEFF's count and rounding rules while validating that the caller-provided
/// table sizes are large enough.
pub fn scmt_energy_grid(input: ScmtEnergyGridInput) -> Result<ScmtEnergyGrid, GridError> {
    validate_finite_scalar("core_valence_energy", input.core_valence_energy)?;
    validate_finite_scalar("fermi_energy", input.fermi_energy)?;
    let energy_span = input.fermi_energy - input.core_valence_energy;
    validate_finite_scalar("energy_span", energy_span)?;
    validate_positive_grid_length("energy", input.max_points)?;
    validate_positive_grid_length("step", input.step_count)?;

    let lower_imaginary_count = input
        .step_count
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "step" })?
        / 2;
    let upper_imaginary_count = input.step_count - 1;
    let minimum_points = lower_imaginary_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(upper_imaginary_count))
        .ok_or(GridError::OutputGridTooShort {
            required: usize::MAX,
            available: input.max_points,
        })?;
    if input.max_points < minimum_points {
        return Err(GridError::OutputGridTooShort {
            required: minimum_points,
            available: input.max_points,
        });
    }

    let real_axis_max = input.max_points - lower_imaginary_count - upper_imaginary_count;
    let minimum_imaginary = 0.05 / FEFF_HARTREE_EV;
    let mut energies = Array1::<Complex>::zeros(input.max_points);
    let mut steps = Array1::<Real>::zeros(input.step_count);

    for index in 1..=lower_imaginary_count {
        let imaginary = minimum_imaginary * square_index_as_real("step", index)?;
        energies[index - 1] = Complex::new(input.core_valence_energy, imaginary);
    }
    steps[input.step_count - 1] = energies[lower_imaginary_count - 1].im / 4.0;

    let bridge_step_guess = energies[lower_imaginary_count - 1].im / 4.0;
    let rounded_bridge_points = (energy_span / bridge_step_guess).round();
    let mut real_axis_count = if rounded_bridge_points <= 0.0 {
        0
    } else if rounded_bridge_points >= real_axis_max as Real {
        real_axis_max
    } else {
        rounded_bridge_points as usize
    };
    if real_axis_count < lower_imaginary_count {
        real_axis_count = lower_imaginary_count;
    }

    let real_step = energy_span / real_axis_count as Real;
    for index in lower_imaginary_count + 1..=lower_imaginary_count + real_axis_count {
        energies[index - 1] = energies[index - 2] + Complex::new(real_step, 0.0);
    }

    let active_len = lower_imaginary_count + real_axis_count + upper_imaginary_count;
    for index in 1..=upper_imaginary_count {
        let imaginary = minimum_imaginary * square_index_as_real("step", index + 1)? / 4.0;
        steps[index - 1] = imaginary / 4.0;
        energies[active_len - index] = Complex::new(input.fermi_energy, imaginary);
    }

    Ok(ScmtEnergyGrid {
        energies,
        steps,
        active_len,
        lower_imaginary_count,
        real_axis_count,
        upper_imaginary_count,
    })
}

/// Add one FEFF `sumax` spherical overlap contribution on the Loucks grid.
///
/// This ports `POT/sumax.f90`, used by FEFF's overlapped potential/density
/// setup. The input and accumulated arrays use the fixed Loucks spacing
/// `delta = 0.05`; only grid points through the neighbor distance are updated,
/// matching FEFF's `jtop = ii(rn)` behavior.
pub fn sum_loucks_spherical_overlap(
    input: LoucksSphericalOverlapInput<'_>,
) -> Result<LoucksSphericalOverlap, GridError> {
    if !(input.neighbor_distance.is_finite() && input.neighbor_distance > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.neighbor_distance,
        });
    }
    validate_finite_scalar("multiplicity", input.multiplicity)?;

    let source_len = input.source.len();
    let accumulated_len = input.accumulated.len();
    if source_len != accumulated_len {
        return Err(GridError::OverlapLengthMismatch {
            source_len,
            accumulated_len,
        });
    }
    validate_positive_grid_length("source", source_len)?;
    validate_component_values("source", input.source)?;
    validate_component_values("accumulated", input.accumulated)?;

    let cutoff_index = loucks_index_below(SUMAX_WIGNER_SEITZ_RADIUS)?;
    let active_len = loucks_index_below(input.neighbor_distance)?;
    ensure_source_length("source", cutoff_index, source_len)?;
    ensure_source_length("accumulated", active_len, accumulated_len)?;

    let source = input.source.iter().copied().collect::<Vec<_>>();
    let mut accumulated = input.accumulated.iter().copied().collect::<Array1<_>>();
    if active_len == 0 {
        return Ok(LoucksSphericalOverlap {
            accumulated,
            active_len,
        });
    }

    let top_x = loucks_x(cutoff_index);

    for index in 1..=active_len {
        let x = loucks_x(index);
        let radius = x.exp();
        let contribution = sumax_integral_contribution(
            input.neighbor_distance,
            input.multiplicity,
            &source,
            top_x,
            radius,
        )?;
        accumulated[index - 1] += contribution;
    }

    Ok(LoucksSphericalOverlap {
        accumulated,
        active_len,
    })
}

/// Construct FEFF's muffin-tin overlap matrix from `POT/movrlp.f90`.
///
/// FEFF stores only a moving `novp = 50` radial window for each potential and
/// appends one equation for the interstitial potential. This function builds
/// that active matrix, applies FEFF-compatible single-complex LU factorization,
/// and returns the factors for downstream `ovp2mt`-style solves.
pub fn muffin_tin_overlap_matrix(
    input: MuffinTinOverlapMatrixInput<'_>,
) -> Result<MuffinTinOverlapMatrix, GridError> {
    validate_muffin_tin_overlap_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let active_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .and_then(|value| value.checked_add(1))
        .ok_or(GridError::GridLengthOverflow { name: "movrlp" })?;
    let radii = (1..=251).map(loucks_radius).collect::<Array1<_>>();
    let grid_half_step = (LOUCKS_DELTA / 2.0).exp();
    let radius_mode = (input.interstitial_selector - (input.interstitial_selector % 2)) / 2;
    let absorber_only = input.interstitial_selector % 2 == 1;

    let mut matrix = Array2::<Complex32>::zeros((active_order, active_order));
    for row in 0..active_order {
        for column in 0..(active_order - 1) {
            matrix[(row, column)] = Complex32::new(0.0, 0.0);
        }
        matrix[(row, row)] = Complex32::new(1.0, 0.0);
        matrix[(row, active_order - 1)] = Complex32::new(0.01, 0.0);
    }

    let mut bmat = Array2::<f32>::zeros((potential_count, active_order - 1));
    let mut interstitial_volume = input.interstitial_volume;
    validate_finite_scalar("interstitial_volume", interstitial_volume)?;
    let mut atom_count = 0.0;

    for target in 0..potential_count {
        let rav = movrlp_average_radius(input, &radii, target, radius_mode)?;
        let neighbors = movrlp_neighbors(input, target)?;
        for neighbor in neighbors {
            let source = neighbor.source_potential;
            let distance = neighbor.distance;
            let multiplicity = neighbor.multiplicity as Real;
            let pair = MovrlpPair {
                target,
                source,
                distance,
                multiplicity,
            };

            if distance < input.muffin_tin_radii[target] + input.muffin_tin_radii[source] {
                interstitial_volume += input.potential_multiplicities[target]
                    * multiplicity
                    * sphere_overlap_cap_volume(
                        input.muffin_tin_radii[target],
                        input.muffin_tin_radii[source],
                        distance,
                    )?;
            }

            if rav + input.muffin_tin_radii[source] > distance {
                movrlp_fill_boundary_row(input, &radii, &mut bmat, pair, rav, grid_half_step)?;
            }

            if input.muffin_tin_radii[target] + input.muffin_tin_radii[source] > distance {
                movrlp_fill_overlap_matrix(input, &radii, &mut matrix, pair, grid_half_step)?;
            }
        }
        atom_count += input.potential_multiplicities[target];
    }
    validate_nonzero_finite_scalar("atom_count", atom_count)?;

    if absorber_only {
        for column in 0..(active_order - 1) {
            matrix[(active_order - 1, column)] += Complex32::new(bmat[(0, column)], 0.0);
        }
    } else {
        for potential in 0..potential_count {
            let weight = (input.potential_multiplicities[potential] / atom_count) as f32;
            for column in 0..(active_order - 1) {
                matrix[(active_order - 1, column)] +=
                    Complex32::new(weight * bmat[(potential, column)], 0.0);
            }
        }
    }

    let lu = complex32_lu_factor(matrix.view())?;
    let final_pivot =
        lu.pivots()
            .get(active_order - 1)
            .copied()
            .ok_or(GridError::LengthTooShort {
                name: "movrlp_pivots",
                required: active_order,
                actual: lu.pivots().len(),
            })?;
    if final_pivot != active_order {
        return Err(GridError::IllegalFinalPivot {
            expected: active_order,
            actual: final_pivot,
        });
    }

    Ok(MuffinTinOverlapMatrix {
        radii,
        lu,
        interstitial_volume,
        active_order,
    })
}

/// Project overlapped potentials or densities onto FEFF muffin-tin spheres.
///
/// This ports `POT/ovp2mt.f90`. FEFF solves only the active `novp = 50`
/// radial window for each potential; when the interstitial potential is fixed
/// or the input is a density, it intentionally solves a prefix of the LU system
/// produced by `movrlp`. This function preserves that behavior and returns a
/// cloned output table rather than mutating the caller's array.
pub fn project_muffin_tin_overlap(
    input: MuffinTinOverlapProjectionInput<'_>,
) -> Result<MuffinTinOverlapProjection, GridError> {
    validate_muffin_tin_projection_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let window_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;
    let full_order = window_order
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;
    let solve_order = match input.mode {
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial => full_order,
        MuffinTinOverlapProjectionMode::Density { .. }
        | MuffinTinOverlapProjectionMode::PotentialFixedInterstitial => window_order,
    };

    let mut rhs = Array1::<Complex32>::zeros(solve_order);
    for potential in 0..potential_count {
        let first_row = input.muffin_tin_indices[potential] - MOVRLP_NOVP;
        for offset in 0..MOVRLP_NOVP {
            let mut value = input.values[(first_row + offset, potential)];
            if input.mode == MuffinTinOverlapProjectionMode::PotentialFixedInterstitial {
                value -= input.interstitial_value;
            }
            rhs[potential * MOVRLP_NOVP + offset] =
                Complex32::new(movrlp_real32("ovp2mt_rhs", value)?, 0.0);
        }
    }

    let absorber_only = input.interstitial_selector % 2 == 1;
    let radius_mode = input.interstitial_selector / 2;
    if input.mode == MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial {
        let average_values = ovp2mt_average_values(input, potential_count, radius_mode)?;
        let last_potential = if absorber_only {
            0
        } else {
            input.highest_potential_index
        };
        let mut average_sum = 0.0;
        let mut multiplicity_sum = 0.0;
        for potential in 0..=last_potential {
            average_sum += average_values[potential] * input.potential_multiplicities[potential];
            multiplicity_sum += input.potential_multiplicities[potential];
        }
        validate_nonzero_finite_scalar("ovp2mt_multiplicity_sum", multiplicity_sum)?;
        rhs[window_order] = Complex32::new(
            movrlp_real32("ovp2mt_rhs", average_sum / multiplicity_sum)?,
            0.0,
        );
    }

    let solved =
        complex32_lu_solve_prefix_vector(&input.overlap_matrix.lu, rhs.view(), solve_order)?;
    let window_values = solved
        .iter()
        .take(window_order)
        .map(|value| value.re as Real)
        .collect::<Array1<_>>();
    let mut output_values = input.values.to_owned();

    let interstitial_value = match input.mode {
        MuffinTinOverlapProjectionMode::Density { total_charge } => {
            total_charge - ovp2mt_density_muffin_tin_charge(input, &window_values, potential_count)?
        }
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial => {
            solved[window_order].re as Real / 100.0
        }
        MuffinTinOverlapProjectionMode::PotentialFixedInterstitial => input.interstitial_value,
    };

    match input.mode {
        MuffinTinOverlapProjectionMode::Density { .. } => {}
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial
        | MuffinTinOverlapProjectionMode::PotentialFixedInterstitial => {
            ovp2mt_rewrite_potentials(
                input,
                &window_values,
                interstitial_value,
                potential_count,
                &mut output_values,
            )?;
        }
    }

    Ok(MuffinTinOverlapProjection {
        values: output_values,
        interstitial_value,
        window_values,
    })
}

/// Volume of one FEFF spherical-overlap cap from `POT/istprm.f90` `calcvl`.
///
/// `sphere_radius` is the radius of the sphere whose cap is being measured,
/// `other_radius` is the radius of the overlapping sphere, and
/// `center_distance` is the distance between sphere centers. FEFF callers use
/// this only after confirming the spheres overlap; this function preserves the
/// algebraic `calcvl` formula and validates only finite positive inputs.
pub fn sphere_overlap_cap_volume(
    sphere_radius: Real,
    other_radius: Real,
    center_distance: Real,
) -> Result<Real, GridError> {
    validate_positive_finite_scalar("sphere_radius", sphere_radius)?;
    validate_positive_finite_scalar("other_radius", other_radius)?;
    validate_positive_finite_scalar("center_distance", center_distance)?;

    let plane_distance = (sphere_radius.powi(2) - other_radius.powi(2) + center_distance.powi(2))
        / (2.0 * center_distance);
    let cap_height = sphere_radius - plane_distance;
    let volume = PI / 3.0 * cap_height.powi(2) * (3.0 * sphere_radius - cap_height);
    validate_finite_scalar("sphere_overlap_cap_volume", volume)?;
    Ok(volume)
}

/// Total lens volume of two overlapping spheres using FEFF `calcvl` caps.
pub fn sphere_overlap_lens_volume(
    radius_a: Real,
    radius_b: Real,
    center_distance: Real,
) -> Result<Real, GridError> {
    Ok(
        sphere_overlap_cap_volume(radius_a, radius_b, center_distance)?
            + sphere_overlap_cap_volume(radius_b, radius_a, center_distance)?,
    )
}

/// Average FEFF potential and overlapped density over an interstitial shell.
///
/// This ports `POT/istval.f90`. FEFF integrates `r**3 * value` over the
/// logarithmic Loucks coordinate and divides by `(rws**3 - rmt**3) / 3`, leaving
/// out the common `4*pi` factor in both the integral and the shell volume.
pub fn interstitial_shell_values(
    input: InterstitialShellValuesInput<'_>,
) -> Result<InterstitialShellValues, GridError> {
    if !(input.muffin_tin_radius.is_finite() && input.muffin_tin_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.muffin_tin_radius,
        });
    }
    if !(input.wigner_seitz_radius.is_finite() && input.wigner_seitz_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.wigner_seitz_radius,
        });
    }
    if input.wigner_seitz_radius <= input.muffin_tin_radius {
        return Err(GridError::InvalidRadiusOrder {
            inner_radius: input.muffin_tin_radius,
            outer_radius: input.wigner_seitz_radius,
        });
    }
    validate_grid_index("muffin_tin", input.muffin_tin_index)?;
    validate_grid_index("wigner_seitz", input.wigner_seitz_index)?;
    if input.wigner_seitz_index < input.muffin_tin_index {
        return Err(GridError::InvalidGridIndexRange {
            lower_index: input.muffin_tin_index,
            upper_index: input.wigner_seitz_index,
        });
    }
    validate_positive_grid_length("total_potential", input.total_potential.len())?;
    validate_positive_grid_length("overlapped_density", input.overlapped_density.len())?;
    validate_component_values("total_potential", input.total_potential)?;
    validate_component_values("overlapped_density", input.overlapped_density)?;

    let required =
        input
            .wigner_seitz_index
            .checked_add(1)
            .ok_or(GridError::GridLengthOverflow {
                name: "interstitial",
            })?;
    ensure_source_length("total_potential", required, input.total_potential.len())?;
    ensure_source_length(
        "overlapped_density",
        required,
        input.overlapped_density.len(),
    )?;

    let shell_volume = (input.wigner_seitz_radius.powi(3) - input.muffin_tin_radius.powi(3)) / 3.0;
    let potential_integral = interstitial_shell_integral(
        input.total_potential,
        input.muffin_tin_radius,
        input.muffin_tin_index,
        input.wigner_seitz_radius,
        input.wigner_seitz_index,
    )?;
    let density_integral = interstitial_shell_integral(
        input.overlapped_density,
        input.muffin_tin_radius,
        input.muffin_tin_index,
        input.wigner_seitz_radius,
        input.wigner_seitz_index,
    )?;

    Ok(InterstitialShellValues {
        interstitial_potential: potential_integral / shell_volume,
        interstitial_density: density_integral / shell_volume,
        shell_volume,
    })
}

/// Locate FEFF overlapped-density tail indices and adjust radii when needed.
///
/// This ports the defined behavior of `POT/sidx.f90`. FEFF scans `rholap`
/// from `imt = ii(rmt)` until the first value at or below `1.0e-5`, then moves
/// the Norman radius inward if its index lies beyond the last positive-density
/// point. The original Fortran leaves `imax` undefined when the first scanned
/// density value is already below cutoff; Rust reports that case as
/// [`GridError::NoActiveDensityTail`].
pub fn overlap_density_indices(
    input: OverlapDensityIndicesInput<'_>,
) -> Result<OverlapDensityIndices, GridError> {
    if !(input.muffin_tin_radius.is_finite() && input.muffin_tin_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.muffin_tin_radius,
        });
    }
    if !(input.norman_radius.is_finite() && input.norman_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.norman_radius,
        });
    }
    validate_positive_grid_length("overlapped_density", input.overlapped_density.len())?;
    validate_component_values("overlapped_density", input.overlapped_density)?;

    let muffin_tin_index = feff_legacy_loucks_index_below(input.muffin_tin_radius)?;
    let initial_norman_index = feff_legacy_loucks_index_below(input.norman_radius)?;
    validate_grid_index("muffin_tin", muffin_tin_index)?;
    validate_grid_index("norman", initial_norman_index)?;
    ensure_source_length(
        "overlapped_density",
        muffin_tin_index,
        input.overlapped_density.len(),
    )?;

    let mut max_density_index = None;
    for index in muffin_tin_index..=input.overlapped_density.len() {
        if view_value(input.overlapped_density, index, "overlapped_density")? <= SIDX_DENSITY_CUTOFF
        {
            break;
        }
        max_density_index = Some(index);
    }
    let max_density_index = max_density_index.ok_or(GridError::NoActiveDensityTail {
        start_index: muffin_tin_index,
        threshold: SIDX_DENSITY_CUTOFF,
    })?;

    let (norman_index, norman_radius, moved_norman_radius) =
        if initial_norman_index > max_density_index {
            (
                max_density_index,
                feff_legacy_loucks_radius(max_density_index),
                true,
            )
        } else {
            (initial_norman_index, input.norman_radius, false)
        };

    Ok(OverlapDensityIndices {
        max_density_index,
        muffin_tin_index,
        norman_index,
        muffin_tin_radius: input.muffin_tin_radius,
        norman_radius,
        moved_norman_radius,
    })
}

/// Find FEFF's Norman radius from an overlapped density profile.
///
/// This ports `POT/frnrm.f90`. FEFF integrates `rho * r**2 dr`, with `rho`
/// already stored as `4*pi*density`, until the accumulated charge reaches the
/// atom's `Z`. The first pass follows FEFF's hand-coded Simpson recurrence, then
/// the returned radius is refined by the same `somm2` endpoint correction used
/// in the original routine. The radial grid intentionally preserves FEFF's
/// default-real `xx.f90` constants before widening to double precision.
pub fn norman_radius_from_density(input: NormanRadiusInput<'_>) -> Result<NormanRadius, GridError> {
    if input.atomic_number == 0 {
        return Err(GridError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    ensure_source_length(
        "overlapped_density",
        FRNRM_DENSITY_POINTS,
        input.overlapped_density.len(),
    )?;
    let density = input
        .overlapped_density
        .iter()
        .take(FRNRM_DENSITY_POINTS)
        .copied()
        .collect::<Vec<_>>();
    validate_slice_values("overlapped_density", &density)?;
    let radii = (1..=FRNRM_DENSITY_POINTS)
        .map(feff_legacy_loucks_radius)
        .collect::<Vec<_>>();
    let density_moments = density
        .iter()
        .zip(radii.iter())
        .map(|(&rho, &radius)| rho * radius * radius * radius)
        .collect::<Vec<_>>();

    let target_charge = input.atomic_number as Real;
    let scan = frnrm_initial_scan(&density, &radii, &density_moments, target_charge)?;
    let (index, mut fraction) = scan.crossing.ok_or(GridError::InsufficientNormanCharge {
        atomic_number: input.atomic_number,
        charge_found: scan.charge,
        max_radius: radii[FRNRM_DENSITY_POINTS - 1],
    })?;

    let mut radius = radii[index - 1] * (1.0 + fraction * FRNRM_LITERAL_DELTA);
    let correction_len = frnrm_correction_len(radius)?;
    ensure_source_length("overlapped_density", correction_len, FRNRM_DENSITY_POINTS)?;
    ensure_source_length("norman_correction", index + 1, correction_len)?;

    let correction_radii = &radii[..correction_len];
    let correction_values = correction_radii
        .iter()
        .zip(density.iter())
        .map(|(&ri, &rho)| rho * ri * ri)
        .collect::<Vec<_>>();

    let first_charge = somm2(
        correction_radii,
        &correction_values,
        FRNRM_LITERAL_DELTA,
        2.0,
        radius,
        0,
    )?;
    let first_delta = first_charge - target_charge;
    let density_at_radius =
        (1.0 - fraction) * correction_values[index - 1] + fraction * correction_values[index];
    validate_nonzero_finite_scalar("norman_correction_density", density_at_radius)?;

    let second_fraction = fraction - first_delta / density_at_radius;
    if (second_fraction - fraction).abs() > FRNRM_CORRECTION_THRESHOLD {
        radius = radii[index - 1] * (1.0 + second_fraction * FRNRM_LITERAL_DELTA);
        let second_charge = somm2(
            correction_radii,
            &correction_values,
            FRNRM_LITERAL_DELTA,
            2.0,
            radius,
            0,
        )?;
        let second_delta = second_charge - target_charge;
        let delta_difference = second_delta - first_delta;
        validate_nonzero_finite_scalar("norman_correction_delta", delta_difference)?;
        fraction = second_fraction - second_delta * (second_fraction - fraction) / delta_difference;
    } else {
        fraction = second_fraction;
    }

    Ok(NormanRadius {
        radius: radii[index - 1] * (1.0 + fraction * FRNRM_LITERAL_DELTA),
        index,
        fraction,
    })
}

/// Calculate FEFF's interstitial Fermi level from density and potential.
///
/// This ports `POT/fermi.f90`. FEFF stores `rhoint` as `4*pi*density`, so the
/// density parameter is `rs = (3 / rhoint)^(1/3)`, the Fermi momentum is
/// `xf = fa / rs`, and the chemical potential is `xmu = vint + xf**2 / 2`.
pub fn interstitial_fermi_level(input: FermiLevelInput) -> Result<FermiLevel, GridError> {
    validate_positive_finite_scalar("interstitial_density", input.interstitial_density)?;
    validate_finite_scalar("interstitial_potential", input.interstitial_potential)?;

    let density_parameter = (3.0 / input.interstitial_density).powf(1.0 / 3.0);
    let fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / density_parameter;
    let chemical_potential = input.interstitial_potential + fermi_momentum.powi(2) / 2.0;

    Ok(FermiLevel {
        chemical_potential,
        density_parameter,
        fermi_momentum,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FrnrmInitialScan {
    crossing: Option<(usize, Real)>,
    charge: Real,
}

fn frnrm_initial_scan(
    density: &[Real],
    radii: &[Real],
    density_moments: &[Real],
    target_charge: Real,
) -> Result<FrnrmInitialScan, GridError> {
    let mut charge =
        (9.0 * density_moments[0] + 28.0 * density_moments[1] + 23.0 * density_moments[2]) / 480.0;
    charge += frnrm_initial_origin_correction(density, radii)?;

    let mut left = density_moments[3];
    let mut center = density_moments[4];
    let mut right = density_moments[5];

    for index in 7..=FRNRM_NRPTX {
        let far_left = left;
        left = center;
        center = right;
        right = if index <= FRNRM_DENSITY_POINTS {
            density_moments[index - 1]
        } else {
            0.0
        };
        let previous_charge = charge;
        charge += (13.0 * (center + left) - far_left - right) / 480.0;
        if charge >= target_charge {
            let increment = charge - previous_charge;
            validate_nonzero_finite_scalar("norman_charge_increment", increment)?;
            return Ok(FrnrmInitialScan {
                crossing: Some((index - 2, (target_charge - previous_charge) / increment)),
                charge,
            });
        }
    }

    Ok(FrnrmInitialScan {
        crossing: None,
        charge,
    })
}

fn frnrm_initial_origin_correction(density: &[Real], radii: &[Real]) -> Result<Real, GridError> {
    let d1 = 3.0;
    let delta = FRNRM_LITERAL_DELTA.exp() - 1.0;
    let second_coefficient =
        radii[0] / (d1 * (d1 + 1.0) * delta * ((d1 - 1.0) * FRNRM_LITERAL_DELTA).exp());
    let first_coefficient = radii[0] * (1.0 + 1.0 / (delta * (d1 + 1.0))) / d1;
    let correction = first_coefficient * density[0] * radii[0] * radii[0]
        - second_coefficient * density[1] * radii[1] * radii[1];
    validate_finite_scalar("norman_origin_correction", correction)?;
    Ok(correction)
}

fn frnrm_correction_len(radius: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    let grid_index =
        fortran_truncated_index((radius.ln() + FRNRM_LITERAL_OFFSET) / FRNRM_LITERAL_DELTA + 2.0);
    grid_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow {
            name: "norman_correction",
        })
}

fn sumax_integral_contribution(
    neighbor_distance: Real,
    multiplicity: Real,
    source: &[Real],
    top_x: Real,
    radius: Real,
) -> Result<Real, GridError> {
    let lower_radius = neighbor_distance - radius;
    if lower_radius <= 0.0 {
        return Ok(0.0);
    }

    let lower_x = lower_radius.ln();
    if lower_x >= top_x {
        return Ok(0.0);
    }

    let mut integral = 0.0;
    let mut lower_index =
        fortran_truncated_index(2.0 + 20.0 * (lower_x + SUMAX_LITERAL_OFFSET)).max(1);
    let mut lower_grid_x = sumax_literal_x(lower_index);
    if lower_index >= 2 {
        let cap_width = lower_grid_x - lower_x;
        let lower_value = source_value(source, lower_index, "source")?;
        let previous_value = source_value(source, lower_index - 1, "source")?;
        integral += 0.5
            * cap_width
            * (lower_value * (2.0 - 20.0 * cap_width) * (2.0 * lower_grid_x).exp()
                + 20.0
                    * cap_width
                    * previous_value
                    * (2.0 * (lower_grid_x - SUMAX_LITERAL_DELTA)).exp());
    }

    let upper_x = (neighbor_distance + radius).ln();
    let upper_index = if upper_x >= top_x {
        radial_index_below(SUMAX_WIGNER_SEITZ_RADIUS, LOUCKS_DELTA)?
    } else {
        let index = fortran_truncated_index(1.0 + 20.0 * (upper_x + SUMAX_LITERAL_OFFSET));
        if index < lower_index {
            let near_zero = source_value(source, index, "source")?
                * (2.0 * (lower_grid_x - SUMAX_LITERAL_DELTA)).exp();
            let lower_value =
                source_value(source, lower_index, "source")? * (2.0 * lower_grid_x).exp();
            let upper_interp = near_zero
                + 20.0 * (lower_value - near_zero) * (upper_x - lower_grid_x + SUMAX_LITERAL_DELTA);
            let lower_interp = near_zero
                + 20.0 * (lower_value - near_zero) * (lower_x - lower_grid_x + SUMAX_LITERAL_DELTA);
            integral = 0.5 * (lower_interp + upper_interp) * (upper_x - lower_x);
            return Ok(0.5 * integral * multiplicity / (neighbor_distance * radius));
        }

        let upper_grid_x = sumax_literal_x(index);
        let cap_width = upper_x - upper_grid_x;
        let upper_value = source_value(source, index, "source")?;
        let next_value = source_value(source, index + 1, "source")?;
        integral += 0.5
            * cap_width
            * (upper_value * (2.0 - 20.0 * cap_width) * (2.0 * upper_grid_x).exp()
                + next_value
                    * 20.0
                    * cap_width
                    * (2.0 * (upper_grid_x + SUMAX_LITERAL_DELTA)).exp());
        index
    };

    while upper_index > lower_index {
        let current = source_value(source, lower_index, "source")? * (2.0 * lower_grid_x).exp();
        let next = source_value(source, lower_index + 1, "source")?
            * (2.0 * (lower_grid_x + SUMAX_LITERAL_DELTA)).exp();
        integral += 0.5 * (current + next) * SUMAX_LITERAL_DELTA;
        lower_index += 1;
        if lower_index < upper_index {
            lower_grid_x += SUMAX_LITERAL_DELTA;
        }
    }

    Ok(0.5 * integral * multiplicity / (neighbor_distance * radius))
}

fn ovp2mt_average_values(
    input: MuffinTinOverlapProjectionInput<'_>,
    potential_count: usize,
    radius_mode: usize,
) -> Result<Array1<Real>, GridError> {
    let mut average_values = Array1::<Real>::zeros(potential_count);
    let radii = input.radii.iter().copied().collect::<Vec<_>>();
    for potential in 0..potential_count {
        let active_len =
            checked_index_offset("norman_indices", input.norman_indices[potential], 2)?;
        let values = input
            .values
            .column(potential)
            .iter()
            .take(active_len)
            .copied()
            .collect::<Vec<_>>();
        let average_radius = muffin_tin_average_radius(
            input.radii,
            input.muffin_tin_indices,
            input.muffin_tin_radii,
            input.norman_radii,
            input.near_neighbor_flags,
            potential,
            radius_mode,
        )?;
        average_values[potential] = terp(&radii[..active_len], &values, 3, average_radius)?.value;
    }
    Ok(average_values)
}

fn ovp2mt_density_muffin_tin_charge(
    input: MuffinTinOverlapProjectionInput<'_>,
    window_values: &Array1<Real>,
    potential_count: usize,
) -> Result<Real, GridError> {
    let radii = input.radii.iter().take(251).copied().collect::<Vec<_>>();
    let mut total_charge = 0.0;
    for potential in 0..potential_count {
        let muffin_index = input.muffin_tin_indices[potential];
        let active_len = checked_index_offset("muffin_tin_indices", muffin_index, 2)?;
        let window_start = muffin_index - MOVRLP_NOVP + 1;
        let mut density_moment = Array1::<Real>::zeros(251);
        for radial_index in 1..=muffin_index {
            let density = if radial_index < window_start {
                input.values[(radial_index - 1, potential)]
            } else {
                let window_index = potential * MOVRLP_NOVP + radial_index - window_start;
                window_values[window_index]
            };
            density_moment[radial_index - 1] = density * input.radii[radial_index - 1].powi(2);
        }

        let interpolation_radii = radii[..muffin_index].to_vec();
        let interpolation_values = density_moment
            .iter()
            .take(muffin_index)
            .copied()
            .collect::<Vec<_>>();
        for radial_index in (muffin_index + 1)..=active_len {
            density_moment[radial_index - 1] = terp(
                &interpolation_radii,
                &interpolation_values,
                2,
                input.radii[radial_index - 1],
            )?
            .value;
        }

        let density_values = density_moment
            .iter()
            .take(active_len)
            .copied()
            .collect::<Vec<_>>();
        let charge = somm2(
            &radii[..active_len],
            &density_values,
            LOUCKS_DELTA,
            0.0,
            input.muffin_tin_radii[potential],
            0,
        )?;
        total_charge += input.potential_multiplicities[potential] * charge;
    }
    Ok(total_charge)
}

fn ovp2mt_rewrite_potentials(
    input: MuffinTinOverlapProjectionInput<'_>,
    window_values: &Array1<Real>,
    interstitial_value: Real,
    potential_count: usize,
    output_values: &mut Array2<Real>,
) -> Result<(), GridError> {
    let radii = input.radii.iter().take(251).copied().collect::<Vec<_>>();
    for potential in 0..potential_count {
        let muffin_index = input.muffin_tin_indices[potential];
        let tail_start = checked_index_offset("muffin_tin_indices", muffin_index, 1)?;
        let first_row = muffin_index - MOVRLP_NOVP;
        for offset in 0..MOVRLP_NOVP {
            output_values[(first_row + offset, potential)] =
                window_values[potential * MOVRLP_NOVP + offset] + interstitial_value;
        }

        let interpolation_values = output_values
            .column(potential)
            .iter()
            .take(muffin_index)
            .copied()
            .collect::<Vec<_>>();
        output_values[(muffin_index, potential)] = terp(
            &radii[..muffin_index],
            &interpolation_values,
            2,
            input.radii[muffin_index],
        )?
        .value;
        for radial_index in tail_start..251 {
            output_values[(radial_index, potential)] = interstitial_value;
        }
    }
    Ok(())
}

fn complex32_lu_solve_prefix_vector(
    lu: &Complex32Lu,
    right_hand_side: ArrayView1<'_, Complex32>,
    order: usize,
) -> Result<Array1<Complex32>, GridError> {
    if right_hand_side.len() != order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side",
            left: right_hand_side.len(),
            right_name: "solve order",
            right: order,
        }
        .into());
    }
    ensure_shape("overlap_lu", lu.factors().shape(), order, order)?;
    ensure_len("overlap_pivots", lu.pivots().len(), order)?;

    let factors = lu.factors();
    let mut solution = right_hand_side.to_owned();
    for (pivot, &pivot_row) in lu.pivots().iter().take(order).enumerate() {
        if pivot_row == 0 || pivot_row > order {
            return Err(GridError::InvalidGridIndex {
                name: "overlap_pivot",
                index: pivot_row,
            });
        }
        let swap_row = pivot_row - 1;
        if swap_row != pivot {
            let left = solution[pivot];
            solution[pivot] = solution[swap_row];
            solution[swap_row] = left;
        }
    }

    for pivot in 0..order {
        for row in (pivot + 1)..order {
            let factor = factors[(row, pivot)];
            let pivot_value = solution[pivot];
            solution[row] -= factor * pivot_value;
        }
    }

    for pivot in (0..order).rev() {
        let diagonal = factors[(pivot, pivot)];
        if diagonal == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot }.into());
        }
        solution[pivot] /= diagonal;
        let pivot_value = solution[pivot];
        for row in 0..pivot {
            let factor = factors[(row, pivot)];
            solution[row] -= factor * pivot_value;
        }
    }

    Ok(solution)
}

fn muffin_tin_average_radius(
    radii: ArrayView1<'_, Real>,
    muffin_tin_indices: ArrayView1<'_, usize>,
    muffin_tin_radii: ArrayView1<'_, Real>,
    norman_radii: ArrayView1<'_, Real>,
    near_neighbor_flags: ArrayView1<'_, bool>,
    potential: usize,
    radius_mode: usize,
) -> Result<Real, GridError> {
    let after_muffin = radii[movrlp_radii_index_after_muffin(muffin_tin_indices[potential])?];
    if near_neighbor_flags[potential] {
        return Ok(after_muffin);
    }
    if radius_mode == 1 {
        Ok((muffin_tin_radii[potential] + norman_radii[potential]) / 2.0)
    } else if radius_mode == 0 {
        Ok(norman_radii[potential])
    } else {
        Ok(after_muffin)
    }
}

fn movrlp_average_radius(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    potential: usize,
    radius_mode: usize,
) -> Result<Real, GridError> {
    muffin_tin_average_radius(
        radii.view(),
        input.muffin_tin_indices,
        input.muffin_tin_radii,
        input.norman_radii,
        input.near_neighbor_flags,
        potential,
        radius_mode,
    )
}

fn movrlp_radii_index_after_muffin(muffin_tin_index: usize) -> Result<usize, GridError> {
    muffin_tin_index
        .checked_add(1)
        .filter(|&index| index <= 251)
        .map(|index| index - 1)
        .ok_or(GridError::SourceGridTooShort {
            name: "radii",
            required: muffin_tin_index.saturating_add(1),
            available: 251,
        })
}

fn movrlp_neighbors(
    input: MuffinTinOverlapMatrixInput<'_>,
    target: usize,
) -> Result<Vec<MuffinTinOverlapNeighbor>, GridError> {
    let explicit = input.explicit_overlaps[target];
    if !explicit.is_empty() {
        return Ok(explicit.to_vec());
    }

    let representative = input.representative_atoms[target];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut neighbors = Vec::new();
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        neighbors.push(MuffinTinOverlapNeighbor {
            source_potential: input.atom_potentials[atom],
            multiplicity: 1,
            distance: distance_between(center, position),
        });
    }
    Ok(neighbors)
}

#[derive(Debug, Clone, Copy)]
struct MovrlpPair {
    target: usize,
    source: usize,
    distance: Real,
    multiplicity: Real,
}

fn movrlp_fill_boundary_row(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    bmat: &mut Array2<f32>,
    pair: MovrlpPair,
    average_radius: Real,
    grid_half_step: Real,
) -> Result<(), GridError> {
    let check_index = loucks_index_below(pair.distance - average_radius)?;
    if input.muffin_tin_indices[pair.source].saturating_sub(check_index) >= MOVRLP_NOVP - 1 {
        return Err(GridError::MuffinTinOverlapTooLarge {
            left: pair.target,
            right: pair.source,
        });
    }
    let start = movrlp_window_start(input.muffin_tin_indices[pair.source], pair.source)?;
    for radial in start..=input.muffin_tin_indices[pair.source] {
        let radius = radii[radial - 1];
        let mut r1 = radius / grid_half_step;
        let mut r2 = radius * grid_half_step;
        if radial == input.muffin_tin_indices[pair.source] {
            r2 = input.muffin_tin_radii[pair.source];
            r1 = (r1 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                - input.muffin_tin_radii[pair.source])
                / 2.0;
        }
        if radial + 1 == input.muffin_tin_indices[pair.source] {
            r2 = (r2 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                - input.muffin_tin_radii[pair.source])
                / 2.0;
        }
        if r2 + average_radius < pair.distance {
            continue;
        }

        if r1 + average_radius < pair.distance {
            let mut fraction = (pair.distance - average_radius - r1) / (r2 - r1);
            r1 = pair.distance - average_radius;
            let contribution = (r2.powi(2) - r1.powi(2)) / (4.0 * pair.distance * average_radius)
                * pair.multiplicity;
            let neighbor_index = if radial == input.muffin_tin_indices[pair.source] {
                radial - 1
            } else {
                radial + 1
            };
            fraction *= (r2 - radius) / (radii[neighbor_index - 1] - radius);
            let column = pair.source * MOVRLP_NOVP + radial - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution * (1.0 - fraction))?;
            let column = pair.source * MOVRLP_NOVP + neighbor_index - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution * fraction)?;
        } else {
            let contribution = (r2.powi(2) - r1.powi(2)) / (4.0 * pair.distance * average_radius)
                * pair.multiplicity;
            let column = pair.source * MOVRLP_NOVP + radial - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution)?;
        }
    }
    Ok(())
}

fn movrlp_fill_overlap_matrix(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    matrix: &mut Array2<Complex32>,
    pair: MovrlpPair,
    grid_half_step: Real,
) -> Result<(), GridError> {
    let check_target = loucks_index_below(pair.distance - input.muffin_tin_radii[pair.source])?;
    let check_source = loucks_index_below(pair.distance - input.muffin_tin_radii[pair.target])?;
    if input.muffin_tin_indices[pair.target].saturating_sub(check_target) >= MOVRLP_NOVP - 1
        || input.muffin_tin_indices[pair.source].saturating_sub(check_source) >= MOVRLP_NOVP - 1
    {
        return Err(GridError::MuffinTinOverlapTooLarge {
            left: pair.target,
            right: pair.source,
        });
    }

    let target_start = movrlp_window_start(input.muffin_tin_indices[pair.target], pair.target)?;
    let source_start = movrlp_window_start(input.muffin_tin_indices[pair.source], pair.source)?;
    for target_radial in target_start..=input.muffin_tin_indices[pair.target] {
        let target_radius = radii[target_radial - 1];
        let mut target_r1 = target_radius / grid_half_step;
        let mut target_r2 = target_radius * grid_half_step;
        if target_radial == input.muffin_tin_indices[pair.target] {
            target_r2 = input.muffin_tin_radii[pair.target];
            target_r1 = (target_r1 + 2.0 * radii[input.muffin_tin_indices[pair.target] - 1]
                - input.muffin_tin_radii[pair.target])
                / 2.0;
        }
        if target_radial + 1 == input.muffin_tin_indices[pair.target] {
            target_r2 = (target_r2 + 2.0 * radii[input.muffin_tin_indices[pair.target] - 1]
                - input.muffin_tin_radii[pair.target])
                / 2.0;
        }
        let target_column = pair.target * MOVRLP_NOVP + target_radial - target_start;

        for source_radial in source_start..=input.muffin_tin_indices[pair.source] {
            let source_radius = radii[source_radial - 1];
            let mut source_r1 = source_radius / grid_half_step;
            let mut source_r2 = source_radius * grid_half_step;
            if source_radial == input.muffin_tin_indices[pair.source] {
                source_r2 = input.muffin_tin_radii[pair.source];
                source_r1 = (source_r1 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                    - input.muffin_tin_radii[pair.source])
                    / 2.0;
            }
            if source_radial + 1 == input.muffin_tin_indices[pair.source] {
                source_r2 = (source_r2 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                    - input.muffin_tin_radii[pair.source])
                    / 2.0;
            }
            if source_r2 + target_r2 < pair.distance {
                continue;
            }

            let mut contribution = sphere_overlap_lens_volume(target_r2, source_r2, pair.distance)?;
            if target_r1 + source_r2 > pair.distance {
                contribution -= sphere_overlap_lens_volume(target_r1, source_r2, pair.distance)?;
            }
            if target_r2 + source_r1 > pair.distance {
                contribution -= sphere_overlap_lens_volume(target_r2, source_r1, pair.distance)?;
            }
            if target_r1 + source_r1 > pair.distance {
                contribution += sphere_overlap_lens_volume(target_r1, source_r1, pair.distance)?;
            }
            contribution = contribution
                / (4.0 / 3.0 * PI * (target_r2.powi(3) - target_r1.powi(3)))
                * pair.multiplicity;

            if source_r1 + target_r2 < pair.distance {
                let mut fraction =
                    (pair.distance - target_radius - source_r1) / (source_r2 - source_r1);
                let neighbor_index = if source_radial == input.muffin_tin_indices[pair.source] {
                    source_radial - 1
                } else {
                    source_radial + 1
                };
                fraction *=
                    (source_r2 - source_radius) / (radii[neighbor_index - 1] - source_radius);
                let column = pair.source * MOVRLP_NOVP + source_radial - source_start;
                matrix[(target_column, column)] += Complex32::new(
                    movrlp_real32("cmovp", contribution * (1.0 - fraction))?,
                    0.0,
                );
                let column = pair.source * MOVRLP_NOVP + neighbor_index - source_start;
                matrix[(target_column, column)] +=
                    Complex32::new(movrlp_real32("cmovp", contribution * fraction)?, 0.0);
            } else {
                let column = pair.source * MOVRLP_NOVP + source_radial - source_start;
                matrix[(target_column, column)] +=
                    Complex32::new(movrlp_real32("cmovp", contribution)?, 0.0);
            }
        }
    }
    Ok(())
}

fn movrlp_window_start(muffin_tin_index: usize, potential: usize) -> Result<usize, GridError> {
    if muffin_tin_index < MOVRLP_NOVP {
        Err(GridError::MuffinTinIndexTooSmall {
            name: "muffin_tin_indices",
            potential,
            minimum: MOVRLP_NOVP,
            index: muffin_tin_index,
        })
    } else {
        Ok(muffin_tin_index - MOVRLP_NOVP + 1)
    }
}

fn movrlp_real32(name: &'static str, value: Real) -> Result<f32, GridError> {
    validate_finite_scalar(name, value)?;
    let narrowed = value as f32;
    if narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
}

fn validate_muffin_tin_overlap_input(
    input: MuffinTinOverlapMatrixInput<'_>,
) -> Result<(), GridError> {
    let potential_count = input
        .highest_potential_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "potential" })?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        potential_count,
    )?;
    ensure_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len(
        "explicit_overlaps",
        input.explicit_overlaps.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_indices",
        input.muffin_tin_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_radii",
        input.muffin_tin_radii.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len(
        "near_neighbor_flags",
        input.near_neighbor_flags.len(),
        potential_count,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(GridError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values("atom_potentials", input.atom_potentials, potential_count)?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;
    validate_real_values("potential_multiplicities", input.potential_multiplicities)?;
    validate_real_values("muffin_tin_radii", input.muffin_tin_radii)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    for potential in 0..potential_count {
        validate_positive_finite_scalar(
            "potential_multiplicities",
            input.potential_multiplicities[potential],
        )?;
        validate_positive_finite_scalar("muffin_tin_radii", input.muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        if input.muffin_tin_indices[potential] < MOVRLP_NOVP {
            return Err(GridError::MuffinTinIndexTooSmall {
                name: "muffin_tin_indices",
                potential,
                minimum: MOVRLP_NOVP,
                index: input.muffin_tin_indices[potential],
            });
        }
        if input.muffin_tin_indices[potential] >= 251 {
            return Err(GridError::SourceGridTooShort {
                name: "radii",
                required: input.muffin_tin_indices[potential] + 1,
                available: 251,
            });
        }
        for neighbor in input.explicit_overlaps[potential] {
            if neighbor.source_potential >= potential_count {
                return Err(GridError::InvalidPotentialIndex {
                    name: "explicit_overlaps.source_potential",
                    index: neighbor.source_potential,
                    available: potential_count,
                });
            }
            if neighbor.multiplicity == 0 {
                return Err(GridError::InvalidGridIndex {
                    name: "explicit_overlaps.multiplicity",
                    index: 0,
                });
            }
            validate_positive_finite_scalar("explicit_overlaps.distance", neighbor.distance)?;
        }
    }
    Ok(())
}

fn validate_muffin_tin_projection_input(
    input: MuffinTinOverlapProjectionInput<'_>,
) -> Result<(), GridError> {
    let potential_count = input
        .highest_potential_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "potential" })?;
    let window_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;
    let full_order = window_order
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "ovp2mt" })?;

    ensure_shape("values", input.values.shape(), 251, potential_count)?;
    ensure_len("radii", input.radii.len(), 251)?;
    ensure_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len(
        "norman_indices",
        input.norman_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_indices",
        input.muffin_tin_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_radii",
        input.muffin_tin_radii.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len(
        "near_neighbor_flags",
        input.near_neighbor_flags.len(),
        potential_count,
    )?;
    if input.overlap_matrix.active_order != full_order {
        return Err(GridError::OverlapMatrixOrderMismatch {
            required: full_order,
            actual: input.overlap_matrix.active_order,
        });
    }
    ensure_shape(
        "overlap_lu",
        input.overlap_matrix.lu.factors().shape(),
        full_order,
        full_order,
    )?;
    ensure_len(
        "overlap_pivots",
        input.overlap_matrix.lu.pivots().len(),
        full_order,
    )?;

    validate_positive_radii(input.radii, 251)?;
    validate_real_table("values", input.values)?;
    validate_real_values("potential_multiplicities", input.potential_multiplicities)?;
    validate_real_values("muffin_tin_radii", input.muffin_tin_radii)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_finite_scalar("interstitial_value", input.interstitial_value)?;
    if let MuffinTinOverlapProjectionMode::Density { total_charge } = input.mode {
        validate_finite_scalar("total_charge", total_charge)?;
    }

    for potential in 0..potential_count {
        validate_positive_finite_scalar(
            "potential_multiplicities",
            input.potential_multiplicities[potential],
        )?;
        validate_positive_finite_scalar("muffin_tin_radii", input.muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        if input.muffin_tin_indices[potential] < MOVRLP_NOVP {
            return Err(GridError::MuffinTinIndexTooSmall {
                name: "muffin_tin_indices",
                potential,
                minimum: MOVRLP_NOVP,
                index: input.muffin_tin_indices[potential],
            });
        }
        let muffin_required =
            checked_index_offset("muffin_tin_indices", input.muffin_tin_indices[potential], 2)?;
        let norman_required =
            checked_index_offset("norman_indices", input.norman_indices[potential], 2)?;
        ensure_source_length("values", muffin_required, input.values.nrows())?;
        ensure_source_length("radii", muffin_required, input.radii.len())?;
        ensure_source_length("values", norman_required, input.values.nrows())?;
        ensure_source_length("radii", norman_required, input.radii.len())?;
        validate_grid_index("muffin_tin_indices", input.muffin_tin_indices[potential])?;
        validate_grid_index("norman_indices", input.norman_indices[potential])?;
    }

    Ok(())
}

fn interstitial_shell_integral(
    values: ArrayView1<'_, Real>,
    muffin_tin_radius: Real,
    muffin_tin_index: usize,
    wigner_seitz_radius: Real,
    wigner_seitz_index: usize,
) -> Result<Real, GridError> {
    let trapezoid_sum = (muffin_tin_index..wigner_seitz_index).try_fold(0.0, |sum, index| {
        let right = radius_cubed_grid_value(values, index + 1, "grid")?;
        let left = radius_cubed_grid_value(values, index, "grid")?;
        Ok::<_, GridError>(sum + 0.5 * (right + left) * LOUCKS_DELTA)
    })?;
    let upper_cap = interstitial_shell_cap(values, wigner_seitz_radius, wigner_seitz_index)?;
    let lower_cap = interstitial_shell_cap(values, muffin_tin_radius, muffin_tin_index)?;
    Ok(trapezoid_sum + upper_cap - lower_cap)
}

fn interstitial_shell_cap(
    values: ArrayView1<'_, Real>,
    radius: Real,
    index: usize,
) -> Result<Real, GridError> {
    let cap_width = radius.ln() - loucks_x(index);
    let ratio = cap_width / LOUCKS_DELTA;
    let left = radius_cubed_grid_value(values, index, "grid")?;
    let right = radius_cubed_grid_value(values, index + 1, "grid")?;
    Ok(0.5 * cap_width * ((2.0 - ratio) * left + ratio * right))
}

fn validate_delta(delta: Real) -> Result<(), GridError> {
    if delta.is_finite() && delta > 0.0 {
        Ok(())
    } else {
        Err(GridError::InvalidDelta { delta })
    }
}

fn validate_positive_grid_length(name: &'static str, len: usize) -> Result<(), GridError> {
    if len > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridLength { name })
    }
}

fn ensure_len(name: &'static str, actual: usize, required: usize) -> Result<(), GridError> {
    if actual >= required {
        Ok(())
    } else {
        Err(GridError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}

fn ensure_shape(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
) -> Result<(), GridError> {
    let rows = shape[0];
    let columns = shape[1];
    if rows >= required_rows && columns >= required_columns {
        Ok(())
    } else {
        Err(GridError::ShapeTooSmall {
            name,
            rows,
            columns,
            required_rows,
            required_columns,
        })
    }
}

fn validate_grid_index(name: &'static str, index: usize) -> Result<(), GridError> {
    if index > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridIndex { name, index })
    }
}

fn validate_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
}

fn validate_positive_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(GridError::NonPositiveScalar { name, value })
    }
}

fn validate_nonzero_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(GridError::ZeroScalar { name, value })
    }
}

fn validate_real_values(name: &'static str, values: ArrayView1<'_, Real>) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_real_table(name: &'static str, values: ArrayView2<'_, Real>) -> Result<(), GridError> {
    let columns = values.ncols();
    for ((row, column), &value) in values.indexed_iter() {
        if !value.is_finite() {
            let index = row
                .checked_mul(columns)
                .and_then(|value| value.checked_add(column))
                .ok_or(GridError::GridLengthOverflow { name })?;
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_component_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_position_table(positions: ArrayView2<'_, Real>) -> Result<(), GridError> {
    if positions.ncols() != 3 {
        return Err(GridError::InvalidPositionShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    ensure_shape("atom_positions", positions.shape(), positions.nrows(), 3)?;
    for ((atom_index, axis), &value) in positions.indexed_iter() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue {
                name: "atom_positions",
                index: atom_index * 3 + axis,
                value,
            });
        }
    }
    Ok(())
}

fn validate_usize_potential_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
    available: usize,
) -> Result<(), GridError> {
    for &index in values {
        if index >= available {
            return Err(GridError::InvalidPotentialIndex {
                name,
                index,
                available,
            });
        }
    }
    Ok(())
}

fn validate_component_prefix_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().take(active_len).enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_slice_values(name: &'static str, values: &[Real]) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_positive_radii(
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), GridError> {
    for &radius in values.iter().take(active_len) {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(GridError::InvalidRadius { radius });
        }
    }
    Ok(())
}

fn validate_source_len_at_least(
    name: &'static str,
    available: usize,
    required: usize,
) -> Result<(), GridError> {
    if available >= required {
        Ok(())
    } else {
        Err(GridError::SourceGridTooShort {
            name,
            required,
            available,
        })
    }
}

fn ensure_source_length(
    name: &'static str,
    required: usize,
    available: usize,
) -> Result<(), GridError> {
    if available >= required {
        Ok(())
    } else {
        Err(GridError::SourceGridTooShort {
            name,
            required,
            available,
        })
    }
}

fn checked_index_offset(
    name: &'static str,
    index: usize,
    offset: usize,
) -> Result<usize, GridError> {
    index
        .checked_add(offset)
        .ok_or(GridError::GridLengthOverflow { name })
}

fn square_index_as_real(name: &'static str, index: usize) -> Result<Real, GridError> {
    index
        .checked_mul(index)
        .map(|value| value as Real)
        .ok_or(GridError::GridLengthOverflow { name })
}

fn fortran_truncated_index(value: Real) -> usize {
    if value <= 0.0 {
        0
    } else {
        value.trunc() as usize
    }
}

fn sumax_literal_x(index_1based: usize) -> Real {
    SUMAX_LITERAL_DELTA * (index_1based as Real - 1.0) - SUMAX_LITERAL_OFFSET
}

fn feff_legacy_loucks_x(index_1based: usize) -> Real {
    sumax_literal_x(index_1based)
}

fn feff_legacy_loucks_radius(index_1based: usize) -> Real {
    feff_legacy_loucks_x(index_1based).exp()
}

fn feff_legacy_loucks_index_below(radius: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    Ok(fortran_truncated_index(
        (radius.ln() + SUMAX_LITERAL_OFFSET) / SUMAX_LITERAL_DELTA + 1.0,
    ))
}

fn radius_cubed_grid_value(
    values: ArrayView1<'_, Real>,
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    Ok(loucks_radius(index_1based).powi(3) * view_value(values, index_1based, name)?)
}

fn view_value(
    values: ArrayView1<'_, Real>,
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    if index_1based == 0 || index_1based > values.len() {
        Err(GridError::SourceGridTooShort {
            name,
            required: index_1based.max(1),
            available: values.len(),
        })
    } else {
        Ok(values[index_1based - 1])
    }
}

fn source_value(
    values: &[Real],
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    if index_1based == 0 || index_1based > values.len() {
        Err(GridError::SourceGridTooShort {
            name,
            required: index_1based.max(1),
            available: values.len(),
        })
    } else {
        Ok(values[index_1based - 1])
    }
}

#[cfg(test)]
mod tests;
