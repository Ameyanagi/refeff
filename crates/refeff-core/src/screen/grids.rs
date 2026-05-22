//! SCREEN radial and energy grid helpers.

use ndarray::Array1;

use crate::{Complex, ComplexVec, Real, RealVec};

use super::types::*;
use super::validation::{
    checked_radial_add, positive_radial_bound, validate_count_at_least, validate_finite,
    validate_increasing, validate_positive,
};

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

/// Port the shared SCREEN/CRPA `jri`, `jnrm`, and `ilast` setup.
///
/// `screensub.f90` and `CRPA/chi_crpa.f90` both derive active radial bounds
/// from `getiat`: `jri = getiat(rmt) + 1`, `jri1 = jri + 1`,
/// `jnrm = getiat(rnrm) + 1`, and `ilast = min(jnrm + 6 + iend, nrx)`.
/// The returned indices keep those FEFF 1-based names so callers can mirror
/// the original handoff logic while converting to zero-based slices locally.
pub fn screen_radial_bounds(
    input: ScreenRadialBoundsInput,
) -> Result<ScreenRadialBounds, ScreenError> {
    validate_positive("dx", input.dx)?;
    validate_finite("x0", input.x0)?;
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_positive("norman_radius", input.norman_radius)?;
    validate_count_at_least("radial_capacity", input.radial_capacity, 1)?;
    validate_count_at_least("response_capacity", input.response_capacity, 1)?;

    let muffin_tin_base = screen_radial_index_1based(input.x0, input.dx, input.muffin_tin_radius)?;
    let muffin_tin_value = checked_radial_add("muffin_tin_index_1based", muffin_tin_base, 1)?;
    let muffin_tin_index_1based =
        positive_radial_bound("muffin_tin_index_1based", muffin_tin_value)?;
    let muffin_tin_next_value =
        checked_radial_add("muffin_tin_next_index_1based", muffin_tin_value, 1)?;
    let muffin_tin_next_index_1based =
        positive_radial_bound("muffin_tin_next_index_1based", muffin_tin_next_value)?;
    if muffin_tin_next_index_1based > input.radial_capacity {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: muffin_tin_next_index_1based,
            capacity: input.radial_capacity,
        });
    }

    let norman_base = screen_radial_index_1based(input.x0, input.dx, input.norman_radius)?;
    let norman_value = checked_radial_add("norman_index_1based", norman_base, 1)?;
    let norman_index_1based = positive_radial_bound("norman_index_1based", norman_value)?;
    let active_tail_value = checked_radial_add("active_count", norman_value, 6)?;
    let active_value = checked_radial_add("active_count", active_tail_value, input.tail_extension)?;
    let unclamped_active_count = positive_radial_bound("active_count", active_value)?;
    let active_count = unclamped_active_count.min(input.response_capacity);

    Ok(ScreenRadialBounds {
        muffin_tin_index_1based,
        muffin_tin_next_index_1based,
        norman_index_1based,
        active_count,
    })
}

/// Port the radial-bound setup from SCREEN `getph.f90`.
///
/// `getph` uses the same Loucks-grid index helper as `screensub`, but its
/// bounds are slightly different: only `jri` is checked against `nrptx`, there
/// is no `jri + 1` reference-potential bound, and `ilast` is clamped to the
/// radial wavefunction capacity rather than a response workspace.
pub fn screen_getph_radial_bounds(
    input: ScreenGetphRadialBoundsInput,
) -> Result<ScreenGetphRadialBounds, ScreenError> {
    validate_positive("dx", input.dx)?;
    validate_finite("x0", input.x0)?;
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_positive("norman_radius", input.norman_radius)?;
    validate_count_at_least("radial_capacity", input.radial_capacity, 1)?;

    let muffin_tin_base = screen_radial_index_1based(input.x0, input.dx, input.muffin_tin_radius)?;
    let muffin_tin_value = checked_radial_add("getph_muffin_tin_index_1based", muffin_tin_base, 1)?;
    let muffin_tin_index_1based =
        positive_radial_bound("getph_muffin_tin_index_1based", muffin_tin_value)?;
    if muffin_tin_index_1based > input.radial_capacity {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "getph_muffin_tin_index_1based",
            value: muffin_tin_index_1based,
            capacity: input.radial_capacity,
        });
    }

    let norman_base = screen_radial_index_1based(input.x0, input.dx, input.norman_radius)?;
    let norman_value = checked_radial_add("getph_norman_index_1based", norman_base, 1)?;
    let norman_index_1based = positive_radial_bound("getph_norman_index_1based", norman_value)?;
    let active_value = checked_radial_add("getph_active_count", norman_value, 6)?;
    let unclamped_active_count = positive_radial_bound("getph_active_count", active_value)?;
    let active_count = unclamped_active_count.min(input.radial_capacity);

    Ok(ScreenGetphRadialBounds {
        muffin_tin_index_1based,
        norman_index_1based,
        active_count,
    })
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
