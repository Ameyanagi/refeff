//! FULLSPECTRUM edge assembly and transition blending.

use ndarray::{Array1, ArrayView1};

use crate::{Complex, Real};

use super::constants::FEFF_FULLSPECTRUM_IMAGINARY_EXIT_MULTIPLIER;
use super::types::*;
use super::validation::{validate_finite_value, validate_matching_len, validate_positive};

/// Port of `FULLSPECTRUM/addedg.f90`: assemble one edge contribution.
///
/// FEFF first reads the smooth FPRIME background, then overlays FMS/path
/// fine-structure data with separate real and imaginary transition intervals.
/// Before mixing, FEFF conjugates all scattering factors to match its optical
/// sign convention. After the seams are blended, it subtracts `fp0` from the
/// full signal and atomic background so insulating spectra satisfy `f(0)=0`.
pub fn full_spectrum_assemble_edge(
    input: FullSpectrumEdgeAssemblyInput<'_>,
) -> Result<FullSpectrumEdgeAssembly, FullSpectrumError> {
    validate_edge_assembly_input(input)?;

    let mut background = input
        .background
        .scattering_factor
        .mapv(|value| value.conj());
    let mut scattering = input
        .fine_structure
        .scattering_factor
        .mapv(|value| value.conj());
    let fine_background = input.fine_structure.background.mapv(|value| value.conj());

    let real_low = input.fine_structure.real_energy_interval[0];
    let real_high = input.fine_structure.real_energy_interval[1];
    let imaginary_low = input.fine_structure.imaginary_energy_interval[0];
    let imaginary_high = input.fine_structure.imaginary_energy_interval[1];
    let real_low_row = last_omega_le(input.omega, real_low);
    let real_high_row = last_omega_le(input.omega, real_high);
    let imaginary_low_row = last_omega_le(input.omega, imaginary_low);
    let imaginary_high_row = last_omega_le(input.omega, imaginary_high);
    let overlap_points = edge_transition_overlap(
        input.omega,
        imaginary_low_row,
        imaginary_high_row,
        imaginary_low,
        input.transition_size,
    );
    let background_overlap_points = (overlap_points / 5).max(1);

    if input.omega[0] <= real_low
        && let Some(end) = real_low_row
    {
        for row in 0..=end {
            scattering[row] = Complex::new(background[row].re, scattering[row].im);
        }
    }
    if input.omega[0] <= imaginary_low
        && let Some(end) = imaginary_low_row
    {
        for row in 0..=end {
            scattering[row] = Complex::new(scattering[row].re, 0.0);
            background[row] = Complex::new(background[row].re, 0.0);
        }
    }

    if input.omega[0] <= real_low
        && let Some(start) = real_low_row
    {
        apply_real_background_entry(
            &mut background,
            &fine_background,
            start,
            background_overlap_points,
            overlap_points,
        );
        apply_real_scattering_entry(&mut scattering, &background, start, overlap_points);
    }
    if input.omega[0] <= imaginary_low
        && let Some(start) = imaginary_low_row
    {
        apply_imaginary_background_entry(
            &mut background,
            &fine_background,
            start,
            background_overlap_points,
            overlap_points,
        );
        apply_imaginary_scattering_entry(&mut scattering, &background, start, overlap_points);
    }

    if let Some(end) = real_high_row
        && end >= overlap_points
    {
        apply_real_exit(
            &mut scattering,
            &mut background,
            &fine_background,
            end - overlap_points,
            end,
            overlap_points,
        );
    }

    let imaginary_exit_start = match (imaginary_low_row, imaginary_high_row) {
        (Some(low), Some(high)) => {
            let wide_start = high.saturating_sub(
                overlap_points.saturating_mul(FEFF_FULLSPECTRUM_IMAGINARY_EXIT_MULTIPLIER),
            );
            Some((low + overlap_points).max(wide_start))
        }
        _ => None,
    };
    if let (Some(start), Some(end)) = (imaginary_exit_start, imaginary_high_row)
        && start < end
    {
        apply_imaginary_exit(
            &mut scattering,
            &mut background,
            &fine_background,
            start,
            end,
        );
    }

    if let (Some(start), Some(end)) = (real_low_row, real_high_row) {
        let middle_start = start + overlap_points - 1;
        let middle_end = end.saturating_sub(overlap_points).saturating_add(1);
        if middle_start <= middle_end {
            for row in middle_start..=middle_end.min(background.len() - 1) {
                background[row] = Complex::new(fine_background[row].re, background[row].im);
            }
        }
    }
    if let (Some(low), Some(exit_start)) = (imaginary_low_row, imaginary_exit_start) {
        let middle_start = low + overlap_points;
        if middle_start <= exit_start {
            for row in middle_start..=exit_start.min(background.len() - 1) {
                background[row] = Complex::new(background[row].re, fine_background[row].im);
            }
        }
    }

    let real_tail_start = real_high_row.unwrap_or(0);
    for row in real_tail_start..scattering.len() {
        scattering[row] = Complex::new(background[row].re, scattering[row].im);
    }
    let imaginary_tail_start = imaginary_high_row.unwrap_or(0);
    for row in imaginary_tail_start..scattering.len() {
        scattering[row] = Complex::new(scattering[row].re, background[row].im);
    }

    let shift = Complex::new(input.background.zero_energy_fprime, 0.0);
    for row in 0..scattering.len() {
        scattering[row] -= shift;
        background[row] -= shift;
    }

    Ok(FullSpectrumEdgeAssembly {
        scattering_factor: scattering,
        background,
        effective_electron_count: input.background.effective_electron_count,
        zero_energy_fprime: input.background.zero_energy_fprime,
        overlap_points,
    })
}

fn validate_edge_assembly_input(
    input: FullSpectrumEdgeAssemblyInput<'_>,
) -> Result<(), FullSpectrumError> {
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable {
            name: "edge_assembly",
        });
    }
    validate_positive("transition_size", input.transition_size)?;
    validate_finite_value("zero_energy_fprime", 0, input.background.zero_energy_fprime)?;
    validate_finite_value(
        "effective_electron_count",
        0,
        input.background.effective_electron_count,
    )?;
    validate_matching_len(
        "background scattering_factor",
        input.background.scattering_factor.len(),
        input.omega.len(),
    )?;
    validate_matching_len(
        "fine_structure scattering_factor",
        input.fine_structure.scattering_factor.len(),
        input.omega.len(),
    )?;
    validate_matching_len(
        "fine_structure background",
        input.fine_structure.background.len(),
        input.omega.len(),
    )?;
    validate_energy_interval(
        "real_energy_interval",
        input.fine_structure.real_energy_interval,
    )?;
    validate_energy_interval(
        "imaginary_energy_interval",
        input.fine_structure.imaginary_energy_interval,
    )?;
    validate_energy_interval(
        "real_transition_interval",
        input.fine_structure.real_transition_interval,
    )?;
    validate_energy_interval(
        "imaginary_transition_interval",
        input.fine_structure.imaginary_transition_interval,
    )?;

    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("edge_assembly omega", row, value)?;
        if row > 0 && value < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: value,
            });
        }
    }
    validate_complex_grid(
        "background scattering_factor",
        input.background.scattering_factor.view(),
    )?;
    validate_complex_grid(
        "fine_structure scattering_factor",
        input.fine_structure.scattering_factor.view(),
    )?;
    validate_complex_grid(
        "fine_structure background",
        input.fine_structure.background.view(),
    )?;
    Ok(())
}

fn validate_energy_interval(
    name: &'static str,
    interval: [Real; 2],
) -> Result<(), FullSpectrumError> {
    validate_finite_value(name, 0, interval[0])?;
    validate_finite_value(name, 1, interval[1])?;
    if interval[1] < interval[0] {
        Err(FullSpectrumError::InvalidEnergyRange {
            name,
            min: interval[0],
            max: interval[1],
        })
    } else {
        Ok(())
    }
}

fn validate_complex_grid(
    field: &'static str,
    values: ArrayView1<'_, Complex>,
) -> Result<(), FullSpectrumError> {
    for (row, value) in values.iter().copied().enumerate() {
        validate_finite_value(field, row, value.re)?;
        validate_finite_value(field, row, value.im)?;
    }
    Ok(())
}

fn last_omega_le(omega: ArrayView1<'_, Real>, limit: Real) -> Option<usize> {
    omega
        .iter()
        .copied()
        .enumerate()
        .rev()
        .find_map(|(row, energy)| (energy <= limit).then_some(row))
}

fn edge_transition_overlap(
    omega: ArrayView1<'_, Real>,
    low_row: Option<usize>,
    high_row: Option<usize>,
    low_energy: Real,
    transition_size: Real,
) -> usize {
    let mut overlap_points = 5;
    if let (Some(start), Some(end)) = (low_row, high_row) {
        for row in start..=end {
            if omega[row] <= low_energy + transition_size {
                overlap_points = row - start + 1;
            }
        }
    }
    overlap_points.max(1)
}

fn apply_real_background_entry(
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    background_overlap: usize,
    overlap: usize,
) {
    if start >= background.len() {
        return;
    }
    for row in start..=(start + background_overlap).min(background.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, background_overlap);
        background[row] = Complex::new(
            background[row].re * cos_squared + fine_background[row].re * sin_squared,
            background[row].im,
        );
    }
    for row in (start + background_overlap)..=(start + overlap).min(background.len() - 1) {
        background[row] = Complex::new(fine_background[row].re, background[row].im);
    }
}

fn apply_imaginary_background_entry(
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    background_overlap: usize,
    overlap: usize,
) {
    if start >= background.len() {
        return;
    }
    for row in start..=(start + background_overlap).min(background.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, background_overlap);
        background[row] = Complex::new(
            background[row].re,
            background[row].im * cos_squared + fine_background[row].im * sin_squared,
        );
    }
    for row in (start + background_overlap)..=(start + overlap).min(background.len() - 1) {
        background[row] = Complex::new(background[row].re, fine_background[row].im);
    }
}

fn apply_real_scattering_entry(
    scattering: &mut Array1<Complex>,
    background: &Array1<Complex>,
    start: usize,
    overlap: usize,
) {
    if start >= scattering.len() {
        return;
    }
    for row in start..=(start + overlap).min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, overlap);
        scattering[row] = Complex::new(
            background[row].re * cos_squared + scattering[row].re * sin_squared,
            scattering[row].im,
        );
    }
}

fn apply_imaginary_scattering_entry(
    scattering: &mut Array1<Complex>,
    background: &Array1<Complex>,
    start: usize,
    overlap: usize,
) {
    if start >= scattering.len() {
        return;
    }
    for row in start..=(start + overlap).min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(row - start, overlap);
        scattering[row] = Complex::new(
            scattering[row].re,
            background[row].im * cos_squared + scattering[row].im * sin_squared,
        );
    }
}

fn apply_real_exit(
    scattering: &mut Array1<Complex>,
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    end: usize,
    overlap: usize,
) {
    if start >= scattering.len() {
        return;
    }
    for row in start..=end.min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(end - row, overlap);
        background[row] = Complex::new(
            background[row].re * cos_squared + fine_background[row].re * sin_squared,
            background[row].im,
        );
        scattering[row] = Complex::new(
            background[row].re * cos_squared + scattering[row].re * sin_squared,
            scattering[row].im,
        );
    }
}

fn apply_imaginary_exit(
    scattering: &mut Array1<Complex>,
    background: &mut Array1<Complex>,
    fine_background: &Array1<Complex>,
    start: usize,
    end: usize,
) {
    if start >= scattering.len() || start >= end {
        return;
    }
    let width = end - start;
    for row in start..=end.min(scattering.len() - 1) {
        let (cos_squared, sin_squared) = transition_weights(end - row, width);
        scattering[row] = Complex::new(
            scattering[row].re,
            background[row].im * cos_squared + scattering[row].im * sin_squared,
        );
        background[row] = Complex::new(
            background[row].re,
            background[row].im * cos_squared + fine_background[row].im * sin_squared,
        );
    }
}

fn transition_weights(offset: usize, width: usize) -> (Real, Real) {
    let fraction = offset as Real / width.max(1) as Real;
    let theta = fraction * std::f64::consts::FRAC_PI_2;
    (theta.cos().powi(2), theta.sin().powi(2))
}
