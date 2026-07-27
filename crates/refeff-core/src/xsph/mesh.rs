//! FEFF XSPH phase-energy mesh builders.

use ndarray::{Array1, ArrayView1};

use crate::{Complex, Real};

use super::{
    XSPH_BOHR_ANGSTROM, XSPH_HARTREE_EV, XSPH_PHASE_SORT_TOLERANCE, XsphError,
    XsphFprimeEnergyGrid84, XsphJasPhaseEnergyMesh, XsphJasPhaseEnergyMeshInput,
    XsphPhaseEnergyMesh84, XsphPhaseEnergyMesh84Input, XsphPhaseUserGridInput,
    XsphRhorrpPhaseEnergyMesh, XsphRhorrpPhaseEnergyMeshInput, XsphSortedEnergyGrid,
    XsphThermalPhaseEnergyMesh, XsphThermalPhaseEnergyMeshInput, XsphXanesEnergyGrid84,
    XsphXesEnergyGrid84, append_danes_phase_extension, append_phase_mesh_segment, nint,
    phase_mesh_count, validate_finite_complex, validate_finite_real, validate_phase_mesh_capacity,
    validate_phase_mesh_endpoint, validate_phase_mesh_step, validate_phase_user_grid_records,
    xsph_default_thermal_horizontal_grid, xsph_thermal_contour_height,
    xsph_thermal_phase_mesh_count, xsph_user_phase_horizontal_grid,
};

const JAS_PHASE_BELOW_COUNT: usize = 10;
const JAS_PHASE_EXAFS_VERTICAL_RESERVE: usize = 50;
const DANES_LEGACY_MAIN_ENERGY_COUNT: usize = 100;
const DANES_MINIMUM_VERTICAL_CONTOUR_COUNT: usize = 3;
const JAS_PHASE_SPECIAL_WAVE_NUMBERS: [Real; 9] = [
    0.0, 0.5123, 1.0123, 1.5123, 2.0123, 3.0123, 4.0123, 5.0123, 6.0123,
];

/// Port of FEFF `XSPH/phmesh2.f90` `MkEMesh`.
///
/// Returns the real-energy points FEFF writes starting at `iStart`, capped at
/// `capacity`. When `max_energy < min_energy`, FEFF reports no points; this
/// wrapper returns an empty array rather than treating that branch as an error.
pub fn xsph_even_energy_mesh(
    min_energy: Real,
    max_energy: Real,
    energy_step: Real,
    capacity: usize,
) -> Result<Array1<Complex>, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_finite_real("min_energy", min_energy)?;
    validate_finite_real("max_energy", max_energy)?;
    validate_phase_mesh_step("energy_step", energy_step)?;

    let final_offset = nint((max_energy - min_energy) / energy_step);
    if final_offset < 0 {
        return Ok(Array1::zeros(0));
    }
    let requested = phase_mesh_count(final_offset)?;
    let count = requested.min(capacity);
    Ok(Array1::from_shape_fn(count, |index| {
        Complex::new(min_energy + energy_step * index as Real, 0.0)
    }))
}

/// Port of FEFF `XSPH/phmesh2.f90` `MkKMesh`.
///
/// The returned values are energies `sign(k_min) * k^2 / 2` on a uniform
/// k-grid. FEFF returns no points when the rounded final offset is zero or
/// negative; this wrapper preserves that behavior.
pub fn xsph_k_energy_mesh(
    min_wave_number: Real,
    max_wave_number: Real,
    wave_number_step: Real,
    capacity: usize,
) -> Result<Array1<Complex>, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_finite_real("min_wave_number", min_wave_number)?;
    validate_finite_real("max_wave_number", max_wave_number)?;
    validate_phase_mesh_step("wave_number_step", wave_number_step)?;

    let final_offset = nint((max_wave_number - min_wave_number) / wave_number_step);
    if final_offset <= 0 {
        return Ok(Array1::zeros(0));
    }
    let requested = phase_mesh_count(final_offset)?;
    let sign = if min_wave_number < 0.0 { -1.0 } else { 1.0 };
    let count = requested.min(capacity);
    Ok(Array1::from_shape_fn(count, |index| {
        let wave_number = min_wave_number + wave_number_step * index as Real;
        Complex::new(sign * wave_number * wave_number / 2.0, 0.0)
    }))
}

/// Port of FEFF `XSPH/phmesh2.f90` `MkExpMesh`.
///
/// Builds the inclusive exponential mesh `emin * exp(del * i)` used by the
/// phase-grid vertical tail and DANES extension.
pub fn xsph_exponential_energy_mesh(
    min_energy: Real,
    max_energy: Real,
    exponent_step: Real,
    capacity: usize,
) -> Result<Array1<Complex>, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_phase_mesh_endpoint("min_energy", min_energy)?;
    validate_phase_mesh_endpoint("max_energy", max_energy)?;
    validate_phase_mesh_step("exponent_step", exponent_step)?;

    let final_offset = nint((max_energy / min_energy).ln() / exponent_step);
    if final_offset < 0 {
        return Ok(Array1::zeros(0));
    }
    let requested = phase_mesh_count(final_offset)?;
    let count = requested.min(capacity);
    Ok(Array1::from_shape_fn(count, |index| {
        Complex::new(min_energy * (exponent_step * index as Real).exp(), 0.0)
    }))
}

/// Port of FEFF `XSPH/phmesh2.f90` `MkVGrid84`.
///
/// Builds the vertical contour branch used by the FEFF8.4 phase mesh. FEFF
/// writes two fixed imaginary points and then an exponential imaginary tail
/// tuned so `xloss` lies midway between two neighboring tail points. This safe
/// wrapper caps the returned grid at `capacity` instead of writing past it.
pub fn xsph_vertical_energy_mesh_84(
    xloss: Real,
    capacity: usize,
) -> Result<Array1<Complex>, XsphError> {
    if capacity < 2 {
        return Err(XsphError::InvalidPhaseMeshCapacity { capacity });
    }
    validate_phase_mesh_endpoint("xloss", xloss)?;

    let first_step = Real::from(0.01_f32) / XSPH_HARTREE_EV;
    let exponent_step: Real = 0.4;
    let mut exponent_count = nint((xloss / first_step).ln() / exponent_step - 0.5);
    if exponent_count <= 0 {
        exponent_count = 1;
    }
    let exp_step = exponent_step.exp();
    let mut min_energy = 2.0 * xloss / (1.0 + exp_step) / exp_step.powi(exponent_count);
    if min_energy <= first_step {
        min_energy *= exp_step;
    }
    let max_energy = (50.0 / XSPH_HARTREE_EV).min(20.0 * xloss);

    let mut values = Vec::with_capacity(capacity);
    values.push(Complex::new(0.0, first_step / 2.0));
    values.push(Complex::new(0.0, first_step));
    if capacity > 2 {
        values.extend(
            xsph_exponential_energy_mesh(min_energy, max_energy, exponent_step, capacity - 2)?
                .iter()
                .map(|energy| Complex::new(0.0, energy.re)),
        );
    }

    Ok(Array1::from_vec(values))
}

/// Port of the vertical contour used by FEFF `XSPH/phmeshjas.f90`.
///
/// This is the same two-point-plus-exponential contour shape as `phmesh2`,
/// except JAS keeps the exponential tail up to `50 eV` in Hartree instead of
/// clipping it at `20 * xloss`.
pub fn xsph_jas_vertical_energy_mesh(xloss: Real) -> Result<Array1<Complex>, XsphError> {
    validate_phase_mesh_endpoint("xloss", xloss)?;

    let base_step = Real::from(0.01_f32) / XSPH_HARTREE_EV;
    let exponent_step = Real::from(0.4_f32);
    let mut exponent_count = nint((xloss / base_step).ln() / exponent_step - 0.5);
    if exponent_count <= 0 {
        exponent_count = 1;
    }
    let exp_step = exponent_step.exp();
    let mut min_energy = 2.0 * xloss / (1.0 + exp_step) / exp_step.powi(exponent_count);
    if min_energy <= base_step {
        min_energy *= exp_step;
    }
    if min_energy <= base_step || min_energy >= xloss {
        return Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "jas_vertical_min_energy",
            value: min_energy,
        });
    }
    let max_energy = 50.0 / XSPH_HARTREE_EV;
    validate_phase_mesh_endpoint("jas_vertical_max_energy", max_energy)?;

    let tail = xsph_exponential_energy_mesh(min_energy, max_energy, exponent_step, usize::MAX)?;
    let mut values = Vec::with_capacity(tail.len() + 2);
    values.push(Complex::new(0.0, base_step / 2.0));
    values.push(Complex::new(0.0, base_step));
    values.extend(tail.iter().map(|energy| Complex::new(0.0, energy.re)));

    Ok(Array1::from_vec(values))
}

/// Port of FEFF `XSPH/phmeshjas.f90`.
///
/// `phmeshjas` is selected for JAS/NRIXS phase generation and uses a constant
/// energy step for the main horizontal grid. In the positive-XANES branch FEFF
/// fills all `nex` horizontal slots and then appends the vertical contour past
/// that fixed array. This wrapper makes the FEFF horizontal budget explicit and
/// returns the full mesh in an owned vector, avoiding the original out-of-bounds
/// write while preserving the generated sequence.
pub fn xsph_jas_phase_energy_mesh(
    input: XsphJasPhaseEnergyMeshInput,
) -> Result<XsphJasPhaseEnergyMesh, XsphError> {
    validate_jas_phase_energy_mesh_input(input)?;

    let mut xloss = input.core_hole_broadening / 2.0 + input.constant_imaginary;
    if xloss < 0.0 {
        xloss = 0.0;
    }
    xloss = xloss.max(Real::from(0.02_f32) / XSPH_HARTREE_EV);
    validate_phase_mesh_endpoint("xloss", xloss)?;
    let near_edge_step = if input.xanes_energy_step > Real::from(0.0001_f32) {
        input.xanes_energy_step
    } else {
        xloss * 0.5
    };
    validate_phase_mesh_endpoint("jas_near_edge_step", near_edge_step)?;

    let vertical = xsph_jas_vertical_energy_mesh(xloss)?;
    let (mut values, zero_index) = if input.spectroscopy > 0 {
        jas_xanes_horizontal_grid(&input, xloss, near_edge_step)?
    } else {
        jas_exafs_horizontal_grid(&input, xloss, near_edge_step)?
    };
    let horizontal_count = values.len();
    values.extend(
        vertical
            .iter()
            .map(|energy| Complex::new(input.edge, energy.im)),
    );

    Ok(XsphJasPhaseEnergyMesh {
        energies: Array1::from_vec(values),
        horizontal_count,
        vertical_count: vertical.len(),
        zero_index,
        xloss,
    })
}

fn validate_jas_phase_energy_mesh_input(
    input: XsphJasPhaseEnergyMeshInput,
) -> Result<(), XsphError> {
    validate_phase_mesh_capacity(input.horizontal_capacity)?;
    if input.spectroscopy == 2 || input.spectroscopy >= 3 {
        return Err(XsphError::UnsupportedPhaseMeshSpectroscopy {
            spectroscopy: input.spectroscopy,
        });
    }
    validate_finite_real("edge", input.edge)?;
    validate_finite_real("constant_imaginary", input.constant_imaginary)?;
    validate_finite_real("core_hole_broadening", input.core_hole_broadening)?;
    validate_finite_real("core_valence_separation", input.core_valence_separation)?;
    validate_phase_mesh_endpoint("max_wave_number", input.max_wave_number)?;
    validate_phase_mesh_endpoint("wave_number_step", input.wave_number_step)?;
    validate_finite_real("xanes_energy_step", input.xanes_energy_step)?;
    Ok(())
}

fn jas_xanes_horizontal_grid(
    input: &XsphJasPhaseEnergyMeshInput,
    xloss: Real,
    near_edge_step: Real,
) -> Result<(Vec<Complex>, usize), XsphError> {
    if input.horizontal_capacity <= JAS_PHASE_BELOW_COUNT {
        return Err(XsphError::InvalidPhaseMeshCapacity {
            capacity: input.horizontal_capacity,
        });
    }

    let mut values =
        jas_below_edge_grid(input.edge, xloss, near_edge_step, input.wave_number_step)?;
    let above_count = input.horizontal_capacity - JAS_PHASE_BELOW_COUNT;
    let energy_step = input.max_wave_number * input.max_wave_number / 2.0 / above_count as Real;
    validate_phase_mesh_endpoint("jas_xanes_energy_step", energy_step)?;
    values.extend(
        (0..above_count).map(|index| Complex::new(input.edge + energy_step * index as Real, xloss)),
    );
    Ok((values, JAS_PHASE_BELOW_COUNT))
}

fn jas_exafs_horizontal_grid(
    input: &XsphJasPhaseEnergyMeshInput,
    xloss: Real,
    near_edge_step: Real,
) -> Result<(Vec<Complex>, usize), XsphError> {
    let mut values = if input.spectroscopy < 0 {
        jas_below_edge_grid(input.edge, xloss, near_edge_step, input.wave_number_step)?
    } else {
        Vec::new()
    };
    let retained_below_count = values.len();
    let reserved = JAS_PHASE_EXAFS_VERTICAL_RESERVE
        .checked_add(retained_below_count)
        .ok_or(XsphError::SizeOutOfRange {
            name: "jas_phase_reserved_count",
            value: retained_below_count,
        })?;
    let generated_count = input.horizontal_capacity.checked_sub(reserved).ok_or(
        XsphError::InvalidPhaseMeshCapacity {
            capacity: input.horizontal_capacity,
        },
    )?;
    if generated_count <= JAS_PHASE_SPECIAL_WAVE_NUMBERS.len() {
        return Err(XsphError::InvalidPhaseMeshCapacity {
            capacity: input.horizontal_capacity,
        });
    }

    let denominator = generated_count - JAS_PHASE_SPECIAL_WAVE_NUMBERS.len();
    let energy_step = input.max_wave_number * input.max_wave_number / 2.0 / denominator as Real;
    validate_phase_mesh_endpoint("jas_exafs_energy_step", energy_step)?;
    append_jas_exafs_generated_grid(
        &mut values,
        input.edge,
        xloss,
        input.max_wave_number,
        energy_step,
        generated_count,
    )?;
    Ok((values, 0))
}

fn jas_below_edge_grid(
    edge: Real,
    xloss: Real,
    near_edge_step: Real,
    wave_number_step: Real,
) -> Result<Vec<Complex>, XsphError> {
    let wave_step = 2.0 * wave_number_step;
    validate_phase_mesh_endpoint("jas_below_wave_step", wave_step)?;
    let mut energy_count = (near_edge_step / (2.0 * wave_step * wave_step)).trunc() as i32;
    let wave_start =
        ((f64::from(energy_count) * 2.0 * near_edge_step).sqrt() / wave_step).trunc() as i32;
    if (wave_step * f64::from(wave_start + 1)).powi(2)
        > f64::from(energy_count + 1) * 2.0 * near_edge_step
    {
        energy_count += 1;
    }
    let energy_count = energy_count.clamp(0, JAS_PHASE_BELOW_COUNT as i32) as usize;
    let wave_count = JAS_PHASE_BELOW_COUNT - energy_count;

    let mut values = vec![Complex::new(0.0, 0.0); JAS_PHASE_BELOW_COUNT];
    for index in 1..=energy_count {
        values[JAS_PHASE_BELOW_COUNT - index] =
            Complex::new(edge - near_edge_step * index as Real, xloss);
    }
    for index in 1..=wave_count {
        let wave_number = wave_step * (f64::from(wave_start) + index as Real);
        values[wave_count - index] = Complex::new(edge - wave_number * wave_number / 2.0, xloss);
    }
    Ok(values)
}

fn append_jas_exafs_generated_grid(
    values: &mut Vec<Complex>,
    edge: Real,
    xloss: Real,
    max_wave_number: Real,
    energy_step: Real,
    generated_count: usize,
) -> Result<(), XsphError> {
    let special_wave_numbers = jas_phase_special_wave_numbers();
    let missing_count = jas_phase_missing_special_count(max_wave_number, &special_wave_numbers);
    let regular_limit =
        generated_count
            .checked_sub(missing_count)
            .ok_or(XsphError::InvalidPhaseMeshCapacity {
                capacity: generated_count,
            })?;
    let mut regular_index = 0usize;
    let mut special_index = 1usize;

    for _ in 0..regular_limit {
        let candidate_index = regular_index + 1;
        let wave_number = (2.0 * energy_step * regular_index as Real).sqrt();
        if special_index < special_wave_numbers.len()
            && wave_number > special_wave_numbers[special_index]
        {
            values.push(Complex::new(
                edge + special_wave_numbers[special_index].powi(2) / 2.0,
                xloss,
            ));
            special_index += 1;
        } else {
            values.push(Complex::new(
                edge + energy_step * (candidate_index - 1) as Real,
                xloss,
            ));
            regular_index = candidate_index;
        }
    }

    let first_missing_index = special_wave_numbers.len() - missing_count;
    for &wave_number in special_wave_numbers.iter().skip(first_missing_index) {
        values.push(Complex::new(edge + wave_number * wave_number / 2.0, xloss));
    }

    Ok(())
}

fn jas_phase_special_wave_numbers() -> [Real; 9] {
    JAS_PHASE_SPECIAL_WAVE_NUMBERS.map(|wave_number| wave_number * XSPH_BOHR_ANGSTROM)
}

fn jas_phase_missing_special_count(
    max_wave_number: Real,
    special_wave_numbers: &[Real; 9],
) -> usize {
    let mut last_below = 1usize;
    let abs_wave_number = max_wave_number.abs();
    for (index, &special) in special_wave_numbers.iter().enumerate() {
        if abs_wave_number > special {
            last_below = index + 1;
        }
    }
    special_wave_numbers.len() - last_below
}

/// Port of FEFF `XSPH/phmesh2.f90` `ExafsGrid84`.
///
/// Builds the legacy FEFF8.4 EXAFS horizontal phase mesh. `max_wave_number`
/// uses FEFF's internal inverse-Bohr wave-number units, matching the Fortran
/// `xkmax` argument. The returned grid is capped at `min(capacity, 100)`,
/// preserving FEFF's hard grid limit without allowing out-of-bounds writes.
pub fn xsph_exafs_energy_grid_84(
    max_wave_number: Real,
    capacity: usize,
) -> Result<Array1<Complex>, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_phase_mesh_endpoint("max_wave_number", max_wave_number)?;

    let limit = capacity.min(100);
    let mut values = Vec::with_capacity(limit);

    let first_step = XSPH_BOHR_ANGSTROM / 10.0;
    let segment = xsph_k_energy_mesh(
        0.0,
        XSPH_BOHR_ANGSTROM * 1.9 + first_step * 0.01,
        first_step,
        limit.saturating_sub(values.len()),
    )?;
    append_phase_mesh_segment(&mut values, limit, segment);

    let second_step = XSPH_BOHR_ANGSTROM / 5.0;
    let segment = xsph_k_energy_mesh(
        XSPH_BOHR_ANGSTROM * 2.0,
        XSPH_BOHR_ANGSTROM * 5.8 + second_step * 0.01,
        second_step,
        limit.saturating_sub(values.len()),
    )?;
    append_phase_mesh_segment(&mut values, limit, segment);

    let third_step = XSPH_BOHR_ANGSTROM * 0.5;
    let segment = xsph_k_energy_mesh(
        XSPH_BOHR_ANGSTROM * 6.0,
        XSPH_BOHR_ANGSTROM * 10.0 + second_step * 0.01,
        third_step,
        limit.saturating_sub(values.len()),
    )?;
    append_phase_mesh_segment(&mut values, limit, segment);

    if values.len() < limit {
        let final_step = XSPH_BOHR_ANGSTROM;
        if let Some(&last_energy) = values.last() {
            let min_wave_number = (2.0 * last_energy.re).sqrt() + final_step;
            let requested_count = nint((max_wave_number - min_wave_number) / final_step) + 1;
            if requested_count > 0 {
                let next_index_1based = values.len() + 1;
                let count = (requested_count as usize).min(limit.saturating_sub(next_index_1based));
                if count > 0 {
                    let max_segment_wave_number =
                        min_wave_number + (count as Real - 1.0) * final_step + final_step * 0.01;
                    let segment = xsph_k_energy_mesh(
                        min_wave_number,
                        max_segment_wave_number,
                        final_step,
                        count,
                    )?;
                    append_phase_mesh_segment(&mut values, limit, segment);
                }
            }
        }
    }

    Ok(Array1::from_vec(values))
}

/// Port of FEFF `XSPH/phmesh2.f90` `XanesGrid84` for XANES/DANES grids.
///
/// Builds the legacy FEFF8.4 horizontal grid used by XANES and DANES before
/// edge shifting and vertical-contour insertion. Wave-number inputs use FEFF's
/// internal inverse-Bohr units, matching the Fortran `xkmax` and `xkstep`
/// arguments. The near-edge segment must fit so the returned `zero_index`
/// always points at FEFF's Fermi-level grid point; the high-energy tail is
/// capped at `capacity`.
pub fn xsph_xanes_energy_grid_84(
    max_wave_number: Real,
    wave_number_step: Real,
    energy_step: Real,
    capacity: usize,
) -> Result<XsphXanesEnergyGrid84, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_phase_mesh_endpoint("max_wave_number", max_wave_number)?;
    validate_phase_mesh_endpoint("wave_number_step", wave_number_step)?;
    validate_phase_mesh_endpoint("energy_step", energy_step)?;

    let below_limit: i32 = 10;
    let below_step = 2.0 * wave_number_step;
    let mut below_energy_count = (energy_step / (2.0 * below_step * below_step)).trunc() as i32;
    let below_start_index =
        ((f64::from(below_energy_count) * 2.0 * energy_step).sqrt() / below_step).trunc() as i32;
    if (below_step * f64::from(below_start_index + 1)).powi(2)
        > f64::from(below_energy_count + 1) * 2.0 * energy_step
    {
        below_energy_count += 1;
    }
    below_energy_count = below_energy_count.min(below_limit);
    let below_wave_count = below_limit - below_energy_count;

    let below_wave_min = -below_step * f64::from(below_start_index + below_wave_count);
    let below_wave_max = -below_step * f64::from(below_start_index + 1);
    let below = xsph_k_energy_mesh(
        below_wave_min,
        below_wave_max,
        below_step,
        below_limit as usize,
    )?;
    let near_edge = xsph_even_energy_mesh(
        -energy_step * f64::from(below_energy_count),
        0.0,
        energy_step,
        phase_mesh_count(below_energy_count)?,
    )?;
    let fixed_count = below.len() + near_edge.len();
    if capacity < fixed_count {
        return Err(XsphError::InvalidPhaseMeshCapacity { capacity });
    }

    let mut values = Vec::with_capacity(capacity);
    append_phase_mesh_segment(&mut values, capacity, below);
    append_phase_mesh_segment(&mut values, capacity, near_edge);
    let zero_index = values.len().saturating_sub(1);

    let next_index_1based = values.len() + 1;
    let above_limit = capacity.saturating_sub(next_index_1based);
    if above_limit > 0 {
        let above_limit_i32 =
            i32::try_from(above_limit).map_err(|_| XsphError::SizeOutOfRange {
                name: "phase_mesh_capacity",
                value: above_limit,
            })?;
        let above_base_count =
            (energy_step / (2.0 * wave_number_step * wave_number_step)).trunc() as i32;
        let above_start_index = ((f64::from(above_base_count) * 2.0 * energy_step).sqrt()
            / wave_number_step)
            .trunc() as i32;
        let mut above_energy_count = above_base_count + 1;
        if (wave_number_step * f64::from(above_start_index + 1)).powi(2)
            > f64::from(above_energy_count + 1) * 2.0 * energy_step
        {
            above_energy_count += 1;
        }
        above_energy_count = above_energy_count.min(above_limit_i32);
        let maximum_energy_count =
            (max_wave_number * max_wave_number / energy_step / 2.0).trunc() as i32 + 1;
        if maximum_energy_count <= above_energy_count {
            above_energy_count = maximum_energy_count;
        }
        let mut above_wave_count = above_limit_i32 - above_energy_count;

        let mut above_energy_max = f64::from(above_energy_count - 1) * energy_step;
        if (2.0 * above_energy_max).sqrt() > max_wave_number {
            above_energy_max = max_wave_number * max_wave_number / 2.0;
            above_wave_count = 0;
        }
        let segment = xsph_even_energy_mesh(
            energy_step,
            above_energy_max,
            energy_step,
            capacity.saturating_sub(values.len()),
        )?;
        append_phase_mesh_segment(&mut values, capacity, segment);

        let above_wave_min = wave_number_step * f64::from(above_start_index + 1);
        let mut above_wave_max = wave_number_step * f64::from(above_start_index + above_wave_count);
        if above_wave_max > max_wave_number {
            above_wave_max = max_wave_number;
        }
        let segment = xsph_k_energy_mesh(
            above_wave_min,
            above_wave_max,
            wave_number_step,
            capacity.saturating_sub(values.len()),
        )?;
        append_phase_mesh_segment(&mut values, capacity, segment);
    }

    Ok(XsphXanesEnergyGrid84 {
        energies: Array1::from_vec(values),
        zero_index,
    })
}

/// Port of the FEFF `XSPH/phmesh2.f90` default XES horizontal grid.
///
/// FEFF reuses the `xkmax` and `xkstep` arguments as lower and upper XES
/// energy bounds, converts them through `/(bohr*hart) - edge`, builds a regular
/// energy grid with the XANES step, and then applies `SortE`. This helper
/// returns the unshifted horizontal grid; callers add `edge + i*xloss`.
pub fn xsph_xes_energy_grid_84(
    min_energy_bound: Real,
    max_energy_bound: Real,
    energy_step: Real,
    edge: Real,
    capacity: usize,
) -> Result<XsphXesEnergyGrid84, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_finite_real("min_energy_bound", min_energy_bound)?;
    validate_finite_real("max_energy_bound", max_energy_bound)?;
    validate_finite_real("edge", edge)?;
    validate_phase_mesh_step("energy_step", energy_step)?;

    let min_energy = min_energy_bound / XSPH_BOHR_ANGSTROM / XSPH_HARTREE_EV - edge;
    let max_energy = max_energy_bound / XSPH_BOHR_ANGSTROM / XSPH_HARTREE_EV - edge;
    validate_finite_real("xes_min_energy", min_energy)?;
    validate_finite_real("xes_max_energy", max_energy)?;

    let unsorted = xsph_even_energy_mesh(min_energy, max_energy, energy_step, capacity)?;
    xsph_sort_energy_grid(unsorted.view()).map(|sorted| XsphXesEnergyGrid84 {
        energies: sorted.energies,
        zero_index: sorted.zero_index,
    })
}

/// Port of FEFF `XSPH/phmesh2.f90` `FPrimeGrid84`.
///
/// Builds the legacy FEFF8.4 FPRIME mesh. `min_energy` and `max_energy` are
/// the FPRIME card bounds before FEFF's `/(bohr*hart) - emu` conversion;
/// `energy_step`, `edge`, and `reference_energy` are already in Hartree.
/// FEFF allows a nonpositive `energy_step` to request an automatic regular
/// step, and this wrapper preserves that branch. Returned data are capped at
/// `capacity` instead of writing past FEFF's fixed `em` buffer.
pub fn xsph_fprime_energy_grid_84(
    min_energy: Real,
    max_energy: Real,
    energy_step: Real,
    reference_energy: Real,
    edge: Real,
    capacity: usize,
) -> Result<XsphFprimeEnergyGrid84, XsphError> {
    validate_phase_mesh_capacity(capacity)?;
    validate_finite_real("min_energy", min_energy)?;
    validate_finite_real("max_energy", max_energy)?;
    validate_finite_real("energy_step", energy_step)?;
    validate_finite_real("reference_energy", reference_energy)?;
    validate_finite_real("edge", edge)?;

    let regular_min = min_energy / XSPH_BOHR_ANGSTROM / XSPH_HARTREE_EV - reference_energy;
    let regular_max = max_energy / XSPH_BOHR_ANGSTROM / XSPH_HARTREE_EV - reference_energy;
    validate_finite_real("regular_min", regular_min)?;
    validate_finite_real("regular_max", regular_max)?;

    let mut values = Vec::with_capacity(capacity);
    if regular_min < regular_max {
        let mut step = energy_step;
        if step <= 0.0 {
            step = (regular_max - regular_min) / 99.0;
        }
        validate_phase_mesh_endpoint("fprime_energy_step", step)?;
        let requested_count = phase_mesh_count(nint((regular_max - regular_min) / step))?;
        let count = requested_count.min(100).min(capacity);
        values
            .extend((0..count).map(|index| Complex::new(regular_min + step * index as Real, 0.0)));
    } else if capacity > 0 {
        values.push(Complex::new(regular_min, 0.0));
    }
    let regular_count = values.len();

    let kk_count = capacity.saturating_sub(regular_count).min(100);
    if kk_count > 0 {
        let delta = 3.0 / XSPH_HARTREE_EV;
        let limit = (1000.0 / XSPH_HARTREE_EV)
            .max((20.0 * reference_energy).min(200_000.0 / XSPH_HARTREE_EV))
            - reference_energy;
        validate_phase_mesh_endpoint("fprime_limit", limit)?;

        let mut previous = edge;
        values.push(Complex::new(previous, 0.0));
        for index in 1..kk_count {
            let scaled_step = if previous > 0.0 {
                previous * ((limit / previous).ln() / (kk_count - index) as Real).exp_m1()
            } else {
                0.0
            };
            previous += delta.max(scaled_step);
            values.push(Complex::new(previous, 0.0));
        }
    }

    Ok(XsphFprimeEnergyGrid84 {
        energies: Array1::from_vec(values),
        regular_count,
        kk_count,
    })
}

/// Port of the default-grid branch of FEFF `XSPH/phmesh2.f90`.
///
/// This composes the FEFF8.4 horizontal-grid helpers, applies the FEFF
/// `edge + i*xloss` horizontal shift, appends the vertical contour for
/// EXAFS/XANES/XES/DANES and negative no-FMS spectra, and preserves the FPRIME
/// KK-extension branch. User `grid.inp` meshes and the finite-temperature
/// `phmesh2T` path are separate routines and intentionally not folded into this
/// default FEFF84 adapter.
pub fn xsph_phase_energy_mesh_84(
    input: XsphPhaseEnergyMesh84Input,
) -> Result<XsphPhaseEnergyMesh84, XsphError> {
    validate_phase_mesh_capacity(input.capacity)?;
    validate_finite_real("edge", input.edge)?;
    validate_finite_real("reference_energy", input.reference_energy)?;
    validate_finite_real("constant_imaginary", input.constant_imaginary)?;
    validate_finite_real("core_hole_broadening", input.core_hole_broadening)?;
    validate_finite_real("core_valence_separation", input.core_valence_separation)?;
    validate_finite_real("max_wave_number", input.max_wave_number)?;
    validate_finite_real("wave_number_step", input.wave_number_step)?;
    validate_finite_real("xanes_energy_step", input.xanes_energy_step)?;

    let xloss =
        (input.core_hole_broadening / 2.0 + input.constant_imaginary).max(0.02 / XSPH_HARTREE_EV);
    validate_phase_mesh_endpoint("xloss", xloss)?;

    if input.spectroscopy == 4 {
        let fprime = xsph_fprime_energy_grid_84(
            input.max_wave_number,
            input.wave_number_step,
            input.xanes_energy_step,
            input.reference_energy,
            input.edge,
            input.capacity,
        )?;
        return Ok(XsphPhaseEnergyMesh84 {
            energies: fprime.energies,
            horizontal_count: fprime.regular_count,
            extension_count: fprime.kk_count,
            zero_index: 0,
            xloss,
        });
    }

    let horizontal_step = if input.xanes_energy_step > 0.0001 {
        input.xanes_energy_step
    } else {
        xloss / 2.0
    };

    let full_vertical = xsph_vertical_energy_mesh_84(xloss, input.capacity)?;
    let (mut horizontal, zero_index) = match input.spectroscopy {
        0 => (
            xsph_exafs_energy_grid_84(input.max_wave_number, input.capacity)?,
            0,
        ),
        1 | 3 => {
            let grid = xsph_xanes_energy_grid_84(
                input.max_wave_number,
                input.wave_number_step,
                horizontal_step,
                input.capacity,
            )?;
            (grid.energies, grid.zero_index)
        }
        2 => {
            let reserved_capacity = input
                .capacity
                .saturating_sub(full_vertical.len().saturating_add(1));
            let grid = xsph_xes_energy_grid_84(
                input.max_wave_number,
                input.wave_number_step,
                horizontal_step,
                input.edge,
                reserved_capacity,
            )?;
            (grid.energies, grid.zero_index)
        }
        -3..=-1 => {
            let xanes = xsph_xanes_energy_grid_84(
                input.max_wave_number,
                input.wave_number_step,
                horizontal_step,
                input.capacity,
            )?;
            let exafs = xsph_exafs_energy_grid_84(
                input.max_wave_number,
                input.capacity.saturating_sub(xanes.zero_index),
            )?;
            let mut no_fms = Vec::with_capacity(input.capacity);
            no_fms.extend(xanes.energies.iter().take(xanes.zero_index).copied());
            no_fms.extend(
                exafs
                    .iter()
                    .take(input.capacity.saturating_sub(no_fms.len()))
                    .copied(),
            );
            (Array1::from_vec(no_fms), xanes.zero_index)
        }
        spectroscopy => {
            return Err(XsphError::UnsupportedPhaseMeshSpectroscopy { spectroscopy });
        }
    };

    // FEFF's working DANES mesh keeps the legacy 100-point main branch when
    // an unconstrained XANES grid would consume the contour slots. Preserve
    // ordinary BN/Cu grids, and only recover the starvation case.
    if input.spectroscopy.abs() == 3
        && input.capacity.saturating_sub(horizontal.len()) < DANES_MINIMUM_VERTICAL_CONTOUR_COUNT
        && horizontal.len() > DANES_LEGACY_MAIN_ENERGY_COUNT
    {
        horizontal = Array1::from_iter(
            horizontal
                .iter()
                .take(DANES_LEGACY_MAIN_ENERGY_COUNT)
                .copied(),
        );
    }

    let mut values: Vec<_> = horizontal
        .iter()
        .map(|&energy| energy + Complex::new(input.edge, xloss))
        .collect();
    let horizontal_count = values.len();
    let vertical_capacity = input.capacity.saturating_sub(horizontal_count);
    if input.spectroscopy.abs() == 3 && vertical_capacity < DANES_MINIMUM_VERTICAL_CONTOUR_COUNT {
        return Err(XsphError::InsufficientDanesVerticalContourPoints {
            points: vertical_capacity,
        });
    }
    let vertical = if input.spectroscopy == 2 {
        full_vertical
    } else {
        xsph_vertical_energy_mesh_84(xloss, vertical_capacity)?
    };
    values.extend(
        vertical
            .iter()
            .map(|&energy| energy + Complex::new(input.edge, 0.0)),
    );

    let extension_count = if input.spectroscopy.abs() == 3 {
        append_danes_phase_extension(&mut values, horizontal_count, input.capacity)?
    } else {
        0
    };

    Ok(XsphPhaseEnergyMesh84 {
        energies: Array1::from_vec(values),
        horizontal_count,
        extension_count,
        zero_index,
        xloss,
    })
}

/// Port of FEFF `XSPH/phmesh2.f90` `mk_rhorrp_grid`.
///
/// FEFF uses this branch for NRIXS/RHORRP-style XSPH meshes (`ispec = 5`).
/// The grid contains a ten-point quadratic vertical leg from `ecv`, a capped
/// horizontal contour leg at the selected imaginary height, and Matsubara poles
/// at the shifted edge. `ik0` is left at zero by the caller, so this routine
/// returns only the FEFF `ne1` contour count and pole count.
pub fn xsph_rhorrp_phase_energy_mesh(
    input: XsphRhorrpPhaseEnergyMeshInput,
) -> Result<XsphRhorrpPhaseEnergyMesh, XsphError> {
    validate_phase_mesh_capacity(input.capacity)?;
    validate_finite_real("edge", input.edge)?;
    validate_finite_real("core_valence_separation", input.core_valence_separation)?;
    validate_finite_real("scf_temperature", input.scf_temperature)?;

    let mut temperature = input.scf_temperature / XSPH_HARTREE_EV;
    if temperature < 0.001 {
        temperature = 0.001;
    }
    validate_phase_mesh_endpoint("rhorrp_temperature", temperature)?;

    let minimum_upper_imaginary = 0.05;
    let base_pole_spacing = 2.0 * std::f64::consts::PI * temperature;
    validate_phase_mesh_endpoint("rhorrp_pole_spacing", base_pole_spacing)?;
    let mut pole_count = 1_usize;
    let mut upper_imaginary = base_pole_spacing;
    if upper_imaginary < minimum_upper_imaginary {
        pole_count = (minimum_upper_imaginary / base_pole_spacing).ceil() as usize;
        upper_imaginary = pole_count as Real * base_pole_spacing;
    }
    validate_phase_mesh_endpoint("rhorrp_upper_imaginary", upper_imaginary)?;

    let maximum_energy = input.edge + 10.0 * temperature;
    validate_finite_real("rhorrp_maximum_energy", maximum_energy)?;
    ensure_rhorrp_span(input.core_valence_separation, maximum_energy)?;

    let vertical_count = 10_usize;
    let horizontal_step_trial = upper_imaginary / 4.0;
    validate_phase_mesh_step("rhorrp_horizontal_step_trial", horizontal_step_trial)?;
    let horizontal_count = ((maximum_energy - input.core_valence_separation)
        / horizontal_step_trial)
        .ceil()
        .max(1.0) as usize;
    let horizontal_count = horizontal_count.min(101);
    let horizontal_step =
        (maximum_energy - input.core_valence_separation) / horizontal_count as Real;
    validate_phase_mesh_step("rhorrp_horizontal_step", horizontal_step)?;

    let contour_count = vertical_count + horizontal_count;
    let total_count =
        contour_count
            .checked_add(pole_count)
            .ok_or(XsphError::InvalidPhaseMeshCapacity {
                capacity: input.capacity,
            })?;
    if total_count > input.capacity {
        return Err(XsphError::InvalidPhaseMeshCapacity {
            capacity: input.capacity,
        });
    }

    let mut values = Vec::with_capacity(total_count);
    let vertical_step = upper_imaginary / (vertical_count * vertical_count) as Real;
    for index in 1..=vertical_count {
        values.push(Complex::new(
            input.core_valence_separation,
            vertical_step * (index * index) as Real,
        ));
    }
    for index in 1..=horizontal_count {
        values.push(Complex::new(
            input.core_valence_separation + index as Real * horizontal_step,
            upper_imaginary,
        ));
    }
    for index in 1..=pole_count {
        values.push(Complex::new(
            input.edge,
            (2 * index - 1) as Real * std::f64::consts::PI * temperature,
        ));
    }

    Ok(XsphRhorrpPhaseEnergyMesh {
        energies: Array1::from_vec(values),
        contour_count,
        pole_count,
        temperature,
        upper_imaginary,
    })
}

fn ensure_rhorrp_span(
    core_valence_separation: Real,
    maximum_energy: Real,
) -> Result<(), XsphError> {
    if maximum_energy <= core_valence_separation {
        return Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "rhorrp_maximum_energy",
            value: maximum_energy,
        });
    }
    Ok(())
}

/// Port of the user-grid branch of FEFF `XSPH/phmesh2.f90`.
///
/// This composes parsed `grid.inp` records with FEFF's `RdGrid` unit
/// conversions, `last` continuation rules, `SortE` ordering, horizontal
/// `edge + i*xloss` shift, vertical contour insertion, and DANES extension.
/// The returned shape and counters match [`xsph_phase_energy_mesh_84`], while
/// the horizontal grid comes from caller-supplied `grid.inp` records. FEFF also
/// lets `ispec = 5` use this generic user-grid branch, in which case the sorted
/// horizontal mesh is kept unshifted and no vertical contour is appended.
pub fn xsph_phase_energy_mesh_user(
    input: XsphPhaseUserGridInput<'_>,
) -> Result<XsphPhaseEnergyMesh84, XsphError> {
    validate_phase_mesh_capacity(input.capacity)?;
    validate_finite_real("edge", input.edge)?;
    validate_finite_real("constant_imaginary", input.constant_imaginary)?;
    validate_finite_real("core_hole_broadening", input.core_hole_broadening)?;

    validate_phase_user_grid_records(input.records)?;

    let spectroscopy_abs =
        input
            .spectroscopy
            .checked_abs()
            .ok_or(XsphError::IntegerOutOfRange {
                name: "spectroscopy",
                value: input.spectroscopy,
            })?;
    if spectroscopy_abs > 5 {
        return Err(XsphError::UnsupportedPhaseMeshSpectroscopy {
            spectroscopy: input.spectroscopy,
        });
    }

    let xloss =
        (input.core_hole_broadening / 2.0 + input.constant_imaginary).max(0.02 / XSPH_HARTREE_EV);
    validate_phase_mesh_endpoint("xloss", xloss)?;

    let vertical_capacity = xsph_vertical_energy_mesh_84(xloss, input.capacity)?;
    let horizontal_limit = input
        .capacity
        .saturating_sub(vertical_capacity.len().saturating_add(1));
    validate_phase_mesh_capacity(horizontal_limit)?;

    let mut horizontal = xsph_user_phase_horizontal_grid(input.records, horizontal_limit)?;
    if horizontal.len() + 1 < input.capacity {
        horizontal.push(Complex::new(0.0, 0.0));
    } else if let Some(first) = horizontal.first_mut() {
        *first = Complex::new(0.0, 0.0);
    }

    let sorted = xsph_sort_energy_grid(ArrayView1::from(horizontal.as_slice()))?;
    let mut values: Vec<_> = sorted
        .energies
        .iter()
        .map(|&energy| {
            if spectroscopy_abs < 4 {
                energy + Complex::new(input.edge, xloss)
            } else {
                energy
            }
        })
        .collect();
    let horizontal_count = values.len();

    if spectroscopy_abs <= 3 {
        values.extend(
            vertical_capacity
                .iter()
                .map(|&energy| energy + Complex::new(input.edge, 0.0)),
        );
    }

    let extension_count = if spectroscopy_abs == 3 {
        append_danes_phase_extension(&mut values, horizontal_count, input.capacity)?
    } else {
        0
    };

    Ok(XsphPhaseEnergyMesh84 {
        energies: Array1::from_vec(values),
        horizontal_count,
        extension_count,
        zero_index: sorted.zero_index,
        xloss,
    })
}

/// Port of the active normal branch of FEFF `XSPH/phmesh2T.f90`.
///
/// FEFF currently sets `normal_mesh = .TRUE.` before dispatching the thermal
/// mesh builder. This routine covers that active branch for both the default
/// thermal contour and the `grid.inp` user-grid contour, including the two
/// horizontal legs, ten-point vertical leg, Matsubara poles, and the fake
/// zero-temperature pole used by downstream code.
pub fn xsph_thermal_phase_energy_mesh(
    input: XsphThermalPhaseEnergyMeshInput<'_>,
) -> Result<XsphThermalPhaseEnergyMesh, XsphError> {
    validate_phase_mesh_capacity(input.capacity)?;
    validate_finite_real("edge", input.edge)?;
    validate_finite_real("constant_imaginary", input.constant_imaginary)?;
    validate_finite_real("core_hole_broadening", input.core_hole_broadening)?;
    validate_finite_real("core_valence_separation", input.core_valence_separation)?;
    validate_phase_mesh_endpoint("electronic_temperature", input.electronic_temperature)?;

    let xloss =
        (input.core_hole_broadening / 2.0 + input.constant_imaginary).max(0.02 / XSPH_HARTREE_EV);
    validate_phase_mesh_endpoint("xloss", xloss)?;

    let temperature = input.electronic_temperature / XSPH_HARTREE_EV;
    validate_phase_mesh_endpoint("thermal_temperature", temperature)?;
    let (pole_count, upper_imaginary) = xsph_thermal_contour_height(temperature)?;

    let (horizontal, zero_index, horizontal_shift) = if let Some(records) = input.user_records {
        validate_phase_user_grid_records(records)?;
        let horizontal = xsph_user_phase_horizontal_grid(records, input.capacity)?;
        let sorted = xsph_sort_energy_grid(ArrayView1::from(horizontal.as_slice()))?;
        (sorted.energies, sorted.zero_index, input.edge)
    } else {
        let horizontal =
            xsph_default_thermal_horizontal_grid(input.core_valence_separation, upper_imaginary)?;
        let zero_index = horizontal
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.re.abs().total_cmp(&right.re.abs()))
            .map_or(0, |(index, _)| index);
        (horizontal, zero_index, 0.0)
    };

    let horizontal_count = horizontal.len();
    let total_count = xsph_thermal_phase_mesh_count(horizontal_count, pole_count)?;
    if total_count > input.capacity {
        return Err(XsphError::InvalidPhaseMeshCapacity {
            capacity: input.capacity,
        });
    }

    let mut values = Vec::with_capacity(total_count);
    values.extend(
        horizontal
            .iter()
            .map(|energy| Complex::new(energy.re + horizontal_shift, upper_imaginary)),
    );
    values.extend(
        horizontal
            .iter()
            .map(|energy| Complex::new(energy.re + horizontal_shift, xloss)),
    );

    let vertical_count = 10_usize;
    let vertical_step = upper_imaginary / (vertical_count * vertical_count) as Real;
    values.extend((1..=vertical_count).map(|index| {
        Complex::new(
            input.core_valence_separation,
            vertical_step * (index * index) as Real,
        )
    }));

    values.extend((1..=pole_count).map(|index| {
        Complex::new(
            input.edge,
            (2 * index - 1) as Real * std::f64::consts::PI * temperature,
        )
    }));
    values.push(Complex::new(
        input.edge,
        Real::from(0.01_f32) / XSPH_HARTREE_EV / 2.0,
    ));

    Ok(XsphThermalPhaseEnergyMesh {
        energies: Array1::from_vec(values),
        horizontal_count,
        pole_count,
        zero_index,
        xloss,
        upper_imaginary,
    })
}

/// Port of FEFF `XSPH/phmesh2.f90` `ReverseGrid`.
///
/// FEFF first reflects every point about `zero_point` and then reverses the
/// array in place. This wrapper returns the transformed grid.
pub fn xsph_reverse_energy_grid(
    energies: ArrayView1<'_, Complex>,
    zero_point: Real,
) -> Result<Array1<Complex>, XsphError> {
    validate_finite_real("zero_point", zero_point)?;
    for (index, &energy) in energies.iter().enumerate() {
        validate_finite_complex("energies", index, energy)?;
    }
    Ok(Array1::from_iter(
        energies
            .iter()
            .rev()
            .map(|&energy| Complex::new(zero_point, 0.0) - energy),
    ))
}

/// Port of FEFF `XSPH/phmesh2.f90` `SortE`.
///
/// Sorts by real energy, drops imaginary parts, removes points within FEFF's
/// fixed `0.001` tolerance of the last retained point, and snaps the remaining
/// point closest to zero to exactly zero.
pub fn xsph_sort_energy_grid(
    energies: ArrayView1<'_, Complex>,
) -> Result<XsphSortedEnergyGrid, XsphError> {
    if energies.is_empty() {
        return Err(XsphError::EmptyPhaseMesh);
    }
    for (index, &energy) in energies.iter().enumerate() {
        validate_finite_complex("energies", index, energy)?;
    }

    let real_energies: Vec<_> = energies.iter().map(|energy| energy.re).collect();
    let mut order: Vec<_> = (0..real_energies.len()).collect();
    order.sort_by(|&left, &right| {
        real_energies[left]
            .total_cmp(&real_energies[right])
            .then_with(|| left.cmp(&right))
    });

    let mut sorted = Vec::with_capacity(real_energies.len());
    let first = real_energies[order[0]];
    sorted.push(if first.abs() < XSPH_PHASE_SORT_TOLERANCE {
        0.0
    } else {
        first
    });
    for &source_index in order.iter().skip(1) {
        let value = real_energies[source_index];
        let Some(&previous) = sorted.last() else {
            return Err(XsphError::EmptyPhaseMesh);
        };
        if (value - previous).abs() > XSPH_PHASE_SORT_TOLERANCE {
            sorted.push(value);
        }
    }

    let zero_index = sorted
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .ok_or(XsphError::EmptyPhaseMesh)?;
    sorted[zero_index] = 0.0;

    Ok(XsphSortedEnergyGrid {
        energies: Array1::from_iter(sorted.into_iter().map(|energy| Complex::new(energy, 0.0))),
        zero_index,
    })
}
