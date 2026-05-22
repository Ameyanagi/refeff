//! FEFF radial-grid resampling helpers.

use std::f64::consts::PI;

use crate::Real;
use crate::interpolation::terp;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};

use super::validation::{
    ensure_source_length, validate_component_values, validate_delta, validate_finite_scalar,
    validate_positive_grid_length, validate_positive_radii, validate_real_table,
    validate_source_len_at_least,
};
use super::{
    AtomicQuantitiesGrid, AtomicQuantitiesGridInput, DiracSpinorGrid, DiracSpinorGridInput,
    DiracSpinorOrbitalsGrid, DiracSpinorOrbitalsGridInput, GridError, PotentialGrid,
    PotentialGridInput, SPINOR_ZERO_THRESHOLD, loucks_x, radial_index_below, radial_radius,
    radial_x,
};

/// Interpolate one FEFF Dirac spinor pair from `dxorg` to `dxnew`.
///
/// This ports the deterministic numerical part of `COMMON/fixdsp.f90`. FEFF
/// finds the last nonzero source-grid spinor point, adds one source point as
/// the zero boundary, interpolates both components with cubic `terp` on the
/// logarithmic `x` grid, and zero-fills the target tail.
pub fn fix_dirac_spinor_grid(
    input: DiracSpinorGridInput<'_>,
) -> Result<DiracSpinorGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;

    let source_len = input.large_component.len();
    if source_len != input.small_component.len() {
        return Err(GridError::SpinorLengthMismatch {
            large_len: source_len,
            small_len: input.small_component.len(),
        });
    }
    validate_positive_grid_length("source", source_len)?;
    validate_component_values("large_component", input.large_component)?;
    validate_component_values("small_component", input.small_component)?;

    let mut large_component = Array1::<Real>::zeros(input.output_len);
    let mut small_component = Array1::<Real>::zeros(input.output_len);
    let Some(last_nonzero) =
        last_nonzero_spinor_index(input.large_component, input.small_component)
    else {
        return Ok(DiracSpinorGrid {
            large_component,
            small_component,
            active_len: 0,
        });
    };

    let source_window_len = (last_nonzero + 2).min(source_len);
    let source_x = (1..=source_window_len)
        .map(|index| radial_x(index, input.original_delta))
        .collect::<Vec<_>>();
    let source_large = input
        .large_component
        .iter()
        .take(source_window_len)
        .copied()
        .collect::<Vec<_>>();
    let source_small = input
        .small_component
        .iter()
        .take(source_window_len)
        .copied()
        .collect::<Vec<_>>();

    let rmax = radial_radius(source_window_len, input.original_delta);
    let active_len = radial_index_below(rmax, input.new_delta)?;
    if active_len > input.output_len {
        return Err(GridError::OutputGridTooShort {
            required: active_len,
            available: input.output_len,
        });
    }

    for target_index in 1..=active_len {
        let x = radial_x(target_index, input.new_delta);
        let index = target_index - 1;
        large_component[index] = terp(&source_x, &source_large, 3, x)?.value;
        small_component[index] = terp(&source_x, &source_small, 3, x)?.value;
    }

    Ok(DiracSpinorGrid {
        large_component,
        small_component,
        active_len,
    })
}

/// Interpolate FEFF Dirac spinor orbital columns from `dxorg` to `dxnew`.
///
/// This ports the deterministic resampling behavior of `COMMON/fixdsx.f90`.
/// Each orbital column is treated independently with the same zero-tail
/// detection and cubic interpolation used by [`fix_dirac_spinor_grid`].
pub fn fix_dirac_spinor_orbitals_grid(
    input: DiracSpinorOrbitalsGridInput<'_>,
) -> Result<DiracSpinorOrbitalsGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;

    let large_shape = input.large_components.shape();
    let small_shape = input.small_components.shape();
    if large_shape != small_shape {
        return Err(GridError::SpinorShapeMismatch {
            large_rows: large_shape[0],
            large_columns: large_shape[1],
            small_rows: small_shape[0],
            small_columns: small_shape[1],
        });
    }
    validate_positive_grid_length("source", large_shape[0])?;
    validate_positive_grid_length("orbital", large_shape[1])?;

    let orbital_count = large_shape[1];
    let mut large_components = Array2::<Real>::zeros((input.output_len, orbital_count).f());
    let mut small_components = Array2::<Real>::zeros((input.output_len, orbital_count).f());
    let mut active_lengths = Array1::<usize>::zeros(orbital_count);

    for orbital in 0..orbital_count {
        let spinor = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: input.original_delta,
            new_delta: input.new_delta,
            large_component: input.large_components.column(orbital),
            small_component: input.small_components.column(orbital),
            output_len: input.output_len,
        })?;
        large_components
            .column_mut(orbital)
            .assign(&spinor.large_component);
        small_components
            .column_mut(orbital)
            .assign(&spinor.small_component);
        active_lengths[orbital] = spinor.active_len;
    }

    Ok(DiracSpinorOrbitalsGrid {
        large_components,
        small_components,
        active_lengths,
    })
}

/// Interpolate FEFF potential, charge density, and magnetization onto a target grid.
///
/// This ports the deterministic numerical behavior of `COMMON/fixvar.f90`.
/// Values through the first target interstitial point are cubic-interpolated on
/// FEFF's logarithmic `x` grid, optional potential jumps are applied exactly as
/// `jumprm` specifies, charge density is divided by `4*pi`, and the remaining
/// tail is filled with interstitial values.
pub fn fix_potential_grid(input: PotentialGridInput<'_>) -> Result<PotentialGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;
    validate_finite_scalar("interstitial_potential", input.interstitial_potential)?;
    validate_finite_scalar("interstitial_density", input.interstitial_density)?;
    validate_finite_scalar("potential_jump", input.potential_jump)?;

    let density_len = input.electron_density.len();
    let potential_len = input.total_potential.len();
    let magnetization_len = input.magnetization.len();
    if density_len != potential_len || density_len != magnetization_len {
        return Err(GridError::PotentialLengthMismatch {
            density_len,
            potential_len,
            magnetization_len,
        });
    }
    validate_positive_grid_length("source", density_len)?;
    validate_component_values("electron_density", input.electron_density)?;
    validate_component_values("total_potential", input.total_potential)?;
    validate_component_values("magnetization", input.magnetization)?;

    let muffin_tin_index_source =
        radial_index_below(input.muffin_tin_radius, input.original_delta)?;
    let interstitial_index_source = muffin_tin_index_source + 1;
    let density_window_len = interstitial_index_source + 1;
    ensure_source_length("total_potential", interstitial_index_source, potential_len)?;
    ensure_source_length("electron_density", density_window_len, density_len)?;
    ensure_source_length("magnetization", density_window_len, magnetization_len)?;

    let muffin_tin_index = radial_index_below(input.muffin_tin_radius, input.new_delta)?;
    let interstitial_index = muffin_tin_index + 1;
    if interstitial_index > input.output_len {
        return Err(GridError::OutputGridTooShort {
            required: interstitial_index,
            available: input.output_len,
        });
    }

    let source_x = (1..=density_window_len)
        .map(|index| radial_x(index, input.original_delta))
        .collect::<Vec<_>>();
    let source_density = input
        .electron_density
        .iter()
        .take(density_window_len)
        .copied()
        .collect::<Vec<_>>();
    let source_potential = input
        .total_potential
        .iter()
        .take(interstitial_index_source)
        .copied()
        .collect::<Vec<_>>();
    let source_magnetization = input
        .magnetization
        .iter()
        .take(density_window_len)
        .copied()
        .collect::<Vec<_>>();

    let radii = (1..=input.output_len)
        .map(|index| radial_radius(index, input.new_delta))
        .collect::<Array1<_>>();
    let mut total_potential = Array1::<Real>::zeros(input.output_len);
    let mut charge_density = Array1::<Real>::zeros(input.output_len);
    let mut magnetization = Array1::<Real>::zeros(input.output_len);

    for target_index in 1..=interstitial_index {
        let x = radial_x(target_index, input.new_delta);
        let index = target_index - 1;
        total_potential[index] = terp(
            &source_x[..interstitial_index_source],
            &source_potential,
            3,
            x,
        )?
        .value;
        charge_density[index] = terp(&source_x, &source_density, 3, x)?.value;
        magnetization[index] = terp(&source_x, &source_magnetization, 3, x)?.value;
    }

    let mut potential_jump = input.potential_jump;
    if input.jump_mode == 1 {
        let muffin_tin_potential = terp(
            &source_x[..interstitial_index_source],
            &source_potential,
            3,
            input.muffin_tin_radius.ln(),
        )?
        .value;
        potential_jump = input.interstitial_potential - muffin_tin_potential;
    }
    if input.jump_mode > 0 {
        total_potential
            .iter_mut()
            .take(interstitial_index)
            .for_each(|value| *value += potential_jump);
    }

    charge_density
        .iter_mut()
        .take(interstitial_index)
        .for_each(|value| *value /= 4.0 * PI);

    total_potential
        .iter_mut()
        .zip(charge_density.iter_mut())
        .zip(magnetization.iter_mut())
        .skip(interstitial_index)
        .for_each(|((potential, density), moment)| {
            *potential = input.interstitial_potential;
            *density = input.interstitial_density / (4.0 * PI);
            *moment = 0.0;
        });

    Ok(PotentialGrid {
        radii,
        total_potential,
        charge_density,
        magnetization,
        muffin_tin_index,
        interstitial_index,
        potential_jump,
    })
}

/// Resample ATOM potentials, densities, and spinors onto FEFF's regular grid.
///
/// This ports `ATOM/scfdat.f90` `FixAtomicQuantities`. FEFF builds
/// `xorg = log(dr)`, evaluates the regular `xx(i)` grid, and applies cubic
/// `terp` independently to `vcoul`, `srho`, `dmag`, `srhovl`, `dgc0`, `dpc0`,
/// and every orbital column of `dgc` and `dpc`.
pub fn fix_atomic_quantities_grid(
    input: AtomicQuantitiesGridInput<'_>,
) -> Result<AtomicQuantitiesGrid, GridError> {
    let source_len = input.source_radii.len();
    validate_positive_grid_length("source", source_len)?;
    validate_source_len_at_least("source", source_len, 4)?;
    validate_positive_grid_length("output", input.output_len)?;
    validate_positive_radii(input.source_radii, source_len)?;

    validate_atomic_quantities_lengths(input, source_len)?;
    validate_component_values("vcoul", input.coulomb_potential)?;
    validate_component_values("srho", input.charge_density)?;
    validate_component_values("dmag", input.magnetization)?;
    validate_component_values("srhovl", input.valence_density)?;
    validate_component_values("dgc0", input.initial_large_component)?;
    validate_component_values("dpc0", input.initial_small_component)?;
    validate_atomic_spinor_shapes(input.large_components, input.small_components, source_len)?;
    validate_real_table("dgc", input.large_components)?;
    validate_real_table("dpc", input.small_components)?;

    let source_x = input
        .source_radii
        .iter()
        .map(|radius| radius.ln())
        .collect::<Vec<_>>();
    let target_x = (1..=input.output_len).map(loucks_x).collect::<Vec<_>>();
    let radii = target_x.iter().map(|&x| x.exp()).collect::<Array1<_>>();

    let coulomb_potential =
        interpolate_atomic_quantity_table(&source_x, input.coulomb_potential, &target_x)?;
    let charge_density =
        interpolate_atomic_quantity_table(&source_x, input.charge_density, &target_x)?;
    let magnetization =
        interpolate_atomic_quantity_table(&source_x, input.magnetization, &target_x)?;
    let valence_density =
        interpolate_atomic_quantity_table(&source_x, input.valence_density, &target_x)?;
    let initial_large_component =
        interpolate_atomic_quantity_table(&source_x, input.initial_large_component, &target_x)?;
    let initial_small_component =
        interpolate_atomic_quantity_table(&source_x, input.initial_small_component, &target_x)?;
    let large_components =
        interpolate_atomic_quantity_matrix(&source_x, input.large_components, &target_x)?;
    let small_components =
        interpolate_atomic_quantity_matrix(&source_x, input.small_components, &target_x)?;

    Ok(AtomicQuantitiesGrid {
        radii,
        coulomb_potential,
        charge_density,
        magnetization,
        valence_density,
        initial_large_component,
        initial_small_component,
        large_components,
        small_components,
    })
}

fn validate_atomic_quantities_lengths(
    input: AtomicQuantitiesGridInput<'_>,
    source_len: usize,
) -> Result<(), GridError> {
    let coulomb_len = input.coulomb_potential.len();
    let density_len = input.charge_density.len();
    let magnetization_len = input.magnetization.len();
    let valence_len = input.valence_density.len();
    let large_len = input.initial_large_component.len();
    let small_len = input.initial_small_component.len();
    if coulomb_len == source_len
        && density_len == source_len
        && magnetization_len == source_len
        && valence_len == source_len
        && large_len == source_len
        && small_len == source_len
    {
        Ok(())
    } else {
        Err(GridError::AtomicQuantitiesLengthMismatch {
            radii_len: source_len,
            coulomb_len,
            density_len,
            magnetization_len,
            valence_len,
            large_len,
            small_len,
        })
    }
}

fn validate_atomic_spinor_shapes(
    large_components: ArrayView2<'_, Real>,
    small_components: ArrayView2<'_, Real>,
    source_len: usize,
) -> Result<(), GridError> {
    let large_shape = large_components.shape();
    let small_shape = small_components.shape();
    if large_shape != small_shape {
        return Err(GridError::SpinorShapeMismatch {
            large_rows: large_shape[0],
            large_columns: large_shape[1],
            small_rows: small_shape[0],
            small_columns: small_shape[1],
        });
    }
    if large_components.nrows() == source_len {
        Ok(())
    } else {
        Err(GridError::AtomicQuantitiesSpinorRowMismatch {
            radial_len: source_len,
            rows: large_components.nrows(),
            columns: large_components.ncols(),
        })
    }
}

fn interpolate_atomic_quantity_table(
    source_x: &[Real],
    values: ArrayView1<'_, Real>,
    target_x: &[Real],
) -> Result<Array1<Real>, GridError> {
    let source = values.iter().copied().collect::<Vec<_>>();
    target_x
        .iter()
        .map(|&x| Ok(terp(source_x, &source, 3, x)?.value))
        .collect::<Result<Array1<_>, GridError>>()
}

fn interpolate_atomic_quantity_matrix(
    source_x: &[Real],
    values: ArrayView2<'_, Real>,
    target_x: &[Real],
) -> Result<Array2<Real>, GridError> {
    let mut output = Array2::<Real>::zeros((target_x.len(), values.ncols()).f());
    for column in 0..values.ncols() {
        let source = values.column(column).to_vec();
        for (row, &x) in target_x.iter().enumerate() {
            output[(row, column)] = terp(source_x, &source, 3, x)?.value;
        }
    }
    Ok(output)
}

fn last_nonzero_spinor_index(
    large_component: ArrayView1<'_, Real>,
    small_component: ArrayView1<'_, Real>,
) -> Option<usize> {
    large_component
        .iter()
        .zip(small_component.iter())
        .enumerate()
        .rev()
        .find_map(|(index, (&large, &small))| {
            (large.abs() >= SPINOR_ZERO_THRESHOLD || small.abs() >= SPINOR_ZERO_THRESHOLD)
                .then_some(index)
        })
}
