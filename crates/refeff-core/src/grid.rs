//! FEFF common energy and radial-grid helpers.
//!
//! These functions port the small common routines `getxk.f90`, `xx.f90`,
//! `m_ifuns.f90`, and radial resampling helpers from `COMMON/`. FEFF uses a
//! 1-based logarithmic radial grid with `x = -8.8 + (j - 1) * delta` and
//! `r = exp(x)`.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use thiserror::Error;

use crate::interpolation::{InterpolationError, terp};
use crate::{Complex, Real};

/// Default FEFF logarithmic radial-grid spacing.
pub const LOUCKS_DELTA: Real = 0.05;

/// Offset used by FEFF's Loucks radial grid.
pub const LOUCKS_X_OFFSET: Real = 8.8;

/// FEFF Hartree constant in eV, from `COMMON/m_constants.f90`.
pub const FEFF_HARTREE_EV: Real = 27.211_396;

const SPINOR_ZERO_THRESHOLD: Real = 1.0e-11;

/// Inputs for FEFF `COMMON/fixdsp.f90` spinor grid interpolation.
#[derive(Debug, Clone, Copy)]
pub struct DiracSpinorGridInput<'a> {
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// Original large Dirac component on the source grid.
    pub large_component: ArrayView1<'a, Real>,
    /// Original small Dirac component on the source grid.
    pub small_component: ArrayView1<'a, Real>,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixdsp` spinor components on a target logarithmic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DiracSpinorGrid {
    /// Interpolated large Dirac component.
    pub large_component: Array1<Real>,
    /// Interpolated small Dirac component.
    pub small_component: Array1<Real>,
    /// Number of target-grid points filled before the zero tail.
    pub active_len: usize,
}

/// Inputs for FEFF `COMMON/fixdsx.f90` multi-orbital spinor interpolation.
#[derive(Debug, Clone, Copy)]
pub struct DiracSpinorOrbitalsGridInput<'a> {
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// Original large Dirac components as `(source_radial, orbital)`.
    pub large_components: ArrayView2<'a, Real>,
    /// Original small Dirac components as `(source_radial, orbital)`.
    pub small_components: ArrayView2<'a, Real>,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixdsx` spinor components on a target logarithmic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DiracSpinorOrbitalsGrid {
    /// Interpolated large Dirac components as `(target_radial, orbital)`.
    pub large_components: Array2<Real>,
    /// Interpolated small Dirac components as `(target_radial, orbital)`.
    pub small_components: Array2<Real>,
    /// Per-orbital target-grid active lengths before zero tails.
    pub active_lengths: Array1<usize>,
}

/// Inputs for FEFF `COMMON/fixvar.f90` potential and density interpolation.
#[derive(Debug, Clone, Copy)]
pub struct PotentialGridInput<'a> {
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Overlapping charge density on the source grid, matching FEFF `edens`.
    ///
    /// FEFF callers pass density multiplied by `4*pi`; [`fix_potential_grid`]
    /// divides the interpolated output by `4*pi`.
    pub electron_density: ArrayView1<'a, Real>,
    /// Total potential on the source grid, matching FEFF `vtot`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Magnetization density on the source grid, matching FEFF `dmag`.
    pub magnetization: ArrayView1<'a, Real>,
    /// Interstitial potential `vint` used to fill the target-grid tail.
    pub interstitial_potential: Real,
    /// Interstitial charge density `rhoint` used to fill the target-grid tail.
    pub interstitial_density: Real,
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// FEFF jump mode `jumprm`: `0` disables, `1` recomputes, `>0` applies.
    pub jump_mode: i32,
    /// Input potential jump `vjump`, or the initial value for `jump_mode == 1`.
    pub potential_jump: Real,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixvar` potential, charge-density, and magnetization target grid.
#[derive(Debug, Clone, PartialEq)]
pub struct PotentialGrid {
    /// Target radial coordinates `ri`.
    pub radii: Array1<Real>,
    /// Target total potential `vtotph`.
    pub total_potential: Array1<Real>,
    /// Target charge density `rhoph`, after FEFF's `4*pi` normalization.
    pub charge_density: Array1<Real>,
    /// Target magnetization density `dmagx`.
    pub magnetization: Array1<Real>,
    /// 1-based target muffin-tin index `jmtnew`.
    pub muffin_tin_index: usize,
    /// 1-based first target interstitial index `jrinew`.
    pub interstitial_index: usize,
    /// Final potential jump after optional `jumprm == 1` recomputation.
    pub potential_jump: Real,
}

/// Inputs for FEFF `POT/grids.f90` SCMT complex-energy mesh construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScmtEnergyGridInput {
    /// Core-valence separation energy `ecv`, in Hartrees.
    pub core_valence_energy: Real,
    /// Fermi energy `xmu`, in Hartrees.
    pub fermi_energy: Real,
    /// Length of the output complex-energy table, equivalent to `negx`.
    pub max_points: usize,
    /// FEFF step table length, equivalent to `nflrx`.
    pub step_count: usize,
}

/// FEFF SCMT complex-energy mesh and integration step table.
#[derive(Debug, Clone, PartialEq)]
pub struct ScmtEnergyGrid {
    /// Complex energies `emg`, zero-filled after [`ScmtEnergyGrid::active_len`].
    pub energies: Array1<Complex>,
    /// Integration step table `step`.
    pub steps: Array1<Real>,
    /// Number of active complex-energy points `neg`.
    pub active_len: usize,
    /// Number of initial off-axis points `neg1`.
    pub lower_imaginary_count: usize,
    /// Number of real-step bridge points `neg2`.
    pub real_axis_count: usize,
    /// Number of final off-axis points `neg3`.
    pub upper_imaginary_count: usize,
}

/// Error returned by radial-grid indexing helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GridError {
    /// The radius must be positive and finite before `ln(r)` is meaningful.
    #[error("radius must be positive and finite, got {radius}")]
    InvalidRadius { radius: Real },
    /// The logarithmic grid spacing must be positive and finite.
    #[error("grid delta must be positive and finite, got {delta}")]
    InvalidDelta { delta: Real },
    /// Source spinor component arrays must have matching lengths.
    #[error("spinor component length mismatch: large={large_len}, small={small_len}")]
    SpinorLengthMismatch { large_len: usize, small_len: usize },
    /// Source spinor component tables must have matching shapes.
    #[error(
        "spinor component shape mismatch: large=({large_rows},{large_columns}), small=({small_rows},{small_columns})"
    )]
    SpinorShapeMismatch {
        large_rows: usize,
        large_columns: usize,
        small_rows: usize,
        small_columns: usize,
    },
    /// Source potential, density, and magnetization arrays must have matching lengths.
    #[error(
        "potential-grid length mismatch: density={density_len}, potential={potential_len}, magnetization={magnetization_len}"
    )]
    PotentialLengthMismatch {
        density_len: usize,
        potential_len: usize,
        magnetization_len: usize,
    },
    /// A grid length must be positive.
    #[error("{name} length must be positive")]
    InvalidGridLength { name: &'static str },
    /// A grid length or derived table size overflowed Rust indexing.
    #[error("{name} grid length is too large")]
    GridLengthOverflow { name: &'static str },
    /// A scalar grid parameter must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// A source or interpolated grid value must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteGridValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// The caller's output grid is too short for FEFF's active interpolation range.
    #[error("output grid length {available} is shorter than required active length {required}")]
    OutputGridTooShort { required: usize, available: usize },
    /// The source grid is too short for FEFF's muffin-tin interpolation range.
    #[error("{name} source length {available} is shorter than required length {required}")]
    SourceGridTooShort {
        name: &'static str,
        required: usize,
        available: usize,
    },
    /// FEFF `terp` failed while resampling grid data.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
}

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

fn validate_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
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

fn square_index_as_real(name: &'static str, index: usize) -> Result<Real, GridError> {
    index
        .checked_mul(index)
        .map(|value| value as Real)
        .ok_or(GridError::GridLengthOverflow { name })
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ShapeBuilder};

    #[test]
    fn converts_energy_to_signed_wave_number() {
        assert_eq!(wave_number_from_hartree(2.0), 2.0);
        assert_eq!(wave_number_from_hartree(-2.0), -2.0);
        assert_eq!(wave_number_from_hartree(0.0), 0.0);
    }

    #[test]
    fn reproduces_loucks_log_grid_points() {
        assert!((loucks_x(1) + 8.8).abs() < 1.0e-12);
        assert!((loucks_x(2) + 8.75).abs() < 1.0e-12);
        assert!((loucks_radius(1) - (-8.8_f64).exp()).abs() < 1.0e-16);
    }

    #[test]
    fn maps_radius_to_index_below() -> Result<(), GridError> {
        let radius = loucks_radius(42);
        assert_eq!(loucks_index_below(radius)?, 42);

        let midpoint = (loucks_x(42) + 0.5 * LOUCKS_DELTA).exp();
        assert_eq!(loucks_index_below(midpoint)?, 42);
        Ok(())
    }

    #[test]
    fn rejects_invalid_radius_or_delta() {
        assert!(matches!(
            loucks_index_below(0.0),
            Err(GridError::InvalidRadius { .. })
        ));
        assert!(matches!(
            radial_index_below(1.0, 0.0),
            Err(GridError::InvalidDelta { .. })
        ));
    }

    #[test]
    fn fix_dirac_spinor_grid_matches_feff_fixdsp_reference() -> Result<(), GridError> {
        let mut large = vec![0.0; 251];
        let mut small = vec![0.0; 251];
        for i in 1..=80 {
            let i_real = i as Real;
            large[i - 1] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
            small[i - 1] = (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
        }
        let large = Array1::from_vec(large);
        let small = Array1::from_vec(small);

        let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 180,
        })?;

        assert_eq!(result.active_len, 161);
        assert_spinor_value(
            &result,
            1,
            0.098_856_582_548_901_49,
            0.981_461_262_295_415_9,
        );
        assert_spinor_value(&result, 2, 0.146_525_001_614_189, 0.969_970_868_040_543_4);
        assert_spinor_value(
            &result,
            3,
            0.192_879_394_911_354_22,
            0.957_050_307_749_104_5,
        );
        assert_spinor_value(&result, 10, 0.473_738_853_193_487_96, 0.830_355_320_320_026);
        assert_spinor_value(
            &result,
            80,
            -0.310_280_702_093_608_3,
            -0.562_325_207_440_241_6,
        );
        assert_spinor_value(
            &result,
            120,
            -0.008_407_166_503_866_128,
            0.021_105_137_955_943_806,
        );
        assert_spinor_value(
            &result,
            160,
            0.191_266_534_139_204_64,
            0.176_750_359_590_577_94,
        );
        assert_spinor_value(&result, 161, 0.0, 0.0);
        assert_spinor_value(&result, 180, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_grid_zero_fills_empty_spinor() -> Result<(), GridError> {
        let large = Array1::zeros(251);
        let small = Array1::zeros(251);

        let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 16,
        })?;

        assert_eq!(result.active_len, 0);
        assert!(result.large_component.iter().all(|&value| value == 0.0));
        assert!(result.small_component.iter().all(|&value| value == 0.0));
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_grid_rejects_invalid_inputs() {
        let large = Array1::zeros(4);
        let small = Array1::zeros(3);
        assert_eq!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: large.view(),
                small_component: small.view(),
                output_len: 16,
            }),
            Err(GridError::SpinorLengthMismatch {
                large_len: 4,
                small_len: 3,
            })
        );

        let nonfinite = Array1::from_vec(vec![0.0, f64::NAN, 0.0, 0.0]);
        let zeros = Array1::zeros(4);
        assert!(matches!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: nonfinite.view(),
                small_component: zeros.view(),
                output_len: 16,
            }),
            Err(GridError::NonFiniteGridValue {
                name: "large_component",
                index: 1,
                ..
            })
        ));

        assert_eq!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.0,
                new_delta: 0.025,
                large_component: zeros.view(),
                small_component: zeros.view(),
                output_len: 16,
            }),
            Err(GridError::InvalidDelta { delta: 0.0 })
        );
    }

    #[test]
    fn fix_dirac_spinor_orbitals_grid_matches_feff_fixdsx_reference() -> Result<(), GridError> {
        let mut large = Array2::<Real>::zeros((251, 4).f());
        let mut small = Array2::<Real>::zeros((251, 4).f());
        for i in 1..=40 {
            let i_real = i as Real;
            large[(i - 1, 0)] = (0.07 * i_real).sin() * (-0.01 * i_real).exp();
            small[(i - 1, 0)] = (0.05 * i_real).cos() * (-0.02 * i_real).exp();
        }
        for i in 1..=75 {
            let i_real = i as Real;
            large[(i - 1, 2)] = 0.2 * (0.11 * i_real).sin() + 0.002 * i_real;
            small[(i - 1, 2)] = 0.3 * (0.09 * i_real).cos() - 0.001 * i_real;
        }
        for i in 1..=5 {
            let i_real = i as Real;
            large[(i - 1, 3)] = 0.05 * i_real;
            small[(i - 1, 3)] = -0.04 * i_real;
        }

        let result = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_components: large.view(),
            small_components: small.view(),
            output_len: 260,
        })?;

        assert_eq!(result.large_components.shape(), &[260, 4]);
        assert_eq!(result.large_components.strides(), &[1, 260]);
        assert_eq!(result.active_lengths.to_vec(), vec![81, 0, 151, 11]);
        assert_orbital_value(
            &result,
            1,
            1,
            0.069_246_904_378_467_77,
            0.978_973_680_203_922_3,
        );
        assert_orbital_value(&result, 81, 1, 0.0, 0.0);
        assert_orbital_value(&result, 82, 1, 0.0, 0.0);
        assert_orbital_value(&result, 1, 2, 0.0, 0.0);
        assert_orbital_value(&result, 100, 2, 0.0, 0.0);
        assert_orbital_value(
            &result,
            1,
            3,
            0.023_955_660_167_434_965,
            0.297_785_819_903_598_26,
        );
        assert_orbital_value(
            &result,
            150,
            3,
            0.228_834_221_332_933_4,
            0.130_219_461_349_623_98,
        );
        assert_orbital_value(&result, 151, 3, 0.0, 0.0);
        assert_orbital_value(&result, 1, 4, 0.05, -0.04);
        assert_orbital_value(&result, 11, 4, 0.0, 0.0);
        assert_orbital_value(&result, 12, 4, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_orbitals_grid_rejects_shape_mismatch() {
        let large = Array2::<Real>::zeros((4, 2));
        let small = Array2::<Real>::zeros((4, 3));

        assert_eq!(
            fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_components: large.view(),
                small_components: small.view(),
                output_len: 16,
            }),
            Err(GridError::SpinorShapeMismatch {
                large_rows: 4,
                large_columns: 2,
                small_rows: 4,
                small_columns: 3,
            })
        );
    }

    #[test]
    fn fix_potential_grid_matches_feff_fixvar_nojump_reference() -> Result<(), GridError> {
        let result = run_sample_potential_grid(0, 0.125)?;

        assert_eq!(result.muffin_tin_index, 121);
        assert_eq!(result.interstitial_index, 122);
        assert_close(result.potential_jump, 0.125);
        assert_potential_value(
            &result,
            1,
            1.507_330_750_954_765e-4,
            -1.935_022_498_312_550_8,
            3.208_561_106_457_231e-2,
            6.991_469_396_917_269e-4,
        );
        assert_potential_value(
            &result,
            2,
            1.545_489_010_585_363e-4,
            -1.927_550_614_879_478_5,
            3.221_287_457_552_784e-2,
            1.047_124_998_462_492_8e-3,
        );
        assert_potential_value(
            &result,
            60,
            6.588_596_634_060_351e-4,
            -1.512_010_471_807_159,
            3.892_714_881_737_269e-2,
            3.404_343_790_471_498e-3,
        );
        assert_potential_value(
            &result,
            121,
            3.027_554_745_375_812_7e-3,
            -1.097_815_545_411_376,
            4.308_030_270_342_605e-2,
            -1.595_986_127_361_670_4e-2,
        );
        assert_potential_value(
            &result,
            122,
            3.104_197_658_649_308_7e-3,
            -1.091_039_022_689_942_5,
            4.312_310_486_424_463e-2,
            -1.593_525_191_056_929e-2,
        );
        assert_potential_value(
            &result,
            123,
            3.182_780_796_509_667e-3,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        assert_potential_value(
            &result,
            180,
            1.323_355_009_654_092_8e-2,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn fix_potential_grid_matches_feff_fixvar_auto_jump_reference() -> Result<(), GridError> {
        let result = run_sample_potential_grid(1, 0.125)?;

        assert_close(result.potential_jump, 3.423_945_657_555_365e-1);
        assert_potential_value(
            &result,
            1,
            1.507_330_750_954_765e-4,
            -1.592_627_932_557_014_3,
            3.208_561_106_457_231e-2,
            6.991_469_396_917_269e-4,
        );
        assert_potential_value(
            &result,
            2,
            1.545_489_010_585_363e-4,
            -1.585_156_049_123_942,
            3.221_287_457_552_784e-2,
            1.047_124_998_462_492_8e-3,
        );
        assert_potential_value(
            &result,
            60,
            6.588_596_634_060_351e-4,
            -1.169_615_906_051_622_5,
            3.892_714_881_737_269e-2,
            3.404_343_790_471_498e-3,
        );
        assert_potential_value(
            &result,
            121,
            3.027_554_745_375_812_7e-3,
            -7.554_209_796_558_395e-1,
            4.308_030_270_342_605e-2,
            -1.595_986_127_361_670_4e-2,
        );
        assert_potential_value(
            &result,
            122,
            3.104_197_658_649_308_7e-3,
            -7.486_444_569_344_06e-1,
            4.312_310_486_424_463e-2,
            -1.593_525_191_056_929e-2,
        );
        assert_potential_value(
            &result,
            123,
            3.182_780_796_509_667e-3,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        assert_potential_value(
            &result,
            180,
            1.323_355_009_654_092_8e-2,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn fix_potential_grid_rejects_invalid_inputs() {
        let density = Array1::<Real>::zeros(4);
        let potential = Array1::<Real>::zeros(5);
        let magnetization = Array1::<Real>::zeros(4);
        assert!(matches!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(2, 0.05),
                electron_density: density.view(),
                total_potential: potential.view(),
                magnetization: magnetization.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: 0.125,
                output_len: 8,
            }),
            Err(GridError::PotentialLengthMismatch {
                density_len: 4,
                potential_len: 5,
                magnetization_len: 4,
            })
        ));

        assert!(matches!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(2, 0.05),
                electron_density: density.view(),
                total_potential: density.view(),
                magnetization: density.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: f64::NAN,
                output_len: 8,
            }),
            Err(GridError::NonFiniteScalar {
                name: "potential_jump",
                ..
            })
        ));

        let nonfinite_density = Array1::from_vec(vec![0.0, f64::INFINITY, 0.0, 0.0]);
        assert!(matches!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(2, 0.05),
                electron_density: nonfinite_density.view(),
                total_potential: density.view(),
                magnetization: density.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: 0.125,
                output_len: 8,
            }),
            Err(GridError::NonFiniteGridValue {
                name: "electron_density",
                index: 1,
                ..
            })
        ));

        assert_eq!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(6, 0.05),
                electron_density: density.view(),
                total_potential: density.view(),
                magnetization: density.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: 0.125,
                output_len: 16,
            }),
            Err(GridError::SourceGridTooShort {
                name: "total_potential",
                required: 7,
                available: 4,
            })
        );
    }

    #[test]
    fn scmt_energy_grid_matches_feff_grids_reference() -> Result<(), GridError> {
        let result = scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: -0.50,
            fermi_energy: 0.20,
            max_points: 120,
            step_count: 9,
        })?;

        assert_eq!(result.active_len, 74);
        assert_eq!(result.lower_imaginary_count, 5);
        assert_eq!(result.real_axis_count, 61);
        assert_eq!(result.upper_imaginary_count, 8);
        assert_energy(&result, 1, -0.5, 1.837_465_450_137_141e-3);
        assert_energy(&result, 2, -0.5, 7.349_861_800_548_564e-3);
        assert_energy(&result, 3, -0.5, 1.653_718_905_123_426_8e-2);
        assert_energy(&result, 4, -0.5, 2.939_944_720_219_425_7e-2);
        assert_energy(&result, 5, -0.5, 4.593_663_625_342_852e-2);
        assert_energy(
            &result,
            37,
            -1.327_868_852_459_011_8e-1,
            4.593_663_625_342_852e-2,
        );
        assert_energy(&result, 72, 0.2, 7.349_861_800_548_564e-3);
        assert_energy(&result, 73, 0.2, 4.134_297_262_808_567e-3);
        assert_energy(&result, 74, 0.2, 1.837_465_450_137_141e-3);
        assert_eq!(result.energies[74], Complex::new(0.0, 0.0));
        assert_step(&result, 1, 4.593_663_625_342_853e-4);
        assert_step(&result, 5, 4.134_297_262_808_567e-3);
        assert_step(&result, 9, 1.148_415_906_335_713_1e-2);
        Ok(())
    }

    #[test]
    fn scmt_energy_grid_matches_feff_grids_clamped_reference() -> Result<(), GridError> {
        let result = scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: -0.20,
            fermi_energy: 20.00,
            max_points: 42,
            step_count: 8,
        })?;

        assert_eq!(result.active_len, 42);
        assert_eq!(result.lower_imaginary_count, 4);
        assert_eq!(result.real_axis_count, 31);
        assert_eq!(result.upper_imaginary_count, 7);
        assert_energy(&result, 1, -0.2, 1.837_465_450_137_141e-3);
        assert_energy(&result, 4, -0.2, 2.939_944_720_219_425_7e-2);
        assert_energy(
            &result,
            5,
            4.516_129_032_258_064_4e-1,
            2.939_944_720_219_425_7e-2,
        );
        assert_energy(
            &result,
            21,
            1.087_741_935_483_871_1e1,
            2.939_944_720_219_425_7e-2,
        );
        assert_energy(&result, 40, 20.0, 7.349_861_800_548_564e-3);
        assert_energy(&result, 41, 20.0, 4.134_297_262_808_567e-3);
        assert_energy(&result, 42, 20.0, 1.837_465_450_137_141e-3);
        assert_step(&result, 1, 4.593_663_625_342_853e-4);
        assert_step(&result, 7, 7.349_861_800_548_564e-3);
        assert_step(&result, 8, 7.349_861_800_548_564e-3);
        Ok(())
    }

    #[test]
    fn scmt_energy_grid_rejects_invalid_inputs() {
        assert!(matches!(
            scmt_energy_grid(ScmtEnergyGridInput {
                core_valence_energy: f64::NAN,
                fermi_energy: 0.2,
                max_points: 120,
                step_count: 9,
            }),
            Err(GridError::NonFiniteScalar {
                name: "core_valence_energy",
                ..
            })
        ));
        assert_eq!(
            scmt_energy_grid(ScmtEnergyGridInput {
                core_valence_energy: -0.5,
                fermi_energy: 0.2,
                max_points: 120,
                step_count: 0,
            }),
            Err(GridError::InvalidGridLength { name: "step" })
        );
        assert_eq!(
            scmt_energy_grid(ScmtEnergyGridInput {
                core_valence_energy: -0.5,
                fermi_energy: 0.2,
                max_points: 14,
                step_count: 8,
            }),
            Err(GridError::OutputGridTooShort {
                required: 15,
                available: 14,
            })
        );
    }

    fn assert_spinor_value(
        spinor: &DiracSpinorGrid,
        index_1based: usize,
        expected_large: Real,
        expected_small: Real,
    ) {
        let index = index_1based - 1;
        assert_close(spinor.large_component[index], expected_large);
        assert_close(spinor.small_component[index], expected_small);
    }

    fn assert_orbital_value(
        spinor: &DiracSpinorOrbitalsGrid,
        index_1based: usize,
        orbital_1based: usize,
        expected_large: Real,
        expected_small: Real,
    ) {
        let radial = index_1based - 1;
        let orbital = orbital_1based - 1;
        assert_close(spinor.large_components[(radial, orbital)], expected_large);
        assert_close(spinor.small_components[(radial, orbital)], expected_small);
    }

    fn assert_potential_value(
        grid: &PotentialGrid,
        index_1based: usize,
        expected_radius: Real,
        expected_potential: Real,
        expected_density: Real,
        expected_magnetization: Real,
    ) {
        let index = index_1based - 1;
        assert_close(grid.radii[index], expected_radius);
        assert_close(grid.total_potential[index], expected_potential);
        assert_close(grid.charge_density[index], expected_density);
        assert_close(grid.magnetization[index], expected_magnetization);
    }

    fn assert_energy(
        grid: &ScmtEnergyGrid,
        index_1based: usize,
        expected_real: Real,
        expected_imaginary: Real,
    ) {
        let value = grid.energies[index_1based - 1];
        assert_close(value.re, expected_real);
        assert_close(value.im, expected_imaginary);
    }

    fn assert_step(grid: &ScmtEnergyGrid, index_1based: usize, expected: Real) {
        assert_close(grid.steps[index_1based - 1], expected);
    }

    fn run_sample_potential_grid(
        jump_mode: i32,
        potential_jump: Real,
    ) -> Result<PotentialGrid, GridError> {
        let (density, potential, magnetization) = sample_potential_sources();
        fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: (-8.8 + 60.4 * 0.05_f64).exp(),
            electron_density: density.view(),
            total_potential: potential.view(),
            magnetization: magnetization.view(),
            interstitial_potential: -0.75,
            interstitial_density: 0.28,
            original_delta: 0.05,
            new_delta: 0.025,
            jump_mode,
            potential_jump,
            output_len: 180,
        })
    }

    fn sample_potential_sources() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let source_len = 251;
        let density = (1..=source_len)
            .map(|index| {
                let i = index as Real;
                0.4 + 0.002 * i + 0.03 * (0.04 * i).sin()
            })
            .collect::<Array1<_>>();
        let potential = (1..=source_len)
            .map(|index| {
                let i = index as Real;
                -2.0 + 0.015 * i + 0.05 * (0.03 * i).cos()
            })
            .collect::<Array1<_>>();
        let magnetization = (1..=source_len)
            .map(|index| {
                let i = index as Real;
                0.01 * (0.08 * i).sin() - 0.0001 * i
            })
            .collect::<Array1<_>>();
        (density, potential, magnetization)
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}
