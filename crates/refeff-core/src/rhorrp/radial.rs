use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, Axis, ShapeBuilder, Slice};
use refeff_linalg::{complex_polyfit, complex_polyval};

use crate::fovrg::{FovrgDiracSolverInput, fovrg_dirac_solver};
use crate::grid::{
    DiracSpinorOrbitalsGridInput, PotentialGridInput, fix_dirac_spinor_orbitals_grid,
    fix_potential_grid,
};
use crate::{Complex, ComplexMat, ComplexVec, Real};
use crate::{bessel::exjlnl, phase::muffin_tin_phase_amplitude};

use super::constants::{
    FEFF_ALPHA_INVERSE, FEFF_FINE_STRUCTURE_ALPHA, IRREGULAR_FIX_POINT_COUNT, RHORRP_RADIAL_X0,
};
use super::types::{
    RhorrpError, RhorrpExactRadialContinuation, RhorrpExactRadialContinuationInput,
    RhorrpExactRadialTail, RhorrpExactRadialTailInput, RhorrpFermiDistributionInput,
    RhorrpIrregularFixInput, RhorrpIrregularInitialCondition, RhorrpIrregularInitialConditionInput,
    RhorrpIrregularSolutionTransform, RhorrpIrregularSolutionTransformInput,
    RhorrpIrregularWronskianScale, RhorrpIrregularWronskianScaleInput, RhorrpMuffinTinMatch,
    RhorrpMuffinTinMatchInput, RhorrpPotentialReferenceShift, RhorrpPotentialReferenceShiftInput,
    RhorrpPotentialReferenceShifts, RhorrpPotentialReferenceShiftsInput,
    RhorrpPotentialWavefunctions, RhorrpPotentialWavefunctionsInput,
    RhorrpPreparedPotentialWavefunctionsInput, RhorrpPreparedWavefunctionTablesInput,
    RhorrpRadialInterpolationInput, RhorrpRadialInterpolationLocation,
    RhorrpRadialSolutionAssembly, RhorrpRadialSolutionAssemblyInput, RhorrpRegularSolutionScale,
    RhorrpRegularSolutionScaleInput, RhorrpWavefunctionChannel, RhorrpWavefunctionChannelInput,
    RhorrpWavefunctionGridPreparation, RhorrpWavefunctionGridPreparationInput,
    RhorrpWavefunctionInterpolationInput, RhorrpWavefunctionSetup, RhorrpWavefunctionSetupInput,
    RhorrpWavefunctionTables, RhorrpWavefunctionTablesInput,
};
use super::validation::{
    validate_irregular_fix_input, validate_radial_interpolation_input, validate_scalar,
    validate_wavefunction_interpolation_input,
};

/// Port of FEFF `rhoerrp` radial interpolation index setup.
///
/// FEFF maps a radius to `f = (log(r) + x0) / dx + 1`, clamps `f` to
/// `1..=nr`, truncates it to the one-based lower index, then keeps the
/// fractional remainder for `interpwf`.
pub fn rhorrp_radial_interpolation_location(
    input: RhorrpRadialInterpolationInput,
) -> Result<RhorrpRadialInterpolationLocation, RhorrpError> {
    validate_radial_interpolation_input(input)?;

    let mut position = if input.radius == 0.0 {
        1.0
    } else {
        (input.radius.ln() + input.x0) / input.dx + 1.0
    };
    position = position.clamp(1.0, input.radial_count as Real);

    let index_below_1based = position.trunc() as isize;
    Ok(RhorrpRadialInterpolationLocation {
        index_below_1based,
        fraction: position - index_below_1based as Real,
    })
}

/// Port of FEFF `interpwf`.
///
/// The index is FEFF's one-based lower radial index. Negative indices return a
/// zero matrix, `0` returns `wf(:,:,1) * fraction`, and positive indices linearly
/// blend FEFF radial samples `i` and `i+1`.
pub fn rhorrp_interpolate_wavefunction(
    input: RhorrpWavefunctionInterpolationInput<'_>,
) -> Result<ComplexMat, RhorrpError> {
    validate_wavefunction_interpolation_input(input)?;

    let (energy_count, angular_count, radial_count) = input.wavefunctions.dim();
    let mut output = Array2::zeros((energy_count, angular_count).f());
    if input.index_below_1based < 0 {
        return Ok(output);
    }

    if input.index_below_1based == 0 {
        for energy in 0..energy_count {
            for angular in 0..angular_count {
                output[(energy, angular)] =
                    input.wavefunctions[(energy, angular, 0)] * input.fraction;
            }
        }
        return Ok(output);
    }

    let lower = usize::try_from(input.index_below_1based - 1).map_err(|_| {
        RhorrpError::InvalidWavefunctionIndex {
            index: input.index_below_1based,
            radial: radial_count,
        }
    })?;
    let upper = lower + 1;
    if upper >= radial_count {
        return Err(RhorrpError::InvalidWavefunctionIndex {
            index: input.index_below_1based,
            radial: radial_count,
        });
    }

    let lower_weight = 1.0 - input.fraction;
    for energy in 0..energy_count {
        for angular in 0..angular_count {
            output[(energy, angular)] = input.wavefunctions[(energy, angular, lower)]
                * lower_weight
                + input.wavefunctions[(energy, angular, upper)] * input.fraction;
        }
    }
    Ok(output)
}

/// Port of FEFF `fermi_dist`.
///
/// FEFF uses the override chemical potential when COMPTON asks for one, applies
/// a step function for temperatures below `1e-5` Hartree, and otherwise returns
/// `1 / (exp((E - mu) / T) + 1)` for complex contour energies.
pub fn rhorrp_fermi_distribution(
    input: RhorrpFermiDistributionInput,
) -> Result<Complex, RhorrpError> {
    validate_scalar("energy_hartree.real", 0, input.energy_hartree.re)?;
    validate_scalar("energy_hartree.imag", 0, input.energy_hartree.im)?;
    validate_scalar(
        "chemical_potential_hartree",
        0,
        input.chemical_potential_hartree,
    )?;
    validate_scalar("temperature_hartree", 0, input.temperature_hartree)?;

    let mu = if let Some(override_mu) = input.chemical_potential_override_hartree {
        validate_scalar("chemical_potential_override_hartree", 0, override_mu)?;
        override_mu
    } else {
        input.chemical_potential_hartree
    };

    let value = if input.temperature_hartree < 1.0e-5 {
        if input.energy_hartree.re < mu {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        }
    } else {
        let exponent = (input.energy_hartree - Complex::new(mu, 0.0)) / input.temperature_hartree;
        Complex::new(1.0, 0.0) / (exponent.exp() + Complex::new(1.0, 0.0))
    };

    validate_scalar("fermi_distribution.real", 0, value.re)?;
    validate_scalar("fermi_distribution.imag", 0, value.im)?;
    Ok(value)
}

/// Port of FEFF `fix_irreg`.
///
/// FEFF fits a cubic polynomial to radial samples `50:100` and replaces samples
/// `1:100` with the polynomial evaluation. The tail after sample 100 is left
/// unchanged. This function returns the updated vector instead of mutating the
/// caller's data.
pub fn rhorrp_fix_irregular_origin(
    input: RhorrpIrregularFixInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    validate_irregular_fix_input(input)?;

    let coefficients = complex_polyfit(
        &input.radii[49..100],
        input.values.slice_axis(Axis(0), Slice::from(49..100)),
        3,
    )?;
    let smoothed = complex_polyval(coefficients.view(), &input.radii[..100]);
    let mut output = input.values.to_owned();
    output
        .slice_axis_mut(Axis(0), Slice::from(..100))
        .assign(&smoothed);
    Ok(output)
}

/// Port of the FEFF `init_wavefunctions` `eref0` potential shift.
///
/// RHORRP derives a potential-local reference energy from
/// `jri = int((log(rmt) + x0) / dx + 2)` and `eref0 = vtotph(jri + 1)`, then
/// subtracts that value from the total-potential prefix through `jri + 1`.
/// For `ixc >= 5`, FEFF shifts the valence-potential prefix by the same
/// reference; otherwise it replaces that prefix with the shifted total
/// potential before solving the radial Dirac equation.
pub fn rhorrp_potential_reference_shift(
    input: RhorrpPotentialReferenceShiftInput<'_>,
) -> Result<RhorrpPotentialReferenceShift, RhorrpError> {
    validate_potential_reference_shift_input(input)?;

    let reference_index = potential_reference_index(input)?;
    let reference = input.total_potential[reference_index];
    let mut total_potential = input.total_potential.to_owned();
    let mut valence_potential = input.valence_potential.to_owned();

    for index in 0..=reference_index {
        total_potential[index] -= reference;
        if input.exchange_index >= 5 {
            valence_potential[index] -= reference;
        } else {
            valence_potential[index] = total_potential[index];
        }
    }

    Ok(RhorrpPotentialReferenceShift {
        reference_index_1based: reference_index + 1,
        reference_energy_hartree: Complex::new(reference, 0.0),
        total_potential,
        valence_potential,
    })
}

/// Port the all-potential FEFF `init_wavefunctions` `eref0` shift.
///
/// FEFF applies the same potential-local shift independently for each
/// potential type before building the photoelectron wavefunction tables. This
/// helper preserves the `(radial, potential)` table shape used by the handoff
/// data while exposing the sampled `eref0(iph)` values.
pub fn rhorrp_potential_reference_shifts(
    input: RhorrpPotentialReferenceShiftsInput<'_>,
) -> Result<RhorrpPotentialReferenceShifts, RhorrpError> {
    validate_potential_reference_shifts_input(input)?;

    let (radial_count, potential_count) = input.total_potential.dim();
    let mut reference_indices_1based = Vec::with_capacity(potential_count);
    let mut reference_energies_hartree = Array1::<Complex>::zeros(potential_count);
    let mut total_potential = Array2::<Real>::zeros((radial_count, potential_count).f());
    let mut valence_potential = Array2::<Real>::zeros((radial_count, potential_count).f());

    for potential in 0..potential_count {
        let shifted = rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
            muffin_tin_radius: input.muffin_tin_radii[potential],
            radial_x0: input.radial_x0,
            radial_dx: input.radial_dx,
            total_potential: input.total_potential.index_axis(Axis(1), potential),
            valence_potential: input.valence_potential.index_axis(Axis(1), potential),
            exchange_index: input.exchange_index,
        })?;
        reference_indices_1based.push(shifted.reference_index_1based);
        reference_energies_hartree[potential] = shifted.reference_energy_hartree;
        total_potential
            .index_axis_mut(Axis(1), potential)
            .assign(&shifted.total_potential);
        valence_potential
            .index_axis_mut(Axis(1), potential)
            .assign(&shifted.valence_potential);
    }

    Ok(RhorrpPotentialReferenceShifts {
        reference_indices_1based,
        reference_energies_hartree,
        total_potential,
        valence_potential,
    })
}

/// Port the FEFF `init_wavefunctions` grid-preparation sequence.
///
/// This composes the RHORRP calls to `fixvar`, the energy-dependent valence
/// `fixvar` branch with FEFF's `jumprm -> 2` adjustment, `fixdsx`, and the
/// potential-local `eref0` shift. The returned arrays own their storage so
/// callers can safely borrow per-potential columns when constructing FOVRG
/// solver inputs.
pub fn rhorrp_prepare_wavefunction_grids(
    input: RhorrpWavefunctionGridPreparationInput<'_>,
) -> Result<RhorrpWavefunctionGridPreparation, RhorrpError> {
    validate_wavefunction_grid_preparation_input(input)?;

    let (_, potential_count) = input.total_potential.dim();
    let (_, orbital_count, _) = input.bound_large_components.dim();
    let mut radii = Array1::<Real>::zeros(input.radial_count);
    let mut potential_jumps = Array1::<Real>::zeros(potential_count);
    let mut total_potential = Array2::<Real>::zeros((input.radial_count, potential_count).f());
    let mut valence_potential = Array2::<Real>::zeros((input.radial_count, potential_count).f());
    let mut bound_large_components =
        Array3::<Real>::zeros((input.radial_count, orbital_count, potential_count).f());
    let mut bound_small_components =
        Array3::<Real>::zeros((input.radial_count, orbital_count, potential_count).f());
    let mut bound_active_lengths = Array2::<usize>::zeros((orbital_count, potential_count).f());
    let fix_valence_potential = input.exchange_index.rem_euclid(10) >= 5;

    for potential in 0..potential_count {
        let total_grid = fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: input.muffin_tin_radii[potential],
            electron_density: input.electron_density.index_axis(Axis(1), potential),
            total_potential: input.total_potential.index_axis(Axis(1), potential),
            magnetization: input.magnetization.index_axis(Axis(1), potential),
            interstitial_potential: input.interstitial_potential,
            interstitial_density: input.interstitial_density,
            original_delta: input.original_radial_dx,
            new_delta: input.target_radial_dx,
            jump_mode: input.jump_mode,
            potential_jump: input.potential_jump,
            output_len: input.radial_count,
        })?;
        if potential == 0 {
            radii.assign(&total_grid.radii);
        }
        potential_jumps[potential] = total_grid.potential_jump;
        total_potential
            .index_axis_mut(Axis(1), potential)
            .assign(&total_grid.total_potential);

        let prepared_valence_potential = if fix_valence_potential {
            let valence_jump_mode = if input.jump_mode > 0 {
                2
            } else {
                input.jump_mode
            };
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: input.muffin_tin_radii[potential],
                electron_density: input.valence_density.index_axis(Axis(1), potential),
                total_potential: input.valence_potential.index_axis(Axis(1), potential),
                magnetization: input.magnetization.index_axis(Axis(1), potential),
                interstitial_potential: input.interstitial_potential,
                interstitial_density: input.interstitial_density,
                original_delta: input.original_radial_dx,
                new_delta: input.target_radial_dx,
                jump_mode: valence_jump_mode,
                potential_jump: total_grid.potential_jump,
                output_len: input.radial_count,
            })?
            .total_potential
        } else {
            total_grid.total_potential.clone()
        };
        valence_potential
            .index_axis_mut(Axis(1), potential)
            .assign(&prepared_valence_potential);

        let spinors = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: input.original_radial_dx,
            new_delta: input.target_radial_dx,
            large_components: input.bound_large_components.index_axis(Axis(2), potential),
            small_components: input.bound_small_components.index_axis(Axis(2), potential),
            output_len: input.radial_count,
        })?;
        bound_large_components
            .index_axis_mut(Axis(2), potential)
            .assign(&spinors.large_components);
        bound_small_components
            .index_axis_mut(Axis(2), potential)
            .assign(&spinors.small_components);
        bound_active_lengths
            .index_axis_mut(Axis(1), potential)
            .assign(&spinors.active_lengths);
    }

    let shifted = rhorrp_potential_reference_shifts(RhorrpPotentialReferenceShiftsInput {
        muffin_tin_radii: input.muffin_tin_radii,
        radial_x0: RHORRP_RADIAL_X0,
        radial_dx: input.target_radial_dx,
        total_potential: total_potential.view(),
        valence_potential: valence_potential.view(),
        exchange_index: input.exchange_index,
    })?;

    Ok(RhorrpWavefunctionGridPreparation {
        radii,
        radial_dx: input.target_radial_dx,
        potential_jumps,
        reference_indices_1based: shifted.reference_indices_1based,
        reference_energies_hartree: shifted.reference_energies_hartree,
        total_potential: shifted
            .total_potential
            .mapv(|value| Complex::new(value, 0.0)),
        valence_potential: shifted
            .valence_potential
            .mapv(|value| Complex::new(value, 0.0)),
        bound_large_components,
        bound_small_components,
        bound_active_lengths,
    })
}

fn validate_wavefunction_grid_preparation_input(
    input: RhorrpWavefunctionGridPreparationInput<'_>,
) -> Result<(), RhorrpError> {
    if input.radial_count == 0 {
        return Err(RhorrpError::InvalidRadialCount { radial_count: 0 });
    }
    validate_scalar(
        "wavefunction_grid_interstitial_potential",
        0,
        input.interstitial_potential,
    )?;
    validate_scalar(
        "wavefunction_grid_interstitial_density",
        0,
        input.interstitial_density,
    )?;
    validate_scalar(
        "wavefunction_grid_original_radial_dx",
        0,
        input.original_radial_dx,
    )?;
    validate_scalar(
        "wavefunction_grid_target_radial_dx",
        0,
        input.target_radial_dx,
    )?;
    validate_scalar("wavefunction_grid_potential_jump", 0, input.potential_jump)?;

    let (source_radial, potential_count) = input.total_potential.dim();
    if potential_count == 0 {
        return Err(RhorrpError::InvalidWavefunctionGridPotentialCount { potential_count });
    }
    if input.muffin_tin_radii.len() != potential_count {
        return Err(RhorrpError::PotentialReferenceShiftShapeMismatch {
            total_radial: source_radial,
            total_potentials: potential_count,
            valence_radial: input.valence_potential.dim().0,
            valence_potentials: input.valence_potential.dim().1,
            muffin_tin_radii: input.muffin_tin_radii.len(),
        });
    }
    validate_wavefunction_grid_matrix_shape(
        "electron_density",
        source_radial,
        potential_count,
        input.electron_density.dim(),
    )?;
    validate_wavefunction_grid_matrix_shape(
        "valence_density",
        source_radial,
        potential_count,
        input.valence_density.dim(),
    )?;
    validate_wavefunction_grid_matrix_shape(
        "valence_potential",
        source_radial,
        potential_count,
        input.valence_potential.dim(),
    )?;
    validate_wavefunction_grid_matrix_shape(
        "magnetization",
        source_radial,
        potential_count,
        input.magnetization.dim(),
    )?;

    let (large_radial, large_orbital, large_potential) = input.bound_large_components.dim();
    let (small_radial, small_orbital, small_potential) = input.bound_small_components.dim();
    if input.bound_large_components.dim() != input.bound_small_components.dim() {
        return Err(RhorrpError::WavefunctionGridSpinorShapeMismatch {
            large_radial,
            large_orbital,
            large_potential,
            small_radial,
            small_orbital,
            small_potential,
        });
    }
    if large_radial == 0 || large_orbital == 0 || large_potential == 0 {
        return Err(RhorrpError::InvalidWavefunctionGridSpinorShape {
            radial: large_radial,
            orbital: large_orbital,
            potential: large_potential,
        });
    }
    if large_potential != potential_count {
        return Err(RhorrpError::WavefunctionGridMatrixShapeMismatch {
            component: "bound_spinors",
            expected_radial: large_radial,
            expected_potentials: potential_count,
            actual_radial: large_radial,
            actual_potentials: large_potential,
        });
    }
    for (potential, &radius) in input.muffin_tin_radii.iter().enumerate() {
        validate_positive_radius("wavefunction_grid_muffin_tin_radius", radius)?;
        validate_scalar(
            "wavefunction_grid_muffin_tin_radius_index",
            potential,
            radius,
        )?;
    }
    Ok(())
}

fn validate_wavefunction_grid_matrix_shape(
    component: &'static str,
    expected_radial: usize,
    expected_potentials: usize,
    actual: (usize, usize),
) -> Result<(), RhorrpError> {
    let (actual_radial, actual_potentials) = actual;
    if actual_radial != expected_radial || actual_potentials != expected_potentials {
        return Err(RhorrpError::WavefunctionGridMatrixShapeMismatch {
            component,
            expected_radial,
            expected_potentials,
            actual_radial,
            actual_potentials,
        });
    }
    Ok(())
}

/// Build one potential's RHORRP wavefunction table from prepared FEFF grids.
///
/// FEFF stores `eref0` at `jri1 = jri + 1` before calling `dfovrg` with
/// `jri`. The prepared-grid metadata stores that one-based `jri1`, so this
/// bridge converts it to the zero-based FOVRG match row and then delegates to
/// [`rhorrp_potential_wavefunctions`].
pub fn rhorrp_prepared_potential_wavefunctions(
    input: RhorrpPreparedPotentialWavefunctionsInput<'_>,
) -> Result<RhorrpPotentialWavefunctions, RhorrpError> {
    let radial_match_index = validate_prepared_potential_wavefunctions_input(input)?;
    let prepared = input.prepared;
    let potential = input.potential_index;
    let zero = Complex::new(0.0, 0.0);

    let solver = FovrgDiracSolverInput {
        exchange_cycle_count: 0,
        target_kappa: -1,
        muffin_tin_radius: input.muffin_tin_radius,
        target_last_index: 0,
        energy: zero,
        step: prepared.radial_dx,
        radii: prepared.radii.view(),
        exchange_correlation_potential: prepared.total_potential.index_axis(Axis(1), potential),
        valence_exchange_correlation_potential: prepared
            .valence_potential
            .index_axis(Axis(1), potential),
        bound_large_components: prepared
            .bound_large_components
            .index_axis(Axis(2), potential),
        bound_small_components: prepared
            .bound_small_components
            .index_axis(Axis(2), potential),
        bound_large_coefficients: input.bound_large_coefficients,
        bound_small_coefficients: input.bound_small_coefficients,
        electron_counts: input.electron_counts,
        valence_counts: input.valence_counts,
        kappa: input.kappa,
        muffin_tin_large_component: zero,
        muffin_tin_small_component: zero,
        atomic_number: input.atomic_number,
        irregular: false,
        c3_scale: 0,
        radial_match_index,
        bound_orbital_count: input.bound_orbital_count,
    };

    rhorrp_potential_wavefunctions(RhorrpPotentialWavefunctionsInput {
        solver,
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: prepared.reference_energies_hartree[potential],
        norman_radius: input.norman_radius,
        radial_x0: RHORRP_RADIAL_X0,
        radial_dx: prepared.radial_dx,
        exchange_index: input.exchange_index,
        angular_momentum_count: input.angular_momentum_count,
    })
}

/// Recover the FEFF `eref0` scalar used by `rhoerrp` density prefactors.
///
/// `init_wavefunctions` updates the module-level `eref0` while iterating over
/// potentials. When `rhoerrp` later computes `p2 = em(ie) - eref0`, that scalar
/// still contains the final potential's reference energy, not a per-atom value.
pub fn rhorrp_density_reference_energy_hartree(
    prepared: &RhorrpWavefunctionGridPreparation,
) -> Result<Complex, RhorrpError> {
    let potential_count = prepared.potential_count();
    if potential_count == 0 {
        return Err(RhorrpError::InvalidWavefunctionGridPotentialCount { potential_count });
    }
    if prepared.reference_energies_hartree.len() != potential_count {
        return Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "reference_energies_hartree",
            expected_potentials: potential_count,
            actual_potentials: prepared.reference_energies_hartree.len(),
        });
    }

    let reference_energy_hartree = prepared.reference_energies_hartree[potential_count - 1];
    validate_complex_result("density_reference_energy_hartree", reference_energy_hartree)?;
    Ok(reference_energy_hartree)
}

fn validate_prepared_potential_wavefunctions_input(
    input: RhorrpPreparedPotentialWavefunctionsInput<'_>,
) -> Result<usize, RhorrpError> {
    let prepared = input.prepared;
    let radial_count = prepared.radial_count();
    if radial_count == 0 {
        return Err(RhorrpError::InvalidRadialCount { radial_count: 0 });
    }
    validate_scalar("prepared_wavefunctions_radial_dx", 0, prepared.radial_dx)?;
    if prepared.radial_dx <= 0.0 {
        return Err(RhorrpError::InvalidRadialStep {
            value: prepared.radial_dx,
        });
    }

    let potential_count = prepared.potential_count();
    if potential_count == 0 {
        return Err(RhorrpError::InvalidWavefunctionGridPotentialCount { potential_count });
    }
    if input.potential_index >= potential_count {
        return Err(RhorrpError::PreparedWavefunctionPotentialOutOfRange {
            potential: input.potential_index,
            potential_count,
        });
    }
    if prepared.reference_energies_hartree.len() != potential_count {
        return Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "reference_energies_hartree",
            expected_potentials: potential_count,
            actual_potentials: prepared.reference_energies_hartree.len(),
        });
    }
    validate_wavefunction_grid_matrix_shape(
        "prepared_total_potential",
        radial_count,
        potential_count,
        prepared.total_potential.dim(),
    )?;
    validate_wavefunction_grid_matrix_shape(
        "prepared_valence_potential",
        radial_count,
        potential_count,
        prepared.valence_potential.dim(),
    )?;

    let (large_radial, large_orbital, large_potential) = prepared.bound_large_components.dim();
    let (small_radial, small_orbital, small_potential) = prepared.bound_small_components.dim();
    if prepared.bound_large_components.dim() != prepared.bound_small_components.dim() {
        return Err(RhorrpError::WavefunctionGridSpinorShapeMismatch {
            large_radial,
            large_orbital,
            large_potential,
            small_radial,
            small_orbital,
            small_potential,
        });
    }
    if large_radial == 0 || large_orbital == 0 || large_potential == 0 {
        return Err(RhorrpError::InvalidWavefunctionGridSpinorShape {
            radial: large_radial,
            orbital: large_orbital,
            potential: large_potential,
        });
    }
    if large_radial != radial_count || large_potential != potential_count {
        return Err(RhorrpError::WavefunctionGridMatrixShapeMismatch {
            component: "prepared_bound_spinors",
            expected_radial: radial_count,
            expected_potentials: potential_count,
            actual_radial: large_radial,
            actual_potentials: large_potential,
        });
    }

    let reference_index_1based = prepared.reference_indices_1based[input.potential_index];
    if reference_index_1based < 2 || reference_index_1based > radial_count {
        return Err(RhorrpError::PreparedWavefunctionReferenceIndexOutOfRange {
            potential: input.potential_index,
            index_1based: reference_index_1based,
            radial_count,
        });
    }
    Ok(reference_index_1based - 2)
}

fn validate_potential_reference_shifts_input(
    input: RhorrpPotentialReferenceShiftsInput<'_>,
) -> Result<(), RhorrpError> {
    if input.muffin_tin_radii.is_empty() {
        return Err(RhorrpError::InvalidPotentialReferencePotentialCount { potential_count: 0 });
    }
    let (total_radial, total_potentials) = input.total_potential.dim();
    let (valence_radial, valence_potentials) = input.valence_potential.dim();
    if total_radial != valence_radial
        || total_potentials != valence_potentials
        || total_potentials != input.muffin_tin_radii.len()
    {
        return Err(RhorrpError::PotentialReferenceShiftShapeMismatch {
            total_radial,
            total_potentials,
            valence_radial,
            valence_potentials,
            muffin_tin_radii: input.muffin_tin_radii.len(),
        });
    }
    if total_radial == 0 {
        return Err(RhorrpError::InvalidRadialCount { radial_count: 0 });
    }
    Ok(())
}

fn validate_potential_reference_shift_input(
    input: RhorrpPotentialReferenceShiftInput<'_>,
) -> Result<(), RhorrpError> {
    validate_scalar("muffin_tin_radius", 0, input.muffin_tin_radius)?;
    validate_scalar("potential_reference_x0", 0, input.radial_x0)?;
    validate_scalar("potential_reference_dx", 0, input.radial_dx)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    if input.radial_dx <= 0.0 {
        return Err(RhorrpError::InvalidRadialStep {
            value: input.radial_dx,
        });
    }
    if input.total_potential.len() != input.valence_potential.len() {
        return Err(RhorrpError::PotentialReferenceShiftLengthMismatch {
            total: input.total_potential.len(),
            valence: input.valence_potential.len(),
        });
    }
    if input.total_potential.is_empty() {
        return Err(RhorrpError::PotentialReferenceIndexOutOfRange {
            index_1based: 1,
            radial_count: 0,
        });
    }
    for (index, &value) in input.total_potential.iter().enumerate() {
        validate_scalar("total_potential", index, value)?;
    }
    for (index, &value) in input.valence_potential.iter().enumerate() {
        validate_scalar("valence_potential", index, value)?;
    }
    potential_reference_index(input)?;
    Ok(())
}

fn potential_reference_index(
    input: RhorrpPotentialReferenceShiftInput<'_>,
) -> Result<usize, RhorrpError> {
    let index_1based_float =
        ((input.muffin_tin_radius.ln() + input.radial_x0) / input.radial_dx + 2.0).trunc() + 1.0;
    if index_1based_float < 1.0 || index_1based_float > usize::MAX as Real {
        return Err(RhorrpError::PotentialReferenceIndexOutOfRange {
            index_1based: 0,
            radial_count: input.total_potential.len(),
        });
    }
    let index_1based = index_1based_float as usize;
    if index_1based == 0 || index_1based > input.total_potential.len() {
        return Err(RhorrpError::PotentialReferenceIndexOutOfRange {
            index_1based,
            radial_count: input.total_potential.len(),
        });
    }
    Ok(index_1based - 1)
}

/// Port of FEFF `init_wavefunctions` setup immediately before `dfovrg`.
///
/// For each potential and contour energy, RHORRP computes the active radial
/// integration endpoint `ilast`, referenced kinetic energy `p2`, relativistic
/// wave number `ck`, muffin-tin argument `xkmt`, and the `ncycle` selector used
/// by the radial Dirac solver.
pub fn rhorrp_wavefunction_setup(
    input: RhorrpWavefunctionSetupInput,
) -> Result<RhorrpWavefunctionSetup, RhorrpError> {
    validate_wavefunction_setup_input(input)?;

    let last_integration_index_1based = wavefunction_last_integration_index(input)?;
    let kinetic_energy_hartree = input.energy_hartree - input.reference_energy_hartree;
    validate_complex_result(
        "wavefunction_kinetic_energy_hartree",
        kinetic_energy_hartree,
    )?;
    let alpha_kinetic = kinetic_energy_hartree * FEFF_FINE_STRUCTURE_ALPHA;
    let wave_number = (kinetic_energy_hartree * 2.0 + alpha_kinetic * alpha_kinetic).sqrt();
    validate_complex_result("wavefunction_wave_number", wave_number)?;
    let muffin_tin_wave_number = wave_number * input.muffin_tin_radius;
    validate_complex_result(
        "wavefunction_muffin_tin_wave_number",
        muffin_tin_wave_number,
    )?;
    let dirac_cycle_count = if input.exchange_index % 10 < 5 { 0 } else { 3 };

    Ok(RhorrpWavefunctionSetup {
        last_integration_index_1based,
        dirac_cycle_count,
        kinetic_energy_hartree,
        wave_number,
        muffin_tin_wave_number,
    })
}

fn validate_wavefunction_setup_input(
    input: RhorrpWavefunctionSetupInput,
) -> Result<(), RhorrpError> {
    validate_scalar(
        "wavefunction_energy_hartree.real",
        0,
        input.energy_hartree.re,
    )?;
    validate_scalar(
        "wavefunction_energy_hartree.imag",
        0,
        input.energy_hartree.im,
    )?;
    validate_scalar(
        "wavefunction_reference_energy_hartree.real",
        0,
        input.reference_energy_hartree.re,
    )?;
    validate_scalar(
        "wavefunction_reference_energy_hartree.imag",
        0,
        input.reference_energy_hartree.im,
    )?;
    validate_scalar("muffin_tin_radius", 0, input.muffin_tin_radius)?;
    validate_scalar("norman_radius", 0, input.norman_radius)?;
    validate_scalar("wavefunction_radial_x0", 0, input.radial_x0)?;
    validate_scalar("wavefunction_radial_dx", 0, input.radial_dx)?;
    if input.muffin_tin_radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: input.muffin_tin_radius,
        });
    }
    if input.norman_radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "norman_radius",
            value: input.norman_radius,
        });
    }
    if input.radial_dx <= 0.0 {
        return Err(RhorrpError::InvalidRadialStep {
            value: input.radial_dx,
        });
    }
    if input.radial_capacity == 0 {
        return Err(RhorrpError::InvalidRadialCount { radial_count: 0 });
    }
    wavefunction_last_integration_index(input)?;
    Ok(())
}

fn wavefunction_last_integration_index(
    input: RhorrpWavefunctionSetupInput,
) -> Result<usize, RhorrpError> {
    let index_1based_float =
        ((input.norman_radius.ln() + input.radial_x0) / input.radial_dx + 8.0).trunc();
    if !index_1based_float.is_finite()
        || index_1based_float < 1.0
        || index_1based_float > isize::MAX as Real
    {
        return Err(RhorrpError::WavefunctionSetupIndexOutOfRange {
            index_1based: 0,
            radial_capacity: input.radial_capacity,
        });
    }

    let index_1based = index_1based_float as usize;
    Ok(index_1based.min(input.radial_capacity))
}

fn validate_complex_result(name: &'static str, value: Complex) -> Result<(), RhorrpError> {
    validate_scalar(name, 0, value.re)?;
    validate_scalar(name, 1, value.im)?;
    Ok(())
}

/// Port of the RHORRP muffin-tin matching sequence in `init_wavefunctions`.
///
/// After the regular `dfovrg` call, FEFF evaluates exact spherical
/// Bessel/Neumann values at `xkmt`, calls `phamp` to recover the phase shift
/// and amplitude, then sets `xfnorm = 1 / temp`. The returned Bessel values are
/// the same values reused by the subsequent irregular-solution boundary setup.
pub fn rhorrp_muffin_tin_match(
    input: RhorrpMuffinTinMatchInput,
) -> Result<RhorrpMuffinTinMatch, RhorrpError> {
    validate_positive_radius("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_complex_result("muffin_tin_match_wave_number", input.wave_number)?;
    validate_complex_result(
        "muffin_tin_regular_large_at_muffin_tin",
        input.regular_large_at_muffin_tin,
    )?;
    validate_complex_result(
        "muffin_tin_regular_small_at_muffin_tin",
        input.regular_small_at_muffin_tin,
    )?;

    let muffin_tin_wave_number = input.wave_number * input.muffin_tin_radius;
    validate_complex_result("muffin_tin_wave_number", muffin_tin_wave_number)?;
    let current = exjlnl(muffin_tin_wave_number, input.angular_momentum)?;
    let next = exjlnl(
        muffin_tin_wave_number,
        input.angular_momentum.saturating_add(1),
    )?;
    let phase = muffin_tin_phase_amplitude(
        input.muffin_tin_radius,
        input.regular_large_at_muffin_tin,
        input.regular_small_at_muffin_tin,
        input.wave_number,
        current.j,
        current.y,
        next.j,
        next.y,
        input.kappa,
    )?;
    let regular_scale = rhorrp_regular_solution_scale(RhorrpRegularSolutionScaleInput {
        phase_amplitude: phase.amplitude,
    })?;

    Ok(RhorrpMuffinTinMatch {
        muffin_tin_wave_number,
        bessel_j_l: current.j,
        neumann_l: current.y,
        bessel_j_l_plus_1: next.j,
        neumann_l_plus_1: next.y,
        phase_shift: phase.phase,
        phase_amplitude: phase.amplitude,
        regular_solution_scale: regular_scale.scale,
    })
}

/// Port of FEFF regular-solution normalization in `init_wavefunctions`.
///
/// After `phamp` returns `temp`, RHORRP normalizes the regular radial solution
/// with `xfnorm = 1 / temp` before copying `pn`/`qn` into `pr`/`qr`.
pub fn rhorrp_regular_solution_scale(
    input: RhorrpRegularSolutionScaleInput,
) -> Result<RhorrpRegularSolutionScale, RhorrpError> {
    validate_complex_result("regular_solution_phase_amplitude", input.phase_amplitude)?;
    if input.phase_amplitude == Complex::new(0.0, 0.0) {
        return Err(RhorrpError::ZeroComplexResult {
            name: "regular_solution_phase_amplitude",
        });
    }

    let scale = Complex::new(1.0, 0.0) / input.phase_amplitude;
    validate_complex_result("regular_solution_scale", scale)?;
    Ok(RhorrpRegularSolutionScale { scale })
}

/// Port of RHORRP irregular-solution boundary values before `dfovrg`.
///
/// This is the standing-wave branch used by `m_rhorrp.f90`:
/// `pu = (nl*cos(phx)+jl*sin(phx))*rmt` and
/// `qu = (nlp1*cos(phx)+jlp1*sin(phx))*factor*rmt`, with
/// `factor = -ck*alphfs/(1+sqrt(1+(ck*alphfs)^2))`.
pub fn rhorrp_irregular_initial_condition(
    input: RhorrpIrregularInitialConditionInput,
) -> Result<RhorrpIrregularInitialCondition, RhorrpError> {
    validate_positive_radius("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_complex_result("irregular_phase_shift", input.phase_shift)?;
    validate_complex_result("irregular_wave_number", input.wave_number)?;
    validate_complex_result("irregular_bessel_j_l", input.bessel_j_l)?;
    validate_complex_result("irregular_neumann_l", input.neumann_l)?;
    validate_complex_result("irregular_bessel_j_l_plus_1", input.bessel_j_l_plus_1)?;
    validate_complex_result("irregular_neumann_l_plus_1", input.neumann_l_plus_1)?;

    let small_component_factor = rhorrp_small_component_factor(input.wave_number)?;
    let cos_phase = input.phase_shift.cos();
    let sin_phase = input.phase_shift.sin();
    validate_complex_result("irregular_cos_phase", cos_phase)?;
    validate_complex_result("irregular_sin_phase", sin_phase)?;

    let radius = input.muffin_tin_radius;
    let large_component = (input.neumann_l * cos_phase + input.bessel_j_l * sin_phase) * radius;
    let small_component = (input.neumann_l_plus_1 * cos_phase
        + input.bessel_j_l_plus_1 * sin_phase)
        * small_component_factor
        * radius;
    validate_complex_result("irregular_large_component", large_component)?;
    validate_complex_result("irregular_small_component", small_component)?;

    Ok(RhorrpIrregularInitialCondition {
        large_component,
        small_component,
    })
}

/// Port of RHORRP irregular-solution Wronskian scaling.
///
/// After the irregular `dfovrg` pass, FEFF computes
/// `qu = 1 / (2*alpinv*exp(i*phx)*(pn(jri)*qr(jri)-pr(jri)*qn(jri))) / ck`.
/// This helper returns the phase factor and reciprocal scale separately so the
/// caller can apply them across all rows.
pub fn rhorrp_irregular_wronskian_scale(
    input: RhorrpIrregularWronskianScaleInput,
) -> Result<RhorrpIrregularWronskianScale, RhorrpError> {
    validate_complex_result("wronskian_phase_shift", input.phase_shift)?;
    validate_complex_result("wronskian_wave_number", input.wave_number)?;
    validate_complex_result(
        "wronskian_regular_large_at_match",
        input.regular_large_at_match,
    )?;
    validate_complex_result(
        "wronskian_regular_small_at_match",
        input.regular_small_at_match,
    )?;
    validate_complex_result(
        "wronskian_irregular_large_at_match",
        input.irregular_large_at_match,
    )?;
    validate_complex_result(
        "wronskian_irregular_small_at_match",
        input.irregular_small_at_match,
    )?;

    let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
    validate_complex_result("wronskian_phase_factor", phase_factor)?;
    let denominator = 2.0
        * FEFF_ALPHA_INVERSE
        * phase_factor
        * (input.irregular_large_at_match * input.regular_small_at_match
            - input.regular_large_at_match * input.irregular_small_at_match);
    validate_complex_result("wronskian_denominator", denominator)?;
    if denominator == Complex::new(0.0, 0.0) {
        return Err(RhorrpError::ZeroComplexResult {
            name: "wronskian_denominator",
        });
    }
    if input.wave_number == Complex::new(0.0, 0.0) {
        return Err(RhorrpError::ZeroComplexResult {
            name: "wronskian_wave_number",
        });
    }

    let reciprocal_wave_scale = Complex::new(1.0, 0.0) / denominator / input.wave_number;
    validate_complex_result("wronskian_reciprocal_wave_scale", reciprocal_wave_scale)?;
    Ok(RhorrpIrregularWronskianScale {
        phase_factor,
        denominator,
        reciprocal_wave_scale,
    })
}

/// Port of the RHORRP `N = iR - H*exp(i*phx)` row transform.
///
/// FEFF applies this after computing the irregular Wronskian scale:
/// `pn(i) = i*pr(i) - temp*pn(i)*qu` and
/// `qn(i) = i*qr(i) - temp*qn(i)*qu`.
pub fn rhorrp_irregular_solution_transform(
    input: RhorrpIrregularSolutionTransformInput,
) -> Result<RhorrpIrregularSolutionTransform, RhorrpError> {
    validate_complex_result("irregular_transform_phase_factor", input.phase_factor)?;
    validate_complex_result(
        "irregular_transform_reciprocal_wave_scale",
        input.reciprocal_wave_scale,
    )?;
    validate_complex_result(
        "irregular_transform_regular_large_component",
        input.regular_large_component,
    )?;
    validate_complex_result(
        "irregular_transform_regular_small_component",
        input.regular_small_component,
    )?;
    validate_complex_result(
        "irregular_transform_irregular_large_component",
        input.irregular_large_component,
    )?;
    validate_complex_result(
        "irregular_transform_irregular_small_component",
        input.irregular_small_component,
    )?;

    let imaginary = Complex::new(0.0, 1.0);
    let large_component = imaginary * input.regular_large_component
        - input.phase_factor * input.irregular_large_component * input.reciprocal_wave_scale;
    let small_component = imaginary * input.regular_small_component
        - input.phase_factor * input.irregular_small_component * input.reciprocal_wave_scale;
    validate_complex_result("irregular_transform_large_component", large_component)?;
    validate_complex_result("irregular_transform_small_component", small_component)?;

    Ok(RhorrpIrregularSolutionTransform {
        large_component,
        small_component,
    })
}

/// Port of RHORRP exact free-particle continuation for samples `jri:nr`.
///
/// After both `dfovrg` passes, FEFF overwrites the tail with exact regular and
/// irregular combinations of spherical Bessel and Neumann functions at
/// `xck = ck * ri(i)`.
pub fn rhorrp_exact_radial_continuation(
    input: RhorrpExactRadialContinuationInput,
) -> Result<RhorrpExactRadialContinuation, RhorrpError> {
    validate_positive_radius("radius", input.radius)?;
    validate_complex_result("exact_phase_shift", input.phase_shift)?;
    validate_complex_result("exact_wave_number", input.wave_number)?;
    validate_complex_result("exact_bessel_j_l", input.bessel_j_l)?;
    validate_complex_result("exact_neumann_l", input.neumann_l)?;
    validate_complex_result("exact_bessel_j_l_plus_1", input.bessel_j_l_plus_1)?;
    validate_complex_result("exact_neumann_l_plus_1", input.neumann_l_plus_1)?;

    let small_component_factor = rhorrp_small_component_factor(input.wave_number)?;
    let cos_phase = input.phase_shift.cos();
    let sin_phase = input.phase_shift.sin();
    validate_complex_result("exact_cos_phase", cos_phase)?;
    validate_complex_result("exact_sin_phase", sin_phase)?;

    let radius = input.radius;
    let regular_large_component =
        (input.bessel_j_l * cos_phase - input.neumann_l * sin_phase) * radius;
    let regular_small_component = (input.bessel_j_l_plus_1 * cos_phase
        - input.neumann_l_plus_1 * sin_phase)
        * small_component_factor
        * radius;
    let irregular_large_component =
        (input.neumann_l * cos_phase + input.bessel_j_l * sin_phase) * radius;
    let irregular_small_component = (input.neumann_l_plus_1 * cos_phase
        + input.bessel_j_l_plus_1 * sin_phase)
        * small_component_factor
        * radius;
    validate_complex_result("exact_regular_large_component", regular_large_component)?;
    validate_complex_result("exact_regular_small_component", regular_small_component)?;
    validate_complex_result("exact_irregular_large_component", irregular_large_component)?;
    validate_complex_result("exact_irregular_small_component", irregular_small_component)?;

    Ok(RhorrpExactRadialContinuation {
        regular_large_component,
        regular_small_component,
        irregular_large_component,
        irregular_small_component,
    })
}

/// Port of the RHORRP exact radial tail loop after muffin-tin matching.
///
/// FEFF overwrites rows `jri:nr` with free-particle Bessel/Neumann
/// combinations, evaluating `exjlnl(ck*ri(i), l)` and `exjlnl(ck*ri(i), l+1)`
/// for each row. This helper returns only the overwritten tail so later callers
/// can assign it into the full `prel/qrel/pnel/qnel` tables without losing the
/// FEFF one-based starting row.
pub fn rhorrp_exact_radial_tail(
    input: RhorrpExactRadialTailInput<'_>,
) -> Result<RhorrpExactRadialTail, RhorrpError> {
    validate_complex_result("exact_tail_phase_shift", input.phase_shift)?;
    validate_complex_result("exact_tail_wave_number", input.wave_number)?;
    if input.radii.is_empty() {
        return Err(RhorrpError::InvalidRadialCount { radial_count: 0 });
    }
    if input.start_index_1based == 0 || input.start_index_1based > input.radii.len() {
        return Err(RhorrpError::ExactRadialTailStartOutOfRange {
            start_index_1based: input.start_index_1based,
            radial_count: input.radii.len(),
        });
    }

    let start = input.start_index_1based - 1;
    let tail_len = input.radii.len() - start;
    let mut regular_large_components = Vec::with_capacity(tail_len);
    let mut regular_small_components = Vec::with_capacity(tail_len);
    let mut irregular_large_components = Vec::with_capacity(tail_len);
    let mut irregular_small_components = Vec::with_capacity(tail_len);

    for &radius in &input.radii[start..] {
        validate_positive_radius("exact_tail_radius", radius)?;
        let argument = input.wave_number * radius;
        validate_complex_result("exact_tail_bessel_argument", argument)?;
        let current = exjlnl(argument, input.angular_momentum)?;
        let next = exjlnl(argument, input.angular_momentum.saturating_add(1))?;
        let continued = rhorrp_exact_radial_continuation(RhorrpExactRadialContinuationInput {
            radius,
            phase_shift: input.phase_shift,
            wave_number: input.wave_number,
            bessel_j_l: current.j,
            neumann_l: current.y,
            bessel_j_l_plus_1: next.j,
            neumann_l_plus_1: next.y,
        })?;
        regular_large_components.push(continued.regular_large_component);
        regular_small_components.push(continued.regular_small_component);
        irregular_large_components.push(continued.irregular_large_component);
        irregular_small_components.push(continued.irregular_small_component);
    }

    Ok(RhorrpExactRadialTail {
        start_index_1based: input.start_index_1based,
        regular_large_components: Array1::from_vec(regular_large_components),
        regular_small_components: Array1::from_vec(regular_small_components),
        irregular_large_components: Array1::from_vec(irregular_large_components),
        irregular_small_components: Array1::from_vec(irregular_small_components),
    })
}

/// Assemble RHORRP radial solution rows after regular and irregular `dfovrg`.
///
/// This ports the FEFF vector operations following the two radial solver passes:
/// scale the regular solution by `xfnorm`, compute the Wronskian scale at the
/// match row, transform the raw irregular solution into `N = iR - H*exp(i*phx)`,
/// overwrite the free-particle tail with exact Bessel/Neumann rows, and apply
/// FEFF `fix_irreg` smoothing to `l=0` irregular-origin rows when the table is
/// long enough for that 100-point fit.
pub fn rhorrp_assemble_radial_solutions(
    input: RhorrpRadialSolutionAssemblyInput<'_>,
) -> Result<RhorrpRadialSolutionAssembly, RhorrpError> {
    validate_radial_solution_assembly_input(input)?;

    let regular_solution_scale = rhorrp_regular_solution_scale(RhorrpRegularSolutionScaleInput {
        phase_amplitude: input.phase_amplitude,
    })?
    .scale;
    let mut regular_large_components = input
        .raw_regular_large
        .mapv(|value| value * regular_solution_scale);
    let mut regular_small_components = input
        .raw_regular_small
        .mapv(|value| value * regular_solution_scale);
    validate_complex_array(
        "radial_regular_large_component",
        regular_large_components.view(),
    )?;
    validate_complex_array(
        "radial_regular_small_component",
        regular_small_components.view(),
    )?;

    let match_index = input.match_index_1based - 1;
    let irregular_wronskian_scale =
        rhorrp_irregular_wronskian_scale(RhorrpIrregularWronskianScaleInput {
            phase_shift: input.phase_shift,
            wave_number: input.wave_number,
            regular_large_at_match: regular_large_components[match_index],
            regular_small_at_match: regular_small_components[match_index],
            irregular_large_at_match: input.raw_irregular_large[match_index],
            irregular_small_at_match: input.raw_irregular_small[match_index],
        })?;

    let mut irregular_large_components = Array1::<Complex>::zeros(input.radii.len());
    let mut irregular_small_components = Array1::<Complex>::zeros(input.radii.len());
    for row in 0..input.radii.len() {
        let transformed =
            rhorrp_irregular_solution_transform(RhorrpIrregularSolutionTransformInput {
                phase_factor: irregular_wronskian_scale.phase_factor,
                reciprocal_wave_scale: irregular_wronskian_scale.reciprocal_wave_scale,
                regular_large_component: regular_large_components[row],
                regular_small_component: regular_small_components[row],
                irregular_large_component: input.raw_irregular_large[row],
                irregular_small_component: input.raw_irregular_small[row],
            })?;
        irregular_large_components[row] = transformed.large_component;
        irregular_small_components[row] = transformed.small_component;
    }

    let exact_tail = rhorrp_exact_radial_tail(RhorrpExactRadialTailInput {
        radii: input.radii,
        start_index_1based: input.exact_tail_start_index_1based,
        angular_momentum: input.angular_momentum,
        phase_shift: input.phase_shift,
        wave_number: input.wave_number,
    })?;
    let tail_start = exact_tail.start_index_1based - 1;
    for offset in 0..exact_tail.row_count() {
        let row = tail_start + offset;
        regular_large_components[row] = exact_tail.regular_large_components[offset];
        regular_small_components[row] = exact_tail.regular_small_components[offset];
        irregular_large_components[row] = exact_tail.irregular_large_components[offset];
        irregular_small_components[row] = exact_tail.irregular_small_components[offset];
    }

    let irregular_origin_smoothed =
        input.angular_momentum == 0 && input.radii.len() >= IRREGULAR_FIX_POINT_COUNT;
    if irregular_origin_smoothed {
        irregular_large_components = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: input.radii,
            values: irregular_large_components.view(),
        })?;
        irregular_small_components = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: input.radii,
            values: irregular_small_components.view(),
        })?;
    }

    Ok(RhorrpRadialSolutionAssembly {
        regular_solution_scale,
        irregular_wronskian_scale,
        regular_large_components,
        regular_small_components,
        irregular_large_components,
        irregular_small_components,
        irregular_origin_smoothed,
    })
}

/// Port one RHORRP `init_wavefunctions` energy/angular radial channel.
///
/// This composes the regular `dfovrg` pass, muffin-tin `phamp` match,
/// irregular-boundary setup, irregular `dfovrg` pass, and final
/// `prel/qrel/pnel/qnel` assembly for one `(energy, l, potential)` channel.
pub fn rhorrp_wavefunction_channel(
    input: RhorrpWavefunctionChannelInput<'_>,
) -> Result<RhorrpWavefunctionChannel, RhorrpError> {
    validate_complex_result("wavefunction_channel_wave_number", input.wave_number)?;

    let zero = Complex::new(0.0, 0.0);
    let regular_input = FovrgDiracSolverInput {
        irregular: false,
        muffin_tin_large_component: zero,
        muffin_tin_small_component: zero,
        ..input.solver
    };
    let regular_solution = fovrg_dirac_solver(regular_input)?;
    let muffin_tin_match = rhorrp_muffin_tin_match(RhorrpMuffinTinMatchInput {
        muffin_tin_radius: input.solver.muffin_tin_radius,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: input.solver.target_kappa,
    })?;
    let irregular_initial_condition =
        rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
            muffin_tin_radius: input.solver.muffin_tin_radius,
            phase_shift: muffin_tin_match.phase_shift,
            wave_number: input.wave_number,
            bessel_j_l: muffin_tin_match.bessel_j_l,
            neumann_l: muffin_tin_match.neumann_l,
            bessel_j_l_plus_1: muffin_tin_match.bessel_j_l_plus_1,
            neumann_l_plus_1: muffin_tin_match.neumann_l_plus_1,
        })?;

    let irregular_input = FovrgDiracSolverInput {
        irregular: true,
        muffin_tin_large_component: irregular_initial_condition.large_component,
        muffin_tin_small_component: irregular_initial_condition.small_component,
        ..input.solver
    };
    let irregular_solution = fovrg_dirac_solver(irregular_input)?;

    validate_radial_solution_len(
        "irregular_large_component",
        regular_solution.active_len,
        irregular_solution.large_component.len(),
    )?;
    validate_radial_solution_len(
        "irregular_small_component",
        regular_solution.active_len,
        irregular_solution.small_component.len(),
    )?;
    let active_radii = input
        .solver
        .radii
        .slice_axis(Axis(0), Slice::from(..regular_solution.active_len))
        .to_vec();
    let radial_solutions = rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
        radii: &active_radii,
        raw_regular_large: regular_solution.large_component.view(),
        raw_regular_small: regular_solution.small_component.view(),
        raw_irregular_large: irregular_solution.large_component.view(),
        raw_irregular_small: irregular_solution.small_component.view(),
        phase_shift: muffin_tin_match.phase_shift,
        phase_amplitude: muffin_tin_match.phase_amplitude,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        match_index_1based: input.solver.radial_match_index + 1,
        exact_tail_start_index_1based: input.solver.radial_match_index + 1,
    })?;

    Ok(RhorrpWavefunctionChannel {
        muffin_tin_match,
        irregular_initial_condition,
        radial_solutions,
        regular_active_len: regular_solution.active_len,
        irregular_active_len: irregular_solution.active_len,
        regular_iteration_count: regular_solution.iteration_count,
        irregular_iteration_count: irregular_solution.iteration_count,
        difficult_iterations: regular_solution.difficult_iterations
            + irregular_solution.difficult_iterations,
    })
}

/// FEFF RHORRP photoelectron kappa for ordinary angular momentum `l`.
///
/// The scalar RHORRP wavefunction loop uses the negative branch
/// `ikap = -(l + 1)` for each ordinary `l` channel.
pub fn rhorrp_photoelectron_kappa(angular_momentum: usize) -> Result<i32, RhorrpError> {
    let abs_kappa = angular_momentum
        .checked_add(1)
        .ok_or(RhorrpError::PhotoelectronKappaOutOfRange { angular_momentum })?;
    let abs_kappa = i32::try_from(abs_kappa)
        .map_err(|_| RhorrpError::PhotoelectronKappaOutOfRange { angular_momentum })?;
    Ok(-abs_kappa)
}

/// FEFF RHORRP C3 correction selector for ordinary angular momentum `l`.
///
/// `init_wavefunctions` sets `ic3 = 0` for the s-wave channel and `ic3 = 1`
/// for all higher ordinary angular momenta before both `dfovrg` calls.
#[must_use]
pub fn rhorrp_c3_scale_for_angular_momentum(angular_momentum: usize) -> i32 {
    if angular_momentum == 0 { 0 } else { 1 }
}

/// Port one FEFF `init_wavefunctions` potential block.
///
/// This builds the per-energy/per-`l` RHORRP phase and radial wavefunction
/// tables for one potential from the already shifted potential arrays. The
/// caller supplies a base FOVRG input, and this helper applies FEFF's
/// `init_wavefunctions` setup for each contour energy and angular channel.
pub fn rhorrp_potential_wavefunctions(
    input: RhorrpPotentialWavefunctionsInput<'_>,
) -> Result<RhorrpPotentialWavefunctions, RhorrpError> {
    validate_potential_wavefunctions_input(input)?;

    let energy_count = input.energies_hartree.len();
    let angular_count = input.angular_momentum_count;
    let mut setups = Vec::with_capacity(energy_count);
    let mut wave_numbers = Array1::<Complex>::zeros(energy_count);
    let mut phase_shifts = Array2::<Complex>::zeros((energy_count, angular_count).f());
    let mut regular_large = None;
    let mut irregular_large = None;
    let mut regular_small = None;
    let mut irregular_small = None;
    let mut radial_count = None;
    let mut regular_iteration_count = 0usize;
    let mut irregular_iteration_count = 0usize;
    let mut difficult_iterations = 0usize;

    for energy_index in 0..energy_count {
        let setup = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
            energy_hartree: input.energies_hartree[energy_index],
            reference_energy_hartree: input.reference_energy_hartree,
            muffin_tin_radius: input.solver.muffin_tin_radius,
            norman_radius: input.norman_radius,
            radial_x0: input.radial_x0,
            radial_dx: input.radial_dx,
            radial_capacity: input.solver.radii.len(),
            exchange_index: input.exchange_index,
        })?;
        wave_numbers[energy_index] = setup.wave_number;

        for angular in 0..angular_count {
            let solver = FovrgDiracSolverInput {
                exchange_cycle_count: setup.dirac_cycle_count,
                target_kappa: rhorrp_photoelectron_kappa(angular)?,
                target_last_index: setup.last_integration_index_1based - 1,
                energy: setup.kinetic_energy_hartree,
                irregular: false,
                c3_scale: rhorrp_c3_scale_for_angular_momentum(angular),
                muffin_tin_large_component: Complex::new(0.0, 0.0),
                muffin_tin_small_component: Complex::new(0.0, 0.0),
                ..input.solver
            };
            let channel = rhorrp_wavefunction_channel(RhorrpWavefunctionChannelInput {
                solver,
                angular_momentum: angular,
                wave_number: setup.wave_number,
            })?;
            let channel_radial_count = channel.radial_solutions.row_count();
            if let Some(expected) = radial_count {
                if channel_radial_count != expected {
                    return Err(RhorrpError::WavefunctionChannelLengthMismatch {
                        energy: energy_index,
                        angular,
                        expected,
                        actual: channel_radial_count,
                    });
                }
            } else {
                radial_count = Some(channel_radial_count);
                regular_large = Some(Array3::<Complex>::zeros(
                    (energy_count, angular_count, channel_radial_count).f(),
                ));
                irregular_large = Some(Array3::<Complex>::zeros(
                    (energy_count, angular_count, channel_radial_count).f(),
                ));
                regular_small = Some(Array3::<Complex>::zeros(
                    (energy_count, angular_count, channel_radial_count).f(),
                ));
                irregular_small = Some(Array3::<Complex>::zeros(
                    (energy_count, angular_count, channel_radial_count).f(),
                ));
            }

            phase_shifts[(energy_index, angular)] = channel.muffin_tin_match.phase_shift;
            let (
                Some(regular_large),
                Some(irregular_large),
                Some(regular_small),
                Some(irregular_small),
            ) = (
                regular_large.as_mut(),
                irregular_large.as_mut(),
                regular_small.as_mut(),
                irregular_small.as_mut(),
            )
            else {
                return Err(RhorrpError::UninitializedWavefunctionTables);
            };
            assign_wavefunction_channel(
                regular_large,
                irregular_large,
                regular_small,
                irregular_small,
                energy_index,
                angular,
                &channel.radial_solutions,
            );
            regular_iteration_count += channel.regular_iteration_count;
            irregular_iteration_count += channel.irregular_iteration_count;
            difficult_iterations += channel.difficult_iterations;
        }

        setups.push(setup);
    }

    Ok(RhorrpPotentialWavefunctions {
        setups,
        wave_numbers,
        phase_shifts,
        regular_large: regular_large.ok_or(RhorrpError::UninitializedWavefunctionTables)?,
        irregular_large: irregular_large.ok_or(RhorrpError::UninitializedWavefunctionTables)?,
        regular_small: regular_small.ok_or(RhorrpError::UninitializedWavefunctionTables)?,
        irregular_small: irregular_small.ok_or(RhorrpError::UninitializedWavefunctionTables)?,
        regular_iteration_count,
        irregular_iteration_count,
        difficult_iterations,
    })
}

fn assign_wavefunction_channel(
    regular_large: &mut Array3<Complex>,
    irregular_large: &mut Array3<Complex>,
    regular_small: &mut Array3<Complex>,
    irregular_small: &mut Array3<Complex>,
    energy: usize,
    angular: usize,
    channel: &RhorrpRadialSolutionAssembly,
) {
    for radial in 0..channel.row_count() {
        regular_large[(energy, angular, radial)] = channel.regular_large_components[radial];
        irregular_large[(energy, angular, radial)] = channel.irregular_large_components[radial];
        regular_small[(energy, angular, radial)] = channel.regular_small_components[radial];
        irregular_small[(energy, angular, radial)] = channel.irregular_small_components[radial];
    }
}

fn validate_potential_wavefunctions_input(
    input: RhorrpPotentialWavefunctionsInput<'_>,
) -> Result<(), RhorrpError> {
    let energy_count = input.energies_hartree.len();
    let radial_count = input.solver.radii.len();
    if energy_count == 0 || input.angular_momentum_count == 0 || radial_count == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy: energy_count,
            angular: input.angular_momentum_count,
            radial: radial_count,
        });
    }
    validate_complex_result(
        "potential_wavefunctions_reference_energy",
        input.reference_energy_hartree,
    )?;
    validate_scalar(
        "potential_wavefunctions_norman_radius",
        0,
        input.norman_radius,
    )?;
    validate_scalar("potential_wavefunctions_radial_x0", 0, input.radial_x0)?;
    validate_scalar("potential_wavefunctions_radial_dx", 0, input.radial_dx)?;
    rhorrp_photoelectron_kappa(input.angular_momentum_count - 1)?;
    Ok(())
}

/// Port the all-potential FEFF `init_wavefunctions` table assembly from prepared grids.
///
/// This mirrors the FEFF `IPH_LOOP` after `fixvar`, `fixdsx`, and `eref0`
/// have already prepared the radial arrays. RHORRP currently assumes the same
/// angular table width for all potentials, matching the FEFF note that
/// `lmaxph` must be uniform for the saved `gg_slice` indexing.
pub fn rhorrp_prepared_wavefunction_tables(
    input: RhorrpPreparedWavefunctionTablesInput<'_>,
) -> Result<RhorrpWavefunctionTables, RhorrpError> {
    let potential_count = validate_prepared_wavefunction_tables_input(input)?;
    let first =
        rhorrp_prepared_potential_wavefunctions(prepared_potential_wavefunctions_input(input, 0))?;
    let energy_count = first.energy_count();
    let angular_count = first.angular_momentum_count();
    let radial_count = first.radial_count();
    let mut output = RhorrpWavefunctionTables {
        setups_by_potential: Vec::with_capacity(potential_count),
        wave_numbers: Array2::<Complex>::zeros((energy_count, potential_count).f()),
        phase_shifts: Array3::<Complex>::zeros((energy_count, angular_count, potential_count).f()),
        regular_large: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        irregular_large: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        regular_small: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        irregular_small: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        regular_iteration_count: 0,
        irregular_iteration_count: 0,
        difficult_iterations: 0,
    };

    assign_potential_wavefunctions(&mut output, 0, &first);
    output.regular_iteration_count += first.regular_iteration_count;
    output.irregular_iteration_count += first.irregular_iteration_count;
    output.difficult_iterations += first.difficult_iterations;
    output.setups_by_potential.push(first.setups.clone());

    for potential_index in 1..potential_count {
        let potential = rhorrp_prepared_potential_wavefunctions(
            prepared_potential_wavefunctions_input(input, potential_index),
        )?;
        validate_wavefunction_potential_shape(
            potential_index,
            energy_count,
            angular_count,
            radial_count,
            &potential,
        )?;
        assign_potential_wavefunctions(&mut output, potential_index, &potential);
        output.regular_iteration_count += potential.regular_iteration_count;
        output.irregular_iteration_count += potential.irregular_iteration_count;
        output.difficult_iterations += potential.difficult_iterations;
        output.setups_by_potential.push(potential.setups.clone());
    }

    Ok(output)
}

fn prepared_potential_wavefunctions_input<'a>(
    input: RhorrpPreparedWavefunctionTablesInput<'a>,
    potential_index: usize,
) -> RhorrpPreparedPotentialWavefunctionsInput<'a> {
    RhorrpPreparedPotentialWavefunctionsInput {
        prepared: input.prepared,
        potential_index,
        energies_hartree: input.energies_hartree,
        muffin_tin_radius: input.muffin_tin_radii[potential_index],
        norman_radius: input.norman_radii[potential_index],
        bound_large_coefficients: input
            .bound_large_coefficients_by_potential
            .index_axis_move(Axis(2), potential_index),
        bound_small_coefficients: input
            .bound_small_coefficients_by_potential
            .index_axis_move(Axis(2), potential_index),
        electron_counts: input
            .electron_counts_by_potential
            .index_axis_move(Axis(1), potential_index),
        valence_counts: input
            .valence_counts_by_potential
            .index_axis_move(Axis(1), potential_index),
        kappa: input
            .kappa_by_potential
            .index_axis_move(Axis(1), potential_index),
        atomic_number: input.atomic_numbers[potential_index],
        exchange_index: input.exchange_index,
        angular_momentum_count: input.angular_momentum_count,
        bound_orbital_count: input.bound_orbital_counts[potential_index],
    }
}

fn validate_prepared_wavefunction_tables_input(
    input: RhorrpPreparedWavefunctionTablesInput<'_>,
) -> Result<usize, RhorrpError> {
    let potential_count = input.prepared.potential_count();
    if potential_count == 0 {
        return Err(RhorrpError::InvalidWavefunctionPotentialCount { potential_count });
    }
    validate_prepared_wavefunction_metadata_len(
        "muffin_tin_radii",
        potential_count,
        input.muffin_tin_radii.len(),
    )?;
    validate_prepared_wavefunction_metadata_len(
        "norman_radii",
        potential_count,
        input.norman_radii.len(),
    )?;
    validate_prepared_wavefunction_metadata_len(
        "atomic_numbers",
        potential_count,
        input.atomic_numbers.len(),
    )?;
    validate_prepared_wavefunction_metadata_len(
        "bound_large_coefficients_by_potential",
        potential_count,
        input.bound_large_coefficients_by_potential.dim().2,
    )?;
    validate_prepared_wavefunction_metadata_len(
        "bound_small_coefficients_by_potential",
        potential_count,
        input.bound_small_coefficients_by_potential.dim().2,
    )?;
    validate_prepared_wavefunction_metadata_len(
        "electron_counts_by_potential",
        potential_count,
        input.electron_counts_by_potential.dim().1,
    )?;
    validate_prepared_wavefunction_metadata_len(
        "valence_counts_by_potential",
        potential_count,
        input.valence_counts_by_potential.dim().1,
    )?;
    validate_prepared_wavefunction_metadata_len(
        "kappa_by_potential",
        potential_count,
        input.kappa_by_potential.dim().1,
    )?;
    validate_prepared_wavefunction_metadata_len(
        "bound_orbital_counts",
        potential_count,
        input.bound_orbital_counts.len(),
    )?;
    Ok(potential_count)
}

fn validate_prepared_wavefunction_metadata_len(
    component: &'static str,
    expected_potentials: usize,
    actual_potentials: usize,
) -> Result<(), RhorrpError> {
    if actual_potentials != expected_potentials {
        return Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component,
            expected_potentials,
            actual_potentials,
        });
    }
    Ok(())
}

/// Port the all-potential FEFF `init_wavefunctions` table assembly.
///
/// This lifts [`rhorrp_potential_wavefunctions`] into the RHORRP handoff table
/// shapes consumed by `rhoerrp`: `ph2(ne,l,iph)` and
/// `prel/qrel/pnel/qnel(ne,l,radial,iph)`.
pub fn rhorrp_wavefunction_tables(
    input: RhorrpWavefunctionTablesInput<'_>,
) -> Result<RhorrpWavefunctionTables, RhorrpError> {
    if input.potentials.is_empty() {
        return Err(RhorrpError::InvalidWavefunctionPotentialCount { potential_count: 0 });
    }

    let first = rhorrp_potential_wavefunctions(input.potentials[0])?;
    let energy_count = first.energy_count();
    let angular_count = first.angular_momentum_count();
    let radial_count = first.radial_count();
    let potential_count = input.potentials.len();
    let mut output = RhorrpWavefunctionTables {
        setups_by_potential: Vec::with_capacity(potential_count),
        wave_numbers: Array2::<Complex>::zeros((energy_count, potential_count).f()),
        phase_shifts: Array3::<Complex>::zeros((energy_count, angular_count, potential_count).f()),
        regular_large: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        irregular_large: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        regular_small: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        irregular_small: Array4::<Complex>::zeros(
            (energy_count, angular_count, radial_count, potential_count).f(),
        ),
        regular_iteration_count: 0,
        irregular_iteration_count: 0,
        difficult_iterations: 0,
    };

    assign_potential_wavefunctions(&mut output, 0, &first);
    output.regular_iteration_count += first.regular_iteration_count;
    output.irregular_iteration_count += first.irregular_iteration_count;
    output.difficult_iterations += first.difficult_iterations;
    output.setups_by_potential.push(first.setups.clone());

    for (potential_index, &potential_input) in input.potentials.iter().enumerate().skip(1) {
        let potential = rhorrp_potential_wavefunctions(potential_input)?;
        validate_wavefunction_potential_shape(
            potential_index,
            energy_count,
            angular_count,
            radial_count,
            &potential,
        )?;
        assign_potential_wavefunctions(&mut output, potential_index, &potential);
        output.regular_iteration_count += potential.regular_iteration_count;
        output.irregular_iteration_count += potential.irregular_iteration_count;
        output.difficult_iterations += potential.difficult_iterations;
        output.setups_by_potential.push(potential.setups.clone());
    }

    Ok(output)
}

fn assign_potential_wavefunctions(
    output: &mut RhorrpWavefunctionTables,
    potential_index: usize,
    potential: &RhorrpPotentialWavefunctions,
) {
    for energy in 0..potential.energy_count() {
        output.wave_numbers[(energy, potential_index)] = potential.wave_numbers[energy];
        for angular in 0..potential.angular_momentum_count() {
            output.phase_shifts[(energy, angular, potential_index)] =
                potential.phase_shifts[(energy, angular)];
            for radial in 0..potential.radial_count() {
                output.regular_large[(energy, angular, radial, potential_index)] =
                    potential.regular_large[(energy, angular, radial)];
                output.irregular_large[(energy, angular, radial, potential_index)] =
                    potential.irregular_large[(energy, angular, radial)];
                output.regular_small[(energy, angular, radial, potential_index)] =
                    potential.regular_small[(energy, angular, radial)];
                output.irregular_small[(energy, angular, radial, potential_index)] =
                    potential.irregular_small[(energy, angular, radial)];
            }
        }
    }
}

fn validate_wavefunction_potential_shape(
    potential_index: usize,
    expected_energy: usize,
    expected_angular: usize,
    expected_radial: usize,
    potential: &RhorrpPotentialWavefunctions,
) -> Result<(), RhorrpError> {
    let actual_energy = potential.energy_count();
    let actual_angular = potential.angular_momentum_count();
    let actual_radial = potential.radial_count();
    if actual_energy != expected_energy
        || actual_angular != expected_angular
        || actual_radial != expected_radial
    {
        return Err(RhorrpError::WavefunctionPotentialShapeMismatch {
            potential: potential_index,
            expected_energy,
            expected_angular,
            expected_radial,
            actual_energy,
            actual_angular,
            actual_radial,
        });
    }
    Ok(())
}

fn validate_radial_solution_assembly_input(
    input: RhorrpRadialSolutionAssemblyInput<'_>,
) -> Result<(), RhorrpError> {
    let radial_count = input.radii.len();
    if radial_count == 0 {
        return Err(RhorrpError::InvalidRadialCount { radial_count: 0 });
    }
    validate_radial_solution_len(
        "raw_regular_large",
        radial_count,
        input.raw_regular_large.len(),
    )?;
    validate_radial_solution_len(
        "raw_regular_small",
        radial_count,
        input.raw_regular_small.len(),
    )?;
    validate_radial_solution_len(
        "raw_irregular_large",
        radial_count,
        input.raw_irregular_large.len(),
    )?;
    validate_radial_solution_len(
        "raw_irregular_small",
        radial_count,
        input.raw_irregular_small.len(),
    )?;
    if input.match_index_1based == 0 || input.match_index_1based > radial_count {
        return Err(RhorrpError::RadialSolutionMatchIndexOutOfRange {
            match_index_1based: input.match_index_1based,
            radial_count,
        });
    }
    validate_complex_result("radial_solution_phase_shift", input.phase_shift)?;
    validate_complex_result("radial_solution_phase_amplitude", input.phase_amplitude)?;
    validate_complex_result("radial_solution_wave_number", input.wave_number)?;
    for (row, &radius) in input.radii.iter().enumerate() {
        validate_scalar("radial_solution_radius", row, radius)?;
        if radius <= 0.0 {
            return Err(RhorrpError::InvalidPositiveRadius {
                name: "radial_solution_radius",
                value: radius,
            });
        }
    }
    validate_complex_array("raw_regular_large", input.raw_regular_large)?;
    validate_complex_array("raw_regular_small", input.raw_regular_small)?;
    validate_complex_array("raw_irregular_large", input.raw_irregular_large)?;
    validate_complex_array("raw_irregular_small", input.raw_irregular_small)?;
    Ok(())
}

fn validate_radial_solution_len(
    component: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), RhorrpError> {
    if actual != expected {
        return Err(RhorrpError::RadialSolutionLengthMismatch {
            component,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_complex_array(
    name: &'static str,
    values: ArrayView1<'_, Complex>,
) -> Result<(), RhorrpError> {
    for (index, &value) in values.iter().enumerate() {
        validate_scalar(name, index * 2, value.re)?;
        validate_scalar(name, index * 2 + 1, value.im)?;
    }
    Ok(())
}

fn rhorrp_small_component_factor(wave_number: Complex) -> Result<Complex, RhorrpError> {
    let one = Complex::new(1.0, 0.0);
    let alpha_wave = wave_number * FEFF_FINE_STRUCTURE_ALPHA;
    let denominator = one + (one + alpha_wave * alpha_wave).sqrt();
    validate_complex_result("small_component_denominator", denominator)?;
    if denominator == Complex::new(0.0, 0.0) {
        return Err(RhorrpError::ZeroComplexResult {
            name: "small_component_denominator",
        });
    }

    let factor = -alpha_wave / denominator;
    validate_complex_result("small_component_factor", factor)?;
    Ok(factor)
}

fn validate_positive_radius(name: &'static str, value: Real) -> Result<(), RhorrpError> {
    validate_scalar(name, 0, value)?;
    if value <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius { name, value });
    }
    Ok(())
}
