//! FEFF atomic-configuration helpers.
//!
//! This module ports the deterministic occupation-compaction part of
//! `COMMON/getorb.f90`. The large FEFF default tables from `m_config.f90` are
//! intentionally kept outside this helper: callers provide the already-selected
//! 40-slot occupation, valence, spin, and next-element occupation rows, and this
//! code applies FEFF's core-hole, screening-electron, ionicity, and high-l
//! freezing rules.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::Real;

/// Number of FEFF relativistic orbital slots in `COMMON/m_config.f90`.
pub const FEFF_ORBITAL_SLOT_COUNT: usize = 40;

/// FEFF `getorb` projection map covers `kappa = -5..=4`.
pub const FEFF_KAPPA_PROJECTION_COUNT: usize = 10;

/// FEFF principal quantum numbers for the 40 configuration slots.
pub const FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS: [i32; FEFF_ORBITAL_SLOT_COUNT] = [
    1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8,
    8, 8, 7, 7, 6, 6, 5, 5,
];

/// FEFF relativistic `kappa` values for the 40 configuration slots.
pub const FEFF_ORBITAL_KAPPAS: [i32; FEFF_ORBITAL_SLOT_COUNT] = [
    -1, -1, 1, -2, -1, 1, -2, 2, -3, -1, 1, -2, 2, -3, 3, -4, -1, 1, -2, 2, -3, 3, -4, -1, 1, -2,
    2, -3, -1, 1, -2, -1, 1, -2, 2, -3, 3, -4, 4, -5,
];

/// Inputs for applying FEFF `COMMON/getorb.f90` occupation rules.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalConfigurationInput<'a> {
    /// Atomic number `iz` of the requested atom.
    pub atomic_number: usize,
    /// FEFF core-hole selector `ihole`; zero means no core hole.
    pub hole_index: usize,
    /// Ionicity `xion`; FEFF rounds this with `nint` and applies the fractional remainder.
    pub ionicity: Real,
    /// Whether to keep f and higher valence occupations. FEFF freezes them when `iunf == 0`.
    pub unfreeze_f_or_higher: bool,
    /// Occupation row selected from FEFF `f_iocc(index, :, iphl)`.
    pub occupations: ArrayView1<'a, Real>,
    /// Valence occupation row selected from FEFF `f_ival(index, :, iphl)`.
    pub valence_occupations: ArrayView1<'a, Real>,
    /// Spin occupation row selected from FEFF `f_ispn(index, :, iphl)`.
    pub spin_occupations: ArrayView1<'a, Real>,
    /// Next-element occupation row selected from FEFF `f_iocc(index + 1, :, -1)`.
    ///
    /// FEFF uses this row to locate the orbital that accepts a screening electron.
    pub next_occupations: ArrayView1<'a, Real>,
}

/// Compacted FEFF orbital configuration returned by [`orbital_configuration`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitalConfiguration {
    /// Number of occupied or explicitly retained orbitals, FEFF `norb`.
    pub orbital_count: usize,
    /// Number of core orbitals at this point in FEFF `getorb`, FEFF `norbco`.
    pub core_orbital_count: usize,
    /// One-based orbital slot for each projection `kappa = -5..=4`, FEFF `iorb(-5:4)`.
    pub projection_orbitals: Array1<usize>,
    /// One-based compacted orbital containing the core hole, FEFF `iholep`; zero when absent.
    pub hole_position: usize,
    /// Principal quantum numbers for compacted orbitals, FEFF `nqn(1:norb)`.
    pub principal_quantum_numbers: Array1<i32>,
    /// Relativistic kappa values for compacted orbitals, FEFF `nk(1:norb)`.
    pub kappa: Array1<i32>,
    /// Electron occupations after core-hole, screening, and ionicity adjustments, FEFF `xnel`.
    pub electron_counts: Array1<Real>,
    /// Valence occupations after FEFF adjustments, FEFF `xnval`.
    pub valence_counts: Array1<Real>,
    /// Spin magnetization copied from the selected configuration row, FEFF `xmag`.
    pub spin_magnetization: Array1<Real>,
    /// One-based orbital slot used for fractional ionicity adjustment, FEFF `iion`.
    pub ionization_orbital: usize,
    /// One-based orbital slot used for the screening electron, FEFF `iscr`.
    pub screening_orbital: usize,
    /// One-based highest occupied orbital slot before adjustments, FEFF `ilast`.
    pub last_occupied_orbital: usize,
    /// Atomic number of the selected base template, FEFF `index = iz - nint(xion)`.
    pub template_atomic_number: usize,
    /// Fractional ionicity remainder, FEFF `delion = xion - nint(xion)`.
    pub ionicity_delta: Real,
}

/// Error returned by FEFF orbital-configuration helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum OrbitalConfigurationError {
    /// FEFF default configuration tables cover `Z = 1..=139`.
    #[error("FEFF atomic number {atomic_number} is outside 1..=139")]
    InvalidAtomicNumber { atomic_number: usize },
    /// FEFF orbital selectors are 1-based slots in `1..=40`, or zero for no core hole.
    #[error("FEFF hole index {hole_index} is outside 0..=40")]
    InvalidHoleIndex { hole_index: usize },
    /// An input row must contain all 40 FEFF orbital slots.
    #[error("{name} length {actual} is shorter than required length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// FEFF scalar inputs must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// FEFF row inputs must be finite.
    #[error("{name} slot {slot} must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        slot: usize,
        value: Real,
    },
    /// Applying `nint(xion)` selected a template outside FEFF's default table range.
    #[error("FEFF template atomic number {template_atomic_number} is outside 1..=138")]
    InvalidTemplateAtomicNumber { template_atomic_number: isize },
    /// FEFF cannot create a core hole in an orbital containing less than one electron.
    #[error(
        "cannot remove a core electron from hole slot {hole_index} with occupation {occupation}"
    )]
    CoreHoleNotOccupied { hole_index: usize, occupation: Real },
    /// FEFF rejects a core hole combined with ionicity that removes too much charge.
    #[error(
        "cannot remove a core electron from hole slot {hole_index}; occupation {occupation} minus ionicity delta {ionicity_delta} is below one"
    )]
    CoreHoleOverIonized {
        hole_index: usize,
        occupation: Real,
        ionicity_delta: Real,
    },
    /// The adjusted output occupation or valence count is outside the slot capacity.
    #[error(
        "{name} for compacted orbital {orbital} with kappa {kappa} is outside [0,{capacity}], got {value}"
    )]
    OccupationOutOfRange {
        name: &'static str,
        orbital: usize,
        kappa: i32,
        capacity: Real,
        value: Real,
    },
}

/// Port of FEFF `COMMON/getorb.f90` occupation compaction.
///
/// The caller is responsible for selecting the base rows from FEFF's default or
/// user-provided configuration tables. This function then mirrors `getorb`:
/// it rounds ionicity with FEFF `nint`, identifies the core-hole screening and
/// fractional-ionicity orbitals, compacts occupied slots, freezes f and higher
/// valence when requested, and returns the compacted arrays consumed by POT,
/// FOVRG, and related atomic solvers.
pub fn orbital_configuration(
    input: OrbitalConfigurationInput<'_>,
) -> Result<OrbitalConfiguration, OrbitalConfigurationError> {
    validate_input(&input)?;

    let ion = input.ionicity.round() as isize;
    let template_atomic_number = input.atomic_number as isize - ion;
    if !(1..139).contains(&template_atomic_number) {
        return Err(OrbitalConfigurationError::InvalidTemplateAtomicNumber {
            template_atomic_number,
        });
    }
    let template_atomic_number = usize::try_from(template_atomic_number).map_err(|_| {
        OrbitalConfigurationError::InvalidTemplateAtomicNumber {
            template_atomic_number,
        }
    })?;
    let ionicity_delta = input.ionicity - ion as Real;

    let mut ionization_orbital = 0_usize;
    let mut last_occupied_orbital = 0_usize;
    for slot in (1..=FEFF_ORBITAL_SLOT_COUNT).rev() {
        let occupation = input.occupations[slot - 1];
        if ionization_orbital == 0 && occupation > ionicity_delta {
            ionization_orbital = slot;
        }
        if last_occupied_orbital == 0 && occupation > 0.0 {
            last_occupied_orbital = slot;
        }
    }

    let mut hole_position = input.hole_index;
    if input.hole_index > 0 {
        let occupation = input.occupations[input.hole_index - 1];
        if occupation < 1.0 {
            return Err(OrbitalConfigurationError::CoreHoleNotOccupied {
                hole_index: input.hole_index,
                occupation,
            });
        }
        if input.hole_index == ionization_orbital
            && ionicity_delta > 0.0
            && occupation - ionicity_delta < 1.0
        {
            return Err(OrbitalConfigurationError::CoreHoleOverIonized {
                hole_index: input.hole_index,
                occupation,
                ionicity_delta,
            });
        }
    }

    let mut screening_orbital = (1..=FEFF_ORBITAL_SLOT_COUNT)
        .find(|&slot| input.next_occupations[slot - 1] - input.occupations[slot - 1] > 0.5)
        .unwrap_or(0);
    if input.hole_index > 0 && input.occupations[input.hole_index - 1] < 1.5 {
        screening_orbital = input.hole_index;
    }

    if ionicity_delta < 0.0 {
        ionization_orbital = screening_orbital;
        if input.hole_index != 0 && screening_orbital != 0 {
            let screening_capacity = slot_capacity(screening_orbital);
            if input.occupations[screening_orbital - 1] + 1.0 - ionicity_delta > screening_capacity
            {
                ionization_orbital = last_occupied_orbital;
                if last_occupied_orbital == screening_orbital
                    || input.occupations[last_occupied_orbital - 1] - ionicity_delta
                        > slot_capacity(last_occupied_orbital)
                {
                    ionization_orbital = last_occupied_orbital + 1;
                }
            }
        }
    }

    let mut projection_orbitals = Array1::<usize>::zeros(FEFF_KAPPA_PROJECTION_COUNT);
    let mut principal_quantum_numbers = Vec::new();
    let mut kappa_values = Vec::new();
    let mut electron_counts = Vec::new();
    let mut valence_counts = Vec::new();
    let mut spin_magnetization = Vec::new();

    for slot in 1..=FEFF_ORBITAL_SLOT_COUNT {
        let occupation = input.occupations[slot - 1];
        let retained_for_screening = slot == screening_orbital && input.hole_index > 0;
        let retained_for_ionicity = slot == ionization_orbital && occupation - ionicity_delta > 0.0;
        if !(occupation > 0.0 || retained_for_screening || retained_for_ionicity)
            || (slot == input.hole_index && occupation < 1.0)
        {
            continue;
        }

        let kappa = FEFF_ORBITAL_KAPPAS[slot - 1];
        let mut electron_count = occupation;
        if slot == input.hole_index {
            electron_count -= 1.0;
            hole_position = principal_quantum_numbers.len() + 1;
        }
        if retained_for_screening {
            electron_count += 1.0;
        }

        let mut valence_count = input.valence_occupations[slot - 1];
        if !input.unfreeze_f_or_higher && matches!(kappa, -4 | 3 | -5 | 4) {
            valence_count = 0.0;
        }
        if slot == input.hole_index && valence_count >= 1.0 {
            valence_count -= 1.0;
        }
        if retained_for_screening {
            valence_count += 1.0;
        }
        if slot == ionization_orbital {
            electron_count -= ionicity_delta;
            valence_count -= ionicity_delta;
        }

        let compact_orbital = principal_quantum_numbers.len() + 1;
        validate_output_count("electron count", compact_orbital, kappa, electron_count)?;
        validate_output_count("valence count", compact_orbital, kappa, valence_count)?;

        if let Some(projection_index) = kappa_projection_index(kappa) {
            projection_orbitals[projection_index] = slot;
        }
        principal_quantum_numbers.push(FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS[slot - 1]);
        kappa_values.push(kappa);
        electron_counts.push(electron_count);
        valence_counts.push(valence_count);
        spin_magnetization.push(input.spin_occupations[slot - 1]);
    }

    let orbital_count = principal_quantum_numbers.len();
    Ok(OrbitalConfiguration {
        orbital_count,
        core_orbital_count: orbital_count,
        projection_orbitals,
        hole_position,
        principal_quantum_numbers: Array1::from_vec(principal_quantum_numbers),
        kappa: Array1::from_vec(kappa_values),
        electron_counts: Array1::from_vec(electron_counts),
        valence_counts: Array1::from_vec(valence_counts),
        spin_magnetization: Array1::from_vec(spin_magnetization),
        ionization_orbital,
        screening_orbital,
        last_occupied_orbital,
        template_atomic_number,
        ionicity_delta,
    })
}

fn validate_input(input: &OrbitalConfigurationInput<'_>) -> Result<(), OrbitalConfigurationError> {
    if !(1..=139).contains(&input.atomic_number) {
        return Err(OrbitalConfigurationError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    if input.hole_index > FEFF_ORBITAL_SLOT_COUNT {
        return Err(OrbitalConfigurationError::InvalidHoleIndex {
            hole_index: input.hole_index,
        });
    }
    if !input.ionicity.is_finite() {
        return Err(OrbitalConfigurationError::NonFiniteScalar {
            name: "ionicity",
            value: input.ionicity,
        });
    }
    validate_len("occupations", input.occupations.len())?;
    validate_len("valence_occupations", input.valence_occupations.len())?;
    validate_len("spin_occupations", input.spin_occupations.len())?;
    validate_len("next_occupations", input.next_occupations.len())?;
    validate_values("occupations", input.occupations)?;
    validate_values("valence_occupations", input.valence_occupations)?;
    validate_values("spin_occupations", input.spin_occupations)?;
    validate_values("next_occupations", input.next_occupations)?;
    Ok(())
}

fn validate_len(name: &'static str, len: usize) -> Result<(), OrbitalConfigurationError> {
    if len < FEFF_ORBITAL_SLOT_COUNT {
        Err(OrbitalConfigurationError::LengthTooShort {
            name,
            required: FEFF_ORBITAL_SLOT_COUNT,
            actual: len,
        })
    } else {
        Ok(())
    }
}

fn validate_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), OrbitalConfigurationError> {
    for slot in 1..=FEFF_ORBITAL_SLOT_COUNT {
        let value = values[slot - 1];
        if !value.is_finite() {
            return Err(OrbitalConfigurationError::NonFiniteValue { name, slot, value });
        }
    }
    Ok(())
}

fn validate_output_count(
    name: &'static str,
    orbital: usize,
    kappa: i32,
    value: Real,
) -> Result<(), OrbitalConfigurationError> {
    let capacity = 2.0 * Real::from(kappa.abs());
    if value < 0.0 || value > capacity {
        Err(OrbitalConfigurationError::OccupationOutOfRange {
            name,
            orbital,
            kappa,
            capacity,
            value,
        })
    } else {
        Ok(())
    }
}

fn slot_capacity(slot: usize) -> Real {
    2.0 * Real::from(FEFF_ORBITAL_KAPPAS[slot - 1].abs())
}

fn kappa_projection_index(kappa: i32) -> Option<usize> {
    let index = kappa + 5;
    usize::try_from(index)
        .ok()
        .filter(|&index| index < FEFF_KAPPA_PROJECTION_COUNT)
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::{
        FEFF_ORBITAL_SLOT_COUNT, OrbitalConfigurationError, OrbitalConfigurationInput,
        orbital_configuration,
    };
    use crate::Real;

    #[test]
    fn orbital_configuration_matches_feff_getorb_core_hole_reference()
    -> Result<(), OrbitalConfigurationError> {
        let (occupations, valence_occupations, spin_occupations, next_occupations) =
            iron_like_configuration();

        let configuration = orbital_configuration(OrbitalConfigurationInput {
            atomic_number: 26,
            hole_index: 1,
            ionicity: 0.0,
            unfreeze_f_or_higher: false,
            occupations: occupations.view(),
            valence_occupations: valence_occupations.view(),
            spin_occupations: spin_occupations.view(),
            next_occupations: next_occupations.view(),
        })?;

        assert_eq!(configuration.orbital_count, 10);
        assert_eq!(configuration.core_orbital_count, 10);
        assert_eq!(configuration.hole_position, 1);
        assert_eq!(configuration.ionization_orbital, 10);
        assert_eq!(configuration.screening_orbital, 9);
        assert_eq!(configuration.last_occupied_orbital, 10);
        assert_eq!(configuration.template_atomic_number, 26);
        assert_close(configuration.ionicity_delta, 0.0);
        assert_eq!(
            configuration.projection_orbitals.to_vec(),
            vec![0, 0, 9, 7, 10, 0, 6, 8, 0, 0]
        );
        assert_eq!(
            configuration.principal_quantum_numbers.to_vec(),
            vec![1, 2, 2, 2, 3, 3, 3, 3, 3, 4]
        );
        assert_eq!(
            configuration.kappa.to_vec(),
            vec![-1, -1, 1, -2, -1, 1, -2, 2, -3, -1]
        );
        assert_close_vec(
            &configuration.electron_counts,
            &[1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 3.0, 4.0, 2.0],
        );
        assert_close_vec(
            &configuration.valence_counts,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 2.0],
        );
        assert_close_vec(
            &configuration.spin_magnetization,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn orbital_configuration_matches_feff_getorb_ionicity_reference()
    -> Result<(), OrbitalConfigurationError> {
        let (mut occupations, mut valence_occupations, spin_occupations, _) =
            iron_like_configuration();
        occupations[8] = 2.0;
        valence_occupations[8] = 2.0;
        let mut next_occupations = occupations.clone();
        next_occupations[8] = 3.0;

        let configuration = orbital_configuration(OrbitalConfigurationInput {
            atomic_number: 26,
            hole_index: 0,
            ionicity: 0.6,
            unfreeze_f_or_higher: false,
            occupations: occupations.view(),
            valence_occupations: valence_occupations.view(),
            spin_occupations: spin_occupations.view(),
            next_occupations: next_occupations.view(),
        })?;

        assert_eq!(configuration.orbital_count, 10);
        assert_eq!(configuration.hole_position, 0);
        assert_eq!(configuration.ionization_orbital, 9);
        assert_eq!(configuration.screening_orbital, 9);
        assert_eq!(configuration.template_atomic_number, 25);
        assert_close(configuration.ionicity_delta, -0.4);
        assert_close_vec(
            &configuration.electron_counts,
            &[2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 3.0, 2.4, 2.0],
        );
        assert_close_vec(
            &configuration.valence_counts,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.4, 2.0],
        );
        Ok(())
    }

    #[test]
    fn orbital_configuration_freezes_f_and_higher_like_feff_getorb()
    -> Result<(), OrbitalConfigurationError> {
        let (occupations, valence_occupations, spin_occupations, next_occupations) =
            f_state_configuration();

        let frozen = orbital_configuration(OrbitalConfigurationInput {
            atomic_number: 58,
            hole_index: 0,
            ionicity: 0.0,
            unfreeze_f_or_higher: false,
            occupations: occupations.view(),
            valence_occupations: valence_occupations.view(),
            spin_occupations: spin_occupations.view(),
            next_occupations: next_occupations.view(),
        })?;
        let unfrozen = orbital_configuration(OrbitalConfigurationInput {
            atomic_number: 58,
            hole_index: 0,
            ionicity: 0.0,
            unfreeze_f_or_higher: true,
            occupations: occupations.view(),
            valence_occupations: valence_occupations.view(),
            spin_occupations: spin_occupations.view(),
            next_occupations: next_occupations.view(),
        })?;

        assert_eq!(frozen.orbital_count, 7);
        assert_eq!(
            frozen.projection_orbitals.to_vec(),
            vec![0, 16, 0, 4, 10, 0, 3, 0, 15, 0]
        );
        assert_close_vec(&frozen.valence_counts, &[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        assert_close_vec(
            &unfrozen.valence_counts,
            &[0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn orbital_configuration_rejects_invalid_inputs() {
        let (occupations, valence_occupations, spin_occupations, next_occupations) =
            iron_like_configuration();
        assert_eq!(
            orbital_configuration(OrbitalConfigurationInput {
                atomic_number: 0,
                hole_index: 0,
                ionicity: 0.0,
                unfreeze_f_or_higher: false,
                occupations: occupations.view(),
                valence_occupations: valence_occupations.view(),
                spin_occupations: spin_occupations.view(),
                next_occupations: next_occupations.view(),
            }),
            Err(OrbitalConfigurationError::InvalidAtomicNumber { atomic_number: 0 })
        );
        assert_eq!(
            orbital_configuration(OrbitalConfigurationInput {
                atomic_number: 26,
                hole_index: 41,
                ionicity: 0.0,
                unfreeze_f_or_higher: false,
                occupations: occupations.view(),
                valence_occupations: valence_occupations.view(),
                spin_occupations: spin_occupations.view(),
                next_occupations: next_occupations.view(),
            }),
            Err(OrbitalConfigurationError::InvalidHoleIndex { hole_index: 41 })
        );
        assert_eq!(
            orbital_configuration(OrbitalConfigurationInput {
                atomic_number: 26,
                hole_index: 1,
                ionicity: 0.0,
                unfreeze_f_or_higher: false,
                occupations: Array1::zeros(3).view(),
                valence_occupations: valence_occupations.view(),
                spin_occupations: spin_occupations.view(),
                next_occupations: next_occupations.view(),
            }),
            Err(OrbitalConfigurationError::LengthTooShort {
                name: "occupations",
                required: FEFF_ORBITAL_SLOT_COUNT,
                actual: 3,
            })
        );

        let mut empty_core = occupations.clone();
        empty_core[0] = 0.5;
        assert_eq!(
            orbital_configuration(OrbitalConfigurationInput {
                atomic_number: 26,
                hole_index: 1,
                ionicity: 0.0,
                unfreeze_f_or_higher: false,
                occupations: empty_core.view(),
                valence_occupations: valence_occupations.view(),
                spin_occupations: spin_occupations.view(),
                next_occupations: next_occupations.view(),
            }),
            Err(OrbitalConfigurationError::CoreHoleNotOccupied {
                hole_index: 1,
                occupation: 0.5,
            })
        );
    }

    fn iron_like_configuration() -> (Array1<Real>, Array1<Real>, Array1<Real>, Array1<Real>) {
        let mut occupations = Array1::<Real>::zeros(FEFF_ORBITAL_SLOT_COUNT);
        let mut valence_occupations = Array1::<Real>::zeros(FEFF_ORBITAL_SLOT_COUNT);
        let mut spin_occupations = Array1::<Real>::zeros(FEFF_ORBITAL_SLOT_COUNT);
        for (slot, value) in [
            (1, 2.0),
            (2, 2.0),
            (3, 2.0),
            (4, 4.0),
            (5, 2.0),
            (6, 2.0),
            (7, 4.0),
            (8, 3.0),
            (9, 3.0),
            (10, 2.0),
        ] {
            occupations[slot - 1] = value;
        }
        valence_occupations[7] = 3.0;
        valence_occupations[8] = 3.0;
        valence_occupations[9] = 2.0;
        spin_occupations[7] = 1.0;
        spin_occupations[8] = 2.0;
        spin_occupations[9] = 1.0;
        let mut next_occupations = occupations.clone();
        next_occupations[8] += 1.0;
        (
            occupations,
            valence_occupations,
            spin_occupations,
            next_occupations,
        )
    }

    fn f_state_configuration() -> (Array1<Real>, Array1<Real>, Array1<Real>, Array1<Real>) {
        let mut occupations = Array1::<Real>::zeros(FEFF_ORBITAL_SLOT_COUNT);
        let mut valence_occupations = Array1::<Real>::zeros(FEFF_ORBITAL_SLOT_COUNT);
        let mut spin_occupations = Array1::<Real>::zeros(FEFF_ORBITAL_SLOT_COUNT);
        for (slot, value) in [
            (1, 2.0),
            (2, 2.0),
            (3, 2.0),
            (4, 4.0),
            (10, 2.0),
            (15, 5.0),
            (16, 1.0),
        ] {
            occupations[slot - 1] = value;
            valence_occupations[slot - 1] = value;
        }
        valence_occupations[0] = 0.0;
        valence_occupations[1] = 0.0;
        valence_occupations[2] = 0.0;
        valence_occupations[3] = 0.0;
        spin_occupations[14] = 2.0;
        spin_occupations[15] = 1.0;
        let mut next_occupations = occupations.clone();
        next_occupations[15] = 2.0;
        (
            occupations,
            valence_occupations,
            spin_occupations,
            next_occupations,
        )
    }

    fn assert_close_vec(actual: &Array1<Real>, expected: &[Real]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "index={index} actual={actual} expected={expected}",
            );
        }
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}",
        );
    }
}
