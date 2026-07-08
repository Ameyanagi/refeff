//! FEFF state-ket construction.
//!
//! This ports the common `getkts` loop from `m_stkets.f90`, which enumerates
//! the scattering basis states `|iat,l,m,spin>` for the current cluster and
//! records where each potential's representative atom starts in that state
//! list.

use ndarray::{Array2, ShapeBuilder};

/// One FEFF scattering basis ket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateKet {
    /// FEFF one-based atom index.
    pub atom: usize,
    /// Orbital angular momentum.
    pub angular_momentum: usize,
    /// Magnetic quantum number.
    pub magnetic: isize,
    /// FEFF one-based spin index.
    pub spin: usize,
}

/// Generated state-ket table and representative offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateKetSet {
    /// State kets in FEFF enumeration order.
    pub states: Vec<StateKet>,
    /// For each potential index, the zero-based offset before that
    /// representative atom's first state. `None` means no atom for the
    /// potential was present.
    pub representative_offsets: Vec<Option<usize>>,
}

/// Error returned by state-ket construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StateKetError {
    /// FEFF requires at least one spin channel.
    #[error("spin channel count must be positive")]
    InvalidSpinCount,
    /// A cluster atom referenced a potential outside the `lipotx` table.
    #[error(
        "atom {atom} references potential {potential}, but only {potential_count} potentials are available"
    )]
    PotentialOutOfRange {
        atom: usize,
        potential: usize,
        potential_count: usize,
    },
    /// The generated state count exceeded a caller-provided capacity.
    #[error("state count exceeded capacity {capacity}")]
    CapacityExceeded { capacity: usize },
    /// A generated state field does not fit in `i32` for FEFF array export.
    #[error("state field {field}={value} does not fit in i32")]
    IntegerOverflow { field: &'static str, value: usize },
}

impl StateKetSet {
    /// Return the FEFF `lrstat(4, istatx)` representation in Fortran order.
    ///
    /// Rows are atom index, `l`, `m`, and spin index. The returned array has
    /// shape `(4, states.len())` and column-major strides, matching FEFF's
    /// original storage order.
    pub fn to_lrstat_array(&self) -> Result<Array2<i32>, StateKetError> {
        let mut array = Array2::zeros((4, self.states.len()).f());
        for (column, state) in self.states.iter().enumerate() {
            array[[0, column]] = checked_i32("atom", state.atom)?;
            array[[1, column]] = checked_i32("angular_momentum", state.angular_momentum)?;
            array[[2, column]] =
                i32::try_from(state.magnetic).map_err(|_| StateKetError::IntegerOverflow {
                    field: "magnetic",
                    value: state.magnetic.unsigned_abs(),
                })?;
            array[[3, column]] = checked_i32("spin", state.spin)?;
        }
        Ok(array)
    }
}

/// Construct FEFF state kets without an explicit capacity limit.
pub fn construct_state_kets(
    spin_count: usize,
    atom_potentials: &[usize],
    potential_lmax: &[usize],
    global_lmax: usize,
) -> Result<StateKetSet, StateKetError> {
    construct_state_kets_with_limit(
        spin_count,
        atom_potentials,
        potential_lmax,
        global_lmax,
        None,
    )
}

/// Construct FEFF state kets with an optional `istatx`-style capacity limit.
pub fn construct_state_kets_with_limit(
    spin_count: usize,
    atom_potentials: &[usize],
    potential_lmax: &[usize],
    global_lmax: usize,
    capacity: Option<usize>,
) -> Result<StateKetSet, StateKetError> {
    if spin_count == 0 {
        return Err(StateKetError::InvalidSpinCount);
    }

    let mut states = Vec::new();
    let mut representative_offsets = vec![None; potential_lmax.len()];

    for (atom_index, &potential) in atom_potentials.iter().enumerate() {
        if potential >= potential_lmax.len() {
            return Err(StateKetError::PotentialOutOfRange {
                atom: atom_index + 1,
                potential,
                potential_count: potential_lmax.len(),
            });
        }

        if representative_offsets[potential].is_none() {
            representative_offsets[potential] = Some(states.len());
        }

        let lmax = global_lmax.min(potential_lmax[potential]);
        for angular_momentum in 0..=lmax {
            for magnetic in -(angular_momentum as isize)..=(angular_momentum as isize) {
                for spin in 1..=spin_count {
                    if capacity.is_some_and(|capacity| states.len() >= capacity) {
                        return Err(StateKetError::CapacityExceeded {
                            capacity: states.len(),
                        });
                    }
                    states.push(StateKet {
                        atom: atom_index + 1,
                        angular_momentum,
                        magnetic,
                        spin,
                    });
                }
            }
        }
    }

    Ok(StateKetSet {
        states,
        representative_offsets,
    })
}

fn checked_i32(field: &'static str, value: usize) -> Result<i32, StateKetError> {
    i32::try_from(value).map_err(|_| StateKetError::IntegerOverflow { field, value })
}

#[cfg(test)]
mod tests {
    use super::{StateKet, StateKetError, construct_state_kets, construct_state_kets_with_limit};

    #[test]
    fn constructs_single_spin_state_kets_in_feff_order() -> Result<(), StateKetError> {
        let states = construct_state_kets(1, &[0, 1], &[0, 1], 1)?;

        assert_eq!(
            states.states,
            vec![
                StateKet {
                    atom: 1,
                    angular_momentum: 0,
                    magnetic: 0,
                    spin: 1,
                },
                StateKet {
                    atom: 2,
                    angular_momentum: 0,
                    magnetic: 0,
                    spin: 1,
                },
                StateKet {
                    atom: 2,
                    angular_momentum: 1,
                    magnetic: -1,
                    spin: 1,
                },
                StateKet {
                    atom: 2,
                    angular_momentum: 1,
                    magnetic: 0,
                    spin: 1,
                },
                StateKet {
                    atom: 2,
                    angular_momentum: 1,
                    magnetic: 1,
                    spin: 1,
                },
            ]
        );
        assert_eq!(states.representative_offsets, vec![Some(0), Some(1)]);
        Ok(())
    }

    #[test]
    fn repeats_spin_inside_each_lm_channel() -> Result<(), StateKetError> {
        let states = construct_state_kets(2, &[0], &[1], 1)?;

        assert_eq!(states.states.len(), 8);
        assert_eq!(states.states[0].spin, 1);
        assert_eq!(states.states[1].spin, 2);
        assert_eq!(states.states[2].magnetic, -1);
        assert_eq!(states.states[2].spin, 1);
        assert_eq!(states.states[3].spin, 2);
        Ok(())
    }

    #[test]
    fn exports_lrstat_as_fortran_order_ndarray() -> Result<(), StateKetError> {
        let states = construct_state_kets(1, &[0, 1], &[0, 1], 1)?;
        let lrstat = states.to_lrstat_array()?;

        assert_eq!(lrstat.shape(), &[4, 5]);
        assert_eq!(lrstat.strides(), &[1, 4]);
        assert_eq!(lrstat.column(0).to_vec(), vec![1, 0, 0, 1]);
        assert_eq!(lrstat.column(2).to_vec(), vec![2, 1, -1, 1]);
        Ok(())
    }

    #[test]
    fn caps_lmax_by_global_limit() -> Result<(), StateKetError> {
        let states = construct_state_kets(1, &[0], &[3], 1)?;

        assert_eq!(states.states.len(), 4);
        assert_eq!(
            states.states.last().map(|state| state.angular_momentum),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            construct_state_kets(0, &[0], &[0], 0),
            Err(StateKetError::InvalidSpinCount)
        );
        assert_eq!(
            construct_state_kets(1, &[2], &[0], 0),
            Err(StateKetError::PotentialOutOfRange {
                atom: 1,
                potential: 2,
                potential_count: 1,
            })
        );
        assert_eq!(
            construct_state_kets_with_limit(1, &[0], &[1], 1, Some(2)),
            Err(StateKetError::CapacityExceeded { capacity: 2 })
        );
    }
}
