use super::*;
use crate::angular::{TransitionBMatrixInput, transition_b_matrix};

const TDLDA_GETMAT_MAX_SIZE: usize = 78;

/// Port of FEFF `MATH/bcoef.f90` ordinary transition-index setup.
///
/// FEFF fills eight compact transition slots before the angular matrix:
/// three dipole slots followed by either five E2 slots, three M1 slots, or
/// unphysical zero slots depending on `le2`. This helper preserves the exact
/// `kiind`/`jind`/`lind` convention, including `kap = 0`, `jind = 0`,
/// `lind = -1` for inactive or angular-cap-excluded slots.
pub fn xsph_bcoef_transition_indices(
    input: XsphBcoefTransitionIndicesInput,
) -> Result<XsphBcoefTransitionIndices, XsphError> {
    if input.initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if !matches!(input.higher_multipole_selector, 0..=2) {
        return Err(XsphError::IntegerOutOfRange {
            name: "le2",
            value: input.higher_multipole_selector,
        });
    }
    let max_angular_momentum = usize_to_i32("max_angular_momentum", input.max_angular_momentum)?;

    let mut final_kappas = Array1::<i32>::zeros(8);
    let mut j_indices = Array1::<i32>::zeros(8);
    let mut orbital_l = Array1::<i32>::zeros(8);
    let mut transitions = Vec::with_capacity(8);

    for k in -1..=1 {
        let slot = usize::try_from(k + 2).map_err(|_| XsphError::IntegerOutOfRange {
            name: "bcoef_dipole_slot",
            value: k,
        })?;
        let mut final_kappa =
            input
                .initial_kappa
                .checked_add(k)
                .ok_or(XsphError::IntegerOutOfRange {
                    name: "bcoef_dipole_kappa",
                    value: input.initial_kappa,
                })?;
        if k == 0 {
            final_kappa = final_kappa
                .checked_neg()
                .ok_or(XsphError::IntegerOutOfRange {
                    name: "bcoef_dipole_kappa",
                    value: final_kappa,
                })?;
        }
        let mut index = bcoef_transition_index(slot, final_kappa)?;
        if index.orbital_l > max_angular_momentum {
            index = inactive_bcoef_transition_index(slot);
        }
        store_bcoef_transition_index(&mut final_kappas, &mut j_indices, &mut orbital_l, index);
        transitions.push(index);
    }

    let abs_initial_kappa =
        input
            .initial_kappa
            .checked_abs()
            .ok_or(XsphError::IntegerOutOfRange {
                name: "initial_kappa",
                value: input.initial_kappa,
            })?;
    for k in -2_i32..=2 {
        let slot = usize::try_from(k + 6).map_err(|_| XsphError::IntegerOutOfRange {
            name: "bcoef_higher_multipole_slot",
            value: k,
        })?;
        let mut j_index = abs_initial_kappa
            .checked_add(k)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "bcoef_higher_multipole_j",
                value: abs_initial_kappa,
            })?;
        if j_index <= 0 {
            j_index = 0;
        }

        let mut final_kappa = j_index;
        let abs_k = k.checked_abs().ok_or(XsphError::IntegerOutOfRange {
            name: "bcoef_higher_multipole_delta",
            value: k,
        })?;
        if (input.initial_kappa < 0 && abs_k != 1) || (input.initial_kappa > 0 && abs_k == 1) {
            final_kappa = -j_index;
        }

        let mut index = bcoef_transition_index_with_j(slot, final_kappa, j_index)?;
        if index.orbital_l > max_angular_momentum
            || input.higher_multipole_selector == 0
            || (input.higher_multipole_selector == 1 && abs_k == 2)
        {
            index = inactive_bcoef_transition_index(slot);
        }
        store_bcoef_transition_index(&mut final_kappas, &mut j_indices, &mut orbital_l, index);
        transitions.push(index);
    }

    Ok(XsphBcoefTransitionIndices {
        final_kappas,
        j_indices,
        orbital_l,
        transitions,
    })
}

/// Extract the traced `bcoef` entries used by FEFF `XSPH/xsect.f90`.
///
/// FEFF calls `bcoef(..., ltrace=.true., ...)`, then reads
/// `bmat(0,isp,k2,0,isp,k1)` for direct and same-`l` cross terms. This helper
/// keeps that extraction source-backed by the shared Rust `bcoef` port.
pub fn xsph_xsect_bcoef_weights(
    input: XsphXsectBcoefWeightsInput,
) -> Result<XsphXsectBcoefWeights, XsphError> {
    if input.initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if !matches!(input.higher_multipole_selector, 0..=2) {
        return Err(XsphError::IntegerOutOfRange {
            name: "le2",
            value: input.higher_multipole_selector,
        });
    }

    let transition_matrix = transition_b_matrix(TransitionBMatrixInput {
        lmax: input.max_angular_momentum,
        initial_kappa: input.initial_kappa,
        polarization: input.polarization,
        polarization_tensor: input.polarization_tensor,
        multipole: input.higher_multipole_selector,
        trace_orbital: true,
        spin: input.spin,
        spin_channels: input.spin_channels,
        spin_vector_angle: input.spin_vector_angle,
    })?;
    let selected_spin_index = xsect_bcoef_selected_spin_index(input.spin, input.spin_channels)?;
    let mut trace_weights = Array2::<Complex>::zeros((8, 8));
    for transition2 in 1..=8 {
        for transition1 in 1..=8 {
            trace_weights[(transition2 - 1, transition1 - 1)] = transition_matrix
                .value(
                    0,
                    selected_spin_index,
                    transition2,
                    0,
                    selected_spin_index,
                    transition1,
                )
                .ok_or(XsphError::SizeOutOfRange {
                    name: "xsect_bcoef_transition_slot",
                    value: transition2.max(transition1),
                })?;
        }
    }
    let diagonal_weights =
        Array1::from_iter((0..8).map(|transition| trace_weights[(transition, transition)]));

    Ok(XsphXsectBcoefWeights {
        selected_spin_index,
        final_kappas: Array1::from_vec(transition_matrix.kappa_indices.to_vec()),
        orbital_l: Array1::from_vec(transition_matrix.orbital_momenta.to_vec()),
        trace_weights,
        diagonal_weights,
    })
}

/// The three dipole-allowed final-state branches enumerated by FEFF
/// `TDLDA/getmat.f90`'s `id=1,3` loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TdldaDipoleBranch {
    First,
    Second,
    Third,
}

impl TdldaDipoleBranch {
    const ALL: [Self; 3] = [Self::First, Self::Second, Self::Third];
}

/// Port of FEFF `TDLDA/getmat.f90` channel-basis construction.
///
/// `TDLDA/xsectd.f90` builds this matrix once before the `getchi0` energy loop.
/// The rows enumerate spin-orbit split initial edges, dipole-allowed final
/// states, magnetic substates, core-orbital slots, and projector-orbital slots.
/// Later `getchi0` and `dmscf` work on this `matsize x matsize` basis.
pub fn xsph_tdlda_channel_basis(
    input: XsphTdldaChannelBasisInput<'_>,
) -> Result<XsphTdldaChannelBasis, XsphError> {
    validate_tdlda_channel_basis_input(&input)?;

    let initial_l = input.initial_l;
    let plus_basis_count = input.plus_basis_count.max(1);
    let minus_basis_count = input.minus_basis_count.max(0);
    let plus_basis_count_usize = i32_to_usize("tdlda_plus_basis_count", plus_basis_count)?;
    let minus_basis_count_usize = i32_to_usize("tdlda_minus_basis_count", minus_basis_count)?;

    let double_initial_l = checked_i32_mul("tdlda_initial_l_double", 2, initial_l)?;
    let plus_channel_count = checked_i32_mul(
        "tdlda_plus_channel_count",
        3,
        checked_i32_add("tdlda_plus_l_count", double_initial_l, 1)?,
    )?;
    let minus_l_count = checked_i32_add("tdlda_minus_l_count", double_initial_l, -1)?;
    let minus_channel_count =
        checked_i32_mul("tdlda_minus_channel_count", 3, minus_l_count.max(0))?;
    let matrix_size_i32 = checked_i32_add(
        "tdlda_matrix_size",
        checked_i32_mul(
            "tdlda_plus_matrix_size",
            plus_basis_count,
            plus_channel_count,
        )?,
        checked_i32_mul(
            "tdlda_minus_matrix_size",
            minus_basis_count,
            minus_channel_count,
        )?,
    )?;
    let matrix_size = i32_to_usize("tdlda_matrix_size", matrix_size_i32)?;
    if matrix_size > TDLDA_GETMAT_MAX_SIZE {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_matrix_size",
            value: matrix_size,
        });
    }

    let mut rows = Vec::with_capacity(matrix_size);
    for basis_index in 0..plus_basis_count {
        for dipole_branch in TdldaDipoleBranch::ALL {
            let (initial_kappa, final_kappa) = match dipole_branch {
                TdldaDipoleBranch::First => (
                    initial_l,
                    checked_i32_add("tdlda_final_kappa", initial_l, 1)?,
                ),
                TdldaDipoleBranch::Second => (
                    checked_i32_neg_add("tdlda_initial_kappa", initial_l, -1)?,
                    checked_i32_add("tdlda_final_kappa", initial_l, 1)?,
                ),
                TdldaDipoleBranch::Third => (
                    checked_i32_neg_add("tdlda_initial_kappa", initial_l, -1)?,
                    checked_i32_neg_add("tdlda_final_kappa", initial_l, -2)?,
                ),
            };
            if initial_kappa == 0 {
                continue;
            }
            let mut projector = checked_i32_mul("tdlda_projector", -2, basis_index)?;
            projector = checked_i32_add("tdlda_projector", projector, -1)?;
            if final_kappa > 0 {
                projector = checked_i32_add("tdlda_projector", projector, -1)?;
            }
            append_tdlda_channel_rows(&mut rows, input, initial_kappa, final_kappa, projector)?;
        }
    }

    for basis_index in 0..minus_basis_count {
        for dipole_branch in TdldaDipoleBranch::ALL {
            let (initial_kappa, final_kappa) = match dipole_branch {
                TdldaDipoleBranch::First => (
                    initial_l,
                    checked_i32_add("tdlda_final_kappa", initial_l, -1)?,
                ),
                TdldaDipoleBranch::Second => {
                    (initial_l, checked_i32_neg("tdlda_final_kappa", initial_l)?)
                }
                TdldaDipoleBranch::Third => (
                    checked_i32_neg_add("tdlda_initial_kappa", initial_l, -1)?,
                    checked_i32_neg("tdlda_final_kappa", initial_l)?,
                ),
            };
            if initial_kappa == 0 || final_kappa == 0 {
                continue;
            }
            let basis_offset =
                checked_i32_add("tdlda_projector_offset", basis_index, plus_basis_count)?;
            let mut projector = checked_i32_mul("tdlda_projector", -2, basis_offset)?;
            projector = checked_i32_add("tdlda_projector", projector, -1)?;
            if final_kappa > 0 {
                projector = checked_i32_add("tdlda_projector", projector, -1)?;
            }
            append_tdlda_channel_rows(&mut rows, input, initial_kappa, final_kappa, projector)?;
        }
    }

    if rows.len() != matrix_size {
        return Err(XsphError::SizeOutOfRange {
            name: "tdlda_channel_rows",
            value: rows.len(),
        });
    }

    Ok(XsphTdldaChannelBasis {
        plus_basis_count: plus_basis_count_usize,
        minus_basis_count: minus_basis_count_usize,
        matrix_size,
        rows,
    })
}

fn xsect_bcoef_selected_spin_index(spin: i32, spin_channels: usize) -> Result<usize, XsphError> {
    let selected = if spin == 1 {
        spin_channels
            .checked_sub(1)
            .ok_or(XsphError::SizeOutOfRange {
                name: "xsect_bcoef_spin_channels",
                value: spin_channels,
            })?
    } else {
        0
    };
    if selected > 1 {
        return Err(XsphError::SizeOutOfRange {
            name: "xsect_bcoef_spin_index",
            value: selected,
        });
    }
    Ok(selected)
}

fn bcoef_transition_index(
    slot_1based: usize,
    final_kappa: i32,
) -> Result<XsphBcoefTransitionIndex, XsphError> {
    let j_index = final_kappa
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "bcoef_kappa",
            value: final_kappa,
        })?;
    bcoef_transition_index_with_j(slot_1based, final_kappa, j_index)
}

fn bcoef_transition_index_with_j(
    slot_1based: usize,
    final_kappa: i32,
    j_index: i32,
) -> Result<XsphBcoefTransitionIndex, XsphError> {
    let orbital_l = if final_kappa <= 0 {
        final_kappa
            .checked_abs()
            .and_then(|value| value.checked_sub(1))
            .ok_or(XsphError::IntegerOutOfRange {
                name: "bcoef_orbital_l",
                value: final_kappa,
            })?
    } else {
        final_kappa
    };
    Ok(XsphBcoefTransitionIndex {
        slot_1based,
        final_kappa,
        j_index,
        orbital_l,
    })
}

fn inactive_bcoef_transition_index(slot_1based: usize) -> XsphBcoefTransitionIndex {
    XsphBcoefTransitionIndex {
        slot_1based,
        final_kappa: 0,
        j_index: 0,
        orbital_l: -1,
    }
}

fn store_bcoef_transition_index(
    final_kappas: &mut Array1<i32>,
    j_indices: &mut Array1<i32>,
    orbital_l: &mut Array1<i32>,
    index: XsphBcoefTransitionIndex,
) {
    let slot = index.slot_1based - 1;
    final_kappas[slot] = index.final_kappa;
    j_indices[slot] = index.j_index;
    orbital_l[slot] = index.orbital_l;
}

fn append_tdlda_channel_rows(
    rows: &mut Vec<XsphTdldaChannelBasisRow>,
    input: XsphTdldaChannelBasisInput<'_>,
    initial_kappa: i32,
    final_kappa: i32,
    default_projector: i32,
) -> Result<(), XsphError> {
    let initial_j2 = doubled_j_from_kappa("tdlda_initial_kappa", initial_kappa)?;
    let final_j2 = doubled_j_from_kappa("tdlda_final_kappa", final_kappa)?;
    let core_orbital_index_1based =
        if initial_kappa < 0 {
            usize_to_i32("tdlda_core_hole_index", input.core_hole_index_1based)?
        } else {
            let previous = input.core_hole_index_1based.checked_sub(1).ok_or(
                XsphError::InvalidOneBasedIndex {
                    name: "tdlda_core_hole_index",
                    index_1based: input.core_hole_index_1based,
                    active_len: usize::MAX,
                },
            )?;
            usize_to_i32("tdlda_core_hole_index", previous)?
        };

    for initial_m2 in (-initial_j2..=initial_j2).step_by(2) {
        let final_m2 = checked_i32_add("tdlda_final_m2", initial_m2, 2)?;
        if final_m2 < -final_j2 || final_m2 > final_j2 {
            continue;
        }

        rows.push(XsphTdldaChannelBasisRow {
            initial_j2,
            initial_m2,
            initial_kappa,
            final_j2,
            final_m2,
            final_kappa,
            core_orbital_index_1based,
            projector_orbital_selector: tdlda_projector_selector(
                input,
                final_kappa,
                default_projector,
            )?,
        });
    }
    Ok(())
}

fn tdlda_projector_selector(
    input: XsphTdldaChannelBasisInput<'_>,
    final_kappa: i32,
    default_projector: i32,
) -> Result<i32, XsphError> {
    if input.basis_selector != 0 {
        return Ok(default_projector);
    }

    let final_l = tdlda_l_from_kappa(final_kappa)?;
    let mut projector = default_projector;
    for (orbital_index, (&orbital_kappa, &occupation)) in input
        .orbital_kappas
        .iter()
        .zip(input.valence_occupations.iter())
        .enumerate()
    {
        if occupation > 0.0
            && tdlda_l_from_kappa_allow_zero(orbital_kappa)? == final_l
            && (final_kappa == orbital_kappa || final_kappa < 0)
        {
            projector = usize_to_i32("tdlda_projector_orbital", orbital_index + 1)?;
        }
    }
    Ok(projector)
}

/// Decode FEFF `TDLDA/getmat.f90` `nph(im)` projector selectors.
pub fn xsph_tdlda_decode_projector_selector(
    selector: i32,
) -> Result<XsphTdldaProjectorSelector, XsphError> {
    if selector > 0 {
        let selector_1based =
            usize::try_from(selector).map_err(|_| XsphError::IntegerOutOfRange {
                name: "tdlda_projector_selector",
                value: selector,
            })?;
        let orbital_index =
            selector_1based
                .checked_sub(1)
                .ok_or(XsphError::InvalidOneBasedIndex {
                    name: "tdlda_projector_selector",
                    index_1based: selector_1based,
                    active_len: usize::MAX,
                })?;
        return Ok(XsphTdldaProjectorSelector::OccupiedOrbital { orbital_index });
    }

    if selector < 0 {
        let raw = selector.checked_neg().ok_or(XsphError::IntegerOutOfRange {
            name: "tdlda_projector_selector",
            value: selector,
        })?;
        let raw = usize::try_from(raw).map_err(|_| XsphError::IntegerOutOfRange {
            name: "tdlda_projector_selector",
            value: selector,
        })?;
        let basis_index = raw.checked_sub(1).ok_or(XsphError::IntegerOutOfRange {
            name: "tdlda_projector_selector",
            value: selector,
        })? / 2;
        return Ok(XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index,
            positive_final_kappa: raw % 2 == 0,
        });
    }

    Err(XsphError::IntegerOutOfRange {
        name: "tdlda_projector_selector",
        value: selector,
    })
}

fn validate_tdlda_channel_basis_input(
    input: &XsphTdldaChannelBasisInput<'_>,
) -> Result<(), XsphError> {
    if input.core_hole_index_1based == 0 {
        return Err(XsphError::InvalidOneBasedIndex {
            name: "tdlda_core_hole_index",
            index_1based: input.core_hole_index_1based,
            active_len: usize::MAX,
        });
    }
    if input.initial_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "tdlda_initial_l",
            index: 0,
            value: input.initial_l,
        });
    }
    if input.orbital_kappas.len() != input.valence_occupations.len() {
        let (name, required, actual) =
            if input.orbital_kappas.len() > input.valence_occupations.len() {
                (
                    "tdlda_valence_occupations",
                    input.orbital_kappas.len(),
                    input.valence_occupations.len(),
                )
            } else {
                (
                    "tdlda_orbital_kappas",
                    input.valence_occupations.len(),
                    input.orbital_kappas.len(),
                )
            };
        return Err(XsphError::LengthTooShort {
            name,
            required,
            actual,
        });
    }
    for (index, &occupation) in input.valence_occupations.iter().enumerate() {
        validate_finite_real("tdlda_valence_occupation", occupation)?;
        if input.orbital_kappas[index] == i32::MIN {
            return Err(XsphError::IntegerOutOfRange {
                name: "tdlda_orbital_kappa",
                value: input.orbital_kappas[index],
            });
        }
    }
    Ok(())
}

fn tdlda_l_from_kappa(kappa: i32) -> Result<i32, XsphError> {
    if kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    tdlda_l_from_kappa_allow_zero(kappa)
}

fn tdlda_l_from_kappa_allow_zero(kappa: i32) -> Result<i32, XsphError> {
    if kappa > 0 {
        Ok(kappa)
    } else {
        kappa
            .checked_abs()
            .and_then(|value| value.checked_sub(1))
            .ok_or(XsphError::IntegerOutOfRange {
                name: "tdlda_kappa",
                value: kappa,
            })
    }
}

fn checked_i32_add(name: &'static str, left: i32, right: i32) -> Result<i32, XsphError> {
    left.checked_add(right)
        .ok_or(XsphError::IntegerOutOfRange { name, value: left })
}

fn checked_i32_neg(name: &'static str, value: i32) -> Result<i32, XsphError> {
    value
        .checked_neg()
        .ok_or(XsphError::IntegerOutOfRange { name, value })
}

fn checked_i32_neg_add(name: &'static str, value: i32, addend: i32) -> Result<i32, XsphError> {
    checked_i32_add(name, checked_i32_neg(name, value)?, addend)
}

fn checked_i32_mul(name: &'static str, left: i32, right: i32) -> Result<i32, XsphError> {
    left.checked_mul(right)
        .ok_or(XsphError::IntegerOutOfRange { name, value: left })
}

fn i32_to_usize(name: &'static str, value: i32) -> Result<usize, XsphError> {
    usize::try_from(value).map_err(|_| XsphError::IntegerOutOfRange { name, value })
}

/// Port of FEFF `COMMON/m_nrixs.f90:nrixs_init` transition-index generation.
///
/// FEFF stores these arrays in the `nrixs_inp` module and later reuses them in
/// XSPH, FMS, and GENFMTJAS. Rust modules run independently from handoff files,
/// so this helper rebuilds the deterministic `kind/lgind/ljind/lind` rows from
/// the core-hole kappa, NRIXS multipole limit, and FEFF angular capacity.
pub fn xsph_nrixs_transition_indices(
    input: XsphNrixsTransitionIndicesInput,
) -> Result<XsphNrixsTransitionIndices, XsphError> {
    if input.initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    let abs_kappa = input
        .initial_kappa
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: input.initial_kappa,
        })?;
    let abs_multipole = input
        .multipole
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole",
            value: input.multipole,
        })?;
    let final_lj_max =
        usize::try_from(abs_multipole).map_err(|_| XsphError::IntegerOutOfRange {
            name: "multipole",
            value: input.multipole,
        })?;
    let initial_j2 = abs_kappa
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: input.initial_kappa,
        })?;
    let final_j2_max = abs_multipole
        .checked_mul(2)
        .and_then(|value| value.checked_add(initial_j2))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole",
            value: input.multipole,
        })?;
    let final_state_capacity = nrixs_final_state_capacity(abs_kappa, abs_multipole, initial_j2)?;
    let doubled_lmax = usize_to_i32("max_angular_momentum", input.max_angular_momentum)?
        .checked_mul(2)
        .ok_or(XsphError::SizeOutOfRange {
            name: "max_angular_momentum",
            value: input.max_angular_momentum,
        })?;
    let initial_parity = if input.initial_kappa > 0 { -1 } else { 1 };

    let mut transitions = Vec::new();
    for lj in 0..=abs_multipole {
        let doubled_lj = lj.checked_mul(2).ok_or(XsphError::IntegerOutOfRange {
            name: "multipole",
            value: input.multipole,
        })?;
        let mut final_j2 = doubled_lj
            .checked_sub(initial_j2)
            .and_then(i32::checked_abs)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "jfin",
                value: doubled_lj,
            })?
            .max(1);
        let final_j2_max_for_lj = doubled_lj
            .checked_add(initial_j2)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "multipole",
                value: input.multipole,
            })?
            .min(final_j2_max);
        while final_j2 <= final_j2_max_for_lj {
            let final_parity = if (initial_j2 + final_j2 + doubled_lj).rem_euclid(4) == 0 {
                -initial_parity
            } else {
                initial_parity
            };
            let final_l2 = if final_parity > 0 {
                final_j2.checked_sub(1)
            } else {
                final_j2.checked_add(1)
            }
            .ok_or(XsphError::IntegerOutOfRange {
                name: "final_j2",
                value: final_j2,
            })?;

            if final_l2 <= doubled_lmax {
                let kappa = final_j2
                    .checked_add(1)
                    .ok_or(XsphError::IntegerOutOfRange {
                        name: "final_j2",
                        value: final_j2,
                    })?;
                transitions.push(XsphNrixsTransitionIndex {
                    final_state_kappa: -(kappa * final_parity) / 2,
                    decomposition_channel: final_l2 / 2,
                    total_angular_momentum_channel: lj,
                    orbital_angular_momentum: final_l2 / 2,
                });
            }
            final_j2 = final_j2
                .checked_add(2)
                .ok_or(XsphError::IntegerOutOfRange {
                    name: "final_j2",
                    value: final_j2,
                })?;
        }
    }

    if transitions.len() > final_state_capacity {
        return Err(XsphError::InsufficientGeneratedStates {
            required: transitions.len(),
            generated: final_state_capacity,
        });
    }

    Ok(XsphNrixsTransitionIndices {
        initial_j2,
        final_j2_max,
        final_lj_max,
        final_state_capacity,
        transitions,
    })
}

/// Port of FEFF `XSPH/mincalc.f90`.
///
/// The input arrays may contain extra capacity, mirroring FEFF's `kfinmax`;
/// `active_len` is FEFF's `indmax` and selects the active prefix. The returned
/// `calculations` table contains one row for each distinct `kind`, with the
/// maximum `ljind` observed for that kind folded into column 1.
pub fn xsph_minimize_calculations(
    kind: ArrayView1<'_, i32>,
    orbital_l: ArrayView1<'_, i32>,
    final_lj: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<XsphCalculationPlan, XsphError> {
    validate_active_len("kind", kind.len(), active_len)?;
    validate_active_len("orbital_l", orbital_l.len(), active_len)?;
    validate_active_len("final_lj", final_lj.len(), active_len)?;
    validate_final_lj(final_lj, active_len)?;

    let mut calculations = Array2::<i32>::zeros((active_len, 3));
    let mut index_map = Array1::<i32>::zeros(active_len);

    let mut calculation_count = 1_usize;
    calculations[(0, 0)] = kind[0];
    calculations[(0, 1)] = final_lj[0];
    calculations[(0, 2)] = orbital_l[0];
    index_map[0] = 1;
    let mut max_lj = final_lj[0];

    for index in 1..active_len {
        let current_kind = kind[index];
        let current_lj = final_lj[index];
        max_lj = max_lj.max(current_lj);

        let existing = (0..calculation_count)
            .find(|&row| current_kind == calculations[(row, 0)])
            .map(|row| row + 1);

        if let Some(one_based_row) = existing {
            index_map[index] =
                -i32::try_from(one_based_row).map_err(|_| XsphError::IndexMapOverflow {
                    index,
                    value: i32::MIN,
                })?;
            let row = one_based_row - 1;
            calculations[(row, 1)] = calculations[(row, 1)].max(current_lj);
        } else {
            let row = calculation_count;
            calculation_count += 1;
            index_map[index] =
                i32::try_from(calculation_count).map_err(|_| XsphError::IndexMapOverflow {
                    index,
                    value: i32::MIN,
                })?;
            calculations[(row, 0)] = current_kind;
            calculations[(row, 1)] = current_lj;
            calculations[(row, 2)] = orbital_l[index];
        }
    }

    let compact_calculations = Array2::from_shape_fn((calculation_count, 3), |(row, column)| {
        calculations[(row, column)]
    });

    Ok(XsphCalculationPlan {
        max_lj,
        calculations: compact_calculations,
        index_map,
    })
}

fn nrixs_final_state_capacity(
    abs_kappa: i32,
    abs_multipole: i32,
    initial_j2: i32,
) -> Result<usize, XsphError> {
    let kappa_term = abs_kappa
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            value: abs_kappa,
        })?;
    let multipole_term = abs_multipole
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole",
            value: abs_multipole,
        })?;
    let j_term = initial_j2
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: initial_j2,
        })?;
    let capacity = 2_i32
        .checked_mul(kappa_term)
        .and_then(|value| value.checked_mul(multipole_term))
        .and_then(|value| value.checked_mul(j_term))
        .ok_or(XsphError::IntegerOutOfRange {
            name: "kfinmax",
            value: i32::MAX,
        })?;
    usize::try_from(capacity).map_err(|_| XsphError::IntegerOutOfRange {
        name: "kfinmax",
        value: capacity,
    })
}

/// Port of FEFF `XSPH/ljneeded0.f90`.
///
/// Returns FEFF's integer flags for angular channels `0..=ljmax` that are used
/// by the one-based shared calculation `calculation_index`.
pub fn xsph_lj_needed_flags(
    ljmax: usize,
    final_lj: ArrayView1<'_, i32>,
    index_map: ArrayView1<'_, i32>,
    active_len: usize,
    calculation_index: i32,
) -> Result<Array1<i32>, XsphError> {
    validate_active_len("final_lj", final_lj.len(), active_len)?;
    validate_active_len("index_map", index_map.len(), active_len)?;
    validate_final_lj(final_lj, active_len)?;
    if calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex { calculation_index });
    }

    let output_len = ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax })?;
    let mut needed = Array1::<i32>::zeros(output_len);
    for index in 0..active_len {
        let mapped = index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: index_map[index],
            })?;
        if mapped == calculation_index {
            let angular_momentum = usize::try_from(final_lj[index]).map_err(|_| {
                XsphError::NegativeAngularMomentum {
                    name: "final_lj",
                    index,
                    value: final_lj[index],
                }
            })?;
            if angular_momentum > ljmax {
                return Err(XsphError::AngularMomentumOutOfRange {
                    angular_momentum,
                    ljmax,
                });
            }
            needed[angular_momentum] = 1;
        }
    }
    Ok(needed)
}

/// Port of FEFF `XSPH/besjnjas.f90`.
///
/// FEFF's JAS Bessel helper computes both `j_l` and `y_l` through
/// `l = ljmax + 1`, even though `qbesselget` only consumes the `j_l` values
/// through `ljmax`. This wrapper preserves that extra final order because the
/// full JAS cross-section driver also shares the helper.
pub fn xsph_jas_bessel_functions(
    argument: Complex,
    ljmax: usize,
) -> Result<SphericalBessel, XsphError> {
    if ljmax > QBESSEL_MAX_LJ {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: ljmax,
            ljmax: QBESSEL_MAX_LJ,
        });
    }
    let max_l = ljmax + 1;
    if argument.re >= 7.51 || argument.im.abs() >= 7.51 {
        validate_jas_bessel_argument(argument)?;
        return Ok(xsph_jas_bessel_asymptotic(argument, max_l));
    }
    spherical_bessel_j_y(argument, max_l).map_err(XsphError::from)
}

/// Port of FEFF `XSPH/qbesselget.f90`.
///
/// Builds a Fortran-order table `j_l(qtrans * r)` with rows over radii and
/// columns over `l = 0..=ljmax`. FEFF skips Bessel evaluation and stores zeros
/// when `qtrans * r >= 1e8`; this adapter keeps the same cutoff.
pub fn xsph_q_bessel_table(
    qtrans: Real,
    radii: ArrayView1<'_, Real>,
    ljmax: usize,
) -> Result<Array2<Real>, XsphError> {
    validate_finite_real("qtrans", qtrans)?;
    if ljmax > QBESSEL_MAX_LJ {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: ljmax,
            ljmax: QBESSEL_MAX_LJ,
        });
    }

    let column_count = ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax })?;
    let mut table = Array2::<Real>::zeros((radii.len(), column_count).f());
    for (radius_index, &radius) in radii.iter().enumerate() {
        validate_finite_real("radius", radius)?;
        let argument = qtrans * radius;
        validate_finite_real("qtrans * radius", argument)?;
        if argument < QBESSEL_ZERO_CUTOFF {
            let values = xsph_jas_bessel_functions(Complex::new(argument, 0.0), ljmax)?;
            for angular_momentum in 0..=ljmax {
                table[(radius_index, angular_momentum)] = values.j[angular_momentum].re;
            }
        }
    }
    Ok(table)
}

fn validate_jas_bessel_argument(argument: Complex) -> Result<(), XsphError> {
    if !argument.re.is_finite() || !argument.im.is_finite() {
        return Err(XsphError::Bessel(BesselError::NonFiniteArgument {
            real: argument.re,
            imaginary: argument.im,
        }));
    }
    if argument.re <= 0.0 {
        return Err(XsphError::Bessel(BesselError::NonPositiveRealArgument {
            real: argument.re,
        }));
    }
    Ok(())
}

fn xsph_jas_bessel_asymptotic(argument: Complex, max_l: usize) -> SphericalBessel {
    let (sjl, cjl) = xsph_jas_asymptotic_tables(argument, max_l);
    let sin_x = argument.sin();
    let cos_x = argument.cos();
    let mut j = Vec::with_capacity(max_l + 1);
    let mut y = Vec::with_capacity(max_l + 1);

    for angular_l in 0..=max_l {
        j.push(sin_x * sjl[angular_l] + cos_x * cjl[angular_l]);
        y.push(sin_x * cjl[angular_l] - cos_x * sjl[angular_l]);
    }

    SphericalBessel {
        j: Array1::from_vec(j),
        y: Array1::from_vec(y),
    }
}

fn xsph_jas_asymptotic_tables(argument: Complex, max_l: usize) -> (Vec<Complex>, Vec<Complex>) {
    let xi = Complex::new(1.0, 0.0) / argument;
    let powers = xsph_jas_complex_powers(xi, 11);
    let zero = Complex::new(0.0, 0.0);
    let c = |value: i32| value as f32 as Real;

    // XSPH/besjnjas.f90 writes these constants with decimal points, so FEFF
    // evaluates them as single-precision literals before widening.
    let s_seed = [
        powers[1],
        powers[2],
        powers[3] * c(3) - powers[1],
        powers[4] * c(15) - powers[2] * c(6),
        powers[5] * c(105) - powers[3] * c(45) + powers[1],
        powers[6] * c(945) - powers[4] * c(420) + powers[2] * c(15),
        powers[7] * c(10395) - powers[5] * c(4725) + powers[3] * c(210) - powers[1],
        powers[8] * c(135135) - powers[6] * c(62370) + powers[4] * c(3150) - powers[2] * c(28),
        powers[9] * c(2027025) - powers[7] * c(945945) + powers[5] * c(51975) - powers[3] * c(630)
            + powers[1],
        powers[10] * c(34459425) - powers[8] * c(16216200) + powers[6] * c(945945)
            - powers[4] * c(13860)
            + powers[2] * c(45),
        powers[11] * c(654729075) - powers[9] * c(310134825) + powers[7] * c(18918900)
            - powers[5] * c(315315)
            + powers[3] * c(1485)
            - powers[1],
    ];
    let c_seed = [
        zero,
        -powers[1],
        -powers[2] * c(3),
        -powers[3] * c(15) + powers[1],
        -powers[4] * c(105) + powers[2] * c(10),
        -powers[5] * c(945) + powers[3] * c(105) - powers[1],
        -powers[6] * c(10395) + powers[4] * c(1260) - powers[2] * c(21),
        -powers[7] * c(135135) + powers[5] * c(17325) - powers[3] * c(378) + powers[1],
        -powers[8] * c(2027025) + powers[6] * c(270270) - powers[4] * c(6930) + powers[2] * c(36),
        -powers[9] * c(34459425) + powers[7] * c(4729725) - powers[5] * c(135135)
            + powers[3] * c(990)
            - powers[1],
        -powers[10] * c(654729075) + powers[8] * c(91891800) - powers[6] * c(2837835)
            + powers[4] * c(25740)
            - powers[2] * c(55),
    ];

    let mut sjl = vec![zero; max_l + 1];
    let mut cjl = vec![zero; max_l + 1];
    let seed_count = (max_l + 1).min(s_seed.len());
    sjl[..seed_count].copy_from_slice(&s_seed[..seed_count]);
    cjl[..seed_count].copy_from_slice(&c_seed[..seed_count]);

    for angular_l in 11..=max_l {
        let coefficient = (2 * angular_l - 1) as f32 as Real;
        sjl[angular_l] = sjl[angular_l - 1] * coefficient * xi - sjl[angular_l - 2];
        cjl[angular_l] = cjl[angular_l - 1] * coefficient * xi - cjl[angular_l - 2];
    }

    (sjl, cjl)
}

fn xsph_jas_complex_powers(base: Complex, max_power: usize) -> Vec<Complex> {
    let mut powers = vec![Complex::new(1.0, 0.0); max_power + 1];
    for power in 1..=max_power {
        powers[power] = powers[power - 1] * base;
    }
    powers
}

/// Port of FEFF `XSPH/getoccnorm.f90`.
///
/// FEFF normalizes partially occupied initial states by dividing the default
/// occupation for `(Z, ihole)` by the reference occupation for `Z = 100`.
/// The inputs are the original one-based FEFF atomic number and hole selector.
/// Hole selectors whose FEFF denominator is zero are reported as errors instead
/// of returning a non-finite quotient.
pub fn xsph_occupation_normalization(
    atomic_number: usize,
    hole_index: usize,
) -> Result<Real, XsphError> {
    if !(1..=XSPH_OCC_NORM_ATOMIC_NUMBER_MAX).contains(&atomic_number) {
        return Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number,
            max_atomic_number: XSPH_OCC_NORM_ATOMIC_NUMBER_MAX,
        });
    }
    if !(1..=XSPH_OCC_NORM_HOLE_COUNT).contains(&hole_index) {
        return Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index,
            max_hole_index: XSPH_OCC_NORM_HOLE_COUNT,
        });
    }

    let numerator = xsph_occ_norm_numerator(atomic_number, hole_index).ok_or(
        XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number,
            max_atomic_number: XSPH_OCC_NORM_ATOMIC_NUMBER_MAX,
        },
    )?;
    let denominator =
        xsph_occ_norm_denominator(hole_index).ok_or(XsphError::InvalidOccupationNormHoleIndex {
            hole_index,
            max_hole_index: XSPH_OCC_NORM_HOLE_COUNT,
        })?;
    if denominator == 0 {
        return Err(XsphError::ZeroOccupationNormDenominator { hole_index });
    }

    Ok(Real::from(numerator) / Real::from(denominator))
}
