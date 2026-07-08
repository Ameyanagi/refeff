use super::*;
use crate::sortid_order_1based;

const PATH_DEGENERACY_EPSILON: Real = 1.0e-3;
const PATH_LENGTH_RANGE_EPSILON: Real = 1.0e-3;

/// Port of the FEFF `pathsd` hash-range degeneracy reduction.
///
/// `pathsd` canonicalizes every candidate with `timrep`, sorts the resulting
/// `dhash` values with `sortid`, merges equal hashes, and verifies that all
/// same-hash paths have matching potentials and standardized coordinates.
pub fn path_degeneracy_groups(
    input: PathDegeneracyGroupsInput<'_>,
) -> Result<Vec<PathDegeneracyGroup>, PathError> {
    let PathDegeneracyGroupsInput {
        atom_positions,
        atom_potentials,
        first_bounce_degeneracies,
        candidates,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
        force_no_symmetry,
    } = input;

    let mut canonical = Vec::with_capacity(candidates.len());
    let mut hashes = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        validate_nonempty_path(candidate.path_indices)?;
        let Some(&first_atom) = candidate.path_indices.first() else {
            return Err(PathError::EmptyPathCriteria);
        };
        let first_bounce = *first_bounce_degeneracies.get(first_atom).ok_or(
            PathError::PathDegeneracyFirstBounceOutOfRange {
                candidate: candidate_index,
                atom_index: first_atom,
                atoms: first_bounce_degeneracies.len(),
            },
        )?;
        let representation = path_canonical_representation(PathCanonicalRepresentationInput {
            atom_positions,
            path_indices: candidate.path_indices,
            atom_potentials,
            polarization,
            spin,
            electric_vector,
            incident_vector,
            symmetry_case_override,
            force_no_symmetry,
        })?;
        hashes.push(representation.degeneracy_hash);
        canonical.push(CanonicalCandidate {
            original_index: candidate_index,
            first_bounce,
            representation,
        });
    }

    let order = sortid_order_1based(&hashes)
        .map_err(|source| PathError::PathDegeneracyHashSort { source })?
        .into_iter()
        .map(|index| index - 1)
        .collect::<Vec<_>>();

    let mut groups = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let representative_index = order[start];
        let representative = &canonical[representative_index];
        let hash = representative.representation.degeneracy_hash;
        let mut end = start + 1;
        while end < order.len() && canonical[order[end]].representation.degeneracy_hash == hash {
            end += 1;
        }

        let mut degeneracy = 0_usize;
        for &member_index in &order[start..end] {
            let member = &canonical[member_index];
            validate_same_degeneracy_group(representative, member, atom_potentials)?;
            degeneracy = degeneracy.checked_add(member.first_bounce).ok_or(
                PathError::PathDegeneracyOverflow {
                    candidate: member.original_index,
                },
            )?;
        }

        groups.push(PathDegeneracyGroup {
            path_indices: representative.representation.path_indices.clone(),
            degeneracy,
            degeneracy_hash: hash,
            member_count: end - start,
            coordinates: representative.representation.coordinates.clone(),
            reversed: representative.representation.reversed,
            symmetry_case: representative.representation.symmetry_case,
        });
        start = end;
    }

    Ok(groups)
}

/// Port of the FEFF `pathsd` `critpw` retention pass.
///
/// FEFF initializes `xportx`/`ndegx` from the first processed unique path and
/// then retains paths whose degeneracy-weighted relative importance
/// `100 * ndeg * xport / (ndegx * xportx)` is at least `critpw`.
pub fn path_degeneracy_retention(
    input: PathDegeneracyRetentionInput<'_>,
) -> Result<PathDegeneracyRetention, PathError> {
    let PathDegeneracyRetentionInput {
        groups,
        port_importances,
        criterion_percent,
        initial_reference,
    } = input;
    if groups.len() != port_importances.len() {
        return Err(PathError::PathDegeneracyRetentionLengthMismatch {
            groups: groups.len(),
            port_importances: port_importances.len(),
        });
    }
    validate_retention_value("criterion_percent", usize::MAX, criterion_percent)?;
    validate_retention_reference(initial_reference)?;

    let mut decisions = Vec::with_capacity(groups.len());
    let mut retained_unique_count = 0_usize;
    let mut retained_total_degeneracy = 0_usize;
    let mut reference_group_index = None;
    let mut reference = initial_reference;

    for (index, (group, &port_importance)) in groups.iter().zip(port_importances).enumerate() {
        validate_retention_value("port_importance", index, port_importance)?;
        if group.degeneracy == 0 {
            return Err(PathError::NonPositivePathDegeneracy {
                index,
                degeneracy: group.degeneracy,
            });
        }
        let current_reference = if let Some(reference) = reference {
            reference
        } else {
            if port_importance == 0.0 {
                return Err(PathError::ZeroPathDegeneracyRetentionReference { index });
            }
            let initialized_reference = PathDegeneracyRetentionReference {
                port_importance,
                degeneracy: group.degeneracy,
            };
            reference_group_index = Some(index);
            reference = Some(initialized_reference);
            initialized_reference
        };

        let fraction_percent = 100.0 * group.degeneracy as Real * port_importance
            / (current_reference.degeneracy as Real * current_reference.port_importance);
        validate_retention_value("fraction_percent", index, fraction_percent)?;

        let retained = fraction_percent >= criterion_percent;
        if retained {
            retained_unique_count += 1;
            retained_total_degeneracy = retained_total_degeneracy
                .checked_add(group.degeneracy)
                .ok_or(PathError::PathDegeneracyRetentionOverflow { index })?;
        }
        decisions.push(PathDegeneracyRetentionDecision {
            group_index: index,
            port_importance,
            fraction_percent,
            retained,
        });
    }

    Ok(PathDegeneracyRetention {
        decisions,
        retained_unique_count,
        retained_total_degeneracy,
        reference_group_index,
        reference_port_importance: reference.map(|reference| reference.port_importance),
        reference_degeneracy: reference.map(|reference| reference.degeneracy),
        reference,
    })
}

/// Port of one FEFF `pathsd` equal-total-length range.
///
/// This composes `timrep`/hash grouping, `outcrt` output importance, sequential
/// `xcalcx` updates, and the `critpw` filter for one range from `paths.bin`.
pub fn path_degeneracy_range(
    input: PathDegeneracyRangeInput<'_>,
) -> Result<PathDegeneracyRange, PathError> {
    let PathDegeneracyRangeInput {
        grouping,
        fbeta,
        wave_numbers,
        mean_free_paths,
        start_energy_index,
        fbeta_critical,
        critical_wave_numbers,
        critical_mean_free_paths,
        current_normalization,
        criterion_percent,
        retention_reference,
    } = input;

    let groups = path_degeneracy_groups(grouping)?;
    let mut normalization = current_normalization;
    let mut importances = Vec::with_capacity(groups.len());
    for group in &groups {
        let importance = path_output_importance(PathOutputImportanceInput {
            atom_positions: grouping.atom_positions,
            path_indices: &group.path_indices,
            atom_potentials: grouping.atom_potentials,
            fbeta,
            wave_numbers,
            mean_free_paths,
            start_energy_index,
            fbeta_critical,
            critical_wave_numbers,
            critical_mean_free_paths,
            current_normalization: normalization,
        })?;
        normalization = importance.normalization;
        importances.push(importance);
    }

    let port_importances = importances
        .iter()
        .map(|importance| importance.port_importance)
        .collect::<Vec<_>>();
    let retention = path_degeneracy_retention(PathDegeneracyRetentionInput {
        groups: &groups,
        port_importances: &port_importances,
        criterion_percent,
        initial_reference: retention_reference,
    })?;

    Ok(PathDegeneracyRange {
        groups,
        importances,
        retention,
        normalization,
    })
}

/// Port of the outer FEFF `pathsd` candidate-reduction loop.
///
/// Records are consumed in caller-provided order, matching sequential
/// `paths.bin` reads. Contiguous records with `abs(r0 - rcurr) < 1.0e-3` are
/// reduced as one total-length range, while `xcalcx` and `xportx`/`ndegx` state
/// is carried into following ranges.
pub fn path_degeneracy_reduction(
    input: PathDegeneracyReductionInput<'_>,
) -> Result<PathDegeneracyReduction, PathError> {
    let PathDegeneracyReductionInput {
        atom_positions,
        atom_potentials,
        first_bounce_degeneracies,
        records,
        polarization,
        spin,
        electric_vector,
        incident_vector,
        symmetry_case_override,
        force_no_symmetry,
        fbeta,
        wave_numbers,
        mean_free_paths,
        start_energy_index,
        fbeta_critical,
        critical_wave_numbers,
        critical_mean_free_paths,
        current_normalization,
        criterion_percent,
        retention_reference,
    } = input;

    let mut ranges = Vec::new();
    let mut retained_unique_count = 0_usize;
    let mut retained_total_degeneracy = 0_usize;
    let mut normalization = current_normalization;
    let mut reference = retention_reference;
    let mut record_index = 0;

    while record_index < records.len() {
        let range_start = record_index;
        let representative_total_path_length = validated_record_length(records, range_start)?;
        record_index += 1;
        while record_index < records.len()
            && (validated_record_length(records, record_index)? - representative_total_path_length)
                .abs()
                < PATH_LENGTH_RANGE_EPSILON
        {
            record_index += 1;
        }

        let candidates = records[range_start..record_index]
            .iter()
            .map(|record| PathDegeneracyCandidate {
                path_indices: record.path_indices,
            })
            .collect::<Vec<_>>();
        let range = path_degeneracy_range(PathDegeneracyRangeInput {
            grouping: PathDegeneracyGroupsInput {
                atom_positions,
                atom_potentials,
                first_bounce_degeneracies,
                candidates: &candidates,
                polarization,
                spin,
                electric_vector,
                incident_vector,
                symmetry_case_override,
                force_no_symmetry,
            },
            fbeta,
            wave_numbers,
            mean_free_paths,
            start_energy_index,
            fbeta_critical,
            critical_wave_numbers,
            critical_mean_free_paths,
            current_normalization: normalization,
            criterion_percent,
            retention_reference: reference,
        })?;

        normalization = range.normalization;
        reference = range.retention.reference;
        retained_unique_count = retained_unique_count
            .checked_add(range.retention.retained_unique_count)
            .ok_or(PathError::PathDegeneracyReductionOverflow {
                range: ranges.len(),
            })?;
        retained_total_degeneracy = retained_total_degeneracy
            .checked_add(range.retention.retained_total_degeneracy)
            .ok_or(PathError::PathDegeneracyReductionOverflow {
                range: ranges.len(),
            })?;
        ranges.push(PathDegeneracyProcessedRange {
            representative_total_path_length,
            range,
        });
    }

    Ok(PathDegeneracyReduction {
        ranges,
        retained_unique_count,
        retained_total_degeneracy,
        normalization,
        retention_reference: reference,
    })
}

struct CanonicalCandidate {
    original_index: usize,
    first_bounce: usize,
    representation: PathCanonicalRepresentation,
}

fn validate_same_degeneracy_group(
    representative: &CanonicalCandidate,
    member: &CanonicalCandidate,
    atom_potentials: &[usize],
) -> Result<(), PathError> {
    let first = &representative.representation;
    let second = &member.representation;

    if first.path_indices.len() != second.path_indices.len() {
        return Err(hash_collision(representative, member));
    }
    for position in 0..first.path_indices.len() {
        let first_atom = first.path_indices[position];
        let second_atom = second.path_indices[position];
        let first_potential =
            *atom_potentials
                .get(first_atom)
                .ok_or(PathError::PathCriteriaAtomIndexOutOfRange {
                    position,
                    atom_index: first_atom,
                    atoms: atom_potentials.len(),
                })?;
        let second_potential = *atom_potentials.get(second_atom).ok_or(
            PathError::PathCriteriaAtomIndexOutOfRange {
                position,
                atom_index: second_atom,
                atoms: atom_potentials.len(),
            },
        )?;
        if first_potential != second_potential {
            return Err(hash_collision(representative, member));
        }
    }

    for row in 0..first.coordinates.nrows() {
        for column in 0..first.coordinates.ncols() {
            if (first.coordinates[(row, column)] - second.coordinates[(row, column)]).abs()
                > PATH_DEGENERACY_EPSILON
            {
                return Err(hash_collision(representative, member));
            }
        }
    }
    Ok(())
}

fn hash_collision(representative: &CanonicalCandidate, member: &CanonicalCandidate) -> PathError {
    PathError::PathDegeneracyHashCollision {
        first_candidate: representative.original_index,
        second_candidate: member.original_index,
    }
}

fn validate_retention_value(
    quantity: &'static str,
    index: usize,
    value: Real,
) -> Result<(), PathError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PathError::NonFinitePathDegeneracyRetentionValue {
            quantity,
            index,
            value,
        })
    }
}

fn validate_retention_reference(
    reference: Option<PathDegeneracyRetentionReference>,
) -> Result<(), PathError> {
    let Some(reference) = reference else {
        return Ok(());
    };
    if !reference.port_importance.is_finite() || reference.port_importance <= 0.0 {
        return Err(PathError::InvalidPathDegeneracyRetentionReference {
            quantity: "port_importance",
            value: reference.port_importance,
        });
    }
    if reference.degeneracy == 0 {
        return Err(
            PathError::InvalidPathDegeneracyRetentionReferenceDegeneracy {
                degeneracy: reference.degeneracy,
            },
        );
    }
    Ok(())
}

fn validated_record_length(
    records: &[PathDegeneracyRecord<'_>],
    record: usize,
) -> Result<Real, PathError> {
    let value = records[record].total_path_length;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PathError::NonFinitePathDegeneracyRecordLength { record, value })
    }
}
