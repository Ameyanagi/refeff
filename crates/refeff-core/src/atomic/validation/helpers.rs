use super::*;

pub(in crate::atomic) fn validate_positive_finite_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::NonPositiveScalar { field, value })
    }
}

pub(in crate::atomic) fn validate_positive_finite_nuclear_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialScalar { field, value })
    }
}

pub(in crate::atomic) fn validate_nuclear_count(
    field: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), AtomMathError> {
    if actual >= minimum {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialCount {
            field,
            minimum,
            actual,
        })
    }
}

pub(in crate::atomic) fn validate_differential_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNormalizationActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_integration_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > ATOM_INTDIR_HISTORY + 12 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracIntegrationActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_solver_setup_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracSolverSetupActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_solution_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracSolutionNormalizationActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

pub(in crate::atomic) fn validate_dirac_match_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_energy_disagreement_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_energy_disagreement_correction_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len % 2 == 1 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

pub(in crate::atomic) fn validate_dirac_match_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracMatchMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_node_count_index(
    field: &'static str,
    index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if index_1based > 0 && index_1based <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNodeCountIndex {
            field,
            index_1based,
            radial_count,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_energy_correction_matching_index(
    matching_index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange {
                matching_index_1based,
                radial_count,
            },
        )
    }
}

pub(in crate::atomic) fn validate_dirac_integration_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > ATOM_INTDIR_HISTORY && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

pub(in crate::atomic) fn validate_dirac_integration_max_index(
    max_index_1based: usize,
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if max_index_1based <= active_len
        && max_index_1based > matching_index_1based + ATOM_INTDIR_HISTORY
    {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange {
            max_index_1based,
            matching_index_1based,
            active_len,
        })
    }
}

pub(in crate::atomic) fn validate_coefficient_count(
    table: &'static str,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len > 0 {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: 1,
            actual_len,
        })
    }
}

pub(in crate::atomic) fn validate_coefficient_vector_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

pub(in crate::atomic) fn validate_coefficient_vector_capacity(
    table: &'static str,
    required_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len >= required_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: required_len,
            actual_len,
        })
    }
}

pub(in crate::atomic) fn validate_matrix_shape(
    table: &'static str,
    matrix: ArrayView2<'_, Real>,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), AtomMathError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows == expected_rows && columns == expected_columns {
        Ok(())
    } else {
        Err(AtomMathError::MatrixShape {
            table,
            expected_rows,
            expected_columns,
            rows,
            columns,
        })
    }
}

pub(in crate::atomic) fn validate_orbital_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

pub(in crate::atomic) fn validate_radial_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::RadialTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

pub(in crate::atomic) fn validate_occupation_tables(
    occupations: &[Real],
    kappas: &[i32],
) -> Result<(), AtomMathError> {
    if occupations.len() != kappas.len() {
        return Err(AtomMathError::OccupationKappaLengthMismatch {
            occupation_len: occupations.len(),
            kappa_len: kappas.len(),
        });
    }
    validate_finite_slice("occupation", occupations)
}

pub(in crate::atomic) fn validate_positive_occupation(
    context: &'static str,
    orbital: usize,
    occupations: &[Real],
) -> Result<Real, AtomMathError> {
    let occupation = occupations[orbital];
    if occupation > 0.0 {
        Ok(occupation)
    } else {
        Err(AtomMathError::NonPositiveOccupation {
            context,
            orbital_1based: orbital + 1,
            occupation,
        })
    }
}

pub(in crate::atomic) fn validate_orbital_index(
    index: usize,
    len: usize,
) -> Result<(), AtomMathError> {
    if index < len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalIndexOutOfRange { index, len })
    }
}

pub(in crate::atomic) fn validate_coefficient_table(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<(), AtomMathError> {
    let shape = coefficients.shape();
    let rows = shape[0];
    let columns = shape[1];
    let channels = shape[2];
    if rows == 0 || columns == 0 || rows != columns || channels == 0 {
        return Err(AtomMathError::CoefficientTableShape {
            rows,
            columns,
            channels,
        });
    }
    if left >= rows {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: left,
            len: rows,
        });
    }
    if right >= columns {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: right,
            len: columns,
        });
    }
    let channel = rank / 2;
    if channel >= channels {
        return Err(AtomMathError::CoefficientChannelOutOfRange {
            rank,
            channel,
            channels,
        });
    }
    for value in coefficients.iter().copied() {
        if !value.is_finite() {
            return Err(AtomMathError::NonFiniteScalar {
                field: "coefficient",
                value,
            });
        }
    }
    Ok(())
}

pub(in crate::atomic) fn validate_finite_slice(
    field: &'static str,
    values: &[Real],
) -> Result<(), AtomMathError> {
    for &value in values {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

pub(in crate::atomic) fn validate_finite_vector(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in values.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

pub(in crate::atomic) fn validate_finite_matrix(
    field: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in matrix.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

pub(in crate::atomic) fn validate_positive_finite_radii(
    values: ArrayView1<'_, Real>,
) -> Result<(), AtomMathError> {
    for &radius in values {
        validate_finite_scalar("radius", radius)?;
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
    }
    Ok(())
}

pub(in crate::atomic) fn validate_finite_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AtomMathError::NonFiniteScalar { field, value })
    }
}
