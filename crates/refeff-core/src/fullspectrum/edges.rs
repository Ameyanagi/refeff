//! FULLSPECTRUM occupied-edge selection.

use super::constants::{
    FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE, FEFF_FULLSPECTRUM_EDGE_LABELS,
    FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT,
};
use super::types::*;
use super::validation::{validate_finite_value, validate_matching_len};

/// Port of `FULLSPECTRUM/gtedgs.f90`: select occupied component edges.
///
/// FEFF scans the 40 relativistic orbital slots from `getorb`, emits an edge
/// when the occupation is at least one, and sets the convolution flag when the
/// tabulated onset is at or below 50 eV (`1.8374655` Hartree). Fractionally
/// occupied states below one electron are skipped, matching FEFF's warning-only
/// branch.
pub fn full_spectrum_edges_from_occupations(
    input: FullSpectrumEdgeSelectionInput<'_>,
) -> Result<FullSpectrumEdgeSelection, FullSpectrumError> {
    validate_matching_len(
        "occupations",
        input.occupations.len(),
        FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT,
    )?;
    validate_matching_len(
        "edge_onsets_hartree",
        input.edge_onsets_hartree.len(),
        FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT,
    )?;

    let mut edges = Vec::new();
    for (row, label) in FEFF_FULLSPECTRUM_EDGE_LABELS.iter().copied().enumerate() {
        let occupation = input.occupations[row];
        validate_finite_value("occupations", row, occupation)?;
        if occupation < 0.0 {
            return Err(FullSpectrumError::NegativeValue {
                field: "occupations",
                row,
                value: occupation,
            });
        }

        let onset = input.edge_onsets_hartree[row];
        validate_finite_value("edge_onsets_hartree", row, onset)?;
        if occupation >= 1.0 {
            edges.push(FullSpectrumSelectedEdge {
                hole_index: row + 1,
                label,
                occupation,
                convolve: onset <= FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE,
            });
        }
    }

    Ok(FullSpectrumEdgeSelection { edges })
}
