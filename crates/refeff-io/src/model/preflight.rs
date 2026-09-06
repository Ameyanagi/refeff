//! Side-effect-free validation of complete calculations, separate from tolerant parsing.

use std::collections::BTreeSet;

use super::*;

impl FeffDocument {
    /// Validate a complete calculation without generating files or numerical results.
    ///
    /// Parsing alone deliberately accepts partial documents for editors and individual
    /// module workflows. This boundary checks referential integrity and prepares the
    /// typed RDINP outputs in memory so malformed handoffs fail before execution.
    pub fn validate_calculation(&self, input: &FeffInput) -> Result<()> {
        let fail = |card: &str, message: String| {
            if let Some(line) = card_by_feff_name(input, card) {
                parse_error(line, message)
            } else {
                IoError::Parse {
                    path: input.source.clone(),
                    line: 1,
                    message,
                }
            }
        };
        if input.cards().next().is_none() {
            return Err(fail("TITLE", "input contains no calculation cards".into()));
        }
        if let Some(edge) = &self.edge {
            if refeff_core::edge_index(&edge.label).is_none() {
                return Err(fail(
                    "EDGE",
                    format!("unknown absorption edge {:?}", edge.label),
                ));
            }
        }
        if let Some(hole) = self.hole {
            refeff_core::core_hole_quantum_numbers(hole)
                .map_err(|error| fail("HOLE", error.to_string()))?;
        }
        let mut potentials = BTreeSet::new();
        for potential in &self.potentials {
            if potential.ipot < 0 || !potentials.insert(potential.ipot) {
                return Err(fail(
                    "POTENTIALS",
                    format!(
                        "potential index {} must be nonnegative and unique",
                        potential.ipot
                    ),
                ));
            }
            if !potential.z.is_some_and(|z| (1..=138).contains(&z)) {
                return Err(fail(
                    "POTENTIALS",
                    format!(
                        "potential {} requires an atomic number in 1..=138",
                        potential.ipot
                    ),
                ));
            }
        }
        let atom_rows: Vec<_> = input.section_rows("ATOMS").collect();
        for (index, atom) in self.atoms.iter().enumerate() {
            let line = atom_rows.get(index).copied();
            let error = |message: String| {
                line.map_or_else(
                    || fail("ATOMS", message.clone()),
                    |line| parse_error(line, message.clone()),
                )
            };
            if !potentials.contains(&atom.ipot) {
                return Err(error(format!(
                    "atom {} references undeclared potential {}",
                    index + 1,
                    atom.ipot
                )));
            }
            if ![atom.x, atom.y, atom.z]
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(error(format!(
                    "atom {} coordinates must be finite",
                    index + 1
                )));
            }
        }
        for (card, value) in [
            ("S02", self.s02.unwrap_or(1.0)),
            ("RGRID", self.rgrid),
            ("RMULT", self.r_multiplier),
            ("CRITERIA", self.critcw),
            ("CRITERIA", self.critpw),
            ("PCRITERIA", self.pcritk),
            ("PCRITERIA", self.pcrith),
            ("AFOLP", self.afolp),
        ] {
            if !value.is_finite() {
                return Err(fail(card, format!("{card} value must be finite")));
            }
        }
        let controls = self.control.unwrap_or([1; 6]);
        let needs_structure = controls.iter().any(|&value| value != 0)
            && !self.opcons
            && self.full_spectrum_input.m_full_spectrum == 0;
        if needs_structure && self.potentials.is_empty() {
            return Err(fail(
                "POTENTIALS",
                "calculation requires POTENTIALS or a structure source".into(),
            ));
        }
        if needs_structure
            && self.atoms.is_empty()
            && self.overlap_shells.is_empty()
            && !self.no_geom
        {
            return Err(fail(
                "ATOMS",
                "calculation requires ATOMS, OVERLAP, CIF, or lattice geometry".into(),
            ));
        }
        if needs_structure && !potentials.contains(&0) {
            return Err(fail(
                "POTENTIALS",
                "calculation requires absorber potential 0".into(),
            ));
        }
        // Writers perform format-specific shape, range, and finite-value validation.
        crate::rdinp::text_outputs(self)?;
        Ok(())
    }
}
