use ndarray::{Array1, Array2, Array3, Array4};

use crate::error::Result;

use super::validate::validate_dym;

/// Coordinate section from a FEFF `.dym` file.
#[derive(Debug, Clone, PartialEq)]
pub enum DymCoordinates {
    /// Cartesian atom positions in atomic units.
    Cartesian(Array2<f64>),
    /// Reduced atom positions plus cell vectors in atomic units.
    Reduced {
        /// Fractional/reduced positions, one atom per row.
        reduced: Array2<f64>,
        /// Cell vectors, one vector per row.
        cell: Array2<f64>,
    },
}

/// One FEFF type-2 unique-atom group from a `.dym` file.
#[derive(Debug, Clone, PartialEq)]
pub struct DymUniqueAtom {
    /// FEFF `utype` value for this unique-atom group.
    pub atom_type: i32,
    /// Zero-based FEFF `centeratomindex` entries.
    pub center_atom_indices: Array1<usize>,
    /// The auxiliary scalar read between `centeratomindex` and coordinates.
    pub weights: Array1<f64>,
    /// FEFF `u_xyz` rows for each degenerate atom in the group.
    pub coordinates: Array2<f64>,
}

/// FEFF type-2 unique-atom metadata stored after the force constants.
#[derive(Debug, Clone, PartialEq)]
pub struct DymType2Metadata {
    /// FEFF `natom` value from the type-2 metadata block.
    pub cell_atom_count: usize,
    /// Unique-atom groups in file order.
    pub unique_atoms: Vec<DymUniqueAtom>,
}

impl DymCoordinates {
    /// Return Cartesian atom positions in atomic units.
    #[must_use]
    pub fn cartesian_positions(&self) -> Array2<f64> {
        match self {
            Self::Cartesian(positions) => positions.clone(),
            Self::Reduced { reduced, cell } => reduced.dot(cell),
        }
    }
}

/// Parsed FEFF `.dym` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct DymData {
    /// FEFF dynamical-matrix type flag.
    pub dym_type: i32,
    /// Atomic numbers, one per atom.
    pub atomic_numbers: Array1<i32>,
    /// Atomic masses, one per atom.
    pub atomic_masses: Array1<f64>,
    /// Coordinate block.
    pub coordinates: DymCoordinates,
    /// Force-constant blocks indexed as `[iatom, jatom, row, column]`.
    pub force_constants: Array4<f64>,
    /// Optional unique-atom metadata for type 2 DMDW self-energy files.
    pub type2_metadata: Option<DymType2Metadata>,
    /// Optional dipole derivatives for type 3 DMDW IR files.
    ///
    /// The shape is `(atom, displacement_component, dipole_component)`, matching
    /// FEFF's nested `iAt`, `ip`, `jq` read order while keeping atom rows first.
    pub dipole_derivatives: Option<Array3<f64>>,
}

impl DymData {
    /// Number of atoms in the `.dym` file.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atomic_numbers.len()
    }

    /// Build FEFF's mass-weighted dynamical matrix in coordinate-major layout.
    ///
    /// FEFF stores coordinate blocks as `dm_block(iAt,jAt,ip,jq)` and then
    /// maps them to `dm((iAt-1)+nAt*(ip-1),(jAt-1)+nAt*(jq-1))`, divided by
    /// `sqrt(am(iAt)*am(jAt))`.
    pub fn mass_weighted_dynamical_matrix(&self) -> Result<Array2<f64>> {
        validate_dym(self)?;

        let atom_count = self.atom_count();
        let mut matrix = Array2::zeros((3 * atom_count, 3 * atom_count));
        for i_atom in 0..atom_count {
            for j_atom in 0..atom_count {
                let scale = (self.atomic_masses[i_atom] * self.atomic_masses[j_atom]).sqrt();
                for row in 0..3 {
                    for column in 0..3 {
                        matrix[[i_atom + atom_count * row, j_atom + atom_count * column]] =
                            self.force_constants[[i_atom, j_atom, row, column]] / scale;
                    }
                }
            }
        }
        Ok(matrix)
    }
}
