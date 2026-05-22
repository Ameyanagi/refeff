//! Minimal Crystallographic Information File reader for FEFF inputs.
//!
//! FEFF's RDINP stage accepts `CIF` cards and expands cell, symmetry, and
//! atom-site data before writing downstream handoff files.  This module starts
//! with the CIF subset used by the FEFF10 reference examples: scalar cell
//! parameters, space-group metadata, symmetry-operation loops, and fractional
//! atom-site loops.

mod common;
mod expand;
mod parse;
#[cfg(test)]
mod tests;
mod types;

pub use expand::{
    expand_cif_cluster, expand_cif_cluster_with_equivalence, expand_cif_structure,
    expand_cif_structure_with_equivalence,
};
pub use parse::{parse_cif, read_cif};
pub use types::{
    CifAtomSite, CifCell, CifCluster, CifClusterAtom, CifDocument, CifEquivalence,
    CifExpandedStructure, CifPotential,
};
