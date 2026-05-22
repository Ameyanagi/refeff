/// Parsed CIF data needed by FEFF's structure import path.
#[derive(Debug, Clone, PartialEq)]
pub struct CifDocument {
    /// Optional CIF data block name without the `data_` prefix.
    pub data_block: Option<String>,
    /// Unit-cell parameters.
    pub cell: CifCell,
    /// International Tables space-group number, when present.
    pub space_group_number: Option<i32>,
    /// Hermann-Mauguin space-group label, when present.
    pub space_group_hm: Option<String>,
    /// Symmetry operations in CIF text form.
    pub symmetry_operations: Vec<String>,
    /// Fractional atom-site rows.
    pub atom_sites: Vec<CifAtomSite>,
}

/// Unit-cell lengths and angles from CIF scalar fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CifCell {
    /// `a` unit-cell length in angstroms.
    pub a: f64,
    /// `b` unit-cell length in angstroms.
    pub b: f64,
    /// `c` unit-cell length in angstroms.
    pub c: f64,
    /// `alpha` unit-cell angle in degrees.
    pub alpha: f64,
    /// `beta` unit-cell angle in degrees.
    pub beta: f64,
    /// `gamma` unit-cell angle in degrees.
    pub gamma: f64,
}

/// One CIF atom site with fractional coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CifAtomSite {
    /// Element symbol from `_atom_site_type_symbol`, or a stripped label fallback.
    pub symbol: String,
    /// Original atom-site label, when present.
    pub label: Option<String>,
    /// Fractional coordinate along the crystallographic `a` axis.
    pub fract_x: f64,
    /// Fractional coordinate along the crystallographic `b` axis.
    pub fract_y: f64,
    /// Fractional coordinate along the crystallographic `c` axis.
    pub fract_z: f64,
}

/// CIF structure expanded through symmetry operations into FEFF coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CifExpandedStructure {
    /// FEFF lattice type such as `P`, `F`, `H`, or `CXY`.
    pub lattice_name: String,
    /// Compacted Hermann-Mauguin space-group label.
    pub space_group_hm: String,
    /// International Tables space-group number.
    pub space_group: i32,
    /// Lattice vectors in Angstrom Cartesian coordinates.
    pub lattice_vectors: [[f64; 3]; 3],
    /// One-based atom position index for the absorbing atom.
    pub absorber: usize,
    /// Atomic number of the absorbing CIF site.
    pub absorber_atomic_number: i32,
    /// FEFF label of the absorbing CIF site.
    pub absorber_label: String,
    /// FEFF `ppos` coordinates relative to the absorber and divided by `a`.
    pub positions: Vec<[f64; 3]>,
    /// Potential index for each expanded atom.
    pub potentials: Vec<i32>,
    /// Multiplicity of each inequivalent CIF atom site in the imported unit cell.
    pub site_multiplicities: Vec<usize>,
    /// Atomic number of each inequivalent CIF atom site.
    pub site_atomic_numbers: Vec<i32>,
    /// FEFF label of each inequivalent CIF atom site.
    pub site_labels: Vec<String>,
    /// FEFF labels: absorbing label first, followed by each inequivalent site label.
    pub labels: Vec<String>,
}

/// Real-space cluster generated from a CIF unit cell.
#[derive(Debug, Clone, PartialEq)]
pub struct CifCluster {
    /// Atoms sorted by distance from the absorber.
    pub atoms: Vec<CifClusterAtom>,
    /// Generated potential metadata for CIF files without a `POTENTIALS` card.
    pub potentials: Vec<CifPotential>,
}

/// One atom in a CIF-generated real-space cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CifClusterAtom {
    /// Cartesian x coordinate in Angstrom, relative to the absorber.
    pub x: f64,
    /// Cartesian y coordinate in Angstrom, relative to the absorber.
    pub y: f64,
    /// Cartesian z coordinate in Angstrom, relative to the absorber.
    pub z: f64,
    /// FEFF potential index, with `0` for the absorbing atom.
    pub potential: i32,
}

/// One generated potential from CIF inequivalent atom-site metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CifPotential {
    /// FEFF potential index.
    pub ipot: i32,
    /// Atomic number.
    pub atomic_number: i32,
    /// FEFF potential label.
    pub label: String,
    /// Unit-cell multiplicity for this potential.
    pub multiplicity: usize,
    /// Whether this row is the absorbing potential.
    pub absorber: bool,
}

/// FEFF CIF potential-equivalence selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CifEquivalence {
    /// Keep crystallographically inequivalent CIF sites as separate potentials.
    Crystallographic,
    /// Collapse CIF sites with the same atomic number into one potential.
    AtomicNumber,
    /// Use crystallographic sites until FEFF's potential limit, then collapse by atomic number.
    AutomaticLimit,
}
