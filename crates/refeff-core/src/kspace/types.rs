//! Public reciprocal-space data types.

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use refeff_linalg::LinalgError;
use thiserror::Error;

use crate::angular::AngularError;
use crate::{Complex, Real, RealMat, Vector3};

/// FEFF Bravais lattice selector from `BAND/ibravais.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BravaisLattice {
    /// Triclinic primitive.
    TriclinicPrimitive = 1,
    /// Monoclinic primitive.
    MonoclinicPrimitive = 2,
    /// Monoclinic base-centered.
    MonoclinicBaseCentered = 3,
    /// Orthorhombic primitive.
    OrthorhombicPrimitive = 4,
    /// Orthorhombic base-centered.
    OrthorhombicBaseCentered = 5,
    /// Orthorhombic body-centered.
    OrthorhombicBodyCentered = 6,
    /// Orthorhombic face-centered.
    OrthorhombicFaceCentered = 7,
    /// Tetragonal primitive.
    TetragonalPrimitive = 8,
    /// Tetragonal body-centered.
    TetragonalBodyCentered = 9,
    /// Trigonal primitive.
    TrigonalPrimitive = 10,
    /// Hexagonal primitive.
    HexagonalPrimitive = 11,
    /// Cubic primitive.
    CubicPrimitive = 12,
    /// Cubic face-centered.
    CubicFaceCentered = 13,
    /// Cubic body-centered.
    CubicBodyCentered = 14,
}

impl BravaisLattice {
    /// Return FEFF's integer Bravais index.
    #[must_use]
    pub fn index(self) -> i32 {
        self as i32
    }

    /// Convert FEFF's integer Bravais index into a typed selector.
    pub fn from_index(index: i32) -> Result<Self, KSpaceError> {
        match index {
            1 => Ok(Self::TriclinicPrimitive),
            2 => Ok(Self::MonoclinicPrimitive),
            3 => Ok(Self::MonoclinicBaseCentered),
            4 => Ok(Self::OrthorhombicPrimitive),
            5 => Ok(Self::OrthorhombicBaseCentered),
            6 => Ok(Self::OrthorhombicBodyCentered),
            7 => Ok(Self::OrthorhombicFaceCentered),
            8 => Ok(Self::TetragonalPrimitive),
            9 => Ok(Self::TetragonalBodyCentered),
            10 => Ok(Self::TrigonalPrimitive),
            11 => Ok(Self::HexagonalPrimitive),
            12 => Ok(Self::CubicPrimitive),
            13 => Ok(Self::CubicFaceCentered),
            14 => Ok(Self::CubicBodyCentered),
            _ => Err(KSpaceError::InvalidBravaisIndex { index }),
        }
    }
}

/// High-symmetry path segments returned by FEFF `define_kpath`.
#[derive(Debug, Clone, PartialEq)]
pub struct KPath {
    /// Bravais lattice used to select the segment table.
    pub bravais: BravaisLattice,
    /// User-provided FEFF `KPATH` value.
    pub requested_kpath: i32,
    /// FEFF-adjusted `KPATH` value after defaulting rules.
    pub effective_kpath: i32,
    /// FEFF eight-character segment labels.
    pub labels: Vec<String>,
    /// Segment start vectors as `(segment, xyz)`.
    pub starts: RealMat,
    /// Segment end vectors as `(segment, xyz)`.
    pub ends: RealMat,
}

impl KPath {
    /// Number of active FEFF K-path segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the path contains no active segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Return one segment's start vector.
    #[must_use]
    pub fn start(&self, index: usize) -> Option<Vector3> {
        if index < self.len() {
            Some([
                self.starts[(index, 0)],
                self.starts[(index, 1)],
                self.starts[(index, 2)],
            ])
        } else {
            None
        }
    }

    /// Return one segment's end vector.
    #[must_use]
    pub fn end(&self, index: usize) -> Option<Vector3> {
        if index < self.len() {
            Some([
                self.ends[(index, 0)],
                self.ends[(index, 1)],
                self.ends[(index, 2)],
            ])
        } else {
            None
        }
    }
}

/// FEFF `BAND/bandtot.f90` sampled K-path mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKPathMesh {
    /// FEFF eight-character segment labels.
    pub labels: Vec<String>,
    /// Number of sampled points assigned to each segment.
    pub segment_point_counts: Vec<usize>,
    /// FEFF one-based cumulative segment end indices, `INDKDIR`.
    pub segment_end_indices: Vec<usize>,
    /// Cartesian k-points as `(point, xyz)`, equivalent to FEFF `bk(:,ik)`.
    pub k_points: RealMat,
    /// Cumulative scalar path coordinate, equivalent to FEFF `KP(ik)`.
    pub path_distances: Array1<Real>,
}

impl BandKPathMesh {
    /// Number of sampled k-points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.k_points.nrows()
    }
}

/// Explicit inputs for the FEFF `KSPACE/strharpol.f90` harmonic-polynomial kernel.
pub struct KSpaceHarmonicPolynomialsInput<'a> {
    /// Cartesian vector passed as FEFF `X`, `Y`, and `Z`.
    pub vector: Vector3,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX`.
    pub lmax: usize,
    /// FEFF `QJLTAB(JJ,LL)` real-harmonic normalization table.
    pub qjltab: ArrayView2<'a, Real>,
}

/// Explicit inputs for FEFF `KSPACE/straa.f90` reciprocal pair phases.
pub struct KSpaceReciprocalPairPhasesInput<'a> {
    /// Direct basis vectors as `(basis vector, xyz)`, FEFF `BRX/BRY/BRZ`.
    pub direct_basis: [Vector3; 3],
    /// Reciprocal basis vectors as `(basis vector, xyz)`, FEFF `BGX/BGY/BGZ`.
    pub reciprocal_basis: [Vector3; 3],
    /// Reciprocal lattice integer triplets `(N, xyz)`, FEFF `G1/G2/G3`.
    pub reciprocal_indices: ArrayView2<'a, i32>,
    /// Q-pair offsets `(IQQP, xyz)`, FEFF `QQPX/QQPY/QQPZ`.
    pub q_pair_offsets: ArrayView2<'a, Real>,
    /// Ewald splitting parameter, FEFF `ETA`.
    pub eta: Real,
}

/// Explicit inputs for FEFF `KSPACE/straa.f90` base direct terms.
pub struct KSpaceDirectLatticeTermsInput<'a> {
    /// Direct basis vectors as `(basis vector, xyz)`, FEFF `BRX/BRY/BRZ`.
    pub direct_basis: [Vector3; 3],
    /// Direct lattice integer triplets `(I, xyz)`, FEFF `R1/R2/R3`.
    pub direct_indices: ArrayView2<'a, i32>,
    /// Per-pair direct-list row references `(S, IQQP)`, zero-based FEFF `INDR`.
    pub direct_index_by_pair: ArrayView2<'a, usize>,
    /// Number of direct terms to use for each pair, FEFF `SMAX`.
    pub direct_counts: &'a [usize],
    /// Q-pair offsets `(IQQP, xyz)`, FEFF `QQPX/QQPY/QQPZ`.
    pub q_pair_offsets: ArrayView2<'a, Real>,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX`.
    pub lmax: usize,
    /// Maximum FEFF continued-fraction order, FEFF `J22MAX`.
    pub j22max: usize,
    /// FEFF `QJLTAB(JJ,LL)` real-harmonic normalization table.
    pub qjltab: ArrayView2<'a, Real>,
    /// Ewald splitting parameter, FEFF `ETA`.
    pub eta: Real,
}

/// Explicit inputs for FEFF `KSPACE/strcc.f90` energy-dependent tables.
pub struct KSpaceEnergyDependentTermsInput<'a> {
    /// Reduced complex energy, FEFF `EDU = ERYD / (2*pi/ALAT)^2`.
    pub energy: Complex,
    /// Ewald splitting parameter, FEFF `ETA`.
    pub eta: Real,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX`.
    pub lmax: usize,
    /// Base direct terms `(MMLL, S, IQQP)`, FEFF `QQMLRS` before `IILERS`.
    pub base_direct_terms: ArrayView3<'a, Complex>,
    /// Continued-fraction radial terms `(J22, LL, S, IQQP)`, FEFF `GGJLRS`.
    pub radial_terms: ArrayView4<'a, Real>,
    /// Number of direct terms to use for each pair, FEFF `SMAX`.
    pub direct_counts: &'a [usize],
}

/// Explicit inputs for FEFF `change_eta` retry orchestration around `STRCC`.
pub struct KSpaceEwaldEnergyTablesInput<'a> {
    /// Reduced complex energy, FEFF `EDU = ERYD / (2*pi/ALAT)^2`.
    pub energy: Complex,
    /// Initial Ewald splitting parameter, FEFF `ETA`.
    pub initial_eta: Real,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX`.
    pub lmax: usize,
    /// Maximum FEFF continued-fraction order, FEFF `J22MAX`.
    pub j22max: usize,
    /// Direct basis vectors as `(basis vector, xyz)`, FEFF `BRX/BRY/BRZ`.
    pub direct_basis: [Vector3; 3],
    /// Reciprocal basis vectors as `(basis vector, xyz)`, FEFF `BGX/BGY/BGZ`.
    pub reciprocal_basis: [Vector3; 3],
    /// Reciprocal lattice integer triplets `(N, xyz)`, FEFF `G1/G2/G3`.
    pub reciprocal_indices: ArrayView2<'a, i32>,
    /// Direct lattice integer triplets `(I, xyz)`, FEFF `R1/R2/R3`.
    pub direct_indices: ArrayView2<'a, i32>,
    /// Per-pair direct-list row references `(S, IQQP)`, zero-based FEFF `INDR`.
    pub direct_index_by_pair: ArrayView2<'a, usize>,
    /// Number of direct terms to use for each pair, FEFF `SMAX`.
    pub direct_counts: &'a [usize],
    /// Q-pair offsets `(IQQP, xyz)`, FEFF `QQPX/QQPY/QQPZ`.
    pub q_pair_offsets: ArrayView2<'a, Real>,
    /// FEFF `QJLTAB(JJ,LL)` real-harmonic normalization table.
    pub qjltab: ArrayView2<'a, Real>,
}

/// Explicit inputs for the FEFF `KSPACE/strbbdd.f90` lattice-sum kernel.
///
/// This ports the reciprocal and direct accumulation loops after the expensive
/// setup work has populated the lattice lists, pair phases, direct terms, and
/// harmonic-polynomial normalization table. Integer lattice indices are
/// zero-based rows in Rust; `direct_index_by_pair` contains zero-based
/// references into `direct_indices`.
#[derive(Debug, Clone, Copy)]
pub struct KSpaceStrbbddInput<'a> {
    /// FEFF `KX`, `KY`, and `KZ`.
    pub k: Vector3,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX`.
    pub lmax: usize,
    /// Ewald splitting parameter, FEFF `ETA`.
    pub eta: Real,
    /// Complex energy denominator term, FEFF `EDU`.
    pub energy: Complex,
    /// Reciprocal cutoff applied to `real(DENOM)`, FEFF `GMAXSQ`.
    pub gmax_squared: Real,
    /// Reciprocal basis vectors as `(basis vector, xyz)`, FEFF `BGX/BGY/BGZ`.
    pub reciprocal_basis: [Vector3; 3],
    /// Reciprocal lattice integer triplets `(N, xyz)`, FEFF `G1/G2/G3`.
    pub reciprocal_indices: ArrayView2<'a, i32>,
    /// Precomputed pair phases for reciprocal vectors `(N, IQQP)`, FEFF `EXPGNQ`.
    pub reciprocal_pair_phases: ArrayView2<'a, Complex>,
    /// FEFF `D1TERM3(LL)` weights, indexed by zero-based angular momentum `LL`.
    pub d1term3: ArrayView1<'a, Complex>,
    /// FEFF `QJLTAB(JJ,LL)` real-harmonic normalization table.
    pub qjltab: ArrayView2<'a, Real>,
    /// Q-pair offsets `(IQQP, xyz)`, FEFF `QQPX/QQPY/QQPZ`.
    pub q_pair_offsets: ArrayView2<'a, Real>,
    /// Direct basis vectors as `(basis vector, xyz)`, FEFF `BRX/BRY/BRZ`.
    pub direct_basis: [Vector3; 3],
    /// Direct lattice integer triplets `(I, xyz)`, FEFF `R1/R2/R3`.
    pub direct_indices: ArrayView2<'a, i32>,
    /// Per-pair direct-list row references `(S, IQQP)`, zero-based FEFF `INDR`.
    pub direct_index_by_pair: ArrayView2<'a, usize>,
    /// Number of direct terms to use for each pair, FEFF `SMAX`.
    pub direct_counts: &'a [usize],
    /// Direct lattice pre-summed terms `(MMLL, S, IQQP)`, FEFF `QQMLRS`.
    pub direct_terms: ArrayView3<'a, Complex>,
    /// FEFF `D300` correction added to `(MMLL=1, IQQP=1)`.
    pub d300: Complex,
}

/// Result of FEFF `STRBBDD -> STRSET` structure-constant assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceStrsetMatrices {
    /// FEFF `DLLMMKE(LM,IQQP)` lattice-sum output from `STRBBDD`.
    pub dllmmke: Array2<Complex>,
    /// FEFF `TAUKINV` matrix produced by `STRSET`, in SPRKKR basis.
    pub taukinv: Array2<Complex>,
}

/// FEFF `STRGAUNT` and `STRAA` angular tables used by `STRBBDD -> STRSET`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceAngularTables {
    /// Maximum scattering angular momentum, FEFF `LMAX = NL - 1`.
    pub angular_lmax: usize,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX = 2 * LMAX`.
    pub harmonic_lmax: usize,
    /// FEFF `NLM = (LMAX + 1)**2`, the non-relativistic angular state count.
    pub angular_state_count: usize,
    /// FEFF `QJLTAB(JJ,LL)` real-harmonic normalization table.
    pub qjltab: Array2<Real>,
    /// FEFF `NGNT` triangular `(LM1,LM2)` row counts.
    pub gaunt_counts: Vec<usize>,
    /// Flattened zero-based FEFF `IGNT` indices into `DLLMMKE`.
    pub gaunt_indices: Vec<usize>,
    /// Flattened FEFF `GNT` coefficients aligned with `gaunt_indices`.
    pub gaunt_values: Vec<Real>,
    /// FEFF `CIPWL(LM)=i**L` phase table for `L=0..2*LMAX`.
    pub cipwl: Array1<Complex>,
}

/// Inputs for composing FEFF `STRBBDD` with non-relativistic `STRSET`.
#[derive(Debug, Clone, Copy)]
pub struct KSpaceStrsetNonRelFromLatticeSumInput<'a> {
    /// Full lattice-sum input consumed by FEFF `STRBBDD`.
    pub lattice_sum: KSpaceStrbbddInput<'a>,
    /// FEFF `NLM = NL**2`, the number of non-relativistic states per site.
    pub angular_state_count: usize,
    /// Representative and equivalent q-pair site indices.
    pub q_pair_sites: ArrayView3<'a, usize>,
    /// Number of equivalent site pairs to use for each q-pair, FEFF `NIJQ`.
    pub q_pair_counts: &'a [usize],
    /// Per-site matrix offsets, FEFF `IND0Q`, converted to zero-based offsets.
    pub site_offsets: &'a [usize],
    /// Per-site state counts, FEFF `NKMQ`.
    pub site_state_counts: &'a [usize],
    /// FEFF `NGNT` triangular `(LM1,LM2)` row counts.
    pub gaunt_counts: &'a [usize],
    /// Flattened zero-based FEFF `IGNT` indices into `DLLMMKE`.
    pub gaunt_indices: &'a [usize],
    /// Flattened FEFF `GNT` coefficients aligned with `gaunt_indices`.
    pub gaunt_values: &'a [Real],
    /// FEFF `CIPWL(LM)` phase table.
    pub cipwl: ArrayView1<'a, Complex>,
    /// Complex wave number `P`.
    pub wave_number: Complex,
}

/// Inputs for composing FEFF `STRBBDD` with relativistic `STRSET`.
#[derive(Debug, Clone, Copy)]
pub struct KSpaceStrsetRelFromLatticeSumInput<'a> {
    /// Full lattice-sum input consumed by FEFF `STRBBDD`.
    pub lattice_sum: KSpaceStrbbddInput<'a>,
    /// FEFF `NLM = NL**2`, the non-relativistic angular state count.
    pub angular_state_count: usize,
    /// Representative and equivalent q-pair site indices.
    pub q_pair_sites: ArrayView3<'a, usize>,
    /// Number of equivalent site pairs to use for each q-pair, FEFF `NIJQ`.
    pub q_pair_counts: &'a [usize],
    /// Per-site matrix offsets, FEFF `IND0Q`, converted to zero-based offsets.
    pub site_offsets: &'a [usize],
    /// Per-site relativistic state counts, FEFF `NKMQ`.
    pub site_state_counts: &'a [usize],
    /// FEFF `NGNT` triangular `(LM1,LM2)` row counts.
    pub gaunt_counts: &'a [usize],
    /// Flattened zero-based FEFF `IGNT` indices into `DLLMMKE`.
    pub gaunt_indices: &'a [usize],
    /// Flattened FEFF `GNT` coefficients aligned with `gaunt_indices`.
    pub gaunt_values: &'a [Real],
    /// FEFF `CIPWL(LM)` phase table.
    pub cipwl: ArrayView1<'a, Complex>,
    /// Number of non-rel components for each spin/state, FEFF `NRREL(IS,IKM)`.
    pub rel_component_counts: ArrayView2<'a, usize>,
    /// Non-rel angular indices for each spin/state component, zero-based FEFF `IRREL`.
    pub rel_component_indices: ArrayView3<'a, usize>,
    /// Relativistic transform coefficients, FEFF `SRREL`.
    pub rel_component_coefficients: ArrayView3<'a, Complex>,
    /// Complex wave number `P`.
    pub wave_number: Complex,
}

/// Explicit inputs for FEFF `KSPACE/strset.f90` with `IREL < 2`.
///
/// Site indices are zero-based and `q_pair_sites` is shaped as
/// `(q_pair, equivalent_pair, [row_site, column_site])`, replacing FEFF's
/// packed `IJQ = 100*IQ + JQ` representation.
pub struct KSpaceStrsetNonRelInput<'a> {
    /// FEFF `NLM = NL**2`, the number of non-relativistic states per site.
    pub angular_state_count: usize,
    /// FEFF `DLLMMKE(LM3,IQQP)` from `STRBBDD`.
    pub dllmmke: ArrayView2<'a, Complex>,
    /// Representative and equivalent q-pair site indices.
    pub q_pair_sites: ArrayView3<'a, usize>,
    /// Number of equivalent site pairs to use for each q-pair, FEFF `NIJQ`.
    pub q_pair_counts: &'a [usize],
    /// Per-site matrix offsets, FEFF `IND0Q`, converted to zero-based offsets.
    pub site_offsets: &'a [usize],
    /// Per-site state counts, FEFF `NKMQ`.
    pub site_state_counts: &'a [usize],
    /// FEFF `NGNT` triangular `(LM1,LM2)` row counts.
    pub gaunt_counts: &'a [usize],
    /// Flattened zero-based FEFF `IGNT` indices into `dllmmke`.
    pub gaunt_indices: &'a [usize],
    /// Flattened FEFF `GNT` coefficients aligned with `gaunt_indices`.
    pub gaunt_values: &'a [Real],
    /// FEFF `CIPWL(LM)` phase table.
    pub cipwl: ArrayView1<'a, Complex>,
    /// Complex wave number `P`; FEFF subtracts `i*P` from first-pair diagonals.
    pub wave_number: Complex,
}

/// Explicit inputs for FEFF `KSPACE/strset.f90` with `IREL >= 2`.
///
/// Site indices are zero-based and `q_pair_sites` is shaped as
/// `(q_pair, equivalent_pair, [row_site, column_site])`, replacing FEFF's
/// packed `IJQ = 100*IQ + JQ` representation. Relativistic transform tables
/// use FEFF axes `(term, spin, relativistic_state)` for `IRREL/SRREL` and
/// `(spin, relativistic_state)` for `NRREL`.
pub struct KSpaceStrsetRelInput<'a> {
    /// FEFF `NLM = NL**2`, the non-relativistic angular state count.
    pub angular_state_count: usize,
    /// FEFF `DLLMMKE(LM3,IQQP)` from `STRBBDD`.
    pub dllmmke: ArrayView2<'a, Complex>,
    /// Representative and equivalent q-pair site indices.
    pub q_pair_sites: ArrayView3<'a, usize>,
    /// Number of equivalent site pairs to use for each q-pair, FEFF `NIJQ`.
    pub q_pair_counts: &'a [usize],
    /// Per-site matrix offsets, FEFF `IND0Q`, converted to zero-based offsets.
    pub site_offsets: &'a [usize],
    /// Per-site relativistic state counts, FEFF `NKMQ`.
    pub site_state_counts: &'a [usize],
    /// FEFF `NGNT` triangular `(LM1,LM2)` row counts.
    pub gaunt_counts: &'a [usize],
    /// Flattened zero-based FEFF `IGNT` indices into `dllmmke`.
    pub gaunt_indices: &'a [usize],
    /// Flattened FEFF `GNT` coefficients aligned with `gaunt_indices`.
    pub gaunt_values: &'a [Real],
    /// FEFF `CIPWL(LM)` phase table.
    pub cipwl: ArrayView1<'a, Complex>,
    /// Number of non-rel components for each spin/state, FEFF `NRREL(IS,IKM)`.
    pub rel_component_counts: ArrayView2<'a, usize>,
    /// Non-rel angular indices for each spin/state component, zero-based FEFF `IRREL`.
    pub rel_component_indices: ArrayView3<'a, usize>,
    /// Relativistic transform coefficients, FEFF `SRREL`.
    pub rel_component_coefficients: ArrayView3<'a, Complex>,
    /// Complex wave number `P`; FEFF adds `i*P` to first-pair `GNR` diagonals before `TAUKINV=-G`.
    pub wave_number: Complex,
}

/// FEFF `STRVECGEN` q-pair grouping data shared by `STRBBDD` and `STRSET`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceQPairGroups {
    /// Unique pair offsets `(IQQP, xyz)`, FEFF `QQPX/QQPY/QQPZ`.
    pub offsets: RealMat,
    /// Equivalent site pairs `(IQQP, equivalent, [row_site, column_site])`.
    pub sites: Array3<usize>,
    /// Number of valid equivalent site pairs for each q-pair, FEFF `NIJQ`.
    pub counts: Vec<usize>,
    /// Longest q-pair offset norm, FEFF `RMAX1`.
    pub max_offset_norm: Real,
}

impl KSpaceQPairGroups {
    /// Number of unique q-pair offsets, FEFF `NQQP`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.nrows()
    }

    /// Whether no q-pair offsets were generated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// FEFF `STRVECGEN` direct-lattice setup shared by `STRAA` and `STRBBDD`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceDirectLatticeSetup {
    /// Sorted direct-lattice integer triplets `(I, xyz)`, FEFF `R1/R2/R3`.
    pub direct_indices: Array2<i32>,
    /// Per-pair direct-list row references `(S, IQQP)`, zero-based FEFF `INDR`.
    pub direct_index_by_pair: Array2<usize>,
    /// Adjusted direct-term counts for each q-pair, FEFF `SMAX`.
    pub direct_counts: Vec<usize>,
    /// Search half-width for integer lattice enumeration, FEFF `NUMRH`.
    pub index_radius: i32,
}

/// FEFF `STRVECGEN` reciprocal-lattice setup used by `STRBBDD`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceReciprocalLatticeSetup {
    /// Sorted reciprocal-lattice integer triplets `(N, xyz)`, FEFF `G1/G2/G3`.
    pub reciprocal_indices: Array2<i32>,
    /// Squared reciprocal cutoff, FEFF `GMAXSQ`.
    pub gmax_squared: Real,
    /// Search half-width for integer lattice enumeration, FEFF `NUMGH`.
    pub index_radius: i32,
}

/// FEFF `STRAA` reciprocal pair phase table used by `STRBBDD`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceReciprocalPairPhases {
    /// Precomputed pair phases for reciprocal vectors `(N, IQQP)`, FEFF `EXPGNQ`.
    pub reciprocal_pair_phases: Array2<Complex>,
    /// Largest absolute reciprocal integer index, FEFF `G123MAX`.
    pub max_index_abs: i32,
    /// FEFF `D1TERM1 = -4*pi / ATVOL` prefactor included in `EXPGNQ`.
    pub d1term1: Real,
}

/// FEFF `STRAA` base direct lattice table used by `STRCC` and `STRBBDD`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceDirectLatticeTerms {
    /// Base direct terms `(MMLL, S, IQQP)`, FEFF `QQMLRS` before `IILERS`.
    pub direct_terms: Array3<Complex>,
    /// Continued-fraction radial terms `(J22, LL, S, IQQP)`, FEFF `GGJLRS`.
    pub radial_terms: Array4<Real>,
    /// Largest absolute direct integer index among accepted terms, FEFF `R123MAX`.
    pub max_index_abs: i32,
    /// FEFF `Q1 = -sqrt(ETA/pi)/2` prefactor included in the base terms.
    pub q1: Real,
}

/// FEFF `STRCC` energy-dependent KSPACE table products.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceEnergyDependentTerms {
    /// Direct terms after multiplying by `IILERS`, FEFF current-energy `QQMLRS`.
    pub direct_terms: Array3<Complex>,
    /// Energy-dependent direct multipliers `(LL, S, IQQP)`, FEFF `IILERS`.
    pub direct_multipliers: Array3<Complex>,
    /// Reciprocal DLM1 angular weights, FEFF `D1TERM3(LL)`.
    pub d1term3: Array1<Complex>,
    /// FEFF missing-term correction added to `DLLMMKE(1,1)`.
    pub d300: Complex,
    /// FEFF Ewald term threshold test result; callers may rerun setup with a changed `ETA`.
    pub ewald_terms_exceed_threshold: bool,
}

/// Energy-independent FEFF `STRAA` tables at the initial Ewald parameter.
///
/// These tables may be reused across energies while `ETA` remains unchanged.
/// A `STRCC` threshold retry must rebuild both tables at the retry `ETA`.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceInitialEwaldTables {
    /// Initial FEFF Ewald splitting parameter used to build both tables.
    pub eta: Real,
    /// Initial-`ETA` FEFF `EXPGNQ` reciprocal pair phase table.
    pub reciprocal_pair_phases: KSpaceReciprocalPairPhases,
    /// Initial-`ETA` FEFF `QQMLRS`/`GGJLRS` base direct tables.
    pub direct_lattice_terms: KSpaceDirectLatticeTerms,
}

/// Complete KSPACE Ewald tables for one energy after FEFF `change_eta` retries.
#[derive(Debug, Clone, PartialEq)]
pub struct KSpaceEwaldEnergyTables {
    /// Final Ewald splitting parameter after zero or more FEFF `change_eta` retries.
    pub eta: Real,
    /// Number of times FEFF's `ETA *= 1.4` retry policy was applied.
    pub retry_count: usize,
    /// Rebuilt FEFF `EXPGNQ` reciprocal pair phase table.
    pub reciprocal_pair_phases: KSpaceReciprocalPairPhases,
    /// Rebuilt FEFF `QQMLRS`/`GGJLRS` base direct tables.
    pub direct_lattice_terms: KSpaceDirectLatticeTerms,
    /// Rebuilt FEFF `STRCC` energy-dependent products.
    pub energy_dependent_terms: KSpaceEnergyDependentTerms,
}

/// Vector reduction result from FEFF reciprocal-space helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedVector {
    /// Reduced vector. `subtract_a` returns reduced coordinates; `reduce`
    /// returns Cartesian coordinates reconstructed inside the unit cell.
    pub vector: Vector3,
    /// Sum of absolute nearest-integer translations FEFF accumulated in `l`.
    pub translation_count: usize,
}

/// FEFF `basdiv` k-mesh division result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KMeshDivisions {
    /// Number of intervals for each reciprocal-lattice vector.
    pub divisions: [usize; 3],
    /// FEFF-adjusted `(n1 + 1) * (n2 + 1) * (n3 + 1)` mesh-point count.
    pub mesh_points: usize,
}

/// FEFF `ARBMSH` generated k-mesh data.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshArbitraryMesh {
    /// Requested FEFF k-point count, `nka`.
    pub requested_point_count: usize,
    /// Number of intervals for each reciprocal-lattice vector, FEFF `n`.
    pub divisions: [usize; 3],
    /// FEFF work-mesh point count, `nkw = (n1 + 1) * (n2 + 1) * (n3 + 1)`.
    pub work_point_count: usize,
    /// FEFF full Brillouin-zone point count, `nkf = n1 * n2 * n3`.
    pub full_point_count: usize,
    /// FEFF irreducible Brillouin-zone point count, `nki`.
    pub irreducible_point_count: usize,
    /// FEFF unnormalized reduction weight sum, `sumwgt`.
    pub total_weight: Real,
    /// FEFF work, full, and irreducible k-point arrays from `REDUZ`.
    pub reduction: KMeshReduction,
    /// Optional FEFF `TETCNT` records when tetrahedra are requested.
    pub tetrahedra: Option<KMeshTetrahedronRecords>,
}

/// FEFF `divisi` common-factor reduction result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KMeshDivisionReduction {
    /// Reduced integer k-point list as `(point, xyz)`.
    pub k_list: Array2<i32>,
    /// FEFF-adjusted divisor value after integer division and lower-bound clamp.
    pub division: usize,
    /// Product of prime factors removed from every k-point component.
    pub common_divisor: usize,
}

/// Weyl-distributed LDOS k-point replacement from FEFF `LDOS/changeklist`.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosWeylKMesh {
    /// Generated k-points as `(point, xyz)`, equivalent to FEFF `ktab(:,i)`.
    pub k_points: RealMat,
    /// Uniform FEFF weights, `1 / nktab`.
    pub weights: Array1<Real>,
}

/// FEFF `TETCNT` tetrahedron record table.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshTetrahedronRecords {
    /// Number of irreducible k-points FEFF writes as `nki`.
    pub irreducible_point_count: usize,
    /// Total tetrahedra before equivalent records are merged.
    pub tetrahedron_count: usize,
    /// Number of unique tetrahedron records after FEFF ordering and merging.
    pub unique_tetrahedron_count: usize,
    /// Per-tetrahedron integration weight `1 / (6 * n1 * n2 * n3)`.
    pub tetrahedron_weight: Real,
    /// FEFF write chunk size from `m_tetrahedra.f90`.
    pub write_chunk_size: usize,
    /// Number of FEFF record chunks required for `records`.
    pub record_count: usize,
    /// Rows are FEFF `ITTFL` records: multiplicity and four 1-based corners.
    pub records: Array2<usize>,
}

/// FEFF `REDUZ` k-mesh reduction data.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshReduction {
    /// FEFF half-mesh shift selector `ishift`.
    pub shift: [usize; 3],
    /// Sum of unnormalized work-point boundary weights before normalization.
    pub total_weight: Real,
    /// Integer work-mesh coordinates as `(point, xyz)`.
    pub work_grid: Array2<usize>,
    /// Work-mesh Cartesian k-vectors as `(point, xyz)`, FEFF `bkw`.
    pub work_vectors: RealMat,
    /// Full-mesh Cartesian k-vectors as `(point, xyz)`, FEFF `bkf`.
    pub full_vectors: RealMat,
    /// Irreducible Cartesian k-vectors as `(point, xyz)`, FEFF `bki`.
    pub irreducible_vectors: RealMat,
    /// Irreducible fractional k-vectors as `(point, xyz)`, FEFF `bki2`.
    pub irreducible_fractional_vectors: RealMat,
    /// Normalized work-point weights, FEFF `ww`.
    pub work_weights: Array1<Real>,
    /// Normalized full-mesh weights, FEFF `wf`.
    pub full_weights: Array1<Real>,
    /// Normalized irreducible weights, FEFF `wi`.
    pub irreducible_weights: Array1<Real>,
    /// Final 1-based work-to-irreducible links, FEFF `linkw`.
    pub work_links: Vec<usize>,
    /// 1-based full-to-irreducible links, FEFF `linkf`.
    pub full_links: Vec<usize>,
    /// 1-based symmetry operation used for each work point, FEFF `lsymw`.
    pub work_symmetry: Vec<usize>,
    /// 1-based symmetry operation used for each full point, FEFF `lsymf`.
    pub full_symmetry: Vec<usize>,
}

/// FEFF `bravais` lattice-basis construction result.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeshBravaisBasis {
    /// FEFF-adjusted lengths after centering-specific scaling.
    pub adjusted_lengths: Vector3,
    /// Direct lattice vectors as `(vector, xyz)`, matching FEFF `rbas(i,*)`.
    pub direct_vectors: RealMat,
    /// Reciprocal lattice matrix exactly as returned by FEFF `bravais`.
    ///
    /// FEFF calls `GBASS` immediately afterward when row-vector reciprocal
    /// storage is needed for the mesh generator.
    pub reciprocal_vectors: RealMat,
    /// FEFF `afact` centering factor.
    pub afact: Real,
    /// FEFF `iarb` dependency flags for equal k-mesh divisions.
    pub dependencies: [bool; 3],
    /// FEFF `ortho` flag.
    pub orthogonal: bool,
    /// FEFF Brillouin-zone volume `v` from `KSPACE/kmesh.f90` `bravais`.
    pub brillouin_zone_volume: Real,
}

/// Point-group operations returned by FEFF `KSPACE/pointgroup.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointGroup {
    /// Rotation operations as `(operation, row, column)`.
    pub operations: Array3<Real>,
}

impl PointGroup {
    /// Number of point-group operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.shape()[0]
    }

    /// Whether the point group has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return one operation as a fixed 3 by 3 matrix.
    #[must_use]
    pub fn operation(&self, index: usize) -> Option<[[Real; 3]; 3]> {
        if index < self.len() {
            Some([
                [
                    self.operations[(index, 0, 0)],
                    self.operations[(index, 0, 1)],
                    self.operations[(index, 0, 2)],
                ],
                [
                    self.operations[(index, 1, 0)],
                    self.operations[(index, 1, 1)],
                    self.operations[(index, 1, 2)],
                ],
                [
                    self.operations[(index, 2, 0)],
                    self.operations[(index, 2, 1)],
                    self.operations[(index, 2, 2)],
                ],
            ])
        } else {
            None
        }
    }
}

/// FEFF `symmetrycheck` multiplication table and error selector.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryCheck {
    /// FEFF-style multiplication table as `(left_operation, right_operation)`.
    ///
    /// Positive entries are one-based operation indices. A `-1` entry means
    /// the rotational product exists but the corresponding translation check
    /// failed, matching FEFF `mult(i,j) = -1`.
    pub multiplication: Array2<i32>,
    /// FEFF `ierr`: zero when no invalid entries were found, otherwise the
    /// one-based operation index with the largest number of invalid products.
    pub ierr: usize,
}

impl SymmetryCheck {
    /// Number of operations represented by the square multiplication table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.multiplication.nrows()
    }

    /// Whether the check contains no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert FEFF's one-based `ierr` value into a Rust zero-based index.
    #[must_use]
    pub fn invalid_operation_index(&self) -> Option<usize> {
        self.ierr.checked_sub(1)
    }
}

/// Error returned by reciprocal-space helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum KSpaceError {
    /// FEFF space-group numbers are in the crystallographic range 1..=230.
    #[error("space group must be in 1..=230, got {space_group}")]
    InvalidSpaceGroup { space_group: i32 },
    /// FEFF produced no valid Bravais lattice for this lattice centering.
    #[error("invalid Bravais lattice for space group {space_group} and lattice '{lattice}'")]
    InvalidBravaisResult { space_group: i32, lattice: char },
    /// FEFF Bravais indices are 1..=14.
    #[error("invalid Bravais lattice index {index}")]
    InvalidBravaisIndex { index: i32 },
    /// `define_kpath` only implements the FEFF Bravais tables that do not stop.
    #[error("Bravais lattice {bravais} has no FEFF K-path table")]
    UnsupportedBravais { bravais: i32 },
    /// The requested `KPATH` selector is not defined for the Bravais lattice.
    #[error("invalid KPATH {kpath} for Bravais lattice {bravais}")]
    InvalidKPath { bravais: i32, kpath: i32 },
    /// Matrix inputs must be 3 by 3.
    #[error("{name} must have shape (3, 3), got ({rows}, {columns})")]
    InvalidMatrixShape {
        name: &'static str,
        rows: usize,
        columns: usize,
    },
    /// Coordinate values must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// Reduced coordinates must fit in FEFF's nearest-integer translation count.
    #[error("nearest-integer translation for component {component} is too large: {value}")]
    TranslationOverflow { component: usize, value: Real },
    /// The accumulated translation count overflowed a Rust `usize`.
    #[error("translation count overflowed")]
    TranslationCountOverflow,
    /// Internal guard for malformed segment tables.
    #[error(
        "KPATH {kpath} for Bravais lattice {bravais} produced only {available} of {required} segments"
    )]
    KPathDefinitionIncomplete {
        bravais: i32,
        kpath: i32,
        available: usize,
        required: usize,
    },
    /// FEFF `bandtot` K-path sampling needs at least two requested points.
    #[error("BAND K-path point count must be at least 2, got {point_count}")]
    InvalidBandKPathPointTarget { point_count: usize },
    /// Public K-path data must keep labels, starts, and ends aligned.
    #[error(
        "KPATH table shape mismatch: labels={labels}, starts=({start_rows}, {start_columns}), ends=({end_rows}, {end_columns})"
    )]
    InvalidKPathSegmentShape {
        labels: usize,
        start_rows: usize,
        start_columns: usize,
        end_rows: usize,
        end_columns: usize,
    },
    /// FEFF `bandtot` divides by total K-path length for multi-segment paths.
    #[error("BAND K-path total segment length is degenerate")]
    DegenerateBandKPathLength,
    /// FEFF-compatible BAND K-path point counts must fit addressable memory.
    #[error("BAND K-path point count overflowed")]
    BandKPathPointCountOverflow,
    /// FEFF `strbbdd` expects lattice and pair arrays with exact shapes.
    #[error(
        "{name} must have shape ({expected_rows}, {expected_columns}), got ({rows}, {columns})"
    )]
    InvalidStructureFactorShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    /// FEFF `strbbdd` expects 3-D direct-term arrays with exact shapes.
    #[error(
        "{name} must have shape ({expected_first}, {expected_second}, {expected_third}), got ({first}, {second}, {third})"
    )]
    InvalidStructureFactorCubeShape {
        name: &'static str,
        first: usize,
        second: usize,
        third: usize,
        expected_first: usize,
        expected_second: usize,
        expected_third: usize,
    },
    /// FEFF `strcc` expects 4-D radial-term arrays with exact shapes.
    #[error(
        "{name} must have shape ({expected_first}, {expected_second}, {expected_third}, {expected_fourth}), got ({first}, {second}, {third}, {fourth})"
    )]
    InvalidStructureFactorArray4Shape {
        name: &'static str,
        first: usize,
        second: usize,
        third: usize,
        fourth: usize,
        expected_first: usize,
        expected_second: usize,
        expected_third: usize,
        expected_fourth: usize,
    },
    /// FEFF `strbbdd` vector tables must align with pair and angular momentum counts.
    #[error("{name} length must be {expected}, got {actual}")]
    InvalidStructureFactorLength {
        name: &'static str,
        actual: usize,
        expected: usize,
    },
    /// FEFF `strbbdd` Ewald parameters must be positive.
    #[error("{name} must be positive, got {value}")]
    InvalidStructureFactorPositiveParameter { name: &'static str, value: Real },
    /// FEFF structure-factor ranges must be ordered.
    #[error("{name} range is invalid: min={min}, max={max}")]
    InvalidStructureFactorRange {
        name: &'static str,
        min: Real,
        max: Real,
    },
    /// FEFF structure-factor counts must be positive and consistent with setup.
    #[error("{name} count is invalid: {count}")]
    InvalidStructureFactorCount { name: &'static str, count: usize },
    /// FEFF structure-factor array sizes must fit addressable memory.
    #[error("{name} size overflowed")]
    StructureFactorSizeOverflow { name: &'static str },
    /// FEFF structure-factor phase ratios divide by this value.
    #[error("{name}[{index}] is degenerate")]
    DegenerateStructureFactorValue { name: &'static str, index: usize },
    /// FEFF `strbbdd` direct-list indirection must reference an existing row.
    #[error("{name} index {index} must be less than {len}")]
    StructureFactorIndexOutOfRange {
        name: &'static str,
        index: usize,
        len: usize,
    },
    /// FEFF `change_eta` stops when the new Ewald parameter exceeds the hard maximum.
    #[error("Ewald eta {eta} exceeds FEFF maximum {max}")]
    EwaldEtaExceeded { eta: Real, max: Real },
    /// FEFF `pointgroup` requires a positive output capacity.
    #[error("point-group operation capacity must be positive, got {capacity}")]
    InvalidPointGroupCapacity { capacity: usize },
    /// FEFF `pointgroup` divides by the metric diagonal values.
    #[error("point-group metric diagonal {index} must be positive and non-degenerate, got {value}")]
    DegenerateMetricDiagonal { index: usize, value: Real },
    /// FEFF `pointgroup` stops when a cutoff denominator is too small.
    #[error("point-group metric denominator {index} is degenerate: {value}")]
    DegenerateMetricDenominator { index: usize, value: Real },
    /// Candidate reciprocal-vector enumeration exceeded addressable memory.
    #[error("point-group candidate vector count overflowed")]
    PointGroupSearchOverflow,
    /// FEFF `pointgroup` stops when the supplied operation array is too small.
    #[error("point group produced more than {capacity} operations")]
    TooManyPointGroupOperations { capacity: usize },
    /// FEFF `pointgroup` expects at least one matching operation.
    #[error("no point-group operations matched the supplied metric")]
    NoPointGroupOperations,
    /// FEFF `GBASS` divides by the cell volume.
    #[error("lattice basis has a degenerate reciprocal volume determinant: {determinant}")]
    DegenerateLatticeVolume { determinant: Real },
    /// FEFF `basdiv` requires a positive requested mesh-point count.
    #[error("requested k-mesh point count must be positive, got {mesh_points}")]
    InvalidKMeshPointTarget { mesh_points: usize },
    /// FEFF `basdiv` scales by reciprocal-vector lengths.
    #[error("reciprocal vector {index} has degenerate length {length}")]
    DegenerateReciprocalVector { index: usize, length: Real },
    /// FEFF `basdiv` could not compute a finite scale from the input basis.
    #[error(
        "k-mesh scale is invalid for requested point count {mesh_points} and reciprocal-vector length product {length_product}"
    )]
    InvalidKMeshScale {
        mesh_points: usize,
        length_product: Real,
    },
    /// FEFF-compatible k-mesh divisions must fit addressable Rust memory.
    #[error("k-mesh division component {component} is too large: {value}")]
    KMeshDivisionOverflow { component: usize, value: Real },
    /// FEFF `basdiv` mesh-point count overflowed Rust `usize`.
    #[error("k-mesh point count overflowed")]
    KMeshPointCountOverflow,
    /// FEFF `tetdiv` divides by each k-mesh division component.
    #[error("k-mesh division component {component} must be positive, got {value}")]
    InvalidKMeshDivision { component: usize, value: usize },
    /// FEFF `tetcnt` expects six tetrahedra with four 3-D corner offsets.
    #[error(
        "tetrahedron offsets must have shape (6, 4, 3), got ({tetrahedra}, {corners}, {coordinates})"
    )]
    InvalidTetrahedronOffsetShape {
        tetrahedra: usize,
        corners: usize,
        coordinates: usize,
    },
    /// FEFF `tetcnt` consumes zero/one offsets generated by `tetdiv`.
    #[error("tetrahedron offset ({tetrahedron}, {corner}, {axis}) must be 0 or 1, got {value}")]
    InvalidTetrahedronOffset {
        tetrahedron: usize,
        corner: usize,
        axis: usize,
        value: i32,
    },
    /// FEFF `tetcnt` needs a positive irreducible k-point count.
    #[error("irreducible k-point count must be positive, got {count}")]
    InvalidIrreducibleKPointCount { count: usize },
    /// FEFF `tetcnt` needs one link for every work mesh point.
    #[error("work mesh link count must be {expected}, got {actual}")]
    InvalidWorkMeshLinkCount { expected: usize, actual: usize },
    /// FEFF `tetcnt` uses 1-based irreducible k-point links.
    #[error("work mesh link {index} must be in 1..={irreducible_point_count}, got {value}")]
    InvalidWorkMeshLink {
        index: usize,
        value: usize,
        irreducible_point_count: usize,
    },
    /// FEFF `tetcnt` tetrahedron count overflowed Rust `usize`.
    #[error("k-mesh tetrahedron count overflowed")]
    KMeshTetrahedronCountOverflow,
    /// FEFF `REDUZ` full-mesh links are internal 1-based indices.
    #[error("full mesh link {index} must be in 1..={full_point_count}, got {value}")]
    InvalidFullMeshLink {
        index: usize,
        value: usize,
        full_point_count: usize,
    },
    /// FEFF `REDUZ` checks that each irreducible point is emitted exactly once.
    #[error("k-mesh reduction emitted {actual} irreducible points, expected {expected}")]
    KMeshReductionIrreducibleCountMismatch { expected: usize, actual: usize },
    /// FEFF `REDUZ` checks that full points do not duplicate link/symmetry pairs.
    #[error(
        "full mesh points {first} and {second} both map to link {link} with symmetry {symmetry}"
    )]
    DuplicateFullMeshLinkSymmetry {
        first: usize,
        second: usize,
        link: usize,
        symmetry: usize,
    },
    /// FEFF `divisi` expects a non-empty `(n, 3)` integer k-point list.
    #[error("k-mesh list must have shape (n, 3) with n > 0, got ({rows}, {columns})")]
    InvalidKMeshListShape { rows: usize, columns: usize },
    /// FEFF `divisi` common divisor overflowed Rust `usize`.
    #[error("k-mesh common divisor overflowed")]
    KMeshCommonDivisorOverflow,
    /// FEFF `symmetrycheck` requires at least one operation.
    #[error("at least one symmetry operation is required")]
    NoSymmetryOperations,
    /// FEFF `symmetrycheck` expects operations as `(operation, 3, 3)`.
    #[error("symmetry operations must have shape (n, 3, 3), got ({operations}, {rows}, {columns})")]
    InvalidSymmetryOperationShape {
        operations: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF `symmetrycheck` expects one 3-vector translation per operation.
    #[error("symmetry translations must have shape ({operations}, 3), got ({rows}, {columns})")]
    InvalidSymmetryTranslationShape {
        operations: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF-compatible multiplication entries must fit in a signed integer.
    #[error("symmetry operation count {count} does not fit FEFF multiplication entries")]
    SymmetryOperationCountOverflow { count: usize },
    /// FEFF stops when the rotational products are not closed.
    #[error(
        "symmetry product for operations {left} and {right} is missing from the operation table"
    )]
    SymmetryProductMissing { left: usize, right: usize },
    /// FEFF-compatible symmetry entries must round back into signed integers.
    #[error(
        "transformed symmetry operation {operation} entry ({row}, {column}) is not representable as i32: {value}"
    )]
    SymmetryOperationValueOverflow {
        operation: usize,
        row: usize,
        column: usize,
        value: Real,
    },
    /// FEFF matrix inversion failed while preparing point-group operations.
    #[error(transparent)]
    Linalg(#[from] LinalgError),
    /// FEFF angular helper failed while preparing KSPACE angular tables.
    #[error(transparent)]
    Angular(#[from] AngularError),
}
