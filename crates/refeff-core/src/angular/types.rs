use super::*;

/// Error returned by angular normalization helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AngularError {
    /// Integer indices must fit in `u32` before conversion to `f64`.
    #[error("angular index {value} is too large for stable floating-point conversion")]
    IndexTooLarge { value: usize },
    /// FEFF angular helpers accept only integer (`1`) and half-integer (`2`) scales.
    #[error("invalid angular momentum scale {scale}; expected 1 or 2")]
    InvalidWignerScale { scale: i32 },
    /// A Wigner 3j argument did not divide evenly by the selected scale.
    #[error("Wigner 3j argument {argument} is not divisible by scale {scale}")]
    InvalidWignerParity { argument: i32, scale: i32 },
    /// FEFF's common `cwig3j` table is limited to factorial arguments up to 58.
    #[error("Wigner 3j factorial argument {argument} exceeds FEFF limit {limit}")]
    WignerFactorialOutOfRange { argument: i32, limit: i32 },
    /// FEFF relativistic state indexing uses nonzero kappa values.
    #[error("invalid relativistic kappa {kappa}; expected nonzero finite i32 range")]
    InvalidRelativisticKappa { kappa: i32 },
    /// FEFF `MUEM05` must lie in `-abs(kappa)..abs(kappa)-1`.
    #[error("relativistic MUEM05 {mu_minus_half} is outside kappa {kappa} range")]
    RelativisticMagneticIndexOutOfRange { kappa: i32, mu_minus_half: i32 },
    /// The requested magnetic index does not fit the allocated table.
    #[error("magnetic index {magnetic} is outside table range for lmax {lmax}")]
    MagneticIndexOutOfRange { magnetic: isize, lmax: usize },
    /// A FEFF angular helper received an inconsistent output table dimension.
    #[error("angular table dimension {name} must be at least {minimum}, got {value}")]
    InvalidAngularTableDimension {
        name: &'static str,
        value: usize,
        minimum: usize,
    },
    /// FEFF Wigner rotations require a finite angle.
    #[error("Wigner rotation angle must be finite")]
    NonFiniteRotationAngle,
    /// FEFF `ylm` requires finite Cartesian vector components.
    #[error("spherical-harmonic vector components must be finite")]
    NonFiniteVector,
    /// FEFF transition matrices require finite polarization tensor entries.
    #[error("polarization tensor entry ({row}, {column}) must be finite")]
    NonFinitePolarizationTensor { row: isize, column: isize },
    /// FEFF basis transformations consume square matrices of the compiled order.
    #[error("basis-transform matrix {name} must be {expected}x{expected}, got {rows}x{columns}")]
    InvalidBasisTransformShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected: usize,
    },
    /// FEFF `iniptz` accepts tensor selectors `1..=10`.
    #[error("invalid polarization tensor selector {index}; expected 1..=10")]
    InvalidPolarizationTensorIndex { index: usize },
    /// FEFF spin-folding expects at least one compiled spin channel.
    #[error("invalid FEFF spin channel count {value}; expected at least 1")]
    InvalidSpinChannelCount { value: usize },
}

/// Spin-orbit Clebsch-Gordon tables used by FEFF's FMS and POT paths.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinOrbitCouplingTables {
    /// `j = l + 1/2` coefficients, indexed as `[l, m + m_offset, spin - 1]`.
    pub plus: Array3<Real>,
    /// `j = l - 1/2` coefficients, indexed as `[l, m + m_offset, spin - 1]`.
    pub minus: Array3<Real>,
    /// Offset added to signed `m` before indexing the second axis.
    pub m_offset: usize,
}

/// FEFF `CALCCGC` relativistic Clebsch-Gordan coefficient table.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativisticClebschGordanCoefficients {
    /// FEFF `CGC(IKM, IS)` coefficients as `(state, spin_component)`.
    pub coefficients: RealMat,
    /// FEFF `LTAB` branch orbital momentum values.
    pub orbital_momentum: Vec<usize>,
    /// FEFF `KAPTAB` branch relativistic kappa values.
    pub kappa: Vec<i32>,
    /// FEFF `NMUETAB` branch state counts.
    pub spin_multiplicity: Vec<usize>,
}

/// FEFF `BASTRMAT` basis-transformation matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisTransformMatrices {
    /// Maximum orbital momentum used to construct the matrices.
    pub lmax: usize,
    /// Matrix order, FEFF `NKM = 2 * (LMAX + 1)^2`.
    pub order: usize,
    /// FEFF `RC`: transforms real spherical harmonics to complex harmonics.
    pub real_to_complex: ComplexMat,
    /// FEFF `CREL`: transforms complex harmonics to relativistic `(kappa,mue)`.
    pub complex_to_relativistic: ComplexMat,
    /// FEFF `RREL`: transforms real harmonics to relativistic `(kappa,mue)`.
    pub real_to_relativistic: ComplexMat,
}

/// FEFF `CHANGEREP` representation-conversion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisTransformMode {
    /// FEFF `REL>RLM`.
    RelativisticToReal,
    /// FEFF `RLM>REL`.
    RealToRelativistic,
    /// FEFF `REL>CLM`.
    RelativisticToComplex,
    /// FEFF `CLM>REL`.
    ComplexToRelativistic,
    /// FEFF `CLM>RLM`.
    ComplexToReal,
    /// FEFF `RLM>CLM`.
    RealToComplex,
}

/// Coordinate system used by FEFF `iniptz` when constructing `ptz`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarizationTensorMode {
    /// Direct spherical tensor basis, with selector `1..=9` choosing one entry.
    Spherical,
    /// Cartesian products rewritten into FEFF's spherical-index tensor basis.
    Cartesian,
}

/// Inputs for FEFF `bcoef`, the transition B-matrix builder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionBMatrixInput {
    /// Maximum orbital momentum `lx`, used for the signed magnetic-index axes.
    pub lmax: usize,
    /// Initial-state relativistic kappa.
    pub initial_kappa: i32,
    /// FEFF polarization selector. `0` uses the orientational average branch;
    /// nonzero values use the full tensor branch.
    pub polarization: i32,
    /// Polarization tensor indexed as `[p + 1][p_prime + 1]` for `p=-1..=1`.
    pub polarization_tensor: [[Complex; 3]; 3],
    /// FEFF `le2` multipole selector.
    pub multipole: i32,
    /// Whether to trace the resulting matrix over orbital `m_l`.
    pub trace_orbital: bool,
    /// FEFF spin selector `ispin`.
    pub spin: i32,
    /// Compiled number of spin channels, FEFF `nspu`.
    pub spin_channels: usize,
    /// Angle between x-ray k-vector and spin vector.
    pub spin_vector_angle: Real,
}

/// FEFF `bcoef` output in Rust-owned storage.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionBMatrix {
    /// Final-state kappa indices for FEFF transition slots `1..=8`.
    pub kappa_indices: [i32; 8],
    /// Orbital angular momenta for FEFF transition slots `1..=8`.
    pub orbital_momenta: [i32; 8],
    /// `bmat(ml2, ms2, k2, ml1, ms1, k1)` in FEFF axis order.
    pub matrix: Array6<Complex>,
    /// Offset added to signed `m_l` before indexing `matrix` axes 0 and 3.
    pub l_offset: usize,
}

impl TransitionBMatrix {
    /// Return a matrix element using FEFF's signed magnetic indices and
    /// one-based transition slots.
    #[must_use]
    pub fn value(
        &self,
        ml2: isize,
        ms2: usize,
        transition2: usize,
        ml1: isize,
        ms1: usize,
        transition1: usize,
    ) -> Option<Complex> {
        if ms1 > 1 || ms2 > 1 || !(1..=8).contains(&transition1) || !(1..=8).contains(&transition2)
        {
            return None;
        }
        let ml2 = self.magnetic_index(ml2)?;
        let ml1 = self.magnetic_index(ml1)?;
        self.matrix
            .get([ml2, ms2, transition2 - 1, ml1, ms1, transition1 - 1])
            .copied()
    }

    fn magnetic_index(&self, magnetic: isize) -> Option<usize> {
        let offset = isize::try_from(self.l_offset).ok()?;
        let index = magnetic + offset;
        let len = offset.checked_mul(2)?.checked_add(1)?;
        if index < 0 || index >= len {
            None
        } else {
            usize::try_from(index).ok()
        }
    }
}
