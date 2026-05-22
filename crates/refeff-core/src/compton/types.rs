use super::*;

/// FEFF COMPTON apodization mode used by `jpq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComptonWindow {
    /// FEFF `window = 0`: rectangular cutoff in `z'`.
    Rectangular,
    /// FEFF `window = 1`: squared cosine taper up to the cutoff.
    CosineSquared,
    /// FEFF fallback branch for any other integer `window` value.
    Unwindowed,
}

/// Input values for FEFF `compton_build_grid`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonGridInput {
    /// Number of cylindrical-radius samples, FEFF `ns`.
    pub ns: usize,
    /// Number of azimuth samples, FEFF `nphi`.
    pub nphi: usize,
    /// Number of `z` samples, FEFF `nz`.
    pub nz: usize,
    /// Number of `z'` samples, FEFF `nzp`.
    pub nzp: usize,
    /// Maximum cylindrical radius. A zero value uses [`Self::norman_radius`].
    pub smax: Real,
    /// Maximum azimuth angle.
    pub phimax: Real,
    /// Maximum `z` coordinate. A zero value uses [`Self::norman_radius`].
    pub zmax: Real,
    /// Maximum `z'` coordinate.
    pub zpmax: Real,
    /// FEFF `rnrm(0)` fallback used when `smax` or `zmax` is zero.
    pub norman_radius: Real,
    /// Momentum-transfer direction, FEFF `qhat`.
    pub qhat: Vector3,
}

/// FEFF COMPTON integration grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ComptonGrid {
    /// Cylindrical radial grid `s`.
    pub s: RealVec,
    /// Azimuth grid `phi`.
    pub phi: RealVec,
    /// Longitudinal grid `z`.
    pub z: RealVec,
    /// Companion longitudinal grid `z'`.
    pub zp: RealVec,
    /// Whether FEFF rotates sample points from the q-axis frame.
    pub rotate: bool,
    /// FEFF rotation matrix from q-axis coordinates to cluster coordinates.
    pub rotation_matrix: RealMat,
}

impl ComptonGrid {
    /// Number of cylindrical-radius samples.
    #[must_use]
    pub fn ns(&self) -> usize {
        self.s.len()
    }

    /// Number of azimuth samples.
    #[must_use]
    pub fn nphi(&self) -> usize {
        self.phi.len()
    }

    /// Number of `z` samples.
    #[must_use]
    pub fn nz(&self) -> usize {
        self.z.len()
    }

    /// Number of `z'` samples.
    #[must_use]
    pub fn nzp(&self) -> usize {
        self.zp.len()
    }
}

/// Inputs for FEFF `jpq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonProfileInput {
    /// Projected momentum value `p_q`.
    pub pq: Real,
    /// FEFF window branch.
    pub window: ComptonWindow,
    /// FEFF `window_cutoff`; zero means use the upper end of `grid.zp`.
    pub window_cutoff: Real,
}

/// Inputs for FEFF's `rhozzp.dat` diagnostic slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonRhoZzpInput {
    /// Number of diagnostic samples. FEFF writes 1000 points.
    pub sample_count: usize,
    /// Fixed `z` coordinate and starting `z'` coordinate. FEFF uses `0.01`.
    pub base_z: Real,
}

/// FEFF `rhozzp.dat` diagnostic values.
#[derive(Debug, Clone, PartialEq)]
pub struct ComptonRhoZzpSlice {
    /// Unrotated `z'` coordinate written as the first output column.
    pub z_prime: RealVec,
    /// Density callback values written as the second output column.
    pub rho: RealVec,
}

/// Rotation axis and angle returned by FEFF `rotation_axis_angle`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonRotationAxisAngle {
    /// Cross-product axis `a x b`.
    pub axis: Vector3,
    /// Rotation angle in radians.
    pub theta: Real,
}

/// Error returned by FEFF COMPTON helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ComptonError {
    /// Scalar inputs must be finite real values.
    #[error("COMPTON input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Vector components must be finite.
    #[error("COMPTON vector {name}[{axis}] must be finite, got {value}")]
    NonFiniteVector {
        name: &'static str,
        axis: usize,
        value: Real,
    },
    /// Grid counts must allow FEFF's `(n - 1)` denominators.
    #[error("COMPTON grid count {name} must be at least 2, got {value}")]
    InvalidGridCount { name: &'static str, value: usize },
    /// Extents used to build linear grids must be nonnegative and finite.
    #[error("COMPTON grid extent {name} must be nonnegative and finite, got {value}")]
    InvalidGridExtent { name: &'static str, value: Real },
    /// A vector norm is required in the corresponding FEFF formula.
    #[error("COMPTON vector {name} must be nonzero")]
    ZeroVector { name: &'static str },
    /// The computed Wigner-style rotation ratio is outside the real asin domain.
    #[error("COMPTON rotation ratio must be in [0, 1], got {value}")]
    InvalidRotationRatio { value: Real },
    /// Rotation matrices must have FEFF's 3x3 shape.
    #[error("COMPTON rotation matrix must have shape (3, 3), got ({rows}, {columns})")]
    InvalidRotationMatrixShape { rows: usize, columns: usize },
    /// `J(z,z')` must match the supplied grid dimensions.
    #[error(
        "COMPTON jzzp shape ({rows}, {columns}) does not match grid shape ({expected_rows}, {expected_columns})"
    )]
    InvalidJzzpShape {
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    /// Piecewise-linear Fourier intervals cannot have zero width.
    #[error("COMPTON {axis} interval {index} has zero width")]
    ZeroFourierInterval { axis: &'static str, index: usize },
    /// Active FEFF windows require a positive cutoff after defaulting.
    #[error("COMPTON window cutoff must be positive, got {value}")]
    InvalidWindowCutoff { value: Real },
    /// A computed result became non-finite.
    #[error("COMPTON result {name} must be finite, got {value}")]
    NonFiniteResult { name: &'static str, value: Real },
    /// The density callback returned a non-finite value.
    #[error("COMPTON density callback must return finite values, got {value}")]
    NonFiniteDensity { value: Real },
}
