use thiserror::Error;

/// Error returned by FEFF linear-algebra helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LinalgError {
    /// FEFF determinant and inversion helpers require square matrices.
    #[error("matrix must be square, got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },
    /// Matrix/vector dimensions must match.
    #[error("length mismatch: {left_name} has {left} values but {right_name} has {right}")]
    LengthMismatch {
        left_name: &'static str,
        left: usize,
        right_name: &'static str,
        right: usize,
    },
    /// FEFF `invertmatrix` stops when a pivot is exactly zero.
    #[error("matrix is singular at pivot {pivot}")]
    SingularMatrix { pivot: usize },
    /// Least-squares design matrices must contain at least one column.
    #[error("design matrix must have at least one column")]
    EmptyDesign,
    /// Selected matrix entries must be finite before calling the eigensolver.
    #[error("matrix entry ({row},{col}) must be finite")]
    NonFiniteMatrixEntry { row: usize, col: usize },
    /// Scalar inputs must be finite before evaluating FEFF helper formulas.
    #[error("{name} must be finite")]
    NonFiniteScalar { name: &'static str },
    /// FEFF's `SSYEV` reports positive `INFO` when eigenvalue iteration fails.
    #[error("symmetric eigensolver did not converge")]
    EigenDidNotConverge,
    /// An owned matrix buffer did not match the shape computed by the solver.
    #[error("matrix buffer has {len} values but shape is {rows}x{cols}")]
    InvalidOwnedShape {
        rows: usize,
        cols: usize,
        len: usize,
    },
    /// FEFF-compatible LU factors are expected to use contiguous row-major storage.
    #[error("LU factor storage is not contiguous")]
    NonContiguousLuFactors,
}
