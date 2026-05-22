use ndarray::ArrayView2;

/// FEFF `UPLO` selector for real symmetric eigensolvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricTriangle {
    /// Read the lower triangle, matching FEFF `UPLO = 'L'`.
    Lower,
    /// Read the upper triangle, matching FEFF `UPLO = 'U'`.
    Upper,
}

impl SymmetricTriangle {
    pub(crate) fn includes(self, row: usize, col: usize) -> bool {
        match self {
            Self::Lower => row >= col,
            Self::Upper => row <= col,
        }
    }

    pub(crate) fn selected_entry<T: Copy>(
        self,
        matrix: ArrayView2<'_, T>,
        row: usize,
        col: usize,
    ) -> T {
        match self {
            Self::Lower if row >= col => matrix[(row, col)],
            Self::Lower => matrix[(col, row)],
            Self::Upper if row <= col => matrix[(row, col)],
            Self::Upper => matrix[(col, row)],
        }
    }
}
