use faer::{Mat, MatRef};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::{Complex32, Complex64};

/// Returns whether a `(rows, cols)` array with the given `strides` is laid
/// out in column-major order, i.e. the order `faer`'s `MatRef`/`MatMut`
/// `from_column_major_slice` constructors expect and that `ndarray`'s `.f()`
/// shape builder produces.
///
/// A single column (`cols <= 1`) or an empty matrix (`rows == 0 || cols ==
/// 0`) is column-major regardless of the reported column stride, since there
/// is no second column for that stride to separate.
pub(crate) fn is_column_major_2d(rows: usize, cols: usize, strides: &[isize]) -> bool {
    if rows == 0 || cols == 0 {
        return true;
    }
    strides.first().copied() == Some(1)
        && (cols == 1 || strides.get(1).copied() == Some(rows as isize))
}

/// Borrow `view`'s backing storage as a column-major slice without copying,
/// when its memory layout already matches the column-major order `faer`
/// expects (as `ndarray`'s `.f()`-shaped matrices do, e.g. the FMS system and
/// scattering matrices). Returns `None` for row-major or otherwise
/// non-contiguous storage so callers can fall back to a copy.
pub(crate) fn column_major_slice<'a, T>(view: ArrayView2<'a, T>) -> Option<&'a [T]> {
    if !is_column_major_2d(view.nrows(), view.ncols(), view.strides()) {
        return None;
    }
    view.to_slice_memory_order()
}

/// Mutable counterpart to [`column_major_slice`], used to solve directly into
/// a caller-owned buffer without an intervening `faer` copy.
pub(crate) fn column_major_slice_mut<T>(view: ArrayViewMut2<'_, T>) -> Option<&'_ mut [T]> {
    if !is_column_major_2d(view.nrows(), view.ncols(), view.strides()) {
        return None;
    }
    view.into_slice_memory_order()
}

/// A `faer` matrix view borrowed directly from `ndarray` storage when it is
/// already column-major, falling back to an owned element-wise copy
/// otherwise. This avoids the `Mat::from_fn` copy on every call for the
/// column-major matrices FMS builds throughout (`.f()`-shaped system and
/// scattering matrices).
pub(crate) enum FaerView<'a, T> {
    Borrowed(MatRef<'a, T>),
    Owned(Mat<T>),
}

impl<T> FaerView<'_, T> {
    pub(crate) fn as_ref(&self) -> MatRef<'_, T> {
        match self {
            Self::Borrowed(matrix) => *matrix,
            Self::Owned(matrix) => matrix.as_ref(),
        }
    }
}

/// Zero-copy (when column-major) view of a real `ndarray` matrix as `faer`.
pub(crate) fn real_view(view: ArrayView2<'_, f64>) -> FaerView<'_, f64> {
    match column_major_slice(view) {
        Some(slice) => FaerView::Borrowed(MatRef::from_column_major_slice(
            slice,
            view.nrows(),
            view.ncols(),
        )),
        None => FaerView::Owned(real_to_faer(view)),
    }
}

/// Zero-copy (when column-major) view of a complex `ndarray` matrix as `faer`.
pub(crate) fn complex_view(view: ArrayView2<'_, Complex64>) -> FaerView<'_, Complex64> {
    match column_major_slice(view) {
        Some(slice) => FaerView::Borrowed(MatRef::from_column_major_slice(
            slice,
            view.nrows(),
            view.ncols(),
        )),
        None => FaerView::Owned(complex_to_faer(view)),
    }
}

/// Zero-copy (when column-major) view of a single-precision complex
/// `ndarray` matrix as `faer`.
pub(crate) fn complex32_view(view: ArrayView2<'_, Complex32>) -> FaerView<'_, Complex32> {
    match column_major_slice(view) {
        Some(slice) => FaerView::Borrowed(MatRef::from_column_major_slice(
            slice,
            view.nrows(),
            view.ncols(),
        )),
        None => FaerView::Owned(complex32_to_faer(view)),
    }
}

/// Copy a real `ndarray` matrix view into a `faer` matrix.
pub fn real_to_faer(view: ArrayView2<'_, f64>) -> Mat<f64> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

/// Copy a complex `ndarray` matrix view into a `faer` matrix.
pub fn complex_to_faer(view: ArrayView2<'_, Complex64>) -> Mat<Complex64> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

/// Copy a single-precision complex `ndarray` matrix view into a `faer` matrix.
pub fn complex32_to_faer(view: ArrayView2<'_, Complex32>) -> Mat<Complex32> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

/// Copy a real `faer` matrix into row-indexed `ndarray` storage.
pub fn real_from_faer(matrix: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

/// Copy a complex `faer` matrix into row-indexed `ndarray` storage.
pub fn complex_from_faer(matrix: &Mat<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

/// Copy a single-precision complex `faer` matrix into row-indexed `ndarray` storage.
pub fn complex32_from_faer(matrix: &Mat<Complex32>) -> Array2<Complex32> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

/// Multiply two real matrices through the pure-Rust `faer` backend.
///
/// Borrows `lhs`/`rhs` directly into `faer` without an element-wise copy when
/// their storage is already column-major (as `ndarray`'s `.f()`-shaped
/// matrices are), falling back to a copy otherwise.
pub fn real_matmul(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>) -> Array2<f64> {
    let lhs = real_view(lhs);
    let rhs = real_view(rhs);
    real_from_faer(&(lhs.as_ref() * rhs.as_ref()))
}

/// Multiply two complex matrices through the pure-Rust `faer` backend.
///
/// Borrows `lhs`/`rhs` directly into `faer` without an element-wise copy when
/// their storage is already column-major (as `ndarray`'s `.f()`-shaped
/// matrices are), falling back to a copy otherwise.
pub fn complex_matmul(
    lhs: ArrayView2<'_, Complex64>,
    rhs: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let lhs = complex_view(lhs);
    let rhs = complex_view(rhs);
    complex_from_faer(&(lhs.as_ref() * rhs.as_ref()))
}
