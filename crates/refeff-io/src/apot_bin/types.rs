//! Public data structures for FEFF `apot.bin` section streams.

use ndarray::Array2;
use num_complex::Complex64;

/// Scalar type marker written in a FEFF `#DT#` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApotBinType {
    /// Fortran `INTEGER`.
    Int,
    /// Fortran `REAL`.
    Real,
    /// Fortran `DOUBLE PRECISION`.
    Double,
    /// Fortran `COMPLEX`.
    Complex,
    /// Fortran `COMPLEX*16`.
    DComplex,
    /// Fixed-width Fortran character payload.
    Text,
}

impl ApotBinType {
    pub(super) fn record_name(self) -> &'static str {
        match self {
            Self::Int => "Int",
            Self::Real => "Real",
            Self::Double => "Double",
            Self::Complex => "Complex",
            Self::DComplex => "DComplex",
            Self::Text => "String",
        }
    }

    pub(super) fn matrix_name(self) -> &'static str {
        match self {
            Self::Int => "integer",
            Self::Real => "real",
            Self::Double => "double",
            Self::Complex => "complex",
            Self::DComplex => "double complex",
            Self::Text => "string",
        }
    }

    pub(super) fn token_width(self) -> usize {
        match self {
            Self::Complex | Self::DComplex => 2,
            Self::Int | Self::Real | Self::Double | Self::Text => 1,
        }
    }

    pub(super) fn is_real(self) -> bool {
        matches!(self, Self::Real | Self::Double)
    }

    pub(super) fn is_complex(self) -> bool {
        matches!(self, Self::Complex | Self::DComplex)
    }
}

/// One typed scalar value from a FEFF `WriteData` or `WriteArrayData` section.
#[derive(Debug, Clone, PartialEq)]
pub enum ApotBinValue {
    /// Integer value.
    Int(i64),
    /// Real or double-precision value.
    Real(f64),
    /// Complex or double-complex value.
    Complex(Complex64),
    /// Character value.
    Text(String),
}

/// Typed row data from a FEFF `WriteData` or `WriteArrayData` section.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinRecords {
    /// Column types from the section `#DT#` line.
    pub column_types: Vec<ApotBinType>,
    /// Parsed rows. Scalar `WriteData` sections have one row; `WriteArrayData`
    /// sections have one row per array element.
    pub rows: Vec<Vec<ApotBinValue>>,
}

impl ApotBinRecords {
    /// Number of parsed data rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of typed values per row.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.column_types.len()
    }
}

/// Storage for a FEFF `Write2D` matrix section.
#[derive(Debug, Clone, PartialEq)]
pub enum ApotBinMatrixValues {
    /// Integer matrix.
    Int(Array2<i64>),
    /// Real or double-precision matrix.
    Real(Array2<f64>),
    /// Complex or double-complex matrix.
    Complex(Array2<Complex64>),
    /// Character matrix.
    Text(Array2<String>),
}

impl ApotBinMatrixValues {
    fn dim(&self) -> (usize, usize) {
        match self {
            Self::Int(values) => values.dim(),
            Self::Real(values) => values.dim(),
            Self::Complex(values) => values.dim(),
            Self::Text(values) => values.dim(),
        }
    }
}

/// Typed payload from a FEFF `Write2D` section.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinMatrix {
    /// Matrix element type from the section `#DT#` line.
    pub value_type: ApotBinType,
    /// Matrix values in FEFF row/column order.
    pub values: ApotBinMatrixValues,
}

impl ApotBinMatrix {
    /// Matrix shape as `(rows, columns)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.values.dim()
    }

    /// Number of matrix rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.shape().0
    }

    /// Number of matrix columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.shape().1
    }
}

/// Parsed payload variant for one FEFF `apot.bin` section.
#[derive(Debug, Clone, PartialEq)]
pub enum ApotBinPayload {
    /// Header-only section or header lines attached outside a data section.
    HeadersOnly,
    /// `WriteData` or `WriteArrayData` row payload.
    Records(ApotBinRecords),
    /// `Write2D` matrix payload.
    Matrix(ApotBinMatrix),
}

/// One `#SN#` section from FEFF `apot.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinSection {
    /// One-based section number from the `#SN#` marker.
    pub section_number: usize,
    /// Semantic `#H#` lines before the payload.
    pub headers: Vec<String>,
    /// Raw payloads after `#H#` for [`Self::headers`], preserving FEFF padding.
    pub header_texts: Vec<String>,
    /// `#CL#` column labels when FEFF supplied them.
    pub column_labels: Vec<String>,
    /// Raw `#CL#` payload after the marker, preserving FEFF field padding.
    pub column_label_text: Option<String>,
    /// Parsed payload.
    pub payload: ApotBinPayload,
    /// Semantic `#H#` lines emitted after the payload, before the next section.
    ///
    /// FEFF sometimes uses a header-only `WriteData` call to annotate the next
    /// group of matrix sections. Because that call does not force a new section,
    /// those headers are attached to the previous section in the raw file.
    pub trailing_headers: Vec<String>,
    /// Raw payloads after `#H#` for [`Self::trailing_headers`].
    pub trailing_header_texts: Vec<String>,
}

impl ApotBinSection {
    /// Return row-record payload when this is a `WriteData` or
    /// `WriteArrayData` section.
    #[must_use]
    pub fn records(&self) -> Option<&ApotBinRecords> {
        match &self.payload {
            ApotBinPayload::Records(records) => Some(records),
            ApotBinPayload::HeadersOnly | ApotBinPayload::Matrix(_) => None,
        }
    }

    /// Return matrix payload when this is a `Write2D` section.
    #[must_use]
    pub fn matrix(&self) -> Option<&ApotBinMatrix> {
        match &self.payload {
            ApotBinPayload::Matrix(matrix) => Some(matrix),
            ApotBinPayload::HeadersOnly | ApotBinPayload::Records(_) => None,
        }
    }
}

/// Parsed FEFF `apot.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinData {
    /// Sections in file order.
    pub sections: Vec<ApotBinSection>,
}

impl ApotBinData {
    /// Number of parsed sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Number of parsed matrix sections.
    #[must_use]
    pub fn matrix_count(&self) -> usize {
        self.sections
            .iter()
            .filter(|section| section.matrix().is_some())
            .count()
    }
}
