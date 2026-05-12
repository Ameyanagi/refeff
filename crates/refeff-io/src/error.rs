use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("too many nested include/load files while reading {path}")]
    IncludeDepth { path: PathBuf },

    #[error("recursive include/load of {path}")]
    RecursiveInclude { path: PathBuf },

    #[error("{path}:{line}: {message}")]
    Parse {
        path: PathBuf,
        line: usize,
        message: String,
    },

    #[error("failed to format FEFF text output")]
    Format {
        #[source]
        source: std::fmt::Error,
    },

    #[error(
        "invalid potential output shape for {field}: got {rows}x{cols}, expected at least {min_rows}x{min_cols}"
    )]
    PotentialOutputShape {
        field: &'static str,
        rows: usize,
        cols: usize,
        min_rows: usize,
        min_cols: usize,
    },

    #[error("invalid potential output value for {field}: {message}")]
    InvalidPotentialOutput {
        field: &'static str,
        message: String,
    },

    #[error(
        "invalid MTDP shape for {field}: got {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    MtdpShape {
        field: &'static str,
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },

    #[error("invalid MTDP length for {field}: got {len}, expected {expected}")]
    MtdpLength {
        field: &'static str,
        len: usize,
        expected: usize,
    },

    #[error("missing MTDP field {field}")]
    MtdpMissing { field: &'static str },

    #[error("could not parse MTDP field {field} from token {token:?}")]
    MtdpParse { field: &'static str, token: String },

    #[error("invalid MTDP value for {field}: {message}")]
    InvalidMtdp {
        field: &'static str,
        message: String,
    },

    #[error("MTDP input has {count} trailing token(s)")]
    MtdpTrailingTokens { count: usize },

    #[error("invalid pot.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    PotBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("missing pot.bin field {field}")]
    PotBinMissing { field: &'static str },

    #[error("could not parse pot.bin field {field} from token {token:?}")]
    PotBinParse { field: &'static str, token: String },

    #[error("invalid pot.bin value for {field}: {message}")]
    InvalidPotBin {
        field: &'static str,
        message: String,
    },

    #[error("pot.bin input has {count} trailing line(s)")]
    PotBinTrailingLines { count: usize },

    #[error("invalid phase.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    PhaseBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("missing phase.bin field {field}")]
    PhaseBinMissing { field: &'static str },

    #[error("could not parse phase.bin field {field} from token {token:?}")]
    PhaseBinParse { field: &'static str, token: String },

    #[error("invalid phase.bin value for {field}: {message}")]
    InvalidPhaseBin {
        field: &'static str,
        message: String,
    },

    #[error("phase.bin input has {count} trailing line(s)")]
    PhaseBinTrailingLines { count: usize },

    #[error("invalid feff.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    FeffBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("missing feff.bin field {field}")]
    FeffBinMissing { field: &'static str },

    #[error("could not parse feff.bin field {field} from token {token:?}")]
    FeffBinParse { field: &'static str, token: String },

    #[error("invalid feff.bin value for {field}: {message}")]
    InvalidFeffBin {
        field: &'static str,
        message: String,
    },

    #[error("missing list.dat field {field}")]
    ListDatMissing { field: &'static str },

    #[error("could not parse list.dat field {field} on line {line} from token {token:?}")]
    ListDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("list.dat row on line {line} has {actual} token(s), expected {expected}")]
    ListDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid list.dat value for {field}: {message}")]
    InvalidListDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid xsect.dat shape for {field}: got {actual}, expected {expected}")]
    XsectDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("missing xsect.dat field {field}")]
    XsectDatMissing { field: &'static str },

    #[error("could not parse xsect.dat field {field} on line {line} from token {token:?}")]
    XsectDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("xsect.dat row on line {line} has {actual} token(s), expected {expected}")]
    XsectDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid xsect.dat value for {field}: {message}")]
    InvalidXsectDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid fms.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    FmsBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("missing fms.bin field {field}")]
    FmsBinMissing { field: &'static str },

    #[error("could not parse fms.bin field {field} from token {token:?}")]
    FmsBinParse { field: &'static str, token: String },

    #[error("invalid fms.bin value for {field}: {message}")]
    InvalidFmsBin {
        field: &'static str,
        message: String,
    },

    #[error("invalid fmsl.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    FmslBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("invalid fmsl.bin value for {field}: {message}")]
    InvalidFmslBin {
        field: &'static str,
        message: String,
    },

    #[error("invalid xsecl.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    XseclBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("missing xsecl.bin field {field}")]
    XseclBinMissing { field: &'static str },

    #[error("could not parse xsecl.bin field {field} on line {line} from token {token:?}")]
    XseclBinParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("xsecl.bin row on line {line} has {actual} token(s), expected {expected}")]
    XseclBinRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid xsecl.bin value for {field}: {message}")]
    InvalidXseclBin {
        field: &'static str,
        message: String,
    },

    #[error("invalid feffl.bin shape for {field}: got {actual:?}, expected {expected:?}")]
    FefflBinShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("invalid feffl.bin value for {field}: {message}")]
    InvalidFefflBin {
        field: &'static str,
        message: String,
    },

    #[error("invalid PAD width {0}; expected at least 3")]
    InvalidPadWidth(usize),

    #[error("PAD line uses marker {found:?}, expected {expected:?}")]
    PadMarker { expected: char, found: char },

    #[error("PAD payload length {payload_len} is not a multiple of {unit_len}")]
    PadPayload { payload_len: usize, unit_len: usize },

    #[error("PAD encoder produced out-of-range byte {value}")]
    PadByte { value: i32 },

    #[error("PAD exponent index {index} does not fit in i32")]
    PadIndex { index: usize },

    #[error("polarization vector norm {norm} is too small")]
    InvalidPolarizationVector { norm: f64 },

    #[error("polarization vector is almost parallel to incidence vector; dot product {dot}")]
    InvalidPolarizationGeometry { dot: f64 },

    #[error("PAD encoded bytes were not valid UTF-8: {source}")]
    PadUtf8 {
        #[source]
        source: std::string::FromUtf8Error,
    },

    #[error("PAD payload chunk was not valid UTF-8: {source}")]
    PadChunkUtf8 {
        #[source]
        source: std::str::Utf8Error,
    },
}

impl IoError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

impl From<std::fmt::Error> for IoError {
    fn from(source: std::fmt::Error) -> Self {
        Self::Format { source }
    }
}
