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

    #[error("could not parse apot.bin field {field} on line {line} from token {token:?}")]
    ApotBinParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("invalid apot.bin data on line {line}: {message}")]
    InvalidApotBin { line: usize, message: String },

    #[error("invalid emesh.bin data: {message}")]
    InvalidEmeshBin { message: String },

    #[error("invalid gtrNN.bin data: {message}")]
    InvalidGtrBin { message: String },

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

    #[error("invalid xmu.dat shape for {field}: got {actual}, expected {expected}")]
    XmuDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse xmu.dat field {field} on line {line} from token {token:?}")]
    XmuDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("xmu.dat row on line {line} has {actual} token(s), expected {expected}")]
    XmuDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid xmu.dat value for {field}: {message}")]
    InvalidXmuDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid chi.dat shape for {field}: got {actual}, expected {expected}")]
    ChiDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse chi.dat field {field} on line {line} from token {token:?}")]
    ChiDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("chi.dat row on line {line} has {actual} token(s), expected {expected}")]
    ChiDatRowWidth {
        line: usize,
        actual: usize,
        expected: &'static str,
    },

    #[error("invalid chi.dat value for {field}: {message}")]
    InvalidChiDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid eels.dat shape for {field}: got {actual}, expected {expected}")]
    EelsDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error(
        "invalid eels.dat tensor shape: got {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    EelsDatTensorShape {
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },

    #[error("could not parse eels.dat field {field} on line {line} from token {token:?}")]
    EelsDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("eels.dat row on line {line} has {actual} token(s), expected {expected}")]
    EelsDatRowWidth {
        line: usize,
        actual: usize,
        expected: &'static str,
    },

    #[error("invalid eels.dat value for {field}: {message}")]
    InvalidEelsDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid danes.dat shape for {field}: got {actual}, expected {expected}")]
    DanesDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse danes.dat field {field} on line {line} from token {token:?}")]
    DanesDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("danes.dat row on line {line} has {actual} token(s), expected {expected}")]
    DanesDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid danes.dat value for {field}: {message}")]
    InvalidDanesDat {
        field: &'static str,
        message: String,
    },

    #[error(
        "invalid ldosNN.dat shape for {field}: got {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    LdosDatShape {
        field: &'static str,
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: &'static str,
    },

    #[error("could not parse ldosNN.dat field {field} on line {line} from token {token:?}")]
    LdosDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("ldosNN.dat row on line {line} has {actual} token(s), expected {expected}")]
    LdosDatRowWidth {
        line: usize,
        actual: usize,
        expected: &'static str,
    },

    #[error("invalid ldosNN.dat value for {field}: {message}")]
    InvalidLdosDat {
        field: &'static str,
        message: String,
    },

    #[error("missing log.dat field {field}")]
    LogDatMissing { field: &'static str },

    #[error("could not parse log.dat field {field} on line {line} from token {token:?}")]
    LogDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("invalid log.dat value for {field}: {message}")]
    InvalidLogDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid misc.dat line {line}: {message}")]
    InvalidMiscDat { line: usize, message: String },

    #[error("invalid compton.dat shape for {field}: got {actual}, expected {expected}")]
    ComptonDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse compton.dat field {field} on line {line} from token {token:?}")]
    ComptonDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("compton.dat row on line {line} has {actual} token(s), expected {expected}")]
    ComptonDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid compton.dat value for {field}: {message}")]
    InvalidComptonDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid rhozzp.dat shape for {field}: got {actual}, expected {expected}")]
    RhozzpDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse rhozzp.dat field {field} on line {line} from token {token:?}")]
    RhozzpDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("rhozzp.dat row on line {line} has {actual} token(s), expected {expected}")]
    RhozzpDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid rhozzp.dat value for {field}: {message}")]
    InvalidRhozzpDat {
        field: &'static str,
        message: String,
    },

    #[error(
        "invalid jzzp.dat shape for {field}: got {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    JzzpDatShape {
        field: &'static str,
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },

    #[error("missing jzzp.dat field {field}")]
    JzzpDatMissing { field: &'static str },

    #[error("could not parse jzzp.dat field {field} on line {line} from token {token:?}")]
    JzzpDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("invalid jzzp.dat value for {field}: {message}")]
    InvalidJzzpDat {
        field: &'static str,
        message: String,
    },

    #[error("missing crpa.dat field {field}")]
    CrpaDatMissing { field: &'static str },

    #[error("could not parse crpa.dat field {field} on line {line} from token {token:?}")]
    CrpaDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("crpa.dat row on line {line} has {actual} token(s), expected {expected}")]
    CrpaDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid crpa.dat value for {field}: {message}")]
    InvalidCrpaDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid loss.dat shape for {field}: got {actual}, expected {expected}")]
    LossDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse loss.dat field {field} on line {line} from token {token:?}")]
    LossDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("loss.dat row on line {line} has {actual} token(s), expected {expected}")]
    LossDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid loss.dat value for {field}: {message}")]
    InvalidLossDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid mpse.dat shape for {field}: got {actual}, expected {expected}")]
    MpseDatShape {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("could not parse mpse.dat field {field} on line {line} from token {token:?}")]
    MpseDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("mpse.dat row on line {line} has {actual} token(s), expected {expected}")]
    MpseDatRowWidth {
        line: usize,
        actual: usize,
        expected: &'static str,
    },

    #[error("invalid mpse.dat value for {field}: {message}")]
    InvalidMpseDat {
        field: &'static str,
        message: String,
    },

    #[error(
        "invalid RIXS data shape for {field}: got {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    RixsDatShape {
        field: &'static str,
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: &'static str,
    },

    #[error("could not parse RIXS data field {field} on line {line} from token {token:?}")]
    RixsDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("RIXS data row on line {line} has {actual} token(s), expected {expected}")]
    RixsDatRowWidth {
        line: usize,
        actual: usize,
        expected: &'static str,
    },

    #[error("invalid RIXS data value for {field}: {message}")]
    InvalidRixsDat {
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

    #[error("missing paths.dat field {field}")]
    PathsDatMissing { field: &'static str },

    #[error("could not parse paths.dat field {field} on line {line} from token {token:?}")]
    PathsDatParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("paths.dat row on line {line} has {actual} token(s), expected at least {expected}")]
    PathsDatRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid paths.dat value for {field}: {message}")]
    InvalidPathsDat {
        field: &'static str,
        message: String,
    },

    #[error("invalid FEFF .dym shape for {field}: got {actual:?}, expected {expected:?}")]
    DymShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("missing FEFF .dym field {field}")]
    DymMissing { field: &'static str },

    #[error("could not parse FEFF .dym field {field} on line {line} from token {token:?}")]
    DymParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("invalid FEFF .dym value for {field}: {message}")]
    InvalidDym {
        field: &'static str,
        message: String,
    },

    #[error("FEFF .dym input has {count} trailing token(s)")]
    DymTrailingTokens { count: usize },

    #[error("missing grid.inp field {field}")]
    GridInpMissing { field: &'static str },

    #[error("could not parse grid.inp field {field} on line {line} from token {token:?}")]
    GridInpParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("grid.inp row on line {line} has {actual} token(s), expected {expected}")]
    GridInpRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid grid.inp value for {field}: {message}")]
    InvalidGridInp {
        field: &'static str,
        message: String,
    },

    #[error("missing config.inp field {field} on line {line}")]
    ConfigInpMissing { field: &'static str, line: usize },

    #[error("could not parse config.inp field {field} on line {line} from token {token:?}")]
    ConfigInpParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("config.inp row on line {line} has {actual} token(s), expected at least {expected}")]
    ConfigInpRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid config.inp value for {field}: {message}")]
    InvalidConfigInp {
        field: &'static str,
        message: String,
    },

    #[error("missing spring.inp field {field}")]
    SpringInpMissing { field: &'static str },

    #[error("could not parse spring.inp field {field} on line {line} from token {token:?}")]
    SpringInpParse {
        field: &'static str,
        line: usize,
        token: String,
    },

    #[error("spring.inp row on line {line} has {actual} token(s), expected at least {expected}")]
    SpringInpRowWidth {
        line: usize,
        actual: usize,
        expected: usize,
    },

    #[error("invalid spring.inp value for {field}: {message}")]
    InvalidSpringInp {
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
