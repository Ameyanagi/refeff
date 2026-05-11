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

    #[error("invalid PAD width {0}; expected at least 3")]
    InvalidPadWidth(usize),

    #[error("PAD line uses marker {found:?}, expected {expected:?}")]
    PadMarker { expected: char, found: char },

    #[error("PAD payload length {payload_len} is not a multiple of {unit_len}")]
    PadPayload { payload_len: usize, unit_len: usize },
}

impl IoError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
