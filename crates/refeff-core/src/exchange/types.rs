use super::*;

/// Error returned by exchange-potential helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ExchangeError {
    /// Inputs must be finite real values.
    #[error("exchange input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Inputs used as positive physical scales must be strictly positive.
    #[error("exchange input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Inputs used as nonnegative physical factors must be zero or positive.
    #[error("exchange input {name} must be nonnegative, got {value}")]
    NegativeInput { name: &'static str, value: Real },
    /// A square-root radicand fell outside the real branch used by FEFF.
    #[error("exchange radicand {name} must be nonnegative, got {value}")]
    NegativeRadicand { name: &'static str, value: Real },
    /// A logarithm argument fell outside the real branch used by FEFF.
    #[error("exchange logarithm argument {name} must be positive, got {value}")]
    NonPositiveLogArgument { name: &'static str, value: Real },
}

/// Exchange-correlation energy and potential from FEFF LDA helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExchangeCorrelation {
    /// Exchange-correlation energy per particle in Hartrees.
    pub energy_per_particle: Real,
    /// Exchange-correlation potential in Hartrees.
    pub potential: Real,
}

/// Spin branch used by FEFF `fxc_ksdt_01` and `exc_ksdt_01`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KsdTSpin {
    /// FEFF `iz = 0`, spin-unpolarized.
    Unpolarized,
    /// FEFF `iz = 1`, fully spin-polarized.
    FullyPolarized,
}

/// KSDT exchange-correlation free energy and potential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsdTFreeEnergy {
    /// Exchange-correlation free energy per particle in Hartrees.
    pub free_energy_per_particle: Real,
    /// Exchange-correlation potential in Hartrees.
    pub potential: Real,
}

/// Result from FEFF `imhl`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedinLundqvistImaginary {
    /// Imaginary self-energy returned by FEFF `imhl`.
    pub value: Real,
    /// FEFF `icusp` flag, true at the beginning of the imaginary branch cusp.
    pub cusp: bool,
}

/// Result from FEFF `rhl`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedinLundqvistSelfEnergy {
    /// Real Hedin-Lundqvist self-energy from FEFF `rhl`.
    pub real: Real,
    /// Imaginary self-energy from FEFF `imhl`, as returned by `rhl`.
    pub imaginary: Real,
    /// FEFF `imhl` cusp flag used to choose the real-branch interpolation.
    pub cusp: bool,
}
