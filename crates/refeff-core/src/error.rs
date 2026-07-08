//! Crate-level error umbrella over refeff-core's per-module error enums.
//!
//! Every module in this crate defines its own `thiserror` error enum scoped
//! to the FEFF routines it ports (see e.g. [`crate::AngularError`],
//! [`crate::XsphError`]). [`enum@Error`] wraps all of them behind one type so
//! callers composing helpers from several modules (a driver, a CLI stage, a
//! test harness) can propagate a single error type with `?` instead of
//! threading each module's error through by hand. Module-scoped `Result`
//! aliases are unaffected; this is purely an additive convenience type.
//!
//! `refeff_core::fms::FmsError` is intentionally not included here: the FMS
//! driver/types/solvers are under active development in parallel and adding
//! a `#[from]` arm here would create merge churn against that work.

use thiserror::Error;

use crate::{
    AngularError, AtomMathError, AtomicError, BandError, BesselError, ComptonError,
    ConvolutionError, CoreHoleError, DebyeError, DensityError, EelsError, ElamError, ExchangeError,
    FovrgError, FprimeError, FullSpectrumError, GenfmtError, GridError, InterpolationError,
    KSpaceError, OpconsError, OptimizationError, OrbitalConfigurationError, PathError, PhaseError,
    QuadratureError, RhorrpError, RixsError, RootError, ScreenError, SelfEnergyError, SfconvError,
    SortError, SpecialFunctionError, StateKetError, VectorError, XscorrError, XsphError,
};

/// Umbrella error over every non-FMS module error enum in refeff-core.
///
/// See the module documentation for why `fms::FmsError` is excluded.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum Error {
    /// Error from [`crate::angular`].
    #[error(transparent)]
    Angular(#[from] AngularError),
    /// Error from [`crate::atomic`].
    #[error(transparent)]
    Atomic(#[from] AtomicError),
    /// Error from [`crate::atomic`] math helpers.
    #[error(transparent)]
    AtomMath(#[from] AtomMathError),
    /// Error from [`crate::band`].
    #[error(transparent)]
    Band(#[from] BandError),
    /// Error from [`crate::bessel`].
    #[error(transparent)]
    Bessel(#[from] BesselError),
    /// Error from [`crate::compton`].
    #[error(transparent)]
    Compton(#[from] ComptonError),
    /// Error from [`crate::convolution`].
    #[error(transparent)]
    Convolution(#[from] ConvolutionError),
    /// Error from [`crate::core_hole`].
    #[error(transparent)]
    CoreHole(#[from] CoreHoleError),
    /// Error from [`crate::debye`].
    #[error(transparent)]
    Debye(#[from] DebyeError),
    /// Error from [`crate::density`].
    #[error(transparent)]
    Density(#[from] DensityError),
    /// Error from [`crate::eels`].
    #[error(transparent)]
    Eels(#[from] EelsError),
    /// Error from [`crate::elam`].
    #[error(transparent)]
    Elam(#[from] ElamError),
    /// Error from [`crate::exchange`].
    #[error(transparent)]
    Exchange(#[from] ExchangeError),
    /// Error from [`crate::fovrg`].
    #[error(transparent)]
    Fovrg(#[from] FovrgError),
    /// Error from [`crate::fprime`].
    #[error(transparent)]
    Fprime(#[from] FprimeError),
    /// Error from [`crate::fullspectrum`].
    #[error(transparent)]
    FullSpectrum(#[from] FullSpectrumError),
    /// Error from [`crate::genfmt`].
    #[error(transparent)]
    Genfmt(#[from] GenfmtError),
    /// Error from [`crate::grid`].
    #[error(transparent)]
    Grid(#[from] GridError),
    /// Error from [`crate::interpolation`].
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
    /// Error from [`crate::kspace`].
    #[error(transparent)]
    KSpace(#[from] KSpaceError),
    /// Error from [`crate::opcons`].
    #[error(transparent)]
    Opcons(#[from] OpconsError),
    /// Error from [`crate::optimization`].
    #[error(transparent)]
    Optimization(#[from] OptimizationError),
    /// Error from [`crate::configuration`].
    #[error(transparent)]
    OrbitalConfiguration(#[from] OrbitalConfigurationError),
    /// Error from [`crate::path`].
    #[error(transparent)]
    Path(#[from] PathError),
    /// Error from [`crate::phase`].
    #[error(transparent)]
    Phase(#[from] PhaseError),
    /// Error from [`crate::quadrature`].
    #[error(transparent)]
    Quadrature(#[from] QuadratureError),
    /// Error from [`crate::rhorrp`].
    #[error(transparent)]
    Rhorrp(#[from] RhorrpError),
    /// Error from [`crate::rixs`].
    #[error(transparent)]
    Rixs(#[from] RixsError),
    /// Error from [`crate::roots`].
    #[error(transparent)]
    Root(#[from] RootError),
    /// Error from [`crate::screen`].
    #[error(transparent)]
    Screen(#[from] ScreenError),
    /// Error from [`crate::self_energy`].
    #[error(transparent)]
    SelfEnergy(#[from] SelfEnergyError),
    /// Error from [`crate::sfconv`].
    #[error(transparent)]
    Sfconv(#[from] SfconvError),
    /// Error from [`crate::sort`].
    #[error(transparent)]
    Sort(#[from] SortError),
    /// Error from [`crate::special`].
    #[error(transparent)]
    SpecialFunction(#[from] SpecialFunctionError),
    /// Error from [`crate::state`].
    #[error(transparent)]
    StateKet(#[from] StateKetError),
    /// Error from [`crate::vector`].
    #[error(transparent)]
    Vector(#[from] VectorError),
    /// Error from [`crate::xscorr`].
    #[error(transparent)]
    Xscorr(#[from] XscorrError),
    /// Error from [`crate::xsph`].
    #[error(transparent)]
    Xsph(#[from] XsphError),
}

/// Convenience alias for `Result<T, refeff_core::Error>`.
pub type Result<T> = core::result::Result<T, Error>;
