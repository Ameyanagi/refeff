use ndarray::{Array1, ArrayView1};

use super::plasma::sfconv_q_limits_with_upper;
use super::support::*;
use super::*;

mod analytic;
mod broadened;
mod limits;
mod real;
mod satellite;

pub use analytic::{
    sfconv_extrinsic_beta, sfconv_free_electron_exchange, sfconv_imaginary_self_energy,
    sfconv_imaginary_self_energy_derivative,
};
pub use broadened::{
    sfconv_broadened_self_energy, sfconv_broadened_self_energy_derivative,
    sfconv_broadened_self_energy_derivative_integrands, sfconv_broadened_self_energy_integrands,
};
pub use real::{
    sfconv_real_self_energy, sfconv_real_self_energy_derivative,
    sfconv_real_self_energy_derivative_integrand_lower,
    sfconv_real_self_energy_derivative_integrand_middle,
    sfconv_real_self_energy_derivative_integrand_upper, sfconv_real_self_energy_integrand_lower,
    sfconv_real_self_energy_integrand_middle, sfconv_real_self_energy_integrand_upper,
};
pub use satellite::{
    sfconv_extrinsic_satellite_broadened, sfconv_extrinsic_satellite_debroadened,
    sfconv_interference_quasiparticle, sfconv_interference_quasiparticle_integrand,
    sfconv_interference_satellite, sfconv_interference_satellite_integrand,
    sfconv_intrinsic_satellite, sfconv_intrinsic_satellite_integrand,
};

use limits::*;
