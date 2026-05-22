use crate::Real;

pub(super) const ATOMIC_DENSITY_CUTOFF_SQUARED: Real = 4.0;
pub(super) const ATOMIC_DENSITY_MIN_RADIUS: Real = 1.0e-4;
pub(super) const ATOMIC_DENSITY_INTERPOLATION_ORDER: usize = 2;
pub(super) const DENSITY_INTEGRATION_HORIZONTAL_EPSILON: Real = 1.0e-15;
pub(super) const DENSITY_INTEGRATION_INTERPOLATION_ORDER: usize = 2;
pub(super) const DENSITY_INTEGRATION_SUBDIVISIONS: usize = 10;
pub(super) const FEFF_FINE_STRUCTURE_ALPHA: Real = 1.0 / 137.03598956;
pub(super) const RHORRP_ORIGIN_EPSILON: Real = 1.0e-3;
