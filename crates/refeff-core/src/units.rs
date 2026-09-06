//! Checked physical units for public boundaries. Kernels retain contiguous floats.
//! Conversions use COMMON FEFF constants; legacy solver constants remain unchanged.
use crate::constants::{BOHR_ANGSTROM, HARTREE_EV};
#[derive(Debug, Clone, Copy, thiserror::Error, PartialEq)]
#[error("physical quantity must be finite, got {0}")]
pub struct NonFiniteQuantity(pub f64);
macro_rules! unit {
    ($name:ident) => {
        #[doc = "Finite physical quantity, stored without changing its unit."]
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);
        impl $name {
            pub fn new(value: f64) -> Result<Self, NonFiniteQuantity> {
                if value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err(NonFiniteQuantity(value))
                }
            }
            pub const fn value(self) -> f64 {
                self.0
            }
        }
    };
}
unit!(ElectronVolts);
unit!(Hartrees);
unit!(Angstroms);
unit!(Bohrs);
unit!(InverseAngstroms);
impl Hartrees {
    pub fn to_ev(self) -> Result<ElectronVolts, NonFiniteQuantity> {
        ElectronVolts::new(self.0 * HARTREE_EV)
    }
}
impl ElectronVolts {
    pub fn to_hartrees(self) -> Result<Hartrees, NonFiniteQuantity> {
        Hartrees::new(self.0 / HARTREE_EV)
    }
}
impl Bohrs {
    pub fn to_angstroms(self) -> Result<Angstroms, NonFiniteQuantity> {
        Angstroms::new(self.0 * BOHR_ANGSTROM)
    }
}
impl Angstroms {
    pub fn to_bohrs(self) -> Result<Bohrs, NonFiniteQuantity> {
        Bohrs::new(self.0 / BOHR_ANGSTROM)
    }
}
