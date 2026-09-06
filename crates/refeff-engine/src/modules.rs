//! Single CLI-independent registry for stage identities, aliases and features.
use crate::{EngineError, Result};
#[derive(Debug, Clone, Copy)]
pub struct ModuleDescriptor {
    pub module: ModuleName,
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// Required Cargo feature, or an empty string for the always-available parser.
    pub feature: &'static str,
    pub available: bool,
}
macro_rules! modules {
    ($($variant:ident => ($name:literal, [$($alias:literal),*], $feature:literal)),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(usize)]
        pub enum ModuleName { $($variant),* }
        pub const MODULES: &[ModuleDescriptor] = &[$(ModuleDescriptor {
            module:ModuleName::$variant, name:$name, aliases:&[$($alias),*],
            feature:if matches!(ModuleName::$variant, ModuleName::Rdinp) { "" } else { $feature },
            available:matches!(ModuleName::$variant, ModuleName::Rdinp) || cfg!(feature=$feature)
        }),*];
    }
}
modules! {
    Rdinp => ("rdinp", [], "exafs"),
    Pot => ("pot", [], "exafs"),
    Atomic => ("atomic", ["atom"], "exafs"),
    Band => ("band", [], "full"),
    Mdff => ("eelsmdff", ["mdff"], "full"),
    Wpot => ("wpot", [], "exafs"),
    Opcons => ("opconsat", ["opcons"], "full"),
    Compton => ("compton", [], "full"),
    Fullspectrum => ("fullspectrum", [], "full"),
    Crpa => ("crpa", [], "full"),
    Screen => ("screen", [], "exafs"),
    Ldos => ("ldos", [], "full"),
    Eels => ("eels", [], "full"),
    Dmdw => ("dmdw", [], "full"),
    Path => ("path", ["paths"], "exafs"),
    Genfmt => ("genfmt", [], "exafs"),
    Ff2x => ("ff2x", [], "exafs"),
    Xsph => ("xsph", [], "exafs"),
    Fms => ("fms", [], "full"),
    Mkgtr => ("mkgtr", [], "full"),
    Rixs => ("rixs", [], "full"),
    Rhorrp => ("rhorrp", [], "full"),
    Sfconv => ("sfconv", [], "sfconv"),
    SelfEnergy => ("self", ["selfenergy"], "sfconv"),
}
impl ModuleName {
    pub fn parse(value: &str) -> Result<Self> {
        MODULES
            .iter()
            .find(|item| {
                item.name.eq_ignore_ascii_case(value)
                    || item
                        .aliases
                        .iter()
                        .any(|alias| alias.eq_ignore_ascii_case(value))
            })
            .map(|item| item.module)
            .ok_or_else(|| {
                EngineError::UnsupportedModule {
                    module: value.into(),
                }
                .into()
            })
    }
    pub const fn descriptor(self) -> &'static ModuleDescriptor {
        &MODULES[self as usize]
    }
    pub const fn as_str(self) -> &'static str {
        self.descriptor().name
    }
    pub const fn disabled_feature(self) -> Option<&'static str> {
        let descriptor = self.descriptor();
        if descriptor.available {
            None
        } else {
            Some(descriptor.feature)
        }
    }
}
impl std::str::FromStr for ModuleName {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
    }
}
