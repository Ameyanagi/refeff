use refeff_engine::{EngineError, ModuleName};

#[cfg(all(feature = "exafs", not(feature = "full")))]
use std::path::PathBuf;

#[test]
fn unsupported_module_name_is_typed() -> Result<(), Box<dyn std::error::Error>> {
    let error = match ModuleName::parse("definitely-not-feff") {
        Ok(_) => return Err("unsupported module name was accepted".into()),
        Err(error) => error,
    };
    assert!(matches!(
        error.downcast_ref::<EngineError>(),
        Some(EngineError::UnsupportedModule { module })
            if module == "definitely-not-feff"
    ));
    Ok(())
}

#[cfg(all(feature = "exafs", not(feature = "full")))]
#[test]
fn exafs_build_rejects_full_only_module_before_io() -> Result<(), Box<dyn std::error::Error>> {
    let error = match refeff_engine::run_named_module(
        ModuleName::Rixs,
        PathBuf::from("input-does-not-need-to-exist"),
    ) {
        Ok(_) => return Err("full-only RIXS module was accepted".into()),
        Err(error) => error,
    };
    assert!(matches!(
        error.downcast_ref::<EngineError>(),
        Some(EngineError::FeatureDisabled {
            module: "rixs",
            feature: "full"
        })
    ));
    Ok(())
}

#[cfg(all(feature = "exafs", not(feature = "sfconv")))]
#[test]
fn ordinary_exafs_does_not_enable_sfconv() {
    assert_eq!(ModuleName::Sfconv.disabled_feature(), Some("sfconv"));
    assert_eq!(ModuleName::Path.disabled_feature(), None);
    assert_eq!(ModuleName::Genfmt.disabled_feature(), None);
    assert_eq!(ModuleName::Ff2x.disabled_feature(), None);
}
