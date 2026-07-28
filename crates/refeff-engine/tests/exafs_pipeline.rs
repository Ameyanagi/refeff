#![cfg(feature = "exafs")]

use std::path::Path;

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "ZnSe parity runs in the release-profile test gate"
)]
fn znse_generates_the_expected_path_set() -> Result<(), Box<dyn std::error::Error>> {
    let output = tempfile::tempdir()?;
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/znse.inp");

    refeff_engine::execute_feff(&input, output.path())?;

    let mut paths = std::fs::read_dir(output.path())?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.file_name())
        .filter(|name| {
            name.to_str().is_some_and(|name| {
                name.len() == 12
                    && name.starts_with("feff")
                    && name.ends_with(".dat")
                    && name[4..name.len() - 4]
                        .bytes()
                        .all(|byte| byte.is_ascii_digit())
            })
        })
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(paths.len(), 15);
    assert_eq!(
        paths.first().and_then(|name| name.to_str()),
        Some("feff0001.dat")
    );
    assert_eq!(
        paths.last().and_then(|name| name.to_str()),
        Some("feff0016.dat")
    );

    #[cfg(not(feature = "full"))]
    {
        assert!(!output.path().join("fms.bin").exists());
        assert!(!output.path().join("gtr.dat").exists());
        assert!(!output.path().join("logband.dat").exists());
    }

    Ok(())
}
