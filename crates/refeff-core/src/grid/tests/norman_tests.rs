use super::{support::*, *};

#[test]
fn norman_radius_matches_feff_frnrm_oxygen_like_reference() -> Result<(), GridError> {
    let density = sample_frnrm_oxygen_density();

    let result = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: 8,
    })?;

    assert_close(result.radius, 1.063_980_446_859_560_2);
    Ok(())
}

#[test]
fn norman_radius_matches_feff_frnrm_iron_like_reference() -> Result<(), GridError> {
    let density = sample_frnrm_iron_density();

    let result = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: 26,
    })?;

    assert_close(result.radius, 8.688_945_443_598_616e-1);
    Ok(())
}

#[test]
fn norman_radius_matches_feff_frnrm_gold_like_reference() -> Result<(), GridError> {
    let density = sample_frnrm_gold_density();

    let result = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: 79,
    })?;

    assert_close(result.radius, 6.973_687_583_509_427e-1);
    Ok(())
}

#[test]
fn norman_radius_rejects_invalid_inputs() {
    let density = Array1::<Real>::from_elem(FRNRM_DENSITY_POINTS, 1.0);
    assert_eq!(
        norman_radius_from_density(NormanRadiusInput {
            overlapped_density: density.view(),
            atomic_number: 0,
        }),
        Err(GridError::InvalidAtomicNumber { atomic_number: 0 })
    );

    let short_density = Array1::<Real>::zeros(FRNRM_DENSITY_POINTS - 1);
    assert_eq!(
        norman_radius_from_density(NormanRadiusInput {
            overlapped_density: short_density.view(),
            atomic_number: 1,
        }),
        Err(GridError::SourceGridTooShort {
            name: "overlapped_density",
            required: FRNRM_DENSITY_POINTS,
            available: FRNRM_DENSITY_POINTS - 1,
        })
    );

    let zero_density = Array1::<Real>::zeros(FRNRM_DENSITY_POINTS);
    assert!(matches!(
        norman_radius_from_density(NormanRadiusInput {
            overlapped_density: zero_density.view(),
            atomic_number: 1,
        }),
        Err(GridError::InsufficientNormanCharge {
            atomic_number: 1,
            ..
        })
    ));
}

#[test]
fn interstitial_fermi_level_matches_feff_fermi_reference() -> Result<(), GridError> {
    let shell = interstitial_fermi_level(FermiLevelInput {
        interstitial_density: 8.430_358_921_763_391e-1,
        interstitial_potential: -1.294_131_834_592_241_2,
    })?;
    assert_fermi_level(
        shell,
        -5.040_450_363_824_843e-1,
        1.526_716_490_479_997_5,
        1.257_049_560_049_051_4,
    );

    let dense = interstitial_fermi_level(FermiLevelInput {
        interstitial_density: 3.2,
        interstitial_potential: -0.42,
    })?;
    assert_fermi_level(
        dense,
        1.502_548_984_343_600_6,
        9.787_169_102_922_159e-1,
        1.960_892_135_913_447_2,
    );
    Ok(())
}

#[test]
fn interstitial_fermi_level_rejects_invalid_inputs() {
    assert_eq!(
        interstitial_fermi_level(FermiLevelInput {
            interstitial_density: 0.0,
            interstitial_potential: -1.0,
        }),
        Err(GridError::NonPositiveScalar {
            name: "interstitial_density",
            value: 0.0,
        })
    );
    assert!(matches!(
        interstitial_fermi_level(FermiLevelInput {
            interstitial_density: 1.0,
            interstitial_potential: Real::NAN,
        }),
        Err(GridError::NonFiniteScalar {
            name: "interstitial_potential",
            ..
        })
    ));
}
