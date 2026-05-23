use super::{support::*, *};

#[test]
fn exponential_energy_grid_matches_feff_setegrid_reference() -> Result<(), ScreenError> {
    let grid = screen_exponential_energy_grid(8.0, 5)?;

    assert_complex_close(grid[0], 0.0, 8.000_000_000_000_002, 1.0e-14);
    assert_complex_close(grid[1], 0.0, 4.196_152_422_706_632, 1.0e-14);
    assert_complex_close(grid[2], 0.0, 2.000_000_000_000_000_4, 1.0e-14);
    assert_complex_close(grid[3], 0.0, 0.732_050_807_568_877_4, 1.0e-14);
    assert_complex_close(grid[4], 0.0, 0.0, 1.0e-14);
    Ok(())
}

#[test]
fn contour_energy_grid_matches_feff_setegi_reference() -> Result<(), ScreenError> {
    let grid = screen_contour_energy_grid(ScreenContourEnergyGridInput {
        min_real_energy: -0.2,
        max_real_energy: 0.4,
        max_imaginary_energy: 0.5,
        min_imaginary_energy: 0.0,
        real_points: 4,
        imaginary_points: 4,
        max_points: 20,
    })?;

    assert_eq!(grid.active_len, 10);
    assert_close(grid.effective_min_imaginary_energy, 0.05, 1.0e-15);
    assert_complex_close(grid.energies[0], -0.2, 0.05, 1.0e-14);
    assert_complex_close(grid.energies[1], -0.2, 0.2, 1.0e-14);
    assert_complex_close(grid.energies[2], -0.2, 0.35, 1.0e-14);
    assert_complex_close(grid.energies[3], -0.2, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[4], -5.551_115_123_125_783e-17, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[5], 0.2, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[6], 0.4, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[7], 0.4, 0.35, 1.0e-14);
    assert_complex_close(grid.energies[8], 0.4, 0.2, 1.0e-14);
    assert_complex_close(grid.energies[9], 0.4, 0.05, 1.0e-14);
    assert_complex_close(grid.energies[10], 0.0, 0.0, 1.0e-15);
    Ok(())
}

#[test]
fn radial_grid_matches_feff_setri_reference() -> Result<(), ScreenError> {
    let grid = screen_radial_grid(0.05, 8.8, 5)?;

    assert_close(grid[0], 0.000_150_733_075_095_476_5, 1.0e-15);
    assert_close(grid[1], 0.000_158_461_325_115_751_26, 1.0e-15);
    assert_close(grid[2], 0.000_166_585_810_987_633_24, 1.0e-15);
    assert_close(grid[3], 0.000_175_126_848_157_658_42, 1.0e-15);
    assert_close(grid[4], 0.000_184_105_793_667_578_87, 1.0e-15);
    assert_eq!(screen_radial_index_1based(8.8, 0.05, grid[2])?, 3);
    assert_eq!(screen_radial_index_1based(8.8, 0.05, 1.0)?, 177);
    assert_eq!(screen_radial_index_1based(0.0, 1.0, 0.01)?, -3);
    Ok(())
}

#[test]
fn radial_bounds_match_feff_screensub_reference() -> Result<(), ScreenError> {
    let bounds = screen_radial_bounds(ScreenRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 1.2,
        tail_extension: 3,
        radial_capacity: 251,
        response_capacity: 251,
    })?;

    assert_eq!(bounds.muffin_tin_index_1based, 164);
    assert_eq!(bounds.muffin_tin_next_index_1based, 165);
    assert_eq!(bounds.norman_index_1based, 181);
    assert_eq!(bounds.active_count, 190);
    Ok(())
}

#[test]
fn radial_bounds_clamp_ilast_to_response_capacity() -> Result<(), ScreenError> {
    let bounds = screen_radial_bounds(ScreenRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 1.2,
        tail_extension: 3,
        radial_capacity: 251,
        response_capacity: 185,
    })?;

    assert_eq!(bounds.norman_index_1based, 181);
    assert_eq!(bounds.active_count, 185);
    Ok(())
}

#[test]
fn getph_radial_bounds_match_feff_reference() -> Result<(), ScreenError> {
    let bounds = screen_getph_radial_bounds(ScreenGetphRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 1.2,
        radial_capacity: 251,
    })?;

    assert_eq!(bounds.muffin_tin_index_1based, 164);
    assert_eq!(bounds.norman_index_1based, 181);
    assert_eq!(bounds.active_count, 187);
    Ok(())
}

#[test]
fn getph_radial_bounds_clamp_ilast_to_radial_capacity() -> Result<(), ScreenError> {
    let bounds = screen_getph_radial_bounds(ScreenGetphRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 38.474_666_049_032_14,
        radial_capacity: 251,
    })?;

    assert_eq!(bounds.muffin_tin_index_1based, 164);
    assert_eq!(bounds.norman_index_1based, 251);
    assert_eq!(bounds.active_count, 251);
    Ok(())
}

#[test]
fn energy_state_matches_feff_per_energy_reference() -> Result<(), ScreenError> {
    let state = screen_energy_state(ScreenEnergyStateInput {
        energy: Complex::new(0.4, 0.5),
        reference_energy: Complex::new(0.1, 0.05),
        muffin_tin_radius: 1.7,
        exchange_selector: 7,
    })?;

    assert_complex_close(state.kinetic_energy, 0.3, 0.45, 1.0e-15);
    assert_complex_close(
        state.wave_number,
        0.916_970_019_128_716_1,
        0.490_754_528_006_756_5,
        1.0e-14,
    );
    assert_complex32_close(
        state.fms_wave_number,
        0.916_970_014_572_143_6,
        0.490_754_514_932_632_45,
        1.0e-6,
    );
    assert_complex_close(
        state.muffin_tin_argument,
        1.558_849_032_518_817_3,
        0.834_282_697_611_486,
        1.0e-14,
    );
    assert_eq!(state.dirac_cycle_count, 3);

    let low_exchange = screen_energy_state(ScreenEnergyStateInput {
        exchange_selector: 14,
        ..ScreenEnergyStateInput {
            energy: Complex::new(0.4, 0.5),
            reference_energy: Complex::new(0.1, 0.05),
            muffin_tin_radius: 1.7,
            exchange_selector: 7,
        }
    })?;
    assert_eq!(low_exchange.dirac_cycle_count, 0);
    Ok(())
}

#[test]
fn getph_lmax_matches_feff_light_element_overrides() -> Result<(), ScreenError> {
    assert_eq!(screen_getph_lmax(29, 5, 3)?, 3);
    assert_eq!(screen_getph_lmax(8, 2, 3)?, 2);
    assert_eq!(screen_getph_lmax(4, 5, 10)?, 2);
    assert_eq!(screen_getph_lmax(2, 5, 10)?, 1);
    assert_eq!(screen_getph_lmax(1, 0, 0)?, 1);
    Ok(())
}
