use super::{support::*, *};

#[test]
fn xsph_phase_energy_mesh_84_matches_feff_phmesh2_reference() -> Result<(), XsphError> {
    let exafs = xsph_phase_energy_mesh_84(phase_mesh84_input(0))?;
    assert_eq!(exafs.energies.len(), 79);
    assert_eq!(exafs.horizontal_count, 57);
    assert_eq!(exafs.extension_count, 0);
    assert_eq!(exafs.zero_index, 0);
    assert_close(exafs.xloss, 0.05);
    assert_complex_close(exafs.energies[0], Complex::new(-0.4, 0.05));
    assert_complex_close(
        exafs.energies[56],
        Complex::new(44.964_626_859_191_69, 0.05),
    );
    assert_complex_close(
        exafs.energies[57],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        exafs.energies[78],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let xanes = xsph_phase_energy_mesh_84(phase_mesh84_input(1))?;
    assert_eq!(xanes.energies.len(), 69);
    assert_eq!(xanes.horizontal_count, 47);
    assert_eq!(xanes.extension_count, 0);
    assert_eq!(xanes.zero_index, 10);
    assert_complex_close(
        xanes.energies[0],
        Complex::new(-11.741_156_714_797_922, 0.05),
    );
    assert_complex_close(xanes.energies[10], Complex::new(-0.4, 0.05));
    assert_complex_close(
        xanes.energies[46],
        Complex::new(44.964_626_859_191_69, 0.05),
    );
    assert_complex_close(
        xanes.energies[47],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    let no_fms = xsph_phase_energy_mesh_84(phase_mesh84_input(-1))?;
    assert_eq!(no_fms.energies.len(), 89);
    assert_eq!(no_fms.horizontal_count, 67);
    assert_eq!(no_fms.extension_count, 0);
    assert_eq!(no_fms.zero_index, 10);
    assert_complex_close(
        no_fms.energies[0],
        Complex::new(-11.741_156_714_797_922, 0.05),
    );
    assert_complex_close(no_fms.energies[10], Complex::new(-0.4, 0.05));
    assert_complex_close(
        no_fms.energies[11],
        Complex::new(-3.985_998_571_957_039_5e-1, 0.05),
    );
    assert_complex_close(
        no_fms.energies[66],
        Complex::new(44.964_626_859_191_69, 0.05),
    );
    assert_complex_close(
        no_fms.energies[67],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        no_fms.energies[88],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let xes_input = XsphPhaseEnergyMesh84Input {
        spectroscopy: 2,
        max_wave_number: -5.0,
        wave_number_step: 10.0,
        xanes_energy_step: 0.25,
        ..phase_mesh84_input(2)
    };
    let xes = xsph_phase_energy_mesh_84(xes_input)?;
    assert_eq!(xes.energies.len(), 27);
    assert_eq!(xes.horizontal_count, 5);
    assert_eq!(xes.extension_count, 0);
    assert_eq!(xes.zero_index, 0);
    assert_complex_close(xes.energies[0], Complex::new(-0.4, 0.05));
    assert_complex_close(
        xes.energies[1],
        Complex::new(-9.723_062_142_400_252e-2, 0.05),
    );
    assert_complex_close(
        xes.energies[4],
        Complex::new(6.527_693_785_759_748e-1, 0.05),
    );
    assert_complex_close(
        xes.energies[5],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    let danes = xsph_phase_energy_mesh_84(phase_mesh84_input(3))?;
    assert_eq!(danes.energies.len(), 119);
    assert_eq!(danes.horizontal_count, 47);
    assert_eq!(danes.extension_count, 50);
    assert_eq!(danes.zero_index, 10);
    assert_complex_close(
        danes.energies[69],
        Complex::new(47.449_880_336_817_16, 2.0e-8),
    );
    assert_complex_close(
        danes.energies[118],
        Complex::new(60_495.181_164_572_314, 2.0e-8),
    );

    let no_fms_danes = xsph_phase_energy_mesh_84(phase_mesh84_input(-3))?;
    assert_eq!(no_fms_danes.energies.len(), 119);
    assert_eq!(no_fms_danes.horizontal_count, 67);
    assert_eq!(no_fms_danes.extension_count, 30);
    assert_eq!(no_fms_danes.zero_index, 10);
    assert_complex_close(
        no_fms_danes.energies[89],
        Complex::new(49.865_126_674_227_82, 2.0e-8),
    );
    assert_complex_close(
        no_fms_danes.energies[118],
        Complex::new(54_977.881_757_676_99, 2.0e-8),
    );

    let fprime_input = XsphPhaseEnergyMesh84Input {
        spectroscopy: 4,
        max_wave_number: -5.0,
        wave_number_step: 10.0,
        xanes_energy_step: 0.25,
        ..phase_mesh84_input(4)
    };
    let fprime = xsph_phase_energy_mesh_84(fprime_input)?;
    assert_eq!(fprime.energies.len(), 105);
    assert_eq!(fprime.horizontal_count, 5);
    assert_eq!(fprime.extension_count, 100);
    assert_eq!(fprime.zero_index, 0);
    assert_complex_close(
        fprime.energies[0],
        Complex::new(-9.347_230_621_424_002, 0.0),
    );
    assert_complex_close(fprime.energies[5], Complex::new(-0.4, 0.0));
    assert_complex_close(fprime.energies[104], Complex::new(171.0, 0.0));

    Ok(())
}

#[test]
fn xsph_user_phase_energy_mesh_matches_feff_phmesh2_reference() -> Result<(), XsphError> {
    let points = arr1(&[
        Complex::new(-5.0, 0.2),
        Complex::new(0.0004, 0.0),
        Complex::new(12.0, -0.1),
    ]);
    let records = [
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::Energy,
            minimum: XsphPhaseUserGridMinimum::Value(-2.0),
            maximum: 2.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::WaveNumber,
            minimum: XsphPhaseUserGridMinimum::Last,
            maximum: 3.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::User(points.view()),
    ];
    let mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
        spectroscopy: 1,
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        records: &records,
        capacity: 120,
    })?;
    assert_eq!(mesh.energies.len(), 31);
    assert_eq!(mesh.horizontal_count, 9);
    assert_eq!(mesh.extension_count, 0);
    assert_eq!(mesh.zero_index, 3);
    assert_complex_close(
        mesh.energies[0],
        Complex::new(-5.837_465_450_137_141e-1, 0.05),
    );
    assert_complex_close(mesh.energies[3], Complex::new(-0.4, 0.05));
    assert_complex_close(
        mesh.energies[6],
        Complex::new(1.640_061_233_229_339_6e-2, 0.05),
    );
    assert_complex_close(
        mesh.energies[8],
        Complex::new(6.393_311_675_183_092e-1, 0.05),
    );
    assert_complex_close(
        mesh.energies[9],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        mesh.energies[30],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let exp_points = arr1(&[Complex::new(-1.0, 0.0), Complex::new(0.002, 0.0)]);
    let exp_records = [
        XsphPhaseUserGridRecord::User(exp_points.view()),
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::Exponential,
            minimum: XsphPhaseUserGridMinimum::Last,
            maximum: 20.0,
            step: 0.5,
        }),
    ];
    let exp_mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
        spectroscopy: 1,
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        records: &exp_records,
        capacity: 120,
    })?;
    assert_eq!(exp_mesh.energies.len(), 54);
    assert_eq!(exp_mesh.horizontal_count, 32);
    assert_eq!(exp_mesh.extension_count, 0);
    assert_eq!(exp_mesh.zero_index, 1);
    assert_complex_close(
        exp_mesh.energies[0],
        Complex::new(-4.367_493_090_027_428_4e-1, 0.05),
    );
    assert_complex_close(exp_mesh.energies[1], Complex::new(-0.4, 0.05));
    assert_complex_close(
        exp_mesh.energies[31],
        Complex::new(3.222_548_489_185_893_5e-1, 0.05),
    );
    assert_complex_close(
        exp_mesh.energies[32],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );
    assert_complex_close(
        exp_mesh.energies[53],
        Complex::new(-0.4, 9.845_207_096_763_882e-1),
    );

    let danes_mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
        spectroscopy: -3,
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        records: &exp_records,
        capacity: 120,
    })?;
    assert_eq!(danes_mesh.energies.len(), 119);
    assert_eq!(danes_mesh.horizontal_count, 32);
    assert_eq!(danes_mesh.extension_count, 65);
    assert_eq!(danes_mesh.zero_index, 1);
    assert_complex_close(
        danes_mesh.energies[54],
        Complex::new(3.532_758_355_442_124e-1, 2.0e-8),
    );
    assert_complex_close(
        danes_mesh.energies[118],
        Complex::new(58_023.774_523_482_745, 2.0e-8),
    );

    Ok(())
}

#[test]
fn xsph_thermal_phase_energy_mesh_matches_feff_phmesh2t_reference() -> Result<(), XsphError> {
    let thermal = xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        core_valence_separation: -1.5,
        electronic_temperature: 5.0,
        user_records: None,
        capacity: 240,
    })?;
    assert_eq!(thermal.energies.len(), 72);
    assert_eq!(thermal.horizontal_count, 30);
    assert_eq!(thermal.pole_count, 1);
    assert_eq!(thermal.zero_index, 5);
    assert_complex_close(
        thermal.energies[0],
        Complex::new(-1.25, 1.154_513_591_875_180_8),
    );
    assert_complex_close(
        thermal.energies[5],
        Complex::new(0.0, 1.154_513_591_875_180_8),
    );
    assert_complex_close(thermal.energies[30], Complex::new(-1.25, 0.05));
    assert_complex_close(
        thermal.energies[60],
        Complex::new(-1.5, 1.154_513_591_875_180_7e-2),
    );
    assert_complex_close(
        thermal.energies[70],
        Complex::new(-0.4, 5.772_567_959_375_904e-1),
    );
    assert_complex_close(
        thermal.energies[71],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    let points = arr1(&[
        Complex::new(-5.0, 0.2),
        Complex::new(0.0004, 0.0),
        Complex::new(12.0, -0.1),
    ]);
    let records = [
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::Energy,
            minimum: XsphPhaseUserGridMinimum::Value(-2.0),
            maximum: 2.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::WaveNumber,
            minimum: XsphPhaseUserGridMinimum::Last,
            maximum: 3.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::User(points.view()),
    ];
    let user = xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
        edge: -0.4,
        constant_imaginary: 0.01,
        core_hole_broadening: 0.08,
        core_valence_separation: -1.5,
        electronic_temperature: 5.0,
        user_records: Some(&records),
        capacity: 240,
    })?;
    assert_eq!(user.energies.len(), 30);
    assert_eq!(user.horizontal_count, 9);
    assert_eq!(user.pole_count, 1);
    assert_eq!(user.zero_index, 3);
    assert_complex_close(
        user.energies[0],
        Complex::new(-5.837_465_450_137_141e-1, 1.154_513_591_875_180_8),
    );
    assert_complex_close(
        user.energies[8],
        Complex::new(6.393_311_675_183_092e-1, 1.154_513_591_875_180_8),
    );
    assert_complex_close(
        user.energies[18],
        Complex::new(-1.5, 1.154_513_591_875_180_7e-2),
    );
    assert_complex_close(
        user.energies[28],
        Complex::new(-0.4, 5.772_567_959_375_904e-1),
    );
    assert_complex_close(
        user.energies[29],
        Complex::new(-0.4, 1.837_465_409_066_587_8e-4),
    );

    Ok(())
}

#[test]
fn xsph_phase_mesh_primitives_match_feff_phmesh2_reference() -> Result<(), XsphError> {
    let even = xsph_even_energy_mesh(-0.2, 0.35, 0.11, 4)?;
    assert_eq!(even.len(), 4);
    for (&actual, expected) in even.iter().zip([-0.2, -0.09, 0.02, 0.13]) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let k_mesh = xsph_k_energy_mesh(-1.2, -0.2, 0.25, 32)?;
    let expected_k = [-0.72, -0.45125, -0.245, -0.10125, -0.02];
    assert_eq!(k_mesh.len(), expected_k.len());
    for (&actual, expected) in k_mesh.iter().zip(expected_k) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let exp_mesh = xsph_exponential_energy_mesh(0.02, 0.5, 0.4, 32)?;
    let expected_exp = [
        2e-2,
        2.983_649_395_282_541e-2,
        4.451_081_856_984_936e-2,
        6.640_233_845_473_097e-2,
        9.906_064_848_790_23e-2,
        1.477_811_219_786_13e-1,
        2.204_635_276_128_321e-1,
        3.288_929_354_219_411e-1,
        4.906_506_039_421_871e-1,
    ];
    assert_eq!(exp_mesh.len(), expected_exp.len());
    for (&actual, expected) in exp_mesh.iter().zip(expected_exp) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let vertical = xsph_vertical_energy_mesh_84(0.05, 32)?;
    let expected_vertical = [
        1.837_465_409_066_59e-4,
        3.674_930_818_133_18e-4,
        4.927_048_004_095_16e-4,
        7.350_291_898_973_28e-4,
        1.096_534_698_976_09e-3,
        1.635_837_545_753_17e-3,
        2.440_382_852_083_45e-3,
        3.640_623_410_438_34e-3,
        5.431_171_918_502_91e-3,
        8.102_356_405_158_36e-3,
        1.208_729_539_430_72e-2,
        1.803_212_579_691_30e-2,
        2.690_077_061_480_91e-2,
        4.013_123_398_875_48e-2,
        5.986_876_601_124_52e-2,
        8.931_370_375_288_18e-2,
        1.332_403_890_963_65e-1,
        1.987_713_031_772_90e-1,
        2.965_319_392_622_22e-1,
        4.423_736_706_308_43e-1,
        6.599_439_674_333_17e-1,
        9.845_207_096_763_88e-1,
    ];
    assert_eq!(vertical.len(), expected_vertical.len());
    for (&actual, expected) in vertical.iter().zip(expected_vertical) {
        assert_complex_close(actual, Complex::new(0.0, expected));
    }
    let clipped_vertical = xsph_vertical_energy_mesh_84(0.05, 4)?;
    assert_eq!(clipped_vertical.len(), 4);
    for (&actual, expected) in clipped_vertical.iter().zip(expected_vertical) {
        assert_complex_close(actual, Complex::new(0.0, expected));
    }

    let exafs84 = xsph_exafs_energy_grid_84(18.0 * XSPH_BOHR_ANGSTROM, 160)?;
    let expected_exafs84 = [
        0.0,
        1.400_142_804_296_04e-3,
        5.600_571_217_184_16e-3,
        1.260_128_523_866_44e-2,
        2.240_228_486_873_66e-2,
        3.500_357_010_740_10e-2,
        5.040_514_095_465_74e-2,
        6.860_699_741_050_60e-2,
        8.960_913_947_494_66e-2,
        1.134_115_671_479_79e-1,
        1.400_142_804_296_04e-1,
        1.694_172_793_198_21e-1,
        2.016_205_638_186_30e-1,
        2.366_241_339_260_31e-1,
        2.744_279_896_420_24e-1,
        3.150_321_309_666_09e-1,
        3.584_365_578_997_86e-1,
        4.046_412_704_415_56e-1,
        4.536_462_685_919_17e-1,
        5.054_515_523_508_70e-1,
        5.600_571_217_184_16e-1,
        6.776_691_172_792_83e-1,
        8.064_822_552_745_19e-1,
        9.464_965_357_041_23e-1,
        1.097_711_958_568_10,
        1.260_128_523_866_44,
        1.433_746_231_599_14,
        1.618_565_081_766_22,
        1.814_585_074_367_67,
        2.021_806_209_403_48,
        2.240_228_486_873_66,
        2.469_851_906_778_21,
        2.710_676_469_117_13,
        2.962_702_173_890_42,
        3.225_929_021_098_08,
        3.500_357_010_740_10,
        3.785_986_142_816_49,
        4.082_816_417_327_25,
        4.390_847_834_272_38,
        4.710_080_393_651_88,
        5.040_514_095_465_74,
        5.915_603_348_150_77,
        6.860_699_741_050_60,
        7.875_803_274_165_22,
        8.960_913_947_494_65,
        10.116_031_761_038_9,
        11.341_156_714_797_9,
        12.636_288_808_771_8,
        14.001_428_042_960_4,
        16.941_727_931_982_1,
        20.162_056_381_863_0,
        23.662_413_392_603_1,
        27.442_798_964_202_4,
        31.503_213_096_660_9,
        35.843_655_789_978_6,
        40.464_127_044_155_6,
        45.364_626_859_191_7,
    ];
    assert_eq!(exafs84.len(), expected_exafs84.len());
    for (&actual, expected) in exafs84.iter().zip(expected_exafs84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    let clipped_exafs84 = xsph_exafs_energy_grid_84(18.0 * XSPH_BOHR_ANGSTROM, 45)?;
    assert_eq!(clipped_exafs84.len(), 45);
    for (&actual, expected) in clipped_exafs84.iter().zip(expected_exafs84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let xanes84 =
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.5 * XSPH_BOHR_ANGSTROM, 0.02, 80)?;
    let expected_xanes84 = [
        -11.341_156_714_797_9,
        -8.960_913_947_494_65,
        -6.860_699_741_050_60,
        -5.040_514_095_465_74,
        -3.500_357_010_740_10,
        -2.240_228_486_873_66,
        -1.260_128_523_866_44,
        -5.600_571_217_184_16e-1,
        -1.400_142_804_296_04e-1,
        -2.0e-2,
        0.0,
        3.500_357_010_740_10e-2,
        1.400_142_804_296_04e-1,
        3.150_321_309_666_09e-1,
        5.600_571_217_184_16e-1,
        8.750_892_526_850_25e-1,
        1.260_128_523_866_44,
        1.715_174_935_262_65,
        2.240_228_486_873_66,
    ];
    assert_eq!(xanes84.zero_index, 10);
    assert_eq!(xanes84.energies.len(), expected_xanes84.len());
    for (&actual, expected) in xanes84.energies.iter().zip(expected_xanes84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    let clipped_xanes84 =
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.5 * XSPH_BOHR_ANGSTROM, 0.02, 15)?;
    assert_eq!(clipped_xanes84.zero_index, 10);
    assert_eq!(clipped_xanes84.energies.len(), 13);
    for (&actual, expected) in clipped_xanes84.energies.iter().zip(expected_xanes84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let xes84 = xsph_xes_energy_grid_84(-5.0, 10.0, 0.25, -0.4, 64)?;
    let expected_xes84 = [
        0.0,
        3.027_693_785_759_975e-1,
        5.527_693_785_759_975e-1,
        8.027_693_785_759_975e-1,
        1.052_769_378_575_997_5,
    ];
    assert_eq!(xes84.zero_index, 0);
    assert_eq!(xes84.energies.len(), expected_xes84.len());
    for (&actual, expected) in xes84.energies.iter().zip(expected_xes84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }

    let fprime84 = xsph_fprime_energy_grid_84(-5.0, 10.0, 0.25, 9.0, -0.4, 32)?;
    let expected_fprime84 = [
        -9.347_230_621_424_00,
        -9.097_230_621_424_00,
        -8.847_230_621_424_00,
        -8.597_230_621_424_00,
        -8.347_230_621_424_00,
        -4.0e-1,
        -2.897_520_729_917_72e-1,
        -1.795_041_459_835_43e-1,
        -6.925_621_897_531_47e-2,
        4.099_170_803_291_38e-2,
        1.512_396_350_411_42e-1,
        2.614_875_620_493_71e-1,
        3.717_354_890_575_99e-1,
        5.133_096_128_907_97e-1,
        7.088_017_325_278_11e-1,
        9.787_463_227_214_28e-1,
        1.351_498_339_069_21,
        1.866_211_620_012_10,
        2.576_951_602_520_49,
        3.558_374_350_755_49,
        4.913_568_422_367_72,
        6.784_883_281_367_88,
        9.368_881_673_088_11,
        12.936_986_557_362_0,
        17.863_991_351_936_7,
        24.667_428_199_534_5,
        34.061_929_497_811_9,
        47.034_292_822_459_8,
        64.947_134_056_249_1,
        89.682_016_439_421_3,
        123.837_089_804_069,
        171.0,
    ];
    assert_eq!(fprime84.regular_count, 5);
    assert_eq!(fprime84.kk_count, 27);
    assert_eq!(fprime84.energies.len(), expected_fprime84.len());
    for (&actual, expected) in fprime84.energies.iter().zip(expected_fprime84) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    let fprime_auto_step = xsph_fprime_energy_grid_84(-5.0, 10.0, 0.0, 9.0, -0.4, 140)?;
    assert_eq!(fprime_auto_step.regular_count, 100);
    assert_eq!(fprime_auto_step.kk_count, 40);
    assert_eq!(fprime_auto_step.energies.len(), 140);
    assert_complex_close(
        fprime_auto_step.energies[0],
        Complex::new(expected_fprime84[0], 0.0),
    );
    assert_complex_close(fprime_auto_step.energies[139], Complex::new(171.0, 0.0));

    let reverse_input = arr1(&[
        Complex::new(1.0, 0.2),
        Complex::new(2.0, -0.1),
        Complex::new(-0.5, 0.0),
    ]);
    let reversed = xsph_reverse_energy_grid(reverse_input.view(), 0.25)?;
    let expected_reverse = [
        Complex::new(0.75, 0.0),
        Complex::new(-1.75, 0.1),
        Complex::new(-0.75, -0.2),
    ];
    for (&actual, expected) in reversed.iter().zip(expected_reverse) {
        assert_complex_close(actual, expected);
    }

    let sort_input = arr1(&[
        Complex::new(0.002, 9.0),
        Complex::new(-0.004, 8.0),
        Complex::new(0.0004, 7.0),
        Complex::new(0.0012, 6.0),
        Complex::new(-0.0036, 5.0),
        Complex::new(0.25, 4.0),
    ]);
    let sorted = xsph_sort_energy_grid(sort_input.view())?;
    assert_eq!(sorted.zero_index, 1);
    let expected_sorted = [-0.004, 0.0, 0.002, 0.25];
    assert_eq!(sorted.energies.len(), expected_sorted.len());
    for (&actual, expected) in sorted.energies.iter().zip(expected_sorted) {
        assert_complex_close(actual, Complex::new(expected, 0.0));
    }
    Ok(())
}

#[test]
fn xsph_phase_mesh_primitives_reject_invalid_inputs() {
    assert_eq!(
        xsph_even_energy_mesh(0.0, 1.0, 0.1, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_even_energy_mesh(0.0, 1.0, 0.0, 4),
        Err(XsphError::InvalidPhaseMeshStep {
            name: "energy_step",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_exponential_energy_mesh(0.0, 1.0, 0.4, 4),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "min_energy",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_vertical_energy_mesh_84(0.05, 1),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 1 })
    );
    assert_eq!(
        xsph_vertical_energy_mesh_84(0.0, 4),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "xloss",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_exafs_energy_grid_84(18.0 * XSPH_BOHR_ANGSTROM, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_exafs_energy_grid_84(0.0, 4),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "max_wave_number",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.5 * XSPH_BOHR_ANGSTROM, 0.02, 10),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 10 })
    );
    assert_eq!(
        xsph_xanes_energy_grid_84(4.0 * XSPH_BOHR_ANGSTROM, 0.0, 0.02, 80),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "wave_number_step",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_fprime_energy_grid_84(-5.0, 10.0, 0.25, 9.0, -0.4, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_xes_energy_grid_84(-5.0, 10.0, 0.25, -0.4, 0),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert!(matches!(
        xsph_fprime_energy_grid_84(-5.0, 10.0, Real::NAN, 9.0, -0.4, 32),
        Err(XsphError::NonFiniteScalar {
            name: "energy_step",
            value,
        }) if value.is_nan()
    ));
    assert_eq!(
        xsph_phase_energy_mesh_84(XsphPhaseEnergyMesh84Input {
            capacity: 0,
            ..phase_mesh84_input(0)
        }),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 0 })
    );
    assert_eq!(
        xsph_phase_energy_mesh_84(phase_mesh84_input(6)),
        Err(XsphError::UnsupportedPhaseMeshSpectroscopy { spectroscopy: 6 })
    );
    assert_eq!(
        xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: 1,
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            records: &[],
            capacity: 120,
        }),
        Err(XsphError::EmptyPhaseGridRecords)
    );
    let too_many = [XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
        kind: XsphPhaseUserGridKind::Energy,
        minimum: XsphPhaseUserGridMinimum::Value(0.0),
        maximum: 1.0,
        step: 0.1,
    }); 11];
    assert_eq!(
        xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: 1,
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            records: &too_many,
            capacity: 120,
        }),
        Err(XsphError::TooManyPhaseGridRecords { count: 11, max: 10 })
    );
    assert_eq!(
        xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: 6,
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            records: &too_many[..1],
            capacity: 120,
        }),
        Err(XsphError::UnsupportedPhaseMeshSpectroscopy { spectroscopy: 6 })
    );
    assert_eq!(
        xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            core_valence_separation: -1.5,
            electronic_temperature: 0.0,
            user_records: None,
            capacity: 240,
        }),
        Err(XsphError::InvalidPhaseMeshEndpoint {
            name: "electronic_temperature",
            value: 0.0,
        })
    );
    assert_eq!(
        xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
            edge: -0.4,
            constant_imaginary: 0.01,
            core_hole_broadening: 0.08,
            core_valence_separation: -1.5,
            electronic_temperature: 5.0,
            user_records: None,
            capacity: 4,
        }),
        Err(XsphError::InvalidPhaseMeshCapacity { capacity: 4 })
    );
    let descending = xsph_even_energy_mesh(1.0, 0.0, 0.1, 4);
    assert_eq!(descending, Ok(Array1::zeros(0)));

    let empty = Array1::<Complex>::zeros(0);
    assert_eq!(
        xsph_sort_energy_grid(empty.view()),
        Err(XsphError::EmptyPhaseMesh)
    );

    let bad = arr1(&[Complex::new(0.0, 0.0), Complex::new(Real::NAN, 0.0)]);
    assert!(matches!(
        xsph_sort_energy_grid(bad.view()),
        Err(XsphError::NonFiniteComplex {
            name: "energies",
            index: 1,
            ..
        })
    ));
}
