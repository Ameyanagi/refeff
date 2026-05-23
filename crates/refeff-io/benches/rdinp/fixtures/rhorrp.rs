use super::*;

pub(crate) fn rhozzp_dat_bench_data() -> RhozzpDatData {
    let point_count = 1000;
    RhozzpDatData {
        header_lines: Vec::new(),
        z_prime: Array1::from_shape_fn(point_count, |index| 0.01 + 10.0 * index as f64 / 999.0),
        density: Array1::from_shape_fn(point_count, |index| {
            let z_prime = 0.01 + 10.0 * index as f64 / 999.0;
            3.7 * (-4.0 * z_prime).exp() - 0.55 * (-0.9 * z_prime).exp()
        }),
    }
}

pub(crate) fn rhorrp_density_bench_data() -> RhorrpDensityTextData {
    let point_count = 10_000;
    RhorrpDensityTextData {
        points_angstrom: Array2::from_shape_fn((point_count, 3), |(point, axis)| {
            let point = point as f64;
            match axis {
                0 => 0.01 * point,
                1 => (0.001 * point).sin(),
                _ => (0.0015 * point).cos(),
            }
        }),
        density_per_angstrom3: Array1::from_shape_fn(point_count, |point| {
            let x = point as f64 / point_count as f64;
            0.25 * (-2.5 * x).exp()
        }),
        nearest: Some(RhorrpNearestAtomColumns {
            displacement_bohr: Array2::from_shape_fn((point_count, 3), |(point, axis)| {
                0.001 * (point % 97) as f64 - 0.02 * axis as f64
            }),
            atom_indices: Array1::from_shape_fn(point_count, |point| point % 64),
            potential_indices: Array1::from_shape_fn(point_count, |point| point % 8),
        }),
    }
}

pub(crate) fn rhorrp_density_bin_bench_data() -> RhorrpDensityBinData {
    let points_per_axis = vec![100, 50, 20];
    let point_count = points_per_axis.iter().product::<usize>();
    RhorrpDensityBinData {
        origin_angstrom: [0.1, -0.2, 0.3],
        axes_angstrom: ndarray::arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]]),
        points_per_axis,
        density_per_angstrom3: Array1::from_shape_fn(point_count, |point| {
            let scaled = point as f64 / point_count as f64;
            0.15 * (-3.0 * scaled).exp() + 0.01 * (13.0 * scaled).sin()
        }),
    }
}

pub(crate) fn jzzp_dat_bench_data() -> JzzpDatData {
    let nz = 64;
    let nzp = 120;
    JzzpDatData {
        ns: 32,
        nphi: 32,
        nz,
        nzp,
        smax: 4.5,
        phimax: std::f64::consts::PI,
        zmax: 4.5,
        zpmax: 10.0,
        values: Array2::from_shape_fn((nz, nzp), |(z, zp)| {
            let z_coord = z as f64 / (nz - 1) as f64;
            let zp_coord = zp as f64 / (nzp - 1) as f64;
            (1.0 + z_coord).exp() * (-2.0 * zp_coord).exp()
        }),
    }
}
