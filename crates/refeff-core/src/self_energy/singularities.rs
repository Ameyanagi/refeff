use std::cmp::Ordering;

use super::*;

/// Port of FEFF `FndSng`: find real self-energy integrand singularities.
///
/// `limits` are the lower and upper integration limits. `dp_parameters`
/// corresponds to FEFF `DPPar(1:4)`, with `DPPar(1)` as `Wp/EFermi` and
/// `DPPar(3)` as `Energy/EFermi`. `complex_parameters` corresponds to
/// `CPar(1:2)`, where `CPar(1)` is `ck/kFermi`.
pub fn find_self_energy_singularities(
    limits: [Complex; 2],
    dp_parameters: [Real; 4],
    complex_parameters: [Complex; 2],
    function: SingularityFunction,
) -> Result<Vec<Real>, SelfEnergyError> {
    ensure_finite_complex("lower limit", limits[0])?;
    ensure_finite_complex("upper limit", limits[1])?;
    ensure_finite_complex("CPar(1)", complex_parameters[0])?;
    ensure_finite_complex("CPar(2)", complex_parameters[1])?;
    for (index, &value) in dp_parameters.iter().enumerate() {
        ensure_finite_real(dp_name(index), value)?;
    }

    let k = complex_parameters[0];
    let energy = dp_parameters[2];
    let plasma = dp_parameters[0];
    let lower = limits[0].re;
    let upper = limits[1].re;
    let mut singularities = Vec::new();

    let base_cubic = [
        4.0 * k,
        2.0 * (3.0 * k * k - energy - 2.0 / 3.0),
        4.0 * k * (k * k - energy),
        (k * k - energy) * (k * k - energy) - plasma * plasma,
    ];

    let plus_roots = cubic_zeros(base_cubic)?;
    singularities.extend(
        plus_roots
            .roots()
            .iter()
            .copied()
            .filter(|&root| accepts_cubic_root(k, energy, plasma, root, true, lower, upper))
            .map(|root| root.re),
    );

    let minus_roots = cubic_zeros([-base_cubic[0], base_cubic[1], -base_cubic[2], base_cubic[3]])?;
    singularities.extend(
        minus_roots
            .roots()
            .iter()
            .copied()
            .filter(|&root| accepts_cubic_root(k, energy, plasma, root, false, lower, upper))
            .map(|root| root.re),
    );

    if function == SingularityFunction::First {
        let roots = quadratic_zeros([
            Complex::new(1.0, 0.0),
            Complex::new(4.0 / 3.0, 0.0),
            Complex::new(plasma * plasma, 0.0),
        ])?;
        for &root in roots.roots() {
            if root.im.abs() <= SINGULARITY_TOLERANCE {
                let square_root = root.sqrt();
                let positive = square_root.re;
                let negative = -square_root.re;
                if positive >= lower && positive <= upper {
                    singularities.push(positive);
                }
                if negative >= lower && negative <= upper {
                    singularities.push(negative);
                }
            }
        }
    }

    sort_like_feff(&mut singularities);
    Ok(singularities)
}
fn accepts_cubic_root(
    k: Complex,
    energy: Real,
    plasma: Real,
    root: Complex,
    positive_branch: bool,
    lower: Real,
    upper: Real,
) -> bool {
    let radical = (root * root * root * root + root * root * (4.0 / 3.0) + plasma * plasma).sqrt();
    let test = if positive_branch {
        ((k + root) * (k + root) - energy + radical).norm()
    } else {
        ((k - root) * (k - root) - energy - radical).norm()
    };

    test < SINGULARITY_TOLERANCE
        && root.re >= lower
        && root.re <= upper
        && root.im.abs() <= SINGULARITY_TOLERANCE
}

fn sort_like_feff(values: &mut [Real]) {
    values.sort_by(|left, right| {
        if left < right {
            Ordering::Less
        } else if left > right {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}
