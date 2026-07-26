//! FEFF `atomNN.dat` diagnostic output.
//!
//! ATOM writes these files when `pot.inp` requests `ipr1 >= 3`. At print
//! level five the file includes the `tabrat` orbital binding energies, radial
//! moments, and same-kappa overlaps used by the upstream HIGHZ harness.

use std::fmt::Write as _;

use refeff_core::{AtomicTabulation, AtomicTotalEnergy};

use crate::error::Result;

/// Typed inputs for one FEFF-compatible `atomNN.dat` file.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomDatData {
    /// Zero-based unique-potential index used in the filename and heading.
    pub potential_index: usize,
    /// FEFF `ipr1` print level.
    pub print_level: i32,
    /// Number of SCF orbital iterations requested by ATOM.
    pub max_orbital_iterations: usize,
    /// Orbital-energy convergence target.
    pub energy_precision: f64,
    /// Wavefunction convergence target.
    pub wavefunction_precision: f64,
    /// Number of radial integration points.
    pub radial_count: usize,
    /// First logarithmic-grid radius.
    pub first_radius: f64,
    /// Logarithmic radial-grid step.
    pub radial_step: f64,
    /// Primary wavefunction matching precision.
    pub matching_precision: f64,
    /// Maximum matching attempts.
    pub matching_attempts: usize,
    /// Whether the finite-nucleus branch was used.
    pub finite_nucleus: bool,
    /// FEFF `etotal` contributions and total.
    pub total_energy: AtomicTotalEnergy,
    /// FEFF `tabrat` table, present at print level five and above.
    pub tabulation: Option<AtomicTabulation>,
}

/// Render canonical `atomNN.dat` text.
pub fn atom_dat_string(data: &AtomDatData) -> Result<String> {
    let mut output = String::new();
    writeln!(output, "  free atom {:12}", data.potential_index)?;
    if data.print_level >= 5 {
        writeln!(
            output,
            "     number of iterations{:4}\n\n     precision of the energies {:8.2E}\n\n                       wave functions   {:8.2E}\n",
            data.max_orbital_iterations, data.energy_precision, data.wavefunction_precision
        )?;
        writeln!(
            output,
            " the integration is made on {:3} points-the first is equal to {:7.4}\n and the step-size pas = {:7.4}\n",
            data.radial_count, data.first_radius, data.radial_step
        )?;
        writeln!(
            output,
            "matching of w.f. with precision {:8.2E} in {:3} attempts \n",
            data.matching_precision, data.matching_attempts
        )?;
        if data.finite_nucleus {
            writeln!(
                output,
                "0                              finite nucleus case used\n"
            )?;
        }
    }

    if data.print_level >= 5 {
        writeln!(output, "etot{:18.7E}", data.total_energy.total)?;
        writeln!(output, "coul{:18.7E}", data.total_energy.direct_coulomb)?;
        writeln!(output, "ech.{:18.7E}", data.total_energy.exchange_coulomb)?;
        writeln!(output, "mag.{:18.7E}", data.total_energy.magnetic_breit)?;
        writeln!(output, "ret.{:18.7E}", data.total_energy.retarded_breit)?;
    }
    writeln!(output, " Total energy:   {:20.14}", data.total_energy.total)?;

    if let Some(tabulation) = &data.tabulation {
        writeln!(
            output,
            " number of electrons nel and average values of r**n in a.u."
        )?;
        writeln!(
            output,
            "     nel     -E      n= 6         4         2         1        -1        -2        -3"
        )?;
        for orbital in &tabulation.orbitals {
            write!(
                output,
                "{}{}{:6.3}{:20.6E}",
                orbital.principal_quantum_number,
                orbital.orbital_label,
                orbital.occupation,
                orbital.binding_energy_ev
            )?;
            for moment in &orbital.moments {
                write!(output, "{:20.6E}", moment.value)?;
            }
            writeln!(output)?;
        }

        if !tabulation.overlaps.is_empty() {
            writeln!(output, "          overlap integrals")?;
            for overlap in &tabulation.overlaps {
                writeln!(
                    output,
                    "    {:>3}{:<2}{:>3}{:<2}{:14.7}",
                    overlap.left_principal_quantum_number,
                    overlap.left_orbital_label,
                    overlap.right_principal_quantum_number,
                    overlap.right_orbital_label,
                    overlap.value
                )?;
            }
        }
    }
    Ok(output)
}

/// Write one FEFF-compatible `atomNN.dat` file.
pub fn write_atom_dat(path: impl AsRef<std::path::Path>, data: &AtomDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, atom_dat_string(data)?)
        .map_err(|source| crate::error::IoError::io(path, source))
}

#[cfg(test)]
mod tests {
    use refeff_core::{
        AtomicTabulatedMoment, AtomicTabulatedOrbital, AtomicTabulation, AtomicTotalEnergy,
    };

    use super::*;

    #[test]
    fn highz_binding_energy_is_the_third_whitespace_field() -> Result<()> {
        let text = atom_dat_string(&AtomDatData {
            potential_index: 0,
            print_level: 5,
            max_orbital_iterations: 40,
            energy_precision: 5.0e-6,
            wavefunction_precision: 1.0e-5,
            radial_count: 251,
            first_radius: 1.0e-4,
            radial_step: 0.05,
            matching_precision: 1.0e-7,
            matching_attempts: 50,
            finite_nucleus: true,
            total_energy: AtomicTotalEnergy {
                total: -10.0,
                direct_coulomb: 1.0,
                exchange_coulomb: -0.5,
                magnetic_breit: 0.0,
                retarded_breit: 0.0,
            },
            tabulation: Some(AtomicTabulation {
                orbitals: vec![AtomicTabulatedOrbital {
                    principal_quantum_number: 1,
                    orbital_label: "s ",
                    occupation: 2.0,
                    binding_energy_ev: 116_443.3,
                    moments: vec![AtomicTabulatedMoment {
                        power: 2,
                        value: 1.5e-7,
                    }],
                }],
                overlaps: Vec::new(),
            }),
        })?;
        let row = text
            .lines()
            .find(|line| line.trim_start().starts_with("1s"))
            .expect("writer should contain a 1s row");
        assert_eq!(row.split_whitespace().nth(2), Some("1.164433E5"));
        Ok(())
    }
}
