//! Parse an embedded FEFF `xmu.dat` spectrum into typed `ndarray` columns.
//!
//! `xmu.dat` is FEFF's final normalized absorption spectrum: a comment-rich
//! header followed by six numeric columns (photon energy, edge-relative
//! energy, photoelectron wave number, normalized total absorption `mu`,
//! normalized atomic background `mu0`, and fine structure `chi = mu - mu0`).
//! See [`refeff_io::XmuDatData`] for the full column documentation.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p refeff-io --example read_spectrum
//! ```

use refeff_io::parse_xmu_dat;

/// A three-point sample in FEFF's compact `xmu.dat` layout, matching the
/// shape FEFF10 writes for a Cu K-edge EXAFS run.
const CU_XMU_DAT: &str = r#"# # Cu                                                           FEFF 10.0.0
#  S02=1.000  Temp=   0.00  Debye_temp=   0.00  Global_sig2= 0.00000
#     0/   0 paths used
#  xsedge+ 50, used to normalize mu           1.2667E-04
#  -----------------------------------------------------------------------
#  omega    e    k    mu    mu0     chi     @#
   11076.317    -40.000  -3.016  9.93209E-03  9.60242E-03  3.29662E-04
   11076.888    -39.429  -2.991  8.72601E-03  8.38540E-03  3.40613E-04
   11077.459    -38.858  -2.965  7.66539E-03  7.31069E-03  3.54700E-04
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spectrum = parse_xmu_dat(CU_XMU_DAT)?;

    println!("points: {}", spectrum.photon_energy_ev.len());
    println!(
        "normalization (xsedge+50, used to convert normalized mu to an absolute \
         cross section in square Angstrom): {:?}",
        spectrum.normalization
    );

    println!("omega(eV)     e(eV)      k(1/Ang)   mu           mu0          chi");
    for i in 0..spectrum.photon_energy_ev.len() {
        println!(
            "{:>10.3}  {:>9.3}  {:>9.3}  {:>11.5e}  {:>11.5e}  {:>11.5e}",
            spectrum.photon_energy_ev[i],
            spectrum.relative_energy_ev[i],
            spectrum.wave_number[i],
            spectrum.mu[i],
            spectrum.mu0[i],
            spectrum.chi[i],
        );
    }

    if let Some(absolute) = spectrum.absolute_mu() {
        println!(
            "absolute mu at first point (square Angstrom): {:.6e}",
            absolute[0]
        );
    }

    Ok(())
}
