//! Run a caller-supplied EXAFS input and read its final absorption spectrum.
//! cargo run --release -p refeff --example spectrum -- /absolute/feff.inp /absolute/output
use refeff::{FileRunRequest, Runner, io};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("provide input and output paths")?;
    let output = args.next().ok_or("provide an output directory")?;
    let report = Runner::new().run_files(FileRunRequest::new(input, &output))?;
    let spectrum = io::xmu_dat::read_xmu_dat(std::path::Path::new(&output).join("xmu.dat"))?;
    println!(
        "{} completed stages; {} spectrum points",
        report.stages.len(),
        spectrum.photon_energy_ev.len()
    );
    if let Some(point) = spectrum.point(0) {
        println!(
            "First point: {} eV, μ={}",
            point.photon_energy.value(),
            point.mu
        );
    }
    Ok(())
}
