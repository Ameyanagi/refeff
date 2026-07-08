//! Evaluate FEFF's Wigner 3j coefficient kernel (a port of `MATH/cwig3j.f90`,
//! `wigner_3j` in `refeff_core::angular`).
//!
//! Units / argument conventions:
//! - `j1`, `j2`, `j3` are angular momentum quantum numbers `l` (orbital) or
//!   `j` (total angular momentum), scaled by `scale` so they can be passed as
//!   `i32`: use `scale = 1` for integer angular momenta (orbital `l`) and
//!   `scale = 2` for half-integer angular momenta doubled to an integer
//!   (e.g. `j = 3/2` is passed as `3` with `scale = 2`).
//! - `m1`, `m2` are the corresponding magnetic quantum numbers, scaled the
//!   same way as `j1`/`j2`; the function derives `m3 = -m1 - m2` internally,
//!   matching the 3j symbol's `m1 + m2 + m3 = 0` selection rule.
//! - The return value is the dimensionless Wigner 3j symbol
//!   `(j1 j2 j3; m1 m2 m3)`, used throughout FEFF to couple partial-wave
//!   angular momenta into multipole matrix elements.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p refeff-core --example wigner_3j
//! ```

use refeff_core::wigner_3j;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Integer angular momenta (p-p coupling to a scalar, l1 = l2 = 1, l3 = 0),
    // scale = 1: (1 1 0; 0 0 0) = -1/sqrt(3).
    let coupling_to_scalar = wigner_3j(1, 1, 0, 0, 0, 1)?;
    println!("(1 1 0; 0 0 0) = {coupling_to_scalar:.6}  [expected -1/sqrt(3) = -0.577350]");

    // Triangle-inequality violation (l3 = 3 cannot couple l1 = l2 = 1):
    // returns exactly zero rather than an error.
    let forbidden = wigner_3j(1, 1, 3, 0, 0, 1)?;
    println!("(1 1 3; 0 0 0) = {forbidden:.6}  [forbidden by the triangle rule]");

    // Half-integer angular momenta doubled for scale = 2: j1 = 0, j2 = 1,
    // j3 = 1, m1 = 0, m2 = -1 -> (0 1 1; 0 -1 1) = -1/sqrt(2).
    let half_integer = wigner_3j(0, 1, 1, 0, -1, 2)?;
    println!("(0 1 1; 0 -1 1) = {half_integer:.6}  [expected -1/sqrt(2) = -0.707107]");

    Ok(())
}
