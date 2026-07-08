#![allow(unused_parens)]

use super::*;

const SELF_ENERGY_BPR_ZERO: Real = 1.0e-10;

#[derive(Clone, Copy)]
struct BprContext {
    q: Complex,
    k: Complex,
    en: Complex,
    emk: Complex,
    wq: Complex,
    gam: Complex,
}

fn bpr_context(
    input: SelfEnergyIntegrandInput,
    force_on_shell: bool,
) -> Result<BprContext, SelfEnergyError> {
    validate_bpr_integrand_input(input)?;
    let q = input.q;
    let k = input.normalized_momentum;
    let en = input.normalized_energy;
    let wq = Complex::new(omega_q(input.plasmon_over_fermi, q.re)?, 0.0);
    let gam = Complex::new(gamma_q(input.width_over_fermi, q.re)?, 0.0);
    let emk = if input.on_shell || force_on_shell {
        Complex::new(0.0, 0.0)
    } else {
        en - k * k
    };
    Ok(BprContext {
        q,
        k,
        en,
        emk,
        wq,
        gam,
    })
}

fn validate_bpr_integrand_input(input: SelfEnergyIntegrandInput) -> Result<(), SelfEnergyError> {
    ensure_finite_complex("q", input.q)?;
    ensure_finite_complex("CPar(1)", input.normalized_momentum)?;
    ensure_finite_complex("CPar(2)", input.normalized_energy)?;
    ensure_positive_real("DPPar(1)", input.plasmon_over_fermi)?;
    ensure_nonnegative_real("DPPar(2)", input.width_over_fermi)?;
    ensure_finite_real("DPPar(4)", input.gap_energy)?;
    ensure_positive_real("BPR q real", input.q.re)
}

fn bpr_log_abs(value: Complex) -> Result<Complex, SelfEnergyError> {
    ensure_finite_complex("BPR log argument", value)?;
    let norm = value.norm();
    if norm == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "BPR log argument",
        });
    }
    Ok(Complex::new(norm.ln(), 0.0))
}

fn cpow(value: Complex, power: i32) -> Complex {
    value.powi(power)
}

fn bpr_sum(
    name: &'static str,
    amp: Complex,
    rq: &[Complex; 13],
) -> Result<Complex, SelfEnergyError> {
    let value = amp * rq.iter().copied().sum::<Complex>();
    ensure_finite_complex(name, value)?;
    Ok(value)
}
/// Port of FEFF `bpr1`: the middle broadened-pole integrand.
pub fn self_energy_bpr1_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let BprContext {
        q,
        k,
        en,
        emk,
        wq,
        gam,
    } = bpr_context(input, false)?;
    let (amp, rq) = if (wq - gam).re > 0.0 {
        {
            let mut rq = [Complex::new(0.0, 0.0); 13];
            let a1 = (emk - gam) + (2.0e0 * k * q + wq - cpow(q, 2));
            let a2 = (emk + gam) - (2.0e0 * k * q + wq + cpow(q, 2));
            let a3 = (emk + gam) + (2.0e0 * k * q + wq - cpow(q, 2));
            let a4 = (emk - gam) - (2.0e0 * k * q + wq + cpow(q, 2));
            let b1 = (1.0e0 - gam) + (wq - en);
            let b2 = (1.0e0 - gam) - (wq + en);
            let b3 = (1.0e0 + gam) + (wq - en);
            let b4 = (1.0e0 + gam) - (wq + en);
            let l1 = bpr_log_abs((a1 / a3) * (a4 / a2))? + log_i(a1, -1)?
                - log_i(-a2, -1)?
                - log_i(a3, -1)?
                + log_i(-a4, -1)?;
            let l2 = bpr_log_abs((a1 / a3) * (a2 / a4))? + log_i(a1, -1)? + log_i(-a2, -1)?
                - log_i(a3, -1)?
                - log_i(-a4, -1)?;
            let l3 = bpr_log_abs((a1 / a3) * (a4 / a2))? + log_i(a1, -1)?
                - log_i(a2, 1)?
                - log_i(a3, -1)?
                + log_i(a4, 1)?;
            let l4 = bpr_log_abs((a1 / a4) * (a3 / a2))? + log_i(a1, -1)? - log_i(a2, 1)?
                + log_i(a3, -1)?
                - log_i(a4, 1)?;
            let l5 = bpr_log_abs((b1 / b3) * (b2 / b4))? + log_i(-b1, 1)? + log_i(-b2, -1)?
                - log_i(-b3, 1)?
                - log_i(-b4, -1)?;
            let l6 = bpr_log_abs((b1 / b3) * (b4 / b2))? + log_i(b1, -1)?
                - log_i(-b2, -1)?
                - log_i(b3, -1)?
                + log_i(-b4, -1)?;
            let l7 = bpr_log_abs((b1 / b3) * (b2 / b4))? + log_i(b1, -1)? + log_i(-b2, -1)?
                - log_i(b3, -1)?
                - log_i(-b4, -1)?;
            let l8 = bpr_log_abs((b1 / b2) * (b3 / b4))? + log_i(-b1, 1)? - log_i(-b2, -1)?
                + log_i(-b3, 1)?
                - log_i(-b4, -1)?;
            let amp = 7.0e0 / (480.0e0 * cpow(gam, 4) * q * (cpow(gam, 2) + 7.0e0 * cpow(wq, 2)));
            rq[12] = (15.0
                * ((l3 + l5) * cpow(wq, 6)
                    + 8.0 * l2 * cpow(q, 6) * wq * (3.0 * cpow(q, 4) + 5.0 * cpow(wq, 2))
                    - 5.0
                        * l1
                        * cpow(q, 4)
                        * (cpow(q, 8) + 9.0 * cpow(q, 4) * cpow(wq, 2) + 3.0 * cpow(wq, 4))))
                / gam;
            rq[11] = (900.0
                * k
                * cpow(q, 3)
                * (-4.0 * l1 * cpow(q, 2) * wq * (cpow(q, 4) + cpow(wq, 2))
                    + l2 * (cpow(q, 8) + 6.0 * cpow(q, 4) * cpow(wq, 2) + cpow(wq, 4))))
                / gam;
            rq[10] = (30.0
                * cpow(q, 2)
                * (5.0 * (2.0 * gam + 3.0 * emk * l1 - 30.0 * cpow(k, 2) * l1) * cpow(q, 8)
                    - 60.0 * (emk - 8.0 * cpow(k, 2)) * l2 * cpow(q, 6) * wq
                    + 2.0
                        * (26.0 * gam + 45.0 * (emk - 6.0 * cpow(k, 2)) * l1)
                        * cpow(q, 4)
                        * cpow(wq, 2)
                    - 60.0 * (emk - 4.0 * cpow(k, 2)) * l2 * cpow(q, 2) * cpow(wq, 3)
                    + (2.0 * gam + 15.0 * (emk - 2.0 * cpow(k, 2)) * l1) * cpow(wq, 4)))
                / gam;
            rq[9] = (60.0
                * k
                * q
                * (-25.0 * (3.0 * emk - 8.0 * cpow(k, 2)) * l2 * cpow(q, 8)
                    + 8.0 * (19.0 * gam + 30.0 * (emk - 2.0 * cpow(k, 2)) * l1) * cpow(q, 6) * wq
                    - 90.0 * (3.0 * emk - 4.0 * cpow(k, 2)) * l2 * cpow(q, 4) * cpow(wq, 2)
                    + 8.0
                        * (7.0 * gam + 15.0 * emk * l1 - 10.0 * cpow(k, 2) * l1)
                        * cpow(q, 2)
                        * cpow(wq, 3)
                    - 15.0 * emk * l2 * cpow(wq, 4)))
                / gam;
            rq[8] = (15.0
                * (-5.0
                    * (20.0 * gam * (emk - 8.0 * cpow(k, 2)) - 3.0 * cpow(gam, 2) * l1
                        + 15.0
                            * (cpow(emk, 2) - 16.0 * emk * cpow(k, 2) + 16.0 * cpow(k, 4))
                            * l1)
                    * cpow(q, 8)
                    + 40.0
                        * (6.0 * cpow(emk, 2) - 1.0 * cpow(gam, 2) - 72.0 * emk * cpow(k, 2)
                            + 48.0 * cpow(k, 4))
                        * l2
                        * cpow(q, 6)
                        * wq
                    - 6.0
                        * (52.0 * gam * (emk - 4.0 * cpow(k, 2)) - 5.0 * cpow(gam, 2) * l1
                            + 15.0
                                * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2)
                                    + 8.0 * cpow(k, 4))
                                * l1)
                        * cpow(q, 4)
                        * cpow(wq, 2)
                    + 120.0 * emk * (emk - 4.0 * cpow(k, 2)) * l2 * cpow(q, 2) * cpow(wq, 3)
                    - 1.0
                        * (4.0 * emk * gam
                            + 15.0 * cpow(emk, 2) * l1
                            + gam * (4.0 - 4.0 * en + 5.0 * gam * (l3 + l5))
                            + 15.0 * cpow(en - 1.0, 2) * l7)
                        * cpow(wq, 4)))
                / gam;
            rq[7] = (120.0
                * k
                * q
                * (15.0
                    * (5.0 * cpow(emk, 2) - 1.0 * cpow(gam, 2) - 20.0 * emk * cpow(k, 2)
                        + 8.0 * cpow(k, 4))
                    * l2
                    * cpow(q, 6)
                    - 2.0
                        * (38.0 * gam * (3.0 * emk - 4.0 * cpow(k, 2))
                            - 15.0 * cpow(gam, 2) * l1
                            + 6.0
                                * (15.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 8.0 * cpow(k, 4))
                                * l1)
                        * cpow(q, 4)
                        * wq
                    - 15.0
                        * (-9.0 * cpow(emk, 2) + cpow(gam, 2) + 12.0 * emk * cpow(k, 2))
                        * l2
                        * cpow(q, 2)
                        * cpow(wq, 2)
                    - 2.0 * emk * (14.0 * gam + 15.0 * emk * l1) * cpow(wq, 3)))
                / gam;
            rq[6] = (20.0
                * (5.0
                    * (-8.0 * cpow(gam, 3)
                        + 30.0
                            * gam
                            * (cpow(emk, 2) - 12.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                        - 9.0 * cpow(gam, 2) * (emk - 6.0 * cpow(k, 2)) * l1
                        + 3.0
                            * (5.0 * cpow(emk, 3) - 90.0 * cpow(emk, 2) * cpow(k, 2)
                                + 120.0 * emk * cpow(k, 4)
                                - 16.0 * cpow(k, 6))
                            * l1)
                    * cpow(q, 6)
                    - 90.0
                        * (-1.0 * cpow(gam, 2) * (emk - 4.0 * cpow(k, 2))
                            + 2.0
                                * emk
                                * (cpow(emk, 2) - 12.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4)))
                        * l2
                        * cpow(q, 4)
                        * wq
                    + 3.0
                        * (-4.0 * cpow(gam, 3)
                            + 78.0 * emk * gam * (emk - 4.0 * cpow(k, 2))
                            + 45.0 * cpow(emk, 2) * (emk - 6.0 * cpow(k, 2)) * l1
                            - 15.0 * cpow(gam, 2) * (emk - 2.0 * cpow(k, 2)) * l1)
                        * cpow(q, 2)
                        * cpow(wq, 2)
                    - 30.0 * (cpow(emk, 3) * l2 - 1.0 * cpow(en - 1.0, 3) * l6) * cpow(wq, 3)))
                / gam;
            rq[5] = (120.0
                * k
                * q
                * (-15.0
                    * (5.0 * cpow(emk, 3)
                        - 3.0 * emk * cpow(gam, 2)
                        - 20.0 * cpow(emk, 2) * cpow(k, 2)
                        + 4.0 * cpow(gam, 2) * cpow(k, 2)
                        + 8.0 * emk * cpow(k, 4))
                    * l2
                    * cpow(q, 4)
                    + 4.0
                        * (-11.0 * cpow(gam, 3)
                            + 19.0 * emk * gam * (3.0 * emk - 4.0 * cpow(k, 2))
                            + 30.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * l1
                            - 5.0 * cpow(gam, 2) * (3.0 * emk - 2.0 * cpow(k, 2)) * l1)
                        * cpow(q, 2)
                        * wq
                    - 15.0 * emk * (3.0 * cpow(emk, 2) - 1.0 * cpow(gam, 2)) * l2 * cpow(wq, 2)))
                / gam;
            rq[4] = 15.0
                * ((5.0
                    * (32.0 * cpow(gam, 3) * (emk - 4.0 * cpow(k, 2))
                        - 40.0
                            * emk
                            * gam
                            * (cpow(emk, 2) - 12.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                        - 3.0 * cpow(gam, 4) * l1
                        + 6.0
                            * cpow(gam, 2)
                            * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * l1
                        - 15.0
                            * cpow(emk, 2)
                            * (cpow(emk, 2) - 16.0 * emk * cpow(k, 2) + 16.0 * cpow(k, 4))
                            * l1)
                    * cpow(q, 4))
                    / gam
                    + (120.0
                        * emk
                        * (cpow(emk, 2) * (emk - 8.0 * cpow(k, 2))
                            - 1.0 * cpow(gam, 2) * (emk - 4.0 * cpow(k, 2)))
                        * l2
                        * cpow(q, 2)
                        * wq)
                        / gam
                    + (-104.0 * cpow(emk, 3)
                        + 104.0 * cpow(en - 1.0, 3)
                        + 16.0 * (1.0 + emk - 1.0 * en) * cpow(gam, 2)
                        + 15.0 * cpow(gam, 3) * (l3 + l5)
                        + 30.0 * gam * (cpow(emk, 2) * l1 + cpow(en - 1.0, 2) * l7)
                        - (45.0 * (cpow(emk, 4) * l1 + cpow(en - 1.0, 4) * l7)) / gam)
                        * cpow(wq, 2));
            rq[3] = (60.0
                * k
                * q
                * (5.0
                    * (3.0 * cpow(gam, 4) + 5.0 * cpow(emk, 3) * (3.0 * emk - 8.0 * cpow(k, 2))
                        - 6.0 * emk * cpow(gam, 2) * (3.0 * emk - 4.0 * cpow(k, 2)))
                    * l2
                    * cpow(q, 2)
                    - 4.0
                        * emk
                        * (38.0 * cpow(emk, 2) * gam - 22.0 * cpow(gam, 3)
                            + 15.0 * cpow(emk, 3) * l1
                            - 15.0 * emk * cpow(gam, 2) * l1)
                        * wq))
                / gam;
            rq[2] = (30.0
                * ((22.0 * cpow(gam, 5) + 50.0 * cpow(emk, 3) * gam * (emk - 8.0 * cpow(k, 2))
                    - 80.0 * emk * cpow(gam, 3) * (emk - 4.0 * cpow(k, 2))
                    + 15.0 * cpow(emk, 4) * (emk - 10.0 * cpow(k, 2)) * l1
                    - 30.0 * cpow(emk, 2) * cpow(gam, 2) * (emk - 6.0 * cpow(k, 2)) * l1
                    + 15.0 * cpow(gam, 4) * (emk - 2.0 * cpow(k, 2)) * l1)
                    * cpow(q, 2)
                    - 4.0
                        * (3.0 * cpow(emk, 5) * l2 - 5.0 * cpow(emk, 3) * cpow(gam, 2) * l2
                            + cpow(en - 1.0, 3)
                                * (-3.0 * cpow(en - 1.0, 2) + 5.0 * cpow(gam, 2))
                                * l6
                            + 2.0 * cpow(gam, 5) * (l4 + l8))
                        * wq))
                / gam;
            rq[1] =
                (-900.0 * emk * cpow((cpow(emk, 2) - 1.0 * cpow(gam, 2)), 2) * k * l2 * q) / gam;
            rq[0] = -300.0 * cpow(emk, 5)
                + 300.0 * cpow(en - 1.0, 5)
                + 800.0 * (cpow(emk, 3) - 1.0 * cpow(en - 1.0, 3)) * cpow(gam, 2)
                - 660.0 * (1.0 + emk - 1.0 * en) * cpow(gam, 4)
                + 75.0 * cpow(gam, 5) * (l3 + l5)
                - 225.0 * cpow(gam, 3) * (cpow(emk, 2) * l1 + cpow(en - 1.0, 2) * l7)
                + 225.0 * gam * (cpow(emk, 4) * l1 + cpow(en - 1.0, 4) * l7)
                + (-75.0 * cpow(emk, 6) * l1 - 75.0 * cpow(en - 1.0, 6) * l7) / gam;
            (amp, rq)
        }
    } else {
        {
            let mut rq = [Complex::new(0.0, 0.0); 13];
            let amp = 7.0e0
                / (120.0e0
                    * q
                    * cpow((gam + wq), 5)
                    * (8.0e0 * cpow(gam, 2) - 5.0e0 * gam * wq + cpow(wq, 2)));
            let a1 = emk - (2.0e0 * k * q + cpow(q, 2) + wq) - gam;
            let a2 = emk + (2.0e0 * k * q - cpow(q, 2) + wq) + gam;
            let a3 = emk + 2.0e0 * k * q - cpow(q, 2);
            let a4 = emk - 2.0e0 * k * q - cpow(q, 2);
            let b1 = -1.0e0 + (en - wq) - gam;
            let b2 = -1.0e0 + (en + wq) + gam;
            let b3 = en - 1.0e0;
            let b4 = en - 1.0e0;
            let l1 = bpr_log_abs((a1 / a2) * (b2 / b1))? + log_i(a1, 1)?
                - log_i(a2, -1)?
                - log_i(b1, 1)?
                + log_i(b2, -1)?;
            let l2 = bpr_log_abs((a1 / a2) * (a3 / a4))? + log_i(-a1, -1)? - log_i(a2, -1)?
                + log_i(a3, -1)?
                - log_i(-a4, -1)?;
            let l3 = bpr_log_abs((a1 / a4) * (a2 / a3))? + log_i(-a1, -1)? + log_i(a2, -1)?
                - log_i(a3, -1)?
                - log_i(-a4, -1)?;
            let l4 = bpr_log_abs(b2 / b1)? - log_i(-b1, -1)? + log_i(b2, -1)? + log_i(-b3, -1)?
                - log_i(b4, -1)?;
            let l5 = if b3.norm() < SELF_ENERGY_BPR_ZERO {
                Complex::new(0.0, 0.0)
            } else {
                bpr_log_abs((b1 / b3) * (b2 / b4))? + log_i(-b1, -1)? + log_i(b2, -1)?
                    - log_i(-b3, -1)?
                    - log_i(b4, -1)?
            };
            rq[6] = 300.0 * cpow(gam, 6) * l1;
            rq[5] = 120.0
                * cpow(gam, 5)
                * (-11.0 * emk + 11.0 * (-1.0 + en + cpow(q, 2)) + 8.0 * l1 * wq);
            rq[4] = 60.0
                * cpow(gam, 4)
                * (-38.0 * (1.0 + emk - 1.0 * en) * wq
                    + 2.0 * cpow(q, 2) * (15.0 * (emk - 2.0 * cpow(k, 2)) * l2 + 19.0 * wq)
                    - 5.0
                        * (3.0 * (-2.0 + en) * en * l4
                            + 2.0 * k * (-5.0 + 6.0 * l3) * q * (-1.0 * emk + cpow(q, 2))
                            + 3.0
                                * (l4 + l2 * (cpow(emk, 2) + cpow(q, 4))
                                    - 1.0 * l1 * cpow(wq, 2))));
            rq[3] = 160.0
                * cpow(gam, 3)
                * (-10.0 * cpow(en - 1.0, 3)
                    + 10.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                    + 66.0 * k * q * (emk - 1.0 * cpow(q, 2)) * wq
                    + 3.0 * (1.0 + emk - 1.0 * en - 1.0 * cpow(q, 2)) * cpow(wq, 2));
            rq[2] = 60.0
                * cpow(gam, 2)
                * (15.0 * (-2.0 + en) * en * (2.0 + (-2.0 + en) * en) * l4
                    + 15.0 * (l4 + l2 * (cpow(emk, 4) + cpow(q, 8)))
                    + (cpow(emk, 3) * (44.0 - 40.0 * l3)
                        + 4.0 * cpow(en - 1.0, 3) * (-11.0 + 10.0 * l5))
                        * wq
                    + 30.0 * (cpow(emk, 2) * l2 + cpow(en - 1.0, 2) * l4) * cpow(wq, 2)
                    + 32.0 * (1.0 + emk - 1.0 * en) * cpow(wq, 3)
                    - 5.0 * l1 * cpow(wq, 4)
                    + 40.0
                        * k
                        * cpow(q, 5)
                        * (emk * (3.0 - 9.0 * l3)
                            + 4.0 * cpow(k, 2) * (-1.0 + 3.0 * l3)
                            + 6.0 * l2 * wq)
                    - 4.0
                        * cpow(q, 6)
                        * (15.0 * (emk - 6.0 * cpow(k, 2)) * l2 + (11.0 - 10.0 * l3) * wq)
                    + 6.0
                        * cpow(q, 4)
                        * (5.0
                            * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * l2
                            - 2.0 * (emk - 4.0 * cpow(k, 2)) * (-11.0 + 10.0 * l3) * wq
                            + 5.0 * l2 * cpow(wq, 2))
                    + 4.0
                        * k
                        * cpow(q, 3)
                        * (10.0 * emk * (3.0 * emk - 4.0 * cpow(k, 2)) * (-1.0 + 3.0 * l3)
                            - 40.0 * (3.0 * emk - 2.0 * cpow(k, 2)) * l2 * wq
                            + (-77.0 + 30.0 * l3) * cpow(wq, 2))
                    + 4.0
                        * k
                        * q
                        * (10.0 * (-1.0 + 3.0 * l3) * (-1.0 * cpow(emk, 3) + cpow(q, 6))
                            + 60.0 * cpow(emk, 2) * l2 * wq
                            - 1.0 * emk * (-77.0 + 30.0 * l3) * cpow(wq, 2))
                    + 4.0
                        * cpow(q, 2)
                        * (-15.0 * cpow(emk, 2) * (emk - 6.0 * cpow(k, 2)) * l2
                            + 3.0 * emk * (emk - 4.0 * cpow(k, 2)) * (-11.0 + 10.0 * l3) * wq
                            - 15.0 * (emk - 2.0 * cpow(k, 2)) * l2 * cpow(wq, 2)
                            - 8.0 * cpow(wq, 3)));
            rq[1] = 120.0
                * gam
                * (5.0 * cpow(en - 1.0, 5)
                    - 5.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 4)
                            - 4.0 * cpow(emk, 2) * (emk - 10.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (3.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 40.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 4.0 * (emk - 10.0 * cpow(k, 2)) * cpow(q, 6)
                            + cpow(q, 8))
                    + 152.0
                        * k
                        * q
                        * (-1.0 * emk + cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * wq
                    - 26.0
                        * (-1.0 * cpow(en - 1.0, 3)
                            + (emk - 1.0 * cpow(q, 2))
                                * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                                    + cpow(q, 4)))
                        * cpow(wq, 2)
                    + 56.0 * k * q * (-1.0 * emk + cpow(q, 2)) * cpow(wq, 3)
                    - 1.0 * (1.0 + emk - 1.0 * en - 1.0 * cpow(q, 2)) * cpow(wq, 4));
            let rq00 = 40.0
                * wq
                * (15.0 * cpow(en - 1.0, 5)
                    - 15.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 4)
                            - 4.0 * cpow(emk, 2) * (emk - 10.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (3.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 40.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 4.0 * (emk - 10.0 * cpow(k, 2)) * cpow(q, 6)
                            + cpow(q, 8))
                    + 516.0
                        * k
                        * q
                        * (-1.0 * emk + cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * wq
                    - 104.0
                        * (-1.0 * cpow(en - 1.0, 3)
                            + (emk - 1.0 * cpow(q, 2))
                                * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                                    + cpow(q, 4)))
                        * cpow(wq, 2)
                    + 291.0 * k * q * (-1.0 * emk + cpow(q, 2)) * cpow(wq, 3)
                    - 15.0 * (1.0 + emk - 1.0 * en - 1.0 * cpow(q, 2)) * cpow(wq, 4));
            let rq0l1 = 60.0 * l1 * cpow(wq, 6);
            let rq0l2 = 60.0
                * l2
                * (-5.0
                    * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2) + cpow(q, 4))
                    * (cpow(emk, 4) - 4.0 * cpow(emk, 2) * (emk - 14.0 * cpow(k, 2)) * cpow(q, 2)
                        + 2.0
                            * (3.0 * cpow(emk, 2) - 56.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * cpow(q, 4)
                        - 4.0 * (emk - 14.0 * cpow(k, 2)) * cpow(q, 6)
                        + cpow(q, 8))
                    - 48.0
                        * k
                        * q
                        * (5.0 * cpow(emk, 4)
                            - 20.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (15.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 8.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 20.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 6)
                            + 5.0 * cpow(q, 8))
                        * wq
                    - 45.0
                        * (cpow(emk, 4)
                            - 4.0 * cpow(emk, 2) * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2)
                                    + 8.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 4.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 6)
                            + cpow(q, 8))
                        * cpow(wq, 2)
                    - 80.0
                        * k
                        * q
                        * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                            + 4.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * cpow(q, 4))
                        * cpow(wq, 3)
                    - 15.0
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 4));
            let rq0l3 = 240.0
                * l3
                * (emk - 1.0 * cpow(q, 2))
                * (5.0
                    * k
                    * q
                    * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2) + cpow(q, 4))
                    * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                        + 4.0 * cpow(k, 2) * cpow(q, 2)
                        + 3.0 * cpow(q, 4))
                    + 6.0
                        * (cpow(emk, 4)
                            - 4.0 * cpow(emk, 2) * (emk - 10.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (3.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 40.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 4.0 * (emk - 10.0 * cpow(k, 2)) * cpow(q, 6)
                            + cpow(q, 8))
                        * wq
                    + 90.0
                        * k
                        * q
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 2)
                    + 10.0
                        * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 3)
                    + 15.0 * k * q * cpow(wq, 4));
            let rq0l4 = 300.0
                * cpow(en - 1.0, 2)
                * l4
                * (-1.0 * cpow(en - 1.0, 4)
                    - 9.0 * cpow(en - 1.0, 2) * cpow(wq, 2)
                    - 3.0 * cpow(wq, 4));
            let rq0l5 = -480.0
                * cpow(en - 1.0, 3)
                * l5
                * wq
                * (3.0 * cpow(en - 1.0, 2) + 5.0 * cpow(wq, 2));
            rq[0] = rq00 + rq0l1 + rq0l2 + rq0l3 + rq0l4 + rq0l5;
            (amp, rq)
        }
    };
    bpr_sum("bpr1", amp, &rq)
}

/// Port of FEFF `bpr2`: the upper-branch broadened-pole integrand.
pub fn self_energy_bpr2_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let BprContext {
        q,
        k,
        en: _,
        emk,
        wq,
        gam,
    } = bpr_context(input, false)?;
    let (amp, rq) = if (wq - gam).re > 0.0 {
        {
            let mut rq = [Complex::new(0.0, 0.0); 13];
            let amp = 1.0e0 / (96.0e0 * cpow(gam, 4) * q * (cpow(gam, 2) + 7.0e0 * cpow(wq, 2)));
            let a1 = (emk + gam) + (2.0e0 * k * q - cpow(q, 2) - wq);
            let a2 = (emk + gam) - (2.0e0 * k * q + cpow(q, 2) + wq);
            let a3 = (emk - gam) + (2.0e0 * k * q - cpow(q, 2) - wq);
            let a4 = (emk - gam) - (2.0e0 * k * q + cpow(q, 2) + wq);
            let l1 = bpr_log_abs((a1 / a3) * (a4 / a2))? + log_i(-a1, -1)?
                - log_i(-a2, -1)?
                - log_i(-a3, -1)?
                + log_i(-a4, -1)?;
            let l2 =
                bpr_log_abs((a1 / a3) * (a4 / a2))? + log_i(a1, 1)? - log_i(a2, 1)? - log_i(a3, 1)?
                    + log_i(a4, 1)?;
            let l3 = bpr_log_abs((a1 / a3) * (a2 / a4))? + log_i(-a1, -1)? + log_i(-a2, -1)?
                - log_i(-a3, -1)?
                - log_i(-a4, -1)?;
            let l4 = bpr_log_abs((a1 / a2) * (a3 / a4))? + log_i(a1, 1)? - log_i(a2, 1)?
                + log_i(a3, 1)?
                - log_i(a4, 1)?;
            rq[12] = (21.0
                * (-5.0 * l1 * cpow(q, 12)
                    - 24.0 * l1 * cpow(q, 10) * wq
                    - 45.0 * l1 * cpow(q, 8) * cpow(wq, 2)
                    - 40.0 * l1 * cpow(q, 6) * cpow(wq, 3)
                    - 15.0 * l1 * cpow(q, 4) * cpow(wq, 4)
                    + l2 * cpow(wq, 6)))
                / gam;
            rq[11] = (1260.0 * k * l3 * cpow(q, 3) * cpow((cpow(q, 2) + wq), 4)) / gam;
            rq[10] = (630.0
                * l1
                * cpow(q, 2)
                * cpow((cpow(q, 2) + wq), 3)
                * (emk * (cpow(q, 2) + wq) - 2.0 * cpow(k, 2) * (5.0 * cpow(q, 2) + wq)))
                / gam;
            rq[9] = (-84.0
                * k
                * q
                * cpow((cpow(q, 2) + wq), 2)
                * (15.0 * emk * l3 * (cpow(q, 2) + wq) * (5.0 * cpow(q, 2) + wq)
                    - 2.0 * gam * (cpow(q, 2) + wq) * (25.0 * cpow(q, 2) + wq)
                    - 40.0 * cpow(k, 2) * l3 * cpow(q, 2) * (5.0 * cpow(q, 2) + 2.0 * wq)))
                / gam;
            rq[8] = (105.0
                * (-3.0
                    * cpow(emk, 2)
                    * l1
                    * cpow((cpow(q, 2) + wq), 3)
                    * (5.0 * cpow(q, 2) + wq)
                    + 48.0
                        * emk
                        * cpow(k, 2)
                        * l1
                        * cpow(q, 2)
                        * cpow((cpow(q, 2) + wq), 2)
                        * (5.0 * cpow(q, 2) + 2.0 * wq)
                    - 48.0
                        * cpow(k, 4)
                        * l1
                        * cpow(q, 4)
                        * (cpow(q, 2) + wq)
                        * (5.0 * cpow(q, 2) + 3.0 * wq)
                    + cpow(gam, 2)
                        * (3.0 * l1 * cpow(q, 8)
                            + 8.0 * l1 * cpow(q, 6) * wq
                            + 6.0 * l1 * cpow(q, 4) * cpow(wq, 2)
                            - 1.0 * l2 * cpow(wq, 4))))
                / gam;
            rq[7] = (168.0
                * k
                * q
                * (15.0
                    * cpow(emk, 2)
                    * l3
                    * cpow((cpow(q, 2) + wq), 2)
                    * (5.0 * cpow(q, 2) + 2.0 * wq)
                    - 4.0
                        * emk
                        * (cpow(q, 2) + wq)
                        * (15.0 * cpow(k, 2) * l3 * cpow(q, 2) * (5.0 * cpow(q, 2) + 3.0 * wq)
                            + gam * (cpow(q, 2) + wq) * (25.0 * cpow(q, 2) + 7.0 * wq))
                    + cpow(q, 2)
                        * (-15.0 * cpow(gam, 2) * l3 * cpow((cpow(q, 2) + wq), 2)
                            + 24.0
                                * cpow(k, 4)
                                * l3
                                * cpow(q, 2)
                                * (5.0 * cpow(q, 2) + 4.0 * wq)
                            + 8.0
                                * gam
                                * cpow(k, 2)
                                * (cpow(q, 2) + wq)
                                * (25.0 * cpow(q, 2) + 13.0 * wq))))
                / gam;
            rq[6] = (420.0
                * l1
                * (cpow(emk, 3) * cpow((cpow(q, 2) + wq), 2) * (5.0 * cpow(q, 2) + 2.0 * wq)
                    - 18.0
                        * cpow(emk, 2)
                        * cpow(k, 2)
                        * cpow(q, 2)
                        * (cpow(q, 2) + wq)
                        * (5.0 * cpow(q, 2) + 3.0 * wq)
                    + 2.0
                        * cpow(k, 2)
                        * cpow(q, 2)
                        * (-8.0 * cpow(k, 4) * cpow(q, 4)
                            + 3.0 * cpow(gam, 2) * (cpow(q, 2) + wq) * (3.0 * cpow(q, 2) + wq))
                    + 3.0
                        * emk
                        * cpow(q, 2)
                        * (-1.0 * cpow(gam, 2) * cpow((cpow(q, 2) + wq), 2)
                            + 8.0 * cpow(k, 4) * (5.0 * cpow(q, 4) + 4.0 * cpow(q, 2) * wq))))
                / gam;
            rq[5] = (-168.0
                * k
                * q
                * (15.0 * cpow(emk, 3) * l3 * (cpow(q, 2) + wq) * (5.0 * cpow(q, 2) + 3.0 * wq)
                    + 4.0
                        * gam
                        * (-20.0 * cpow(k, 4) * cpow(q, 4)
                            + cpow(gam, 2) * (cpow(q, 2) + wq) * (10.0 * cpow(q, 2) + wq)
                            + 5.0
                                * gam
                                * cpow(k, 2)
                                * l3
                                * cpow(q, 2)
                                * (3.0 * cpow(q, 2) + 2.0 * wq))
                    - 6.0
                        * cpow(emk, 2)
                        * (10.0 * cpow(k, 2) * l3 * cpow(q, 2) * (5.0 * cpow(q, 2) + 4.0 * wq)
                            + gam * (cpow(q, 2) + wq) * (25.0 * cpow(q, 2) + 13.0 * wq))
                    + emk
                        * (120.0 * cpow(k, 4) * l3 * cpow(q, 4)
                            - 15.0
                                * cpow(gam, 2)
                                * l3
                                * (cpow(q, 2) + wq)
                                * (3.0 * cpow(q, 2) + wq)
                            + 16.0
                                * gam
                                * cpow(k, 2)
                                * cpow(q, 2)
                                * (25.0 * cpow(q, 2) + 19.0 * wq))))
                / gam;
            rq[4] = (315.0
                * (-1.0
                    * (cpow(gam, 4)
                        - 2.0
                            * cpow(gam, 2)
                            * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                        + 5.0
                            * cpow(emk, 2)
                            * (cpow(emk, 2) - 16.0 * emk * cpow(k, 2) + 16.0 * cpow(k, 4)))
                    * l1
                    * cpow(q, 4)
                    - 8.0
                        * emk
                        * (cpow(emk, 2) * (emk - 8.0 * cpow(k, 2))
                            - 1.0 * cpow(gam, 2) * (emk - 4.0 * cpow(k, 2)))
                        * l1
                        * cpow(q, 2)
                        * wq
                    + (-3.0 * cpow(emk, 4) * l1
                        + 2.0 * cpow(emk, 2) * cpow(gam, 2) * l1
                        + cpow(gam, 4) * l2)
                        * cpow(wq, 2)))
                / gam;
            rq[3] = (28.0
                * k
                * q
                * (-5.0 * cpow(gam, 3) * (64.0 * cpow(k, 2) - 9.0 * gam * l3) * cpow(q, 2)
                    + 45.0 * cpow(emk, 4) * l3 * (5.0 * cpow(q, 2) + 4.0 * wq)
                    + 24.0
                        * emk
                        * cpow(gam, 2)
                        * (5.0 * (4.0 * gam + 3.0 * cpow(k, 2) * l3) * cpow(q, 2)
                            + 11.0 * gam * wq)
                    - 24.0
                        * cpow(emk, 3)
                        * (25.0 * (gam + cpow(k, 2) * l3) * cpow(q, 2) + 19.0 * gam * wq)
                    + 30.0
                        * cpow(emk, 2)
                        * gam
                        * (40.0 * cpow(k, 2) * cpow(q, 2)
                            - 3.0 * gam * l3 * (3.0 * cpow(q, 2) + 2.0 * wq))))
                / gam;
            rq[2] = (42.0
                * (15.0
                    * (emk - 1.0 * gam)
                    * (emk + gam)
                    * (cpow(emk, 2) * (emk - 10.0 * cpow(k, 2))
                        - 1.0 * cpow(gam, 2) * (emk - 2.0 * cpow(k, 2)))
                    * l1
                    * cpow(q, 2)
                    + 4.0
                        * (3.0 * cpow(emk, 5) * l1
                            - 5.0 * cpow(emk, 3) * cpow(gam, 2) * l1
                            - 2.0 * cpow(gam, 5) * l4)
                        * wq))
                / gam;
            rq[1] = (84.0
                * k
                * (50.0 * cpow(emk, 4) * gam - 80.0 * cpow(emk, 2) * cpow(gam, 3)
                    + 22.0 * cpow(gam, 5)
                    - 15.0 * cpow(emk, 5) * l3
                    + 30.0 * cpow(emk, 3) * cpow(gam, 2) * l3
                    - 15.0 * emk * cpow(gam, 4) * l3)
                * q)
                / gam;
            rq[0] = (105.0
                * (-1.0 * cpow(emk, 6) * l1 + 3.0 * cpow(emk, 4) * cpow(gam, 2) * l1
                    - 3.0 * cpow(emk, 2) * cpow(gam, 4) * l1
                    + cpow(gam, 6) * l2))
                / gam;
            (amp, rq)
        }
    } else {
        {
            let mut rq = [Complex::new(0.0, 0.0); 13];
            let amp = 7.0e0
                / (120.0e0
                    * q
                    * cpow((gam + wq), 5)
                    * (8.0e0 * cpow(gam, 2) - 5.0e0 * gam * wq + cpow(wq, 2)));
            let a1 = emk - 1.0 * gam - 2.0 * k * q - 1.0 * cpow(q, 2) - 1.0 * wq;
            let a2 = emk - 1.0 * gam + 2.0 * k * q - 1.0 * cpow(q, 2) - 1.0 * wq;
            let a3 = emk + 2.0 * k * q - 1.0 * cpow(q, 2);
            let a4 = emk - 2.0 * k * q - 1.0 * cpow(q, 2);
            let l1 = bpr_log_abs(a1 / a2)? + log_i(a1, 1)? - log_i(a2, 1)?;
            let l2 = bpr_log_abs((a1 / a4) * (a2 / a3))? + log_i(-a1, -1)? + log_i(-a2, -1)?
                - log_i(-a3, -1)?
                - log_i(-a4, -1)?;
            let l3 = bpr_log_abs((a1 / a2) * (a3 / a4))? + log_i(-a1, -1)? - log_i(-a2, -1)?
                + log_i(-a3, -1)?
                - log_i(-a4, -1)?;
            rq[6] = 300.0 * cpow(gam, 6) * l1;
            rq[5] = 240.0 * cpow(gam, 5) * (11.0 * k * q + 4.0 * l1 * wq);
            rq[4] = 60.0
                * cpow(gam, 4)
                * (-15.0
                    * l3
                    * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2) + cpow(q, 4))
                    + 15.0 * l1 * cpow(wq, 2)
                    + 2.0
                        * k
                        * q
                        * (5.0 * (-5.0 + 6.0 * l2) * (emk - 1.0 * cpow(q, 2)) + 38.0 * wq));
            rq[3] = -320.0
                * cpow(gam, 3)
                * k
                * q
                * (30.0 * cpow(emk, 2)
                    + 40.0 * cpow(k, 2) * cpow(q, 2)
                    + 3.0 * (cpow(q, 2) + wq) * (10.0 * cpow(q, 2) + wq)
                    - 3.0 * emk * (20.0 * cpow(q, 2) + 11.0 * wq));
            rq[2] = 60.0
                * cpow(gam, 2)
                * (-5.0 * l1 * cpow(wq, 4)
                    + 4.0
                        * k
                        * q
                        * (10.0
                            * (-1.0 + 3.0 * l2)
                            * (-1.0 * emk + cpow(q, 2))
                            * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                                + cpow(q, 4))
                            + 2.0
                                * (-11.0 + 10.0 * l2)
                                * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                                    + 4.0 * cpow(k, 2) * cpow(q, 2)
                                    + 3.0 * cpow(q, 4))
                                * wq
                            + (-77.0 + 30.0 * l2) * (-1.0 * emk + cpow(q, 2)) * cpow(wq, 2)
                            - 16.0 * cpow(wq, 3))
                    + 5.0
                        * l3
                        * (3.0 * (cpow(emk, 4) + cpow(q, 8)) - 8.0 * cpow(emk, 3) * wq
                            + 6.0 * cpow(emk, 2) * cpow(wq, 2)
                            + 6.0
                                * cpow(q, 4)
                                * (3.0 * cpow(emk, 2)
                                    + 8.0 * cpow(k, 4)
                                    + 16.0 * cpow(k, 2) * wq
                                    + cpow(wq, 2)
                                    - 4.0 * emk * (6.0 * cpow(k, 2) + wq))
                            - 12.0
                                * cpow(q, 2)
                                * (emk - 1.0 * wq)
                                * (cpow(emk, 2) + 2.0 * cpow(k, 2) * wq
                                    - 1.0 * emk * (6.0 * cpow(k, 2) + wq))
                            - 4.0 * cpow(q, 6) * (3.0 * emk - 2.0 * (9.0 * cpow(k, 2) + wq))));
            rq[1] = 240.0
                * gam
                * k
                * q
                * (25.0 * cpow(emk, 4)
                    + 80.0 * cpow(k, 4) * cpow(q, 4)
                    + cpow((cpow(q, 2) + wq), 3) * (25.0 * cpow(q, 2) + wq)
                    + 8.0
                        * cpow(k, 2)
                        * cpow(q, 2)
                        * (cpow(q, 2) + wq)
                        * (25.0 * cpow(q, 2) + 13.0 * wq)
                    - 4.0 * cpow(emk, 3) * (25.0 * cpow(q, 2) + 19.0 * wq)
                    + 2.0
                        * cpow(emk, 2)
                        * (100.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * (cpow(q, 2) + wq) * (25.0 * cpow(q, 2) + 13.0 * wq))
                    - 4.0
                        * emk
                        * (cpow((cpow(q, 2) + wq), 2) * (25.0 * cpow(q, 2) + 7.0 * wq)
                            + 4.0 * cpow(k, 2) * (25.0 * cpow(q, 4) + 19.0 * cpow(q, 2) * wq)));
            let rq00 = 40.0
                * k
                * q
                * (30.0
                    * (5.0 * cpow(emk, 4)
                        - 20.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                        + 2.0
                            * (15.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * cpow(q, 4)
                        - 20.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 6)
                        + 5.0 * cpow(q, 8))
                    * wq
                    - 516.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 2)
                    + 208.0
                        * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                            + 4.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * cpow(q, 4))
                        * cpow(wq, 3)
                    - 291.0 * (emk - 1.0 * cpow(q, 2)) * cpow(wq, 4)
                    + 30.0 * cpow(wq, 5));
            let rq0l1 = 60.0 * cpow(wq, 6) * l1;
            let rq0l2 = -240.0
                * k
                * q
                * (-15.0 * cpow(emk, 5) + 15.0 * cpow(emk, 4) * (5.0 * cpow(q, 2) + 4.0 * wq)
                    - 10.0
                        * cpow(emk, 3)
                        * (20.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * (cpow(q, 2) + wq) * (5.0 * cpow(q, 2) + 3.0 * wq))
                    - 15.0
                        * emk
                        * (16.0 * cpow(k, 4) * cpow(q, 4)
                            + cpow((cpow(q, 2) + wq), 3) * (5.0 * cpow(q, 2) + wq)
                            + 8.0
                                * cpow(k, 2)
                                * cpow(q, 2)
                                * (cpow(q, 2) + wq)
                                * (5.0 * cpow(q, 2) + 3.0 * wq))
                    + 30.0
                        * cpow(emk, 2)
                        * (cpow((cpow(q, 2) + wq), 2) * (5.0 * cpow(q, 2) + 2.0 * wq)
                            + 4.0 * cpow(k, 2) * (5.0 * cpow(q, 4) + 4.0 * cpow(q, 2) * wq))
                    + cpow(q, 2)
                        * (15.0 * cpow((cpow(q, 2) + wq), 4)
                            + 40.0
                                * cpow(k, 2)
                                * cpow((cpow(q, 2) + wq), 2)
                                * (5.0 * cpow(q, 2) + 2.0 * wq)
                            + 48.0 * cpow(k, 4) * (5.0 * cpow(q, 4) + 4.0 * cpow(q, 2) * wq)))
                * l2;
            let rq0l3 = 60.0
                * (-5.0
                    * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2) + cpow(q, 4))
                    * (cpow(emk, 4) - 4.0 * cpow(emk, 2) * (emk - 14.0 * cpow(k, 2)) * cpow(q, 2)
                        + 2.0
                            * (3.0 * cpow(emk, 2) - 56.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * cpow(q, 4)
                        - 4.0 * (emk - 14.0 * cpow(k, 2)) * cpow(q, 6)
                        + cpow(q, 8))
                    + 24.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 4)
                            - 4.0 * cpow(emk, 2) * (emk - 10.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (3.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 40.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 4.0 * (emk - 10.0 * cpow(k, 2)) * cpow(q, 6)
                            + cpow(q, 8))
                        * wq
                    - 45.0
                        * (cpow(emk, 4)
                            - 4.0 * cpow(emk, 2) * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2)
                                    + 8.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 4.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 6)
                            + cpow(q, 8))
                        * cpow(wq, 2)
                    + 40.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 3)
                    - 15.0
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 4))
                * l3;
            rq[0] = rq00 + rq0l1 + rq0l2 + rq0l3;
            (amp, rq)
        }
    };
    bpr_sum("bpr2", amp, &rq)
}

/// Port of FEFF `bpr3`: the lower-branch broadened-pole integrand.
pub fn self_energy_bpr3_integrand(
    input: SelfEnergyIntegrandInput,
) -> Result<Complex, SelfEnergyError> {
    let BprContext {
        q,
        k,
        en: _,
        emk,
        wq,
        gam,
    } = bpr_context(input, true)?;
    let (amp, rq) = if (wq - gam).re > 0.0 {
        {
            let mut rq = [Complex::new(0.0, 0.0); 13];
            let a1 = (emk - gam) - (2.0e0 * k * q + cpow(q, 2) - wq);
            let a2 = (emk + gam) + (2.0e0 * k * q - cpow(q, 2) + wq);
            let a3 = (emk - gam) + (2.0e0 * k * q - cpow(q, 2) + wq);
            let a4 = (emk + gam) - (2.0e0 * k * q + cpow(q, 2) - wq);
            let l1 = bpr_log_abs((a1 / a4) * (a2 / a3))? + log_i(a1, -1)? + log_i(a2, -1)?
                - log_i(a3, -1)?
                - log_i(a4, -1)?;
            let l2 = bpr_log_abs((a1 / a4) * (a3 / a2))? + log_i(a1, -1)? - log_i(a2, -1)?
                + log_i(a3, -1)?
                - log_i(a4, -1)?;
            let l3 = bpr_log_abs((a1 / a3) * (a4 / a2))? + log_i(a1, -1)?
                - log_i(a2, -1)?
                - log_i(a3, -1)?
                + log_i(a4, -1)?;
            let amp = 1.0e0 / (96.0e0 * cpow(gam, 4) * q * (cpow(gam, 2) + 7.0e0 * cpow(wq, 2)));
            rq[12] = (21.0 * l1 * cpow((cpow(q, 2) - 1.0 * wq), 5) * (5.0 * cpow(q, 2) + wq)) / gam;
            rq[11] = (1260.0 * k * l2 * cpow(q, 3) * cpow((cpow(q, 2) - 1.0 * wq), 4)) / gam;
            rq[10] = (-630.0
                * l1
                * cpow(q, 2)
                * cpow((cpow(q, 2) - 1.0 * wq), 3)
                * (emk * (cpow(q, 2) - 1.0 * wq) + 2.0 * cpow(k, 2) * (-5.0 * cpow(q, 2) + wq)))
                / gam;
            rq[9] = (84.0
                * k
                * q
                * cpow((cpow(q, 2) - 1.0 * wq), 2)
                * (-25.0 * (2.0 * gam + 3.0 * emk * l2 - 8.0 * cpow(k, 2) * l2) * cpow(q, 4)
                    + 2.0
                        * (26.0 * gam + 45.0 * emk * l2 - 40.0 * cpow(k, 2) * l2)
                        * cpow(q, 2)
                        * wq
                    - 1.0 * (2.0 * gam + 15.0 * emk * l2) * cpow(wq, 2)))
                / gam;
            rq[8] = (105.0
                * l1
                * (cpow(q, 2) - 1.0 * wq)
                * (3.0
                    * cpow(emk, 2)
                    * cpow((cpow(q, 2) - 1.0 * wq), 2)
                    * (5.0 * cpow(q, 2) - 1.0 * wq)
                    - 1.0
                        * cpow(gam, 2)
                        * cpow((cpow(q, 2) - 1.0 * wq), 2)
                        * (3.0 * cpow(q, 2) + wq)
                    + 48.0 * cpow(k, 4) * (5.0 * cpow(q, 6) - 3.0 * cpow(q, 4) * wq)
                    - 48.0
                        * emk
                        * cpow(k, 2)
                        * cpow(q, 2)
                        * (5.0 * cpow(q, 4) - 7.0 * cpow(q, 2) * wq + 2.0 * cpow(wq, 2))))
                / gam;
            rq[7] = (168.0
                * k
                * q
                * (cpow(q, 2)
                    * (24.0 * cpow(k, 4) * l2 * cpow(q, 2) * (5.0 * cpow(q, 2) - 4.0 * wq)
                        - 8.0
                            * gam
                            * cpow(k, 2)
                            * (25.0 * cpow(q, 2) - 13.0 * wq)
                            * (cpow(q, 2) - 1.0 * wq)
                        - 15.0 * cpow(gam, 2) * l2 * cpow((cpow(q, 2) - 1.0 * wq), 2))
                    + 15.0
                        * cpow(emk, 2)
                        * l2
                        * (5.0 * cpow(q, 2) - 2.0 * wq)
                        * cpow((cpow(q, 2) - 1.0 * wq), 2)
                    - 4.0
                        * emk
                        * (cpow(q, 2) - 1.0 * wq)
                        * (15.0 * cpow(k, 2) * l2 * cpow(q, 2) * (5.0 * cpow(q, 2) - 3.0 * wq)
                            + gam
                                * (-25.0 * cpow(q, 4) + 32.0 * cpow(q, 2) * wq
                                    - 7.0 * cpow(wq, 2)))))
                / gam;
            rq[6] = (420.0
                * l1
                * (-1.0
                    * cpow(emk, 3)
                    * (5.0 * cpow(q, 2) - 2.0 * wq)
                    * cpow((cpow(q, 2) - 1.0 * wq), 2)
                    + 18.0
                        * cpow(emk, 2)
                        * cpow(k, 2)
                        * cpow(q, 2)
                        * (5.0 * cpow(q, 4) - 8.0 * cpow(q, 2) * wq + 3.0 * cpow(wq, 2))
                    + 3.0
                        * emk
                        * cpow(q, 2)
                        * (cpow(gam, 2) * cpow((cpow(q, 2) - 1.0 * wq), 2)
                            - 8.0 * cpow(k, 4) * (5.0 * cpow(q, 4) - 4.0 * cpow(q, 2) * wq))
                    + 2.0
                        * cpow(k, 2)
                        * cpow(q, 2)
                        * (8.0 * cpow(k, 4) * cpow(q, 4)
                            - 3.0
                                * cpow(gam, 2)
                                * (3.0 * cpow(q, 4) - 4.0 * cpow(q, 2) * wq + cpow(wq, 2)))))
                / gam;
            rq[5] = (168.0
                * k
                * q
                * (-5.0
                    * (15.0 * cpow(emk, 3) * l2
                        + 30.0 * cpow(emk, 2) * (gam - 2.0 * cpow(k, 2) * l2)
                        + 4.0
                            * gam
                            * (-2.0 * cpow(gam, 2)
                                + 4.0 * cpow(k, 4)
                                + 3.0 * gam * cpow(k, 2) * l2)
                        + emk
                            * (-80.0 * gam * cpow(k, 2) - 9.0 * cpow(gam, 2) * l2
                                + 24.0 * cpow(k, 4) * l2))
                    * cpow(q, 4)
                    + 4.0
                        * (-11.0 * cpow(gam, 3)
                            + 19.0 * emk * gam * (3.0 * emk - 4.0 * cpow(k, 2))
                            + 30.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * l2
                            - 5.0 * cpow(gam, 2) * (3.0 * emk - 2.0 * cpow(k, 2)) * l2)
                        * cpow(q, 2)
                        * wq
                    - 1.0
                        * (78.0 * cpow(emk, 2) * gam - 4.0 * cpow(gam, 3)
                            + 45.0 * cpow(emk, 3) * l2
                            - 15.0 * emk * cpow(gam, 2) * l2)
                        * cpow(wq, 2)))
                / gam;
            rq[4] = (315.0
                * l1
                * ((cpow(gam, 4)
                    - 2.0
                        * cpow(gam, 2)
                        * (3.0 * cpow(emk, 2) - 24.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                    + 5.0
                        * cpow(emk, 2)
                        * (cpow(emk, 2) - 16.0 * emk * cpow(k, 2) + 16.0 * cpow(k, 4)))
                    * cpow(q, 4)
                    - 8.0
                        * emk
                        * (cpow(emk, 2) * (emk - 8.0 * cpow(k, 2))
                            - 1.0 * cpow(gam, 2) * (emk - 4.0 * cpow(k, 2)))
                        * cpow(q, 2)
                        * wq
                    + (emk - 1.0 * gam)
                        * (emk + gam)
                        * (3.0 * cpow(emk, 2) + cpow(gam, 2))
                        * cpow(wq, 2)))
                / gam;
            rq[3] = (28.0
                * k
                * q
                * (5.0
                    * (120.0 * cpow(emk, 2) * gam * (emk - 2.0 * cpow(k, 2))
                        - 32.0 * cpow(gam, 3) * (3.0 * emk - 2.0 * cpow(k, 2))
                        + 9.0 * cpow(gam, 4) * l2
                        + 15.0 * cpow(emk, 3) * (3.0 * emk - 8.0 * cpow(k, 2)) * l2
                        - 18.0 * emk * cpow(gam, 2) * (3.0 * emk - 4.0 * cpow(k, 2)) * l2)
                    * cpow(q, 2)
                    - 12.0
                        * emk
                        * (38.0 * cpow(emk, 2) * gam - 22.0 * cpow(gam, 3)
                            + 15.0 * cpow(emk, 3) * l2
                            - 15.0 * emk * cpow(gam, 2) * l2)
                        * wq))
                / gam;
            rq[2] = (-630.0
                * (emk - 1.0 * gam)
                * (emk + gam)
                * (cpow(emk, 2) * (emk - 10.0 * cpow(k, 2))
                    - 1.0 * cpow(gam, 2) * (emk - 2.0 * cpow(k, 2)))
                * l1
                * cpow(q, 2)
                + 168.0
                    * (3.0 * cpow(emk, 5) * l1 - 5.0 * cpow(emk, 3) * cpow(gam, 2) * l1
                        + 2.0 * cpow(gam, 5) * l3)
                    * wq)
                / gam;
            rq[1] = (-84.0
                * k
                * (50.0 * cpow(emk, 4) * gam - 80.0 * cpow(emk, 2) * cpow(gam, 3)
                    + 22.0 * cpow(gam, 5)
                    + 15.0 * cpow(emk, 5) * l2
                    - 30.0 * cpow(emk, 3) * cpow(gam, 2) * l2
                    + 15.0 * emk * cpow(gam, 4) * l2)
                * q)
                / gam;
            rq[0] = (105.0 * cpow((cpow(emk, 2) - 1.0 * cpow(gam, 2)), 3) * l1) / gam;
            (amp, rq)
        }
    } else {
        {
            let mut rq = [Complex::new(0.0, 0.0); 13];
            let amp = 7.0e0
                / (120.0e0
                    * q
                    * cpow((gam + wq), 5)
                    * (8.0e0 * cpow(gam, 2) - 5.0e0 * gam * wq + cpow(wq, 2)));
            let a1 = emk + gam - 2.0 * k * q - cpow(q, 2) + wq;
            let a2 = emk + gam + 2.0 * k * q - cpow(q, 2) + wq;
            let a3 = emk - 2.0 * k * q - cpow(q, 2);
            let a4 = emk + 2.0 * k * q - cpow(q, 2);
            let l1 = bpr_log_abs(a1 / a2)? + log_i(a1, -1)? - log_i(a2, -1)?;
            let l2 = bpr_log_abs((a1 / a3) * (a2 / a4))? + log_i(a1, -1)? + log_i(a2, -1)?
                - log_i(a3, -1)?
                - log_i(a4, -1)?;
            let l3 = bpr_log_abs((a1 / a2) * (a4 / a3))? + log_i(a1, -1)?
                - log_i(a2, -1)?
                - log_i(a3, -1)?
                + log_i(a4, -1)?;
            rq[6] = 300.0 * cpow(gam, 6) * l1;
            rq[5] = -240.0 * cpow(gam, 5) * (11.0 * k * q - 4.0 * l1 * wq);
            rq[4] = 60.0
                * cpow(gam, 4)
                * (-15.0
                    * l3
                    * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2) + cpow(q, 4))
                    + 2.0
                        * k
                        * q
                        * (5.0 * (-5.0 + 6.0 * l2) * (emk - 1.0 * cpow(q, 2)) - 38.0 * wq)
                    + 15.0 * l1 * cpow(wq, 2));
            rq[3] = 320.0
                * cpow(gam, 3)
                * k
                * q
                * (10.0
                    * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                        + 4.0 * cpow(k, 2) * cpow(q, 2)
                        + 3.0 * cpow(q, 4))
                    + 33.0 * (emk - 1.0 * cpow(q, 2)) * wq
                    + 3.0 * cpow(wq, 2));
            rq[2] = 60.0
                * cpow(gam, 2)
                * (-5.0 * l1 * cpow(wq, 4)
                    + 40.0 * (k * (-1.0 + 3.0 * l2) * cpow(q, 7) + cpow(emk, 3) * l3 * wq)
                    + 4.0
                        * k
                        * q
                        * (-10.0
                            * (-1.0 + 3.0 * l2)
                            * (cpow(emk, 3)
                                - 1.0 * emk * (3.0 * emk - 4.0 * cpow(k, 2)) * cpow(q, 2)
                                + (3.0 * emk - 4.0 * cpow(k, 2)) * cpow(q, 4))
                            - 2.0
                                * (-11.0 + 10.0 * l2)
                                * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                                    + 4.0 * cpow(k, 2) * cpow(q, 2)
                                    + 3.0 * cpow(q, 4))
                                * wq
                            + (-77.0 + 30.0 * l2) * (-1.0 * emk + cpow(q, 2)) * cpow(wq, 2)
                            + 16.0 * cpow(wq, 3))
                    + 5.0
                        * l3
                        * (3.0 * (cpow(emk, 4) + cpow(q, 8)) + 6.0 * cpow(emk, 2) * cpow(wq, 2)
                            - 4.0 * cpow(q, 6) * (3.0 * emk + 2.0 * (-9.0 * cpow(k, 2) + wq))
                            - 12.0
                                * cpow(q, 2)
                                * (emk + wq)
                                * (cpow(emk, 2) - 2.0 * cpow(k, 2) * wq
                                    + emk * (-6.0 * cpow(k, 2) + wq))
                            + 6.0
                                * cpow(q, 4)
                                * (3.0 * cpow(emk, 2) + 8.0 * cpow(k, 4)
                                    - 16.0 * cpow(k, 2) * wq
                                    + cpow(wq, 2)
                                    + 4.0 * emk * (-6.0 * cpow(k, 2) + wq))));
            rq[1] = -240.0
                * gam
                * k
                * q
                * (5.0
                    * (5.0 * cpow(emk, 4)
                        - 20.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                        + 2.0
                            * (15.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * cpow(q, 4)
                        - 20.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 6)
                        + 5.0 * cpow(q, 8))
                    + 76.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * wq
                    + 26.0
                        * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                            + 4.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * cpow(q, 4))
                        * cpow(wq, 2)
                    + 28.0 * (emk - 1.0 * cpow(q, 2)) * cpow(wq, 3)
                    + cpow(wq, 4));
            let rq00 = 40.0
                * k
                * q
                * (-30.0
                    * (5.0 * cpow(emk, 4)
                        - 20.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                        + 2.0
                            * (15.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                            * cpow(q, 4)
                        - 20.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 6)
                        + 5.0 * cpow(q, 8))
                    * wq
                    - 516.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 2)
                    - 208.0
                        * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                            + 4.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * cpow(q, 4))
                        * cpow(wq, 3)
                    - 291.0 * (emk - 1.0 * cpow(q, 2)) * cpow(wq, 4)
                    - 30.0 * cpow(wq, 5));
            let rq0l1 = 60.0 * cpow(wq, 6) * l2;
            let rq0l2 = -240.0
                * k
                * q
                * (-5.0
                    * (emk - 1.0 * cpow(q, 2))
                    * (cpow(emk, 2) - 2.0 * (emk - 6.0 * cpow(k, 2)) * cpow(q, 2) + cpow(q, 4))
                    * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                        + 4.0 * cpow(k, 2) * cpow(q, 2)
                        + 3.0 * cpow(q, 4))
                    - 12.0
                        * (5.0 * cpow(emk, 4)
                            - 20.0 * cpow(emk, 2) * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + 2.0
                                * (15.0 * cpow(emk, 2) - 40.0 * emk * cpow(k, 2)
                                    + 8.0 * cpow(k, 4))
                                * cpow(q, 4)
                            - 20.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 6)
                            + 5.0 * cpow(q, 8))
                        * wq
                    - 90.0
                        * (emk - 1.0 * cpow(q, 2))
                        * (cpow(emk, 2) - 2.0 * (emk - 2.0 * cpow(k, 2)) * cpow(q, 2)
                            + cpow(q, 4))
                        * cpow(wq, 2)
                    - 20.0
                        * (3.0 * cpow(emk, 2) - 6.0 * emk * cpow(q, 2)
                            + 4.0 * cpow(k, 2) * cpow(q, 2)
                            + 3.0 * cpow(q, 4))
                        * cpow(wq, 3)
                    - 15.0 * (emk - 1.0 * cpow(q, 2)) * cpow(wq, 4))
                * l2;
            let rq0l3 = 60.0
                * (-5.0 * (cpow(emk, 6) + cpow(q, 12))
                    - 24.0 * cpow(emk, 5) * wq
                    - 45.0 * cpow(emk, 4) * cpow(wq, 2)
                    - 40.0 * cpow(emk, 3) * cpow(wq, 3)
                    - 15.0 * cpow(emk, 2) * cpow(wq, 4)
                    + 6.0 * cpow(q, 10) * (5.0 * emk - 50.0 * cpow(k, 2) + 4.0 * wq)
                    - 15.0
                        * cpow(q, 8)
                        * (5.0 * cpow(emk, 2) - 80.0 * emk * cpow(k, 2)
                            + 80.0 * cpow(k, 4)
                            + 8.0 * emk * wq
                            - 64.0 * cpow(k, 2) * wq
                            + 3.0 * cpow(wq, 2))
                    - 15.0
                        * cpow(q, 4)
                        * (emk + wq)
                        * (5.0
                            * emk
                            * (cpow(emk, 2) - 16.0 * emk * cpow(k, 2) + 16.0 * cpow(k, 4))
                            + (11.0 * cpow(emk, 2) - 112.0 * emk * cpow(k, 2)
                                + 48.0 * cpow(k, 4))
                                * wq
                            + (7.0 * emk - 32.0 * cpow(k, 2)) * cpow(wq, 2)
                            + cpow(wq, 3))
                    + 20.0
                        * cpow(q, 6)
                        * (5.0 * cpow(emk, 3) - 90.0 * cpow(emk, 2) * cpow(k, 2)
                            + 120.0 * emk * cpow(k, 4)
                            - 16.0 * cpow(k, 6)
                            + 12.0
                                * (cpow(emk, 2) - 12.0 * emk * cpow(k, 2) + 8.0 * cpow(k, 4))
                                * wq
                            + 9.0 * (emk - 6.0 * cpow(k, 2)) * cpow(wq, 2)
                            + 2.0 * cpow(wq, 3))
                    + 30.0
                        * cpow(q, 2)
                        * cpow((emk + wq), 3)
                        * (cpow(emk, 2) - 2.0 * cpow(k, 2) * wq + emk * (-10.0 * cpow(k, 2) + wq)))
                * l3;
            rq[0] = rq00 + rq0l1 + rq0l2 + rq0l3;
            (amp, rq)
        }
    };
    bpr_sum("bpr3", amp, &rq)
}
