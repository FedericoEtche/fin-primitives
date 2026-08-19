//! Credit default swap (CDS) pricing with hazard rate models.
//!
//! Implements survival probability curves, CDS NPV, par spread, and CS01
//! using piecewise-constant hazard rates bootstrapped from market spreads.

/// Piecewise-flat hazard rate curve for credit default modeling.
#[derive(Debug, Clone)]
pub struct HazardCurve {
    /// Tenors in years (must be sorted ascending).
    pub tenors: Vec<f64>,
    /// Hazard rates (instantaneous default intensity) per tenor segment.
    pub hazard_rates: Vec<f64>,
}

impl HazardCurve {
    /// Survival probability Q(0, t) = exp(-integral_0^t h(s) ds).
    ///
    /// Uses piecewise-constant hazard rates with trapezoidal integration over knots.
    pub fn survival_probability(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let n = self.tenors.len();
        if n == 0 {
            return 1.0;
        }

        // Trapezoidal integration over the knot points
        let mut integral = 0.0;
        let mut t_prev = 0.0;
        let mut h_prev = self.hazard_rates[0]; // flat extrapolation before first knot

        for i in 0..n {
            let t_i = self.tenors[i];
            let h_i = self.hazard_rates[i];

            if t <= t_i {
                // We finish integration in this segment
                let h_interp = h_prev + (h_i - h_prev) * (t - t_prev) / (t_i - t_prev).max(1e-12);
                // Trapezoidal on [t_prev, t]
                integral += 0.5 * (h_prev + h_interp) * (t - t_prev);
                return (-integral).exp();
            }

            // Full segment [t_prev, t_i]
            integral += 0.5 * (h_prev + h_i) * (t_i - t_prev);
            t_prev = t_i;
            h_prev = h_i;
        }

        // t > last knot: flat extrapolation
        integral += h_prev * (t - t_prev);
        (-integral).exp()
    }

    /// Default probability: 1 - survival_probability(t).
    pub fn default_probability(&self, t: f64) -> f64 {
        1.0 - self.survival_probability(t)
    }

    /// Hazard curve from the flat credit-triangle approximation.
    ///
    /// `h_i = spread_i / (1 - recovery)`, applied independently per tenor.
    ///
    /// This is a closed-form approximation, not a bootstrap: it ignores
    /// discounting, the premium-leg accrual, and survival through earlier
    /// segments. It is accurate to first order for flat curves and short
    /// tenors, and degrades as the term structure steepens.
    ///
    /// Prefer [`HazardCurve::bootstrap_from_par_spreads`], which solves each
    /// segment so the CDS actually reprices to par.
    pub fn from_par_spreads_approx(tenors: &[f64], spreads: &[f64], recovery: f64) -> Self {
        assert_eq!(tenors.len(), spreads.len());
        let denom = (1.0 - recovery).max(1e-8);
        HazardCurve {
            tenors: tenors.to_vec(),
            hazard_rates: spreads.iter().map(|s| s / denom).collect(),
        }
    }

    /// Deprecated alias for [`HazardCurve::from_par_spreads_approx`].
    ///
    /// The former documentation claimed a sequential bootstrap "adjusted for
    /// prior-period survival", but the implementation applied the flat credit
    /// triangle independently per tenor. The behaviour is unchanged here; only
    /// the name and documentation now match what it does.
    #[deprecated(
        since = "0.1.1",
        note = "renamed: use `from_par_spreads_approx` for the credit-triangle \
                approximation, or `bootstrap_from_par_spreads` for a true bootstrap"
    )]
    pub fn from_par_spreads(tenors: &[f64], spreads: &[f64], recovery: f64) -> Self {
        Self::from_par_spreads_approx(tenors, spreads, recovery)
    }

    /// Bootstrap a hazard curve from market CDS par spreads.
    ///
    /// Walks the tenors in ascending order and, for each one, solves for the
    /// piecewise-flat hazard rate on `(t_{i-1}, t_i]` that reprices that
    /// tenor's CDS to par, holding all earlier segments fixed. Earlier
    /// segments therefore feed through as survival into every later solve,
    /// which is what makes this a bootstrap rather than a per-tenor formula.
    ///
    /// Each segment is solved by bisection on the CDS NPV, which is monotone
    /// in the hazard rate: raising it raises the protection leg and lowers the
    /// premium leg, so NPV to the protection buyer is strictly increasing.
    ///
    /// `payment_frequency` is the premium payment frequency of the quoted
    /// contracts (4 for the market-standard quarterly CDS).
    ///
    /// Returns the credit-triangle approximation for any segment that fails to
    /// bracket a root — a defensive fallback that should not occur for
    /// well-formed, positive, arbitrage-free quotes.
    ///
    /// # Panics
    /// If `tenors` and `spreads` differ in length, or `tenors` is not sorted
    /// ascending with all entries strictly positive.
    pub fn bootstrap_from_par_spreads(
        tenors: &[f64],
        spreads: &[f64],
        recovery: f64,
        payment_frequency: u32,
        risk_free: &[(f64, f64)],
    ) -> Self {
        assert_eq!(
            tenors.len(),
            spreads.len(),
            "tenors and spreads must have equal length"
        );
        assert!(
            tenors.windows(2).all(|w| w[0] < w[1]),
            "tenors must be strictly ascending"
        );
        assert!(
            tenors.iter().all(|&t| t > 0.0),
            "tenors must be strictly positive"
        );

        let n = tenors.len();
        let freq = payment_frequency.max(1);
        let mut curve = HazardCurve {
            tenors: Vec::with_capacity(n),
            hazard_rates: Vec::with_capacity(n),
        };

        for i in 0..n {
            let spec = CdsSpec {
                notional: 1.0,
                premium_rate: spreads[i],
                tenor_years: tenors[i],
                recovery_rate: recovery,
                payment_frequency: freq,
            };

            // NPV to the protection buyer as a function of this segment's rate,
            // with all earlier segments already pinned.
            let npv_at = |h: f64| -> f64 {
                let mut trial = curve.clone();
                trial.tenors.push(tenors[i]);
                trial.hazard_rates.push(h);
                cds_npv(&spec, &trial, risk_free, true)
            };

            // Bracket: NPV is increasing in h, negative at h = 0 (buyer pays
            // premium for no protection). Grow the upper bound until it turns.
            let mut lo = 0.0_f64;
            let mut hi = (spreads[i] / (1.0 - recovery).max(1e-8)).max(1e-4) * 2.0;
            let mut bracketed = false;
            for _ in 0..60 {
                if npv_at(hi) >= 0.0 {
                    bracketed = true;
                    break;
                }
                lo = hi;
                hi *= 2.0;
                if hi > 50.0 {
                    break;
                }
            }

            let h_solved = if bracketed {
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if npv_at(mid) > 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                    if (hi - lo).abs() < 1e-12 {
                        break;
                    }
                }
                0.5 * (lo + hi)
            } else {
                // Defensive: fall back to the closed-form approximation rather
                // than emitting a rate from an unconverged solve.
                spreads[i] / (1.0 - recovery).max(1e-8)
            };

            curve.tenors.push(tenors[i]);
            curve.hazard_rates.push(h_solved);
        }

        curve
    }
}

/// Specification for a credit default swap.
#[derive(Debug, Clone)]
pub struct CdsSpec {
    /// Notional principal.
    pub notional: f64,
    /// Premium (spread) rate paid by protection buyer (annualized).
    pub premium_rate: f64,
    /// CDS tenor in years.
    pub tenor_years: f64,
    /// Recovery rate on default (e.g. 0.4 = 40%).
    pub recovery_rate: f64,
    /// Number of premium payments per year.
    pub payment_frequency: u32,
}

/// Discount factor computation from a set of (tenor, zero_rate) pairs.
fn df(t: f64, risk_free: &[(f64, f64)]) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    if risk_free.is_empty() {
        return (-0.05 * t).exp(); // fallback 5% flat
    }
    // Linear interpolation
    let n = risk_free.len();
    if t <= risk_free[0].0 {
        return (-risk_free[0].1 * t).exp();
    }
    if t >= risk_free[n - 1].0 {
        return (-risk_free[n - 1].1 * t).exp();
    }
    for i in 0..n - 1 {
        let (t0, r0) = risk_free[i];
        let (t1, r1) = risk_free[i + 1];
        if t >= t0 && t <= t1 {
            let alpha = (t - t0) / (t1 - t0);
            let r = r0 * (1.0 - alpha) + r1 * alpha;
            return (-r * t).exp();
        }
    }
    (-risk_free[n - 1].1 * t).exp()
}

/// Present value of the protection leg.
///
/// PV = integral_0^T (1 - R) * df(t) * (-dQ(t)) dt
/// Numerically integrated using 100 steps over the tenor.
pub fn protection_leg_pv(
    spec: &CdsSpec,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
) -> f64 {
    let n_steps = 100usize;
    let dt = spec.tenor_years / n_steps as f64;
    let lgd = (1.0 - spec.recovery_rate) * spec.notional;

    let mut pv = 0.0;
    let mut q_prev = hazard.survival_probability(0.0);

    for k in 1..=n_steps {
        let t = k as f64 * dt;
        let q_t = hazard.survival_probability(t);
        let dq = q_prev - q_t; // probability of default in [t-dt, t]
        let t_mid = t - 0.5 * dt;
        let discount = df(t_mid, risk_free);
        pv += lgd * discount * dq;
        q_prev = q_t;
    }
    pv
}

/// Present value of the premium leg.
///
/// PV = sum over payment dates of: spread * notional * dcf * df(t_k) * Q(t_k)
pub fn premium_leg_pv(
    spec: &CdsSpec,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
) -> f64 {
    let n_payments = (spec.tenor_years * spec.payment_frequency as f64).round() as u32;
    let dt = 1.0 / spec.payment_frequency as f64;
    let dcf = dt; // simplified: actual fraction ≈ 1/frequency

    let mut pv = 0.0;
    for k in 1..=n_payments {
        let t = k as f64 * dt;
        let discount = df(t, risk_free);
        let survival = hazard.survival_probability(t);
        pv += spec.notional * spec.premium_rate * dcf * discount * survival;
    }
    pv
}

/// Net present value of the CDS.
///
/// `protection_buyer = true`: long protection (pay premium, receive on default).
pub fn cds_npv(
    spec: &CdsSpec,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
    protection_buyer: bool,
) -> f64 {
    let prot = protection_leg_pv(spec, hazard, risk_free);
    let prem = premium_leg_pv(spec, hazard, risk_free);
    if protection_buyer {
        prot - prem
    } else {
        prem - prot
    }
}

/// Risky annuity: the PV of 1 unit of annual running spread paid on a
/// notional of 1, surviving to each payment date.
///
/// `risky_annuity = sum_k df(t_k) * Q(t_k) * accrual_k`
///
/// This is the denominator that converts a PV into a running spread. It is
/// the survival-weighted analogue of a plain annuity, and it is what makes
/// the conversion correct for a distressed credit — where an unweighted
/// annuity (or a yield-based DV01) materially overstates the discounting of
/// spread payments that will not in fact be made.
///
/// See [`risky_pv01`] for the per-basis-point, notional-scaled form, and
/// [`spread_equivalent_bps`] for the conversion itself.
pub fn risky_annuity(
    tenor_years: f64,
    payment_frequency: u32,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
) -> f64 {
    if tenor_years <= 0.0 || payment_frequency == 0 {
        return 0.0;
    }
    let n_payments = (tenor_years * payment_frequency as f64).round() as u32;
    let dt_pay = 1.0 / payment_frequency as f64;
    let mut annuity = 0.0;
    for k in 1..=n_payments {
        let t = k as f64 * dt_pay;
        annuity += df(t, risk_free) * hazard.survival_probability(t) * dt_pay;
    }
    annuity
}

/// Risky PV01: change in PV for a 1bp change in running spread.
///
/// `risky_pv01 = notional * 0.0001 * risky_annuity`
///
/// Use this to express an option or protection PV as a running spread:
///
/// ```text
/// spread_bps = option_pv / risky_pv01
/// ```
pub fn risky_pv01(
    notional: f64,
    tenor_years: f64,
    payment_frequency: u32,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
) -> f64 {
    notional * 1e-4 * risky_annuity(tenor_years, payment_frequency, hazard, risk_free)
}

/// Convert a present value into an equivalent running spread, in basis points.
///
/// Returns `None` when the risky PV01 is zero or non-finite, which happens for
/// a zero tenor or a curve that has already defaulted with certainty — cases
/// where no running spread can express the PV and returning a number would be
/// misleading.
pub fn spread_equivalent_bps(pv: f64, risky_pv01_value: f64) -> Option<f64> {
    if !risky_pv01_value.is_finite() || risky_pv01_value <= 0.0 {
        return None;
    }
    Some(pv / risky_pv01_value)
}

/// Par spread: the spread that sets CDS NPV to zero.
///
/// `par_spread = protection_pv / (notional * risky_annuity)`
pub fn par_spread(
    spec: &CdsSpec,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
) -> f64 {
    let prot = protection_leg_pv(spec, hazard, risk_free);
    let annuity = risky_annuity(
        spec.tenor_years,
        spec.payment_frequency,
        hazard,
        risk_free,
    );
    if annuity == 0.0 {
        return 0.0;
    }
    prot / (spec.notional * annuity)
}

/// CS01: sensitivity of CDS NPV to a 1bp (0.0001) parallel shift in the hazard curve.
pub fn cds_cs01(
    spec: &CdsSpec,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
) -> f64 {
    let bump = 0.0001;
    let bumped_rates: Vec<f64> = hazard.hazard_rates.iter().map(|h| h + bump).collect();
    let bumped_hazard = HazardCurve {
        tenors: hazard.tenors.clone(),
        hazard_rates: bumped_rates,
    };
    let npv_base = cds_npv(spec, hazard, risk_free, true);
    let npv_bumped = cds_npv(spec, &bumped_hazard, risk_free, true);
    npv_bumped - npv_base
}

/// Complete CDS valuation result.
#[derive(Debug, Clone)]
pub struct CdsValuation {
    /// Net present value (positive = in the money for the holder).
    pub npv: f64,
    /// Present value of the protection leg.
    pub protection_pv: f64,
    /// Present value of the premium leg.
    pub premium_pv: f64,
    /// Par spread (spread that sets NPV to zero).
    pub par_spread: f64,
    /// CS01 (dollar value of 1bp hazard rate shift).
    pub cs01: f64,
    /// Survival probability at maturity.
    pub survival_at_maturity: f64,
}

/// Compute a full CDS valuation.
pub fn value_cds(
    spec: &CdsSpec,
    hazard: &HazardCurve,
    risk_free: &[(f64, f64)],
    protection_buyer: bool,
) -> CdsValuation {
    let protection_pv = protection_leg_pv(spec, hazard, risk_free);
    let premium_pv = premium_leg_pv(spec, hazard, risk_free);
    let npv = if protection_buyer {
        protection_pv - premium_pv
    } else {
        premium_pv - protection_pv
    };
    CdsValuation {
        npv,
        protection_pv,
        premium_pv,
        par_spread: par_spread(spec, hazard, risk_free),
        cs01: cds_cs01(spec, hazard, risk_free),
        survival_at_maturity: hazard.survival_probability(spec.tenor_years),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_rf(rate: f64) -> Vec<(f64, f64)> {
        vec![
            (0.5, rate), (1.0, rate), (2.0, rate),
            (3.0, rate), (5.0, rate), (7.0, rate), (10.0, rate),
        ]
    }

    #[test]
    fn test_survival_probability_at_zero_is_one() {
        let hazard = HazardCurve {
            tenors: vec![1.0, 2.0, 5.0],
            hazard_rates: vec![0.02, 0.025, 0.03],
        };
        assert_eq!(hazard.survival_probability(0.0), 1.0);
    }

    #[test]
    fn test_survival_probability_decreasing() {
        let hazard = HazardCurve {
            tenors: vec![1.0, 3.0, 5.0],
            hazard_rates: vec![0.03, 0.04, 0.05],
        };
        let q1 = hazard.survival_probability(1.0);
        let q3 = hazard.survival_probability(3.0);
        let q5 = hazard.survival_probability(5.0);
        assert!(q1 > q3, "Survival should decrease over time");
        assert!(q3 > q5, "Survival should decrease over time");
        assert!(q5 > 0.0, "Survival probability should be positive");
    }

    #[test]
    fn test_par_spread_matches_input_flat_curve() {
        // For a flat hazard curve h, par_spread ≈ h * (1 - R)
        let recovery = 0.4;
        let spread = 0.02; // 200bp
        let hazard = HazardCurve::from_par_spreads_approx(
            &[1.0, 2.0, 3.0, 5.0],
            &[spread; 4],
            recovery,
        );
        let spec = CdsSpec {
            notional: 1_000_000.0,
            premium_rate: spread,
            tenor_years: 3.0,
            recovery_rate: recovery,
            payment_frequency: 4,
        };
        let rf = flat_rf(0.05);
        let ps = par_spread(&spec, &hazard, &rf);
        // The par spread should be close to our input spread
        assert!(
            (ps - spread).abs() / spread < 0.5,
            "Par spread {} should be close to input spread {}",
            ps,
            spread
        );
    }

    #[test]
    fn test_protection_exceeds_premium_high_hazard() {
        // High hazard rate → protection leg PV > premium leg PV
        let hazard = HazardCurve {
            tenors: vec![1.0, 3.0, 5.0],
            hazard_rates: vec![0.30, 0.30, 0.30], // very high default probability
        };
        let spec = CdsSpec {
            notional: 1_000_000.0,
            premium_rate: 0.01, // only 100bp spread
            tenor_years: 3.0,
            recovery_rate: 0.4,
            payment_frequency: 4,
        };
        let rf = flat_rf(0.05);
        let prot = protection_leg_pv(&spec, &hazard, &rf);
        let prem = premium_leg_pv(&spec, &hazard, &rf);
        assert!(prot > prem, "Protection PV {} should exceed premium PV {} for high hazard", prot, prem);
    }

    #[test]
    fn test_from_par_spreads_approx_returns_positive_rates() {
        let tenors = vec![1.0, 2.0, 3.0, 5.0];
        let spreads = vec![0.01, 0.015, 0.018, 0.022];
        let hazard = HazardCurve::from_par_spreads_approx(&tenors, &spreads, 0.4);
        assert_eq!(hazard.hazard_rates.len(), 4);
        // All hazard rates should be positive
        for h in &hazard.hazard_rates {
            assert!(*h > 0.0, "Hazard rates should be positive");
        }
    }

    // ---- risky annuity / PV01 / spread conversion -------------------------

    #[test]
    fn risky_annuity_is_positive_and_below_riskless_annuity() {
        let rf = flat_rf(0.03);
        let hazard = HazardCurve {
            tenors: vec![1.0, 3.0, 5.0],
            hazard_rates: vec![0.02, 0.02, 0.02],
        };
        let risky = risky_annuity(5.0, 4, &hazard, &rf);
        // Same schedule with certain survival.
        let no_default = HazardCurve {
            tenors: vec![1.0, 3.0, 5.0],
            hazard_rates: vec![0.0, 0.0, 0.0],
        };
        let riskless = risky_annuity(5.0, 4, &no_default, &rf);
        assert!(risky > 0.0);
        assert!(
            risky < riskless,
            "survival weighting must reduce the annuity: {risky} vs {riskless}"
        );
    }

    #[test]
    fn risky_annuity_decreases_as_hazard_rises() {
        let rf = flat_rf(0.03);
        let mut prev = f64::INFINITY;
        for h in [0.0, 0.01, 0.05, 0.15] {
            let curve = HazardCurve {
                tenors: vec![5.0],
                hazard_rates: vec![h],
            };
            let a = risky_annuity(5.0, 4, &curve, &rf);
            assert!(a < prev, "annuity must fall as hazard rises (h={h})");
            prev = a;
        }
    }

    #[test]
    fn risky_pv01_scales_with_notional() {
        let rf = flat_rf(0.03);
        let curve = HazardCurve {
            tenors: vec![5.0],
            hazard_rates: vec![0.02],
        };
        let a = risky_pv01(1_000_000.0, 5.0, 4, &curve, &rf);
        let b = risky_pv01(2_000_000.0, 5.0, 4, &curve, &rf);
        assert!((b - 2.0 * a).abs() < 1e-9);
    }

    #[test]
    fn risky_annuity_zero_for_degenerate_schedule() {
        let rf = flat_rf(0.03);
        let curve = HazardCurve { tenors: vec![5.0], hazard_rates: vec![0.02] };
        assert_eq!(risky_annuity(0.0, 4, &curve, &rf), 0.0);
        assert_eq!(risky_annuity(5.0, 0, &curve, &rf), 0.0);
    }

    #[test]
    fn spread_equivalent_bps_inverts_risky_pv01() {
        let rf = flat_rf(0.03);
        let curve = HazardCurve { tenors: vec![5.0], hazard_rates: vec![0.02] };
        let notional = 1_000_000.0;
        let pv01 = risky_pv01(notional, 5.0, 4, &curve, &rf);
        // A PV worth exactly 250bp of running spread must convert back to 250.
        let pv = 250.0 * pv01;
        let bps = spread_equivalent_bps(pv, pv01).expect("finite pv01");
        assert!((bps - 250.0).abs() < 1e-6, "got {bps}");
    }

    #[test]
    fn spread_equivalent_bps_returns_none_when_pv01_is_unusable() {
        assert!(spread_equivalent_bps(100.0, 0.0).is_none());
        assert!(spread_equivalent_bps(100.0, -1.0).is_none());
        assert!(spread_equivalent_bps(100.0, f64::NAN).is_none());
    }

    #[test]
    fn par_spread_equals_protection_pv_over_risky_pv01_in_bps() {
        // par_spread and the risky-PV01 conversion must agree, since
        // par_spread was refactored to share risky_annuity.
        let rf = flat_rf(0.03);
        let curve = HazardCurve {
            tenors: vec![1.0, 3.0, 5.0],
            hazard_rates: vec![0.015, 0.02, 0.025],
        };
        let spec = CdsSpec {
            notional: 10_000_000.0,
            premium_rate: 0.01,
            tenor_years: 5.0,
            recovery_rate: 0.4,
            payment_frequency: 4,
        };
        let prot = protection_leg_pv(&spec, &curve, &rf);
        let pv01 = risky_pv01(spec.notional, spec.tenor_years, spec.payment_frequency, &curve, &rf);
        let via_conversion = spread_equivalent_bps(prot, pv01).unwrap() / 10_000.0;
        let direct = par_spread(&spec, &curve, &rf);
        assert!(
            (via_conversion - direct).abs() < 1e-12,
            "conversion {via_conversion} vs par_spread {direct}"
        );
    }

    #[test]
    fn risky_annuity_matches_hand_computed_closed_form() {
        // Flat r = 4%, flat hazard h = 2%, 5y annual payments, accrual 1.0:
        //   annuity = sum over t=1..5 of exp(-0.04 t) * exp(-0.02 t) * 1.0
        //           = sum over t=1..5 of exp(-0.06 t)
        //           = 4.191401263460949
        // Both curves are flat, so the piecewise-linear interpolation in df()
        // and survival_probability() reproduces the closed form exactly.
        let rf = flat_rf(0.04);
        let curve = HazardCurve { tenors: vec![5.0], hazard_rates: vec![0.02] };
        let expected: f64 = (1..=5).map(|t| (-0.06 * t as f64).exp()).sum();
        assert!((expected - 4.191401263460949).abs() < 1e-12);
        let annuity = risky_annuity(5.0, 1, &curve, &rf);
        assert!(
            (annuity - expected).abs() < 1e-10,
            "risky_annuity {annuity} vs hand-computed {expected}"
        );
        let pv01 = risky_pv01(1.0, 5.0, 1, &curve, &rf);
        assert!(
            (pv01 - expected * 1e-4).abs() < 1e-14,
            "risky_pv01 {pv01} vs hand-computed {}",
            expected * 1e-4
        );
    }

    // ---- genuine bootstrap ------------------------------------------------

    #[test]
    fn bootstrap_reprices_every_quoted_tenor_to_par() {
        // The defining property, and the one the credit-triangle
        // approximation does not have.
        let rf = flat_rf(0.03);
        let tenors = vec![1.0, 3.0, 5.0, 7.0, 10.0];
        let spreads = vec![0.0080, 0.0120, 0.0150, 0.0165, 0.0175];
        let recovery = 0.4;
        let curve =
            HazardCurve::bootstrap_from_par_spreads(&tenors, &spreads, recovery, 4, &rf);

        for (i, &t) in tenors.iter().enumerate() {
            let spec = CdsSpec {
                notional: 1.0,
                premium_rate: spreads[i],
                tenor_years: t,
                recovery_rate: recovery,
                payment_frequency: 4,
            };
            let npv = cds_npv(&spec, &curve, &rf, true);
            assert!(
                npv.abs() < 1e-8,
                "tenor {t}y should reprice to par, NPV = {npv}"
            );
            let reimplied = par_spread(&spec, &curve, &rf);
            assert!(
                (reimplied - spreads[i]).abs() < 1e-9,
                "tenor {t}y: re-implied {reimplied} vs quoted {}",
                spreads[i]
            );
        }
    }

    #[test]
    fn bootstrap_100_150_200_reprices_each_tenor_to_par() {
        // Definition of par: a CDS quoted at the bootstrapped curve's input
        // spread has zero NPV at that curve.
        let rf = flat_rf(0.04);
        let tenors = vec![1.0, 3.0, 5.0];
        let spreads = vec![0.0100, 0.0150, 0.0200]; // 100 / 150 / 200 bps
        let recovery = 0.4;
        let curve =
            HazardCurve::bootstrap_from_par_spreads(&tenors, &spreads, recovery, 4, &rf);

        for (i, &t) in tenors.iter().enumerate() {
            let spec = CdsSpec {
                notional: 1.0,
                premium_rate: spreads[i],
                tenor_years: t,
                recovery_rate: recovery,
                payment_frequency: 4,
            };
            let npv = cds_npv(&spec, &curve, &rf, true);
            assert!(
                npv.abs() < 1e-8,
                "tenor {t}y quoted at {} bps should reprice to par, NPV = {npv}",
                spreads[i] * 1e4
            );
        }
    }

    #[test]
    fn bootstrap_beats_credit_triangle_on_an_upward_sloping_curve() {
        let rf = flat_rf(0.03);
        let tenors = vec![1.0, 3.0, 5.0, 10.0];
        let spreads = vec![0.0050, 0.0110, 0.0160, 0.0200];
        let recovery = 0.4;

        let boot = HazardCurve::bootstrap_from_par_spreads(&tenors, &spreads, recovery, 4, &rf);
        let approx = HazardCurve::from_par_spreads_approx(&tenors, &spreads, recovery);

        let err = |c: &HazardCurve| -> f64 {
            tenors
                .iter()
                .enumerate()
                .map(|(i, &t)| {
                    let spec = CdsSpec {
                        notional: 1.0,
                        premium_rate: spreads[i],
                        tenor_years: t,
                        recovery_rate: recovery,
                        payment_frequency: 4,
                    };
                    cds_npv(&spec, c, &rf, true).abs()
                })
                .sum()
        };
        let (e_boot, e_approx) = (err(&boot), err(&approx));
        assert!(
            e_boot < e_approx,
            "bootstrap repricing error {e_boot} should beat approximation {e_approx}"
        );
        assert!(e_boot < 1e-7, "bootstrap should reprice essentially exactly");
    }

    #[test]
    fn bootstrap_forward_hazards_exceed_flat_triangle_on_steep_curve() {
        // On an upward-sloping curve the later forward hazard must exceed the
        // flat per-tenor triangle rate, because earlier low-hazard segments
        // have to be compensated for.
        let rf = flat_rf(0.03);
        let tenors = vec![1.0, 5.0];
        let spreads = vec![0.0040, 0.0200];
        let recovery = 0.4;
        let boot = HazardCurve::bootstrap_from_par_spreads(&tenors, &spreads, recovery, 4, &rf);
        let triangle = spreads[1] / (1.0 - recovery);
        assert!(
            boot.hazard_rates[1] > triangle,
            "bootstrapped 5y forward hazard {} should exceed triangle {triangle}",
            boot.hazard_rates[1]
        );
    }

    #[test]
    fn bootstrap_survival_is_monotone_decreasing() {
        let rf = flat_rf(0.03);
        let tenors = vec![1.0, 3.0, 5.0, 10.0];
        let spreads = vec![0.0080, 0.0120, 0.0150, 0.0175];
        let curve = HazardCurve::bootstrap_from_par_spreads(&tenors, &spreads, 0.4, 4, &rf);
        let mut prev = 1.0;
        for t in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] {
            let q = curve.survival_probability(t);
            assert!(q > 0.0 && q <= 1.0, "survival out of range at {t}: {q}");
            assert!(q <= prev + 1e-12, "survival must not increase at {t}");
            prev = q;
        }
    }

    #[test]
    fn bootstrap_matches_triangle_closely_on_a_flat_curve() {
        // With a flat quote the two methods should broadly agree; this is the
        // regime where the approximation is defensible.
        let rf = flat_rf(0.0);
        let tenors = vec![1.0, 3.0, 5.0];
        let spreads = vec![0.01, 0.01, 0.01];
        let recovery = 0.4;
        let boot = HazardCurve::bootstrap_from_par_spreads(&tenors, &spreads, recovery, 4, &rf);
        let triangle = 0.01 / (1.0 - recovery);
        for h in &boot.hazard_rates {
            assert!(
                (h - triangle).abs() < 0.002,
                "flat-curve bootstrap {h} should be near triangle {triangle}"
            );
        }
    }
}
