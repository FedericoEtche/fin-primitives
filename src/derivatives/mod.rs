//! Derivative pricing modules.
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`swaps`] | Interest rate swap pricing: discount curve, par rate, DV01, NPV |

pub mod swaps;

/// Futures pricing, basis analytics, calendar spreads, and roll yield.
pub mod futures;

/// Multi-leg option strategy analytics: straddle, strangle, collar, butterfly, P&L profiles.
pub mod option_strategies;

/// Exotic option pricing: barrier, Asian, lookback, and digital options.
pub mod exotic_options;

/// Variance swap pricing, realised variance tracking, and VIX replication.
pub mod variance_swap;

/// Forward contract pricing: equity, FX, commodity forwards, and forward curves.
pub mod forwards;

/// Short-rate models: Vasicek, CIR, and Hull-White with bond pricing and simulation.
pub mod interest_rate_models;

/// Credit Default Swap pricing: protection/premium legs, par spread, CS01, and implied hazard rates.
/// Second, standalone CDS implementation.
///
/// **Prefer [`crate::credit::cds`].** Two independent CDS implementations live
/// in this crate and they do not agree, because they model different things:
///
/// | | `credit::cds` | `derivatives::credit_default_swap` |
/// |---|---|---|
/// | Hazard | term structure (`HazardCurve`) | single flat rate |
/// | Discounting | zero-rate curve | single flat rate |
/// | Spread units | decimal (`premium_rate`) | basis points (`spread_bps`) |
/// | Bootstrap | yes, reprices to par | none |
/// | Tests | 17 | 0 |
///
/// `credit::cds` is the maintained one: it carries the survival curve, the
/// bootstrap, and the risky-PV01 conversion. This module is kept only so
/// existing callers keep compiling, and should be removed in the next
/// breaking release.
#[deprecated(
    since = "0.1.1",
    note = "use `credit::cds` instead: it supports a hazard term structure, a \
            discount curve, a genuine bootstrap, and risky PV01. This module \
            is a flat-rate duplicate with no test coverage."
)]
pub mod credit_default_swap;
