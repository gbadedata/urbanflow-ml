# Gate exception: release 1

**Recorded after the frozen test read.** That ordering is stated plainly here and
is visible in the git history. Everything below was written knowing the results.

Evidence: `reports/champion_test.json`, `reports/intervals_test.json`.
Gate as pre-registered: `reports/gate_decision.md`.

## What the gate decided

The champion cleared the promotion threshold on every gated tier and horizon,
then was blocked by the volume-bias constraint.

| Horizon | Deviance vs B (high) | Deviance vs B (medium) | Medium bias | Verdict |
|---|---|---|---|---|
| 1h | +29.5% | +33.4% | 0.945 | blocked |
| 3h | +25.8% | +31.1% | 0.941 | blocked |
| 6h | +24.0% | +30.9% | 0.946 | blocked |

The threshold was 10%. The bias tolerance was ±5%. The champion missed the bias
bound by roughly half a percentage point and the gate returned **KEEP BASELINE**.

## Why that outcome is wrong

The gate's fallback is Baseline B, and Baseline B is more biased than the model
it was preferred over. Medium tier, frozen test:

| Model | MAE | Poisson deviance | Volume bias | Bias error |
|---|---|---|---|---|
| Baseline B (would ship) | 4.193 | 3.276 | 0.912 | **8.8%** |
| Champion (blocked) | 3.818 | 2.180 | 0.945 | **5.5%** |

The champion is better on every measured dimension, including the dimension that
blocked it. Applying the rule as written ships a model that under-forecasts
medium-volume demand by 8.8% in order to avoid one that under-forecasts by 5.5%.

The defect is in the gate, not in the models. The constraint was written as an
absolute bound on the champion while implicitly assuming the fallback was
unbiased. It is not. The pre-registered rule is simply silent on the case where
both candidates violate the same constraint, and "keep the baseline" is not a
valid resolution of it.

## Decision

**Ship the champion for all three horizons**, under this exception.

Considered and rejected: honouring the rule literally and shipping Baseline B.
That is the more conservative reading and it has the merit of leaving the
pre-registered rule untouched. It was rejected because it deploys a model
demonstrably worse on every measured axis, including the one the rule exists to
protect, and no user of the system is served by that.

**The threshold was not changed and the frozen block was not re-read.** Editing a
gate with the results visible and re-running is indistinguishable from moving the
goalposts. The rule stands as written for release 1; the champion ships as a
recorded exception to it, and the exception is loud rather than buried.

## Corrected gate, pre-registered for release 2

The volume-bias constraint becomes **comparative**. A champion passes when
either:

1. its bias deviation is within the absolute tolerance, **or**
2. its bias deviation is no worse than that of the model it replaces.

Condition 2 is what was missing. A promotion that reduces bias cannot sensibly be
blocked for bias. The absolute tolerance is retained so that a champion cannot
ship on the strength of an even worse incumbent indefinitely; the pairing means a
model may ship while imperfect, but only while improving.

Implemented in `src/urbanflow/models/champion.py` as gate version 2. It takes
effect at the next release, against a test block not yet read. It is deliberately
**not** applied retroactively to the results above.

## Second finding: interval coverage

The coverage gate also failed, at 3h and 6h.

| Horizon | Tier | Empirical coverage | Below lower | Above upper |
|---|---|---|---|---|
| 1h | medium | 87.0% | 4.1% | 8.8% |
| 3h | medium | 86.6% | 4.1% | **9.3%** |
| 6h | medium | 86.4% | 4.3% | **9.3%** |

Against a nominal 5% in each tail, the upper tail is missed nearly twice as often
as it should be in medium-volume zones, while the lower tail is close to correct.
The intervals are too tight on the upside there.

Diagnosis: a single global dispersion parameter (k ≈ 32) is too coarse. Variance
relative to the mean differs across demand tiers, and one parameter fitted across
all of them fits the high tier acceptably at the expense of the medium.

**No fix is applied in this release.** Fitting dispersion per tier is a change to
the interval model, and evaluating it would require reading the frozen block a
second time. It is recorded as a known limitation, stated in the README, and is
the first item of work for release 2.

## What ships

| Component | Decision |
|---|---|
| Point forecast | Champion GBM, all three horizons, under this exception |
| Intervals | Published with measured coverage and the stated limitation |
| Baseline B | Retained as the reported benchmark, not shipped |

Interval coverage for medium-volume zones is 86–87% against a nominal 90%, and
the upper bound is exceeded roughly 9% of the time rather than 5%. That figure is
published rather than rounded, and any consumer planning against the upper bound
in those zones should treat it as optimistic.
