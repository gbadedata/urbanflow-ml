# Model card: UrbanFlow demand forecaster

| Field | Value |
|---|---|
| Model name | `gbm_poisson` |
| Version | `h{1,3,6}-26c6cf6a154f-20260807` |
| Type | Gradient boosted regression trees, Poisson loss |
| Implementation | `sklearn.ensemble.HistGradientBoostingRegressor` |
| Gate version applied | 1, with a recorded exception |
| Snapshot | `yellow-2023-01-2025-06-26c6cf6a154f` |
| Artefacts | `models/champion_h{1,3,6}.joblib` |

Three separate models are trained, one per horizon. They share a feature
definition and differ only in how far back their availability anchor sits.

---

## 1. Intended use

### Primary use

Estimating expected yellow taxi pickup counts for a New York City taxi zone
during a specific future hour, to support fleet positioning and staffing
decisions made before demand materialises.

### Intended users

Operations planners and dispatch analysts with access to a trip feed of
comparable latency to the one assumed here.

### Out of scope

| Use | Why not |
|---|---|
| Causal inference about demand drivers | The model is predictive; correlated features are not causes |
| Individual trip prediction | The prediction unit is a zone-hour aggregate |
| For-hire vehicle or green taxi demand | Trained on yellow taxi records only |
| Total city mobility demand | Yellow taxis are a declining minority of trips |
| Zones excluded by the volume floor | 43 zones receive no forecast at all |
| Deployment with different feed latency | Features would not mean what the model learned |

The final row is enforced rather than advised. Model bundles record the
availability cutoff they were trained under, and loading fails if the runtime is
configured differently.

---

## 2. Model architecture

| Parameter | Value |
|---|---|
| Loss | Poisson |
| Boosting iterations | 400 |
| Learning rate | 0.1 |
| Maximum leaf nodes | 63 |
| Minimum samples per leaf | 100 |
| L2 regularisation | 1.0 |
| Early stopping | Enabled, 10% validation fraction, patience 20 |
| Categorical features | `local_hour`, `local_dow`, `local_month` |
| Random state | 0 |

Poisson loss matches both the target, which is a non-negative count whose
variance rises with its mean, and the promotion metric. The model optimises the
quantity it is judged on rather than a proxy.

**Early stopping never triggered.** All three models ran to the 400 iteration cap
with internal validation loss still improving, which means they are under-trained
rather than over-trained. Raising the cap is the third item in the next
iteration.

### Feature transformation

Demand valued features are `log1p` transformed before fitting. Trees are
invariant to monotone transforms of a single feature, so this does not change
which splits are reachable. It is applied so the champion and the Poisson
regression baseline consume identical inputs and remain comparable.

---

## 3. Input features

19 features. Nothing else reaches the model.

| Feature | Anchored on | Definition |
|---|---|---|
| `recent_1`, `recent_2`, `recent_3` | Availability | Demand in the three most recent fully arrived hours |
| `lag_24h`, `lag_48h`, `lag_168h` | Target | Demand at the same clock hour 1, 2 and 7 days earlier |
| `roll_mean_3h`, `roll_mean_24h`, `roll_mean_168h` | Availability | Rolling means ending at the anchor |
| `zero_rate_168h` | Availability | Share of the past week with zero demand |
| `how_median_4w` | Target | Trailing four week median for this zone and hour of week |
| `local_hour`, `local_dow`, `local_month` | Target | Calendar position in America/New_York |
| `is_weekend` | Target | Weekend indicator |
| `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Target | Cyclical encodings |

### The availability constraint

At origin `as_of`, no feature may use data newer than `as_of` minus the feed
latency. With 15 minutes of latency and an origin on the hour, the newest fully
arrived hour is `as_of` minus two hours, because the intervening hour has not
finished.

| Horizon | Anchor |
|---|---|
| 1 hour | Target minus 3 hours |
| 3 hours | Target minus 5 hours |
| 6 hours | Target minus 8 hours |

**No lag-1h feature exists**, at any horizon. It would require data that has not
arrived. This is enforced in the feature builder and asserted by unit test.

### Prohibited inputs

Fare amount, tip, tolls, total amount, payment type, trip distance, rate code,
dropoff location and dropoff time. All are known only after a trip completes.
Trip distance and total amount are read during validation for duplicate detection
and never reach a model.

---

## 4. Training data

| Property | Value |
|---|---|
| Source | NYC TLC yellow taxi trip records |
| Window | January 2023 through June 2025 |
| Raw records | 103,563,330 |
| Records after validation | 102,956,835 |
| Grid rows | 4,749,479 |
| Feature rows per horizon | 4,603,655 |
| Training rows per horizon | 4,019,057 |
| Zones | 217 |

The window starts in 2023 deliberately. The 2020 and 2021 period is a structural
break, and yellow taxi volume has declined secularly, so older absolute levels do
not transfer.

Full data provenance is in `reports/data_card.md`.

---

## 5. Evaluation

### Protocol

Three temporal blocks with gaps at least as long as the longest forecast horizon.
Never a random split.

| Block | Rows | Purpose |
|---|---|---|
| Train | 4,019,057 | Fitting only |
| Calibrate | 290,346 | Model selection, dispersion, thresholds |
| Test | 291,648 | Read once, at release |

The test block covers the most recent eight weeks and was untouched until the
pipeline was complete and the metric choice was committed.

### Primary metric

Mean Poisson deviance, on the high and medium demand tiers.

The choice was recorded in `reports/gate_decision.md` before any champion model
was trained, because the baselines revealed that mean absolute error is minimised
by the conditional median while Poisson deviance is minimised by the conditional
mean. With 45.9% of the grid at zero, those differ substantially, and the
benchmark is itself a median estimator. Volume bias settled it: the median
baseline under-forecast low tier demand by 12.1% while posting the better MAE.

### Frozen test results, horizon 1 hour

| Model | Tier | MAE | Deviance | WAPE | Bias |
|---|---|---|---|---|---|
| Hour-of-week median (benchmark) | high | 19.760 | 7.280 | 0.174 | 0.994 |
| | medium | 4.193 | 3.276 | 0.328 | 0.912 |
| | low | 1.241 | 5.053 | 0.619 | 0.848 |
| Gradient boosting (this model) | high | 17.198 | 5.132 | 0.152 | 0.982 |
| | medium | 3.818 | 2.180 | 0.299 | 0.945 |
| | low | 1.183 | 1.397 | 0.591 | 0.983 |

### Improvement over the benchmark, all horizons

| Horizon | High | Medium | Low | All |
|---|---|---|---|---|
| 1 hour | +29.5% | +33.4% | +72.4% | +55.4% |
| 3 hours | +25.8% | +31.1% | +72.0% | +53.9% |
| 6 hours | +24.0% | +30.9% | +71.9% | +53.3% |

Relative reduction in Poisson deviance. The threshold was 10%.

### Performance by demand tier

Performance varies substantially by zone volume, and pooled figures conceal that.

| Tier | Zones | Mean pickups per hour | MAE (1h) | WAPE (1h) |
|---|---|---|---|---|
| High | 47 | 20 or more | 17.198 | 0.152 |
| Medium | 31 | 2 to 20 | 3.818 | 0.299 |
| Low | 138 | 0.1 to 2 | 1.183 | 0.591 |

WAPE of 0.591 in the low tier is not a failure of the model. With mean demand
near 1.5 pickups per hour, being wrong by one pickup is a 67% error and carries
almost no operational consequence. This is why percentage metrics are reported
but not gated for that tier.

### Horizon sensitivity

| Horizon | MAE | Deviance |
|---|---|---|
| 1 hour | 5.089 | 2.336 |
| 3 hours | 5.189 | 2.417 |
| 6 hours | 5.232 | 2.449 |

Degradation from one hour to six is under 3%. Most predictable signal is weekly
periodicity rather than short term momentum.

---

## 6. Prediction intervals

| Property | Value |
|---|---|
| Method | Negative binomial quantiles around the predicted mean |
| Nominal coverage | 90%, from the 5th and 95th percentiles |
| Dispersion parameter | `k` approximately 32, fitted on the calibrate block |
| Variance model | `Var = mu + mu^2 / k` |

Quantile gradient boosting was implemented first and abandoned. With 46% of the
grid at zero, pinball loss at the 0.05 quantile never escapes a lower bound of
zero, producing a one-sided interval that met its total coverage target while
conveying nothing about the downside.

### Measured coverage on the frozen test block

| Horizon | Tier | Coverage | Below lower | Above upper |
|---|---|---|---|---|
| 1 hour | high | 88.8% | 4.9% | 6.3% |
| | medium | 87.0% | 4.1% | 8.8% |
| 3 hours | high | 88.4% | 5.1% | 6.5% |
| | medium | 86.6% | 4.1% | 9.3% |
| 6 hours | high | 88.0% | 5.3% | 6.7% |
| | medium | 86.4% | 4.3% | 9.3% |

Nominal is 90% coverage with 5% in each tail.

**The coverage gate fails at 3 and 6 hours.** In medium volume zones the upper
bound is exceeded roughly 9% of the time against 5% expected. A single global
dispersion parameter is too coarse; variance relative to the mean differs by
tier. Users planning against the upper bound in medium volume zones should treat
it as optimistic by approximately four percentage points.

---

## 7. Promotion decision

The model was **blocked by the pre-registered gate and ships under a recorded
exception.**

The champion cleared the deviance threshold on every gated tier, then failed the
volume bias constraint at 0.945 in the medium tier against a 5% tolerance. The
gate's fallback, Baseline B, has a medium tier bias of 0.912, which is worse.

| Model | MAE | Deviance | Bias error |
|---|---|---|---|
| Baseline B (would ship) | 4.193 | 3.276 | 8.8% |
| Champion (blocked) | 3.818 | 2.180 | 5.5% |

The constraint was written as an absolute bound on the champion while implicitly
assuming an unbiased fallback. Applying it literally would ship a model worse on
every measured dimension, including the one the rule exists to protect.

The threshold was not changed and the frozen block was not re-read. The full
record, dated after the frozen read, is in `reports/gate_exception.md`. A
corrected comparative constraint is implemented as gate version 2 and takes
effect at the next release.

---

## 8. Ethical and operational considerations

| Consideration | Assessment |
|---|---|
| Personal data | None. The grain is zone-hour aggregate; no trip, driver or passenger is identifiable |
| Geographic equity | Performance is materially worse in low volume zones, which correlate with outer boroughs. Reported per tier rather than pooled so the disparity is visible |
| Feedback loops | Positioning vehicles by forecast may influence realised demand. Not modelled, and worth monitoring if deployed |
| Excluded zones | 43 zones receive no forecast. They carry 0.02% of trips, but a service using this model should not treat them as zero demand |
| Automation | The system produces estimates with uncertainty, not decisions. It has no authority to dispatch |

The geographic point deserves emphasis. Pooled metrics would hide that the model
serves Midtown considerably better than the outer boroughs. Tiered reporting is a
deliberate choice to keep that visible rather than a statistical convenience.

---

## 9. Caveats

| Caveat | Detail |
|---|---|
| Simulated latency | The 15 minute cutoff is an assumption. A real deployment must measure its feed and retrain if it differs |
| Under-trained | Early stopping never triggered at 400 iterations |
| No weather | Likely explains part of the variance captured by the dispersion parameter |
| Static model | No retraining schedule or drift detection is implemented |
| Single snapshot | Trained on one snapshot; TLC revises published files |
| Interval limitation | See section 6 |
| Structural change | Trained on 2023 to 2025 data. A regime shift comparable to 2020 would invalidate it |

---

## 10. Reproducing these figures

```bash
uv run urbanflow train
uv run urbanflow baselines --champion --frozen-test
uv run urbanflow intervals --frozen-test
```

Results are written to `reports/champion_test.json` and
`reports/intervals_test.json`. The frozen block should be read once. If you are
iterating, work on the calibrate block and read test only when finished.
