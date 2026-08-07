# Gate metric decision

**Recorded before any champion model was trained.** The decision is written down
first so that it cannot be, or appear to be, chosen to suit a result.

Evidence: `reports/baselines_calibrate.json`, calibrate block, 290,346 rows per
horizon. Test block untouched.

## The problem

The baseline ladder produced two metrics that disagree about which model is
better.

| Model | MAE | Poisson deviance | Volume bias |
|---|---|---|---|
| Seasonal naive (A) | 5.935 | 8.446 | 0.990 |
| Hour-of-week median (B) | **5.374** | 5.180 | 0.972 |
| Poisson GLM (C) | 5.224 | **2.624** | **1.000** |

Against Baseline B, the GLM improves MAE by 2.8% but deviance by 49.3%. In the
low tier the GLM is *worse* on MAE (1.087 against 1.064) and far better on
deviance (1.436 against 4.967).

## Why they disagree

MAE is minimised by the conditional **median**; Poisson deviance by the
conditional **mean**. Hourly pickup counts are right-skewed and 45.9% of the grid
is zero, so for sparse zones the median sits well below the mean. Baseline B is
literally a trailing median, so it is near-optimal for MAE by construction.

At higher demand rates a Poisson median and mean coincide on the integers, so
this is specifically a sparse-demand effect — and 138 of 217 forecastable zones
are in the low tier.

## What settled it

Volume bias, which neither MAE nor WAPE can see:

| Model | High | Medium | Low |
|---|---|---|---|
| Hour-of-week median (B) | 0.979 | 0.929 | **0.879** |
| Poisson GLM (C) | 1.001 | 0.989 | 0.993 |

Baseline B forecasts 12.1% fewer pickups than occur in low-demand zones, and 7.1%
fewer in medium. It achieves its MAE advantage by predicting the typical hour
well while consistently missing demand overall.

For the stated use — positioning vehicles ahead of demand — that is the failure
mode that matters. A dispatcher acting on Baseline B under-supplies the outer
boroughs every day, and no absolute-error metric would ever reveal it.

## Decision

1. **Promotion gate: mean Poisson deviance**, measured on the frozen test set,
   high and medium tiers. It matches the training objective, is defined at zero
   counts, does not explode on small denominators, and is minimised by the
   conditional mean — the quantity an operator acts on.

2. **Threshold unchanged at ≥10% relative improvement** over Baseline B.

3. **Volume bias is a hard constraint, not a headline.** A champion must satisfy
   |bias − 1| ≤ 0.05 in every reported tier. A model may not buy deviance with
   systematic over-forecasting.

4. **MAE, WAPE, RMSE and zero-hit rate remain reported** at every tier. MAE is
   the more intuitive figure and belongs in the write-up; it is simply not the
   gate, and the reason is this document.

5. **Baseline B still ships if the gate is not cleared.** Spec §8 gate 3 is a
   two-way gate and this decision does not change that.

## Consequence to state plainly in the model card

Under the original spec wording — a 10% MAE improvement over Baseline B — no
model was ever going to pass, because MAE structurally favours a median
predictor on sparse count data. Discovering that *after* seeing a champion's
results and then changing the metric would be indistinguishable from moving the
goalposts. It was found from the baselines, which is why it is recorded here,
with the test block still unread.
