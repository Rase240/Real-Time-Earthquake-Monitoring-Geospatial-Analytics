# TerraWatcher model evaluation

_Generated automatically by `train_model.py` on 2026-10-18T21:35:45Z. Do not edit by hand._

## What is being predicted

For an earthquake that has been recorded with magnitude >= 4.5, the model estimates the probability that its magnitude is >= 5.0, from its location, depth, the number of M>=4.5 events within +/-1.0 degree in the preceding 30 days, and its distance to the nearest mapped plate boundary. It does not estimate whether, when or where an earthquake will occur.

* Catalogue: USGS FDSN event service, M >= 4.5, 2024-11-28 to 2026-10-18 (15855 events, 15362 usable samples after the 30-day warm-up).
* Positive label: magnitude >= 5.0. Training base rate: 0.252.

## Chronological split

| Split | Samples | Positive rate | From | To |
|---|---:|---:|---|---|
| train | 10753 | 0.252 | 2024-11-28 | 2026-03-23 |
| validation | 2304 | 0.241 | 2026-03-23 | 2026-07-11 |
| test | 2305 | 0.272 | 2026-07-11 | 2026-10-18 |

## Test-period results (newest 15%, never used for training or early stopping)

| Model | ROC-AUC | Average precision | Brier | Log loss |
|---|---:|---:|---:|---:|
| MLP (this model) | 0.5703 | 0.3312 | 0.1970 | 0.5837 |
| Constant train base rate | 0.5000 | 0.2716 | 0.1982 | 0.5859 |
| 5-degree regional rate (train history) | 0.5242 | 0.2946 | 0.1989 | 0.5882 |
| Logistic regression (same features) | 0.5524 | 0.3228 | 0.1967 | 0.5819 |

Brier skill score vs. constant base rate: **+0.0061** (0 = no better than always predicting the base rate; negative = worse).

### Decision threshold 0.152 (chosen to maximise F1 on validation, applied to test)

Precision 0.278 · Recall 0.971 · F1 0.432 · Accuracy 0.305

| | Predicted < M5 | Predicted >= M5 |
|---|---:|---:|
| Actual < M5 | 96 | 1583 |
| Actual >= M5 | 18 | 608 |

Accuracy of always predicting '< M5' on test: 0.728.

### Calibration on test

| Predicted bin | n | Mean predicted | Observed rate |
|---|---:|---:|---:|
| 0.0-0.1 | 28 | 0.091 | 0.250 |
| 0.1-0.2 | 689 | 0.173 | 0.228 |
| 0.2-0.3 | 1307 | 0.243 | 0.277 |
| 0.3-0.4 | 228 | 0.325 | 0.342 |
| 0.4-0.5 | 53 | 0.446 | 0.415 |

## Limitations

* Conditional on an event having occurred: the score is not a probability that an earthquake will happen.
* Magnitude types differ across networks (mb, Mww, ml...); the label mixes them as reported by USGS.
* Aftershock sequences that straddle split boundaries make adjacent periods correlated.
* Latitude/longitude are raw inputs, so the model partly memorises regional behaviour; it is not validated on regions absent from training.
* Evaluated on a single recent test period; results can change with each retraining.
