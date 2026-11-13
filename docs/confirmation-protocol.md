# Confirmation protocol (written before the confirmation run)

This file is committed together with the code that produces the confirmation artifacts,
and before any of them exist. `results/confirm/*.json` record the commit they were run
from; it must be this file's commit or a later one that changes nothing under
`georipple/` or `experiments/run_benchmark.py`.

## Why a confirmation run

The ten development networks (seeds 0-9) were used to build and debug the method. On them
we found that shipping the MILP's existing-lane flows exactly as written (absolute
dispatch) withholds normal deliveries wherever the conservative forecast is wrong, and
that additive dispatch (each existing lane ships the larger of its normal quantity and
the planned flow) avoids this. Because that finding came from those networks, they cannot
confirm it. The confirmation uses ten networks that had not been generated before the
code was frozen.

## Design

- Networks: seeds 10-19 (`--seed-offset 10 --seeds 10`), three fixed hazard families,
  three delay draws per plan, one plan per network and hazard: 90 records per condition.
- Conditions, all on the same ten networks:
  - C0 straight lanes (x1.25), one DC per dealer (the development design);
  - C1 lanes routed on U.S. primary roads, one DC per dealer;
  - C2 straight lanes, two DCs per dealer (70/30);
  - C3 road lanes, two DCs per dealer.
  C1-C3 are paired robustness conditions, not independent replications.
- Every other setting is unchanged from development: M = 8 forecast members, K = 10,
  k = 3, radii 600/1,500 km, tau_max = 4 days, shortage penalty 100 CU/unit, 1% MILP gap,
  60 s time limit, upper-quartile exposure for the full planner.
- Each condition is run once (`make confirm`). Runs execute in a clean checkout and must
  pass `experiments/check_artifacts.py` (complete, clean provenance, every MILP solved to
  the gap).

## Hypotheses (condition C0)

Each is a two-sided paired t test over the ten network means; p-values are Holm-adjusted
across H1-H6 at 0.05. Intervals are reported unadjusted.

| | Quantity | Predicted sign |
|---|---|---|
| H1 | recall(GeoRipple) - recall(B2 node-only) | > 0 |
| H2 | MAE(GeoRipple) - MAE(A1 no buffers) | < 0 |
| H3 | unmet(GeoRipple) - unmet(no action) | < 0 |
| H4 | unmet(GeoRipple) - unmet(A4 absolute dispatch) | < 0 |
| H5 | unmet(GeoRipple) - unmet(B4 mean-exposure MILP) | < 0 |
| H6 | unmet(GeoRipple) - unmet(B3 action-matched greedy) | < 0 |

Secondary, descriptive: realized objective (cost + 0.1 kCU per unmet unit), cost, and a
false-alarm arm in which GeoRipple, A4, and B4 are executed with the same delays but no
hazard.

## Decision rule

Every quantity is reported whatever its sign. A hypothesis counts as confirmed only if its
Holm-adjusted p is below 0.05 and the effect has the predicted sign. If unmet demand falls
but the realized objective or the false-alarm harm rises, both are reported and no net
benefit is claimed. No method or setting may be changed after seeing these results.
