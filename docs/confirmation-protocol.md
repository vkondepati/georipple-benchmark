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

## Clarifications (added 2026-11-13, after the run started and before any result was read)

Code reviewers found edge cases the text above did not settle. These rules change no
method or setting; they say how the analysis treats cases the code could otherwise not
compute.

- A record whose paired difference is undefined (for example recall when no dealer has a
  hazard-attributable stockout) is dropped from that comparison only. Network means use
  the remaining records, and the record and network counts are reported.
- The Holm family is fixed at the six hypotheses. An undefined p-value counts as 1. When
  every network mean of a difference is identical, p is 1 if that value is 0 and 0
  otherwise.
- A batch that stops before writing its file may be rerun unchanged at the same commit.
  An artifact in which any MILP did not reach its 1% gap fails the artifact check; that
  condition is rerun unchanged on an unloaded machine, and a repeat failure is reported.
  Any code change after a confirmation file exists is a deviation and is disclosed.
- Two known properties of the benchmark, present in the development and confirmation
  runs alike: (1) for hazard family 0, the forecast ensemble's random stream
  (`default_rng([seed, 0])`) is the same stream that generates the network
  (`default_rng(seed)`), so those forecast perturbations are tied to supplier placement;
  the held-out event and the transit delays use separate streams. (2) On road lanes,
  candidate routing closes every road piece that touches the buffered forecast union,
  but the short local legs at each end are not rerouted; any exposure they pick up is
  charged to the lane during planning and execution.

## Run log

- 2026-11-13 05:56-06:10 UTC, first attempt at f374dae: all six raw runs (development,
  sensitivity, C0-C3) were started at once on a machine with four performance cores. Seven
  comparator plans (four B4, three A2; no GeoRipple plan) hit the 60 s MILP limit, in the
  development file and in C0-C3. Under the rule above those five files failed the artifact
  check and were not analyzed.
- 2026-11-13 06:12-06:25 UTC, rerun of those five files, unchanged at f374dae, at most two
  at a time: every MILP reached its 1% gap (slowest plan 27.2 s). The sensitivity file from
  the first attempt passed (slowest plan 59.4 s) and is kept.
- Tables were generated with the analysis at 54f59c0; the development records still match
  the published artifact for every unchanged method (`make gate`).
