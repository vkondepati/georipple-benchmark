# GeoRipple benchmark

Reproducible code and synthetic results for *GeoRipple: Geographic and
Network Analysis for Predicting and Resolving Supply Chain Disruptions*
(Venkata Rao Kondepati, Ankit Hemant Lade, Dhruvil Patel).

The benchmark generates layered supply networks, draws every lane either as a
straight segment or as a route over U.S. primary roads, intersects nodes and
lanes with forecast hazard ensembles, predicts the day each dealer runs out,
selects recovery lanes with a time-expanded MILP, dispatches the plan, and
scores it under a held-out hazard with random transit delays. It does **not**
contain the Neo4j/Snowflake/React application sketched in the paper.

## Reproduce the artifact

Python 3.11 is the reference interpreter. Direct requirements are in
`requirements.txt`; the full environment is pinned with hashes in
`requirements.lock`.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
make check        # tests, lint, and the artifact gate
make all          # everything below, from a clean checkout
```

Step by step:

```bash
make bench confirm sensitivity  # raw runs into build/ (ignored)
make collect tables             # copy them into results/ and regenerate every table
make gate                       # compare results/runs.json with the published artifact
make figures                    # static map figures (straight and road lanes)
```

Raw runs write to `build/` first so that one run never sees another's output as a
modified file; run them from a clean checkout with the pinned environment.

Each run records the Python, platform, package, Git-commit, and worktree
provenance. `check_artifacts.py` rejects incomplete runs, runs from a modified
checkout or another environment, confirmation files that hold the wrong seeds
or condition, raw files from different commits, any MILP that did not reach
its 1% gap, and tables that no longer match their records. Results are deterministic for the pinned environment;
floating-point and solver differences on another platform can move numbers
slightly.

`make roads TIGER_ZIP=path/to/tl_2023_us_primaryroads.zip` rebuilds the road
graph from the Census download (see `data/README.md`); the result is
byte-identical.

## What is in `results/`

| File | Contents |
|---|---|
| `runs.json`, `summary.json`, `tables.tex` | Development networks (seeds 0-9), 90 records |
| `confirm/C0.json` ... `confirm/C3.json` | Confirmation networks (seeds 10-19), 90 records per condition |
| `confirm/C*/tables.tex`, `confirm/C*/summary.json` | Per-condition tables, paired effects, Holm-adjusted hypothesis tests |
| `conditions.tex`, `conditions_summary.json` | Cross-condition confirmation table |
| `sensitivity_runs.json`, `sensitivity_summary.json`, `sensitivity.tex` | One-factor-at-a-time study under both dispatch rules |

The pre-registered protocol for the confirmation run is in
`docs/confirmation-protocol.md`; it was committed before any confirmation
artifact existed.

## Layout

| Path | Purpose |
|---|---|
| `georipple/network.py` | Synthetic supplier-DC-dealer network; optional dual sourcing |
| `georipple/lanes.py` | Lane models (straight, primary roads), hazard-avoiding routers, lane enrichment |
| `georipple/roads.py` | U.S. primary-road graph: loading, nearest-point access, shortest paths |
| `georipple/hazards.py` | Stylized footprints, forecast ensembles, node and lane exposure |
| `georipple/simulate.py` | Daily inventory and flow simulator with transit and receiving queues |
| `georipple/predict.py` | Wavefront prediction, baselines B1/B2/A1, STCI |
| `georipple/routing.py` | Visibility-graph router used by the straight lane model |
| `georipple/resolve.py` | Recovery candidates, MILP, additive and absolute dispatch, greedy B3 |
| `georipple/evaluate.py` | Execution under the held-out event, metrics, miss decomposition |
| `experiments/run_benchmark.py` | Runner (`--lanes`, `--dealer-dcs`, `--seed-offset`) and provenance |
| `experiments/make_tables.py` | Network-clustered intervals, paired effects, Holm tests |
| `experiments/make_conditions.py` | Confirmation table across C0-C3 |
| `experiments/run_sensitivity.py`, `make_sensitivity.py` | Sensitivity study |
| `experiments/make_figures.py` | Static map figures for one development network |
| `experiments/prepare_roads.py` | Builds `data/us-primary-roads.json.gz` from Census TIGER/Line |
| `experiments/check_artifacts.py` | Artifact gate used by `make check` |
| `paper/` | Pre-review manuscript draft and its figures; the camera-ready source is not kept here |

## Evaluation design

Development networks (seeds 0-9) were used to build and debug the method. On them
we found that shipping the MILP's existing-lane flows exactly as written (absolute
dispatch, ablation A4) withholds normal deliveries wherever the cautious forecast is
wrong. GeoRipple therefore dispatches additively: each existing lane ships the larger
of its normal quantity and the planned flow. To test that fairly, the code was frozen
and ten new networks (seeds 10-19) were generated and run once under four conditions:

| Condition | Lanes | DCs per dealer |
|---|---|---|
| C0 | straight x1.25 | 1 |
| C1 | U.S. primary roads | 1 |
| C2 | straight x1.25 | 2 (70/30) |
| C3 | U.S. primary roads | 2 (70/30) |

Six hypotheses on C0 were tested with paired t tests over network means and a Holm
correction; C1-C3 are paired robustness conditions on the same networks.
Intervals are descriptive t intervals over network means, because records from one
network are not independent.

Interpretation limits:

- Facilities are jittered around metro centers; the networks are synthetic.
- Hazards are stylized, and the held-out event comes from the same family as the
  forecast, so this tests generalization within a hazard family.
- The road graph treats every crossing of two primary roads as an interchange.
- Networks have three layers, so betweenness carries little information (see below).
- The linked graph/map application is proposed, not implemented or user-tested.

## Current headline results

On the ten confirmation networks (seeds 10-19, condition C0), analyzed as
written in `docs/confirmation-protocol.md`:

- **Prediction.** Precision 0.83, recall 0.67, stockout-day error 8.7 days.
  Lane exposure adds 0.047 recall over node-only exposure (95% interval 0.013
  to 0.080), almost all of it in the river flood; inventory buffers cut the
  timing error by 7.5 days.
- **Resolution.** GeoRipple's plans cut system-wide unmet demand from 1,133 to
  786 units (-347, interval -438 to -255) for 18.0k synthetic cost units, and
  leave less unmet demand than the mean-exposure MILP (990) and the
  action-matched greedy policy (1,327).
- **Dispatch.** The same plans shipped exactly as written (absolute dispatch,
  A4) leave 1,238 units unmet. Run with no hazard at all, they cause 846 units
  of shortage, while additively dispatched plans change unmet demand by less
  than one unit.
- All six pre-registered hypotheses are confirmed (Holm-adjusted p from
  1.4e-6 to 0.012).
- With road-routed lanes, dual-sourced dealers, or both (C1-C3), unmet demand
  falls 30-31% and every comparison keeps its sign; lane exposure helps more on
  real roads (+0.062 recall against +0.047).
- The STCI structural term changed the top-K set in 5 of 120 plans and never
  moved unmet demand by more than one unit; in 88 of the 120 plans, K or fewer
  facilities were exposed at all.
- 54% of missed stockouts do not happen when the held-out event is replayed
  without transit delays, so they need a forecast that samples delays.

On the development networks, where the dispatch problem was found, no action
leaves 899 units unmet, GeoRipple 661, and absolute dispatch 1,053. All
intervals are in `results/conditions_summary.json` and
`results/confirm/C*/summary.json`.

## Using operational data

- replace `base_hazards()` with observed, projected GeoJSON footprints;
- replace `generate()` with facilities, inventories, lanes, and demand from an
  operational source;
- pass a lane model backed by an audited routing service to `enrich_lanes` and
  `avoiding()`; and
- check supplier qualification and lane feasibility before candidate generation.

## License

Code is MIT licensed. See `LICENSE`. Map boundaries and the road graph are derived
from public-domain U.S. Census Bureau data; see `data/README.md`.
