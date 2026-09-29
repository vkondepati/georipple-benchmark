# GeoRipple benchmark

Reproducible code and synthetic results for *GeoRipple: Hazard-Aware
Spatio-Topological Disruption Wavefronts for Resolving Supply-Chain
Disruptions with Linked Graph and GIS Views*.

The benchmark generates layered supply networks, intersects nodes and lanes
with forecast hazard ensembles, predicts dealer stockout times, selects
recovery lanes with a time-expanded MILP, and evaluates each plan under a
realized hazard with stochastic transit delays. It does **not** contain the
proposed Neo4j/Snowflake/React deployment described in the paper.

## Reproduce the artifact

Python 3.11 is the reference interpreter. Direct requirements are documented in
`requirements.txt`; the complete transitive environment is pinned in
`requirements.lock`.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
make test
make check      # tests, focused lint, strict JSON, and artifact drift
make bench       # 10 network seeds x 3 hazards x 3 delay replicates
make tables      # results/summary.json and results/tables.tex
make sensitivity # 7 configurations x 5 seeds x 3 hazards x 3 replicates
make figures     # paper/figures/fig_{wavefront,resolution}.{png,pdf}
```

The primary benchmark and sensitivity study each take about one to two minutes
on the reference Apple Silicon laptop; solver and hardware differences can
change runtime. Results are
deterministic for the pinned environment and record the Python, platform,
package, Git-commit, and tracked-worktree provenance in `results/runs.json`.
Floating-point and solver changes on another platform can cause small numerical
differences.

The repository includes the exact generated artifact used by the manuscript:

- `results/runs.json`: 90 primary evaluation records and provenance
- `results/summary.json`: seed-clustered summaries and paired effects
- `results/tables.tex`: generated manuscript tables
- `results/sensitivity_runs.json`: 315 one-factor-at-a-time records
- `results/sensitivity_summary.json`: sensitivity summaries and intervals
- `results/sensitivity.tex`: generated sensitivity table
- `paper/figures/figure_stats.json`: values underlying the example figures

## Layout

| Path | Purpose |
|---|---|
| `georipple/network.py` | Synthetic supplier--DC--dealer layered DAG |
| `georipple/hazards.py` | Stylized footprints, forecast ensembles, and exposure |
| `georipple/simulate.py` | Daily inventory/flow simulator with transit and receiving queues |
| `georipple/predict.py` | Wavefront prediction, baselines B1/B2/A1, and STCI |
| `georipple/routing.py` | Hazard-avoiding visibility-graph routing proxy |
| `georipple/resolve.py` | Recovery candidates, MILP, and action-count-matched greedy B3 |
| `georipple/evaluate.py` | Stochastic execution and prediction/recovery metrics |
| `experiments/run_benchmark.py` | Benchmark runner and provenance capture |
| `experiments/make_tables.py` | Seed-clustered descriptive intervals and paired comparisons |
| `experiments/run_sensitivity.py` | One-factor-at-a-time sensitivity runner |
| `experiments/make_sensitivity.py` | Sensitivity summary and table generator |
| `experiments/make_figures.py` | Static map figures for one illustrative run |
| `paper/georipple_paper.tex` | Manuscript source |
| `data/README.md` | Attribution for the map boundary data |

## Evaluation design

The checked-in run uses a 28-day horizon, eight forecast members, top-$K=10$,
a 60-second MILP time limit, ten independently generated networks, three fixed
hazard families, and three stochastic execution replicates per network/hazard
pair. Plans and predictions are built once per network/hazard pair and then
evaluated under a separately seeded, held-out perturbation of that hazard
family and the three delay realizations. Descriptive intervals and paired
effects use network seed as the independent cluster ($n=10$); the 90 records are
not treated as independent samples. Reported intervals are descriptive t
intervals over seed means, with bounded metrics intersected with $[0,1]$.

The separate sensitivity artifact uses five seeds and varies one factor at a
time: planning quantile (0.50, 0.75, 0.90), top-$K$ (5, 10, 15), and ensemble
size (4, 8, 16). It is diagnostic and was not used to retune the primary result.

Important interpretation limits:

- Locations are jittered around rounded metro coordinates, not real facilities.
- Existing lanes are straight segments with a 1.25 road-distance factor, not
  road-network routes.
- Hazards are stylized event types, not historical reconstructions.
- The network is a layered DAG: every dealer has one DC, while each DC has two
  supplier parents. It is not a tree and does not model general multi-sourcing.
- Candidate routing uses a planar visibility graph around a buffered polygon as
  a proxy for a production routing engine.
- Forecast members and the independently perturbed realized hazard come from
  the same hand-authored family; this tests within-family generalization, not
  external validation or performance on unseen hazard types.
- The STCI structural term selected the same top-$K$ set as `lambda=0` in this
  benchmark. Hazard-aware routing also has an inconclusive effect. Both are
  reported as negative results.
- The linked graph/map application is a proposed architecture; it has not been
  implemented or evaluated with users.

## Current headline results

Across the primary synthetic benchmark, GeoRipple has precision 0.86, recall
0.60, Brier score 0.017, and stockout-timing MAE 9.3 days. Adding lane exposure
raises recall by 0.073 (seed-clustered descriptive 95% t interval 0.032 to
0.113).

The prescriptive result is negative: the reference upper-quartile plan raises
mean system-wide unmet demand from 899 to 1,053 units (paired difference +154;
95% interval +3 to +305) at a mean 13.9k synthetic CU. A stronger
mean-exposure MILP yields 868 units at 7.3k CU. In sensitivity analysis, only
the median-exposure configuration lowers mean unmet demand. Both reductions
relative to no action have intervals that include zero. The artifact therefore
supports lane-aware prediction
but does **not** establish the effectiveness of the reference resolution
policy. See `results/summary.json` and `results/sensitivity_summary.json` for
all intervals and hazard strata.

## Using operational data

The code exposes the seams needed for a future field evaluation:

- replace `base_hazards()` with observed, projected GeoJSON footprints;
- replace `generate()` with facilities, inventories, lanes, and demand from an
  operational source;
- replace `HazardRouter` with an audited road-network routing service; and
- validate supplier qualification and lane feasibility before candidate
  generation.

Those integrations are future work; they are not present in this repository.

## Paper build

The manuscript build uses [Tectonic](https://tectonic-typesetting.github.io/),
which resolves its TeX package bundle on first use:

```bash
make paper
```

## License

Code is MIT licensed. See `LICENSE`. The bundled map boundary data has separate
source attribution in `data/README.md`.
