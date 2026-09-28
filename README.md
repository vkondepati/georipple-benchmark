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
make bench       # 5 network seeds x 3 hazards x 3 delay replicates
make tables      # results/summary.json and results/tables.tex
make figures     # paper/figures/fig_{wavefront,resolution}.{png,pdf}
```

The full benchmark currently takes about one minute on the reference Apple
Silicon laptop; solver and hardware differences can change runtime. Results are
deterministic for the pinned environment and record the Python, platform,
package, Git-commit, and tracked-worktree provenance in `results/runs.json`.
Floating-point and solver changes on another platform can cause small numerical
differences.

The repository includes the exact generated artifact used by the manuscript:

- `results/runs.json`: 45 evaluation records and provenance
- `results/summary.json`: seed-clustered summaries and paired effects
- `results/tables.tex`: generated manuscript tables
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
| `experiments/make_figures.py` | Static map figures for one illustrative run |
| `paper/georipple_paper.tex` | Manuscript source |
| `data/README.md` | Attribution for the map boundary data |

## Evaluation design

The checked-in run uses a 28-day horizon, eight forecast members, top-$K=10$,
a 60-second MILP time limit, five independently generated networks, three fixed
hazard scenarios, and three stochastic execution replicates per network/hazard
pair. Plans and predictions are built once per network/hazard pair and then
evaluated under the three delay realizations. Descriptive intervals and paired
effects use network seed as the independent cluster ($n=5$); the 45 records are
not treated as independent samples. Reported intervals are descriptive t
intervals over seed means, with bounded metrics intersected with $[0,1]$.

Important interpretation limits:

- Locations are jittered around rounded metro coordinates, not real facilities.
- Existing lanes are straight segments with a 1.25 road-distance factor, not
  road-network routes.
- Hazards are stylized event types, not historical reconstructions.
- The network is a layered DAG: every dealer has one DC, while each DC has two
  supplier parents. It is not a tree and does not model general multi-sourcing.
- Candidate routing uses a planar visibility graph around a buffered polygon as
  a proxy for a production routing engine.
- Forecast members and the realized hazard come from the same hand-authored
  hazard family; this is an internal synthetic stress test, not external
  validation.
- The STCI structural term selected the same top-$K$ set as `lambda=0` in this
  benchmark. Hazard-aware routing also has an inconclusive effect at five seed
  clusters. Both are reported as negative results.

## Current headline results

Across the synthetic benchmark, GeoRipple has precision 0.96, recall 0.65,
Brier score 0.013, and stockout timing MAE 7.9 days. Its plans reduce mean total
unmet demand from 1,027 to 826 units (paired mean difference -202; seed-clustered
descriptive 95% t interval -357 to -46), at mean recovery cost 17.7k synthetic
cost units (CU). This aggregate result is not
uniform: the plan improves the Gulf and river scenarios but slightly worsens the
winter scenario. See `results/summary.json` for all intervals and strata.

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
