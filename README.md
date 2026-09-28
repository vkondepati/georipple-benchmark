# GeoRipple benchmark

Code for the evaluation in *GeoRipple: Hazard-Aware Spatio-Topological
Disruption Wavefronts for Resolving Supply-Chain Disruptions with Linked
Graph and GIS Views*. It generates synthetic supply networks, applies
hazard footprints, predicts per-dealer stockout times (the disruption
wavefront), plans recovery with a time-expanded MILP, and scores
everything against a noisier ground-truth simulation. It produces
Tables I and II of the paper.

## Quick start

```bash
pip install -r requirements.txt
make test      # 3 unit tests
make bench     # 5 seeds x 3 hazards, about 2-5 minutes on one core
make tables    # writes results/tables.tex and results/summary.json
make figures   # writes paper/figures/fig_wavefront.* and fig_resolution.*
```

`results/` already contains the outputs used in the paper
(`runs.json`, `summary.json`, `tables.tex`). Rerunning with the same
seeds reproduces them.

## Layout

| Path | Purpose |
|---|---|
| `georipple/geo.py` | Approximate U.S. metro coordinates, planar projection |
| `georipple/network.py` | Synthetic supplier -> DC -> dealer network |
| `georipple/hazards.py` | Stylized footprints, forecast ensembles, node and lane exposure |
| `georipple/simulate.py` | Daily flow simulation (Algorithm 1, with schedules and delays) |
| `georipple/predict.py` | Wavefront prediction, baselines B1/B2/A1, STCI |
| `georipple/routing.py` | Hazard-avoiding routing via a visibility graph |
| `georipple/resolve.py` | Candidate recovery lanes, MILP (HiGHS via SciPy), greedy baseline B3 |
| `georipple/evaluate.py` | Ground-truth execution and metrics |
| `experiments/run_benchmark.py` | Runs all methods, writes `results/runs.json` |
| `experiments/make_tables.py` | Aggregates runs, writes LaTeX tables |
| `experiments/make_figures.py` | Renders the two map figures for one run |
| `data/us-states.json` | State outlines for figures (PublicaMundi/MappingAPI, from U.S. Census boundaries) |
| `paper/georipple_paper.tex` | Paper source |

## What the benchmark does and does not model

It is a synthetic benchmark. Keep these points in mind when citing it:

- **Locations** are jittered around rounded metro coordinates, not real facilities.
- **Lanes** are straight segments scaled by a 1.25 road factor, not road routes.
- **Hazards** are hand-drawn footprints shaped like a Gulf Coast
  hurricane flood, a statewide winter storm, and a river-corridor flood.
  They are not reconstructions of specific historical events.
- **Hazard routing** uses a visibility graph as a stand-in for a routing
  engine with exclusion polygons (Valhalla, GraphHopper).
- **Networks are trees**: each dealer has one DC and each DC two
  suppliers. Because of this, the STCI structural term (lambda) had no
  effect, and hazard-aware routing changed little. The paper reports
  both as negative results.

## Using real data

To ground the study historically:

- Replace `base_hazards()` in `georipple/hazards.py` with observed GeoJSON
  footprints, projected with `geo.project_coords`.
- Replace `generate()` with facilities loaded from your Neo4j graph.
- Swap `HazardRouter` for calls to a self-hosted Valhalla (`exclude_polygons`)
  or GraphHopper instance.

## Key parameters

Defaults are in `run_benchmark.py`: horizon 28 days, 8 forecast
members, top-K = 10, MILP time limit 60 s. Planning uses the
upper-quartile exposure across forecast members (`PLAN_QUANTILE`). Cost
and penalty constants are at the top of `resolve.py`.

## License

MIT. See `LICENSE`.
