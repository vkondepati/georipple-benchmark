import os
import sys

import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.network import generate, Network, DEALER            # noqa: E402
from georipple.hazards import Hazard, base_hazards, node_exposure, edge_exposure  # noqa: E402
from georipple.routing import HazardRouter                         # noqa: E402
from georipple.resolve import Candidates, execution_schedule, plan_milp  # noqa: E402
from georipple.predict import stci                                 # noqa: E402
from georipple.evaluate import execute_truth, resolution_metrics   # noqa: E402
from georipple.simulate import (Lanes, SimResult, simulate, existing_lanes, baseline_schedule,  # noqa: E402
                                initial_pipeline)

T = 21


def _calm(net):
    return np.zeros((net.n, T)), np.zeros((len(net.src), T))


def test_steady_state_without_hazard_has_no_stockouts():
    net = generate(0)
    xn, xe = _calm(net)
    r = simulate(net, existing_lanes(net), xn, xe, baseline_schedule(net, T), T,
                 pipe=initial_pipeline(net, T))
    dl = net.dealers
    assert np.all(r.sigma[dl] == T)
    assert np.allclose(r.served[dl], net.demand[dl, None])


def test_hazard_causes_stockouts_inside_footprint():
    net = generate(0)
    h = base_hazards()[0]
    xn, xe = node_exposure(net.xy, h, T), edge_exposure(net.geoms, h, T)
    r = simulate(net, existing_lanes(net), xn, xe, baseline_schedule(net, T), T,
                 pipe=initial_pipeline(net, T))
    assert (r.sigma[net.dealers] < T).sum() > 0


def test_router_avoids_obstacle():
    h = base_hazards()[0]
    router = HazardRouter([h.poly])
    c = np.array(h.poly.centroid.coords[0])
    p, q = c + np.array([-600.0, 0.0]), c + np.array([600.0, 0.0])
    g = router.route(p, q)
    assert g is not None
    assert g.intersection(router.obstacle.buffer(-1.0)).length < 1e-6
    assert g.length > np.linalg.norm(q - p)


def test_receiving_capacity_applies_on_arrival_day():
    net = Network(
        xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
        kind=np.array([0, DEALER]), demand=np.array([0.0, 1.0]),
        prod=np.array([0.0, 0.0]), node_cap=np.array([1.0, 0.0]),
        recv_cap=np.array([0.0, 1.0]), I0=np.array([1.0, 0.0]),
        src=np.array([0]), dst=np.array([1]), km=np.array([1.0]),
        tau=np.array([2]), cap=np.array([1.0]), base_f=np.array([0.0]))
    lanes = Lanes(net.src, net.dst, net.tau, net.cap)
    S = np.zeros((1, 4)); S[0, 0] = 1.0
    xn = np.zeros((2, 4)); xn[1, 2] = 1.0
    r = simulate(net, lanes, xn, np.zeros((1, 4)), S, 4)
    assert r.shipped[0, 0] == 1.0
    assert r.served[1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_milp_schedule_is_executed_without_postprocessing():
    net = generate(0, n_dc=4, n_dealer=8)
    T = 5
    xn = np.zeros((net.n, T)); xe = np.zeros((len(net.src), T))
    fe, fc, y, _ = plan_milp(net, Candidates(), xn, xe, np.zeros((0, T)), T,
                             initial_pipeline(net, T), time_limit=5.0)
    scheduled_existing, scheduled_candidates = execution_schedule(net, fe, fc, y, T)
    assert np.array_equal(scheduled_existing, fe)
    assert scheduled_candidates.shape == (0, T)
    r = simulate(net, existing_lanes(net), xn, xe, scheduled_existing, T,
                 pipe=initial_pipeline(net, T))
    assert np.allclose(r.shipped, fe, atol=1e-7)


def test_stci_attributes_last_mile_exposure_to_parent_facility():
    net = Network(
        xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        kind=np.array([0, 1, DEALER]), demand=np.array([0.0, 0.0, 1.0]),
        prod=np.zeros(3), node_cap=np.ones(3), recv_cap=np.ones(3), I0=np.ones(3),
        src=np.array([0, 1]), dst=np.array([1, 2]), km=np.ones(2),
        tau=np.ones(2, int), cap=np.ones(2), base_f=np.ones(2))
    xn = np.zeros((3, 2)); xe = np.zeros((2, 2)); xe[1] = 1.0
    dar = np.array([[0.0, 0.0, 2.0]])
    score = stci(net, [(xn, xe)], dar)
    assert score[1] > 0.0


def test_activated_lane_incurs_fixed_cost_without_flow():
    net = generate(0, n_dc=4, n_dealer=8)
    cand = Candidates()
    s, d = int(np.where(net.kind == 1)[0][0]), int(net.dealers[0])
    cand.add(s, d, LineString([net.xy[s], net.xy[d]]), net.demand[d], "lateral")
    h = Hazard("calm", base_hazards()[0].poly, T + 1, T + 1, 0.0)
    _, cost = execute_truth(net, h, T, 0, cand,
                            baseline_schedule(net, T), np.zeros((1, T)), np.array([True]))
    assert cost == cand.fixed_cost[0]


def test_ttr_is_censored_when_recovery_is_not_observed():
    net = generate(0, n_dc=4, n_dealer=8)
    served = np.zeros((net.n, T))
    r = SimResult(served, np.full(net.n, T), np.zeros((len(net.src), T)))
    affected = np.zeros(len(net.dealers), bool); affected[0] = True
    metrics = resolution_metrics(net, r, affected, base_hazards()[0], T, 0.0)
    assert np.isnan(metrics["ttr95_days"])
    assert metrics["ttr95_recovered"] is False
