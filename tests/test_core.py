import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.network import generate, DEALER                     # noqa: E402
from georipple.hazards import Hazard, base_hazards, node_exposure, edge_exposure  # noqa: E402
from georipple.routing import HazardRouter                         # noqa: E402
from georipple.simulate import (simulate, existing_lanes, baseline_schedule,  # noqa: E402
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
