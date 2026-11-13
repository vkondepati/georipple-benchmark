import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.network import DEALER, generate
from georipple.resolve import Candidates, additive_schedule, execution_schedule, plan_milp
from georipple.simulate import existing_lanes, initial_pipeline, simulate

T = 10


def _forecast_closes_one_dealer_lane():
    net = generate(0, n_dc=4, n_dealer=8)
    xn = np.zeros((net.n, T))
    xe = np.zeros((len(net.src), T))
    lane = int(np.where(net.kind[net.dst] == DEALER)[0][0])
    xe[lane, 2:6] = 1.0
    return net, xn, xe, lane


def _plan(net, xn, xe):
    pipe = initial_pipeline(net, T)
    fe, fc, y, _ = plan_milp(net, Candidates(), xn, xe, np.zeros((0, T)), T, pipe,
                             time_limit=5.0)
    return pipe, fe, fc, y


def test_additive_dispatch_is_curtailed_to_the_plan_when_the_forecast_is_right():
    net, xn, xe, _ = _forecast_closes_one_dealer_lane()
    pipe, fe, fc, y = _plan(net, xn, xe)
    lanes = existing_lanes(net)
    absolute = simulate(net, lanes, xn, xe, execution_schedule(net, fe, fc, y, T)[0], T,
                        pipe=pipe)
    additive = simulate(net, lanes, xn, xe, additive_schedule(net, fe, fc, y, T)[0], T,
                        pipe=pipe)
    assert np.allclose(additive.shipped, absolute.shipped, atol=1e-6)
    assert np.allclose(additive.served, absolute.served, atol=1e-6)


def test_additive_dispatch_keeps_normal_service_when_the_forecast_is_wrong():
    net, xn, xe, lane = _forecast_closes_one_dealer_lane()
    pipe, fe, fc, y = _plan(net, xn, xe)
    calm_n, calm_e = np.zeros_like(xn), np.zeros_like(xe)
    lanes = existing_lanes(net)
    absolute = simulate(net, lanes, calm_n, calm_e, execution_schedule(net, fe, fc, y, T)[0],
                        T, pipe=pipe)
    additive = simulate(net, lanes, calm_n, calm_e, additive_schedule(net, fe, fc, y, T)[0],
                        T, pipe=pipe)
    assert absolute.shipped[lane, 2:6].max() < net.base_f[lane]
    assert additive.shipped[lane, 2:6].min() >= net.base_f[lane] - 1e-9


def test_additive_schedule_is_the_larger_of_normal_and_planned_flow():
    net, xn, xe, _ = _forecast_closes_one_dealer_lane()
    _, fe, fc, y = _plan(net, xn, xe)
    existing, candidates = additive_schedule(net, fe, fc, y, T)
    assert np.array_equal(existing, np.maximum(net.base_f[:, None], fe))
    assert np.any(fe > net.base_f[:, None] + 1e-9) or np.any(fe < net.base_f[:, None] - 1e-9)
    assert candidates.shape == (0, T)
