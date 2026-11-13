import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.network import DC, generate
from georipple.resolve import generate_candidates, greedy_nearest
from georipple.simulate import baseline_schedule, existing_lanes, initial_pipeline, simulate

SHARES = (0.7, 0.3)


def test_each_dealer_draws_from_its_two_nearest_dcs_in_the_stated_shares():
    net = generate(1, n_dc=8, n_dealer=40, dealer_shares=SHARES)
    for d in net.dealers:
        lanes = np.where(net.dst == d)[0]
        assert len(lanes) == 2
        dist = np.linalg.norm(net.xy[net.src[lanes]] - net.xy[d], axis=1)
        all_dc = np.linalg.norm(net.xy[net.kind == DC] - net.xy[d], axis=1)
        assert np.allclose(np.sort(dist), np.sort(all_dc)[:2])
        flows = net.base_f[lanes][np.argsort(dist)]
        assert flows == pytest.approx([0.7 * net.demand[d], 0.3 * net.demand[d]])


def test_dual_sourced_network_is_in_steady_state_without_a_hazard():
    net = generate(1, n_dc=8, n_dealer=40, dealer_shares=SHARES)
    T = 21
    r = simulate(net, existing_lanes(net), np.zeros((net.n, T)), np.zeros((len(net.src), T)),
                 baseline_schedule(net, T), T, pipe=initial_pipeline(net, T))
    assert np.all(r.sigma[net.dealers] == T)


def test_recovery_lanes_never_duplicate_an_existing_supply_lane():
    net = generate(2, n_dc=8, n_dealer=40, dealer_shares=SHARES)
    stranded = np.zeros(net.n, bool)
    stranded[net.dealers] = True
    score = (net.kind == DC).astype(float)
    existing = set(zip(net.src.tolist(), net.dst.tolist(), strict=True))
    cand, _ = generate_candidates(net, stranded, score, np.zeros(net.n), K=8, k=3,
                                  r_lateral=10_000.0, r_alt=10_000.0, tau_max=99)
    assert len(cand) > 0
    assert not existing & set(zip(cand.src, cand.dst, strict=True))
    greedy, *_ = greedy_nearest(net, stranded, np.zeros(net.n), 7, r_lateral=10_000.0)
    assert len(greedy) > 0
    assert not existing & set(zip(greedy.src, greedy.dst, strict=True))


@pytest.mark.parametrize("shares", [(0.7, 0.2), (1.2, -0.2), ()])
def test_dealer_shares_must_be_positive_and_sum_to_one(shares):
    with pytest.raises(ValueError):
        generate(0, n_dc=4, n_dealer=8, dealer_shares=shares)
