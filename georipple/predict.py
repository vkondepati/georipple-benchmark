"""Stockout prediction methods and the Spatio-Topological Criticality Index."""
import numpy as np
import networkx as nx

from .hazards import node_exposure, edge_exposure
from .network import DEALER
from .simulate import simulate, existing_lanes, baseline_schedule, initial_pipeline


def ensemble_exposures(net, ens, T):
    return [(node_exposure(net.xy, h, T), edge_exposure(net.geoms, h, T)) for h in ens]


def predict_wavefront(net, exps, T, use_edges=True, use_buffers=True):
    """Deterministic Algorithm-1 simulation per forecast member.

    Returns (P[stockout by T], median stockout day, per-member DaR per node).
    """
    lanes, S, pipe = existing_lanes(net), baseline_schedule(net, T), initial_pipeline(net, T)
    I0 = net.I0 if use_buffers else np.zeros(net.n)
    sig, dar = [], []
    for xn, xe in exps:
        xe_used = xe if use_edges else np.zeros_like(xe)
        r = simulate(net, lanes, xn, xe_used, S, T, pipe=pipe, I0=I0)
        sig.append(r.sigma)
        dar.append((net.demand[:, None] - r.served).clip(0).sum(1) * (net.kind == DEALER))
    sig = np.array(sig)
    return (sig < T).mean(0), np.median(sig, 0), np.array(dar)


def predict_topology_only(net, ens, T):
    """B1: facilities inside the (majority) forecast footprint fail; every
    downstream dealer is predicted to stock out at hazard onset."""
    inside = np.mean([node_exposure(net.xy, h, h.start + 1)[:, h.start] > 0 for h in ens], 0) >= 0.5
    onset = int(np.median([h.start for h in ens]))
    ch = net.children()
    hit = set()
    for v in np.where(inside)[0]:
        if net.kind[v] == DEALER:
            hit.add(int(v))
        hit.update(net.downstream_dealers(int(v), ch))
    p = np.zeros(net.n)
    sig = np.full(net.n, float(T))
    for v in hit:
        p[v], sig[v] = 1.0, onset
    return p, sig


def stci(net, exps, dar_members, lam=1.0):
    """STCI_v = x_v * DaR_down_v * (1 + lam * B_v), for suppliers and DCs."""
    n = net.n
    xhat = np.zeros(n)
    for xn, xe in exps:
        peak_node = xn.max(1)
        peak_lane = np.zeros(n)
        peak = xe.max(1)
        np.maximum.at(peak_lane, net.src, peak)
        np.maximum.at(peak_lane, net.dst, peak)
        xhat += np.maximum(peak_node, peak_lane)
    xhat /= len(exps)
    dar = dar_members.mean(0)
    ch = net.children()
    down = np.zeros(n)
    for v in range(n):
        if net.kind[v] != DEALER:
            dd = net.downstream_dealers(v, ch)
            down[v] = dar[dd].sum() if dd else 0.0
    g = nx.DiGraph()
    g.add_edges_from(zip(net.src.tolist(), net.dst.tolist()))
    bc = nx.betweenness_centrality(g, normalized=True)
    B = np.array([bc.get(v, 0.0) for v in range(n)])
    norm = lambda a: a / a.max() if a.max() > 0 else a
    score = xhat * norm(down) * (1.0 + lam * norm(B))
    score[net.kind == DEALER] = 0.0
    return score
