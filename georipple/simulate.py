"""Discrete-time flow simulation (Algorithm 1 of the paper, generalized).

Each lane e has a shipment schedule S[e, t]. Each day a node ships its
scheduled quantities, prorated down when on-hand inventory or
hazard-degraded node capacity is short, then clipped by hazard-degraded
lane capacity (clipped quantity stays at the origin). All lanes into a node share
its hazard-degraded receiving capacity on arrival, so opening extra lanes
into a flooded area cannot bypass the hazard. Excess arrivals remain in
transit until receiving capacity becomes available. Shipments arrive after
tau_e (+ optional stochastic delay) days. Dealers serve demand from on-hand
stock; the first day demand is not fully met is the stockout day.
"""
from dataclasses import dataclass
import numpy as np

from .network import SUP, DEALER


@dataclass
class Lanes:
    src: np.ndarray
    dst: np.ndarray
    tau: np.ndarray
    cap: np.ndarray


@dataclass
class SimResult:
    served: np.ndarray      # (n, T) (non-zero for dealers only)
    sigma: np.ndarray       # (n,) stockout day, T if none
    shipped: np.ndarray     # (E, T)


def initial_pipeline(net, T):
    """Arrivals during days 0..tau_e-1 from shipments already in transit."""
    pipe = np.zeros((net.n, T))
    for e in range(len(net.src)):
        pipe[net.dst[e], :min(T, net.tau[e])] += net.base_f[e]
    return pipe


def simulate(net, lanes, x_node, x_edge, S, T, pipe=None, delays=None, I0=None):
    n, E = net.n, len(lanes.src)
    I = (net.I0 if I0 is None else I0).astype(float).copy()
    horizon = T + int(lanes.tau.max()) + (0 if delays is None else int(delays.max())) + 1
    arr = np.zeros((n, horizon))
    if pipe is not None:
        arr[:, :T] += pipe
    is_sup = net.kind == SUP
    is_dl = net.kind == DEALER
    served = np.zeros((n, T))
    shipped = np.zeros((E, T))
    sigma = np.full(n, T)
    for t in range(T):
        inbound = arr[:, t]
        rcap = np.maximum(net.recv_cap * (1.0 - x_node[:, t]), 0.0)
        received = np.minimum(inbound, rcap)
        I += received
        arr[:, t + 1] += inbound - received
        I[is_sup] += net.prod[is_sup] * (1.0 - x_node[is_sup, t])
        s = np.where(is_dl, np.minimum(I, net.demand), 0.0)
        I -= s
        served[:, t] = s
        short = is_dl & (s < net.demand * (1 - 1e-6)) & (sigma == T)
        sigma[short] = t
        req = np.bincount(lanes.src, S[:, t], n)
        avail = np.minimum(I, net.node_cap * (1.0 - x_node[:, t]))
        ratio = np.where(req > 1e-12, np.minimum(1.0, avail / np.maximum(req, 1e-12)), 0.0)
        ship = S[:, t] * ratio[lanes.src]
        ship = np.minimum(ship, lanes.cap * (1.0 - x_edge[:, t]))
        ship = np.maximum(ship, 0.0)
        I -= np.bincount(lanes.src, ship, n)
        due = t + lanes.tau + (0 if delays is None else delays[:, t])
        np.add.at(arr, (lanes.dst, due), ship)
        shipped[:, t] = ship
    return SimResult(served=served, sigma=sigma, shipped=shipped)


def existing_lanes(net):
    return Lanes(net.src, net.dst, net.tau, net.cap)


def baseline_schedule(net, T):
    return np.repeat(net.base_f[:, None], T, axis=1)
