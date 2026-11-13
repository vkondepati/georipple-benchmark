"""Synthetic three-echelon supply network (suppliers -> DCs -> dealers)."""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString

from .geo import SUPPLIER_CITIES, city_table

SUP, DC, DEALER = 0, 1, 2
ROAD_FACTOR = 1.25      # road km per straight-line km
KM_PER_DAY = 700.0      # truck travel per day


def lane_tau(km):
    return np.maximum(1, np.ceil(np.asarray(km) / KM_PER_DAY)).astype(int)


@dataclass
class Network:
    xy: np.ndarray          # (n, 2) projected km
    kind: np.ndarray        # (n,) SUP / DC / DEALER
    demand: np.ndarray      # (n,) units/day (dealers only)
    prod: np.ndarray        # (n,) production units/day (suppliers only)
    node_cap: np.ndarray    # (n,) max outbound units/day
    recv_cap: np.ndarray    # (n,) max inbound units/day (shared by all lanes)
    I0: np.ndarray          # (n,) initial on-hand inventory
    src: np.ndarray         # (E,)
    dst: np.ndarray         # (E,)
    km: np.ndarray          # (E,) road km
    tau: np.ndarray         # (E,) transit days
    cap: np.ndarray         # (E,) lane capacity units/day
    base_f: np.ndarray      # (E,) baseline flow units/day
    geoms: list = field(default_factory=list)   # (E,) shapely LineStrings

    @property
    def n(self):
        return len(self.kind)

    @property
    def dealers(self):
        return np.where(self.kind == DEALER)[0]

    def children(self):
        ch = [[] for _ in range(self.n)]
        for s, d in zip(self.src, self.dst, strict=True):
            ch[s].append(int(d))
        return ch

    def downstream_dealers(self, v, ch=None):
        ch = ch if ch is not None else self.children()
        out, stack, seen = [], [v], {v}
        while stack:
            u = stack.pop()
            for w in ch[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
                    if self.kind[w] == DEALER:
                        out.append(w)
        return out


def _checked_shares(dealer_shares, n_dc):
    shares = tuple(float(s) for s in dealer_shares)
    if (not shares or len(shares) > n_dc or min(shares) <= 0.0
            or abs(sum(shares) - 1.0) > 1e-9):
        raise ValueError("dealer_shares must be positive, sum to 1, and fit within n_dc")
    return shares


def generate(seed, n_dc=60, n_dealer=600, jitter_dc=40.0, jitter_dealer=50.0,
             dealer_shares=(1.0,)):
    """Generate a network whose facilities cluster around U.S. metros.

    ``dealer_shares`` splits each dealer's demand over its nearest DCs: the i-th nearest
    DC supplies share i. Lanes are straight; ``lanes.enrich_lanes`` can redraw them.
    """
    shares = _checked_shares(dealer_shares, n_dc)
    rng = np.random.default_rng(seed)
    names, cxy, pop = city_table()
    idx = {nm: i for i, nm in enumerate(names)}

    sup_xy = np.array([cxy[idx[c]] for c in SUPPLIER_CITIES])
    sup_xy = sup_xy + rng.normal(0, 15.0, sup_xy.shape)
    w_dc = np.sqrt(pop) / np.sqrt(pop).sum()
    dc_xy = cxy[rng.choice(len(pop), n_dc, p=w_dc)] + rng.normal(0, jitter_dc, (n_dc, 2))
    w_r = pop / pop.sum()
    r_xy = cxy[rng.choice(len(pop), n_dealer, p=w_r)] + rng.normal(0, jitter_dealer, (n_dealer, 2))

    xy = np.vstack([sup_xy, dc_xy, r_xy])
    ns = len(sup_xy)
    kind = np.array([SUP] * ns + [DC] * n_dc + [DEALER] * n_dealer)
    sup_ids = np.arange(ns)
    dc_ids = np.arange(ns, ns + n_dc)
    r_ids = np.arange(ns + n_dc, len(kind))

    demand = np.zeros(len(kind))
    demand[r_ids] = rng.lognormal(np.log(10.0), 0.4, n_dealer)

    src, dst, flow = [], [], []
    dc_out = np.zeros(len(kind))
    for r in r_ids:                                   # dealer <- nearest DC(s)
        nearest = np.argsort(np.linalg.norm(dc_xy - xy[r], axis=1), kind="stable")
        for share, j in zip(shares, nearest, strict=False):
            k = dc_ids[j]
            src.append(k); dst.append(r); flow.append(share * demand[r])
            dc_out[k] += share * demand[r]
    for k in dc_ids:                                  # DC <- 2 nearest suppliers
        if dc_out[k] == 0:
            continue
        d = np.linalg.norm(sup_xy - xy[k], axis=1)
        for s in sup_ids[np.argsort(d)[:2]]:
            src.append(s); dst.append(k); flow.append(0.5 * dc_out[k])
    src, dst, base_f = np.array(src), np.array(dst), np.array(flow)

    out = np.bincount(src, base_f, len(kind))
    prod = np.where(kind == SUP, 1.15 * out, 0.0)
    node_cap = 1.3 * out
    inflow = np.bincount(dst, base_f, len(kind))
    recv_cap = 2.0 * inflow
    dos = rng.lognormal(np.log(5.0), 0.5, len(kind))   # days of supply
    I0 = np.where(kind == DEALER, demand * dos, out * dos)
    I0[kind == SUP] = 2.0 * out[kind == SUP]

    geoms = [LineString([tuple(xy[s]), tuple(xy[d])])
             for s, d in zip(src, dst, strict=True)]
    km = np.array([g.length for g in geoms]) * ROAD_FACTOR
    return Network(xy=xy, kind=kind, demand=demand, prod=prod, node_cap=node_cap, recv_cap=recv_cap,
                   I0=I0, src=src, dst=dst, km=km, tau=lane_tau(km),
                   cap=2.0 * base_f, base_f=base_f, geoms=geoms)
