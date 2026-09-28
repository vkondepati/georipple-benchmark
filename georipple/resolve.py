"""Hazard-aware resolution: candidate recovery lanes + time-expanded MILP."""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from shapely.geometry import LineString

from .network import SUP, DC, DEALER, ROAD_FACTOR, lane_tau
from .simulate import pipeline_receipts

UNIT_COST_PER_KM = 0.01     # $ per unit per road km
FIXED_BASE = 1000.0         # $ fixed cost to open a lane
FIXED_PER_KM = 1.0          # $ per road km, fixed part
SHORTAGE_PENALTY = 100.0    # $ per unit of unmet demand


@dataclass
class Candidates:
    src: list = field(default_factory=list)
    dst: list = field(default_factory=list)
    km: list = field(default_factory=list)
    cap: list = field(default_factory=list)
    geoms: list = field(default_factory=list)
    kind: list = field(default_factory=list)   # "lateral" / "alt_source"

    def __len__(self):
        return len(self.src)

    def add(self, s, d, geom, cap, kind):
        self.src.append(int(s)); self.dst.append(int(d))
        self.geoms.append(geom); self.km.append(geom.length * ROAD_FACTOR)
        self.cap.append(float(cap)); self.kind.append(kind)

    def arrays(self):
        km = np.array(self.km, float)
        return (np.array(self.src, int), np.array(self.dst, int), km,
                lane_tau(km) if len(km) else np.zeros(0, int), np.array(self.cap, float))

    @property
    def fixed_cost(self):
        return FIXED_BASE + FIXED_PER_KM * np.array(self.km, float)


def _nearest(net, origin, pool, k, radius):
    d = np.linalg.norm(net.xy[pool] - net.xy[origin], axis=1)
    order = np.argsort(d)
    return [int(pool[i]) for i in order if d[i] <= radius][:k]


def generate_candidates(net, stranded, score, p_exposed, router=None, K=10,
                        k=3, r_lateral=600.0, r_alt=1500.0, tau_max=4):
    """Spatial candidate generation for the top-K STCI facilities.

    ``router`` is a HazardRouter; if None, straight-line lanes are used
    without hazard checking (ablation A2).
    """
    top = [int(v) for v in np.argsort(-score)[:K] if score[v] > 0]
    ch = net.children()
    parent = {int(d): int(s) for s, d in zip(net.src, net.dst) if net.kind[d] == DEALER}
    sup_of = {}
    for s, d in zip(net.src, net.dst):
        if net.kind[d] == DC:
            sup_of.setdefault(int(d), set()).add(int(s))

    target_dealers, target_dcs = set(), set()
    for v in top:
        target_dealers.update(d for d in net.downstream_dealers(v, ch) if stranded[d])
        if net.kind[v] == DC:
            target_dcs.add(v)
        elif net.kind[v] == SUP:
            target_dcs.update(w for w in ch[v] if net.kind[w] == DC)

    ok_dc = np.where((net.kind == DC) & (p_exposed < 0.5))[0]
    ok_sup = np.where((net.kind == SUP) & (p_exposed < 0.5))[0]
    cand = Candidates()

    def lane(a, b):
        g = router.route(net.xy[a], net.xy[b]) if router is not None else \
            LineString([tuple(net.xy[a]), tuple(net.xy[b])])
        if g is None or lane_tau(g.length * ROAD_FACTOR) > tau_max:
            return None
        return g

    for d in sorted(target_dealers):
        pool = ok_dc[ok_dc != parent.get(d, -1)]
        for s in _nearest(net, d, pool, k, r_lateral):
            g = lane(s, d)
            if g is not None:
                cand.add(s, d, g, 1.5 * net.demand[d], "lateral")
    for v in sorted(target_dcs):
        pool = np.array([s for s in ok_sup if s not in sup_of.get(v, set())], int)
        if len(pool) == 0:
            continue
        for s in _nearest(net, v, pool, k, r_alt):
            g = lane(s, v)
            if g is not None:
                cand.add(s, v, g, 0.6 * net.node_cap[v] / 1.3, "alt_source")
    return cand, top


def plan_milp(net, cand, xn, xe, xc, T, pipe, time_limit=60.0):
    """Solve the time-expanded MILP (paper Eqs. 5-10) on expected exposures.

    Returns (flows for existing lanes (E,T), flows for candidates (C,T), y (C,), info).
    """
    E, C, n = len(net.src), len(cand), net.n
    cs, cd, ckm, ctau, ccap = cand.arrays()
    src = np.concatenate([net.src, cs]).astype(int)
    dst = np.concatenate([net.dst, cd]).astype(int)
    tau = np.concatenate([net.tau, ctau]).astype(int)
    km = np.concatenate([net.km, ckm])
    ub_f = np.vstack([net.cap[:, None] * (1 - xe), ccap[:, None] * (1 - xc)]) if C else \
        net.cap[:, None] * (1 - xe)
    nE = E + C
    dealers = net.dealers
    dl_index = {int(v): j for j, v in enumerate(dealers)}
    R = len(dealers)
    oF, oI, oU, oY = 0, nE * T, nE * T + n * T, nE * T + n * T + R * T
    nvar = oY + C

    rows, cols, vals = [], [], []
    def row(v, t):
        return v * T + t
    for e in range(nE):
        for t in range(T):
            j = oF + e * T + t
            rows.append(row(src[e], t)); cols.append(j); vals.append(1.0)
            if t + tau[e] < T:
                rows.append(row(dst[e], t + tau[e])); cols.append(j); vals.append(-1.0)
    for v in range(n):
        for t in range(T):
            j = oI + v * T + t
            rows.append(row(v, t)); cols.append(j); vals.append(1.0)
            if t + 1 < T:
                rows.append(row(v, t + 1)); cols.append(j); vals.append(-1.0)
    for v in dealers:
        for t in range(T):
            rows.append(row(v, t)); cols.append(oU + dl_index[int(v)] * T + t); vals.append(-1.0)
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(n * T, nvar))
    pipe_received, recv_residual = pipeline_receipts(net, xn, pipe, T)
    b = pipe_received.copy()
    is_sup = net.kind == SUP
    b[is_sup] += (net.prod[is_sup, None] * (1 - xn[is_sup]))
    b[dealers] -= net.demand[dealers, None]
    b[:, 0] += net.I0
    b_eq = b.reshape(-1)

    # node outbound capacity
    nd = np.where(net.kind != DEALER)[0]
    nd_index = {int(v): i for i, v in enumerate(nd)}
    r2, c2, v2 = [], [], []
    for e in range(nE):
        if src[e] in nd_index:
            for t in range(T):
                r2.append(nd_index[src[e]] * T + t); c2.append(oF + e * T + t); v2.append(1.0)
    A_cap = sp.csr_matrix((v2, (r2, c2)), shape=(len(nd) * T, nvar))
    ub_cap = (net.node_cap[nd, None] * (1 - xn[nd])).reshape(-1)
    # shared node receiving capacity on the arrival day
    arrivals = [(e, t, t + tau[e]) for e in range(nE) for t in range(T)
                if t + tau[e] < T]
    r4 = [dst[e] * T + at for e, t, at in arrivals]
    c4 = [oF + e * T + t for e, t, at in arrivals]
    A_recv = sp.csr_matrix((np.ones(len(r4)), (r4, c4)), shape=(n * T, nvar))
    ub_recv = recv_residual.reshape(-1)
    cons = [LinearConstraint(A_eq, b_eq, b_eq), LinearConstraint(A_cap, -np.inf, ub_cap),
            LinearConstraint(A_recv, -np.inf, ub_recv)]

    if C:
        r3, c3, v3 = [], [], []
        for c in range(C):
            for t in range(T):
                r = c * T + t
                r3 += [r, r]; c3 += [oF + (E + c) * T + t, oY + c]; v3 += [1.0, -ccap[c]]
        A_act = sp.csr_matrix((v3, (r3, c3)), shape=(C * T, nvar))
        cons.append(LinearConstraint(A_act, -np.inf, 0.0))

    cost = np.zeros(nvar)
    cost[oF:oI] = np.repeat(UNIT_COST_PER_KM * km, T)
    cost[oU:oY] = SHORTAGE_PENALTY
    if C:
        cost[oY:] = cand.fixed_cost
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[oF:oI] = np.maximum(ub_f, 0).reshape(-1)
    ub[oY:] = 1.0
    integrality = np.zeros(nvar)
    integrality[oY:] = 1
    res = milp(cost, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 0.01, "disp": False})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    f = res.x[oF:oI].reshape(nE, T)
    y = res.x[oY:] if C else np.zeros(0)
    info = {"status": int(res.status), "message": res.message, "n_vars": nvar,
            "n_candidates": C, "objective": float(res.fun),
            "gap": float(getattr(res, "mip_gap", np.nan) or 0.0)}
    return f[:E], f[E:], (y > 0.5), info


def execution_schedule(net, f_exist, f_cand, y, T):
    """Execute the absolute lane flows optimized by the MILP."""
    S_exist = f_exist
    S_cand = f_cand * y[:, None] if len(y) else np.zeros((0, T))
    return S_exist, S_cand


def greedy_nearest(net, stranded, p_exposed, T, r_lateral=600.0,
                   max_lanes=None, priority=None):
    """B3: add nearest-DC lanes in priority order, subject to an action budget."""
    parent = {int(d): int(s) for s, d in zip(net.src, net.dst) if net.kind[d] == DEALER}
    ok_dc = np.where((net.kind == DC) & (p_exposed < 0.5))[0]
    cand = Candidates()
    dealers = np.where(stranded)[0]
    if priority is not None:
        dealers = dealers[np.argsort(-np.asarray(priority)[dealers])]
    for d in dealers:
        if max_lanes is not None and len(cand) >= max_lanes:
            break
        pool = ok_dc[ok_dc != parent.get(int(d), -1)]
        near = _nearest(net, int(d), pool, 1, r_lateral)
        if near:
            s = near[0]
            cand.add(s, d, LineString([tuple(net.xy[s]), tuple(net.xy[d])]),
                     1.5 * net.demand[d], "lateral")
    S_cand = np.repeat(np.array([net.demand[d] for d in cand.dst], float)[:, None], T, axis=1) \
        if len(cand) else np.zeros((0, T))
    y = np.ones(len(cand), bool)
    S_exist = np.repeat(net.base_f[:, None], T, axis=1)
    return cand, S_exist, S_cand, y
