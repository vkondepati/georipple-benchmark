"""Hazard-aware resolution: candidate recovery lanes + time-expanded MILP."""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .lanes import StraightLanes
from .network import DC, DEALER, SUP, lane_tau
from .simulate import baseline_schedule, existing_lanes, simulate

# Illustrative synthetic cost units (CU); these are not calibrated dollars.
UNIT_COST_PER_KM = 0.01     # CU per unit per road km
FIXED_BASE = 1000.0         # CU fixed cost to open a lane
FIXED_PER_KM = 1.0          # CU per road km, fixed part
SHORTAGE_PENALTY = 100.0    # CU per unit of unmet demand


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

    def add(self, s, d, lane, cap, kind):
        self.src.append(int(s)); self.dst.append(int(d))
        self.geoms.append(lane.geom); self.km.append(lane.km)
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


def _suppliers_of(net, kind):
    """Every current supplying node of each node of the given kind."""
    out = {}
    for s, d in zip(net.src, net.dst, strict=True):
        if net.kind[d] == kind:
            out.setdefault(int(d), set()).add(int(s))
    return out


def _new_origins(pool, current):
    return pool[~np.isin(pool, sorted(current))]


def generate_candidates(net, stranded, score, p_exposed, router=None, K=10,
                        k=3, r_lateral=600.0, r_alt=1500.0, tau_max=4):
    """Spatial candidate generation for the top-K STCI facilities.

    ``router`` draws the lanes: a lane model's ``avoiding`` router keeps them clear of
    the forecast hazard, a bare lane model ignores it (ablation A2), and None means
    straight lanes. No candidate duplicates a lane the node already receives.
    """
    top = [int(v) for v in np.argsort(-score)[:K] if score[v] > 0]
    ch = net.children()
    dcs_of = _suppliers_of(net, DEALER)
    sup_of = _suppliers_of(net, DC)

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

    route = (router or StraightLanes()).route

    def lane(a, b):
        found = route(net.xy[a], net.xy[b])
        if found is None or lane_tau(found.km) > tau_max:
            return None
        return found

    for d in sorted(target_dealers):
        pool = _new_origins(ok_dc, dcs_of.get(d, set()))
        for s in _nearest(net, d, pool, k, r_lateral):
            found = lane(s, d)
            if found is not None:
                cand.add(s, d, found, 1.5 * net.demand[d], "lateral")
    for v in sorted(target_dcs):
        pool = _new_origins(ok_sup, sup_of.get(v, set()))
        if len(pool) == 0:
            continue
        for s in _nearest(net, v, pool, k, r_alt):
            found = lane(s, v)
            if found is not None:
                cand.add(s, v, found, 0.6 * net.node_cap[v] / 1.3, "alt_source")
    return cand, top


def plan_milp(net, cand, xn, xe, xc, T, pipe, time_limit=60.0):
    """Solve the time-expanded MILP (paper Eqs. 5-10) on planning exposures.

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
    oF = 0
    oI = nE * T
    oU = oI + n * T
    oR = oU + R * T              # receipts after inbound-capacity queueing
    oQ = oR + n * T              # end-of-day inbound queue
    oY = oQ + n * T
    nvar = oY + C

    rows, cols, vals = [], [], []
    def row(v, t):
        return v * T + t
    for e in range(nE):
        for t in range(T):
            j = oF + e * T + t
            rows.append(row(src[e], t)); cols.append(j); vals.append(1.0)
    for v in range(n):
        for t in range(T):
            j = oI + v * T + t
            rows.append(row(v, t)); cols.append(j); vals.append(1.0)
            if t + 1 < T:
                rows.append(row(v, t + 1)); cols.append(j); vals.append(-1.0)
            rows.append(row(v, t)); cols.append(oR + v * T + t); vals.append(-1.0)
    for v in dealers:
        for t in range(T):
            rows.append(row(v, t)); cols.append(oU + dl_index[int(v)] * T + t); vals.append(-1.0)
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(n * T, nvar))
    b = np.zeros((n, T))
    is_sup = net.kind == SUP
    b[is_sup] += (net.prod[is_sup, None] * (1 - xn[is_sup]))
    b[dealers] -= net.demand[dealers, None]
    b[:, 0] += net.I0
    b_eq = b.reshape(-1)

    # inbound queue: Q_t = Q_{t-1} + pipeline_t + arrivals_t - receipts_t
    qr, qc, qv = [], [], []
    for v in range(n):
        for t in range(T):
            qr.append(row(v, t)); qc.append(oQ + v * T + t); qv.append(1.0)
            if t > 0:
                qr.append(row(v, t)); qc.append(oQ + v * T + t - 1); qv.append(-1.0)
            qr.append(row(v, t)); qc.append(oR + v * T + t); qv.append(1.0)
    for e in range(nE):
        for t in range(T):
            at = t + tau[e]
            if at < T:
                qr.append(row(dst[e], at)); qc.append(oF + e * T + t); qv.append(-1.0)
    A_queue = sp.csr_matrix((qv, (qr, qc)), shape=(n * T, nvar))

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
    cons = [LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_queue, pipe.reshape(-1), pipe.reshape(-1)),
            LinearConstraint(A_cap, -np.inf, ub_cap)]

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
    cost[oU:oR] = SHORTAGE_PENALTY
    if C:
        cost[oY:] = cand.fixed_cost
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[oF:oI] = np.maximum(ub_f, 0).reshape(-1)
    baseline = simulate(net, existing_lanes(net), xn, xe, baseline_schedule(net, T),
                        T, pipe=pipe).shipped
    lb[oF:oF + E * T] = np.minimum(baseline, ub_f[:E]).reshape(-1)
    ub[oR:oQ] = np.maximum(net.recv_cap[:, None] * (1 - xn), 0).reshape(-1)
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


def additive_schedule(net, f_exist, f_cand, y, T):
    """Additive dispatch (GeoRipple): each existing lane ships the larger of its normal
    daily quantity and the planned flow; activated candidate lanes ship planned flows.

    The realized hazard still curtails what can move, so a correct forecast leaves
    roughly the planned flows, while a wrong one leaves normal service in place.
    """
    S_exist = np.maximum(net.base_f[:, None], f_exist)
    S_cand = f_cand * y[:, None] if len(y) else np.zeros((0, T))
    return S_exist, S_cand


def execution_schedule(net, f_exist, f_cand, y, T):
    """Absolute dispatch (ablation A4): ship the MILP's existing-lane flows as planned.

    Wherever the forecast impairs a lane or facility, this withholds normal
    replenishment even if the realized hazard never arrives there.
    """
    S_exist = f_exist
    S_cand = f_cand * y[:, None] if len(y) else np.zeros((0, T))
    return S_exist, S_cand


def greedy_nearest(net, stranded, p_exposed, T, r_lateral=600.0,
                   max_lanes=None, priority=None, router=None):
    """B3: add nearest-DC lanes in priority order, subject to an action budget.

    ``router`` is the network's lane model (straight if None); hazards are ignored.
    """
    route = (router or StraightLanes()).route
    dcs_of = _suppliers_of(net, DEALER)
    ok_dc = np.where((net.kind == DC) & (p_exposed < 0.5))[0]
    cand = Candidates()
    dealers = np.where(stranded)[0]
    if priority is not None:
        dealers = dealers[np.argsort(-np.asarray(priority)[dealers])]
    for d in dealers:
        if max_lanes is not None and len(cand) >= max_lanes:
            break
        pool = _new_origins(ok_dc, dcs_of.get(int(d), set()))
        near = _nearest(net, int(d), pool, 1, r_lateral)
        if near:
            s = near[0]
            cand.add(s, d, route(net.xy[s], net.xy[d]), 1.5 * net.demand[d], "lateral")
    S_cand = np.repeat(np.array([net.demand[d] for d in cand.dst], float)[:, None], T, axis=1) \
        if len(cand) else np.zeros((0, T))
    y = np.ones(len(cand), bool)
    S_exist = np.repeat(net.base_f[:, None], T, axis=1)
    return cand, S_exist, S_cand, y
