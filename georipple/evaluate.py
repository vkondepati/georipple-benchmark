"""Ground-truth execution and evaluation metrics."""
import numpy as np

from .hazards import node_exposure, edge_exposure
from .network import DEALER
from .resolve import UNIT_COST_PER_KM
from .simulate import Lanes, simulate, initial_pipeline

DELAY_RATE = 0.3   # mean extra days per shipment (Poisson) in ground truth


def truth_delays(net, seed, T):
    return np.random.default_rng([seed, 7]).poisson(DELAY_RATE, (len(net.src), T))


def _cand_delays(cand, seed, T):
    if not len(cand):
        return np.zeros((0, T), int)
    return np.vstack([np.random.default_rng([seed, 11, s, d]).poisson(DELAY_RATE, T)
                      for s, d in zip(cand.src, cand.dst)])


def execute_truth(net, h_real, T, seed, cand=None, S_exist=None, S_cand=None, y=None):
    """Run the higher-fidelity simulator (realized hazard + stochastic delays)."""
    xn = node_exposure(net.xy, h_real, T)
    xe = edge_exposure(net.geoms, h_real, T)
    d_exist = truth_delays(net, seed, T)
    if S_exist is None:
        S_exist = np.repeat(net.base_f[:, None], T, axis=1)
    if cand is not None and len(cand):
        cs, cd, ckm, ctau, ccap = cand.arrays()
        lanes = Lanes(np.concatenate([net.src, cs]), np.concatenate([net.dst, cd]),
                      np.concatenate([net.tau, ctau]), np.concatenate([net.cap, ccap]))
        xe_all = np.vstack([xe, edge_exposure(cand.geoms, h_real, T)])
        S = np.vstack([S_exist, S_cand])
        delays = np.vstack([d_exist, _cand_delays(cand, seed, T)])
    else:
        lanes = Lanes(net.src, net.dst, net.tau, net.cap)
        xe_all, S, delays = xe, S_exist, d_exist
    r = simulate(net, lanes, xn, xe_all, S, T, pipe=initial_pipeline(net, T), delays=delays)
    cost = 0.0
    if cand is not None and len(cand):
        E = len(net.src)
        _, _, ckm, _, _ = cand.arrays()
        used = y & (r.shipped[E:].sum(1) > 0) if y is not None else r.shipped[E:].sum(1) > 0
        cost = float((cand.fixed_cost * used).sum()
                     + (r.shipped[E:].sum(1) * UNIT_COST_PER_KM * ckm).sum())
    return r, cost


def attributable_stockouts(net, sigma_hazard, sigma_nohazard, T):
    """Dealers whose stockout is caused (or advanced) by the hazard, not by
    ordinary transit-delay noise. Returns a boolean mask over dealers."""
    dl = net.dealers
    return (sigma_hazard[dl] < T) & (sigma_hazard[dl] < sigma_nohazard[dl])


def prediction_metrics(net, p, sig_hat, sigma_true, T, truth_mask):
    dl = net.dealers
    truth = truth_mask
    pred = p[dl] >= 0.5
    tp = (truth & pred).sum()
    precision = tp / pred.sum() if pred.sum() else np.nan
    recall = tp / truth.sum() if truth.sum() else np.nan
    union = truth | pred
    s_hat = np.where(pred, np.minimum(sig_hat[dl], T), T)
    mae = float(np.abs(s_hat[union] - np.minimum(sigma_true[dl][union], T)).mean()) if union.any() else np.nan
    return {"precision": float(precision), "recall": float(recall), "mae_days": mae,
            "n_true": int(truth.sum()), "n_pred": int(pred.sum())}


def resolution_metrics(net, r, affected, h_real, T, cost):
    dl = net.dealers
    A = dl[affected]
    dem = net.demand
    unmet = float((dem[dl, None] - r.served[dl]).clip(0).sum())
    if len(A) == 0:
        return {"fill_rate": np.nan, "unmet_units": unmet, "ttr95_days": np.nan,
                "cost_k": cost / 1000.0, "n_affected": 0}
    fill = float(r.served[A].sum() / (dem[A].sum() * T))
    daily = r.served[A].sum(0) / dem[A].sum()
    bad = np.where(daily < 0.95)[0]
    ttr = 0.0 if len(bad) == 0 else float(bad.max() + 1 - h_real.start)
    return {"fill_rate": fill, "unmet_units": unmet, "ttr95_days": max(ttr, 0.0),
            "cost_k": cost / 1000.0, "n_affected": int(len(A))}
