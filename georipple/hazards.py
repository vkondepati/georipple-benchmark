"""Stylized hazard footprints, forecast ensembles, and node/edge exposure.

The three footprints are hand-drawn polygons loosely shaped after event
types that affect U.S. freight (a Gulf Coast hurricane flood, a statewide
winter storm, a Mississippi River flood). They are NOT reconstructions of
any specific historical event. Replace them with real GeoJSON footprints
(see README) for a historically grounded study.
"""
from dataclasses import dataclass
import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep

from .geo import project_coords

L_MIN_KM = 1.0   # any lane crossing longer than this is fully exposed


@dataclass
class Hazard:
    name: str
    poly: object        # shapely (Multi)Polygon, projected km
    start: int          # first affected day (inclusive)
    end: int            # last affected day (inclusive)
    severity: float     # fraction of capacity lost inside the footprint

    def active(self, T):
        t = np.arange(T)
        return ((t >= self.start) & (t <= self.end)).astype(float)


def base_hazards():
    hurricane = Polygon(project_coords([
        (-97.2, 27.8), (-96.8, 30.6), (-95.0, 31.0), (-92.8, 30.8),
        (-92.6, 29.6), (-94.5, 29.3), (-96.0, 28.2)]))
    winter = Polygon(project_coords([
        (-103.0, 32.0), (-100.0, 34.8), (-95.5, 34.5), (-93.8, 32.5),
        (-94.0, 29.8), (-97.3, 26.5), (-99.5, 27.5), (-101.5, 29.5)]))
    river = LineString(project_coords([
        (-91.4, 40.3), (-90.2, 38.6), (-89.5, 37.0), (-90.05, 35.15),
        (-91.0, 33.4), (-91.2, 32.3), (-91.19, 30.45)])).buffer(30.0)
    return [
        Hazard("gulf_hurricane_flood", hurricane, 2, 10, 0.9),
        Hazard("statewide_winter_storm", winter, 2, 8, 0.6),
        Hazard("river_corridor_flood", river, 3, 14, 1.0),
    ]


def forecast_ensemble(h, rng, M=8, shift_km=40.0):
    """Perturbed forecasts of hazard ``h`` (position, extent, timing, severity)."""
    ens = []
    for _ in range(M):
        dx, dy = rng.normal(0, shift_km, 2)
        s = rng.uniform(0.85, 1.25)
        p = affinity.scale(h.poly, s, s, origin="centroid")
        p = affinity.translate(p, dx, dy)
        start = max(1, h.start + int(rng.integers(-1, 2)))
        end = max(start + 1, h.end + int(rng.integers(-2, 3)))
        sev = float(np.clip(h.severity * rng.uniform(0.85, 1.1), 0, 1))
        ens.append(Hazard(h.name, p, start, end, sev))
    return ens


def node_exposure(xy, h, T):
    pp = prep(h.poly)
    inside = np.array([pp.contains(Point(p)) for p in xy], dtype=float)
    return np.outer(inside * h.severity, h.active(T))


def edge_exposure(geoms, h, T):
    pp = prep(h.poly)
    x = np.zeros(len(geoms))
    for i, g in enumerate(geoms):
        if pp.intersects(g):
            length = g.intersection(h.poly).length
            x[i] = h.severity * min(1.0, length / L_MIN_KM)
    return np.outer(x, h.active(T))
