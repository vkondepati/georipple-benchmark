"""Lane models: how a lane between two facilities is drawn and how long it is.

``StraightLanes`` reproduces the published benchmark: a straight segment whose length is
scaled by a road factor. ``RoadLanes`` follows the shortest path on the U.S. primary-road
graph; each facility reaches that graph by a local leg to the nearest point on a primary
road, counted at the same road factor.

Each model routes freely with ``route(p, q)`` and returns ``avoiding(polygons)``, a
router that keeps candidate lanes out of a buffered hazard union. Every router returns a
``Lane`` or ``None`` when no acceptable lane exists. ``enrich_lanes`` is the pipeline's
lane-geometry enrichment step: it redraws every existing lane of a network with a model.
"""
from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from .network import ROAD_FACTOR, lane_tau
from .roads import RoadSearch
from .routing import HazardRouter

HAZARD_BUFFER_KM = 15.0     # clearance kept around the forecast footprint union


@dataclass(frozen=True)
class Lane:
    geom: LineString
    km: float


def _local(p, q):
    geom = LineString([tuple(p), tuple(q)])
    return Lane(geom, geom.length * ROAD_FACTOR)


class StraightLanes:
    """Straight segments; km = length x ROAD_FACTOR."""

    name = "straight"

    def route(self, p, q):
        return _local(p, q)

    def avoiding(self, polygons):
        return _AvoidingStraight(HazardRouter(polygons, buffer_km=HAZARD_BUFFER_KM))


class _AvoidingStraight:
    """Visibility-graph detours around the buffered union (the published proxy)."""

    def __init__(self, router):
        self.router = router

    def route(self, p, q):
        geom = self.router.route(p, q)
        return None if geom is None else Lane(geom, geom.length * ROAD_FACTOR)


def _road_lane(search, p, q):
    found = search.route(p, q)
    if found is None:
        return None
    road_km, leg_km, line = found
    return Lane(LineString(line), road_km + ROAD_FACTOR * leg_km)


class RoadLanes:
    """Shortest primary-road path plus local legs to the nearest road point."""

    name = "road"

    def __init__(self, graph):
        self.graph = graph
        self._search = RoadSearch(graph)

    def route(self, p, q):
        return _road_lane(self._search, p, q)

    def avoiding(self, polygons):
        return _AvoidingRoads(self, polygons)


class _AvoidingRoads:
    """Road routing with every piece that touches the buffered hazard union closed.

    Same rule as the straight proxy: an origin inside the union has no lane; a
    destination inside it must be entered anyway, so the free route is kept and its
    forecast exposure is left to the planner.
    """

    def __init__(self, lanes, polygons):
        self.lanes = lanes
        obstacle = unary_union(list(polygons)).buffer(HAZARD_BUFFER_KM)
        self._obstacle = prep(obstacle)
        graph = lanes.graph
        open_pieces = np.ones(len(graph.u), bool)
        open_pieces[graph.piece_tree.query(obstacle, predicate="intersects")] = False
        self._search = RoadSearch(graph, open_pieces)

    def route(self, p, q):
        if self._obstacle.contains(Point(p)):
            return None
        if self._obstacle.contains(Point(q)):
            return self.lanes.route(p, q)
        return _road_lane(self._search, p, q)


def enrich_lanes(net, model):
    """Redraw every existing lane with ``model``; only geometry, km and transit change."""
    routed = [model.route(net.xy[s], net.xy[d]) for s, d in zip(net.src, net.dst, strict=True)]
    if any(lane is None for lane in routed):
        raise ValueError("the lane model could not connect every existing lane")
    km = np.array([lane.km for lane in routed])
    return replace(net, geoms=[lane.geom for lane in routed], km=km, tau=lane_tau(km))
