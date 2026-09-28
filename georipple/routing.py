"""Hazard-avoiding lane routing.

Stand-in for a road routing engine with exclusion polygons (e.g. Valhalla
``exclude_polygons`` or GraphHopper custom-model areas). Routes are
straight lines when clear of the obstacle, otherwise the shortest path
through a visibility graph built on the obstacle's (slightly inflated)
boundary vertices. Distances are then scaled by the road factor.
"""
import heapq

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep


class HazardRouter:
    def __init__(self, polygons, buffer_km=15.0, simplify_km=8.0):
        obstacle = unary_union([p for p in polygons]).buffer(buffer_km)
        self.obstacle = obstacle
        self._inner = prep(obstacle.buffer(-0.5))
        self._contains = prep(obstacle)
        ring_src = obstacle.buffer(simplify_km + 1.0).simplify(simplify_km)
        parts = getattr(ring_src, "geoms", [ring_src])
        verts = []
        for p in parts:
            verts.extend(list(p.exterior.coords)[:-1])
        self.V = np.array(verts)
        nv = len(self.V)
        self.adj = [[] for _ in range(nv)]
        for i in range(nv):
            for j in range(i + 1, nv):
                if self._visible(self.V[i], self.V[j]):
                    d = float(np.linalg.norm(self.V[i] - self.V[j]))
                    self.adj[i].append((j, d))
                    self.adj[j].append((i, d))

    def _visible(self, a, b):
        return not self._inner.intersects(LineString([tuple(a), tuple(b)]))

    def route(self, p, q):
        """Return a LineString from p to q avoiding the obstacle, or None.

        If the origin is inside the obstacle the lane is infeasible. If only
        the destination is inside, the lane must enter the hazard anyway, so
        the direct geometry is returned and its exposure is left to the
        planner (it can still be used before hazard onset, e.g. to
        pre-position stock).
        """
        p, q = np.asarray(p, float), np.asarray(q, float)
        if self._contains.contains(Point(p)):
            return None
        if self._contains.contains(Point(q)):
            return LineString([tuple(p), tuple(q)])
        if self._visible(p, q):
            return LineString([tuple(p), tuple(q)])
        nv = len(self.V)
        P, Q = nv, nv + 1
        extra = {P: [], Q: []}
        for i in range(nv):
            if self._visible(p, self.V[i]):
                extra[P].append((i, float(np.linalg.norm(p - self.V[i]))))
            if self._visible(q, self.V[i]):
                extra[Q].append((i, float(np.linalg.norm(q - self.V[i]))))
        q_links = {i: d for i, d in extra[Q]}
        dist, prev = {P: 0.0}, {}
        heap = [(0.0, P)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == Q:
                break
            if d > dist.get(u, np.inf):
                continue
            nbrs = extra[P] if u == P else list(self.adj[u])
            if u != P and u in q_links:
                nbrs = nbrs + [(Q, q_links[u])]
            for w, c in nbrs:
                nd = d + c
                if nd < dist.get(w, np.inf):
                    dist[w], prev[w] = nd, u
                    heapq.heappush(heap, (nd, w))
        if Q not in dist:
            return None
        path, u = [tuple(q)], Q
        while u != P:
            u = prev[u]
            path.append(tuple(p) if u == P else tuple(self.V[u]))
        return LineString(path[::-1])
