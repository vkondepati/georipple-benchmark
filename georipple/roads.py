"""U.S. primary-road graph for road-network lanes.

The graph is built by ``experiments/prepare_roads.py`` from Census TIGER/Line 2023
Primary Roads and stored as ``data/us-primary-roads.json.gz`` (lon/lat). It is loaded
once, projected to the benchmark's planar km frame, and searched with Dijkstra.
"""
import functools
import gzip
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .geo import project

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "us-primary-roads.json.gz"


@dataclass(frozen=True, eq=False)
class RoadGraph:
    """Junctions and the road pieces between them; one piece per junction pair."""
    xy: np.ndarray        # (N, 2) junction coordinates, km
    u: np.ndarray         # (M,) lower junction id of each piece
    v: np.ndarray         # (M,) higher junction id
    km: np.ndarray        # (M,) piece length
    lines: tuple          # (M,) LineStrings in km, oriented u -> v

    def adjacency(self, open_pieces):
        """Symmetric sparse adjacency over the open pieces."""
        u, v, w = self.u[open_pieces], self.v[open_pieces], self.km[open_pieces]
        n = len(self.xy)
        return coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()

    def is_connected(self):
        everything = np.ones(len(self.u), bool)
        return connected_components(self.adjacency(everything), directed=False)[0] == 1

    @functools.cached_property
    def piece_of(self):
        return {(a, b): i for i, (a, b) in enumerate(zip(self.u.tolist(), self.v.tolist(),
                                                          strict=True))}

    @functools.cached_property
    def piece_tree(self):
        return STRtree(self.lines)


@functools.lru_cache(maxsize=2)
def load_roads(path=DEFAULT_PATH):
    with gzip.open(path, "rt") as f:
        payload = json.load(f)
    lonlat = np.asarray(payload["nodes"], float)
    xy = np.column_stack(project(lonlat[:, 0], lonlat[:, 1]))
    u, v, lines = [], [], []
    for a, b, flat in payload["edges"]:
        ll = np.asarray(flat, float).reshape(-1, 2)
        lines.append(LineString(np.column_stack(project(ll[:, 0], ll[:, 1]))))
        u.append(a); v.append(b)
    u, v = np.asarray(u, int), np.asarray(v, int)
    if np.any(u >= v) or len(set(zip(u.tolist(), v.tolist(), strict=True))) != len(u):
        raise ValueError(f"{path}: road pieces must be unique junction pairs with u < v")
    return RoadGraph(xy=xy, u=u, v=v, km=np.array([ln.length for ln in lines]),
                     lines=tuple(lines))


@dataclass(frozen=True)
class Access:
    """Where a facility joins the road: a point on one piece, plus a local leg."""
    piece: int
    along: float      # km from the piece's u end to the access point
    leg: float        # straight km from the facility to the access point


class RoadSearch:
    """Shortest road paths between arbitrary points over one set of open pieces.

    Facilities join the largest connected part of the open network at the nearest point
    on any piece. Paths run from either end of the origin piece to either end of the
    destination piece. Recent single-source trees are kept (about 0.4 MB each).
    """

    CACHED_TREES = 256

    def __init__(self, graph, open_pieces=None):
        self.graph = graph
        keep = np.ones(len(graph.u), bool) if open_pieces is None else open_pieces
        self.matrix = graph.adjacency(keep)
        _, label = connected_components(self.matrix, directed=False)
        piece_label = label[graph.u]
        main = np.bincount(piece_label[keep]).argmax()
        self.usable = np.where(keep & (piece_label == main))[0]
        self._tree = STRtree([graph.lines[i] for i in self.usable])
        self._trees = OrderedDict()

    def access(self, p):
        point = Point(p)
        i = int(self.usable[self._tree.nearest(point)])
        along = float(self.graph.lines[i].project(point))
        leg = float(point.distance(self.graph.lines[i].interpolate(along)))
        return Access(i, along, leg)

    def _distances(self, junction):
        if junction in self._trees:
            self._trees.move_to_end(junction)
        else:
            self._trees[junction] = dijkstra(self.matrix, directed=False, indices=junction,
                                             return_predecessors=True)
            if len(self._trees) > self.CACHED_TREES:
                self._trees.popitem(last=False)
        return self._trees[junction]

    def _junction_path(self, a, b):
        _, pred = self._distances(a)
        nodes = [b]
        while nodes[-1] != a:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        parts = []
        for s, t in zip(nodes[:-1], nodes[1:], strict=True):
            line = self.graph.lines[self.graph.piece_of[(s, t) if s < t else (t, s)]]
            c = np.asarray(line.coords)
            parts.append(c if s < t else c[::-1])
        return parts

    def _ends(self, acc):
        g = self.graph
        return ((int(g.u[acc.piece]), acc.along, 0.0),
                (int(g.v[acc.piece]), g.km[acc.piece] - acc.along, g.km[acc.piece]))

    def route(self, p, q):
        """(road km, local-leg km, polyline from p to q) or None if unreachable."""
        A, B = self.access(p), self.access(q)
        best = None
        if A.piece == B.piece:
            best = (abs(A.along - B.along), None)
        for ea, da, at_a in self._ends(A):
            dist, _ = self._distances(ea)
            for eb, db, at_b in self._ends(B):
                total = da + dist[eb] + db
                if np.isfinite(total) and (best is None or total < best[0]):
                    best = (float(total), (ea, at_a, eb, at_b))
        if best is None:
            return None
        lines = self.graph.lines
        if best[1] is None:
            parts = [_coords(substring(lines[A.piece], A.along, B.along))]
        else:
            ea, at_a, eb, at_b = best[1]
            parts = ([_coords(substring(lines[A.piece], A.along, at_a))]
                     + self._junction_path(ea, eb)
                     + [_coords(substring(lines[B.piece], at_b, B.along))])
        line = np.vstack([np.asarray(p, float)[None]] + parts + [np.asarray(q, float)[None]])
        keep = np.ones(len(line), bool)
        keep[1:] = np.any(line[1:] != line[:-1], axis=1)
        return best[0], A.leg + B.leg, line[keep]


def _coords(geom):
    return np.asarray(geom.coords, float)
