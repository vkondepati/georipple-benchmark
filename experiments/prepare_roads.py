"""Build the compact road graph behind the road-network lane model.

Source: U.S. Census Bureau TIGER/Line 2023 Primary Roads (MTFCC S1100: divided,
limited-access highways with interchanges), public domain:
https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/tl_2023_us_primaryroads.zip

Steps: keep lower-48 features, simplify each at 0.01 degree (about 1 km), split every
line where it crosses another, merge chains between junctions, join separate pieces
whose junctions lie within 2 km, keep the largest connected piece, and round lon/lat
to 4 decimals. Output: data/us-primary-roads.json.gz.

Usage: python experiments/prepare_roads.py path/to/tl_2023_us_primaryroads.zip
"""
import argparse
import gzip
import hashlib
import json
import os
import struct
import sys
import zipfile

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.geo import project  # noqa: E402

SOURCE_URL = ("https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/"
              "tl_2023_us_primaryroads.zip")
SOURCE_SHA256 = "97aad145a0c7b6285fb07e5119d3949f31d94e76803801a4c46eb8966e381a20"
LOWER_48 = (-125.0, 24.0, -66.0, 50.0)   # lon_min, lat_min, lon_max, lat_max
SIMPLIFY_DEG = 0.01
SNAP_KM = 2.0
DECIMALS = 4
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "us-primary-roads.json.gz")


def read_dbf(raw):
    """Attribute rows from a dBASE III table."""
    nrec, hlen, rlen = struct.unpack("<IHH", raw[4:12])
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos:pos + 11].split(b"\x00")[0].decode()
        fields.append((name, raw[pos + 16]))
        pos += 32
    rows = []
    for i in range(nrec):
        rec = raw[hlen + i * rlen:hlen + (i + 1) * rlen]
        at, row = 1, {}
        for name, width in fields:
            row[name] = rec[at:at + width].decode("latin-1").strip()
            at += width
        rows.append(row)
    return rows


def read_polylines(raw):
    """Parts of every PolyLine record in a shapefile, as (k, 2) lon/lat arrays."""
    if struct.unpack("<i", raw[32:36])[0] != 3:
        raise ValueError("expected a PolyLine shapefile")
    shapes, off = [], 100
    while off < len(raw):
        clen = struct.unpack(">i", raw[off + 4:off + 8])[0]
        body = raw[off + 8:off + 8 + 2 * clen]
        off += 8 + 2 * clen
        if struct.unpack("<i", body[:4])[0] == 0:
            shapes.append([])
            continue
        nparts, npts = struct.unpack("<ii", body[36:44])
        starts = list(struct.unpack(f"<{nparts}i", body[44:44 + 4 * nparts])) + [npts]
        pts = np.frombuffer(body[44 + 4 * nparts:44 + 4 * nparts + 16 * npts],
                            "<f8").reshape(-1, 2)
        shapes.append([pts[starts[i]:starts[i + 1]] for i in range(nparts)])
    return shapes


def in_lower_48(part):
    lon0, lat0, lon1, lat1 = LOWER_48
    return bool(((part[:, 0] > lon0) & (part[:, 0] < lon1)
                 & (part[:, 1] > lat0) & (part[:, 1] < lat1)).all())


def load_primary_roads(zip_path):
    with open(zip_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"unexpected input SHA-256 {digest}; expected {SOURCE_SHA256}")
    with zipfile.ZipFile(zip_path) as z:
        rows = read_dbf(z.read("tl_2023_us_primaryroads.dbf"))
        shapes = read_polylines(z.read("tl_2023_us_primaryroads.shp"))
    if len(rows) != len(shapes):
        raise SystemExit("attribute and geometry record counts differ")
    return [LineString(part) for row, parts in zip(rows, shapes, strict=True)
            if row["MTFCC"] == "S1100" for part in parts
            if len(part) >= 2 and in_lower_48(part)]


def node_key(coord):
    return (round(coord[0], DECIMALS), round(coord[1], DECIMALS))


def build_graph(lines):
    """Junction-to-junction road pieces as a networkx graph keyed by rounded lon/lat."""
    noded = unary_union(MultiLineString([ln.simplify(SIMPLIFY_DEG) for ln in lines]))
    merged = linemerge(noded)
    graph = nx.Graph()
    for piece in getattr(merged, "geoms", [merged]):
        coords = [node_key(c) for c in piece.coords]
        coords = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
        if len(coords) < 2 or coords[0] == coords[-1]:
            continue
        u, v = coords[0], coords[-1]
        x, y = project(np.array([c[0] for c in coords]), np.array([c[1] for c in coords]))
        km = float(np.hypot(np.diff(x), np.diff(y)).sum())
        if not graph.has_edge(u, v) or km < graph[u][v]["km"]:
            graph.add_edge(u, v, km=km, coords=coords)
    return graph


def snap_components(graph):
    """Join separate pieces whose junctions lie within SNAP_KM of each other."""
    nodes = sorted(graph.nodes)
    xy = np.column_stack(project(np.array([n[0] for n in nodes]),
                                 np.array([n[1] for n in nodes])))
    component = {n: i for i, c in enumerate(nx.connected_components(graph)) for n in c}
    for i, j in sorted(cKDTree(xy).query_pairs(SNAP_KM)):
        a, b = nodes[i], nodes[j]
        if component[a] != component[b] and not graph.has_edge(a, b):
            graph.add_edge(a, b, km=float(np.hypot(*(xy[i] - xy[j]))), coords=[a, b])
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def serialize(graph):
    nodes = sorted(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    edges = []
    for a, b, data in graph.edges(data=True):
        coords = data["coords"] if data["coords"][0] == a else data["coords"][::-1]
        u, v = index[a], index[b]
        if u > v:
            u, v, coords = v, u, coords[::-1]
        edges.append([u, v, [x for c in coords for x in c]])
    edges.sort(key=lambda e: (e[0], e[1], e[2]))
    return {"source": SOURCE_URL, "sha256": SOURCE_SHA256, "simplify_deg": SIMPLIFY_DEG,
            "snap_km": SNAP_KM, "nodes": [list(n) for n in nodes], "edges": edges}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    graph = snap_components(build_graph(load_primary_roads(args.zip_path)))
    payload = serialize(graph)
    data = json.dumps(payload, separators=(",", ":")).encode()
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(data, compresslevel=9, mtime=0))
    os.replace(tmp, args.out)
    total_km = sum(d["km"] for *_, d in graph.edges(data=True))
    print(f"wrote {args.out}: {graph.number_of_nodes()} junctions, "
          f"{graph.number_of_edges()} road pieces, {total_km:,.0f} km")


if __name__ == "__main__":
    main()
