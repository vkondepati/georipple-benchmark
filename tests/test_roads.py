import os
import sys

import numpy as np
import pytest
from shapely.geometry import Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from georipple.geo import CITIES, project
from georipple.lanes import RoadLanes, StraightLanes, enrich_lanes
from georipple.network import generate
from georipple.roads import load_roads
from georipple.simulate import baseline_schedule, existing_lanes, initial_pipeline, simulate


def city(name):
    _, lat, lon, _ = next(c for c in CITIES if c[0] == name)
    return np.array(project(lon, lat), float)


@pytest.fixture(scope="module")
def roads():
    return RoadLanes(load_roads())


def test_road_graph_is_one_connected_piece():
    graph = load_roads()
    assert len(graph.xy) > 20_000
    assert graph.is_connected()


def test_houston_to_dallas_follows_the_interstate_distance(roads):
    lane = roads.route(city("Houston"), city("Dallas"))
    # I-45 between the two downtowns is about 385 km.
    assert 350.0 < lane.km < 425.0
    assert lane.geom.coords[0] == tuple(city("Houston"))
    assert lane.geom.coords[-1] == tuple(city("Dallas"))
    assert len(lane.geom.coords) > 10


def test_avoiding_router_detours_around_a_closed_corridor(roads):
    houston, dallas = city("Houston"), city("Dallas")
    closure = Point((houston + dallas) / 2).buffer(40.0)
    direct = roads.route(houston, dallas)
    assert direct.geom.intersects(closure)
    detour = roads.avoiding([closure]).route(houston, dallas)
    assert detour is not None
    assert not detour.geom.intersects(closure.buffer(14.0))
    assert detour.km > direct.km


def test_avoiding_router_rejects_origins_inside_and_keeps_destinations_inside(roads):
    houston, dallas = city("Houston"), city("Dallas")
    closure = Point(houston).buffer(30.0)
    router = roads.avoiding([closure])
    assert router.route(houston, dallas) is None
    into = router.route(dallas, houston)
    assert into is not None
    assert into.km == pytest.approx(roads.route(dallas, houston).km)


def test_straight_enrichment_keeps_the_published_geometry():
    net = generate(0, n_dc=6, n_dealer=30)
    same = enrich_lanes(net, StraightLanes())
    assert np.array_equal(same.km, net.km) and np.array_equal(same.tau, net.tau)
    for e, geom in enumerate(same.geoms):
        assert geom.equals_exact(net.geoms[e], 0.0)
        assert same.km[e] == geom.length * 1.25


def test_road_lanes_change_only_lane_geometry_and_timing(roads):
    straight = generate(3, n_dc=10, n_dealer=80)
    road = enrich_lanes(straight, roads)
    for name in ("xy", "kind", "demand", "prod", "node_cap", "recv_cap", "I0",
                 "src", "dst", "cap", "base_f"):
        assert np.array_equal(getattr(straight, name), getattr(road, name)), name
    assert not np.array_equal(straight.km, road.km)
    T = 14
    calm_n, calm_e = np.zeros((road.n, T)), np.zeros((len(road.src), T))
    r = simulate(road, existing_lanes(road), calm_n, calm_e, baseline_schedule(road, T),
                 T, pipe=initial_pipeline(road, T))
    assert np.all(r.sigma[road.dealers] == T)
