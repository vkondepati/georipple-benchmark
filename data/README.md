# Map boundary data

`us-states.json` is used only as a background layer in the static figures. It
is the U.S. Census Bureau's 2022 state cartographic boundary GeoJSON at
1:20,000,000 scale, distributed through the Census CitySDK data service:

<https://ddbc5tjh37x38.cloudfront.net/20m/2022/state.json>

The service is maintained by the U.S. Census Bureau's CitySDK project and
documents these files as translations of Census cartographic boundaries. U.S.
government works are public domain in the United States. Only geometry and the
state name are read by the figure generator.

- CitySDK source and GeoJSON documentation: <https://github.com/uscensusbureau/citysdk>
- Census cartographic boundary documentation: <https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html>

# Road graph

`us-primary-roads.json.gz` is the road network behind the road-lane model
(`georipple/lanes.py`, `RoadLanes`). It is derived from the U.S. Census Bureau's
TIGER/Line 2023 Primary Roads shapefile (MTFCC S1100: divided, limited-access
highways with interchanges), which is public domain:

<https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/tl_2023_us_primaryroads.zip>
(SHA-256 `97aad145a0c7b6285fb07e5119d3949f31d94e76803801a4c46eb8966e381a20`)

`experiments/prepare_roads.py` rebuilds it from that zip: lower-48 features only,
simplified at 0.01 degree, split at every crossing, merged between junctions,
pieces within 2 km joined, largest connected piece kept, lon/lat rounded to four
decimals. The result has 29,969 junctions and 53,899 road pieces (218,050 km of
carriageway). Treating every crossing as a junction ignores overpasses without
ramps; at national scale that is a deliberate simplification.

- TIGER/Line documentation: <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>
