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
