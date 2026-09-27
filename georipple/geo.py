"""Approximate U.S. metro locations and a local planar projection.

Coordinates and populations are rounded approximations intended only to
give the synthetic network a realistic spatial distribution. They are not
an authoritative dataset.
"""
import numpy as np

# name, lat, lon, approx. metro population (millions)
CITIES = [
    ("New York", 40.71, -74.01, 19.5), ("Los Angeles", 34.05, -118.24, 13.0),
    ("Chicago", 41.88, -87.63, 9.4), ("Dallas", 32.78, -96.80, 7.9),
    ("Houston", 29.76, -95.37, 7.3), ("Washington", 38.91, -77.04, 6.3),
    ("Philadelphia", 39.95, -75.17, 6.2), ("Miami", 25.76, -80.19, 6.1),
    ("Atlanta", 33.75, -84.39, 6.2), ("Boston", 42.36, -71.06, 4.9),
    ("Phoenix", 33.45, -112.07, 5.0), ("San Francisco", 37.77, -122.42, 4.6),
    ("Riverside", 33.95, -117.40, 4.6), ("Detroit", 42.33, -83.05, 4.3),
    ("Seattle", 47.61, -122.33, 4.0), ("Minneapolis", 44.98, -93.27, 3.7),
    ("San Diego", 32.72, -117.16, 3.3), ("Tampa", 27.95, -82.46, 3.3),
    ("Denver", 39.74, -104.99, 3.0), ("Baltimore", 39.29, -76.61, 2.8),
    ("St. Louis", 38.63, -90.20, 2.8), ("Orlando", 28.54, -81.38, 2.7),
    ("Charlotte", 35.23, -80.84, 2.7), ("San Antonio", 29.42, -98.49, 2.6),
    ("Portland", 45.52, -122.68, 2.5), ("Sacramento", 38.58, -121.49, 2.4),
    ("Pittsburgh", 40.44, -80.00, 2.4), ("Austin", 30.27, -97.74, 2.4),
    ("Las Vegas", 36.17, -115.14, 2.3), ("Cincinnati", 39.10, -84.51, 2.3),
    ("Kansas City", 39.10, -94.58, 2.2), ("Columbus", 39.96, -83.00, 2.1),
    ("Indianapolis", 39.77, -86.16, 2.1), ("Cleveland", 41.50, -81.69, 2.1),
    ("Nashville", 36.16, -86.78, 2.0), ("Jacksonville", 30.33, -81.66, 1.6),
    ("Milwaukee", 43.04, -87.91, 1.6), ("Oklahoma City", 35.47, -97.52, 1.4),
    ("Memphis", 35.15, -90.05, 1.3), ("Louisville", 38.25, -85.76, 1.3),
    ("Richmond", 37.54, -77.44, 1.3), ("New Orleans", 29.95, -90.07, 1.3),
    ("Salt Lake City", 40.76, -111.89, 1.3), ("Birmingham", 33.52, -86.80, 1.1),
    ("Tulsa", 36.15, -95.99, 1.0), ("Omaha", 41.26, -95.93, 1.0),
    ("Albuquerque", 35.08, -106.65, 0.9), ("El Paso", 31.76, -106.49, 0.9),
    ("Baton Rouge", 30.45, -91.19, 0.9), ("Knoxville", 35.96, -83.92, 0.9),
    ("Little Rock", 34.75, -92.29, 0.75), ("Des Moines", 41.59, -93.62, 0.7),
    ("Jackson", 32.30, -90.18, 0.6), ("Shreveport", 32.52, -93.75, 0.4),
    ("Beaumont", 30.08, -94.13, 0.4), ("Corpus Christi", 27.80, -97.40, 0.4),
    ("Lubbock", 33.58, -101.86, 0.3), ("Amarillo", 35.22, -101.83, 0.27),
    ("Lake Charles", 30.23, -93.22, 0.2),
]

SUPPLIER_CITIES = [
    "Detroit", "Chicago", "Cleveland", "Columbus", "Indianapolis", "Louisville",
    "Nashville", "Birmingham", "Atlanta", "Houston", "Dallas", "San Antonio",
    "Memphis", "St. Louis", "Kansas City", "Los Angeles", "Phoenix",
    "Charlotte", "Pittsburgh", "Milwaukee",
]

LAT0, LON0 = 37.0, -96.0
KX = 111.32 * np.cos(np.radians(LAT0))
KY = 110.57


def project(lon, lat):
    """Equirectangular projection to kilometres around (LAT0, LON0)."""
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY


def project_coords(coords):
    """Project a sequence of (lon, lat) pairs to (x_km, y_km) pairs."""
    return [tuple(map(float, project(lon, lat))) for lon, lat in coords]


def city_table():
    names = [c[0] for c in CITIES]
    lat = np.array([c[1] for c in CITIES])
    lon = np.array([c[2] for c in CITIES])
    pop = np.array([c[3] for c in CITIES])
    x, y = project(lon, lat)
    return names, np.stack([x, y], axis=1), pop
