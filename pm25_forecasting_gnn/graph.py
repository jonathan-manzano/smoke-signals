# Standard library imports
import pickle
import sys
import time
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Set, Tuple, TypedDict, Annotated

# Third-party imports
import numpy as np
import requests
import torch
from bresenham import bresenham
from geopy.distance import geodesic
from metpy.units import units
from metpy.calc import wind_direction
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# Constants
PROJ_DIR = Path(__file__).resolve().parent.parent
PM25GNN_DIR = PROJ_DIR / "pm25_forecasting_gnn"
sys.path.append(str(PM25GNN_DIR))
RAW_DATA_DIR = PM25GNN_DIR / "data" / "raw"
CITY_FILE = RAW_DATA_DIR / "locations.txt"
ALTITUDE_FILE = RAW_DATA_DIR / "alt.pkl"
DEFAULT_ALTITUDE = 0.0
API_DELAY = 1.0  # seconds to wait between API calls
ALTITUDE_API_URL = "https://api.opentopodata.org/v1/test-dataset"


# Type definitions
class NodeData(TypedDict):
    """TypedDict for storing node data."""

    city: str
    altitude: float
    lon: float
    lat: float


Coordinate = Tuple[float, float]
AltitudeDict = Dict[Coordinate, float]
NodeDict = OrderedDict[int, NodeData]
CoordinateArray = Annotated[np.ndarray, "Array of coordinate values"]


class Graph:
    def __init__(
        self,
        distance_threshold: float = 3,
        altitude_threshold: float = 1200,
        factor: int = 10,
        use_altitude: bool = True,
    ) -> None:
        """Initialize the Graph object with geographical and altitude data."""
        self.distance_threshold = distance_threshold
        self.altitude_threshold = altitude_threshold
        self.factor = factor
        self.use_altitude = use_altitude

        self.count: Set[Tuple[float, float]] = set()
        self.alt_dict = self.__class__._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index, self.edge_attr = self._gen_edges()

        if self.use_altitude:
            self._update_edges()

        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    @staticmethod
    def _load_altitude() -> Dict[Tuple[float, float], float]:
        """Load altitude data from pickle file."""
        if not ALTITUDE_FILE.exists():
            raise FileNotFoundError(f"Altitude file not found: {ALTITUDE_FILE}")

        with open(ALTITUDE_FILE, "rb") as f:
            altitude_raw = pickle.load(f)

        # Convert NumPy float64 values to Python floats
        altitude = {
            (float(lat), float(lon)): float(alt)
            for (lat, lon), alt in altitude_raw.items()
        }

        return altitude

    def _get_alt(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """
        Get altitude for given latitude and longitude.

        Args:
            latitude: Array of latitude values
            longitude: Array of longitude values

        Returns:
            Array of altitude values
        """
        alt = np.full(len(latitude), 0.0)

        for i in range(len(latitude)):
            lat = float(latitude[i]) / self.factor
            lon = float(longitude[i]) / self.factor

            if (lat, lon) in self.alt_dict:
                alt[i] = self.alt_dict[(lat, lon)]
                continue

            try:
                response = requests.get(
                    f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}"
                )
                response.raise_for_status()  # Check for HTTP errors
                data = response.json()
                x = data["results"][0]["elevation"]
                alt[i] = float(x)
                time.sleep(1.0)
            except Exception as e:
                print(f"Error fetching altitude data: {e}")
                # Default value of 0.0 is already set

        return alt

    def _get_and_store_altitude(self, lat: int, lon: int) -> float:
        """
        Get altitude for a point and store it in the altitude dictionary.

        Args:
            lat: Latitude value (unscaled)
            lon: Longitude value (unscaled)

        Returns:
            The altitude value as a float
        """
        # Get altitude
        altitude_array = self._get_alt(np.full(1, lat), np.full(1, lon))

        # Scale coordinates
        lat_scaled = lat / self.factor
        lon_scaled = lon / self.factor

        # Store altitude and add to count
        altitude = float(altitude_array[0])
        self.alt_dict[(lat_scaled, lon_scaled)] = altitude
        self.count.add((lat_scaled, lon_scaled))

        return altitude

    def _gen_nodes(self) -> OrderedDict:
        """
        Generate nodes from city data file.

        Returns:
            OrderedDict containing node data
        """
        nodes = OrderedDict()

        with open(CITY_FILE, "r") as f:
            for line in f:
                try:
                    idx, city, lat, lon = line.rstrip("\n").split(" ")
                    idx = int(idx)
                    lon = np.full(1, int(float(lon) * self.factor))
                    lat = np.full(1, int(float(lat) * self.factor))

                    lat_scaled = float(lat[0]) / self.factor
                    lon_scaled = float(lon[0]) / self.factor

                    if (lat_scaled, lon_scaled) not in self.alt_dict:
                        altitude_array = self._get_alt(lat, lon)
                        altitude = float(
                            altitude_array[0]
                        )  # Explicitly convert to float
                        self.alt_dict[(lat_scaled, lon_scaled)] = altitude
                    else:
                        altitude = self.alt_dict[(lat_scaled, lon_scaled)]

                    self.count.add((lat_scaled, lon_scaled))

                    nodes[idx] = {
                        "city": city,
                        "altitude": altitude,
                        "lon": lon_scaled,
                        "lat": lat_scaled,
                    }
                except (ValueError, IndexError) as e:
                    print(f"Error processing line: {line.strip()}. Error: {e}")

        return nodes

    def _add_node_attr(self) -> np.ndarray:
        """
        Add node attributes.

        Returns:
            Array of node attributes
        """
        altitude_arr = []

        for i in self.nodes:
            altitude = self.nodes[i]["altitude"]
            altitude_arr.append(altitude)

        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)

        return node_attr

    def traverse_graph(self) -> Tuple[List[int], List[str], List[float], List[float]]:
        """
        Traverse graph and return node information.

        Returns:
            Tuple containing lists of node indices, city names, longitudes, and latitudes
        """
        idx = []
        cities = []
        longitudes = []
        latitudes = []

        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]["city"]
            lon, lat = self.nodes[i]["lon"], self.nodes[i]["lat"]
            longitudes.append(lon)
            latitudes.append(lat)
            cities.append(city)

        return idx, cities, longitudes, latitudes

    def gen_lines(self) -> List[Tuple[List[float], List[float]]]:
        """
        Generate lines representing edges.

        Returns:
            List of line coordinates
        """
        lines = []

        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]["lat"], self.nodes[src]["lon"]
            dest_lat, dest_lon = self.nodes[dest]["lat"], self.nodes[dest]["lon"]
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate edges and edge attributes.

        Returns:
            Tuple containing edge indices and edge attributes
        """
        coords = []

        for i in self.nodes:
            coords.append([self.nodes[i]["lon"], self.nodes[i]["lat"]])

        dist = distance.cdist(coords, coords, "euclidean")
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.distance_threshold] = 1

        assert adj.shape == dist.shape
        dist = dist * adj

        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direction_arr = []
        dist_kilometer = []

        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]["lat"], self.nodes[src]["lon"]
            dest_lat, dest_lon = self.nodes[dest]["lat"], self.nodes[dest]["lon"]

            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers

            v, u = src_lat - dest_lat, src_lon - dest_lon
            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direction = wind_direction(u, v).m

            direction_arr.append(direction)
            dist_kilometer.append(dist_km)

        direction_arr = np.stack(direction_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direction_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self) -> None:
        """Update edges based on altitude data."""
        edge_idx = []
        edge_attr = []

        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat = int(self.nodes[src]["lat"] * self.factor)
            src_lon = int(self.nodes[src]["lon"] * self.factor)
            dest_lat = int(self.nodes[dest]["lat"] * self.factor)
            dest_lon = int(self.nodes[dest]["lon"] * self.factor)

            points = np.asarray(
                list(bresenham(src_lat, src_lon, dest_lat, dest_lon))
            ).transpose((1, 0))
            altitude_points = self._get_alt(points[0], points[1])

            for j in range(len(points[0])):
                point_lat = float(points[0, j]) / self.factor
                point_lon = float(points[1, j]) / self.factor
                self.count.add((point_lat, point_lon))
                self.alt_dict[(point_lat, point_lon)] = float(altitude_points[j])

            # Get source and destination altitudes
            altitude_src = self._get_and_store_altitude(src_lat, src_lon)
            altitude_dest = self._get_and_store_altitude(dest_lat, dest_lon)

            # Filter edges based on altitude threshold
            if (
                np.sum(altitude_points - altitude_src > self.altitude_threshold) < 3
                and np.sum(altitude_points - altitude_dest > self.altitude_threshold)
                < 3
            ):
                edge_idx.append(self.edge_index[:, i])
                edge_attr.append(self.edge_attr[i])

        # Update edge data - handle empty edge list case
        if edge_idx:  # Check if list is not empty
            self.edge_index = np.stack(edge_idx, axis=1)
            self.edge_attr = np.stack(edge_attr, axis=0)
        else:
            # Handle the case where no edges pass the filter
            self.edge_index = np.empty((2, 0), dtype=int)
            self.edge_attr = np.empty((0, self.edge_attr.shape[1]), dtype=float)


if __name__ == "__main__":
    graph = Graph()
