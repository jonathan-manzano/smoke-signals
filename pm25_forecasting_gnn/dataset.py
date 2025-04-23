"""
HazeData dataset module for air quality prediction.

This module provides dataset functionality for training and evaluating
models that predict PM2.5 concentrations using meteorological data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils.data import Dataset

# Import project utilities with proper path handling
project_dir = Path(__file__).parent.parent.resolve()
import sys
sys.path.append(str(project_dir))
from util import initialize_environment, get_dataset_config, get_time


class HazeData(Dataset):
    """
    Dataset class for air quality prediction with meteorological features.

    This class handles loading, processing, and normalizing meteorological
    and air quality data for training, validation, and testing.

    Attributes:
        feature: Normalized meteorological features
        pm25: Normalized PM2.5 concentration values
        time_arr: Array of timestamps
        feature_mean: Mean values of features (for normalization)
        feature_std: Standard deviation of features (for normalization)
        pm25_mean: Mean value of PM2.5 concentrations
        pm25_std: Standard deviation of PM2.5 concentrations
    """

    def __init__(
        self,
        graph: Any,
        hist_len: int = 1,
        pred_len: int = 24,
        dataset_num: int = 1,
        flag: Literal["Train", "Val", "Test"] = "Train",
    ) -> None:
        """
        Initialize the HazeData dataset.

        Args:
            graph: Graph object representing the spatial relationships
            hist_len: Number of historical time steps to use
            pred_len: Number of future time steps to predict
            dataset_num: Dataset configuration number
            flag: Dataset split type ("Train", "Val", or "Test")

        Raises:
            ValueError: If an invalid flag is provided
        """
        # Load configuration and file paths
        self.config, self.file_dir = initialize_environment()
        self.dataset_config = get_dataset_config(self.config, dataset_num)

        # Determine time range based on dataset flag
        if flag == "Train":
            start_time_str = "train_start"
            end_time_str = "train_end"
        elif flag == "Val":
            start_time_str = "val_start"
            end_time_str = "val_end"
        elif flag == "Test":
            start_time_str = "test_start"
            end_time_str = "test_end"
        else:
            raise ValueError(f"Invalid flag: {flag}. Must be 'Train', 'Val', or 'Test'")

        # Get time ranges from configuration
        # Use the raw time data from the config for the get_time function
        self.start_time = get_time(self.dataset_config[start_time_str])
        self.end_time = get_time(self.dataset_config[end_time_str])
        self.data_start = get_time(self.dataset_config["data_start"])
        self.data_end = get_time(self.dataset_config["data_end"])

        # Set data source and graph
        self.knowair_fp = self.file_dir["knowair_fp"]
        self.graph = graph

        # Process data
        self._load_data()
        self._generate_time_arrays()
        self._process_time_range()
        self._process_features()

        # Convert to float32 for efficiency
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)

        # Calculate statistics and normalize
        self._calculate_statistics()

        # Prepare sequences
        seq_len = hist_len + pred_len
        self._prepare_sequences(seq_len)
        self._normalize_data()
        self._create_time_index()

    def _load_data(self) -> None:
        """Load data from NumPy files."""
        try:
            self.knowair = np.load(self.knowair_fp)
            self.feature = self.knowair[:, :, :-1]
            self.pm25 = self.knowair[:, :, -1:]
            # Uncomment for 'train_ambient.py' if needed:
            # self.frp500 = self.knowair[:, :, 12]
        except (FileNotFoundError, IOError) as e:
            raise IOError(f"Failed to load data from {self.knowair_fp}: {e}") from e

    def _generate_time_arrays(self) -> None:
        """Generate arrays of time objects and timestamps."""
        self.time_arrow: List[arrow.Arrow] = []
        self.time_arr: List[float] = []

        # Create hourly timestamps from start to end (inclusive)
        for time_arrow in arrow.Arrow.interval(
            "hour",
            self.data_start,
            self.data_end.shift(hours=+1),
            1
        ):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)

        self.time_arr = np.array(self.time_arr)

    def _process_time_range(self) -> None:
        """Extract data for the specified time range."""
        start_idx = self._get_index(self.start_time)
        end_idx = self._get_index(self.end_time)

        self.pm25 = self.pm25[start_idx:end_idx+1, :]
        self.feature = self.feature[start_idx:end_idx+1, :]
        # Uncomment for 'train_ambient.py' if needed:
        # self.frp500 = self.frp500[start_idx:end_idx+1, :]
        self.time_arr = self.time_arr[start_idx:end_idx+1]
        self.time_arrow = self.time_arrow[start_idx:end_idx+1]

    def _process_features(self) -> None:
        """Process and augment meteorological features."""
        # Select required meteorological variables
        metero_var = self.config["data"]["metero_var"]
        metero_use = self.config["experiments"]["metero_use"]
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:, :, metero_idx]

        # Extract wind components (indices should be defined as constants)
        WIND_U_IDX = 7  # u_component_of_wind+950
        WIND_V_IDX = 8  # v_component_of_wind+950

        # Calculate wind speed and direction using MetPy
        u = self.feature[:, :, WIND_U_IDX] * units.meter / units.second
        v = self.feature[:, :, WIND_V_IDX] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude  # Convert to km/h
        direction = mpcalc.wind_direction(u, v)._magnitude

        # Extract hour and weekday features
        hour_arr = np.array([t.hour for t in self.time_arrow])
        weekday_arr = np.array([t.isoweekday() for t in self.time_arrow])

        # Expand dimensions for broadcasting
        hour_arr = np.repeat(hour_arr[:, np.newaxis], self.graph.node_num, axis=1)
        weekday_arr = np.repeat(weekday_arr[:, np.newaxis], self.graph.node_num, axis=1)

        # Concatenate all features
        self.feature = np.concatenate([
            self.feature,
            hour_arr[:, :, np.newaxis],
            weekday_arr[:, :, np.newaxis],
            speed[:, :, np.newaxis],
            direction[:, :, np.newaxis]
        ], axis=-1)

    def _calculate_statistics(self) -> None:
        """Calculate mean and standard deviation for normalization."""
        self.feature_mean = self.feature.mean(axis=(0, 1))
        self.feature_std = self.feature.std(axis=(0, 1))

        # Store wind statistics separately if needed
        WIND_U_IDX = 7
        WIND_V_IDX = 8
        self.wind_mean = self.feature_mean[WIND_U_IDX:WIND_V_IDX+1]
        self.wind_std = self.feature_std[WIND_U_IDX:WIND_V_IDX+1]

        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _prepare_sequences(self, seq_len: int) -> None:
        """Prepare data sequences with appropriate offsets.

        Args:
            seq_len: Length of sequence (history + prediction)
        """
        # Create copies for full sequences
        self.pm25_full = self.pm25.copy()
        self.feature_full = self.feature.copy()
        self.time_arr_full = self.time_arr.copy()

        # Adjust arrays to account for sequence length
        self.pm25 = self.pm25[seq_len:]
        self.feature = self.feature[seq_len:]
        self.time_arr = self.time_arr[seq_len:]

    def _normalize_data(self) -> None:
        """Normalize features and target values."""
        # Apply z-score normalization
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std
        self.feature_full = (self.feature_full - self.feature_mean) / self.feature_std
        self.pm25_full = (self.pm25_full - self.pm25_mean) / self.pm25_std

    def _create_time_index(self) -> None:
        """Create dictionary mapping timestamps to indices."""
        self.time_index = {timestamp: i for i, timestamp in enumerate(self.time_arr_full)}

    def _get_index(self, t: arrow.Arrow) -> int:
        """Convert timestamp to index based on data start time.

        Args:
            t: Arrow time object

        Returns:
            Index in the time array
        """
        t0 = self.data_start
        # Calculate hours difference (hourly granularity)
        return int((t.timestamp - t0.timestamp) / (60 * 60))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of data points
        """
        return len(self.pm25)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get item at the specified index.

        Args:
            index: Index to retrieve

        Returns:
            Tuple of (pm25, feature, timestamp)
        """
        return self.pm25[index], self.feature[index], self.time_arr[index]


if __name__ == "__main__":
    # Example usage
    from graph import Graph

    graph = Graph()
    train_data = HazeData(graph, flag="Train")
    val_data = HazeData(graph, flag="Val")
    test_data = HazeData(graph, flag="Test")

    print(f"Train data shape: {train_data.feature.shape}")
    print(f"Validation data shape: {val_data.feature.shape}")
    print(f"Test data shape: {test_data.feature.shape}")