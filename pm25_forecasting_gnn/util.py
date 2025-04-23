"""
Utility module for configuration management.

This module provides functionality for loading and accessing project configuration,
file paths, and environment setup for the air quality prediction system.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

# Third-Party Libraries
import yaml
import arrow


# Type definitions for config.yaml structure
class DatasetTimeConfig(TypedDict):
    """Type definition for dataset time configurations."""
    train_start: List[Union[List[int], str]]
    train_end: List[Union[List[int], str]]
    val_start: List[Union[List[int], str]]
    val_end: List[Union[List[int], str]]
    test_start: List[Union[List[int], str]]
    test_end: List[Union[List[int], str]]


class DatasetConfig(TypedDict):
    """Type definition for dataset configurations."""
    data_start: List[Union[List[int], str]]
    data_end: List[Union[List[int], str]]


class ExperimentsConfig(TypedDict):
    """Type definition for experiment configurations."""
    meteorological_use: List[str]
    save_npy: bool
    dataset_num: int
    model: str


class TrainConfig(TypedDict):
    """Type definition for training configurations."""
    batch_size: int
    epochs: int
    exp_repeat: int
    hist_len: int
    pred_len: int
    weight_decay: float
    early_stop: int
    lr: float


class DataConfig(TypedDict):
    """Type definition for data configurations."""
    meteorological_var: List[str]


class ServerConfig(TypedDict):
    """Type definition for the complete server configuration."""
    experiments: ExperimentsConfig
    train: TrainConfig
    filepath: Dict[str, Dict[str, str]]
    data: DataConfig
    dataset: Dict[str, Any]  # Changed to accommodate mixed key types


@dataclass(frozen=True)
class ProjectPaths:
    """Dataclass to store project paths."""
    project_dir: Path
    config_file: Path
    data_dir: Path
    air_data_file: Path
    results_dir: Path


def get_project_dir() -> Path:
    """
    Return the absolute path to the project directory.

    Returns:
        Path object representing the project root directory
    """
    return Path(__file__).parent.resolve()


def load_config(config_file: Optional[Path] = None) -> ServerConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the configuration file. If None, uses the default config.yaml.

    Returns:
        Configuration dictionary with properly typed structure.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        KeyError: If required configuration keys are missing.
    """
    project_dir = get_project_dir()

    if config_file is None:
        config_file = project_dir / "config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Validate essential config sections
        required_keys = ["experiments", "train", "filepath", "data", "dataset"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise KeyError(f"Missing required configuration sections: {', '.join(missing_keys)}")

        return cast(ServerConfig, config)
    except yaml.YAMLError as error:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {error}")


def get_file_dir(
    config: ServerConfig,
    server_key: str = None,
) -> Dict[str, str]:
    """
    Generate file directory paths based on configuration for a specific server.

    Args:
        config: The configuration dictionary.
        server_key: The server key to look up in the configuration.

    Returns:
        Dictionary of file directory paths.

    Raises:
        KeyError: If the specified server key is not found in the configuration.
    """
    try:
        if server_key is None:
            server_key = "GPU-Server"

        # Handle both hyphenated and underscore versions of server key
        normalized_key = server_key.replace("-", "_")
        if normalized_key not in config["filepath"]:
            normalized_key = server_key.replace("_", "-")

        if normalized_key not in config["filepath"]:
            raise KeyError(f"Server key '{server_key}' not found in configuration")

        file_paths = {}

        # Add the paths directly from config
        for path_name, path_value in config["filepath"][normalized_key].items():
            file_paths[path_name] = path_value

        # Ensure we have default paths if they're not in the config
        if "air_data_file" not in file_paths:
            base_path = Path(file_paths.get("data_dir", "/data/pm25gnn/data"))
            file_paths["air_data_file"] = str(base_path / "dataset_fire_wind_aligned.npy")

        if "results_dir" not in file_paths:
            base_path = Path(file_paths.get("base_dir", "/data/pm25gnn"))
            file_paths["results_dir"] = str(base_path / "results")

        if "model_save_dir" not in file_paths:
            base_path = Path(file_paths.get("results_dir", "/data/pm25gnn/results"))
            file_paths["model_save_dir"] = str(base_path / "models")

        if "logs_dir" not in file_paths:
            base_path = Path(file_paths.get("results_dir", "/data/pm25gnn/results"))
            file_paths["logs_dir"] = str(base_path / "logs")

        return file_paths
    except KeyError as error:
        raise KeyError(f"Missing configuration key: {error}")


def get_dataset_config(config: ServerConfig, dataset_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Get dataset configuration for a specific dataset number.

    Args:
        config: The configuration dictionary.
        dataset_num: Dataset number to retrieve. If None, uses the number from experiments.

    Returns:
        Dataset configuration dictionary.

    Raises:
        KeyError: If the specified dataset number is not found.
    """
    if dataset_num is None:
        dataset_num = config["experiments"].get("dataset_num", 1)

    try:
        # Get dataset-specific config - convert to string to ensure type safety
        dataset_specific = config["dataset"][str(dataset_num)]
        if dataset_specific is None:
            # Try with integer key if string key fails
            dataset_specific = config["dataset"][str(dataset_num)]

        # Merge with general dataset config
        dataset_config = {
            key: value for key, value in config["dataset"].items()
            if not isinstance(key, int) and key != str(dataset_num)
        }

        # Update with dataset-specific config
        if isinstance(dataset_specific, dict):
            dataset_config.update(dataset_specific)

        return dataset_config
    except (KeyError, TypeError) as error:
        raise KeyError(f"Dataset configuration not found for dataset {dataset_num}: {error}")


def get_time(time_yaml: List[Union[List[int], str]]) -> arrow.Arrow:
    """
    Convert YAML time configuration to Arrow object.

    Args:
        time_yaml: List containing [datetime_components, timezone]

    Returns:
        Arrow time object

    Raises:
        ValueError: If the time format is invalid.
    """
    try:
        dt_components = time_yaml[0]
        timezone = time_yaml[1]

        if not isinstance(dt_components, list) or not all(isinstance(i, int) for i in dt_components):
            raise ValueError(f"Invalid datetime components format: {dt_components}")

        # Pad with zeros if needed (e.g., if only year, month, day are provided)
        while len(dt_components) < 5:
            dt_components.append(0)

        dt = datetime(*dt_components)
        return arrow.get(dt, timezone)
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Invalid time format in configuration: {error}")


def initialize_environment() -> tuple[ServerConfig, Dict[str, str]]:
    """
    Set up the environment by loading configuration and adding project directory to path.

    Returns:
        Tuple containing:
            - The loaded configuration
            - Dictionary of file directories

    Example:
        \>>> config, file_dir = initialize_environment()
        \>>> print(f\"Data directory: {file_dir['air_data_file']}")
    """
    project_dir = get_project_dir()

    # Add project directory to Python path if not already present
    if str(project_dir) not in sys.path:
        sys.path.append(str(project_dir))

    # Load the default configuration
    config = load_config()

    # Get file directories
    file_dir = get_file_dir(config)

    return config, file_dir


def get_project_paths(config: ServerConfig, server_key: str = "GPU-Server") -> ProjectPaths:
    """
    Get standardized project paths based on configuration.

    Args:
        config: The configuration dictionary.
        server_key: The server key to look up in the configuration.

    Returns:
        ProjectPaths object containing common project paths
    """
    project_dir = get_project_dir()
    config_path = project_dir / "config.yaml"

    file_paths = get_file_dir(config, server_key)

    # Handle both string and Path inputs
    air_data_file = Path(file_paths["air_data_file"])
    results_dir = Path(file_paths["results_dir"])
    data_dir = air_data_file.parent.resolve()

    return ProjectPaths(
        project_dir=project_dir,
        config_file=config_path,
        data_dir=data_dir,
        air_data_file=air_data_file,
        results_dir=results_dir,
    )


def get_model_config(config: ServerConfig) -> Dict[str, Any]:
    """
    Extract model configuration parameters from config.

    Args:
        config: The configuration dictionary.

    Returns:
        Dictionary with model configuration parameters.
    """
    model_config = {
        "model_name": config["experiments"].get("model", "PM25_GNN"),
        "hist_len": config["train"].get("hist_len", 240),
        "pred_len": config["train"].get("pred_len", 48),
        "batch_size": config["train"].get("batch_size", 16),
        "learning_rate": config["train"].get("lr", 0.0005),
        "weight_decay": config["train"].get("weight_decay", 0.0005),
        "epochs": config["train"].get("epochs", 50),
        "early_stop": config["train"].get("early_stop", 10),
        "meteorological_vars": config["experiments"].get("meteorological_use", []),
    }

    return model_config


def main() -> None:
    """Run utility module as a script for quick testing."""
    config, file_dir = initialize_environment()
    paths = get_project_paths(config)
    model_config = get_model_config(config)

    try:
        dataset_config = get_dataset_config(config)
        print("\nDataset configuration:")
        for key, value in dataset_config.items():
            if key in ["data_start", "data_end"]:
                print(f"  {key}: {value}")
    except KeyError as e:
        print(f"Error getting dataset config: {e}")

    print(f"Project directory: {paths.project_dir}")
    print(f"Configuration file: {paths.config_file}")
    print(f"Data directory: {paths.data_dir}")
    print(f"KnowAir file path: {paths.air_data_file}")
    print(f"Results directory: {paths.results_dir}")

    print("\nModel configuration:")
    for key, value in model_config.items():
        if key != "meteorological_vars":  # Skip printing the long list of variables
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()