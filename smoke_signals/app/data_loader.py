from app.gcs_utils import get_npy_from_gcs
from pathlib import Path
import pandas as pd
import numpy as np
import os

# --- Configuration ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
PROC_DATA_DIR = DATA_DIR / "processed"
LOCATIONS_FILE = PROC_DATA_DIR / "locations-names.csv"

PREDICT_NET_FILE = get_npy_from_gcs("smoke-signal-bucket", "pm25gnn/predict.npy")
PREDICT_AMBIENT_FILE = get_npy_from_gcs("smoke-signal-bucket", "pm25gnn-ambient/predict.npy")
LABEL_FILE = get_npy_from_gcs("smoke-signal-bucket", "pm25gnn-ambient/label.npy")
TIME_FILE = get_npy_from_gcs("smoke-signal-bucket", "pm25gnn/time.npy")

PRED_LEN = 48  # Prediction length used in training

def load_and_preprocess_data():
    """
    Load and preprocess data for time series analysis.
    Returns:
        df (pd.DataFrame): Preprocessed data as a DataFrame.
        locations (list): List of location options for dropdown.
        location_map (dict): Mapping of location indices to labels.
    """
    print("Loading and preprocessing data...")

    # Load location data
    locations = []
    location_map = {}
    if os.path.exists(LOCATIONS_FILE):
        locations_df = pd.read_csv(LOCATIONS_FILE, index_col=0)
        loc_id_col = 'location_id'
        city_name_col = 'city_name'

        for index, row in locations_df.iterrows():
            loc_index = index
            loc_name = row[city_name_col]
            label = f"{loc_name}"
            locations.append({'label': label, 'value': loc_index})
            location_map[loc_index] = label

    # Preprocess .npy data
    min_len = min(PREDICT_NET_FILE.shape[0], PREDICT_AMBIENT_FILE.shape[0], LABEL_FILE.shape[0], TIME_FILE.shape[0])
    predict_net_data_trunc = PREDICT_NET_FILE[:min_len]
    predict_ambient_data_trunc = PREDICT_AMBIENT_FILE[:min_len]
    label_data_trunc = LABEL_FILE[:min_len]
    time_data_trunc = TIME_FILE[:min_len]

    predict_net_actual = predict_net_data_trunc[:, -PRED_LEN:, :, 0].reshape(-1, predict_net_data_trunc.shape[-2])
    predict_ambient_actual = predict_ambient_data_trunc[:, -PRED_LEN:, :, 0].reshape(-1, predict_ambient_data_trunc.shape[-2])
    label_actual = label_data_trunc[:, -PRED_LEN:, :, 0].reshape(-1, label_data_trunc.shape[-2])
    time_data_processed = np.repeat(time_data_trunc, PRED_LEN)

    time_data_dt = pd.to_datetime(time_data_processed, unit='s')

    # Combine into a DataFrame
    df_list = []
    for data_loc_idx in range(predict_net_actual.shape[1]):
        loc_df = pd.DataFrame({
            'timestamp': time_data_dt,
            'location_id': data_loc_idx,
            'observed': label_actual[:, data_loc_idx],
            'predicted_net': predict_net_actual[:, data_loc_idx],
            'predicted_ambient': predict_ambient_actual[:, data_loc_idx]
        })
        df_list.append(loc_df)

    df = pd.concat(df_list, ignore_index=True)
    df['fire_specific'] = df['predicted_net'] - df['predicted_ambient']
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("Data preprocessing complete.")
    return df, locations, location_map