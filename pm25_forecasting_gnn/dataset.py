import os
import sys
from datetime import datetime

import arrow
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

from util import config, file_dir

class HazeData(data.Dataset):
    def __init__(self, graph,
                       hist_len=1,
                       pred_len=24,
                       dataset_num=1,
                       flag='Train',
                       ):
        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])
        self.knowair_fp = file_dir['knowair_fp']
        self.graph = graph
        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()
        # Use .astype() for clarity, although np.float32() also works
        self.feature = self.feature.astype(np.float32)
        self.pm25 = self.pm25.astype(np.float32)
        self.frp500 = self.frp500.astype(np.float32) # uncomment for 'train_ambient.py'
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()
        self._dictionary()
        print(f"[{flag} Dataset] Initialized. Shapes - PM2.5: {self.pm25.shape}, Feature: {self.feature.shape}, Time: {self.time_arr.shape}")


    def _dictionary(self):
        self.time_index = {}
        for i in range(self.time_arr_full.shape[0]):
            # Ensure the key is a basic number type (it should be from timestamp)
            key = self.time_arr_full[i].item() if isinstance(self.time_arr_full[i], np.generic) else self.time_arr_full[i]
            self.time_index[key] = i


    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std
        self.feature_full = (self.feature_full - self.feature_mean) / self.feature_std
        self.pm25_full = (self.pm25_full - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):
        # Ensure these are copies, not views, if modifying later (though looks ok here)
        self.pm25_full = np.copy(self.pm25)
        self.feature_full = np.copy(self.feature)
        self.time_arr_full = np.copy(self.time_arr)

        # Slicing creates views, which is usually fine for reading
        self.pm25 = self.pm25[seq_len:]
        self.feature = self.feature[seq_len:]
        self.time_arr = self.time_arr[seq_len:]


    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        # Ensure indices 7, 8 are correct for wind components after _process_feature
        # Check self.feature.shape[-1] after _process_feature if unsure
        wind_u_idx = config['experiments']['metero_use'].index('u_component_of_wind+950')
        wind_v_idx = config['experiments']['metero_use'].index('v_component_of_wind+950')
        self.wind_mean = self.feature_mean[[wind_u_idx, wind_v_idx]]
        self.wind_std = self.feature_std[[wind_u_idx, wind_v_idx]]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()
        print(f"Calculated Mean/Std - PM2.5 Mean: {self.pm25_mean}, PM2.5 Std: {self.pm25_std}")


    def _process_feature(self):
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:,:,metero_idx]

        # Find indices dynamically based on metero_use config
        try:
            u_idx = metero_use.index('u_component_of_wind+950')
            v_idx = metero_use.index('v_component_of_wind+950')
        except ValueError:
            raise ValueError("u_component_of_wind+950 or v_component_of_wind+950 not found in metero_use config")

        u = self.feature[:, :, u_idx] * units.meter / units.second
        v = self.feature[:, :, v_idx] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v)._magnitude
        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday()) # Monday=1 to Sunday=7
        h_arr = np.array(h_arr) # Use np.array instead of np.stack for 1D
        w_arr = np.array(w_arr)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)
        # Add julian_date and time_of_day if they are in metero_use, otherwise add calculated ones
        # Assuming julian_date and time_of_day might already be handled if present in metero_use
        # If they are NOT in metero_use, we add hour and weekday here.
        # Consider adding them as sin/cos transforms for cyclical nature.
        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)


    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.frp500 = self.frp500[start_idx: end_idx+1, :] # uncomment for 'train_ambient.py'
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        self.time_arrow = []
        time_stamps = []
        # determines time granularity (in this case, 1 hour)
        # Correct interval generation: end date is exclusive, shift by +1 interval unit
        for time_arrow_tuple in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+1), 1):
             # interval returns a tuple (start_time, end_time) for each interval
             current_time = time_arrow_tuple[0]
             self.time_arrow.append(current_time)
             time_stamps.append(current_time.timestamp()) # Use .timestamp() directly
        self.time_arr = np.array(time_stamps) # Use np.array for 1D


    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)
        # Assuming last column is PM2.5, 12th (index 11 or 12?) is FRP
        # Verify indices based on actual data structure
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]
        # Ensure correct index for FRP - Python uses 0-based indexing
        # If it's the 12th column, the index is 11. If it's column *named* 12, check data.
        frp_col_index = 11 # Assuming 12th column means index 11
        self.frp500 = self.knowair[:,:,frp_col_index] # Slicing with integer keeps dims, need [:, :, frp_col_index]

    def _get_idx(self, t):
        t0 = self.data_start
        # determines time granularity (1 hour = 3600 seconds)
        return int((t.timestamp() - t0.timestamp()) / 3600)


    def _get_time(self, time_yaml):
        # time_yaml format: [[YYYY, M, D], 'Timezone']
        # Example: [[2017, 1, 2], 'UTC']
        dt = datetime(time_yaml[0][0], time_yaml[0][1], time_yaml[0][2])
        arrow_time = arrow.get(dt, time_yaml[1])
        return arrow_time

    def __len__(self):
        # Length should be the number of possible start times for sequences
        # Original length was based on self.pm25 after slicing in _add_time_dim
        return len(self.time_arr) # time_arr is sliced in _add_time_dim

    def __getitem__(self, index):
        # # --- DEBUGGING ---
        # print(f"--- HazeData.__getitem__(index={index}) ---")
        try:
            pm_item = self.pm25[index]
            feat_item = self.feature[index]
            time_item = self.time_arr[index] # time_arr is 1D, index directly
            #
            # print(f"  Type pm_item: {type(pm_item)}, Shape: {getattr(pm_item, 'shape', 'N/A')}")
            # print(f"  Type feat_item: {type(feat_item)}, Shape: {getattr(feat_item, 'shape', 'N/A')}")
            # print(f"  Type time_item: {type(time_item)}, Value: {time_item}") # time_item should be a number

            # Check if any is a method
            if callable(pm_item) or callable(feat_item) or callable(time_item):
                 print(f"  !!! ERROR: An item is callable (method?) !!!")
                 # You might want to raise an error here or return dummy data
                 # raise TypeError("Item is callable!")

            return pm_item, feat_item, time_item

        except Exception as e:
            print(f"  !!! EXCEPTION in __getitem__ for index {index}: {e}")
            # Depending on the error, you might re-raise or return dummy data
            # For debugging, let's return dummy data to see if collation proceeds
            # return np.zeros(1), np.zeros(1), 0.0 # Example dummy data
            raise e # Re-raise to see the original error location if it's an indexing issue

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')
