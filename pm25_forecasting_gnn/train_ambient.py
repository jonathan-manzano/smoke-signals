# import os
# import sys
#
# proj_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(proj_dir)
# from util import config, file_dir
#
# from graph import Graph
# import pdb
# from dataset import HazeData
#
# from model.MLP import MLP
# from model.LSTM import LSTM
# from model.GRU import GRU
# from model.GC_LSTM import GC_LSTM
# from model.nodesFC_GRU import nodesFC_GRU
# from model.PM25_GNN import PM25_GNN
# from model.PM25_GNN_nosub import PM25_GNN_nosub
#
# import pdb
# import arrow
# import torch
# from torch import nn
# from tqdm import tqdm
# import numpy as np
# import pickle
# import glob
# import shutil
#
# torch.set_num_threads(1)
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
# print(device)
#
# graph = Graph()
# city_num = graph.node_num
#
# batch_size = config['train']['batch_size']
# epochs = config['train']['epochs']
# hist_len = config['train']['hist_len']
# pred_len = config['train']['pred_len']
# seq_len = hist_len + pred_len
# weight_decay = config['train']['weight_decay']
# early_stop = config['train']['early_stop']
# lr = config['train']['lr']
# results_dir = file_dir['results_dir']
# dataset_num = config['experiments']['dataset_num']
# exp_model = config['experiments']['model']
# exp_repeat = config['train']['exp_repeat']
# save_npy = config['experiments']['save_npy']
# criterion = nn.MSELoss()
#
# train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
# val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
# test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
#
# in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
# wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
# pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std
# print("mean ", pm25_mean, "std ", pm25_std)
#
#
# def get_metric(predict_epoch, label_epoch):
#     haze_threshold = 75
#     predict_haze = predict_epoch >= haze_threshold
#     predict_clear = predict_epoch < haze_threshold
#     label_haze = label_epoch >= haze_threshold
#     label_clear = label_epoch < haze_threshold
#     hit = np.sum(np.logical_and(predict_haze, label_haze))
#     miss = np.sum(np.logical_and(label_haze, predict_clear))
#     falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
#     csi = hit / (hit + falsealarm + miss)
#     pod = hit / (hit + miss)
#     far = falsealarm / (hit + falsealarm)
#     predict = predict_epoch[:, :, :, 0].transpose((0, 2, 1))
#     label = label_epoch[:, :, :, 0].transpose((0, 2, 1))
#     predict = predict.reshape((-1, predict.shape[-1]))
#     label = label.reshape((-1, label.shape[-1]))
#     mae = np.mean(np.mean(np.abs(predict - label), axis=1))
#     rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
#     return rmse, mae, csi, pod, far
#
#
# def get_exp_info():
#     exp_info = '============== Train Info ==============\n' + \
#                'Dataset number: %s\n' % dataset_num + \
#                'Model: %s\n' % exp_model + \
#                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
#                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
#                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
#                'City number: %s\n' % city_num + \
#                'Use metero: %s\n' % config['experiments']['metero_use'] + \
#                'batch_size: %s\n' % batch_size + \
#                'epochs: %s\n' % epochs + \
#                'hist_len: %s\n' % hist_len + \
#                'pred_len: %s\n' % pred_len + \
#                'weight_decay: %s\n' % weight_decay + \
#                'early_stop: %s\n' % early_stop + \
#                'lr: %s\n' % lr + \
#                '========================================\n'
#     return exp_info
#
#
# def get_model():
#     if exp_model == 'MLP':
#         return MLP(hist_len, pred_len, in_dim)
#     elif exp_model == 'LSTM':
#         return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
#     elif exp_model == 'GRU':
#         return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
#     elif exp_model == 'nodesFC_GRU':
#         return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
#     elif exp_model == 'GC_LSTM':
#         return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
#     elif exp_model == 'PM25_GNN':
#         return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr,
#                         wind_mean, wind_std)
#     elif exp_model == 'PM25_GNN_nosub':
#         return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index,
#                               graph.edge_attr, wind_mean, wind_std)
#     else:
#         raise Exception('Wrong model name!')
#
#
# def prepare(pm25, feature, time_arr, flag):
#     seq_len_fire = seq_len
#     pm25_hist = np.full((pm25.shape[0], hist_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float64)
#     pm25_label = np.full((pm25.shape[0], pred_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float64)
#     feature = np.full((feature.shape[0], hist_len + pred_len, feature.shape[1], feature.shape[2]), -1.0, dtype=np.float64)
#     pm = np.full((pm25.shape[0], hist_len + pred_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float64)
#     frp500 = np.full((pm25.shape[0], seq_len_fire, pm25.shape[1]), -1.0, dtype=np.float64)
#
#     for i in range(np.asarray(time_arr).shape[0]):
#         if flag == "Train":
#             end = train_data.time_index[np.asarray(time_arr)[i]]
#             pm25_hist[i, :, :, :] = train_data.pm25_full[end - seq_len + 1:end - pred_len + 1, :, :]
#             pm25_label[i, :, :, :] = train_data.pm25_full[end - pred_len + 1:end + 1, :, :]
#             feature[i, :, :, :] = train_data.feature_full[end - seq_len + 1:end + 1, :, :]
#             pm[i, :, :, :] = train_data.pm25_full[end - seq_len + 1:end + 1, :, :]
#             frp500[i, :, :] = train_data.frp500[end - seq_len_fire + 1:end + 1, :]
#         elif flag == "Val":
#             end = val_data.time_index[np.asarray(time_arr)[i]]
#             pm25_hist[i, :, :, :] = val_data.pm25_full[end - seq_len + 1:end - pred_len + 1, :, :]
#             pm25_label[i, :, :, :] = val_data.pm25_full[end - pred_len + 1:end + 1, :, :]
#             feature[i, :, :, :] = val_data.feature_full[end - seq_len + 1:end + 1, :, :]
#             pm[i, :, :, :] = val_data.pm25_full[end - seq_len + 1:end + 1, :, :]
#         else:
#             end = test_data.time_index[np.asarray(time_arr)[i]]
#             pm25_hist[i, :, :, :] = test_data.pm25_full[end - seq_len + 1:end - pred_len + 1, :, :]
#             pm25_label[i, :, :, :] = test_data.pm25_full[end - pred_len + 1:end + 1, :, :]
#             feature[i, :, :, :] = test_data.feature_full[end - seq_len + 1:end + 1, :, :]
#             pm[i, :, :, :] = test_data.pm25_full[end - seq_len + 1:end + 1, :, :]
#
#         if flag == "Train":
#             return torch.tensor(pm25_hist, dtype=torch.float), torch.tensor(pm25_label,
#                                                                             dtype=torch.float), torch.tensor(feature,
#                                                                                                              dtype=torch.float), pm.astype(
#                 'float'), frp500.astype('float')
#     return torch.tensor(pm25_hist, dtype=torch.float), torch.tensor(pm25_label, dtype=torch.float), torch.tensor(
#         feature, dtype=torch.float), pm.astype('float')
#
#
# def train(train_loader, model, optimizer):
#     model.train()
#     train_loss = 0
#     count = 0
#     pm25_std_train, pm25_mean_train = train_data.pm25_std, train_data.pm25_mean
#     for batch_idx, data in enumerate(tqdm(train_loader)):
#         pm25, feature, time_arr = data
#         pm25_hist, pm25_label, feature, pm25, fire = prepare(pm25, feature, time_arr, "Train")
#         feature = feature.to(device)
#         pm25_hist = pm25_hist.to(device)
#         pm25_label = pm25_label.to(device)
#
#         thresh = 0.15
#         exceed = (fire > thresh).any()
#
#         if not exceed:
#             pm25_pred = model(pm25_hist, feature)
#             loss = criterion(pm25_pred, pm25_label)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             count += 1
#
#     print(count)
#     train_loss /= batch_idx + 1
#     return train_loss
#
#
# def val(val_loader, model):
#     model.eval()
#     val_loss = 0
#     for batch_idx, data in enumerate(tqdm(val_loader)):
#         pm25, feature, time_arr = data
#         pm25_hist, pm25_label, feature, pm25 = prepare(pm25, feature, time_arr, "Val")
#         feature = feature.to(device)
#         pm25_hist = pm25_hist.to(device)
#         pm25_pred = model(pm25_hist, feature)
#         pm25_label = pm25_label.to(device)
#         loss = criterion(pm25_pred, pm25_label)
#         val_loss += loss.item()
#     val_loss /= batch_idx + 1
#     return val_loss
#
#
# def test(test_loader, model):
#     model.eval()
#     predict_list = []
#     label_list = []
#     time_list = []
#     test_loss = 0
#     for batch_idx, data in enumerate(tqdm(test_loader)):
#         pm25, feature, time_arr = data
#         pm25_hist, pm25_label, feature, pm25 = prepare(pm25, feature, time_arr, "Test")
#         feature = feature.to(device)
#         pm25_hist = pm25_hist.to(device)
#         pm25_label = pm25_label.to(device)
#         pm25_pred = model(pm25_hist, feature)
#
#         loss = criterion(pm25_pred, pm25_label)
#         test_loss += loss.item()
#         pdb.set_trace()
#         pm25_pred = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()],
#                                    axis=1) * pm25_std + pm25_mean
#         pm25_label = pm25 * pm25_std + pm25_mean
#         predict_list.append(pm25_pred)
#         label_list.append(pm25_label)
#         time_list.append(time_arr.cpu().detach().numpy())
#
#     test_loss /= batch_idx + 1
#     predict_epoch = np.concatenate(predict_list, axis=0)
#     label_epoch = np.concatenate(label_list, axis=0)
#     time_epoch = np.concatenate(time_list, axis=0)
#     predict_epoch[predict_epoch < 0] = 0
#
#     return test_loss, predict_epoch, label_epoch, time_epoch
#
#
# def get_mean_std(data_list):
#     data = np.asarray(data_list)
#     return data.mean(), data.std()
#
#
# def main():
#     exp_info = get_exp_info()
#     print(exp_info)
#
#     exp_time = arrow.now().format('YYYYMMDDHHmmss')
#
#     train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []
#
#     for exp_idx in range(exp_repeat):
#         print('\nNo.%2d experiment ~~~' % exp_idx)
#         train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
#         test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
#
#         model = get_model()
#         model = model.to(device)
#         model_name = type(model).__name__
#
#         print(str(model))
#
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#         exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name,
#                                      str(exp_time), '%02d' % exp_idx)
#         if not os.path.exists(exp_model_dir):
#             os.makedirs(exp_model_dir)
#         model_fp = os.path.join(exp_model_dir, 'model.pth')
#
#         val_loss_min = 100000
#         best_epoch = 0
#
#         train_loss_, val_loss_ = 0, 0
#
#         for epoch in range(epochs):
#             print('\nTrain epoch %s:' % (epoch))
#
#             train_loss = train(train_loader, model, optimizer)
#             val_loss = val(val_loader, model)
#
#             print('train_loss: %.4f' % train_loss)
#             print('val_loss: %.4f' % val_loss)
#
#             if epoch - best_epoch > early_stop:
#                 break
#
#             if val_loss < val_loss_min:
#                 val_loss_min = val_loss
#                 best_epoch = epoch
#                 print('Minimum val loss!!!')
#                 torch.save(model.state_dict(), model_fp)
#                 print('Save model: %s' % model_fp)
#                 test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
#                 train_loss_, val_loss_ = train_loss, val_loss
#                 rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
#                 print(
#                     'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
#                         train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))
#
#                 if save_npy:
#                     np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
#                     np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
#                     np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)
#
#         train_loss_list.append(train_loss_)
#         val_loss_list.append(val_loss_)
#         test_loss_list.append(test_loss)
#         rmse_list.append(rmse)
#         mae_list.append(mae)
#         csi_list.append(csi)
#         pod_list.append(pod)
#         far_list.append(far)
#
#         print('\nNo.%2d experiment results:' % exp_idx)
#         print(
#             'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
#                 train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))
#
#     exp_metric_str = '---------------------------------------\n' + \
#                      'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
#                      'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
#                      'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
#                      'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
#                      'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
#                      'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
#                      'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
#                      'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))
#
#     metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
#     with open(metric_fp, 'w') as f:
#         f.write(exp_info)
#         f.write(str(model))
#         f.write(exp_metric_str)
#
#     print('=========================\n')
#     print(exp_info)
#     print(exp_metric_str)
#     print(str(model))
#     print(metric_fp)
#
#
# if __name__ == '__main__':
#     main()

import os
import sys

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
# Assuming util provides config and file_dir
# Make sure util.py and config.yaml are correctly set up
try:
    from util import config, file_dir
except ImportError as e:
    print(f"Error importing from util: {e}")
    print("Please ensure util.py and config.yaml are in the correct directory and accessible.")
    sys.exit(1)


from graph import Graph
# Removed pdb import here as set_trace is commented out
# import pdb
from dataset import HazeData # Assuming HazeData is correctly defined in dataset.py

# Import models - ensure these files exist in ./model/
try:
    from model.MLP import MLP
    from model.LSTM import LSTM
    from model.GRU import GRU
    from model.GC_LSTM import GC_LSTM
    from model.nodesFC_GRU import nodesFC_GRU
    from model.PM25_GNN import PM25_GNN
    from model.PM25_GNN_nosub import PM25_GNN_nosub
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Ensure model files (MLP.py, LSTM.py, etc.) exist in the 'model' subdirectory.")
    sys.exit(1)


import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil

# --- Configuration & Setup ---
torch.set_num_threads(1) # Good practice for reproducibility
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")

# Load graph data
try:
    graph = Graph()
    city_num = graph.node_num
except Exception as e:
    print(f"Error initializing Graph: {e}")
    print("Check graph.py and data dependencies (locations.txt, alt.pkl).")
    sys.exit(1)


# Training Hyperparameters from config
try:
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    hist_len = config['train']['hist_len']
    pred_len = config['train']['pred_len']
    weight_decay = config['train']['weight_decay']
    early_stop = config['train']['early_stop']
    lr = config['train']['lr']
    exp_repeat = config['train']['exp_repeat']
    results_dir = file_dir['results_dir'] # Ensure this path is correct in config.yaml
    dataset_num = config['experiments']['dataset_num']
    exp_model = config['experiments']['model']
    save_npy = config['experiments']['save_npy']
    metero_use = config['experiments']['metero_use'] # Get metero list
except KeyError as e:
    print(f"Error: Missing key in config.yaml: {e}")
    sys.exit(1)

seq_len = hist_len + pred_len
criterion = nn.MSELoss() # Mean Squared Error Loss

# --- Load Data ---
# Wrap data loading in try-except blocks for better error handling
try:
    print("Loading Training Data...")
    train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
    print("Loading Validation Data...")
    val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
    print("Loading Test Data...")
    test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
except Exception as e:
    print(f"Error loading HazeData: {e}")
    print("Check dataset.py and the underlying data files (e.g., knowair.npy).")
    sys.exit(1)

# --- Data Statistics ---
# Check if dataset loading was successful before accessing attributes
if 'train_data' in locals() and 'test_data' in locals():
    # Calculate input dimension based on loaded data
    in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1] # Features + PM2.5 itself
    wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
    # Use test_data mean/std for un-normalizing test predictions, as is common practice
    pm25_mean_test, pm25_std_test = test_data.pm25_mean, test_data.pm25_std
    print(f"Test Data PM2.5 Stats for Un-normalization: mean {pm25_mean_test:.6f} std {pm25_std_test:.6f}")
else:
    print("Error: Datasets not loaded successfully. Exiting.")
    sys.exit(1)


# --- Utility Functions ---

def get_metric(predict_epoch, label_epoch):
    """
    Calculates evaluation metrics (RMSE, MAE, CSI, POD, FAR).

    Args:
        predict_epoch (np.ndarray): Predictions array (Batch, Time, Nodes, Feat=1), un-normalized.
        label_epoch (np.ndarray): Ground truth array (Batch, Time, Nodes, Feat=1), un-normalized.

    Returns:
        tuple: rmse, mae, csi, pod, far
    """
    haze_threshold = 75 # PM2.5 threshold for haze event

    # Ensure inputs are numpy arrays
    predict_epoch = np.asarray(predict_epoch)
    label_epoch = np.asarray(label_epoch)

    # Clamp predictions to be non-negative
    predict_epoch[predict_epoch < 0] = 0

    # --- Haze Event Metrics (CSI, POD, FAR) ---
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold

    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))

    # Handle potential division by zero
    csi = hit / (hit + falsealarm + miss) if (hit + falsealarm + miss) > 0 else 0
    pod = hit / (hit + miss) if (hit + miss) > 0 else 0
    far = falsealarm / (hit + falsealarm) if (hit + falsealarm) > 0 else 0

    # --- Regression Metrics (RMSE, MAE) ---
    # Reshape for easier calculation: (Batch * Nodes, Time)
    # Assuming predict_epoch/label_epoch shape is (B, T, N, 1) -> (B, N, T) -> (B*N, T)
    if predict_epoch.ndim == 4 and predict_epoch.shape[-1] == 1:
        predict = predict_epoch[:, :, :, 0].transpose((0, 2, 1)) # (B, N, T)
        label = label_epoch[:, :, :, 0].transpose((0, 2, 1))     # (B, N, T)
    elif predict_epoch.ndim == 3: # If shape is already (B, N, T) or (B, T, N)
         # Check the time dimension, assume it's the last one for now
         if predict_epoch.shape[1] == city_num: # Shape (B, N, T)
             predict = predict_epoch
             label = label_epoch
         elif predict_epoch.shape[2] == city_num: # Shape (B, T, N)
             predict = predict_epoch.transpose((0, 2, 1))
             label = label_epoch.transpose((0, 2, 1))
         else:
             print(f"Warning: Unexpected shape for metric calculation: {predict_epoch.shape}. Assuming (B, T, N).")
             predict = predict_epoch.transpose((0, 2, 1))
             label = label_epoch.transpose((0, 2, 1))
    else:
        raise ValueError(f"Unexpected shape for metric calculation: {predict_epoch.shape}")


    predict = predict.reshape((-1, predict.shape[-1])) # (B*N, T)
    label = label.reshape((-1, label.shape[-1]))     # (B*N, T)

    # Calculate metrics per node-sequence, then average
    mae = np.mean(np.abs(predict - label)) # Global MAE
    rmse = np.sqrt(np.mean(np.square(predict - label))) # Global RMSE

    # Alternative: Calculate per sequence, then average
    # mae_per_seq = np.mean(np.abs(predict - label), axis=1) # MAE for each node-sequence
    # rmse_per_seq = np.sqrt(np.mean(np.square(predict - label), axis=1)) # RMSE for each node-sequence
    # mae = np.mean(mae_per_seq)
    # rmse = np.mean(rmse_per_seq)

    return rmse, mae, csi, pod, far


def get_exp_info():
    """ Returns a formatted string with experiment details. """
    exp_info = '============== Train Info ==============\n' + \
               f'Dataset number: {dataset_num}\n' + \
               f'Model: {exp_model}\n' + \
               f'Train: {train_data.start_time} --> {train_data.end_time}\n' + \
               f'Val: {val_data.start_time} --> {val_data.end_time}\n' + \
               f'Test: {test_data.start_time} --> {test_data.end_time}\n' + \
               f'City number: {city_num}\n' + \
               f'Use metero: {metero_use}\n' + \
               f'batch_size: {batch_size}\n' + \
               f'epochs: {epochs}\n' + \
               f'hist_len: {hist_len}\n' + \
               f'pred_len: {pred_len}\n' + \
               f'weight_decay: {weight_decay}\n' + \
               f'early_stop: {early_stop}\n' + \
               f'lr: {lr}\n' + \
               '========================================\n'
    return exp_info


def get_model():
    """ Initializes and returns the specified model. """
    if exp_model == 'MLP':
        model = MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        model = LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        model = GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        model = nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        # Ensure graph edge_index is tensor and on correct device
        edge_index_tensor = torch.tensor(graph.edge_index, dtype=torch.long).to(device)
        model = GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index_tensor)
    elif exp_model == 'PM25_GNN':
         # Ensure graph attributes are tensors and on correct device
        edge_index_tensor = torch.tensor(graph.edge_index, dtype=torch.long).to(device)
        edge_attr_tensor = torch.tensor(graph.edge_attr, dtype=torch.float).to(device)
        model = PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device,
                         edge_index_tensor, edge_attr_tensor, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        edge_index_tensor = torch.tensor(graph.edge_index, dtype=torch.long).to(device)
        edge_attr_tensor = torch.tensor(graph.edge_attr, dtype=torch.float).to(device)
        model = PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device,
                               edge_index_tensor, edge_attr_tensor, wind_mean, wind_std)
    else:
        raise ValueError(f'Unknown model name: {exp_model}')

    return model.to(device) # Ensure model is on the correct device


def prepare_batch(time_arr_batch, flag, dataset_obj):
    """
    Prepares history and label sequences for a batch using the dataset's full arrays.

    Args:
        time_arr_batch (torch.Tensor): Tensor of timestamps for the current batch.
        flag (str): 'Train', 'Val', or 'Test'.
        dataset_obj (HazeData): The corresponding dataset object (train_data, val_data, or test_data).

    Returns:
        tuple: Tensors for (pm25_hist, pm25_label, feature_seq, pm25_seq, [frp500_seq - only for Train])
               Returns None for a sample if data is insufficient.
    """
    B = time_arr_batch.shape[0]
    T_hist = dataset_obj.hist_len
    T_pred = dataset_obj.pred_len
    T_seq = T_hist + T_pred
    N = dataset_obj.graph.node_num
    F_feat = dataset_obj.feature_full.shape[-1]
    F_pm = dataset_obj.pm25_full.shape[-1] # Should be 1

    # Initialize tensors for the batch on CPU first
    pm25_hist_batch = torch.zeros((B, T_hist, N, F_pm), dtype=torch.float)
    pm25_label_batch = torch.zeros((B, T_pred, N, F_pm), dtype=torch.float)
    feature_seq_batch = torch.zeros((B, T_seq, N, F_feat), dtype=torch.float)
    pm25_seq_batch = torch.zeros((B, T_seq, N, F_pm), dtype=torch.float)
    if flag == "Train":
        # Assuming frp500 shape is (Time, Nodes)
        F_frp = dataset_obj.frp500.shape[-1] if dataset_obj.frp500.ndim > 1 else N # Adjust if frp500 is just (Time,)
        frp500_seq_batch = torch.zeros((B, T_seq, F_frp), dtype=torch.float) # Adjust shape if needed

    valid_indices = [] # Keep track of samples with enough data

    for i in range(B):
        current_time = time_arr_batch[i].item()
        try:
            end_idx = dataset_obj.time_index[current_time]
        except KeyError:
            print(f"Warning: Timestamp {current_time} not found in {flag} dataset time_index. Skipping sample {i}.")
            continue # Skip this sample

        seq_start_idx = end_idx - T_seq + 1
        hist_end_idx = end_idx - T_pred + 1
        label_end_idx = end_idx + 1

        if seq_start_idx < 0:
            # print(f"Warning: Not enough history for index {end_idx} (time {current_time}) in {flag} dataset. Skipping sample {i}.")
            continue # Skip this sample

        # Slice numpy arrays and convert to tensors
        pm25_hist_batch[i] = torch.from_numpy(dataset_obj.pm25_full[seq_start_idx:hist_end_idx, :, :])
        pm25_label_batch[i] = torch.from_numpy(dataset_obj.pm25_full[hist_end_idx:label_end_idx, :, :])
        feature_seq_batch[i] = torch.from_numpy(dataset_obj.feature_full[seq_start_idx:label_end_idx, :, :])
        pm25_seq_batch[i] = torch.from_numpy(dataset_obj.pm25_full[seq_start_idx:label_end_idx, :, :])
        if flag == "Train":
             # Ensure frp500 slicing matches its dimensions
            if dataset_obj.frp500.ndim == 2: # Shape (Time, Nodes)
                 frp500_seq_batch[i] = torch.from_numpy(dataset_obj.frp500[seq_start_idx:label_end_idx, :])
            elif dataset_obj.frp500.ndim == 1: # Shape (Time,) - Broadcast? Or error?
                 # This case needs clarification on how single FRP value applies to nodes
                 print(f"Warning: frp500 has unexpected shape {dataset_obj.frp500.shape}. Assuming broadcast needed.")
                 frp_slice = torch.from_numpy(dataset_obj.frp500[seq_start_idx:label_end_idx])
                 frp500_seq_batch[i] = frp_slice.unsqueeze(1).expand(-1, F_frp) # Example broadcast
            else:
                 raise ValueError(f"Unexpected frp500 shape: {dataset_obj.frp500.shape}")


        valid_indices.append(i)

    # Filter batch tensors to include only valid samples
    if not valid_indices: # Handle case where no samples in the batch are valid
        print(f"Warning: No valid samples found in batch for {flag}.")
        if flag == "Train":
            return None, None, None, None, None
        else:
            return None, None, None, None

    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
    pm25_hist_batch = pm25_hist_batch[valid_indices_tensor]
    pm25_label_batch = pm25_label_batch[valid_indices_tensor]
    feature_seq_batch = feature_seq_batch[valid_indices_tensor]
    pm25_seq_batch = pm25_seq_batch[valid_indices_tensor]

    if flag == "Train":
        frp500_seq_batch = frp500_seq_batch[valid_indices_tensor]
        return pm25_hist_batch, pm25_label_batch, feature_seq_batch, pm25_seq_batch, frp500_seq_batch
    else:
        return pm25_hist_batch, pm25_label_batch, feature_seq_batch, pm25_seq_batch


# --- Training, Validation, Testing Functions ---

def train(train_loader, model, optimizer, criterion):
    """ Trains the model for one epoch. """
    model.train()
    train_loss = 0
    processed_batches = 0
    global train_data # Access the global train_data object

    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc="Train Epoch", leave=False)
    for batch_idx, data in enumerate(pbar):
        # Data from DataLoader is already a batch of tensors (pm25_now, feature_now, time_now)
        _, _, time_arr_batch = data # We only need time_arr to fetch sequences

        # Prepare history and labels for the batch
        prepared_data = prepare_batch(time_arr_batch, "Train", train_data)
        if prepared_data[0] is None: # Skip if batch preparation failed
            continue
        pm25_hist, pm25_label, feature_seq, _, frp500_seq = prepared_data

        # Move data to the target device
        feature_seq = feature_seq.to(device)
        pm25_hist = pm25_hist.to(device)
        pm25_label = pm25_label.to(device)
        frp500_seq = frp500_seq.to(device) # Move fire data too

        # --- Fire condition check (Example) ---
        # Adapt this logic based on how fire data should influence training
        thresh = 0.15
        # Check if *any* value in the sequence for *any* sample in the batch exceeds thresh
        exceed = (frp500_seq > thresh).any()

        if exceed:
             # print(f"Skipping training batch {batch_idx} due to fire condition.")
             continue # Skip this batch if fire condition met

        # --- Forward and Backward Pass ---
        optimizer.zero_grad()
        pm25_pred = model(pm25_hist, feature_seq) # Pass history PM2.5 and feature sequence
        loss = criterion(pm25_pred, pm25_label)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss encountered in training batch {batch_idx}. Skipping backward pass.")
            # Potentially log inputs/outputs here for debugging
            continue

        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        processed_batches += 1
        pbar.set_postfix(loss=loss.item()) # Update progress bar

    pbar.close()

    if processed_batches == 0:
        print("Warning: No batches were processed in this training epoch.")
        return 0.0

    avg_train_loss = train_loss / processed_batches
    print(f"Processed {processed_batches} training batches.")
    return avg_train_loss


@torch.no_grad() # Disable gradient calculation for validation/testing
def evaluate(loader, model, criterion, flag, dataset_obj):
    """ Evaluates the model on validation or test data. """
    model.eval()
    total_loss = 0
    processed_batches = 0
    predict_list, label_list, time_list = [], [], []

    # Use tqdm for progress bar
    pbar = tqdm(loader, desc=f"{flag} Epoch", leave=False)
    for batch_idx, data in enumerate(pbar):
        _, _, time_arr_batch = data

        prepared_data = prepare_batch(time_arr_batch, flag, dataset_obj)
        if prepared_data[0] is None:
            continue

        if flag == "Val":
             pm25_hist, pm25_label, feature_seq, _ = prepared_data
        else: # Test - also need the full pm25 sequence for un-normalization later
             pm25_hist, pm25_label, feature_seq, pm25_seq = prepared_data


        # Move data to device
        feature_seq = feature_seq.to(device)
        pm25_hist = pm25_hist.to(device)
        pm25_label_dev = pm25_label.to(device) # Keep label on CPU for loss calc if needed, move copy

        # --- Forward Pass ---
        pm25_pred = model(pm25_hist, feature_seq)
        loss = criterion(pm25_pred, pm25_label_dev)

        if torch.isnan(loss):
            print(f"Warning: NaN loss encountered in {flag} batch {batch_idx}.")
            continue

        total_loss += loss.item()
        processed_batches += 1
        pbar.set_postfix(loss=loss.item())

        # --- Store results for metrics (only needed for test set usually) ---
        if flag == "Test":
            # Store predictions and labels for final metric calculation
            # Un-normalize using test set mean/std
            # Prediction is only for pred_len, prepend hist for context if needed by metric fn
            pred_unnorm = pm25_pred.cpu().numpy() * pm25_std_test + pm25_mean_test
            # Label needs to be un-normalized too (it came from prepare_batch normalized)
            label_unnorm = pm25_label.cpu().numpy() * pm25_std_test + pm25_mean_test

            predict_list.append(pred_unnorm)
            label_list.append(label_unnorm)
            time_list.append(time_arr_batch.cpu().numpy()) # Store corresponding times

    pbar.close()

    if processed_batches == 0:
        print(f"Warning: No batches were processed during {flag}.")
        if flag == "Test":
             return 0.0, None, None, None
        else:
             return 0.0

    avg_loss = total_loss / processed_batches
    print(f"Processed {processed_batches} {flag} batches.")


    if flag == "Test":
        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        time_epoch = np.concatenate(time_list, axis=0)
        return avg_loss, predict_epoch, label_epoch, time_epoch
    else: # Validation
        return avg_loss


def get_mean_std(data_list):
    """ Calculates mean and standard deviation of a list of numbers. """
    data = np.array(data_list)
    return data.mean(), data.std()


# --- Main Execution Logic ---

def main():
    """ Main function to run the training and evaluation loop. """
    exp_info = get_exp_info()
    print(exp_info)

    # Use UTC time for unique experiment identifier
    exp_time = arrow.utcnow().format('YYYYMMDDHHmmss')

    # Lists to store metrics across repeats
    all_metrics = {'train_loss': [], 'val_loss': [], 'test_loss': [],
                   'rmse': [], 'mae': [], 'csi': [], 'pod': [], 'far': []}

    for exp_idx in range(exp_repeat):
        print(f'\n--- Experiment Repeat {exp_idx + 1}/{exp_repeat} ---')

        # DataLoaders - consider num_workers > 0 if I/O is bottleneck, but start with 0
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

        model = get_model()
        model_name = type(model).__name__
        print(f"Model Architecture:\n{model}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam is often a good default
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Optional LR scheduler

        # --- Setup Results Directory ---
        # Structure: results_dir / hist_pred / dataset_num / model_name / exp_time / repeat_idx
        exp_model_dir = os.path.join(results_dir, f'{hist_len}_{pred_len}', str(dataset_num), model_name, exp_time, f'{exp_idx:02d}')
        os.makedirs(exp_model_dir, exist_ok=True)
        print(f"Results will be saved in: {exp_model_dir}")
        model_fp = os.path.join(exp_model_dir, 'model_best.pth') # Save best model

        # --- Training Loop ---
        val_loss_min = float('inf')
        best_epoch = -1
        epochs_no_improve = 0

        best_test_metrics = {} # Store metrics corresponding to the best validation epoch

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')

            train_loss = train(train_loader, model, optimizer, criterion)
            val_loss = evaluate(val_loader, model, criterion, "Val", val_data)
            # scheduler.step() # Step the scheduler if using one

            print(f'Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # --- Early Stopping and Model Saving ---
            if val_loss < val_loss_min:
                print(f'Validation loss decreased ({val_loss_min:.4f} --> {val_loss:.4f}). Saving model...')
                val_loss_min = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_fp)
                print(f'Saved best model to: {model_fp}')

                # Evaluate on test set only when validation improves
                print("Running test set evaluation for best model...")
                test_loss, predict_epoch, label_epoch, time_epoch = evaluate(test_loader, model, criterion, "Test", test_data)
                if predict_epoch is not None:
                     rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
                     print(f'Test Metrics (Epoch {epoch + 1}): Loss: {test_loss:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, CSI: {csi:.4f}, POD: {pod:.4f}, FAR: {far:.4f}')
                     best_test_metrics = {'test_loss': test_loss, 'rmse': rmse, 'mae': mae, 'csi': csi, 'pod': pod, 'far': far,
                                          'train_loss': train_loss, 'val_loss': val_loss} # Store all relevant losses

                     if save_npy:
                         print("Saving test predictions and labels...")
                         np.save(os.path.join(exp_model_dir, 'predict_best.npy'), predict_epoch)
                         np.save(os.path.join(exp_model_dir, 'label_best.npy'), label_epoch)
                         np.save(os.path.join(exp_model_dir, 'time_best.npy'), time_epoch)
                else:
                     print("Test evaluation skipped as no valid batches were processed.")


            else:
                epochs_no_improve += 1
                print(f'Validation loss did not improve for {epochs_no_improve} epoch(s).')

            if epochs_no_improve >= early_stop:
                print(f'Early stopping triggered after {early_stop} epochs without improvement.')
                break

        # --- Store Results for this Repeat ---
        if best_test_metrics: # Check if best_test_metrics was populated
             print(f'\nEnd of Repeat {exp_idx + 1}. Best Epoch: {best_epoch + 1}')
             print(f'Metrics from Best Epoch: {best_test_metrics}')
             for key in all_metrics.keys():
                 if key in best_test_metrics:
                     all_metrics[key].append(best_test_metrics[key])
                 else:
                      # Append NaN or handle missing metric if test eval didn't run/succeed
                      all_metrics[key].append(float('nan'))
                      print(f"Warning: Metric '{key}' not found for repeat {exp_idx + 1}.")

        else: # Handle case where training finished without improvement or test eval failed
            print(f"\nEnd of Repeat {exp_idx + 1}. No improvement found or test evaluation failed.")
            # Append NaN to all metrics for this repeat
            for key in all_metrics.keys():
                all_metrics[key].append(float('nan'))


    # --- Aggregate and Save Final Results ---
    print('\n--- Overall Experiment Results ---')
    exp_metric_str = '---------------------------------------\n'
    for key, values in all_metrics.items():
        if values: # Check if list is not empty
             mean_val, std_val = get_mean_std(values)
             exp_metric_str += f'{key:<10s} | mean: {mean_val:0.4f} std: {std_val:0.4f}\n'
        else:
             exp_metric_str += f'{key:<10s} | No valid results recorded.\n'


    # Save metrics to file in the parent directory of repeats
    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metrics_summary.txt')
    try:
        with open(metric_fp, 'w') as f:
            f.write(exp_info)
            # Write model architecture (load best model from first repeat if exists)
            first_repeat_model_fp = os.path.join(results_dir, f'{hist_len}_{pred_len}', str(dataset_num), model_name, exp_time, '00', 'model_best.pth')
            if os.path.exists(first_repeat_model_fp):
                 model = get_model() # Re-initialize model structure
                 model.load_state_dict(torch.load(first_repeat_model_fp, map_location='cpu')) # Load weights
                 f.write(f"\nModel Architecture:\n{model}\n")
            else:
                 f.write("\nModel Architecture: Could not load from first repeat.\n")

            f.write("\nAggregated Metrics (Mean +/- Std across repeats):\n")
            f.write(exp_metric_str)
        print(f"\nAggregated metrics saved to: {metric_fp}")
    except Exception as e:
        print(f"Error saving metrics summary: {e}")


    print('\n=========================')
    print("Experiment Finished.")
    print(exp_info)
    print(exp_metric_str)
    print(f"Metrics summary file: {metric_fp}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n--- An error occurred during execution ---")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------")
        sys.exit(1) # Exit with error code
