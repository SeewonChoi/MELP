import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import butter, sosfilt
import neurokit2 as nk
import pandas as pd
import ipdb


class QRS_Tokenizer(nn.Module):
    def __init__(self, window_size, sentence_len, num_beats=1, fs=500, down_fs=100):
        super().__init__()
        self.window_size = window_size
        self.sentence_len = sentence_len
        self.num_beats = num_beats
        self.fs = fs
        self.down_fs = down_fs
    
    def downsample_signal(self, x, original_fs, target_fs):
        # Downsample the signal using neurokit2
        x_np = x.cpu().numpy()  # Convert to numpy array
        x_downsampled = nk.signal_resample(x_np, sampling_rate=original_fs, desired_sampling_rate=target_fs) 
        return torch.tensor(x_downsampled, dtype=x.dtype, device=x.device)
    
    def qrs_detection(self, x):
        # Perform QRS detection using neurokit2
        QRS_index = []
        x_np = x.cpu().numpy()  # Convert to numpy array
        for i in range(x_np.shape[0]):
            _, rpeaks = nk.ecg_peaks(x_np[i, :], sampling_rate=self.down_fs)
            QRS_index.append(rpeaks['ECG_R_Peaks'])

        return QRS_index
    
    def patch(self, x, QRS_index):
        # boundaries be the midpoints of the QRS index
        x_patch = []
        for i in range(len(QRS_index)):
            cur_qrs_index = QRS_index[i]
            if len(cur_qrs_index) == 0:
                continue
            bound_1 = cur_qrs_index[:-1]
            bound_2 = cur_qrs_index[1:]
            midpoints = (bound_1 + bound_2) // 2
            midpoints = np.insert(midpoints, 0, 0)
            midpoints = np.append(midpoints, x.shape[2])
            patched_x = []
            for j in range(0, len(midpoints) - 1, self.num_beats):
                left = midpoints[j]
                right = min(midpoints[j+1], x.shape[2])
                x_slice = x[i, :, left:right]
                if x_slice.shape[-1] < self.window_size:
                    left_pad_num = (self.window_size - x_slice.shape[1]) // 2
                    right_pad_num = self.window_size - x_slice.shape[1] - left_pad_num
                    x_slice = torch.cat((
                        torch.ones(x_slice.shape[0], left_pad_num) * x_slice[:, 0].reshape(-1, 1), 
                        x_slice, 
                        torch.ones(x_slice.shape[0], right_pad_num) * x_slice[:, -1].reshape(-1, 1)
                        ), dim=1)
                elif x_slice.shape[-1] > self.window_size:
                    remove_num = (x_slice.shape[1] - self.window_size) // 2
                    x_slice = x_slice[:, remove_num:remove_num+self.window_size]
                patched_x.append(x_slice)  # x_slice: 12, 96
            x_patch.append(torch.stack(patched_x, dim=0).reshape(-1, self.window_size))
        return x_patch
    
    def padding(self, x):
        for i in range(len(x)):
            if x[i].shape[0] < self.sentence_len:
                padding = self.sentence_len - x[i].shape[0]
                x[i] = F.pad(x[i],(0,0,0,padding) )
            else:
                x[i] = x[i][:self.sentence_len,:]  
        x = torch.stack(x)
        return x
    
    @staticmethod
    def calculate_t_indices(QRS_index, sentence_len):
        t_indices = []
        for i in range(len(QRS_index)):
            ipdb.set_trace()
            idx = QRS_index[i] // 100 + 1
            # repeat idx for 12 leads
            idx = np.tile(idx, 12)
            if len(idx) > sentence_len:
                idx = idx[:sentence_len]
            else:
                n_padding = sentence_len - len(idx)
                idx = np.pad(idx, (0, n_padding))
            idx = torch.tensor(idx, dtype=int)
            t_indices.append(idx)
        torch.stack(t_indices)
        return t_indices

    @staticmethod
    def calculate_s_indices(QRS_index, sentence_len):
        s_indices = []
        for i in range(len(QRS_index)):
            lead_idx = range(1,13) 
            idx = np.repeat(lead_idx, len(QRS_index[i]), axis=-1)
            if len(idx) > sentence_len:
                idx = idx[:sentence_len]
            else:
                idx = np.pad(idx, (0, sentence_len - len(idx)))
            idx = torch.tensor(idx, dtype=int)
            s_indices.append(idx)
        torch.stack(s_indices)
        return s_indices

    def forward(self, x):
        x_downsampled = F.interpolate(x, scale_factor=self.down_fs/self.fs, mode='linear')
        # x_downsampled = torch.stack([torch.stack(
        #     [self.downsample_signal(x[i, j, :], self.fs, self.down_fs) 
        #      for j in range(x.shape[1])]) 
        #      for i in range(x.shape[0])])
        lead_I = x_downsampled[:, 0, :]
        QRS_index = self.qrs_detection(lead_I)
        # x_patch = self.patch(x_downsampled, QRS_index)
        # # # print(x_patch[0].shape)
        # x_pad = self.padding(x_patch)
        t_indices = self.calculate_t_indices(QRS_index, self.sentence_len)
        s_indices = self.calculate_s_indices(QRS_index, self.sentence_len)
        x_pad = x_pad.reshape(x.shape[0], 12, -1, self.window_size)

        return x_pad, QRS_index, t_indices, s_indices



if __name__ == "__main__":
    from melp.datasets.pretrain_datamodule import ECGTextDataModule
    tokenizer = QRS_Tokenizer(window_size=96, sentence_len=252)
    dm = ECGTextDataModule(
        dataset_dir="/disk1/*/ECG/raw",
        dataset_list=["mimic-iv-ecg"],
        batch_size=4,
        num_workers=1,
        train_data_pct=0.1,
    )
    for batch in dm.train_dataloader():
        x = batch["ecg"]
        report = batch["report"]    
        x, q_indices, t_indices, s_indices = tokenizer(x)
        # ipdb.set_trace()
        break

    # x, q_indices, t_indices, s_indices = tokenizer(x)
    # max_len = max(arr.shape[0] for arr in q_indices)
    # # print(max_len)
    # padded_np = [np.pad(arr.astype(float), (0, max_len - arr.shape[0]), constant_values = np.nan) for arr in q_indices]
    # df = pd.DataFrame(padded_np)
    # # df['t_indices'] = t_indices
    # # df['s_indices'] = s_indices
    # # df['patient_id'] = batch['patient_id']
    # df.to_csv('q_indices.csv')
    