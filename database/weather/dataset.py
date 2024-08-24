import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import ast
import torch
import sys

sys.path.append("..")
import yaml

config = yaml.load(open("../database/weather/config.yaml", "r"),
                   Loader=yaml.FullLoader)
total_sequence = np.load(f"../database/weather/data/2m_temperature/total.npy", allow_pickle=True, mmap_mode='r')

class seq_DataBuilder(Dataset): 
    
    def __init__(self,
                 dataset,
                 clip_length,
                 rollout_times,
                 transform=None): 
        
        self.clip_length = clip_length
        self.rollout_times = rollout_times
        self.total_length = clip_length * (rollout_times + 1)

        df = pd.read_csv(
            f"../database/weather/data/2m_temperature/total.csv")
        
        if dataset == 'train':
            year = config['train_year']
        elif dataset == 'valid':
            year = config['valid_year']
        else:
            year = config['test_year']
        
        self.sorted_df = df[(df['Year'] >= year[0]) & (df['Year'] <= year[1])].sort_values(by='Timestep').reset_index(drop=True)

        self.all_clips = np.lib.stride_tricks.sliding_window_view(self.sorted_df['Timestep'].values, self.total_length)

    def __len__(self):
        return len(self.sorted_df) - self.total_length + 1
    
    def __getitem__(self, idx):
        clips = self.all_clips[idx]
        input_clips = clips[:self.clip_length]
        target_clips = clips[self.clip_length:]
        
        input_data = torch.tensor(total_sequence[input_clips], dtype=torch.float32)
        target_data = torch.tensor(total_sequence[target_clips], dtype=torch.float32)

        input_data = input_data.unsqueeze(1)
        target_data = target_data.unsqueeze(1)
        
        return {
            "Input": input_data, 
            "Target": target_data
        }


# from torch.utils.data import DataLoader

# data_loader = DataLoader(seq_DataBuilder(dataset='train', clip_length=10, rollout_times=2), batch_size=64, shuffle=True)

# for i, data in enumerate(data_loader):
#     print(data['Input'].shape, data['Target'].shape)
#     break