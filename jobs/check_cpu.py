import time
import torch
from torch.utils.data import DataLoader
import yaml
import sys
sys.path.append("..")
from database.weather.dataset import seq_DataBuilder


# Function to benchmark DataLoader
def find_optimal_num_workers(dataset, batch_size, max_workers=8):
    num_workers_times = []
    for num_workers in range(0, max_workers + 1):
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        
        start_time = time.time()
        for _ in dataloader:
            pass  # Iterate through the dataloader to measure the time
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        num_workers_times.append((num_workers, elapsed_time))
        print(f"num_workers: {num_workers}, Time: {elapsed_time:.2f} seconds")
    
    # Find the optimal number of workers with the lowest time
    optimal_num_workers = min(num_workers_times, key=lambda x: x[1])[0]
    return optimal_num_workers

# Usage
dataset = seq_DataBuilder('train', 10, 2)
batch_size = 2400
optimal_workers = find_optimal_num_workers(dataset, batch_size)
print(f"Optimal num_workers: {optimal_workers}")
