import os
import json
import torch
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import torch.multiprocessing as mp

def int_or_string(value):
    if value == "best":
        return value
    else:
        return int(value)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_losses(epoch, loss_data, save_path):
    with open(os.path.join(save_path), 'r') as f:
        losses = json.load(f)
    losses[epoch] = loss_data
    with open(os.path.join(save_path), 'w') as f:
        json.dump(losses, f)


def mask(x, mask_mtd="zeros", test_flag=False, mask_ratio=None):
    seq_lenth = len(x[0])
    if test_flag == False:
        mask_ratio = torch.rand(1).item()
    else:
        mask_ratio = mask_ratio
    num_mask = int(1 + mask_ratio * (seq_lenth - 2))
    weights = torch.ones(x.shape[1]).expand(x.shape[0], -1)
    idx = torch.multinomial(weights, num_mask, replacement=False)
    if mask_mtd == "zeros":
        masked_tensor = torch.zeros(x.shape[2], x.shape[3],
                                    x.shape[4]).to(x.device)
    elif mask_mtd == "random":
        masked_tensor = torch.rand(x.shape[2], x.shape[3],
                                   x.shape[4]).to(x.device)
    batch_indices = torch.arange(x.shape[0],
                                 device=x.device).unsqueeze(1).expand(
                                     -1, num_mask)
    x[batch_indices, idx] = masked_tensor
    return x, idx



def plot_losses(folder_path):
    # Helper function to load JSON data
    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    
    # Load the train and valid loss data
    train_losses = load_json(os.path.join(folder_path, 'train_losses.json'))
    valid_losses = load_json(os.path.join(folder_path, 'valid_losses.json'))

    # Split valid losses into predict and rollout components
    split_losses = {}
    for epoch, metrics in valid_losses.items():
        split_losses[epoch] = {}
        for metric, values in metrics.items():
            split_losses[epoch][f'predict_{metric}'] = values[0]
            split_losses[epoch][f'rollout_{metric}'] = values[1]

    # Convert dictionaries to DataFrames
    train_df = pd.DataFrame.from_dict(train_losses, orient='index')
    valid_df = pd.DataFrame.from_dict(split_losses, orient='index')

    # Merge the train and valid DataFrames
    df = pd.concat([train_df, valid_df], axis=1)
    df = df.reset_index().rename(columns={'index': 'epoch'})

    # Define plot settings and metric mappings
    plot_configs = [
        ('predict_loss', 'rollout_loss', 'Train Loss', 'Loss'),
        ('predict_MSE', 'rollout_MSE', 'Validation MSE', 'MSE'),
        ('predict_MAE', 'rollout_MAE', 'Validation MAE', 'MAE'),
        ('predict_RMSE', 'rollout_RMSE', 'Validation RMSE', 'RMSE'),
        ('predict_SSIM', 'rollout_SSIM', 'Validation SSIM', 'SSIM'),
        ('predict_PSNR', 'rollout_PSNR', 'Validation PSNR', 'PSNR'),
        ('predict_loss', 'predict_MSE', 'Direct Prediction MSE', 'Loss'),
        ('rollout_loss', 'rollout_MSE', 'Rollout Prediction MSE', 'Loss')
    ]

    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    axs = axs.flatten()

    for ax, (y1, y2, title, ylabel) in zip(axs, plot_configs):
        if y1 in df.columns and y2 in df.columns:
            ax.plot(df['epoch'], df[y1], label=y1)
            ax.plot(df['epoch'], df[y2], label=y2)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            # ax.set_yscale('log')  # Set y-axis to log scale
            ax.legend()
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'losses.png'))
    plt.close()