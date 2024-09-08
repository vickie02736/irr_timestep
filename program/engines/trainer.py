import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from .engine import Engine
from torch.utils.data import DataLoader, DistributedSampler


class Trainer(Engine):
    def __init__(self, args):
        Engine.__init__(self, args)
        self.init_train_dataloader()
        self.init_loss_function()

    def init_train_dataloader(self):
        if self.args.database == 'shallow_water':
            from database.shallow_water.dataset import seq_DataBuilder, fra_DataBuilder
            if self.args.model_name == 'cae':
                dataset = fra_DataBuilder(self.data_config['train_file'], timestep=100)
            else:
                dataset = seq_DataBuilder(self.data_config['train_file'],
                                                self.config['seq_length'],
                                                self.config['train']['rollout_times'],
                                                timestep=100)
        elif self.args.database == 'weather_2m_temperature':
            from database.weather.dataset import seq_DataBuilder
            dataset = seq_DataBuilder('train',
                                      self.config['seq_length'],
                                      self.config['train']['rollout_times'])
        
        
        sampler = DistributedSampler(dataset,
                                     num_replicas=self.world_size,
                                     rank=self.rank)
        self.train_loader = DataLoader(dataset,
                                batch_size=self.config[self.args.model_name]['train_batch_size'],
                                pin_memory=True,
                                shuffle=False,
                                drop_last=True,
                                sampler=sampler,
                                num_workers=self.args.cpu)
        self.len_dataset = len(dataset)

    def init_training_components(self):
        # optimizer
        if self.config['train']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                betas=(0.9, 0.95),
                weight_decay=0.03)
        elif self.config['train']['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                alpha=0.9)
        elif self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'])
        else:
            pass

        # scheduler
        self.scheduler = self.create_scheduler()

        # scaler
        self.scaler = torch.cuda.amp.GradScaler()

        if self.args.resume_epoch != 1:
            self.optimizer.load_state_dict(self.loaded_checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.loaded_checkpoint['scheduler'])
            self.scaler.load_state_dict(self.loaded_checkpoint['scaler'])
            self.global_step = self.loaded_checkpoint['global_step']

    def init_loss_function(self):
        if self.config['train']['loss_fn'] == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.config['train']['loss_fn'] == 'L1':
            self.loss_fn = nn.L1Loss()
        elif self.config['train']['loss_fn'] == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Invalid loss function')

    def save_checkpoint(self, epoch, save_path):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
        }
        torch.save(save_dict, save_path)

    def create_scheduler(self): 
        total_step = int(self.args.epochs * self.len_dataset // self.config[self.args.model_name]['train_batch_size'])
        cosine_step = int(total_step * 0.95)
        self.warmup_step = int(total_step - cosine_step)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=cosine_step, eta_min=1e-8)

        