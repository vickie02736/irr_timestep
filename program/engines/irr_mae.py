import os
import copy
import json
import torch
from matplotlib import pyplot as plt

from models import VisionTransformer
from .trainer import Trainer
from .evaluator import Evaluator, Tester
from utils import save_losses, mask, plot_losses


class ImaeTrainer(Trainer, Evaluator):

    def __init__(self, args):
        Trainer.__init__(self, args)
        Evaluator.__init__(self, args)
        self.load_model()   # Here
        self.setup()        # Engine
        self.load_checkpoint()
        self.init_training_components() # Trainer

        self.best_losses = {'MSE': None, 'RMSE': None, 'MAE': None, 'SSIM': None, 'PSNR': None}

    def load_model(self):
        self.model = VisionTransformer(self.config)

    def train_epoch(self, epoch):
        torch.manual_seed(epoch)
        self.model.train()
        total_predict_loss = 0.0
        total_rollout_loss = 0.0

        for _, sample in enumerate(self.train_loader):
            origin, _ = mask(sample["Input"], mask_mtd=self.config["mask_method"])
            origin = origin.float().to(self.device)
            target = sample["Target"].float().to(self.device)
            target_chunks = torch.chunk(target, self.config['train']['rollout_times'], dim=1)

            with torch.cuda.amp.autocast():
                output = self.model(origin)
                predict_loss = self.loss_fn(output, target_chunks[0])
                output = self.model(output)
                rollout_loss = self.loss_fn(output, target_chunks[1])
                loss = predict_loss + rollout_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_predict_loss += predict_loss.item()
            total_rollout_loss += rollout_loss.item()

        if self.rank == 0:
            average_predict_loss = total_predict_loss / len(self.train_loader.dataset)
            average_rollout_loss = total_rollout_loss / len(self.train_loader.dataset)

            loss_data = {
                'predict_loss': average_predict_loss,
                'rollout_loss': average_rollout_loss
            }

            save_losses(epoch, loss_data, os.path.join(self.save_loss_path, 'train_losses.json'))
            if epoch % self.args.save_frequency == 0:
                self.save_checkpoint(epoch, os.path.join(self.save_checkpoint_path, f'checkpoint_{epoch}.pth'))

    def evaluate_epoch(self, epoch):
        self.model.eval()
        running_losses = {metric: [0.0] * self.rollout_times for metric in self.loss_functions.keys()}

        with torch.no_grad():
            for i, sample in enumerate(self.eval_loader):
                origin_before_masked = copy.deepcopy(sample["Input"])
                origin, _ = mask(sample["Input"], mask_mtd=self.config["mask_method"])
                origin_plot = copy.deepcopy(origin)
                origin = origin.float().to(self.device)
                target = sample["Target"].float().to(self.device)
                target_chunks = torch.chunk(target, self.rollout_times, dim=1)

                output_chunks = []
                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        output = self.model(origin)
                    else:
                        output = self.model(output)
                    output_chunks.append(output)
                    # Compute losses
                    for metric, loss_fn in self.loss_functions.items():
                        loss = loss_fn(output, chunk)
                        running_losses[metric][j] += loss.item()

                if i == 1 and epoch % 1 == 0:
                    self.plot(epoch, origin_before_masked, origin_plot, 
                              output_chunks, target_chunks)
                            

            chunk_losses = {key: [val / len(self.eval_loader.dataset) for val in values] for key, values in running_losses.items()}
            print(chunk_losses)
            save_losses(epoch, chunk_losses, os.path.join(self.save_loss_path, 'valid_losses.json'))
            
            if epoch % 5 == 0:
                plot_losses(self.save_loss_path)

            # save best checkpoint
            keys = ['MSE', 'RMSE', 'MAE']
            updated = False
            for key in keys:
                if self.best_losses[key] is None or chunk_losses[key] < self.best_losses[key]:
                    self.best_losses[key] = chunk_losses[key]
                    updated = True
            if updated:
                self.save_checkpoint(epoch, os.path.join(self.save_checkpoint_path, 'best_checkpoint.pth'))


    def plot(self, idx, origin, masked_origin, 
             output_chunks, target_chunks):
                
        rollout_times = self.config['train']['rollout_times']
        seq_len = self.config['seq_length']

        _, ax = plt.subplots(rollout_times * 2 + 2, seq_len + 1,
                             figsize=(seq_len * 2 + 2, rollout_times * 4 + 4))
        row_titles = ["Original input", "Masked input", "Direct prediction", 
                      "Target", "Rollout prediction", "Target"]
        for i, title in enumerate(row_titles):
            ax[i][0].text(1.0, 0.5, title, verticalalignment='center', horizontalalignment='right', fontsize=12)
            ax[i][0].axis('off')
        for j in range(seq_len):
            # visualise input
            ax[0][j + 1].imshow(origin[0][j][0].cpu().detach().numpy())
            ax[0][j + 1].set_xticks([])
            ax[0][j + 1].set_yticks([])
            ax[0][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1), fontsize=10)
            # visualise masked input
            ax[1][j + 1].imshow(masked_origin[0][j][0].cpu().detach().numpy())
            ax[1][j + 1].set_xticks([])
            ax[1][j + 1].set_yticks([])
            ax[1][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1), fontsize=10)
        for k in range(rollout_times):
            for j in range(seq_len):
                # visualise output
                ax[2 * k + 2][j + 1].imshow(
                    output_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 2][j + 1].set_xticks([])
                ax[2 * k + 2][j + 1].set_yticks([])
                ax[2 * k + 2][j + 1].set_title("Timestep {timestep}".format(timestep=j + (k + 1) * seq_len), fontsize=10)
                # visualise target
                ax[2 * k + 3][j + 1].imshow(target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 3][j + 1].set_xticks([])
                ax[2 * k + 3][j + 1].set_yticks([])
                ax[2 * k + 3][j + 1].set_title("Timestep {timestep}".format(timestep=j + (k + 1) * seq_len), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_reconstruct_path, f"{idx}.png"))
        plt.close()




class ImaeTester(Tester):
    def __init__(self, args):
        super().__init__(args)
        self.load_model()
        self.setup()
        self.load_checkpoint()

    def load_model(self):
        self.model = VisionTransformer(self.config)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for _, sample in enumerate(self.eval_loader):
                origin, _ = mask(sample["Input"], mask_mtd='zeros', test_flag=True, mask_ratio=self.args.mask_ratio)
                origin = origin.float().to(self.device)
                target = sample["Target"].float().to(self.device)
                target_chunks = torch.chunk(target, self.args.rollout_times, dim=1)

                output_chunks = []
                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        output = self.model(origin)
                    else:
                        output = self.model(output)
                    output_chunks.append(output)
                    # Compute losses
                    for metric, loss_fn in self.loss_functions.items():
                        loss = loss_fn(output, chunk)
                        self.running_losses[metric][j] += loss.item()
            chunk_losses = {}
            for metric, running_loss_list in self.running_losses.items():
                average_loss = [_ / len(self.eval_loader.dataset) for _ in running_loss_list]
                chunk_losses[metric] = average_loss
            loss_savepath = os.path.join(self.config['imae']['save_loss'], f'test_loss_{self.args.mask_ratio}.json')
            with open(loss_savepath, 'w') as file:
                json.dump(chunk_losses, file)