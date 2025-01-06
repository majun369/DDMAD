import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pickle

from model.losses import mmdx, local_infoNCE, infoNCE
from utils.utils import *
from model.MYdetector import MYdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
from scipy.stats import norm
from model.residual_loss import residual_loss_fn

warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save_checkpoint(val_loss, model, path)
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            #if self.counter >= self.patience:
            #    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint_{0:.5f}.pth'.format(val_loss))
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        self.meta_lr = 0.012

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.device = torch.device("cuda:{0}".format(self.gpu) if torch.cuda.is_available() else "cpu")

        self.build_model()
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = MYdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads,
                                d_model=self.d_model, e_layers=self.e_layers, channel=self.input_c, k=self.k,
                                kernel_size=self.kernel_size)

        if torch.cuda.is_available():
            self.model.cuda(device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        criterion = self.criterion
        loss_1 = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                out = self.model(input)
                loss = criterion(out, input)
                all_loss = loss
                loss_1.append(all_loss.item())

        self.model.train()

        return np.average(loss_1)

    def train(self):

        time_now = time.time()
        path = self.model_save_path
        batch_size = self.batch_size
        win_size = self.win_size
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        train_steps = len(self.train_loader)  
        criterion = nn.MSELoss(reduction='none')


        indicies = np.arange(train_steps*batch_size)

        memory_module = dict([(k, []) for k in indicies])  

        mean = (win_size - 1) / 2
        std_dev = win_size / 6  
        weights = torch.exp(-0.5 * ((torch.arange(win_size) - mean) / std_dev) ** 2)
        weights = weights / weights.sum()  
        weights = weights.to(self.device)  

        warm_epochs = self.warm_epochs

        for epoch in range(self.num_epochs):
            if epoch >= warm_epochs:
                epoch_weight = [(1+i)/self.num_epochs for i in range(epoch)]
                instance_mean = {k: np.mean(np.array(v)*epoch_weight) for k, v in sorted(memory_module.items(), key=lambda item: item[1])}

                mu = np.mean(list(instance_mean.values()))
                sd = np.std(list(instance_mean.values()))

                gaussian_norm = norm(mu, sd)

                fp_bound = mu+self.fp*sd

                fp_index = [k for k in instance_mean.keys() if instance_mean[k]>=fp_bound]
                print("identify ok")

            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            index = 0
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                out = self.model(input)
                loss = criterion(out, input)
                loss_per_timestamp = loss.mean(dim=2)  
                all_loss = (loss_per_timestamp)*weights

                pos_loss = all_loss.sum(dim=1)

                index_update = index
                for j in range(batch_size):
                    if index >= len(memory_module):
                        print("index out of range")
                        break
                    memory_module[index].append(pos_loss[j].cpu().item())
                    index = index + 1


                if epoch >= warm_epochs:
                    l = pos_loss.detach().cpu()
                    w = gaussian_norm.pdf(l)
                    for j in range(batch_size):
                        _id = index_update

                        if _id in fp_index:

                            pos_loss[j] *= w[j]
                        index_update = index_update + 1


                loss = torch.mean(pos_loss) 

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            if epoch >= self.num_epochs-1:
                vali_loss = self.vali(self.test_loader)
                early_stopping(vali_loss, self.model, path)
            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            if early_stopping.early_stop:
                break
            if self.optimizer.param_groups[0]['lr'] > 1e-5:
                adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            print("Current learning rate:", self.optimizer.param_groups[0]['lr'])

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path) + '/checkpoint.pth')))
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            out = self.model(input)
            score = torch.mean(self.anomaly_criterion(input,out), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        print("1.finish")

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            out = self.model(input)
            score = torch.mean(self.anomaly_criterion(input, out), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        pred = (test_energy > thresh).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        
        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        return accuracy, precision, recall, f_score

