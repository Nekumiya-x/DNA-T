
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np

from models import DNA
from utils.util import EarlyStopping, calculate_performance

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DNA': DNA,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self):
        self.train_loader, self.valid_loader, self.test_loader = data_provider(self.args)
        return


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def valid(self,valid_loader,criterion):
        total_loss = []
        y_gts = np.array([]).reshape(0)
        y_preds = np.array([]).reshape(0)
        y_scores = np.array([]).reshape(0)

        self.model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                batch = tuple(b.float().to(self.device) for b in batch)
                observation, time, mask, delta, label, real_len = batch

                y_gts = np.hstack([y_gts, label.to('cpu').detach().numpy().flatten()])

                B, L, _ = observation.shape
                attn_mask = torch.ones((B, L), device=observation.device)
                for i in range(B):
                    attn_mask[i, :int(real_len[i])] = 0
                attn_mask = attn_mask.bool()


                outputs = self.model(observation, mask, time, delta,attn_mask=attn_mask)
                label = label.long()
                loss = criterion(outputs,label)

                total_loss.append(loss.item())

                y_score = torch.softmax(outputs, dim=-1).to('cpu').detach().numpy()[:, 1]
                y_pred = torch.softmax(outputs, dim=-1).to('cpu').detach().numpy().argmax(1)
                y_preds = np.hstack([y_preds, y_pred])
                y_scores = np.hstack([y_scores, y_score])

        total_loss = np.average(total_loss)
        acc, auc, auprc, prec, recall, F1, balacc, sen, spec = calculate_performance(y_gts, y_scores, y_preds)
        self.model.train()
        return total_loss, acc, auc, auprc, prec, recall, F1, balacc, sen, spec

    def train(self, setting):

        self._get_data()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_loader = self.train_loader
        valid_loader = self.valid_loader

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = optim.lr_scheduler.StepLR(model_optim, step_size=10, gamma=self.args.lr_ratio)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            y_gts = np.array([]).reshape(0)
            y_preds = np.array([]).reshape(0)
            y_scores = np.array([]).reshape(0)
            train_loss = []

            self.model.train()
            for batch in train_loader:
                batch = tuple(b.float().to(self.device) for b in batch)
                observation, time, mask, delta, label, real_len = batch

                y_gts = np.hstack([y_gts, label.to('cpu').detach().numpy().flatten()])

                B, L, _ = observation.shape
                attn_mask = torch.ones((B, L), device=observation.device)
                for i in range(B):
                    attn_mask[i, :int(real_len[i])] = 0
                attn_mask = attn_mask.bool()

                model_optim.zero_grad()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(observation, mask, time, delta, attn_mask=attn_mask)
                else:
                    outputs = self.model(observation, mask, time, delta, attn_mask=attn_mask)
                loss = criterion(outputs,label.long())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())
                y_score = torch.softmax(outputs, dim=-1).to('cpu').detach().numpy()[:,1]
                y_pred = torch.softmax(outputs, dim=-1).to('cpu').detach().numpy().argmax(1)
                y_preds = np.hstack([y_preds, y_pred])
                y_scores = np.hstack([y_scores, y_score])

            scheduler.step()

            # Results per epoch: loss, acc, auc, auprc, prec, recall, F1, balacc, sen, spec
            train_loss = np.average(train_loss)
            train_acc, train_auc, train_auprc, train_prec, train_recall, train_F1, train_balacc, train_sen, train_spec = calculate_performance(y_gts, y_scores, y_preds)
            valid_loss, valid_acc, valid_auc, valid_auprc, valid_prec, valid_recall, valid_F1, valid_balacc, valid_sen, valid_spec = self.valid(valid_loader, criterion)
            print('Epoch: {}  train loss: {:.5}  train acc: {:.5} train auc: {:.5}  train auprc: {:.5}  train prec: {:.5}  train recall: {:.5}  train F1: {:.5}  train balacc: {:.5}  train sen: {:.5}  train spec: {:.5}'.format(epoch, train_loss, train_acc, train_auc, train_auprc, train_prec, train_recall, train_F1, train_balacc, train_sen, train_spec))
            print('Epoch: {}  valid loss: {:.5}  valid acc: {:.5} valid auc: {:.5}  valid auprc: {:.5}  valid prec: {:.5}  valid recall: {:.5}  valid F1: {:.5}  valid balacc: {:.5}  valid sen: {:.5}  valid spec: {:.5}'.format(epoch, valid_loss, valid_acc, valid_auc, valid_auprc, valid_prec, valid_recall, valid_F1, valid_balacc, valid_sen, valid_spec))

            early_stopping(valid_auc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                print('Breaking because of Early stopping')
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=False):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        y_gts = np.array([]).reshape(0)
        y_preds = np.array([]).reshape(0)
        y_scores = np.array([]).reshape(0)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                batch = tuple(b.float().to(self.device) for b in batch)
                observation, time, mask, delta, label, real_len = batch

                y_gts = np.hstack([y_gts, label.to('cpu').detach().numpy().flatten()])

                B, L, _ = observation.shape
                attn_mask = torch.ones((B, L), device=observation.device)
                for i in range(B):
                    attn_mask[i, :int(real_len[i])] = 0
                attn_mask = attn_mask.bool()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(observation, mask, time, delta,attn_mask=attn_mask)
                else:
                    outputs = self.model(observation, mask, time, delta,attn_mask=attn_mask)

                y_score = torch.softmax(outputs, dim=-1).to('cpu').detach().numpy()[:, 1]
                y_pred = torch.softmax(outputs, dim=-1).to('cpu').detach().numpy().argmax(1)
                y_preds = np.hstack([y_preds, y_pred])
                y_scores = np.hstack([y_scores, y_score])

        acc, auc, auprc, prec, recall, F1, balacc, sen, spec = calculate_performance(y_gts, y_scores, y_preds)
        print("Final Test")
        print('acc:{}'.format(acc))
        print('auc:{}'.format(auc))
        print('auprc:{}'.format(auprc))

        return acc, auc, auprc


