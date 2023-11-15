import argparse
import pandas as pd
import numpy as np

import os
import gc
import pickle

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--l', type=str)
args = parser.parse_args()

l = args.l
filename = './pp_data/data-reg-2d-{}-pp.csv'.format(l)

data = pd.read_csv(filename)

valid_idx = 0

model_path = './models/{}/'.format(l)
if not os.path.exists(model_path):
    os.makedirs(model_path)

from sklearn.model_selection import StratifiedKFold, KFold

data['fold'] = 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for idx, (train_idx, valid_idx) in enumerate(skf.split(data, data['N_seg'])):
    data['fold'].iloc[valid_idx] = idx

valid_idx = 0

train_df = data[data['fold']!=valid_idx].reset_index(drop=True)
valid_df = data[data['fold']==valid_idx].reset_index(drop=True)
print('There are {} samples in the training set.'.format(len(train_df)))
print('There are {} samples in the validation set.'.format(len(valid_df)))

import torch
from torch import nn, optim
from torch.nn import functional as F
#from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
#from torch.nn.utils.rnn import *
from tqdm.auto import tqdm

class AnDiDataset(Dataset):
    def __init__(self, df, label=True):
        self.df = df#.copy()
        self.label = label
        
    def __getitem__(self, index): 
        data_seq_x = torch.Tensor([float(i) for i in self.df['pos_x'].iloc[index].split(',')])
        data_seq_y = torch.Tensor([float(i) for i in self.df['pos_y'].iloc[index].split(',')])
        data_seq = torch.stack((data_seq_x, data_seq_y), dim = 0)
        if self.label:            
            target = torch.Tensor([round(float(i),2) for i in self.df['mask'].iloc[index].split(' ')])
        else:
            target = 0
        return data_seq, torch.Tensor(target)
    
    def __len__(self):
        return len(self.df)

train_loader = DataLoader(AnDiDataset(train_df), batch_size=512, shuffle=True, num_workers=16)
valid_loader = DataLoader(AnDiDataset(valid_df), batch_size=512, shuffle=True, num_workers=16)

#https://github.com/odie2630463/WaveNet/blob/master/model.py#L70
class AnDi_Wave(nn.Module):
    def __init__(self, input_dim, filters, kernel_size, dilation_depth):
        super().__init__()
        self.dilation_depth =  dilation_depth
        
        self.dilations = [2**i for i in range(dilation_depth)]
        self.conv1d_tanh = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, 
                                                    padding=int((dilation*(kernel_size-1))/2), dilation=dilation) 
                                                    for dilation in self.dilations])
        self.conv1d_sigm = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, 
                                                    padding=int((dilation*(kernel_size-1))/2), dilation=dilation) 
                                                    for dilation in self.dilations]) 
        self.conv1d_0 = nn.Conv1d(in_channels=input_dim, out_channels=filters, 
                                  kernel_size=kernel_size, padding=1)
        self.conv1d_1 = nn.Conv1d(in_channels=filters, out_channels=filters, 
                                  kernel_size=1, padding=0)
        self.post = nn.Sequential(nn.BatchNorm1d(filters), nn.Dropout(0.1))
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        # WaveNet Block
        x = self.conv1d_0(x)
        res_x = x
        
        for i in range(self.dilation_depth):
            tahn_out = torch.tanh(self.conv1d_tanh[i](x))
            sigm_out = torch.sigmoid(self.conv1d_sigm[i](x))
            x = tahn_out * sigm_out
            x = self.conv1d_1(x)
            res_x = res_x + x
        
        x = self.post(res_x)
        out = self.pool(x)
        
        return out, x

class AnDiModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dilation_depth):
        super().__init__()
        # Encoder
        self.encoder0 = AnDi_Wave(input_dim, hidden_dim, kernel_size, dilation_depth)
        self.encoder1 = AnDi_Wave(hidden_dim, hidden_dim*2, kernel_size, dilation_depth)
        self.encoder2 = AnDi_Wave(hidden_dim*2, hidden_dim*4, kernel_size, dilation_depth)
        self.encoder3 = AnDi_Wave(hidden_dim*4, hidden_dim*8, kernel_size, dilation_depth)
        self.encoder4 = AnDi_Wave(hidden_dim*8, hidden_dim*16, kernel_size, dilation_depth)
        
        # Decoder
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(hidden_dim*16, hidden_dim*8, 1))
        self.decoder3 = AnDi_Wave(hidden_dim*16, hidden_dim*8, kernel_size, dilation_depth)
        
        self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(hidden_dim*8, hidden_dim*4, 1))
        self.decoder2 = AnDi_Wave(hidden_dim*8, hidden_dim*4, kernel_size, dilation_depth)
        
        self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(hidden_dim*4, hidden_dim*2, 1))
        self.decoder1 = AnDi_Wave(hidden_dim*4, hidden_dim*2, kernel_size, dilation_depth)
        
        self.upconv0 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        self.decoder0 = AnDi_Wave(hidden_dim*2, hidden_dim, kernel_size, dilation_depth)
        
        # last conv
        self.conv = nn.Conv1d(hidden_dim, 1, 1)
    
    def forward(self, x):
        # Encoder
        x, f0 = self.encoder0(x)
        x, f1 = self.encoder1(x)
        x, f2 = self.encoder2(x)
        x, f3 = self.encoder3(x)
        _, x  = self.encoder4(x)
        
        # Decoder
        x = self.upconv3(x)
        x = self.cat(f3, x)
        _, x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = self.cat(f2, x)
        _, x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = self.cat(f1, x)
        _, x = self.decoder1(x)
        
        x = self.upconv0(x)
        x = self.cat(f0, x)
        _, x = self.decoder0(x)
        
        # last conv
        out = self.conv(x)
        
        return out
    
    def cat(self, f_left, f_right):
        if f_right.shape[-1] != f_left.shape[-1]:
            f_right = F.pad(f_right, pad=(0, 1), value=0.0)
        return torch.cat((f_left, f_right), dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = AnDiModel(2, 64, 3, 5).to(device)

criterion = nn.MSELoss()
metric = nn.L1Loss()

lr = 2e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

def evaluate(step, history=None): 
    #model.eval() 
    valid_loss = 0.
    mae_metric = 0.
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for batch_idx, (seq_batch, label_batch) in enumerate(valid_loader):
            all_targets.append(label_batch.numpy().copy())
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)
            
            output = model(seq_batch).squeeze()
            loss = criterion(output, label_batch)
            mae = metric(output, label_batch)
            valid_loss += loss.data
            mae_metric += mae.data
            all_predictions.append(output.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    valid_loss /= (batch_idx+1)
    mae_metric /= (batch_idx+1)
    
    if history is not None:
        history.loc[step, 'valid_loss'] = valid_loss.cpu().numpy()
    
    valid_status = 'Step: {}\tLR: {:.6f}\tValid Loss: {:.4f}\tValid MAE: {:.4f}'.format(
        step, optimizer.state_dict()['param_groups'][0]['lr'], valid_loss, mae_metric)
    print(valid_status)
    with open(model_path+'log.txt', 'a+') as f:
        f.write(valid_status+'\n')
        f.close()
    
    return valid_loss, mae_metric

history_train = pd.DataFrame()
history_valid = pd.DataFrame()

n_epochs = 100
init_epoch = 0
max_lr_changes = 2
valid_losses = []
mae_metrics = []
lr_reset_epoch = init_epoch
patience = 100000
lr_changes = 0
best_valid_loss = 1000.
best_valid_metric = 1000.

eval_step = 1000
current_step = 0

for epoch in range(init_epoch, n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    
    model.train() 
    t = tqdm(train_loader)
    
    for batch_idx, (seq_batch, label_batch) in enumerate(t):
        
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        output = model(seq_batch).squeeze()
        loss = criterion(output, label_batch)
        t.set_description(f'train_loss (l={loss:.4f})')
        
        if history_train is not None:
            history_train.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()    
        optimizer.step()
        
        current_step += 1
        if current_step % eval_step == 0:
            #print(current_step)
            model.eval() 
            valid_loss, mae_metric = evaluate(current_step, history_valid)
            valid_losses.append(valid_loss)
            mae_metrics.append(mae_metric)
            
            if mae_metric < best_valid_metric:
                best_valid_metric = mae_metric
                best_pth_path = model_path+'step{}-{}.pth'.\
                format(current_step, round(mae_metric.cpu().numpy().tolist(), 5))
                if best_valid_metric < 0.2:
                    torch.save(model.state_dict(), best_pth_path)
                
            #torch.save(model.state_dict(), model_path+'step{}.pth'.format(current_step))
            model.train()

history_train.to_csv(model_path+'history_train.csv')
history_valid.to_csv(model_path+'history_valid.csv')
