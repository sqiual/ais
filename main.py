import pandas as pd
import os
import math
import torch
from tqdm import tqdm
import numpy as np
import networkx as nx
import argparse
from metrics import *
from model import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle

# parameter setting
#==============================================================
#==============================================================
# min_voy = 20  # minimum voyage duration in minutes
# min_speed = 1
# re_sample_intvl = 10
# tim_interval = np.arange(0, 3600*24, 10)
# min_ves = 1

parser = argparse.ArgumentParser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=30)
parser.add_argument('--pred_seq_len', type=int, default=30)

# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--stgcnn', default='tag',
                    help='personal tag for the model ')
parser.add_argument('--skip', default=3,
                    help='')

args = parser.parse_args()

skip = args.skip


# Training
#==============================================================
#==============================================================

print('*'*30)
print("Training initiating....")

# def graph_loss(V_pred,V_target):
#     return bivariate_loss(V_pred,V_target)

obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
skip = args.skip

dset_train = TrajectoryDataset(
        data_name='train',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=skip,norm_lap_matr=True)

loader_train = DataLoader(
        dset_train,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=0)

dset_val = TrajectoryDataset(
        data_name='val',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=skip,norm_lap_matr=True)

loader_val = DataLoader(
        dset_val,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=1)

model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
input_feat=args.input_size, output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).to(device)

#Training settings

optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

checkpoint_dir = './checkpoint/' + args.tag + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

def train(epoch):
    global metrics,loader_train
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1


    for cnt,batch in enumerate(loader_train):
        batch_count+=1

        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
                V_obs,A_obs,V_tr,A_tr = batch

        optimizer.zero_grad()
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)  # [128,2,8,57]
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)  # [128,12,57,5]
        V_tr = V_tr.squeeze()  # [128,8,57,2]
        # ================
        # ================
        # ================
        V_pred = V_pred.squeeze()
        # print('V_pred', V_pred)
        # print('V_tr', V_tr)

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = bivariate_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / batch_count)


def vald(epoch):
    global metrics, loader_val, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        V_obs, A_obs, V_tr, A_tr = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = bivariate_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


print('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    vald(epoch)
    if args.use_lrschd:
        scheduler.step()

    print('*' * 30)
    print('Epoch:', args.tag, ":", epoch)
    for k, v in metrics.items():
        if len(v) > 0:
            print(k, v[-1])

    print(constant_metrics)
    print('*' * 30)

    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)


