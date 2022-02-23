import os
import numpy as np
import math
import networkx as nx
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def read_file(_path, delim):
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def anorm(p1,p2):
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)

def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]
    #print('V_trgt[:, :, 0]:',V_trgt[:, :, 0])
    # print('V_pred[:, :, 0]:',V_pred[:, :, 0])
    # print('normx:', normx)
    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    # print('sx:', sx)
    corr = torch.tanh(V_pred[:, :, 4])  # corr
    # print('corr:', corr)
    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    # print('negRhp:', negRho)
    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # print('result:', result)
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    # print('denom:', denom)
    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    # print('result:', result)
    result = torch.mean(result)
    # print('result:', result)
    return result


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]  # number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return sum_all / All


# Dataset preprocessing
#==============================================================
#==============================================================

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_name, obs_len, pred_len, skip, threshold=0.002,
        min_ves =1, delim=' ',norm_lap_matr = True):

        super(TrajectoryDataset, self).__init__()
        self.max_ves_in_frame = 0
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip = skip
        self.data_dir = 'stgcnn/data/'+data_name
        self.processed_data = 'stgcnn/data/processed_data/'+data_name

        num_ves_in_seq = []
        seq_list = []
        seq_list_rel = []

        data_path = self.data_dir+'.txt'

        cleaned = read_file(data_path, delim)
        # cleaned: [time,mmsi,lat,long,sog,cog]
        frames = np.unique(cleaned[:, 0]).tolist()  # 取出unique
        frame_data = []
        for frame in frames:
            frame_data.append(cleaned[frame == cleaned[:, 0], :]) #取出每个时刻对应的数据
        num_sequences = int(
            math.ceil((len(frames) - self.seq_len + 1) / skip))  # 向上取整算序列的长度，计算有多少个序列

        for idx in range(0, num_sequences * self.skip + 1, skip):
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + self.seq_len], axis=0)
            ves_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 每个序列里有哪些船共同存在
            self.max_ves_in_frame = max(self.max_ves_in_frame, len(ves_in_curr_seq))
            curr_seq_rel = np.zeros((len(ves_in_curr_seq), 2, self.seq_len)) #建一个相对位置的框架
            curr_seq = np.zeros((len(ves_in_curr_seq), 2, self.seq_len))  # 建一个绝对位置的框架

            num_ves_considered = 0
            _non_linear_ves = []
            for _, ves_id in enumerate(ves_in_curr_seq):  # 对每一条船来说
                curr_ves_seq = curr_seq_data[curr_seq_data[:, 1] ==  # 取出当前序列当前船的xy坐标
                                     ves_id, :]  # [time,mmsi,lat,long,sog,cog]
                curr_ves_seq = np.around(curr_ves_seq, decimals=4)
                pad_front = frames.index(curr_ves_seq[0, 0]) - idx  # 首帧
                pad_end = frames.index(curr_ves_seq[-1, 0]) - idx + 1  # 尾帧
                if pad_end - pad_front != self.seq_len:
                    continue
                curr_ves_seq = np.transpose(curr_ves_seq[:, 2:])  # 取第二列之后的xy坐标，并转置成
                curr_ves_seq = curr_ves_seq  # [4, 180]

                # Make coordinates relative
                rel_curr_ves_seq = np.zeros(curr_ves_seq.shape)  # [4,180]
                rel_curr_ves_seq[:, 1:] = \
                    curr_ves_seq[:, 1:] - curr_ves_seq[:, :-1]  # 179个下一帧减去179个前一帧求相对变化量
                _idx = num_ves_considered
                curr_seq[_idx, :, pad_front:pad_end] = curr_ves_seq
                # curr_seq [24,4,180]
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ves_seq
                num_ves_considered += 1

            if num_ves_considered > min_ves:
                num_ves_in_seq.append(num_ves_considered)
                seq_list.append(curr_seq[:num_ves_considered])
                seq_list_rel.append(curr_seq_rel[:num_ves_considered])
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_ves_in_seq).tolist()

        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        graph_data_path = self.data_dir+'_graph_data.dat'
        if not os.path.exists(graph_data_path):
            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)
                start, end = self.seq_start_end[ss]
                v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
                # 输入seq 和seq的相对位移输出v和A
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())

            pbar.close()
            graph_data = {'v_obs': self.v_obs, 'A_obs': self.A_obs, 'v_pred': self.v_pred, 'A_pred': self.A_pred}
            torch.save(graph_data, graph_data_path)

        else:
            graph_data = torch.load(graph_data_path)
            self.v_obs, self.A_obs, self.v_pred, self.A_pred = graph_data['v_obs'], graph_data['A_obs'], graph_data[
                'v_pred'], graph_data['A_pred']
            print('Loaded pre-processed graph data at {:s}.'.format(graph_data_path))

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
