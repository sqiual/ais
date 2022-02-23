import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels, # 4 记住这里的in channel是4
                 out_channels, # 5
                 kernel_size, # 8
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        # self.kernel_size = kernel_size
        # dialation 3*3 卷积间隔1 相当于5*5效果
        self.conv = nn.Conv2d(
            in_channels,
            out_channels, # 5 channels 5个卷积核
            kernel_size=(t_kernel_size, 1), # [1,1]
            padding=(t_padding, 0), # [0,0]
            stride=(t_stride, 1), # [1,1]
            dilation=(t_dilation, 1),  # [1,1]
            bias=bias) # [8,57] 卷积核为1 维度不变

    def forward(self, x, A): #A[8,57,57]
        # ++++++++++++++++++++++++
        # assert A.size(0) == self.kernel_size
        x = self.conv(x)
        # [1,5,8,57],[8,57,57] 5个通道，每个通道包含两种人[5*2]，每种人有8个时刻[5*2*8],一种人有57个[5*2*8*57]
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class social_stgcnn(nn.Module):
    # input feature我已经从2改成了4
    def __init__(self, n_stgcnn, n_txpcnn, input_feat, output_feat,
                 seq_len, pred_seq_len, kernel_size):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, kernel_size, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a):

        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v, a


# ===========================================
#             Moxing: RNN LSTM
# ===========================================

class LSTM(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.output_size = output_size
        self.lstm1= nn.LSTMCell(self.input_size,self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, self.output_size)

    def forward(self, x, future=0):
        n_samples = x.shape[1]
        n_timestep = x.shape[0]
        outputs = torch.zeros((future, n_samples, self.output_size)).to(device)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        # V_obs.shape [30, 54, 2]

        for input_t in range(n_timestep):
            # V_obs = [30, 54, 2]
            h_t, c_t = self.lstm1(x[input_t,:,:].squeeze(), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            tmp_out = h_t2

        for i in range(future):
            h_t, c_t = self.lstm2(tmp_out, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs[i, :] = output
            # outputs shape [30, 52, 5]
        return outputs