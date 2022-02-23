import pickle
import glob
import torch.distributions.multivariate_normal as torchdist
import torch
from metrics import *
import copy
from model import *
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(KSTEPS=20):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0
    for batch in loader_test:
        step+=1
        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
         V_obs,A_obs,V_tr,A_tr = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())  # batch,feat,seq,node  [1, 5, 30, 2]
        V_pred = V_pred.permute(0, 2, 3, 1)  # batch, seq, node, feat   [1, 30, 2, 5]

        V_tr = V_tr.squeeze()  # [30,2,2]
        V_pred = V_pred.squeeze() # [30,2,5]
        num_of_objs = obs_traj_rel.shape[1]  # seq, node, feat, pred_len
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)  # seq(12),node, 2,2
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.cpu().data.numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.cpu().data.numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.cpu().data.numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.cpu().data.numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            # V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.cpu().data.numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


paths = ['./checkpoint/stgcnn']
KSTEPS = 20

print("*" * 50)
print('Number of samples:', KSTEPS)
print("*" * 50)

for feta in range(len(paths)):
    ade_ls = []
    fde_ls = []
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f:
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f:
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len

        dset_test = TrajectoryDataset(
                data_name='test',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=3,norm_lap_matr=True)

        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=1)



        #Defining the model
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
                              input_feat=args.input_size,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).to(device)
        model.load_state_dict(torch.load(model_path,map_location=device))


        ade_ =999999
        fde_ =999999
        print("Testing ....")
        ad,fd,raw_data_dic_= test()
        ade_= min(ade_,ad)
        fde_ =min(fde_,fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        print("ADE:",ade_," FDE:",fde_)




    print("*"*50)