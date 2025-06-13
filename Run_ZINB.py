import h5py as h5
import sys

from lib import preprocessH5
from lib.preprocessH5 import pre_normalize, prepro
from model.GNN import GraphConvolution

sys.path.append("..")
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import scanpy as sc
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
# from sknetwork.clustering import Louvain, BiLouvain, modularity, bimodularity
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
import sklearn
from sklearn import metrics
import pandas as pd
import time
from numpy.random import seed
from lib.utils import buildGraphNN, sparse_mx_to_torch_sparse_tensor
from lib.utils import normalize as normalized
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


# def _nelem(x):
#    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
#    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)
#
# def _reduce_mean(x):
#    nelem = _nelem(x)
#    x = _nan2zero(x)
#    return tf.divide(tf.reduce_sum(x), nelem)

def NB(theta, y_true, y_pred, mask=False, debug=False, mean=True):
    eps = 1e-10
    scale_factor = 1.0

    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))

    final = t1 + t2
    final = _nan2inf(final)
    if mean:
        final = torch.mean(final)
    else:
        final = torch.sum(final)

    return final


def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mask=False, debug=False, mean=True):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=True, debug=debug) - torch.log(1.0 - pi + eps)

    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * torch.pow(pi, 2)
    result += ridge
    if mean:
        result = torch.mean(result)
    else:
        result = torch.sum(result)

    result = _nan2inf(result)

    return result


def main(dataset_name, args):
    setup_seed(2)
    # seed(2)
    # torch.manual_seed(6)

    ## Note: this program need GPU to run ########

    #######  first change the data and label file path in your computer ########

    ### In the input, each row is a cell, and each column is a gene  #######
    # Data_file = "your data file path'
    # input = np.loadtxt(data_file, sep='\t')
    # input = np.array(input)
    # label_path = "your label file path"
    # y_true = np.loadtxt(label_path, delimiter='\t')

    dataPath = "/root/autodl-tmp/scSAG2E/DeepCI_ZINB/data"
    # # # dataset = "Klein"
    dataset = dataset_name
    filename = dataPath + "/" + dataset + "/data.h5"
    input, y_true = prepro(filename)

    args.Cell_BATCH_SIZE = input.shape[0]
    args.Gene_BATCH_SIZE = input.shape[1]
    args.Cell_pretrain_BATCH_SIZE = input.shape[0]
    args.Gene_pretrain_BATCH_SIZE = input.shape[1]

    # df = pd.read_csv('/root/autodl-tmp/scCFIB/data/Biase/Biase_Normalized.tsv', sep='\t')
    # df2 = pd.read_csv('/root/autodl-tmp/scCFIB/data/Biase/subtype.ann', sep='\t')
    # y_true = df2["label"]
    # df.drop(columns=df.columns[0], inplace=True)
    # input = df.T
    # adata = sc.AnnData(input)
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=3000,
    #                             subset=True)
    # input = adata.X.astype(np.float32)

    # input = pd.read_csv("/root/autodl-tmp/scSAG2E/DeepCI_ZINB/GSE66688/GSE66688_data.txt",sep=' ')
    # input = input.values
    # input = input.astype('float32')
    # y_true = pd.read_csv("/root/autodl-tmp/scSAG2E/DeepCI_ZINB/GSE66688/GSE66688_lableb.txt",sep=' ')
    # y_true = y_true.T
    # y_true = y_true.values
    # y_true = y_true.flatten()
    # for i in range(len(y_true)):
    #     y_true[i] = y_true[i] - 1
    # adata = sc.AnnData(input)
    # adata = pre_normalize(adata, copy=True, highly_genes=3000, size_factors=False, normalize_input=True,
    #                       logtrans_input=True)
    #
    # input = adata.X.astype(np.float32)

    ##### Change the path to save the pretrained model in your computer #########
    save_path1 = 'save/' + dataset + '_cellautoencoder.pkl'
    save_path2 = 'save/' + dataset + '_geneautoencoder.pkl'

    ## buildnetwork
    def buildNetwork_cell(layers, activation="relu", dropout=0):
        net = []
        for i in range(1, len(layers)):
            net.append(nn.Linear(layers[i - 1], layers[i]))
            if activation == "relu":
                net.append(nn.ReLU())
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        return nn.Sequential(*net)

    class CellAutoEncoder(torch.nn.Module):

        def __init__(self, cellInput_dim=512,
                     cellEncodeLayer=[400], cellDecodeLayer=[400], activation="relu", dropout=0):
            super(self.__class__, self).__init__()
            # self.z_dim = z_dim
            # self.cellLayers = [cellInput_dim] + cellEncodeLayer + [z_dim]
            self.activation = activation
            self.dropout = dropout
            self.cellencoder = buildNetwork_cell([cellInput_dim] + cellEncodeLayer, activation=activation,
                                                 dropout=dropout)
            self.celldecoder = buildNetwork_cell(cellDecodeLayer, activation=activation, dropout=dropout)
            # self._enc_mu = nn.Linear(cellEncodeLayer[-1], z_dim)
            self._dec = nn.Linear(cellDecodeLayer[-1], cellInput_dim)
            self.fc_pi = nn.Linear(cellDecodeLayer[-1], cellInput_dim)
            self.fc_disp = nn.Linear(cellDecodeLayer[-1], cellInput_dim)
            self.fc_mu = nn.Linear(cellDecodeLayer[-1], cellInput_dim)

            # self.n_clusters = n_clusters
            # self.alpha = alpha
            # self.gamma = gamma
            # self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

        def forward(self, celltogene):
            z = self.cellencoder(celltogene)
            # z = self._enc_mu(h)
            h = self.celldecoder(z)
            xrecon = self._dec(h)

            pi = self.fc_pi(h)
            self.pi = torch.sigmoid(pi)
            disp = self.fc_disp(h)
            self.disp = torch.clamp(F.softplus(disp), min=0, max=1e4)
            mean = self.fc_mu(h)
            self.mean = torch.clamp(F.relu(mean), min=0, max=1e6)
            self.output = self.mean
            self.likelihood_loss = ZINB(self.pi, self.disp, celltogene, self.output, ridge_lambda=1.0, mean=True)

            # compute q -> NxK
            # q = self.soft_assign(z)
            # return xrecon, self.likelihood_loss
            return xrecon, self.likelihood_loss, z

    # def buildNetwork_gene(layers, activation="relu", dropout=0):
    #     net = []
    #     for i in range(1, len(layers)):
    #         net.append(nn.Linear(layers[i - 1], layers[i]))
    #         if activation == "relu":
    #             net.append(nn.ReLU())
    #         elif activation == "sigmoid":
    #             net.append(nn.Sigmoid())
    #         if dropout > 0:
    #             net.append(nn.Dropout(dropout))
    #     return nn.Sequential(*net)

    def buildNetwork_gene(layers, activation="relu", dropout=0):
        net = []
        for i in range(1, len(layers)):
            net.append(GraphConvolution(layers[i - 1], layers[i]))
            if activation == "relu":
                net.append(nn.ReLU())
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        return nn.Sequential(*net)

    class GeneAutoEncoder(torch.nn.Module):

        def __init__(self, geneInput_dim=784,
                     geneEncodeLayer=[400], geneDecodeLayer=[400], activation="relu", dropout=0):
            super(self.__class__, self).__init__()
            # self.gene_emd_dim = gene_emd_dim
            # self.geneLayers = [geneInput_dim] + geneEncodeLayer + [gene_emd_dim]
            self.activation = activation
            self.dropout = dropout

            self.geneencoder = buildNetwork_gene([geneInput_dim] + geneEncodeLayer, activation=activation,
                                                 dropout=dropout)
            self.genedecoder = buildNetwork_gene(geneDecodeLayer, activation=activation, dropout=dropout)
            # self._enc_mu_gene = nn.Linear(geneEncodeLayer[-1], gene_emd_dim)
            self._dec_gene = nn.Linear(geneDecodeLayer[-1], geneInput_dim)

        def forward(self, genetocell):
            h = self.geneencoder(torch.tensor(genetocell))
            # z = self._enc_mu(h)
            h = self.genedecoder(h)
            xrecon = self._dec_gene(h)
            # xrecon = torch.mm(h.T, h)

            # compute q -> NxK
            # q = self.soft_assign(z)
            return xrecon

    def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
        if filter_min_counts:
            sc.pp.filter_genes(adata, min_counts=1)
            sc.pp.filter_cells(adata, min_counts=1)

        if size_factors or normalize_input or logtrans_input:
            adata.raw = adata.copy()
        else:
            adata.raw = adata

        if size_factors:
            sc.pp.normalize_per_cell(adata)
            adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        else:
            adata.obs['size_factors'] = 1.0

        if logtrans_input:
            sc.pp.log1p(adata)

        if normalize_input:
            sc.pp.scale(adata)

        return adata

    def acc(y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        # from sklearn.utils.linear_assignment_ import linear_assignment
        from scipy.optimize import linear_sum_assignment as linear_assignment
        ind = linear_assignment(w.max() - w)

        # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

        b = [None] * len(ind[0])
        for k in range(len(ind[0])):
            i = ind[0][k]
            j = ind[1][k]
            b[k] = w[i, j]
        return sum(b) * 1.0 / y_pred.size

    ### the preprocessing can be changed accorrding your data #########
    # 筛选基因：将基因中非0的个数小于细胞个数0.01倍的基因剔除
    # columns = (input != 0).sum(0)
    # input = input[:, columns > np.ceil(input.shape[0] * 0.01)]

    adata = sc.AnnData(input)
    adata.obs['Group'] = y_true

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=False,
                      logtrans_input=True)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000,
                                subset=True)
    input = torch.from_numpy(adata.X).float()

    #########

    # Hyper Parameters
    # LR_pretrain_cell = 0.0001 #pollen：0.001
    # LR_pretrain_gene = 0.0001 #pollen：0.001
    # Cell_BATCH_SIZE = 1024 #512
    # Gene_BATCH_SIZE = 1024
    # EPOCH_pretrain = 200  # 400
    # tol_cell = 0.001
    # tol_gene = 0.001

    # Hyper Parameters
    LR_pretrain_cell = args.LR_pretrain_cell #pollen：0.001
    LR_pretrain_gene = args.LR_pretrain_gene #pollen：0.001
    Cell_BATCH_SIZE = args.Cell_pretrain_BATCH_SIZE #512
    Gene_BATCH_SIZE = args.Gene_pretrain_BATCH_SIZE
    EPOCH_pretrain = args.EPOCH_pretrain  # 400
    tol_cell = args.tol_pretrain_cell
    tol_gene = args.tol_pretrain_gene

    # 生成细胞视图的数据
    loader_cell = Data.DataLoader(
        dataset=input,  # torch TensorDataset format
        batch_size=Cell_BATCH_SIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=2,  # subprocesses for loading data
    )

    # 生成基因视图的数据
    loader_gene = Data.DataLoader(
        dataset=torch.t(input),  # torch TensorDataset format
        batch_size=Gene_BATCH_SIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=2,  # subprocesses for loading data
    )

    from torch.autograd import Variable

    cellautoencoder = CellAutoEncoder(input.shape[1], cellEncodeLayer=[64, 32], cellDecodeLayer=[32, 64],
                                      activation="relu",
                                      dropout=0.1).float()
    print(cellautoencoder)

    # !!!!!!!! Change in here !!!!!!!!! #
    cellautoencoder  # Moves all model parameters and buffers to the GPU.

    celloptimizer = torch.optim.Adam(cellautoencoder.parameters(), lr=LR_pretrain_cell)
    cellloss_func = nn.MSELoss()

    start = time.time()
    for epoch in range(EPOCH_pretrain):  # train entire dataset 3 times
        for step_cell, batch_cell_cpu in enumerate(loader_cell):  # for each training step
            print('Epoch: ', epoch, '| Step_cell: ', step_cell)
            # !!!!!!!! Change in here !!!!!!!!! #
            batch_cell = batch_cell_cpu  # Tensor on GPU
            # celldecoded = cellautoencoder(batch_cell,noise = False,mean=0, stddev=0.01)
            celldecoded, likelihood_loss, z_emb = cellautoencoder(batch_cell)
            cellloss = cellloss_func(celldecoded, batch_cell)  # mean square error
            # cellloss = likelihood_loss
            celloptimizer.zero_grad()  # clear gradients for this training step
            cellloss.backward()  # backpropagation, compute gradients
            celloptimizer.step()  # apply gradients

            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCH_pretrain, cellloss.item()))
        if epoch > 0 and cellloss.item() < tol_cell:
            print('cellloss', cellloss.item(), '< tol ', tol_cell)
            print('Reached tolerance threshold. Stopping training.')
            break

    end = time.time()
    cell_time = end - start
    print('cell time:', cell_time)

    geneautoencoder = GeneAutoEncoder(input.shape[0], geneEncodeLayer=[64, 32], geneDecodeLayer=[32, 64],
                                      activation="relu",
                                      dropout=0.1).float()
    print(geneautoencoder)

    geneautoencoder

    geneoptimizer = torch.optim.Adam(geneautoencoder.parameters(), lr=LR_pretrain_gene)
    geneloss_func = nn.MSELoss()

    start = time.time()
    for epoch in range(EPOCH_pretrain):  # train entire dataset 3 times
        for step_gene, batch_gene_cpu in enumerate(loader_gene):
            print('Epoch: ', epoch, '| Step_gene: ', step_gene)
            # !!!!!!!! Change in here !!!!!!!!! #
            batch_gene = batch_gene_cpu  # Tensor on GPU
            # adj = buildGraphNN(batch_gene, 30)
            # adj = normalized(adj)
            # adj = sparse_mx_to_torch_sparse_tensor(adj)

            # genedecoded = geneautoencoder(batch_gene,noise = False,mean=1, stddev=0.01)
            genedecoded = geneautoencoder(batch_gene.detach().numpy())
            geneloss = geneloss_func(genedecoded, batch_gene)  # mean square error
            geneoptimizer.zero_grad()  # clear gradients for this training step
            geneloss.backward()  # backpropagation, compute gradients
            geneoptimizer.step()  # apply gradients

            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCH_pretrain, geneloss.item()))
        if epoch > 0 and cellloss.item() < tol_gene:
            print('geneloss', geneloss.item(), '< tol ', tol_gene)
            print('Reached tolerance threshold. Stopping training.')
            break

    end = time.time()
    gene_time = end - start
    print('gene time:', gene_time)

    # 2 ways to save the net
    torch.save(cellautoencoder.state_dict(), save_path1)  # save only the parameters
    torch.save(geneautoencoder.state_dict(), save_path2)  # save only the parameters

    #######################Fit predition#####################
    # %load_ext autoreload
    # %autoreload 2
    from DeepCI_ZINB import Net

    n_clusters = len(np.unique(y_true))
    print('unique number of label:', n_clusters)
    # Hyper Parameters
    Cell_BATCH_SIZE = args.Cell_BATCH_SIZE
    Gene_BATCH_SIZE = args.Gene_BATCH_SIZE
    lr = args.lr
    EPOCH = args.EPOCH  # 400
    update_interval = args.update_interval
    tol = args.tol

    idec = Net(input.shape[1], input.shape[0], args.z_dim, n_clusters, cellEncodeLayer=[64, 32],
               cellDecodeLayer=[32, 64], geneEncodeLayer=[64, 32], geneDecodeLayer=[32, 64], activation="relu",
               dropout=0.1, alpha=args.alpha, rec_gamma=args.rec_gamma, clu_gamma=args.clu_gamma,
               cell_gamma=args.cell_gamma, gene_gamma=args.gene_gamma, Graph_gamma=args.Graph_gamma,
               Graph_sparse=args.Graph_sparse, ZINB_gamma=args.ZINB_gamma, Cell_Graph=args.Cell_Graph,
               Cell_sparse=args.Cell_sparse)
    print(idec)

    # cell_path = '/home/lzl/DFM/simulation_lzl/simulation_scale_network/pretrain_weights/' + allf[i] + '_cellautoencoder_dropout0.1_defultPreproFilter0.005_64_32_buildnetwork_epoch200.pkl'
    # gene_path = '/home/lzl/DFM/simulation_lzl/simulation_scale_network/pretrain_weights/' + allf[i] + '_geneautoencoder_dropout0.1_defultPreproFilter0.005_64_32_buildnetwork_epoch200.pkl'

    idec.load_model(save_path1, save_path2)
    # idec.cuda()

    torch.manual_seed(2)

    input_fina = input
    # y = torch.from_numpy(y_true).int()
    y = torch.from_numpy(y_true).int()

    # y_pred_origin = idec.fit(input_fina, y, lr, Cell_BATCH_SIZE, Gene_BATCH_SIZE, EPOCH, update_interval, tol,noise = False,mean=1, stddev=0.01)
    y_pred_origin = idec.fit(input_fina, y, lr, Cell_BATCH_SIZE, Gene_BATCH_SIZE, EPOCH, update_interval, tol)

    [xrecon, cell_emb, gene_emb, z, q, _, _, _] = idec.forward(input_fina.to("cuda"), (torch.t(input_fina)).to("cuda"))

    print('=================Final  统计模型复杂度和参数量=============')
    num_params = sum(param.numel() for param in idec.parameters())
    print('模型参数量：', num_params)




    y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

    # # 绘制聚类图
    # tsne = TSNE(n_components=2,random_state=0)
    #
    #
    # X_tsne = tsne.fit_transform(cell_emb.detach().numpy())
    #
    # plt.figure(figsize=(5, 5))
    # # plt.subplot(121)
    # labelliat = np.unique(y)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y, s=4, alpha=None, marker='o', edgecolors=None)
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, label=labelliat)
    # # plt.legend()
    # imagePath = './images/'+ dataset_name +'_ScAGN.jpg'
    # plt.savefig(imagePath, dpi=120)
    # plt.show()

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = acc(y_true, y_pred)
    fmi = metrics.cluster.fowlkes_mallows_score(y_true, y_pred)
    print("acc, nmi, ari, fmi")
    print("acc: " + str(acc))
    print("nmi: " + str(nmi))
    print("ari: " + str(ari))
    print("fmi: " + str(fmi))

    return acc, nmi, ari, fmi


if __name__ == '__main__':

    # datasetnames = ['Quake_10x_Bladder', 'Quake_10x_Limb_Muscle', 'Quake_Smart-seq2_Diaphragm', 'Romanov', 'Young',
    #                 'Adam', 'Chen', 'Plasschaert', 'Quake_10x_Spleen', 'Quake_10x_Trachea',
    #                 'Quake_Smart-seq2_Heart', 'Quake_Smart-seq2_Limb_Muscle', 'Quake_Smart-seq2_Lung', 'Klein',
    #                 'Muraro', 'Pollen', 'Tosches_turtle', 'Wang_Lung']

    datasetnames = ['Young']

    # 参数设定
    parser = argparse.ArgumentParser()

    # dataset hyper-parameters
    parser.add_argument('--Cell_BATCH_SIZE', type=int, default=512)
    parser.add_argument('--Gene_BATCH_SIZE', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-04)  # 0.00001
    parser.add_argument('--EPOCH', type=int, default=25)
    parser.add_argument('--update_interval', type=int, default=5)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--rec_gamma', type=float, default=0.01)  # 重构损失
    parser.add_argument('--clu_gamma', type=float, default=1)  # 聚类督损失
    parser.add_argument('--cell_gamma', type=float, default=0)  # 细胞重构损失
    parser.add_argument('--gene_gamma', type=float, default=1)  # 基因重构损失
    parser.add_argument('--alpha', type=float, default=10)  # 自监督参数
    # parser.add_argument('--Graph_gamma', type=float, default=0.0001)  # 图正则化
    parser.add_argument('--Graph_gamma', type=float, default=0)  # 图正则化
    parser.add_argument('--Graph_sparse', type=float, default=0)  # 图稀疏化
    parser.add_argument('--Cell_Graph', type=float, default=0.001)  # 细胞图正则化
    parser.add_argument('--Cell_sparse', type=float, default=0.0001)  # 细胞稀疏化
    parser.add_argument('--ZINB_gamma', type=float, default=0)  # ZINB重构损失

    # 预训练参数
    parser.add_argument('--LR_pretrain_cell', type=float, default=0.0001)  # ZINB重构损失
    parser.add_argument('--LR_pretrain_gene', type=float, default=0.0001)  # ZINB重构损失
    parser.add_argument('--Cell_pretrain_BATCH_SIZE', type=int, default=512)  # ZINB重构损失
    parser.add_argument('--Gene_pretrain_BATCH_SIZE', type=int, default=1024)  # ZINB重构损失
    parser.add_argument('--EPOCH_pretrain', type=float, default=25)  # ZINB重构损失
    parser.add_argument('--tol_pretrain_cell', type=float, default=0.001)  # ZINB重构损失
    parser.add_argument('--tol_pretrain_gene', type=float, default=0.001)  # ZINB重构损失

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    result_file = './Result/result_scAG2E_xm.txt'
    start = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

    # 设置输出文件格式
    # ss = '=' * 10 + '新训练：' + start + '(Graph_gamma:' + str(args.Graph_gamma) + ', ZINB_gamma:' + str(
    #     args.ZINB_gamma) + ')' + '=' * 10 + '\n' + str(args)
    ss = '=' * 10 + '参数clu_gamma：新训练：' + start + '=' * 10 + '\n'
    # print(args)
    with open(result_file, 'a') as f:  # 设置文件对象
        f.write(ss)  # 可以是随便对文件的操作

    # 开始训练
    for datasetname in datasetnames:
        # dataset = datasetname
        print("============" + datasetname + "=================")

        # for lri in range(7,13):
        # for lri in range(7,8):
        # args.clu_gamma = 1e-10 * (10 ** lri)
        # args.clu_gamma = 1e-10 * (10 ** 13)
        ss = '=' * 10 + '新参数：' + start + '=' * 10 + '\n' + str(args) + '\n'
        # 运行主程序
        results = pd.DataFrame()
        acc, nmi, ari, fmi = main(datasetname, args)
        results = pd.concat([results, pd.DataFrame([acc, nmi, ari, fmi]).T])
        results.columns = ["acc", "nmi", "ari", "fmi"]
        result_str = ss + datasetname + '\n' + str(results) + '\n'

        with open(result_file, 'a') as f:  # 设置文件对象
            # f.write(ss)
            f.write(result_str)  # 可以是随便对文件的操作
