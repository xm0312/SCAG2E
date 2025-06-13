import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.utils.data as Data

import numpy as np
import math
# from lib.utils import acc
# from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

# from .ZINBloss import ZINB
from lib.utils import buildGraphNN, sparse_mx_to_torch_sparse_tensor, normalize
from model.GNN import GraphConvolution


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


# def buildNetwork_gene(layers, activation="relu", dropout=0):
#     net = []
#     for i in range(1, len(layers)):
#         net.append(nn.Linear(layers[i-1], layers[i]))
#         if activation=="relu":
#             net.append(nn.ReLU())
#         elif activation=="sigmoid":
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


class Net(torch.nn.Module):

    def __init__(self, cellInput_dim=784, geneInput_dim=784, z_dim=10, n_clusters=10,
                 cellEncodeLayer=[400], cellDecodeLayer=[400], geneEncodeLayer=[400], geneDecodeLayer=[400],
                 activation="relu", dropout=0, alpha=1., rec_gamma=1, clu_gamma=1, cell_gamma=1, gene_gamma=1,
                 Graph_gamma=0.01, Graph_sparse=0.01, ZINB_gamma=0.1, Cell_Graph=0.1, Cell_sparse=0.01):
        super(self.__class__, self).__init__()
        self.activation = activation
        self.dropout = dropout
        # 细胞自编码器，自监督模块
        self.cellencoder = buildNetwork_cell([cellInput_dim] + cellEncodeLayer, activation=activation, dropout=dropout)
        self.celldecoder = buildNetwork_cell(cellDecodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(cellEncodeLayer[-1], z_dim)
        self._dec = nn.Linear(cellDecodeLayer[-1], cellInput_dim)
        self.fc_pi = nn.Linear(cellDecodeLayer[-1], cellInput_dim)
        self.fc_disp = nn.Linear(cellDecodeLayer[-1], cellInput_dim)
        self.fc_mu = nn.Linear(cellDecodeLayer[-1], cellInput_dim)

        # 基因自编码器
        self.geneencoder = buildNetwork_gene([geneInput_dim] + geneEncodeLayer, activation=activation, dropout=dropout)
        self.genedecoder = buildNetwork_gene(geneDecodeLayer, activation=activation, dropout=dropout)
        self._enc_gene = nn.Linear(geneEncodeLayer[-1], z_dim)
        self._dec_gene = nn.Linear(geneDecodeLayer[-1], geneInput_dim)

        # 相关参数
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.rec_gamma = rec_gamma
        self.clu_gamma = clu_gamma
        self.cell_gamma = cell_gamma
        self.gene_gamma = gene_gamma
        self.Graph_gamma = Graph_gamma
        self.Graph_sparse = Graph_sparse
        self.ZINB_gamma = ZINB_gamma
        self.Cell_Graph = Cell_Graph
        self.Cell_sparse = Cell_sparse
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def forward(self, celltogene, genetocell):

        # 细胞编码器
        cell_emb = self.cellencoder(celltogene)
        # 细胞低维表示
        z = self._enc_mu(cell_emb)
        # 细胞解码器
        cell_h = self.celldecoder(cell_emb)
        # 细胞重构
        cellrecon = self._dec(cell_h)
        # if self.distribution == "ZINB":

        pi = self.fc_pi(cell_h)
        self.pi = torch.sigmoid(pi)
        disp = self.fc_disp(cell_h)
        self.disp = torch.clamp(F.softplus(disp), min=0, max=1e4)
        mean = self.fc_mu(cell_h)
        self.mean = torch.clamp(F.relu(mean), min=0, max=1e6)
        self.output = self.mean
        self.likelihood_loss = ZINB(self.pi, self.disp, celltogene, self.output, ridge_lambda=1.0, mean=True)

        # 求解自监督q
        q = self.soft_assign(z)

        # 基因编码器
        gene_emb = self.geneencoder(genetocell)

        # 基因低维表示
        # z_gene = self._enc_gene(gene_emb)

        # 基因解码器
        # gene_h = self.genedecoder(gene_emb)
        # 基因重构
        # generecon = self._dec_gene(gene_h)

        generecon = torch.mm(gene_emb, gene_emb.T)

        # NMF重构基因表达矩阵
        xrecon = torch.mm(cell_emb.detach().cpu(), (torch.t(gene_emb).detach().cpu()))

        # return xrecon, cell_emb, gene_emb, z, q, cellrecon, generecon
        return xrecon, cell_emb, gene_emb, z, q, cellrecon, self.likelihood_loss, generecon
        # return xrecon, cell_emb, gene_emb, z, q, self.likelihood_loss, gene_emb

    def load_model(self, cell_path, gene_path):
        pretrained_dict_cell = torch.load(cell_path, map_location=lambda storage, loc: storage)  ### gpu -> cpu
        pretrained_dict_gene = torch.load(gene_path, map_location=lambda storage, loc: storage)
        model_dict_cell = self.state_dict()
        # model_dict_gene = self.state_dict()
        pretrained_dict_cell = {k: v for k, v in pretrained_dict_cell.items() if k in model_dict_cell}
        pretrained_dict_gene = {k: v for k, v in pretrained_dict_gene.items() if k in model_dict_cell}
        # 更新数据
        model_dict_cell.update(pretrained_dict_cell)
        model_dict_cell.update(pretrained_dict_gene)
        self.load_state_dict(model_dict_cell)
        # self.load_state_dict(model_dict_gene)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2,
                                   dim=2) / self.alpha)  ## z.unsqueeze(1)这个函数主要是对数据维度进行扩充，变成二维了，行数*1列
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def loss_function(self, x, xrecon, p, q, matrix_mask, cellrecon, generecon, input_cell, input_gene, cell_emb,
                      gene_emb,
                      likelihood_loss):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        cadj = buildGraphNN(x, 30)
        # adj = normalize(adj)
        CDiag = np.diagflat(cadj.sum(0).tolist()[0])
        CLaplac = CDiag - cadj
        CLaplac = torch.from_numpy(np.array(CLaplac)).float()

        Cadjw = sparse_mx_to_torch_sparse_tensor(cadj).float().to_dense()

        Ggadj = buildGraphNN(torch.t(x), 30)
        # adj = normalize(adj)
        GDiag = np.diagflat(Ggadj.sum(0).tolist()[0])
        GLaplac = GDiag - Ggadj
        GLaplac = torch.from_numpy(np.array(GLaplac)).float()

        adjw = sparse_mx_to_torch_sparse_tensor(Ggadj).float().to_dense()

        kldloss = kld(p, q)  # 自监督损失函数

        mcloss = torch.sum(torch.mul(torch.matmul(q, q.T), Cadjw))

        recon_loss = torch.mean((x - (torch.mul(xrecon, matrix_mask))) ** 2)  # + self.Graph_gamma * torch.mm(
        # torch.mm(cell_emb, CLaplac), cell_emb).trace()  # + self.Graph_sparse * torch.sum(torch.abs(adjw))

        cell_loss = torch.mean((input_cell - cellrecon) ** 2) + 0 * self.Cell_Graph * torch.mm(
            torch.mm(cell_emb.T, CLaplac), cell_emb).trace() + self.Cell_sparse * torch.sum(torch.abs(Cadjw))

        # gene_loss = torch.mean((input_gene - generecon) ** 2) + self.Graph_gamma * torch.mm(
        #     torch.mm(gene_emb.T, GLaplac), gene_emb).trace()+ self.Graph_sparse * torch.sum(torch.abs(adjw))

        # gene_loss = torch.mean((adjw - torch.mm(gene_emb, gene_emb.T)) ** 2)
        gene_loss = torch.mean((adjw - generecon) ** 2)


        loss = self.rec_gamma * recon_loss + self.clu_gamma * kldloss + self.cell_gamma * cell_loss + self.gene_gamma * gene_loss + self.ZINB_gamma * likelihood_loss + 0 * mcloss
        return loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def encodeBatch(self, X, Cell_BATCH_SIZE):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        Cell_BATCH_SIZE = int(Cell_BATCH_SIZE)
        X = X.to("cuda")
        loader_cell = Data.DataLoader(
            dataset=X,  # torch TensorDataset format
            batch_size=Cell_BATCH_SIZE,  # mini batch size
            shuffle=False,  # random shuffle for training
            num_workers=0,  # subprocesses for loading data
        )

        encoded_z = []
        encoded = []
        self.eval()

        for step_cell, batch_cell in enumerate(loader_cell):  # for each training step
            # Gene_X = torch.t(Cell_X)
            # adj = buildGraphNN(torch.t(X), 30)
            # adj = normalize(adj)
            # adj = sparse_mx_to_torch_sparse_tensor(adj)

            [_, cell_emb, _, z, _, _, _, _] = self.forward(batch_cell, torch.t(X))

            encoded_z.append(z.data)
            encoded = torch.cat(encoded_z, dim=0)
        return encoded

    def fit(self, X, y=None, lr=0.001, Cell_BATCH_SIZE=512, Gene_BATCH_SIZE=512, num_epochs=10, update_interval=1,
            tol=1e-3):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print(use_cuda)
            self.cuda()
        print("=====Training IDEC=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)  # s
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, init='k-means++', n_init=100)
        Cell_BATCH_SIZE = torch.tensor(Cell_BATCH_SIZE)
        encoded = self.encodeBatch(X, Cell_BATCH_SIZE)
        y_pred = kmeans.fit_predict(encoded.data.cpu().numpy())
        # y_pred = kmeans.fit_predict(encoded)
        y_pred_last = y_pred
        # 将kemeans结果作为初始化聚类中心
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        # 进入训练模式
        self.train()

        Cell_BATCH_SIZE = int(Cell_BATCH_SIZE)
        loader_cell = Data.DataLoader(
            dataset=X,  # torch TensorDataset format
            batch_size=Cell_BATCH_SIZE,  # mini batch size
            shuffle=False,  # random shuffle for training
            num_workers=0,  # subprocesses for loading data
            drop_last=False,
        )

        loader_gene = Data.DataLoader(
            dataset=torch.t(X),  # torch TensorDataset format
            batch_size=Gene_BATCH_SIZE,  # mini batch size
            shuffle=False,  # random shuffle for training
            num_workers=0,  # subprocesses for loading data

        )

        num_cell = X.shape[0]
        num_gene = X.shape[1]
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X, Cell_BATCH_SIZE)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num_cell  ###
                y_pred_last = y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # train 1 epoch
            train_loss = 0.0
            for step_cell, batch_cell in enumerate(loader_cell):  # for each training step
                print('Epoch: ', epoch, '| Step_cell: ', step_cell)

                pbatch = p[step_cell * Cell_BATCH_SIZE: min((step_cell + 1) * Cell_BATCH_SIZE, num_cell)]

                for step_gene, batch_gene in enumerate(loader_gene):
                    # train your data...
                    # print('Step_gene: ', step_gene)

                    batch_input = batch_cell[:,
                                  step_gene * Gene_BATCH_SIZE:min((step_gene + 1) * Gene_BATCH_SIZE, num_gene)]

                    matrix_mask1_cpu = batch_input.cpu().numpy().copy()
                    matrix_mask1_cpu[matrix_mask1_cpu.nonzero()] = 1
                    matrix_mask1_cpu = torch.from_numpy(matrix_mask1_cpu).float()

                    optimizer.zero_grad()
                    inputs = Variable(batch_input)
                    target = Variable(pbatch)

                    # adj = buildGraphNN(torch.t(batch_gene), 30)
                    # adj = normalize(adj)
                    # adj = sparse_mx_to_torch_sparse_tensor(adj)

                    [xrecon_batch, cell_emb_batch, gene_emb_batch, z_batch, q_batch, cellrecon, likelihood_loss,
                     generecon] = self.forward(batch_cell.to("cuda"), batch_gene.to("cuda"))

                    loss = self.loss_function(inputs, xrecon_batch, target.cpu(), q_batch.cpu(), matrix_mask1_cpu, cellrecon.cpu(),
                                              generecon, batch_cell, batch_gene, cell_emb_batch.cpu(), gene_emb_batch,
                                              likelihood_loss.cpu())
                    train_loss += loss.data * len(inputs)
                    loss.backward()
                    optimizer.step()

            print("#Epoch %3d: Loss: %.4f" % (
                epoch + 1, train_loss / num_cell))

        latent = self.encodeBatch(X, Cell_BATCH_SIZE)
        q = self.soft_assign(latent)
        # p = self.target_distribution(q).data

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

        return y_pred
