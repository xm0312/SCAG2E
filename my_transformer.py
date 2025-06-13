import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import scanpy as sc

from lib import preprocessH5
from sklearn.cluster import KMeans


class TransformerCluster(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, num_clusters):
        super(TransformerCluster, self).__init__()
        self.embedding_dim = embedding_dim

        # 基因嵌入层
        self.gene_embedding = nn.Linear(input_dim, embedding_dim)

        # 位置编码
        self.position_encoding = PositionalEncoding(embedding_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 聚类层
        self.cluster_layer = nn.Linear(embedding_dim, num_clusters)

        # 自监督学习任务的头
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, input_dim)
        )

    def forward(self, x):
        # 基因嵌入
        x = self.gene_embedding(x)  # (batch_size, seq_len, embedding_dim)

        # 添加位置编码
        x = self.position_encoding(x)

        # Transformer编码
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        encoded = self.transformer_encoder(x)
        encoded = encoded.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)

        # 平均池化获取细胞表示
        cell_repr = encoded.mean(dim=1)  # (batch_size, embedding_dim)

        # 聚类分配
        cluster_logits = self.cluster_layer(cell_repr)  # (batch_size, num_clusters)

        return cluster_logits, encoded

    def predict_genes(self, x, mask):
        # 基因嵌入
        x = self.gene_embedding(x)

        # 添加位置编码
        x = self.position_encoding(x)

        # Transformer编码
        x = x.permute(1, 0, 2)
        encoded = self.transformer_encoder(x)
        encoded = encoded.permute(1, 0, 2)

        # 应用掩码
        masked_encoded = encoded * mask.unsqueeze(-1)

        # 预测被掩码的基因
        predicted_genes = self.predictor(masked_encoded)

        return predicted_genes


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        return x + self.pe[:, :x.size(1), :]


class ClusterLoss(nn.Module):
    def __init__(self, num_clusters, alpha=1.0, beta=1.0):
        super(ClusterLoss, self).__init__()
        self.alpha = alpha  # 聚类损失权重
        self.beta = beta  # 自监督损失权重
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, cluster_logits, targets, predicted_genes=None, original_genes=None, mask=None):
        # 聚类损失 (KL散度)
        cluster_loss = self.criterion(torch.log_softmax(cluster_logits, dim=1), targets)

        # 自监督损失 (MSE)
        ssl_loss = 0.0
        if predicted_genes is not None and original_genes is not None and mask is not None:
            ssl_loss = F.mse_loss(predicted_genes, original_genes, reduction='none')
            ssl_loss = (ssl_loss * mask).sum() / mask.sum()

        # 总损失
        total_loss = self.alpha * cluster_loss + self.beta * ssl_loss

        return total_loss, cluster_loss, ssl_loss


def preprocess_data(adata, n_pcs_list=[30, 35, 40, 45, 50], n_hvgs=3000):
    """
    数据预处理和多尺度PCA降维

    参数:
        adata: anndata对象
        n_pcs_list: 不同PCA维度列表
        n_hvgs: 高可变基因数量

    返回:
        多尺度PCA特征列表
    """
    # 标准化和高可变基因选择
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
    adata = adata[:, adata.var.highly_variable]

    # 多尺度PCA
    pca_features = []
    for n_pcs in n_pcs_list:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        pca_features.append(adata.obsm['X_pca'])

    return pca_features


def initialize_clusters(pca_features, n_clusters, weights=None):
    """
    多尺度K-means聚类初始化

    参数:
        pca_features: 多尺度PCA特征列表
        n_clusters: 聚类数量
        weights: 各尺度权重 (None表示等权重)

    返回:
        初始聚类标签 (加权投票结果)
    """
    if weights is None:
        weights = np.ones(len(pca_features)) / len(pca_features)

    # 各尺度K-means聚类
    cluster_votes = np.zeros((pca_features[0].shape[0], n_clusters))
    for i, features in enumerate(pca_features):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # 创建one-hot投票矩阵
        vote = np.zeros((features.shape[0], n_clusters))
        for j in range(features.shape[0]):
            vote[j, labels[j]] = 1.0

        cluster_votes += vote * weights[i]

    # 加权投票确定最终标签
    final_labels = np.argmax(cluster_votes, axis=1)

    return final_labels


def train_model(adata, model, device, epochs=100, batch_size=256, lr=1e-3,
                n_clusters=10, mask_prob=0.2, patience=10):
    """
    训练Transformer聚类模型

    参数:
        adata: 预处理后的anndata对象
        model: TransformerCluster模型
        device: 训练设备
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
        n_clusters: 聚类数量
        mask_prob: 基因掩码概率
        patience: 早停耐心值
    """
    # 准备数据
    X = adata.X.toarray() if isinstance(adata.X, np.matrix) else adata.X
    X = torch.FloatTensor(X).to(device)

    # 初始化聚类中心
    with torch.no_grad():
        # 使用多尺度PCA特征初始化
        pca_features = preprocess_data(adata)
        init_labels = initialize_clusters(pca_features, n_clusters)
        init_labels = torch.LongTensor(init_labels).to(device)

        # 计算初始聚类中心
        cluster_centers = torch.zeros(n_clusters, X.shape[1]).to(device)
        for i in range(n_clusters):
            cluster_centers[i] = X[init_labels == i].mean(dim=0)

    # 优化器
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = ClusterLoss(n_clusters, alpha=1.0, beta=0.5)

    # 训练循环
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # 随机打乱数据
        permutation = torch.randperm(X.shape[0])

        for i in range(0, X.shape[0], batch_size):
            optimizer.zero_grad()

            # 获取批量数据
            indices = permutation[i:i + batch_size]
            batch_X = X[indices]

            # 创建基因掩码
            mask = torch.ones_like(batch_X)
            mask = mask * (torch.rand_like(batch_X) > mask_prob).float()
            masked_X = batch_X * mask

            # 前向传播
            cluster_logits, _ = model(masked_X)

            # 计算目标分布 (软分配)
            q = 1.0 / (1.0 + (torch.sum((batch_X.unsqueeze(1) - cluster_centers) ** 2, dim=2) / 1.0))
            q = q ** 2 / q.sum(dim=1, keepdim=True)

            # 自监督学习任务
            predicted_genes = model.predict_genes(masked_X, mask)

            # 计算损失
            loss, cluster_loss, ssl_loss = criterion(cluster_logits, q, predicted_genes, batch_X, mask)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印训练信息
        avg_loss = total_loss / (X.shape[0] // batch_size)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Cluster Loss: {cluster_loss:.4f}, SSL Loss: {ssl_loss:.4f}")

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


def cluster_cells(adata, model, device, n_clusters=10):
    """
    使用训练好的模型对细胞进行聚类

    参数:
        adata: 预处理后的anndata对象
        model: 训练好的TransformerCluster模型
        device: 计算设备
        n_clusters: 聚类数量

    返回:
        聚类标签
    """
    # 准备数据
    X = adata.X.toarray() if isinstance(adata.X, np.matrix) else adata.X
    X = torch.FloatTensor(X).to(device)

    # 获取聚类分配
    with torch.no_grad():
        # cluster_logits, _ = model(X)
        # cluster_probs = F.softmax(cluster_logits, dim=1)
        # labels = torch.argmax(cluster_probs, dim=1).cpu().numpy()

        # 将数据移到CPU（kmeans函数可能不支持GPU）
        X_cpu = X.cpu() if X.is_cuda else X

        # 运行K-Means
        kmeans = KMeans(n_clusters, init='k-means++', n_init=100)
        cluster_ids = kmeans.fit_predict(X_cpu)

        return cluster_ids  # 返回numpy数组

    return labels


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


# 示例用法
if __name__ == "__main__":
    # 加载数据 (这里使用scanpy的示例数据)
    # input, y_true, _, _, _,adata = preprocessH5.load_h5("/root/autodl-tmp/scSAG2E/DeepCI_ZINB/data/Young/data.h5",
    #                                                     5685)

    df = pd.read_csv('/root/autodl-tmp/scCFIB/data/Biase/Biase_Normalized.tsv', sep='\t')
    df2 = pd.read_csv('/root/autodl-tmp/scCFIB/data/Biase/subtype.ann', sep='\t')
    y_true = df2["label"]
    df.drop(columns=df.columns[0], inplace=True)
    input = df.T
    adata = sc.AnnData(input)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=49,
                                subset=True)
    input = adata.X.astype(np.float32)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = adata.shape[1]  # 基因数量
    model = TransformerCluster(
        input_dim=input_dim,
        embedding_dim=input_dim,
        num_heads=5,
        num_layers=3,
        num_clusters=4
    ).to(device)

    # 训练模型
    train_model(adata, model, device, epochs=1, batch_size=input_dim, lr=1e-3, n_clusters=4)

    # 聚类细胞
    labels = cluster_cells(adata, model, device, n_clusters=6)

    nmi = normalized_mutual_info_score(y_true, labels)
    ari = adjusted_rand_score(y_true, labels)
    acc = acc(y_true, labels)
    fmi = metrics.cluster.fowlkes_mallows_score(y_true, labels)
    print("acc, nmi, ari, fmi")
    print("acc: " + str(acc))
    print("nmi: " + str(nmi))
    print("ari: " + str(ari))
    print("fmi: " + str(fmi))