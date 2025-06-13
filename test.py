#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9
# @Author  : 小龙人
# @Github  : https://github.com/lqh42
# @Software: PyCharm
# @File    : test.py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def p(X,y_prred,name):
    X = X
    labels = y_prred
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(X)
    x_min, x_max = Y.min(0), Y.max(0)
    X_norm = (Y - x_min) / (x_max - x_min)

    area = (20 * np.random.rand(len(Y))) ** 2
    color = []

    # for i in np.arange(len(labels)):
    #         if labels[i] == 1:
    #                 color.append('darkgreen')
    #         if labels[i] == 2:
    #                 color.append('darkorchid')
    #         if labels[i] == 3:
    #                 color.append('darkgoldenrod')
    #         if labels[i] == 4:
    #                 color.append('darkred')
    #         if labels[i] == 0:
    #                 color.append('royalblue')


    plt.scatter(Y[:, 0], Y[:, 1], c=color, s=4, alpha=None, marker='o', edgecolors=None)
    filename = name + '_kmeans' + '.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()

# print('=================Final  统计模型复杂度和参数量=============')
# num_params = sum(param.numel() for param in final_model.parameters())
# print('torchinfo模型复杂度统计')
# print(summary(final_model, (args.batch_size, CONFIG_HSIC[args.datasets]['bands'], args.windows, args.windows)))
# print('thop模型复杂度统计')
# input = torch.randn(args.batch_size, CONFIG_HSIC[args.datasets]['bands'], args.windows, args.windows).to(
#     args.device)
# macs, params = profile(final_model, inputs=(input,))
# print('FLOPs = ' + str(macs * 2 / 1000 ** 3) + 'G')  # macs和flops之间有一个双倍的关系
# print('Params = ' + str(params / 1000 ** 2) + 'M')
# print("模型参数量：")
# print(num_params)