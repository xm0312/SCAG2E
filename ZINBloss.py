#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/22
# @Author  : 小龙人
# @Github  : https://github.com/lqh42
# @Software: PyCharm
# @File    : ZINBloss.py

import torch
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