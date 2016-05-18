import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC as LSVC

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score
from lib.costs import MSE,CCE

from load import svhn_with_valid_set

relu = activations.Rectify()
lrelu = activations.LeakyRectify(leak=0.2)
sigmoid = activations.Sigmoid()

trX, vaX, teX, trY, vaY, teY = svhn_with_valid_set(extra=False)

vaX = floatX(vaX)/127.5-1.
trX = floatX(trX)/127.5-1.
teX = floatX(teX)/127.5-1.

X = T.tensor4()

desc = 'svhn_unsup_all_conv_dcgan_100z_gaussian_lr_0.0005_64mb'
epoch = 200
params = [sharedX(p) for p in joblib.load('../models/%s/%d_discrim_params.jl'%(desc, epoch))]
print desc.upper()
print 'epoch %d'%epoch

def mean_and_var(X):
    u = T.mean(X, axis=[0, 2, 3])
    s = T.mean(T.sqr(X - u.dimshuffle('x', 0, 'x', 'x')), axis=[0, 2, 3])
    return u, s

def bnorm_statistics(X, w, w2, g2, b2, w3, g3, b3, wy):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))

    h2 = dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))
    h2_u, h2_s = mean_and_var(h2)
    h2 = lrelu(batchnorm(h2, g=g2, b=b2))

    h3 = dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2))
    h3_u, h3_s = mean_and_var(h3)
    h3 = lrelu(batchnorm(h3, g=g3, b=b3))

    h_us = [h2_u, h3_u]
    h_ss = [h2_s, h3_s]
    return h_us, h_ss

def infer_bnorm_stats(X, nbatch=128):
    U = [np.zeros(128, dtype=theano.config.floatX), np.zeros(256, dtype=theano.config.floatX)]
    S = [np.zeros(128, dtype=theano.config.floatX), np.zeros(256, dtype=theano.config.floatX)]
    n = 0
    for xmb in iter_data(X, size=nbatch):
        stats = _bnorm_stats(floatX(xmb))
        umb = stats[:2]
        smb = stats[2:]
        for i, u in enumerate(umb):
            U[i] += u
        for i, s in enumerate(smb):
            S[i] += s
        n += 1
    U = [u/n for u in U]
    S = [s/n for s in S]
    return U, S

def model(X,
    h2_u, h3_u,
    h2_s, h3_s,
    w, w2, g2, b2, w3, g3, b3, wy
    ):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2, u=h2_u, s=h2_s))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3, u=h3_u, s=h3_s))
    h = T.flatten(dnn_pool(h, (4, 4), (4, 4), mode='max'), 2)
    h2 = T.flatten(dnn_pool(h2, (2, 2), (2, 2), mode='max'), 2)
    h3 = T.flatten(dnn_pool(h3, (1, 1), (1, 1), mode='max'), 2)
    f = T.concatenate([h, h2, h3], axis=1)
    return [f]

X = T.tensor4()

h_us, h_ss = bnorm_statistics(X, *params)
_bnorm_stats = theano.function([X], h_us + h_ss)

trU, trS = infer_bnorm_stats(trX)

HUs = [sharedX(u) for u in trU]
HSs = [sharedX(s) for s in trS]

targs = [X]+HUs+HSs+params
f = model(*targs)
_features = theano.function([X], f)

def features(X, nbatch=128):
    Xfs = []
    for xmb in iter_data(X, size=nbatch):
        fmbs = _features(floatX(xmb))
        for i, fmb in enumerate(fmbs):
            Xfs.append(fmb)
    return np.concatenate(Xfs, axis=0)

cs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
vaXt = features(vaX)
mean_va_accs = []
for c in cs:
    tr_accs = []
    va_accs = []
    te_accs = []
    for _ in tqdm(range(10), leave=False, ncols=80):
        idxs = np.arange(len(trX))
        classes_idxs = [idxs[trY==y] for y in range(10)]
        sampled_idxs = [py_rng.sample(class_idxs, 100) for class_idxs in classes_idxs]
        sampled_idxs = np.asarray(sampled_idxs).flatten()

        trXt = features(trX[sampled_idxs])

        model = LSVC(C=c)
        model.fit(trXt[:1000], trY[sampled_idxs])
        tr_pred = model.predict(trXt)
        va_pred = model.predict(vaXt)
        tr_acc = metrics.accuracy_score(trY[sampled_idxs], tr_pred[:1000])
        va_acc = metrics.accuracy_score(vaY, va_pred)
        tr_accs.append(100*(1-tr_acc))
        va_accs.append(100*(1-va_acc))
    mean_va_accs.append(np.mean(va_accs))
    print 'c: %.4f train: %.4f %.4f valid: %.4f %.4f'%(c, np.mean(tr_accs), np.std(tr_accs)*1.96, np.mean(va_accs), np.std(va_accs)*1.96)
best_va_idx = np.argmin(mean_va_accs)
best_va_c = cs[best_va_idx]
print 'best c: %.4f'%best_va_c
teXt = features(teX)

tr_accs = []
va_accs = []
te_accs = []
for _ in tqdm(range(100), leave=False, ncols=80):
    idxs = np.arange(len(trX))
    classes_idxs = [idxs[trY==y] for y in range(10)]
    sampled_idxs = [py_rng.sample(class_idxs, 100) for class_idxs in classes_idxs]
    sampled_idxs = np.asarray(sampled_idxs).flatten()

    trXt = features(trX[sampled_idxs])

    model = LSVC(C=best_va_c)
    model.fit(trXt[:1000], trY[sampled_idxs])
    tr_pred = model.predict(trXt)
    va_pred = model.predict(vaXt)
    te_pred = model.predict(teXt)
    tr_acc = metrics.accuracy_score(trY[sampled_idxs], tr_pred[:1000])
    va_acc = metrics.accuracy_score(vaY, va_pred)
    te_acc = metrics.accuracy_score(teY, te_pred)
    # print '%.4f %.4f %.4f %.4f'%(c, 100*(1-tr_acc), 100*(1-va_acc), 100*(1-te_acc))
    tr_accs.append(100*(1-tr_acc))
    va_accs.append(100*(1-va_acc))
    te_accs.append(100*(1-te_acc))
print 'train: %.4f %.4f valid: %.4f %.4f test: %.4f %.4f'%(np.mean(tr_accs), np.std(tr_accs)*1.96, np.mean(va_accs), np.std(va_accs)*1.96, np.mean(te_accs), np.std(te_accs)*1.96)
