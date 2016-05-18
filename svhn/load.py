import sys
sys.path.append('..')

import os
import numpy as np
from scipy.io import loadmat

from lib.data_utils import shuffle
from lib.config import data_dir

def svhn(extra=False):
    data = loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    trX = data['X'].transpose(3, 2, 0, 1)
    trY = data['y'].flatten()-1
    data = loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    teX = data['X'].transpose(3, 2, 0, 1)
    teY = data['y'].flatten()-1
    if extra:
        data = loadmat(os.path.join(data_dir, 'extra_32x32.mat'))
        exX = data['X'].transpose(3, 2, 0, 1)
        exY = data['y'].flatten()-1
        return trX, exX, teX, trY, exY, teY
    return trX, teX, trY, teY

def svhn_with_valid_set(extra=False):
    if extra:
        trX, exX, teX, trY, exY, teY = svhn(extra=extra)
    else:
        trX, teX, trY, teY = svhn(extra=extra)
    trX, trY = shuffle(trX, trY)
    vaX = trX[:10000]
    vaY = trY[:10000]
    trX = trX[10000:]
    trY = trY[10000:]
    if extra:
        trS = np.asarray([1 for _ in range(len(trY))] + [0 for _ in range(len(exY))])
        trX = np.concatenate([trX, exX], axis=0)
        trY = np.concatenate([trY, exY], axis=0)
        trX, trY, trS = shuffle(trX, trY, trS)
    if extra:
        return trX, vaX, teX, trY, vaY, teY, trS
    else:
        return trX, vaX, teX, trY, vaY, teY
