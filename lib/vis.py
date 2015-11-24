import numpy as np
from scipy.misc import imsave

def grayscale_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img

def color_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img

def grayscale_weight_grid_vis(w, (nh, nw), save_path=None):
    w = (w+w.min())/(w.max()-w.min())
    return grayscale_grid_vis(w, (nh, nw), save_path=save_path)