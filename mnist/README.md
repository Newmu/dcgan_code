Modify data_dir in lib/config.py to point to directory with mnist files.

Run train_cond_dcgan.py to train mnist model from appendix. It will create a few folders and save training info, model parameters, and samples periodically. Should take ~ an hour to run on a good GPU. 

Libs you'll need installed/configured to run it:
- theano
- cudnn
- sklearn
- numpy
- scipy
- matplotlib
- tqdm