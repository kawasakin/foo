# import all packages
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from io import StringIO 
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import random 
from scipy.stats.stats import pearsonr
import collections 
from itertools import chain, combinations
from random import shuffle
import gc
from collections import Counter
from math import sqrt
from joblib import Parallel, delayed

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def gen_chip_feature(fname, region_len=50):
    with open(fname) as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_X = dat.reshape(-1, 1, 17, region_len)
    return dat_X

def gen_matches(fname):
    matches = collections.defaultdict(list)
    with open(fname) as fin:
        for line in fin:
            [p, e] = map(int, line.strip().split())
            matches[p].append(e)
    return matches

def gen_target(fname, n=2):
    with open(fname) as fin:
        dat_Y = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_Y = dat_Y + 1
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat_Y)])
    return dat_Y

def accumu(lis):
    total = 0
    for x in lis:
        total += x
        yield total
        
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates
    
def model(X, w1, w2, Max_Pooling_Shape, p_drop_conv):
    # Max_Pooling_Shape has to be large enough to pool only one element out
    l1 = T.flatten(dropout(T.mean(rectify(conv2d(X, w1, border_mode='valid')), axis=3), p_drop_conv), outdim=2)
    pyx = softmax(T.dot(l1, w2))
    return pyx

def all_subsets(ss, num):
    return chain(*map(lambda x: combinations(ss, x), range(num+1)))

def cross_entropy(a, y):
    return np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))

# defining all the parameters
num_class = 2           # number of output
p_drop_conv = 0         # dropout rate for cov_layer
mini_batch_size = 100    # [40 - 100]
lr = 0.001              # learning rate
epchs = 15              # number of iteration
feat_num = 17           # number of chip features
chip_motif_len = 6      # length of motif matrix
chip_motif_num = 100     # number of motifs 
#max_pool_shape = (1, 5000000) # max pool maxtrix size

# graphic parameters
X = T.ftensor4()
Y = T.fmatrix()

w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
w2 = init_weights((chip_motif_num, num_class))

noise_py_x = model(X, w1, w2, max_pool_shape, p_drop_conv)
py_x = model(X, w1, w2, max_pool_shape, 0.)
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2]
updates = RMSprop(cost, params, lr)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

# load input
dat_X = gen_chip_feature("datX_P.dat", 100)    # promoter 5k
dat_Y = gen_target("datY.dat", 2)               # target 

# seperate the data into training and test
index = list(range(dat_X.shape[0]))
random.shuffle(index)
trX = dat_X[index[:round(len(index)*4/5)]]
trY = dat_Y[index[:round(len(index)*4/5)]]
teX = dat_X[index[round(len(index)*4/5):]]
teY = dat_Y[index[round(len(index)*4/5):]]

# model 0 that based only on promoters
index = np.array(range(trX.shape[0]))
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trX), mini_batch_size), range(mini_batch_size, len(trX), mini_batch_size)):
        cost = train(trX[index][start:end], trY[index][start:end])
    te_pred = predict(teX)
    print(i, np.mean(cross_entropy(te_pred, teY)), np.mean(np.argmax(teY, axis=1) == np.argmax(te_pred, axis=1)))
    gc.collect()





l1_pool_feat = T.flatten(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), max_pool_shape), outdim=2)
l1_pool_feat_cal = theano.function(inputs=[X], outputs=l1_pool_feat, allow_input_downcast=True)
feats = l1_pool_feat_cal(dat_X_P)

np.savetxt("tmp.txt", feats)
# R
data = read.table("tmp.txt")
AUC <- function(pos.scores, neg.scores){
    res = c()
    for(i in 1:100){
        res = c(res, mean(sample(pos.scores,2000,replace=T) > sample(neg.scores,2000,replace=T)))         
    }
    return(mean(res))
}
aucs = sapply(1:ncol(data), function(i) AUC(data[4431:nrow(data),i], data[1:4431,i]))
orders = order(aucs, decreasing=TRUE) - 1
aucs[orders+1]
# R-end
feat_order = \
[5, 8, 7, 9, 4, 0, 3, 6, 1, 2]
#[1] 0.929180 0.927285 0.926270 0.920460 0.910855 0.907330 0.896575 0.494940
#[9] 0.414200 0.413785

# get the heatmap
l1_cov2d = rectify(conv2d(X, w1, border_mode='valid'))
l1_cov2d_cal = theano.function(inputs=[X], outputs=l1_cov2d, allow_input_downcast=True)
feats = l1_cov2d_cal(dat_X_P)
res = []
for i in feat_order:
    res.append(feats[:, i, 0, :])
np.savetxt("feat_heatmaps.txt", np.array(res).reshape(11041*10, 95))
# get the motifs
np.savetxt("weights.txt", w1.get_value()[feat_order].reshape(-1, 6))
