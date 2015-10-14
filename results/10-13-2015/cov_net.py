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
    
def model(X, w1, w2, w3, Max_Pooling_Shape, p_drop_conv, p_drop_hidden):
    # Max_Pooling_Shape has to be large enough to pool only one element out
    l1 = T.flatten(dropout(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
    pyx = softmax(T.dot(l2, w3))
    return pyx

def learning(l_X, trY, ii, epchs, mini_batch_size):
    # model 0 - learning from promoter
    final = []
    index = np.array(range(l_X.shape[0]))
    for i in range(epchs):
        random.shuffle(index)
        for start, end in zip(range(0, len(l_X), mini_batch_size), range(mini_batch_size, len(l_X), mini_batch_size)):
            cost = train(l_X[index][start:end], trY[index][start:end])
        preds_tr = predict(l_X)
        costs = np.mean(cross_entropy(preds_tr, trY))
        accur = np.mean(np.argmax(trY, axis=1) == np.argmax(preds_tr, axis=1))
        print(ii, costs, accur)
        final.append((ii, costs, accur))
    gc.collect()
    return final

def all_subsets(ss, num):
    return chain(*map(lambda x: combinations(ss, x), range(num+1)))

def cross_entropy(a, y):
    return np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))

def update_X(matches, enhancer_num, P, E, Y, n_jobs):
    res = Parallel(n_jobs=n_jobs)(delayed(update)(P[i].reshape(1,1,17,-1), E[matches[i]], matches[i], enhancer_num, Y[i], i) for i in range(P.shape[0]))    
    aa = []
    bb = []
    for (a, b) in res:
        aa.append(a)
        bb.append(b)    
    del res
    gc.collect()
    return (np.array(aa), np.array(bb))

def update(p, e, matches, max_enhancer_num, y, k):
    e = np.vstack((e, np.zeros((17*19)).reshape(1, 1, 17, -1)))     
    match_index = range(len(matches))
    matches = np.array(matches)
    res_X = []
    res_E = []
    if len(match_index) != 0:
        tmp = all_subsets(match_index, max_enhancer_num)
        enhancer_index = [list(subset) + [-1]*(max_enhancer_num - len(subset)) for subset in tmp]
        enhancer_index = [sorted(x, key=lambda k: random.random()) for x in enhancer_index]
        matches = np.append(matches, -1)
        pp = p[[0]*len(enhancer_index)]
        ee = np.array(list(map(lambda x: np.dstack(e[x]), enhancer_index)))
        tmp = np.dstack((pp.reshape(-1, 17, pp.shape[3]), ee.reshape(-1, 17, ee.shape[3]))).reshape(pp.shape[0], 1, 17, -1)
        preds = predict(tmp)
        costs = cross_entropy(preds[:,1], y[[1]*len(enhancer_index)])
        res_X = tmp[np.argmin(costs)]
        res_E = matches[enhancer_index[np.argmin(costs)]]
        tmp = None
        del tmp
        del enhancer_index
        del pp
        del ee
        del preds
        del costs
        gc.collect()
    else:
        enhancer_index = [-1]*max_enhancer_num
        ee = np.array(list(map(lambda x: np.dstack(e[x]), [enhancer_index])))[0]
        res_X = np.dstack((p[0], ee))
        res_E = np.array(enhancer_index)
        del enhancer_index
        del ee
        gc.collect()
    gc.collect()
    return (res_X, res_E)
    

# defining all the parameters
num_class = 2           # number of output
p_drop_conv = 0.3       # dropout rate for cov_layer
p_drop_hidden = 0.5     # dropout rate for hidden_layer

mini_batch_size = 50    # [40 - 100]
lr = 0.001              # learning rate
epchs = 15              # number of iteration

feat_num = 17           # number of chip features
chip_motif_len = 6      # length of motif matrix
chip_motif_num = 50     # number of motifs 
hidden_unit_num = 100   # number of hidden units
max_enhancer_num = 3    # max number of enhancers included for each promoter
n_jobs = 15

max_pool_shape = (1, 5000000) # max pool maxtrix size

# graphic parameters
X = T.ftensor4()
Y = T.fmatrix()
w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
w2 = init_weights((chip_motif_num, hidden_unit_num))
w3 = init_weights((hidden_unit_num, num_class))
noise_py_x = model(X, w1, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
py_x = model(X, w1, w2, w3, max_pool_shape, 0., 0.)
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2, w3]
updates = RMSprop(cost, params, lr)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

# load input
dat_X_P = gen_chip_feature("datX_P.dat", 29)    # promoter 3k
dat_X_E = gen_chip_feature("datX_E.dat", 19)    # enhancer 2k
dat_Y = gen_target("datY.dat", 2)               # target 
matches_dist = gen_matches("matches_distance.txt")

# model 0 that based only on promoters
res = learning(dat_X_P, dat_Y, 0, epchs, mini_batch_size)

# update trX
enhancers = []
for i in range(0, 60):
    (trX_update, enh_tmp) = update_X(matches_dist, max_enhancer_num, dat_X_P, dat_X_E, dat_Y, n_jobs)
    # append the enhancers
    enhancers.append(enh_tmp)
    # re-initilize all the parameters
    X = T.ftensor4()
    Y = T.fmatrix()
    w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
    w2 = init_weights((chip_motif_num, hidden_unit_num))
    w3 = init_weights((hidden_unit_num, num_class))
    noise_py_x = model(X, w1, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
    py_x = model(X, w1, w2, w3, max_pool_shape, 0., 0.)
    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w1, w2, w3]
    updates = RMSprop(cost, params, lr)
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)
    # now predict
    res += learning(trX_update, dat_Y, i, epchs, mini_batch_size)
    gc.collect()

np.savetxt("loops_pred.300k.raw.rep1.txt", np.array(enhancers).reshape(-1, max_enhancer_num))
np.savetxt("res_enh.300k.raw.rep1.txt", np.array(res))

# predict against randomly assigned enhancers
res = []
for k in range(15, 50):
    trX_update = []
    for i in range(len(dat_X_P)):
        p = dat_X_P[i]    
        if i in matches_dist:
            if(len(matches_dist[i])>max_enhancer_num):
                enhancer_index = random.sample(matches_dist[i], max_enhancer_num)
            else:
                enhancer_index = matches_dist[i] + [-1]*(max_enhancer_num - len(matches_dist[i]))
            shuffle(enhancer_index)
            e = np.dstack(dat_X_E[enhancer_index])
            p = np.dstack((p, e))
        else: # no nehancers
            enhancer_index = max_enhancer_num*[-1]
            e = np.dstack(dat_X_E[enhancer_index])
            p = np.dstack((p, e)) 
        trX_update.append(p)
    trX_update = np.array(trX_update)    
    # re-initilize all the parameters
    X = T.ftensor4()
    Y = T.fmatrix()
    w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
    w2 = init_weights((chip_motif_num, hidden_unit_num))
    w3 = init_weights((hidden_unit_num, num_class))
    noise_py_x = model(X, w1, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
    py_x = model(X, w1, w2, w3, max_pool_shape, 0., 0.)
    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w1, w2, w3]
    updates = RMSprop(cost, params, lr)
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)
    # now predict
    res += learning(trX_update, dat_Y, k, epchs, mini_batch_size)
    gc.collect()
np.savetxt("res_enh.random.txt", np.array(res))