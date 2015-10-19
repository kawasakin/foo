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

def update_X_random(P, E, matches_dist, max_enhancer_num):
    trX_update = []
    for i in range(len(P)):
        p = P[i]    
        if i in matches_dist:
            if(len(matches_dist[i])>max_enhancer_num):
                enhancer_index = random.sample(matches_dist[i], max_enhancer_num)
            else:
                enhancer_index = matches_dist[i] + [-1]*(max_enhancer_num - len(matches_dist[i]))
            shuffle(enhancer_index)
            e = np.dstack(E[enhancer_index])
            p = np.dstack((p, e))
        else: # no nehancers
            enhancer_index = max_enhancer_num*[-1]
            e = np.dstack(E[enhancer_index])
            p = np.dstack((p, e)) 
        trX_update.append(p)
    trX_update = np.array(trX_update)  
    return trX_update
    
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
mini_batch_size = 100    # [40 - 100]
lr = 0.001              # learning rate
epchs = 15              # number of iteration
feat_num = 17           # number of chip features
chip_motif_len = 6      # length of motif matrix
chip_motif_num = 30     # number of motifs 
hidden_unit_num = 100   # number of hidden units
max_enhancer_num = 3    # max number of enhancers included for each promoter
n_jobs = 1
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
dat_X_P = gen_chip_feature("datX_P.dat", 100)    # promoter 5k
dat_Y = gen_target("datY.dat", 2)               # target 

# model 0 that based only on promoters
res = learning(dat_X_P, dat_Y, 0, epchs, mini_batch_size)


l1_pool_feat = T.flatten(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), max_pool_shape), outdim=2)
l1_pool_feat_cal = theano.function(inputs=[X], outputs=l1_pool_feat, allow_input_downcast=True)
feats = l1_pool_feat_cal(dat_X_P)

np.savetxt("tmp.txt", feats)
# R
data = read.table("tmp.txt")
AUC <- function(pos.scores, neg.scores){
    res = c()
    for(i in 1:100){
        res = c(res, mean(sample(pos.scores,1000,replace=T) > sample(neg.scores,1000,replace=T)))         
    }
    return(mean(res))
}
orders = order(sapply(1:ncol(data), function(i) AUC(data[4431:nrow(data),i], data[1:4431,i])))
orders = data.frame(a = orders, b = 0:(length(orders)-1))
orders[order(orders$a),2]
# R-end

feat_order = \
[14,  4,  9, 11, 25, 28,
  5,  3, 21,  0, 20,  8,
  7, 16, 18, 17, 29, 27, 
 19, 12, 10, 15,  1, 22, 
 13, 24,  2, 23,  6, 26]


w1.get_value()[14]
l1_cov2d = rectify(conv2d(X, w1, border_mode='valid'))
l1_cov2d_cal = theano.function(inputs=[X], outputs=l1_cov2d, allow_input_downcast=True)
feats = l1_cov2d_cal(dat_X_P)
np.savetxt("feat1_heatmap.txt", feats[:, 14, 0, :])

feat1 = np.zeros([17, 6])
num = 0
for i in range(dat_X_P.shape[0]):
    if np.argmax(feats[:, 14, 0, :], axis=1)[i] > 0:
        feat1 += dat_X_P[i, 0, :, np.argmax(feats[:, 14, 0, :], axis=1)[i]:(np.argmax(feats[:, 14, 0, :], axis=1)[i]+6)]
        num += 1
feat1 = feat1/num 
np.savetxt("motif1.txt")

final = []
l_X = dat_X_P
trY = dat_Y
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

res_random = res
# update trX
enhancers = []
for i in range(1, 60):
    # postive prediction
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
            
np.savetxt("loops.300K.3E.raw.rep4.txt", np.array(enhancers).reshape(-1, max_enhancer_num))
np.savetxt("res_enh.300k.3E.rep4.txt", np.array(res))

for i in range(1, 59):
    # random control
    trX_update_random = update_X_random(dat_X_P, dat_X_E, matches_dist, max_enhancer_num)
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
    res_random += learning(trX_update_random, dat_Y, i, epchs, mini_batch_size)
    gc.collect()
np.savetxt("res_enh.300k.3E.random.txt", np.array(res_random))


########################################################################
########################################################################

max_pool_shape = (1, 5000000) # max pool maxtrix size
#l1 = rectify(conv2d(X, w1, border_mode='valid'), )
l1 = T.flatten(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), max_pool_shape), outdim=2)
l1_conv2d = conv2d(X, w1, border_mode='valid')

aa = theano.function(inputs=[X], outputs=l1, allow_input_downcast=True)
l1_conv2d_cal = theano.function(inputs=[X], outputs=l1_conv2d, allow_input_downcast=True)

a = aa(l_X)
np.savetxt("tmp.txt", a)


l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
pyx = softmax(T.dot(l2, w3))


feat1 = w1.get_value()[24].reshape(1,1,17,6)
a = l1_conv2d_cal(l_X)
aaaa = np.argmax(a.reshape(11041,24), axis=1)

tmp_n = np.zeros((17, 6))
for i in range(4431):
    tmp_n = tmp_n + l_X[i].reshape(17, 29)[:,aaaa[i]:(aaaa[i]+6)]

tmp_n = tmp_n/4431

tmp_p = np.zeros((17, 6))
for i in range(4431, 11041):
    tmp_p = tmp_p + l_X[i].reshape(17, 29)[:,aaaa[i]:(aaaa[i]+6)]

tmp_p = tmp_p/6610
tmp_p = np.array(tmp_p)
np.savetxt("feature1_p.txt", tmp_p)
np.savetxt("feature1_n.txt", tmp_n)








