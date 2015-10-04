import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from StringIO import StringIO 
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import random 
from scipy.stats.stats import pearsonr
import collections 
from itertools import chain, combinations
from random import shuffle
import gc
from collections import Counter

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

def all_subsets(ss, num):
    return chain(*map(lambda x: combinations(ss, x), range(num+1)))

def cross_entropy(a, y):
    return np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))

def update_X(start, end, matches, max_enhancer_num, dat_X_P, dat_X_E, dat_Y):
    res = []
    promoters_lst = []
    enhancers_lst = []
    for i in xrange(start, end):
        if i in matches:
            tmp = all_subsets(matches[i], max_enhancer_num)
            enhancer_index = [list(subset) + [-1]*(max_enhancer_num - len(subset)) for subset in tmp]
            enhancer_index = [sorted(x, key=lambda k: random.random()) for x in enhancer_index]
            promoters_lst += [i] * len(enhancer_index)
            enhancers_lst += enhancer_index
        else:
            promoters_lst.append(i)
            enhancers_lst.append(max_enhancer_num*[-1])    
    pp = dat_X_P[promoters_lst]
    ee = np.array(map(np.dstack, dat_X_E[enhancers_lst]))
    dat_X_update = np.dstack((pp.reshape(-1, 17, pp.shape[3]), ee.reshape(-1, 17, ee.shape[3]))).reshape(pp.shape[0], 1, 17, -1)
    preds = predict(dat_X_update)
    costs = cross_entropy(preds[:,1], trY[promoters_lst, 1])
    intervals = [0] + list(accumu(collections.Counter(promoters_lst).values()))
    for i in xrange(len(intervals)-1):
        best_index = np.argmin(costs[intervals[i]:intervals[i+1]])
        res.append(dat_X_update[intervals[i]:intervals[i+1]][best_index])
    gc.collect()
    gc.collect()
    return res

num_class = 2
p_drop_conv = 0.3
p_drop_hidden = 0.5

mini_batch_size = 50 #[40 - 100]
lr = 0.001 # [0.0001 - 0.89 (too slow)] [0.001 - 0.90]
epchs = 10

feat_num = 17
chip_motif_len = 6
chip_motif_num = 50
hidden_unit_num = 100
max_enhancer_num = 3

max_pool_shape = (1, 500000000)

dat_X_P = gen_chip_feature("datX_P.dat", 29)    # promoter 3k
dat_X_E = gen_chip_feature("datX_E.dat", 19)    # enhancer 2k
dat_Y = gen_target("datY.dat", 2)               # target 
matches_dist = gen_matches("matches_distance.txt")
matches_loop = gen_matches("matches_loops.txt")

# add one more fake enhancer to the end of dat_X_E
e = np.zeros((17*19)).reshape(1, 1, 17, -1)  
dat_X_E = np.vstack((dat_X_E, e)) 

tmp = sorted(matches_loop.items(), key=lambda x: x[0]) 
test_index = [x[0] for x in tmp]
train_index = sorted(list(set(range(dat_X_P.shape[0])) - set(test_index)))
# update match_loop
matches_loop = dict([(i, tmp[i][1]) for i in xrange(len(tmp))])
matches_dist = dict([(i, matches_dist[train_index[i]]) for i in xrange(len(train_index))])

# get train and test promoter part
trX = dat_X_P[train_index]
trY = dat_Y[train_index]
teX = dat_X_P[test_index]
teY = dat_Y[test_index]

# Model 0 let's first predict everything only by promoter
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

index = np.array(xrange(trX.shape[0]))
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trX), mini_batch_size), range(mini_batch_size, len(trX), mini_batch_size)):
        cost = train(trX[index][start:end], trY[index][start:end])
    preds_tr = predict(trX)
    preds_te = predict(teX)
    print np.mean(cross_entropy(predict(trX), trY)), np.mean(np.argmax(trY, axis=1) == np.argmax(predict(trX), axis=1)), np.mean(cross_entropy(preds_te, teY)), np.mean(np.argmax(teY, axis=1) == np.argmax(preds_te, axis=1)) 

########################################################################################################################
########################################################################################################################
# add enhancers to promoters for test 
dat_X = []
for i in xrange(len(teX)):
    p = teX[i] 
    if i in matches_loop:
        enhancer_index = matches_loop[i]
        e = np.dstack(dat_X_E[enhancer_index])
        p = np.dstack((p, e))
    else: # no nehancers
        enhancer_index = max_enhancer_num*[-1]
        e = np.dstack(dat_X_E[enhancer_index])
        p = np.dstack((p, e)) 
    dat_X.append(p)
teX = dat_X
# Model 1 randomly assign enhancer to promoters

dat_X = []
for i in xrange(len(trX)):
    p = trX[i]    
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
    dat_X.append(p)
trX_update = np.array(dat_X)

# re-init the weight
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

final = []
index = np.array(xrange(trX.shape[0]))
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trX), mini_batch_size), range(mini_batch_size, len(trX), mini_batch_size)):
        cost = train(trX[index][start:end], trY[index][start:end])
    preds_te = np.array(map(lambda x: predict(x.reshape(1,1,17,-1)), teX)).reshape(-1, 2)
    preds_tr = predict(trX_update)
    if i >= 10:
        final.append((0, 
                np.mean(cross_entropy(preds_tr, trY)), 
                np.mean(np.argmax(trY, axis=1) == np.argmax(preds_tr, axis=1)), 
                np.mean(cross_entropy(preds_te, teY)), 
                np.mean(np.argmax(teY, axis=1) == np.argmax(preds_te, axis=1))
                ))
gc.collect()

# Model 2 update X
for kk in xrange(1, 100):
    print kk
    res = []
    res += update_X(0, 3000, matches_dist, max_enhancer_num, trX, dat_X_E, trY)
    res += update_X(3000, 6000, matches_dist, max_enhancer_num, trX, dat_X_E, trY)
    res += update_X(6000, trX.shape[0], matches_dist, max_enhancer_num, trX, dat_X_E, trY)
    trX_update = np.array(res)
    
    w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
    w2 = init_weights((chip_motif_num, hidden_unit_num))
    w3 = init_weights((hidden_unit_num, num_class))
    
    index = np.array(xrange(trX.shape[0]))
    for i in range(20):
        random.shuffle(index)
        for start, end in zip(range(0, len(trX), mini_batch_size), range(mini_batch_size, len(trX), mini_batch_size)):
            cost = train(trX[index][start:end], trY[index][start:end])
        preds_te = np.array(map(lambda x: predict(x.reshape(1,1,17,-1)), teX)).reshape(-1, 2)
        preds_tr = predict(trX_update)
        if(i>=10):
            final.append((kk, np.mean(cross_entropy(preds_tr, trY)), 
                      np.mean(np.argmax(trY, axis=1) == np.argmax(preds_tr, axis=1)), 
                      np.mean(cross_entropy(preds_te, teY)), 
                      np.mean(np.argmax(teY, axis=1) == np.argmax(preds_te, axis=1))
                     ))
    del res
    del trX_update
    del index
    gc.collect()
                    
