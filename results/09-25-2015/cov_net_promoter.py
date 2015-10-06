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


num_class = 2
p_drop_conv = 0.3
p_drop_hidden = 0.5

mini_batch_size = 50 #[40 - 100]
lr = 0.001 # [0.0001 - 0.89 (too slow)] [0.001 - 0.90]
epchs = 50

feat_num = 17
chip_motif_len = 6
chip_motif_num = 50
hidden_unit_num = 100
max_enhancer_num = 3

max_pool_shape = (1, 500000000)

dat_X = gen_chip_feature("datX_P.dat", 29)      # promoter 3k
dat_Y = gen_target("datY.dat", 2)               # target 

train_index = random.sample(range(dat_X.shape[0]),  dat_X.shape[0]*4/5)
test_index  = list(set(range(dat_X.shape[0])) - set(train_index))

# get train and test promoter part
trX = dat_X[train_index]
trY = dat_Y[train_index]
teX = dat_X[test_index]
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
for i in range(100):
    random.shuffle(index)
    for start, end in zip(range(0, len(trX), mini_batch_size), range(mini_batch_size, len(trX), mini_batch_size)):
        cost = train(trX[index][start:end], trY[index][start:end])
    preds_tr = predict(trX)
    preds_te = predict(teX)
    print np.mean(cross_entropy(predict(trX), trY)), np.mean(np.argmax(trY, axis=1) == np.argmax(predict(trX), axis=1)), np.mean(cross_entropy(preds_te, teY)), np.mean(np.argmax(teY, axis=1) == np.argmax(preds_te, axis=1)) 
