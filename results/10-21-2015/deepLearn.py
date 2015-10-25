######################################### import all packages
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
from operator import itemgetter

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
    loops =  collections.defaultdict(float)
    with open(fname) as fin:
        for line in fin:
            p = int(line.strip().split()[0])-1
            e = int(line.strip().split()[1])-1
            matches[p].append(e)
            loops[(p, e)] = float(line.strip().split()[3])
    return (matches, loops)

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
    
def model(X, X2, w1, w2, Max_Pooling_Shape, p_drop_conv):
    # pool the features for raw data from first layer
    l1a = T.flatten(dropout(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    # add one element to each row of l1
    l1b = T.concatenate([l1a, T.reshape(X2, (-1, 1))], axis=1)    
    pyx = softmax(T.dot(l1b, w2))
    return pyx

def learning(trX1, trX2, trY, ii, epchs, mini_batch_size):
    # model 0 - learning from promoter
    final = []
    index = np.array(range(trX1.shape[0]))
    for i in range(epchs):
        random.shuffle(index)
        for start, end in zip(range(0, len(trX1), mini_batch_size), range(mini_batch_size, len(trX1), mini_batch_size)):
            cost = train(trX1[index][start:end], trX2[index][start:end], trY[index][start:end])
        preds_tr = predict(trX1, trX2)
        costs = np.mean(cross_entropy(preds_tr, trY))
        accur = np.mean(np.argmax(trY, axis=1) == np.argmax(preds_tr, axis=1))
        print(ii, costs, accur)
        final.append((ii, costs, accur))
    gc.collect()
    return final

def cross_entropy(a, y):
    return np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))

def random_assign(num, M, L):
    res_assgn = []
    res_loops = []
    for i in range(num):
        if len(M[i])>0:
            tmp = (i, random.choice(M[i]))
            res_assgn.append(tmp)
            res_loops.append(L[tmp])
        else:
            res_assgn.append((i, -1))
            res_loops.append(0.0)
    res_assgn = np.array(res_assgn)
    res_loops = np.array(res_loops).reshape(-1, 1)
    return (res_assgn, res_loops)    

def random_drop(A, L, p):
    index_drop = random.sample(list(range(A.shape[0])), round(A.shape[0]*p))
    A[index_drop, 1] = -1
    L[index_drop] = 0.0
    return (A, L)

def get_all_EPU(P, E, M, L):
    EPU_ALL = []
    LOP_ALL = []
    for i in range(P.shape[0]):
        p = P[i].reshape(1, 1, 17, -1)
        m = M[i] 
        if len(m)>0:
            EPU_ALL.append(np.concatenate((np.array([p]*len(m)).reshape(-1,1,17,100), E[m].reshape(-1,1,17,100)), axis=3))
            LOP_ALL.append(np.array(list(map(lambda x: L[(i, x)], m))).reshape(-1, 1))
        else:
            EPU_ALL.append(np.concatenate((p, E[-1].reshape(-1,1,17,100)), axis=3))       
            LOP_ALL.append(np.array(0.0).reshape(-1,1))
    return (EPU_ALL, LOP_ALL)

def EPU_update(E, L, Y, M):
    EPU_UPDATED = []
    LOP_UPDATED = []
    MCH_UPDATED = []
    num = len(E)
    PREDS_ALL = list(map(lambda i: predict(E[i], L[i])[:,1], range(num)))
    TARGS_ALL = list(map(lambda i: Y[[i]*(E[i].shape[0]), 1], range(num)))
    COSTS_ALL = list(map(lambda i: cross_entropy(PREDS_ALL[i], TARGS_ALL[i]), range(num)))
    for i in range(len(COSTS_ALL)):
        if len(M[i]) > 0:
            id_min = np.argmin(np.array(COSTS_ALL[i]))
            EPU_UPDATED.append(E[i][id_min])
            LOP_UPDATED.append(L[i][id_min])
            MCH_UPDATED.append(M[i][id_min])
        else:
            EPU_UPDATED.append(E[i][0])
            LOP_UPDATED.append(L[i][0])
            MCH_UPDATED.append(-1)
    return(np.array(MCH_UPDATED), np.array(EPU_UPDATED), np.array(LOP_UPDATED))

# defining all the parameters
num_class = 2           # number of output
p_drop_conv = 0.2       # dropout rate for cov_layer
mini_batch_size = 50    # [40 - 100]
lr = 0.001              # learning rate
epchs = 15              # number of iteration
feat_num = 17           # number of chip features
chip_motif_len = 6      # length of motif matrix
chip_motif_num = 15     # number of motifs 
max_pool_shape = (1, 5000000) # max pool maxtrix size

# graphic parameters
X = T.ftensor4()
X2 = T.fmatrix()
Y = T.fmatrix()
# no hidden layer
w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
w2 = init_weights((chip_motif_num+1, num_class))
noise_py_x = model(X, X2, w1, w2, max_pool_shape, p_drop_conv)
py_x = model(X, X2, w1, w2, max_pool_shape, 0.)
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2]
updates = RMSprop(cost, params, lr)
train = theano.function(inputs=[X, X2, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X, X2], outputs=py_x, allow_input_downcast=True)

# load input
dat_X_P = gen_chip_feature("datX_P.dat", 100)    # promoter 5k
dat_X_E = gen_chip_feature("datX_E.dat", 100)    # enhancer 5k
dat_Y = gen_target("datY.dat", 2)               # target 
(matches, loops) = gen_matches("access.txt")

# add an empty enhancer to the end
dat_X_E = np.concatenate((dat_X_E, np.zeros((1, 1, 17, 100))), axis=0)
# all possible EPU for every promoter
(EPU_ALL, LOP_ALL) = get_all_EPU(dat_X_P, dat_X_E, matches, loops)
# random assign enhancer to promoters
(assgn0, loop0) = random_assign(dat_X_P.shape[0], matches, loops)
# update enhancer-promoter unit (EPU) as the 0th model      
EPU0 = np.concatenate((dat_X_P, dat_X_E[assgn0[:,1]]), axis=3)
# learn paramters from EPU0      
res = learning(EPU0, loop0, dat_Y, 0, epchs, mini_batch_size)
#########################################
freq = collections.defaultdict(int)
# update EPU based on the learned weights
for mm in range(1, 50):
    (MCH_UPDATED, EPU_UPDATED, LOP_UPDATED) = EPU_update(EPU_ALL, LOP_ALL, dat_Y, matches)
    for i in range(len(MCH_UPDATED)):
        if(MCH_UPDATED[i]!=-1):
            freq[(i, MCH_UPDATED[i])] += 1
    # randomly drop our 20% of enhancers.
    ind_sampled = random.sample(list(range(EPU_UPDATED.shape[0])), round(EPU_UPDATED.shape[0]*0.2)) 
    EPU_UPDATED[ind_sampled] = np.concatenate((dat_X_P[ind_sampled], dat_X_E[[-1]*len(ind_sampled)]), axis=3)
    LOP_UPDATED[ind_sampled] = 0.0
    # initlize all the parameters of models.
    w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
    w2 = init_weights((chip_motif_num+1, num_class))
    # learn the model again.
    res = res + learning(EPU_UPDATED, LOP_UPDATED, dat_Y, mm, epchs, mini_batch_size)

with open("rep1.targets.txt", "w") as fout:
    for (key, value) in freq.items():
        fout.write("\t".join(map(str, key)) + "\t" + str(value)+"\n") 

#np.savetxt("rep1.res.txt", np.array(res[15:]).reshape(-1, 3))
#np.savetxt("tmp", np.array(res[:15]))
