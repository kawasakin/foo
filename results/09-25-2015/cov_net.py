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

num_class = 2
p_drop_conv = 0.3
p_drop_hidden = 0.5

mini_batch_size = 400 #[40 - 100]
lr = 0.001 # [0.0001 - 0.89 (too slow)] [0.001 - 0.90]
epchs = 20

feat_num = 17
chip_motif_len = 10
chip_motif_num = 50
hidden_unit_num = 100

max_pool_shape = (1, 500000000)

dat_X = gen_chip_feature("datX_P.dat", 59)
dat_Y = gen_target("datY.dat", 2)
matches = gen_matches("matches.txt")

# Enhancers
dat_X_E = gen_chip_feature("datX_E.dat", 39)

# seperate training and testing data
train_index = random.sample(xrange(dat_X.shape[0]), dat_X.shape[0]*4/5)
test_index  = sorted(list(set(range(dat_X.shape[0]))-set(train_index)))

# symbolic variables
X = T.ftensor4()
Y = T.fmatrix()
Z = T.fmatrix()

w1 = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
w2 = init_weights((chip_motif_num, hidden_unit_num))
w3 = init_weights((hidden_unit_num, num_class))


noise_py_x = model(X, w1, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
py_x = model(X, w1, w2, w3, max_pool_shape, 0., 0.)

cost = T.mean(T.nnet.categorical_crossentropy(Z, Y))
params = [w1, w2, w3]
updates = RMSprop(cost, params, lr)

train = theano.function(inputs=[Z, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

# = theano.function(inputs=[X], outputs=, allow_input_downcast=True)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2, w3]
updates = RMSprop(cost, params, lr)

for i in train_index[:100]:
    promoter = dat_X[i]
    for j in matches[i]:
        enhancer = dat_X_E[j]
        promoter = np.dstack((promoter, enhancer))
    X_tmp = promoter.reshape(1, 1, 17, -1)
    print noise_py_x(X_tmp)
    

train = theano.function(inputs=[C, S, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[C, S], outputs=py_x, allow_input_downcast=True)

index = np.r_[:trC.shape[0]]
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trC), mini_batch_size), range(mini_batch_size, len(trC), mini_batch_size)):
        cost = train(trC[index][start:end], trS[index][start:end], trY[index][start:end])
        print cost
    print cost, np.mean(np.argmax(evY, axis=1) == np.argmax(predict(evC, evS), axis=1))

# test on the test set 
teX_S = gen_seq_feature("teX.fa", 4000)
teX_C = gen_chip_feature("teX_chip.dat")
teY = gen_target("teY.dat", 2)
print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX_C, teX_S), axis=1))

pred = predict(teX_C, teX_S)
res = np.array((np.argmax(teY, axis=1), pred[:,1])).T
np.savetxt("teY_prob.txt", res)

loops = collections.defaultdict(list)
with open("Enhancer_promoter_matches.txt") as fin:
    for line in fin:
        [p, e] = line.strip().split()
        loops[int(p)-1].append(int(e)-1)

# adding enhancers
teE_S = gen_seq_feature("Enhancers.fa", 2000)
teE_C = gen_chip_feature("teX_Enhancer.txt", 39)

res1 = []
res2 = []
for i in loops:
    S = teX_S[i]
    C = teX_C[i]    
    for j in loops[i]:
        S = np.dstack((S, teE_S[j]))
        C = np.dstack((C, teE_C[j]))
    res1.append(np.argmax(predict(C.reshape(1, 1, 17, -1), S.reshape(1, 1, 4, -1))))
    res2.append(np.argmax(teY[i]))
np.mean(np.array(res1)==np.array(res2))
        

# shuffle enhancers 
keys = loops.keys()
keys_shuffle = sorted(keys, key=lambda k: random.random())
a = dict(zip(keys, keys_shuffle))
loops_shuffled = dict([(key, loops[a[key]])for key in keys])

res1 = []
res2 = []
for i in loops_shuffled:
    S = teX_S[i]
    C = teX_C[i]    
    for j in loops_shuffled[i]:
        S = np.dstack((S, teE_S[j]))
        C = np.dstack((C, teE_C[j]))
    res1.append(np.argmax(predict(C.reshape(1, 1, 17, -1), S.reshape(1, 1, 4, -1))))
    res2.append(np.argmax(teY[i]))
np.mean(np.array(res1)==np.array(res2))

# 0.87874360847333821

#train_index = list(set(xrange(dat_X.shape[0])) - set(loops.keys()))
#test_index = sorted(loops.keys())

#train_index = random.sample(xrange(dat_X.shape[0]), dat_X.shape[0]*4/5)
#test_index  = sorted(list(set(range(dat_X.shape[0]))-set(train_index)))






#print np.mean(np.argmax(dat_Y[test_index], axis=1) == np.argmax(predict(dat_X[test_index]), axis=1))
#pearsonr(range(len(test_index)), predict(dat_X[test_index])[:,1])
#
#a = predict(dat_X[test_index])
#np.savetxt("tmp", a, delimiter="t")
#
#sumCorrect = 0
#for i in test_index:
#    X = dat_X[i]
#    Y = dat_Y[i]
#    for j in loops[i]:
#        X = np.dstack((X, dat_E_X[j]))
#    sumCorrect = sumCorrect + [0,1][np.argmax(Y, axis=0) == np.argmax(predict(X.reshape(1, 1, 17, -1))[0], axis=0)]
#
#keys =  list(loops.keys())        
#random.shuffle(keys)
#loops_shuffled = dict([(key, loops[key]) for key in keys])
#sumCorrect = 0
#for i in test_index:
#    X = dat_X[i]
#    Y = dat_Y[i]
#    for j in loops_shuffled[i]:
#        X = np.dstack((X, dat_E_X[j]))
#    sumCorrect = sumCorrect + [0,1][np.argmax(Y, axis=0) == np.argmax(predict(X.reshape(1, 1, 17, -1))[0], axis=0)]
#
#
#
#a = [ list(gene_mapper[:,1]).index(x) if x in gene_mapper[:,1] else None for x in loops[:,0] ]
#teX_tmp = teX[a]
#teY_tmp = teY[a]
#pair = loops[:,1]
## with enhancer
#a = np.dstack((teX_tmp[0], dat_E_X[pair[0]]))
#b = np.dstack((teX_tmp[1], dat_E_X[pair[1]]))
#a = np.vstack((a,b))
##
#for i in xrange(2, teX_tmp.shape[0]):
#    b = np.dstack((teX_tmp[i], dat_E_X[pair[i]]))
#    a = np.vstack((a,b))
#
#teX_tmp = a.reshape(teY_tmp.shape[0], 1, 17, 178)    
#print np.mean(np.argmax(teY_tmp, axis=1) == np.argmax(predict(teX_tmp), axis=1))
#
#
#
#np.random.shuffle(pair[:,1])
#a = np.dstack((teX[0], dat_e[pair[0][1]]))
#b = np.dstack((teX[1], dat_e[pair[1][1]]))
#a = np.vstack((a,b))
#
#for i in xrange(2, teX.shape[0]):
#    b = np.dstack((teX[i], dat_e[pair[i][1]]))
#    a = np.vstack((a,b))
#teX_s = a.reshape(teY.shape[0], 1, 17, 178)    



    