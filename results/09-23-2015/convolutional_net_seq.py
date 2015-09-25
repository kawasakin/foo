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

def model(X1, X2, w1a, w1b, w2, w3, Max_Pooling_Shape, p_drop_conv, p_drop_hidden):
    # X1 is matrix for ChIP-seq features
    # X2 is matrix for sequence features
    # Max_Pooling_Shape has to be large enough to pool only one element out
    l1a = T.flatten(dropout(max_pool_2d(rectify(conv2d(X1, w1a, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    l1b = T.flatten(dropout(max_pool_2d(rectify(conv2d(X2, w1b, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    l1 = T.concatenate([l1a,l1b], axis=1)
    l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
    pyx = softmax(T.dot(l2, w3))
    return pyx

def model1(X1, w1, w2, w3, Max_Pooling_Shape, p_drop_conv, p_drop_hidden):
    # X1 is matrix for ChIP-seq features
    # X2 is matrix for sequence features
    # Max_Pooling_Shape has to be large enough to pool only one element out
    l1 = T.flatten(dropout(max_pool_2d(rectify(conv2d(X1, w1, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
    pyx = softmax(T.dot(l2, w3))
    return pyx
    
def gen_seq_feature(fname, region_len=4000):
    dat_seq = []
    with open(fname) as fin:
        for line in fin:
            if ">" not in line:
                seq = line.strip().upper()
                if "N" in seq:
                    seq = seq.replace("N", "A")
                dat_seq.append([[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]] [["A", "T", "C", "G"].index(x)] for x in seq])
    dat_seq = np.array(dat_seq)
    dat_seq = np.transpose(dat_seq, (0, 2, 1))
    dat_seq = dat_seq.reshape(-1, 1, 4, region_len)
    return dat_seq

def gen_chip_feature(fname, region_len=79):
    with open(fname) as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_X = dat.reshape(-1, 1, 17, region_len)
    return dat_X
    
def gen_target(fname, n=2):
    with open(fname) as fin:
        dat_Y = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_Y = dat_Y + 1
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat_Y)])
    return dat_Y

num_class = 2
p_drop_conv = 0.3
p_drop_hidden = 0.5

mini_batch_size = 100 #[40 - 100]
lr = 0.001 # [0.0001 - 0.89 (too slow)] [0.001 - 0.90]
epchs = 20

feat_num = 17
seq_motif_len = 20
chip_motif_len = 15
seq_motif_num = 40
chip_motif_num = 50
hidden_unit_num = 100


max_pool_shape = (1, 500000000)

dat_S = gen_seq_feature("trX.fa", 4000)
dat_C = gen_chip_feature("trX_chip.dat")
dat_Y = gen_target("trY.dat", 2)

# random chose 870 as evaluation set
train_index = np.array(range(dat_Y.shape[0]))
random.shuffle(train_index)
eval_index = train_index[:870]
train_index = train_index[870:]
trC = dat_C[train_index]
trS = dat_S[train_index]
trY = dat_Y[train_index]
evS = dat_S[eval_index]
evC = dat_C[eval_index]
evY = dat_Y[eval_index]


C = T.ftensor4()
S = T.ftensor4()
Y = T.fmatrix()
chip_motif_num = 0
w1C = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
w1S = init_weights((seq_motif_num, 1, 4, seq_motif_len))
w2 = init_weights((seq_motif_num + chip_motif_num, hidden_unit_num))
w3 = init_weights((hidden_unit_num, num_class))

noise_py_x = model1(S, w1S, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
py_x = model1(S, w1S, w2, w3, max_pool_shape, 0., 0.)

#noise_py_x = model(C, S, w1C, w1S, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
#py_x = model(C, S, w1C, w1S, w2, w3, max_pool_shape, 0., 0.)
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

#params = [w1C, w1S, w2, w3]
params = [w1S, w2, w3]
updates = RMSprop(cost, params, lr)

#train = theano.function(inputs=[C, S, Y], outputs=cost, updates=updates, allow_input_downcast=True)
#predict = theano.function(inputs=[C, S], outputs=py_x, allow_input_downcast=True)

train = theano.function(inputs=[S, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[S], outputs=py_x, allow_input_downcast=True)

index = np.r_[:trC.shape[0]]
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trC), mini_batch_size), range(mini_batch_size, len(trC), mini_batch_size)):
        cost = train(trC[index][start:end], trS[index][start:end], trY[index][start:end])
        print cost
    print cost, np.mean(np.argmax(evY, axis=1) == np.argmax(predict(evC, evS), axis=1))


index = np.r_[:trS.shape[0]]
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trS), mini_batch_size), range(mini_batch_size, len(trS), mini_batch_size)):
        cost = train(trS[index][start:end], trY[index][start:end])
    print cost, np.mean(np.argmax(evY, axis=1) == np.argmax(predict(evS), axis=1))


# test on the test set 
#teX_S = gen_seq_feature("teX.fa", 4000)
#teX_C = gen_chip_feature("teX_chip.dat")
#teY = gen_target("teY.dat", 2)
#print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX_C, teX_S), axis=1))
print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX_S), axis=1))

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
    for j in loops[i]:
        S = np.dstack((S, teE_S[j]))
    res1.append(np.argmax(predict(S.reshape(1, 1, 4, -1))))
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
    res1.append(np.argmax(predict(S.reshape(1, 1, 4, -1))))
    res2.append(np.argmax(teY[i]))
np.mean(np.array(res1)==np.array(res2))
