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

def gen_seq_feature(fname):
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
    dat_seq = dat_seq.reshape(-1, 1, 4, 4000)
    return dat_seq

def gen_chip_feature(fname):
    with open(fname) as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_X = dat.reshape(-1, 1, 17, 79)
    return dat_X
    
def gen_target(fname, n=2):
    with open(fname) as fin:
        dat_Y = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_Y = dat_Y + 1
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat_Y)])
    return dat_Y

num_class = 2
p_drop_conv = 0.2
p_drop_hidden = 0.5

mini_batch_size = 40 #[40 - 100]
lr = 0.005 # [0.0001 - 0.89 (too slow)] [0.001 - 0.90]
epchs = 200

feat_num = 17
seq_motif_len = 20
chip_motif_len = 15
seq_motif_num = 40
chip_motif_num = 30
hidden_unit_num = 100


max_pool_shape = (1, 500000000)

dat_S = gen_seq_feature("trX.fa")
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

w1C = init_weights((chip_motif_num, 1, feat_num, chip_motif_len))
w1S = init_weights((seq_motif_num, 1, 4, seq_motif_len))
w2 = init_weights((seq_motif_num + chip_motif_num, hidden_unit_num))
w3 = init_weights((hidden_unit_num, num_class))

noise_py_x = model(C, S, w1C, w1S, w2, w3, max_pool_shape, p_drop_conv, p_drop_hidden)
py_x = model(C, S, w1C, w1S, w2, w3, max_pool_shape, 0., 0.)
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

params = [w1C, w1S, w2, w3]
updates = RMSprop(cost, params, lr)

train = theano.function(inputs=[C, S, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[C, S], outputs=py_x, allow_input_downcast=True)

index = np.r_[:trC.shape[0]]
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trC), mini_batch_size), range(mini_batch_size, len(trC), mini_batch_size)):
        cost = train(trC[index][start:end], trS[index][start:end], trY[index][start:end])
    print cost, np.mean(np.argmax(trY, axis=1) == np.argmax(predict(trC, trS), axis=1)), np.mean(np.argmax(evY, axis=1) == np.argmax(predict(evC, evS), axis=1))


#loops = collections.defaultdict(list)
#with open("interaction.txt") as fin:
#    for line in fin:
#        [p, e] = line.strip().split()
#        loops[int(p)-1].append(int(e)-1)

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



    