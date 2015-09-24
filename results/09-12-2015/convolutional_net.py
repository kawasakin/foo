import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from StringIO import StringIO 
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import random 
from scipy.stats.stats import pearsonr
srng = RandomStreams()
import collections 

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

def model(X, w1, w2, w3, Max_Pooling_Shape, p_drop_conv, p_drop_hidden):
    l1 = T.flatten(dropout(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
    pyx = softmax(T.dot(l2, w3))
    return pyx


#############################
num_class = 2
p_drop_conv = 0.3
p_drop_hidden = 0.5

mini_batch_size = 40 #[40 - 100]
lr = 0.01 # [0.0001 - 0.89 (too slow)] [0.001 - 0.90]
epchs = 200

feat_num = 17
convolution_window_width = 15
convolution_unit_num = 50 #50-0.827; 100-0.79 
hidden_unit_num = 100


with open("dat_X_P.txt") as fin:
    dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
dat_X = dat.reshape(-1, 1, 17, 99)

with open("dat_X_E.txt") as fin:
    dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
dat_E_X = dat.reshape(-1, 1, 17, 79)

with open("dat_Y.txt") as fin:
    dat_Y = np.loadtxt(StringIO(fin.read()), dtype="float32") 
dat_Y = dat_Y + 1
dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(num_class-x) for x in list(dat_Y)])

loops = collections.defaultdict(list)
with open("interaction.txt") as fin:
    for line in fin:
        [p, e] = line.strip().split()
        loops[int(p)-1].append(int(e)-1)

train_index = list(set(xrange(dat_X.shape[0])) - set(loops.keys()))
test_index = sorted(loops.keys())
random.shuffle(train_index)
eval_index = train_index[:870]
train_index = train_index[870:]

#train_index = random.sample(xrange(dat_X.shape[0]), dat_X.shape[0]*4/5)
#test_index  = sorted(list(set(range(dat_X.shape[0]))-set(train_index)))

trX = dat_X[train_index]
trY = dat_Y[train_index]
teX = dat_X[test_index]
teY = dat_Y[test_index]
evX = dat_X[eval_index]
evY = dat_Y[eval_index]

X = T.ftensor4()
Y = T.fmatrix()

w1 = init_weights((convolution_unit_num, 1, feat_num, convolution_window_width))
w2 = init_weights((convolution_unit_num, hidden_unit_num))
w3 = init_weights((hidden_unit_num, num_class))

noise_py_x = model(X, w1, w2, w3, (1, 500), p_drop_conv, p_drop_hidden)
py_x = model(X, w1, w2, w3, (1, 500), 0., 0.)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

params = [w1, w2, w3]
updates = RMSprop(cost, params, lr)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

index = np.r_[:trX.shape[0]]
for i in range(epchs):
    random.shuffle(index)
    for start, end in zip(range(0, len(trX), mini_batch_size), range(mini_batch_size, len(trX), mini_batch_size)):
        cost = train(trX[index][start:end], trY[index][start:end])
    print cost, np.mean(np.argmax(evY, axis=1) == np.argmax(predict(evX), axis=1))

print np.mean(np.argmax(dat_Y[test_index], axis=1) == np.argmax(predict(dat_X[test_index]), axis=1))
pearsonr(range(len(test_index)), predict(dat_X[test_index])[:,1])

a = predict(dat_X[test_index])
np.savetxt("tmp", a, delimiter="t")

sumCorrect = 0
for i in test_index:
    X = dat_X[i]
    Y = dat_Y[i]
    for j in loops[i]:
        X = np.dstack((X, dat_E_X[j]))
    sumCorrect = sumCorrect + [0,1][np.argmax(Y, axis=0) == np.argmax(predict(X.reshape(1, 1, 17, -1))[0], axis=0)]

keys =  list(loops.keys())        
random.shuffle(keys)
loops_shuffled = dict([(key, loops[key]) for key in keys])
sumCorrect = 0
for i in test_index:
    X = dat_X[i]
    Y = dat_Y[i]
    for j in loops_shuffled[i]:
        X = np.dstack((X, dat_E_X[j]))
    sumCorrect = sumCorrect + [0,1][np.argmax(Y, axis=0) == np.argmax(predict(X.reshape(1, 1, 17, -1))[0], axis=0)]



a = [ list(gene_mapper[:,1]).index(x) if x in gene_mapper[:,1] else None for x in loops[:,0] ]
teX_tmp = teX[a]
teY_tmp = teY[a]
pair = loops[:,1]
# with enhancer
a = np.dstack((teX_tmp[0], dat_E_X[pair[0]]))
b = np.dstack((teX_tmp[1], dat_E_X[pair[1]]))
a = np.vstack((a,b))
#
for i in xrange(2, teX_tmp.shape[0]):
    b = np.dstack((teX_tmp[i], dat_E_X[pair[i]]))
    a = np.vstack((a,b))

teX_tmp = a.reshape(teY_tmp.shape[0], 1, 17, 178)    
print np.mean(np.argmax(teY_tmp, axis=1) == np.argmax(predict(teX_tmp), axis=1))

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



    