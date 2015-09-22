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

def load_data_1(n):
    with open("dat_p.txt") as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_p = np.transpose(dat.reshape(-1, 1, 99, 17), (0, 1, 3, 2))

    with open("dat_e.txt") as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_e = np.transpose(dat.reshape(-1, 1, 79, 17), (0, 1, 3, 2))
    
    with open("dat_Y.txt") as fin:
        dat_Y = np.loadtxt(StringIO(fin.read()), dtype="float32")     

    i = int(max(dat_Y))+1
    for p in [min(dat_Y)-0.1] + [np.percentile(dat_Y, x*100) for x in np.arange(0, 1, 1/float(n))[1:]] + [max(dat_Y)+0.1]:
        dat_Y[dat_Y <= p] = i
        i += 1 
    dat_Y = dat_Y - min(dat_Y) + 1
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat_Y)])

    with open("interaction.txt") as fin:
        loops = np.loadtxt(StringIO(fin.read()), dtype="int") - 1
    
    train_index = list(set(xrange(dat_p.shape[0])) - set(loops[:,0]))
    random.shuffle(train_index)
    test_index = loops[:,0]
    
    trX = dat_p[train_index,]
    trY = dat_Y[train_index,]
    teX = dat_p[test_index,]
    teX_enhancer = dat_e
    teY = dat_Y[test_index,]    
    return trX, trY, teX, teY, teX_enhancer, loops

def load_data_1(n):
    with open("dat_p.txt") as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32") 
    dat_X = np.transpose(dat.reshape(-1, 1, 99, 17), (0, 1, 3, 2))
    
    with open("dat_Y.txt") as fin:
        dat_Y = np.loadtxt(StringIO(fin.read()), dtype="float32")     

    i = int(max(dat_Y))+1
    for p in [min(dat_Y)-0.1] + [np.percentile(dat_Y, x*100) for x in np.arange(0, 1, 1/float(n))[1:]] + [max(dat_Y)+0.1]:
        dat_Y[dat_Y <= p] = i
        i += 1 
    dat_Y = dat_Y - min(dat_Y) + 1
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat_Y)])
 
    train_index = random.sample(xrange(dat_X.shape[0]), dat_X.shape[0]*4/5)
    test_index  = sorted(list(set(range(dat_X.shape[0]))-set(train_index)))
    return [dat_X[train_index,], dat_Y[train_index,], dat_X[test_index,], dat_Y[test_index,]]
    
    
def model(X, w1, w2, w3, Max_Pooling_Shape, p_drop_conv, p_drop_hidden):
    l1 = T.flatten(dropout(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), Max_Pooling_Shape), p_drop_conv), outdim=2)
    l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
    pyx = softmax(T.dot(l2, w3))
    return pyx

num_class = 2
p_drop_conv = 0.3
p_drop_hidden = 0.5

trX, trY, teX, teY, teX_enhancer, pair = load_data(num_class)

trX, trY, teX, teY = load_data_1(num_class)

X = T.ftensor4()
Y = T.fmatrix()

w1 = init_weights((70, 1, 17, 10))
w2 = init_weights((70, 200))
w3 = init_weights((200, 2))

noise_py_x = model(X, w1, w2, w3, (1, 500), 0.2, 0.5)
py_x = model(X, w1, w2, w3, (1, 500), 0., 0.)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

params = [w1, w2, w3]
updates = RMSprop(cost, params, lr=0.02)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

for i in range(500):
    for start, end in zip(range(0, len(trX), 200), range(200, len(trX), 200)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))
    
import numpy
# with enhancer
dat_e = teX_enhancer
np.random.shuffle(pair[:,1])
a = np.dstack((teX[0], dat_e[pair[0][1]]))
b = np.dstack((teX[1], dat_e[pair[1][1]]))
a = numpy.vstack((a,b))

for i in xrange(2, teX.shape[0]):
    b = np.dstack((teX[i], dat_e[pair[i][1]]))
    a = numpy.vstack((a,b))
    
a = a.reshape(teY.shape[0], 1, 17, 178)    
print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(a), axis=1))




    