import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from StringIO import StringIO 
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import random 

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

def load_data(dataset, num_cases, n):
    with open("dat_X.txt") as fin:
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

    num = dat_X.shape[0]
    dat_X = dat_X[range(10000) + range(num-10000, num)]
    dat_Y = dat_Y[range(10000) + range(num-10000, num)]

    train_index = random.sample(xrange(dat_X.shape[0]), dat_X.shape[0]*4/5)
    test_index  = sorted(list(set(range(dat_X.shape[0]))-set(train_index)))
    return [dat_X[train_index,], dat_Y[train_index,], dat_X[test_index,], dat_Y[test_index,]]

def model(X, w1, w2, w3, p_drop_conv, p_drop_hidden):
    l1 = T.flatten(dropout(max_pool_2d(rectify(conv2d(X, w1, border_mode='valid')), (1, 90)), p_drop_conv), outdim=2)
    l2 = dropout(rectify(T.dot(l1, w2)), p_drop_hidden)
    pyx = softmax(T.dot(l2, w3))    
    return pyx

dataset = "mESC-zy27.gene.expr.sel.feat"
num_cases = 100
num_class = 2
sigma = 0.05 # learning rate
epos = 200
min_batch = 100
p_drop_conv = 0.3
p_drop_hidden = 0.5

trX, trY, teX, teY = load_data(dataset, num_cases, num_class)

X = T.ftensor4()
Y = T.fmatrix()

w1 = init_weights((70, 1, 17, 10))
w2 = init_weights((70, 200))
w3 = init_weights((200, 2))

noise_py_x = model(X, w1, w2, w3, 0.2, 0.5)
py_x = model(X, w1, w2, w3, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2, w3]
updates = RMSprop(cost, params, lr=0.01)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 400), range(400, len(trX), 400)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))
    