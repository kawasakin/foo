import theano
from theano import tensor as T
import numpy as np
from StringIO import StringIO 
import random 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
    with open(dataset) as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32")    
    dat = dat[range(num_cases/2) + range((dat.shape[0]-num_cases/2), dat.shape[0]), ]  
    dat_X = dat[:,:(dat.shape[1]-1)]    
    dat_Y = dat[:,dat.shape[1]-1]
    i = int(max(dat_Y))+1
    for p in [min(dat_Y)-0.1] + [np.percentile(dat_Y, x*100) for x in np.arange(0, 1, 1/float(n))[1:]] + [max(dat_Y)+0.1]:
        dat_Y[dat_Y <= p] = i
        i += 1 
    dat_Y = dat_Y - min(dat_Y) + 1
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat_Y)])
    train_index = random.sample(xrange(dat.shape[0]), dat.shape[0]*4/5)
    test_index  = sorted(list(set(range(dat.shape[0]))-set(train_index)))
    return [dat_X[train_index,], dat_Y[train_index,], dat_X[test_index,], dat_Y[test_index,]]

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

dataset = "mESC-zy27.gene.expr.sel.feat"
num_cases = 10000
num_class = 2
lr = 0.005 # learning rate
epos = 200
min_batch = 400

trX, trY, teX, teY = load_data(dataset, num_cases, num_class)

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((trX.shape[1], 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, num_class))

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, lr=lr)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

for i in range(epos):
    for start, end in zip(range(0, len(trX), min_batch), range(min_batch, len(trX), min_batch)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))

pred = predict(teX)
np.savetxt("tmp.txt", pred)
