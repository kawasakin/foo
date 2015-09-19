import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

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

def model(X, w1, w2, w3, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w1, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    dropout(T.flatten(max_pool_2d(rectify(conv2d(X, w2)), (2,2)), outdim=2), 0.3)
    
    l2a = rectify(conv2d(l1, w2))
    l2b = max_pool_2d(l2a, (2, 2))
    l2 = T.flatten(l2b, outdim=2)
    l2 = dropout(l2, p_drop_conv)

    pyx = softmax(T.dot(l2, w3))
    return l1, l2, pyx

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX = trX[1:10000]
trY = trY[1:10000]
teX = teX[1:10000]
teY = teY[1:10000]

X = T.ftensor4()
Y = T.fmatrix()

w1 = init_weights((32, 1, 3, 3))
w2 = init_weights((128, 32, 3, 3))
w3 = init_weights((128 * 7 * 7, 10))

noise_l1, noise_l2, noise_py_x = model(X, w1, w2, w3, 0.2, 0.5)
l1, l2, py_x = model(X, w1, w2, w3, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2, w3]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 10), range(10, len(trX), 10)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

