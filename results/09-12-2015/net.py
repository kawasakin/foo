import theano
from theano import tensor as T
import numpy as np
from StringIO import StringIO 
import random 

def load_data(dataset, n):
    with open(dataset) as fin:
        dat = np.loadtxt(StringIO(fin.read()), dtype="float32")
    dat_X = dat[:,:(dat.shape[1]-1)]        
    dat_Y = np.array([ [0]*(x-1) + [1] + [0]*(n-x) for x in list(dat[:,dat.shape[1]-1])])
    train_index = random.sample(xrange(dat.shape[0]), dat.shape[0]*4/5)
    test_index = list(set(range(dat.shape[0]))-set(train_index))
    return [dat_X[train_index,], dat_Y[train_index,], dat_X[test_index,], dat_Y[test_index,]]

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx

n = 2
trX, trY, teX, teY = load_data("mESC_sample_features.txt", n)
X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((trX.shape[1], 625))
w_o = init_weights((625, n))

py_x = model(X, w_h, w_o)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(400):
    for start, end in zip(range(0, len(trX), 300), range(300, len(trX), 300)):
        cost = train(trX[start:end], trY[start:end])
    print cost
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

preds = predict(teX)
for i in xrange(len(predict(teX))):
    print preds[i], np.argmax(teY, axis=1)[i]
    




