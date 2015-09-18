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
    test_index  = sorted(list(set(range(dat.shape[0]))-set(train_index)))
    return [dat_X[train_index,], dat_Y[train_index,], dat_X[test_index,], dat_Y[test_index,]]

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

n = 2
trX, trY, teX, teY = load_data("mESC_sample_features.txt", n)

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((trX.shape[1], n))

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 500), range(500, len(trX), 500)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))    


pred = predict(teX)
print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))    

pred[,2] = np.argmax(teY, axis=1)
np.append(pred, np.argmax(teY, axis=1), 1)

np.savetxt("tmp.txt", data_Y)
#preds = predict(teX)
#for i in xrange(len(predict(teX))):
#    print preds[i], np.argmax(teY, axis=1)[i]

