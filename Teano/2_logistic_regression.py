# import library
import theano
from theano import tensor as T
import numpy as np
from load import mnist

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# trX, trY is training data
# teX, teY is training data
# 60000 for training 
# 10000 for testing
trX, teX, trY, teY = mnist(onehot=True)

# symbolic 
X = T.fmatrix()
Y = T.fmatrix()
w = theano.shared(floatX(np.random.randn(784, 10) * 0.01))

# model
py_x = T.nnet.softmax(T.dot(X, w))
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(100): # train for 100 times
    # mini-batch size = 128
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))
