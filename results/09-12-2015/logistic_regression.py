import theano
from theano import tensor as T
import numpy as np
from StringIO import StringIO 
import random 

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

def load_data(n):
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
  
    
    
dataset = "mESC-zy27.gene.expr.sel.feat"
epos = 200
min_batch = 500
n=2
sigma = 0.05

trX, trY, teX, teY, teX_enhancer, pair = load_data(num_class)
trX = trX.reshape(8598, -1)
teX = teX.reshape(2547, -1)

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((trX.shape[1], n))

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * sigma]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

for i in range(epos):
    for start, end in zip(range(0, len(trX), min_batch), range(min_batch, len(trX), min_batch)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))    


pred = predict(teX)
print np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))    



np.savetxt("tmp.txt", pred)
#preds = predict(teX)
#for i in xrange(len(predict(teX))):
#    print preds[i], np.argmax(teY, axis=1)[i]

