import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
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

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

def seq_conv(seqs, detector_len, max_seq_len):
    S = np.empty((len(seqs), 1, max_seq_len+(detector_len-1)*2, 4), dtype=theano.config.floatX)
    S.fill(0.25)
    for j in xrange(len(seqs)):
        s = seqs[j]
        for i in xrange(min(len(s), max_seq_len)):
            index = ['A', 'C', 'G', 'T'].index(s[i])
            S[j][0][i+detector_len-1][0] = 0
            S[j][0][i+detector_len-1][1] = 0
            S[j][0][i+detector_len-1][2] = 0
            S[j][0][i+detector_len-1][3] = 0
            S[j][0][i+detector_len-1][index] = 1
    return S
    
def main():    
    m = 3; max_seq_len=120
    fname = "data/dream5/chipseq/TF_23_CHIP_100_full_genomic.seq"
    seqs = []; labels = []; 
    with open(fname) as fin:
        for line in fin:
            [s, l] = line.split()
            seqs.append(s) 
            labels.append(l)
    S = seq_conv(seqs, m, max_seq_len)
    w = init_weights((10, 1, 4, m))
    conv2d(S, w, border_mode='full')
    
if __name__ == '__main__':
    main()
