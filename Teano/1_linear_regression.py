# import library
import numpy
import theano
import theano.tensor as T
from theano import config

rng = numpy.random
N = 400                  # number of points
feats = 784              # number of features
D = (rng.randn(N, feats).astype(numpy.float32), rng.randint(size=N, low=0, high=2).astype(numpy.float32))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x", dtype=config.floatX)
y = T.vector("y", dtype=config.floatX)
# shared variables
w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name="w")
b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))        # Probability that target = 1
prediction = p_1 > 0.5                         # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)  # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()     # The cost to minimize with L2 penality
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

predict = theano.function(inputs=[x], outputs=prediction)
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
