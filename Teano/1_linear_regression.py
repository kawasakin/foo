# import library
import theano
from theano import tensor as T
import numpy as np

# simulate data
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# symbolic declaim
X = T.scalar('X')
Y = T.scalar('Y')

# shared variable
w = theano.shared(np.asarray(0., dtype=theano.config.floatX))

# Model
y = X * w

# define the cost
cost = T.mean(T.sqr(y - Y))

# define the gradiant
gradient = T.grad(cost=cost, wrt=w)

# define updates and learning rate - 0.01
updates = [[w, w - gradient * 0.01]]

# compile the function
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# traning (# of steps)
for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)

# output
print w.get_value() #something around 2
