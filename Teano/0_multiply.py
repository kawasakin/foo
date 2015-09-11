import theano 
from theano import tensor as T
import time

a = T.scalar()
b = T.scalar()

y = a * b

multiply = theano.function(inputs=[a,b], outputs=y)

start_time = time.time()
print [multiply(1.0,2.0) for x in xrange(10)]
print("--- %s seconds ---" % (time.time() - start_time))
