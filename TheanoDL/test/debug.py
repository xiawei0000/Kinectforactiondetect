#ecoding=gbk
import numpy as np
import theano
import theano.tensor as T

import numpy


import theano

def inspect_inputs(i, node, fn):
    print i, node, "\ninput(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "\noutput(s) value(s):", [output[0] for output in fn.outputs]

x = theano.tensor.dscalar('x')
f = theano.function([x], [5 * x],
                    mode=theano.compile.MonitorMode(
                        pre_func=inspect_inputs,
                        post_func=inspect_outputs))
f(3)


"""
x = theano.tensor.dvector('x')

x_printed = theano.printing.Print('this is a very important value')(x)
#要的就是这个，每次调用函数，都会输出变量 

f = theano.function([x], x * 5)
f_with_print = theano.function([x], x_printed * 5)

#this runs the graph without any printing
assert numpy.all( f([1, 2, 3]) == [5, 10, 15])

#this runs the graph with the message, and value printed
assert numpy.all( f_with_print([1, 2, 3]) == [5, 10, 15])
f_with_print([1, 2, 3])
f_with_print([1, 2, 3])
f_with_print([1, 2, 3])
"""

"""
# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.compute_test_value ='warn' # 'off' # Use 'warn' to activate this feature

# configure shared variables
W1val = numpy.random.rand(2, 10, 10).astype(theano.config.floatX)
W1 = theano.shared(W1val, 'W1')
W2val = numpy.random.rand(15, 20).astype(theano.config.floatX)
W2 = theano.shared(W2val, 'W2')

# input which will be of shape (5,10)
x  = T.matrix('x')
# provide Theano with a default test-value
#x.tag.test_value = numpy.random.rand(5, 10)

# transform the shared variable in some way. Theano does not
# know off hand that the matrix func_of_W1 has shape (20, 10)
func_of_W1 = W1.dimshuffle(2, 0, 1).flatten(2).T

# source of error: dot product of 5x10 with 20x10
h1 = T.dot(x, func_of_W1)

# do more stuff
h2 = T.dot(h1, W2.T)

# compile and call the actual function
f = theano.function([x], h2)
f(numpy.random.rand(5, 10))


"""







"""
import theano.tensor as tt
x = tt.vector('x') 
y = tt.vector('y') 
s = tt.sum(x**2 + tt.sin(y))
print s
theano.pprint(s)
#theano.ProfileMode.provided_optimizer=fast_compile #None
x = T.vector()
y = T.vector()
z = x + x
z = z + y
f = theano.function([x, y], z)
#theano.pprint(f)
#theano.printing.debugprint(f) 可以输出他的参数类型
f(np.ones((2,)), np.ones((3,)))

"""

"""
import numpy
import theano
import theano.tensor as T

from theano import ProfileMode
profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
x = T.dvector('x')

f = theano.function([x], 10 * x, mode='DebugMode')
#optimizer=fast_compile
gf = theano.function([x], 10 * x, profile=True)#profmode)
profmode.print_summary()

f([5])
f([0])
f([7])

gf([4])

#"""