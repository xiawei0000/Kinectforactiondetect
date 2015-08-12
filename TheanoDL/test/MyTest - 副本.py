#ecoding =gbk

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

import PIL.Image

def testbinomial():
    rng = numpy.random.RandomState(123)#numpy的随机产生器
    input=rng.rand(3,4)#产生一组，
    print input
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    re=theano_rng.binomial(size=input.shape, n=2,
                                         p=1 - 0.0001,#产生1的概率
                                         dtype=theano.config.floatX) 
    """Sample n times with probability of success p for each trial and
    # return the number of successes.
    就是以概率p随机测n次，返回成功的次数
    """
    print re
    print re[0][0]
    a=re[0]

    b=re[0][0]
   
    #print re(0)
    print re[1]
    re=re* input
    print re
    return  re
returnt=testbinomial()
print returnt

#print returnt.get_value()