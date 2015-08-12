#ecoding=gbk
from rbm import RBM

import theano
import theano.tensor as T

# --------------------------------------------------------------------------
class GBRBM(RBM):#这是python的类继承，直接一个（）

    # --------------------------------------------------------------------------
    # initialize class
    def __init__(self, input, n_in=784, n_hidden=500, \
                 W=None, hbias=None, vbias=None, numpy_rng=None, transpose=False, activation=T.nnet.sigmoid,
                 theano_rng=None, name='grbm', W_r=None, dropout=0, dropconnect=0):

            # initialize parent class (RBM)
            RBM.__init__(self, input=input, n_visible=n_in, n_hidden=n_hidden, \
                         W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng,
                         theano_rng=theano_rng)

    # --------------------------------------------------------------------------
    def type(self):
        return 'gauss-bernoulli'

    # --------------------------------------------------------------------------
    # overwrite free energy function (here only vbias term is different)
    def free_energy(self, v_sample):     #重写       
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        #这一个项 修改了， 原来是直接dot，现在换成了差的平方。
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - T.diagonal(vbias_term)

    # --------------------------------------------------------------------------
    # overwrite sampling function (here you sample from normal distribution)
    def sample_v_given_h(self, h0_sample):#为何只是重写了可视层到隐藏层，没有h到v层的函数？？？
        '''原理，输入层是实数，不是01010了，所以实数的输入产生01010的1000个节点。
        要从0101001 的串转回 实数的可视化节点，只能不用sigma函数，直接返回了。
        '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)


        '''
        theano不能调试的缺点太大了，  这个pre_sigmoid_v1 返回的是001001串，还是数字？？？？？？？

        实数样本输入， 经过权值+偏置后，得到pre_sigmoid_v1 。 再用激活函数sigma得到v1_mean。
        要得到类似的输出，可以直接将点积结果返回。

        如果是0100101的串输入，那么pre_sigmoid_v1 只是点积的值，通过sigma激活函数，得到概率，然后通过伯努利采样
        得到01010的输出。
        Since the input data is normalized to unit variance and zero mean, we do not have to sample
        from a normal distribution and pass the pre_sigmoid instead. If this is not the case, 
        we have to sample the      distribution.
        '''
        #这里说，考虑到数据是正态分布的，不必要再次用正态分布采样了，直接用概率返回结果，
        # in fact, you don't need to sample from normal distribution here and just use pre_sigmoid activation instead        
        # v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):#对原始重构误差的修改
        """
        RMS as the reconstructed cost

        """

        rms_cost = T.mean(T.sum((self.input -  pre_sigmoid_nv)** 2, axis=1))
        return rms_cost
