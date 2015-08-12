# coding=gbk

import sys, os, random, zipfile
from numpy import log
from shutil import copyfile
import matplotlib.pyplot as plt
import time
import cPickle
import numpy
import scipy.io as sio  

####################################
### local import
####################################
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized
from utils import normalize
from utils import imdisplay
from utils import createSubmisionFile
############### viterbi path import
from utils import viterbi_path, viterbi_path_log
from utils import viterbi_colab_clean
from utils import viterbi_endframe

####################################
### theano import
####################################
sys.path.append(r'.\TheanoDL')
try:
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    from logistic_sgd import LogisticRegression
    from mlp import HiddenLayer
    from rbm import RBM
    from grbm import GBRBM
    from utils import zero_mean_unit_variance
    from utils import normalize
    from GRBM_DBN import GRBM_DBN
    from sklearn import preprocessing
except ImportError:
    sys.exit('Please install Theano!')

class RunDbn(object):
    def __init__(self):
        ####################################
        ### Some initialization ############
        ####################################
        self.used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                        'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                        'Head','Spine','HipCenter']#11个点
        self.njoints = len(self.used_joints)
        ### load the pre-store normalization constant
        f = open('SK_normalization.pkl','rb')#预先处理好的
        SK_normalization = cPickle.load(f)#cPicke是 将对象打包保存为文件的类

        self.Mean1 =SK_normalization ['Mean1']#这里的均值和 方差是 后面真实数据归一化要使用的
        self.Std1 = SK_normalization['Std1']

        #这是vitebi算法要的数据
        ## Load Prior and transitional Matrix 预处理好的转换矩阵
        dic=sio.loadmat('Transition_matrix.mat')#scipy.io 是个科学计算模块，i模块实现了MATLAB数据的导入
        self.Transition_matrix = log(dic['Transition_matrix'])
        self.Prior = log(dic['Prior'])
        ##########################
        ### model 1  第一种网络构架模式  #
        ##########################        
        self.numpy_rng = numpy.random.RandomState(123)
        self.dbn = GRBM_DBN(numpy_rng=self.numpy_rng, n_ins=528,
        hidden_layers_sizes=[1000, 1000, 500],
        n_outs=201)
        self.dbn.load('dbn_2014-05-23-20-07-28.npy')#预先训练好的构架
        
        z=theano.tensor.dmatrix('z')
        #这里就是theano的奇葩函数构架
        self.validate_model = theano.function(inputs=[z],
            outputs=self.dbn.logLayer.p_y_given_x,#输出是逻辑回归层的输出
            givens={ self.dbn.x: z})


    def myBuildDBNtest(self):
        #提取所有帧中，原始的骨架点，得到一个矩阵，Skeleton_matrix ，同时返回骨架是否归0化
        #Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, 1, smp.getNumFrames())
        time_tic = time.time()  
        import cPickle
        Skeleton_matrix=cPickle.load(open("testSkeleton_matrix","rb"))
        #print Skeleton_matrix
 
        Feature = Extract_feature_Realtime(Skeleton_matrix, self.njoints)

        Feature_normalized = normalize(Feature, self.Mean1, self.Std1)
       
        '''
        ##########################
        ### model 1  第一种网络构架模式  #
        ##########################
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
        hidden_layers_sizes=[1000, 1000, 500],
        n_outs=201)
        dbn.load('dbn_2014-05-23-20-07-28.npy')#预先训练好的构架
        #这里就是theano的奇葩函数构架
        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,#输出是逻辑回归层的输出
            givens={ dbn.x: shared_x})   '''

        observ_likelihood_1 = self.validate_model(Feature_normalized)#调用函数得到结果



        ##########################
        # viterbi path decoding
        #####################
        observ_likelihood_1=observ_likelihood_1[0:50,:]
        #这里自己改了，只有第一个网络结构的，
        log_observ_likelihood = log(observ_likelihood_1.T) 
        #这里是一个矩阵， 行是样本，列是概率   [1884个样本, 201列] 用T转置了
        print "处理时间 %d sec" % int(time.time() - time_tic)
        time_tic = time.time()


        #下面就是vibiter算法了
        print("\t Viterbi path decoding " )
        # do it in log space avoid numeric underflow
        [path, predecessor_state_index, global_score] =viterbi_path_log(
            self.Prior,  self.Transition_matrix, log_observ_likelihood   )

        label=viterbi_endframe(path,5,30)
        # Some gestures are not within the vocabulary
        #[pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_clean(
        #    path, global_score, threshold=-100, mini_frame=19)
 
        print "标记是:"
        print label
        print "viterbi处理时间 %d sec" % int(time.time() - time_tic)

    def myBuildDBN(self,Skeleton_matrix):
        #提取所有帧中，原始的骨架点，得到一个矩阵，Skeleton_matrix ，同时返回骨架是否归0化
        #Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, 1, smp.getNumFrames())
        time_tic = time.time()  
 
        Feature = Extract_feature_Realtime(Skeleton_matrix, self.njoints)
        Feature_normalized = normalize(Feature, self.Mean1, self.Std1)
        observ_likelihood_1 = self.validate_model(Feature_normalized)#调用函数得到结果

        ##########################
        # viterbi path decoding
        #####################
        #observ_likelihood_1=observ_likelihood_1[0:50,:]
        #这里自己改了，只有第一个网络结构的，
        log_observ_likelihood = log(observ_likelihood_1.T) 
        #这里是一个矩阵， 行是样本，列是概率   [1884个样本, 201列] 用T转置了
        print "处理时间 %d sec" % int(time.time() - time_tic)
        
        return log_observ_likelihood
    def myViterbi(self,log_observ_likelihood):  
        time_tic = time.time()

        #下面就是vibiter算法了
        print("\t Viterbi path decoding " )
        # do it in log space avoid numeric underflow
        [path, predecessor_state_index, global_score] =viterbi_path_log(
            self.Prior,  self.Transition_matrix, log_observ_likelihood   )

        label=viterbi_endframe(path,5,30)
        # Some gestures are not within the vocabulary
        #[pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_clean(
        #    path, global_score, threshold=-100, mini_frame=19)
 
        print "标记是:"
        print label
        print "viterbi处理时间 %d sec" % int(time.time() - time_tic)
        return label

    
            

if __name__=='__main__':
    a=RunDbn(); 
    a.myBuildDBN()
             