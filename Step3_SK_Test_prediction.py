# coding=gbk
#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track3
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
# Author:      Di Wu: stevenwudi@gmail.com
# Created:     24/03/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
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


print("Congratulations, you finish all the reqruired modules!")
####################################
### The path you need to set!!!#####
####################################

# Data folder (Test data)
#data_path=os.path.join("I:\Kaggle_multimodal\Test\Test\\")  
#data_path=os.path.join("D:\\ComputerScienceLanguage\\Kinect\\CODE+Paper\\LAP\\13\\trainning1");
#data_path=os.path.join("M:\\ALP14\\small\\tran\\");


##########################################################################################################测试修改
data_path=os.path.join("D:\\360\\down\\");

# Predictions folder (output)
outPred=r'.\training\test'
# Submision folder (output)
outSubmision=r'.\training\test_submission'

####################################
### Some initialization ############
####################################
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','Spine','HipCenter']#11个点
njoints = len(used_joints)
### load the pre-store normalization constant
f = open('SK_normalization.pkl','rb')#预先处理好的
SK_normalization = cPickle.load(f)#cPicke是 将对象打包保存为文件的类

Mean1 = SK_normalization ['Mean1']#这里的均值和 方差是 后面真实数据归一化要使用的
Std1 = SK_normalization['Std1']

#这是vitebi算法要的数据
## Load Prior and transitional Matrix 预处理好的转换矩阵
dic=sio.loadmat('Transition_matrix.mat')#scipy.io 是个科学计算模块，i模块实现了MATLAB数据的导入
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']



####################################
### Start predicting here ##########
####################################
samples=os.listdir(data_path)#listdir展示所有路径
for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;  
    time_tic = time.time()      
    if  file_count > -1: # do nothing, just for bebug convenience
        ''' 这里 是对一个样本文件的处理，
        先获取动作list，然后对一个文件中的所有动作，一起提取特征。
        通过网络计算 概率可能度， （一个视频开始的前5帧，最后一帧无效）'''
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data_path,file))#读取数据，按特定格式， os.path.join是链接路径用的
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()#得到帧对应的动作标签

        #提取所有帧中，原始的骨架点，得到一个矩阵，Skeleton_matrix ，同时返回骨架是否归0化
        Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, 1, smp.getNumFrames())

        import cPickle
        cPickle.dump(Skeleton_matrix,open("testSkeleton_matrix","wb"))
        #print Skeleton_matrix
        #print Skeleton_matrix.get_value()


        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)

        Feature_normalized = normalize(Feature, Mean1, Std1)

        ### Feed into DBN, theano requires the shared tensor representation
        shared_x = theano.shared(numpy.asarray(Feature_normalized,
                                    dtype=theano.config.floatX),
                                    borrow=True)
        numpy_rng = numpy.random.RandomState(123)

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
            givens={ dbn.x: shared_x})

        observ_likelihood_1 = validate_model()#调用函数得到结果
        del dbn


        """
        ##########################
        ### model 2
        ##########################
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
        hidden_layers_sizes=[1000, 1000, 500],
        n_outs=201)

        dbn.load('dbn_2014-05-24-05-53-17.npy')

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_2 = validate_model()
        del dbn

        ##########################
        ### model 3
        ##########################
        
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
            hidden_layers_sizes=[2000, 2000, 1000],
            n_outs=201, finetune_lr=0.1)
        dbn.load('dbn_2014-05-25-10-11-56.npy')
        # Optimization complete with best validation score of 38.194915 %,with test performance 38.113636 %
        #....The score for this prediction is 0.792685963841

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_3 = validate_model()
        del dbn

        ##########################
        ### model 4
        ##########################
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
            hidden_layers_sizes=[2000, 2000, 1000],
            n_outs=201, finetune_lr=0.1)
        dbn.load('dbn_2014-05-25-11-09-00.npy')

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_4 = validate_model()
        del dbn
        """

        #sio.savemat('observ_likelihood.mat', {'observ_likelihood':observ_likelihood})
        ##########################
        # viterbi path decoding
        #####################

 #       log_observ_likelihood = log(observ_likelihood_1.T) + log(observ_likelihood_2.T) \
 #                                + log(observ_likelihood_3.T) + log(observ_likelihood_4.T)
        ##########################
        # viterbi path decoding
        #####################

        #这里自己改了，只有第一个网络结构的，
        log_observ_likelihood = log(observ_likelihood_1.T) 
        #这里是一个矩阵， 行是样本，列是概率   [1884个样本, 201列] 用T转置了


        '''这时，矩阵是 【201行个id， 1884个样本列】
        【-1，0：5】其实就是一个视频开始的前5帧 第201个动作id 的 log（概率） =0
        log=0，那么概率=1， log（小于1）=负数。
        '''
        #（一个视频开始的前5帧，最后5帧无效）
        log_observ_likelihood[-1, 0:5] = 0 # We force first and last 5 frames are in the ergodic state
        #？？？？？？？？？为何是最后最后一行， 不是全部的？？？
        log_observ_likelihood[-1, -5:] = 0#这 最后5帧也无效


        #下面就是vibiter算法了
        print("\t Viterbi path decoding " )
        # do it in log space avoid numeric underflow
        [path, predecessor_state_index, global_score] =viterbi_path_log(
            log(Prior), log(Transition_matrix), log_observ_likelihood   )
        #[path, predecessor_state_index, global_score] =  viterbi_path(Prior, Transition_matrix, observ_likelihood)
        
        # Some gestures are not within the vocabulary
        [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_clean(
            path, global_score, threshold=-100, mini_frame=19)

        #begin_frame = begin_frame-1
        end_frame = end_frame + 3
        ### plot the path and prediction
        ##绘制结果， 默认false不执行
        if False:
            im  = imdisplay(global_score)
            plt.imshow(im, cmap='gray')
            plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
            plt.xlim((0, global_score.shape[-1]))
            # plot ground truth
            for gesture in gesturesList:
            # Get the gesture ID, and start and end frames for the gesture
                gestureID,startFrame,endFrame=gesture
                frames_count = numpy.array(range(startFrame, endFrame+1))
                pred_label_temp = ((gestureID-1) *10 +5) * numpy.ones(len(frames_count))
                plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
            
            # plot clean path
            for i in range(len(begin_frame)):
                frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
                pred_label_temp = ((pred_label[i]-1) *10 +5) * numpy.ones(len(frames_count))
                plt.plot(frames_count, pred_label_temp, color='#ffff00', linewidth=2.0)

            plt.show()
        else:
            print "Elapsed time %d sec" % int(time.time() - time_tic)

            pred=[]
            for i in range(len(begin_frame)):
                pred.append([ pred_label[i], begin_frame[i], end_frame[i]] )
                #pred_label 是动作的标签，  begin_frame  end_frame是开始和结束

            smp.exportPredictions(pred,outPred)

     # ###############################################
        ## delete the sample
        del smp        


# Prepare submision file (only for validation and final evaluation data sets)
createSubmisionFile(outPred, outSubmision)
#用zip文件保存结果