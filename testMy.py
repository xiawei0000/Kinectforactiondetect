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
import sys, os,random,numpy,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt

from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized


# Data folder (Training data)
print("Extracting the training files")
#data=os.path.join("M:\\ALP14\\small\\trantest\\")  
data=os.path.join("M:\\ALP14\\trainingall\\");
# Get the list of training samples
samples=os.listdir(data)
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','Spine','HipCenter']
njoints = len(used_joints)
STATE_NO = 10
count = 0

# pre-allocating the memory
Feature_all =  numpy.zeros(shape=(100000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
Targets = numpy.zeros( shape=(100000, STATE_NO*20+1), dtype=numpy.uint8)

# Access to each sample
#对于每一个zip文件，获取动作的cvs文件，提取动作的帧，得到对应的骨架点矩阵
#为何这里是对前后5帧去样本？？？，还给了个标志位？？
#20*10+1个空闲状态？？？   这表示是空闲状态吗？？
for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;        
    if file_count<650: 
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()
        # Iterate for each action in this sample
        # Then we also choose 5 frame before and after the ground true data:
        seg_length = 5
        for gesture in gesturesList:
                # Get the gesture ID, and start and end frames for the gesture
                gestureID,startFrame,endFrame=gesture
                # This part is to extract action data

                Skeleton_matrix = numpy.zeros(shape=(5, len(used_joints)*3))
                HipCentre_matrix = numpy.zeros(shape=(5, 3))
                frame_num = 0 
                
                ## extract first 5 frames
                if startFrame-seg_length > 0:
                    Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, startFrame-seg_length+1, startFrame)              
                    if not valid_skel:
                        print "No detected Skeleton: ", gestureID
                    else:
                        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
                        begin_frame = count
                        end_frame = count+seg_length-1
                        Feature_all[begin_frame:end_frame,:] = Feature#前5帧的特征矩阵
                        Targets[begin_frame:end_frame, -1] = 1#最后一位是1，
                        count=count+seg_length-1

                ## extract last 5 frames
                if endFrame+seg_length < smp.getNumFrames():
                    Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, endFrame, endFrame+seg_length-1)              
                    if not valid_skel:
                        print "No detected Skeleton: ", gestureID
                    else:
                        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
                        begin_frame = count
                        end_frame = count+seg_length-1
                        Feature_all[begin_frame:end_frame,:] = Feature
                        Targets[begin_frame:end_frame, -1] = 1
                        count=count+seg_length-1
        # ###############################################
        del smp

# save the skeleton file:
Feature_all_new = Feature_all[0:end_frame, :]
Targets_all_new = Targets[0:end_frame, :]
import cPickle as pickle
Feature_train = { "Feature_all_neutral": Feature_all_new, "Targets_all_new": Targets_all_new }
#将特征矩阵保存为pkl
pickle.dump( Feature_train, open( "Feature_all_neutral_realtime.pkl", "wb" ) )

import scipy.io as sio
sio.savemat('Feature_all_neutral_realtime.mat', Feature_train)
#保存为mat格式











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
import sys, os,random,numpy,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt

from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized


# Data folder (Training data)
print("Extracting the training files")
#data=os.path.join("M:\\ALP14\\small\\trantest\\")  
data=os.path.join("M:\\ALP14\\trainingall\\");
#data=os.path.join("D:\\ComputerScienceLanguage\\Kinect\\CODE+Paper\\LAP\\13\\trainning1\\")  
# Get the list of training samples
samples=os.listdir(data)
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','Spine','HipCenter']
njoints = len(used_joints)
STATE_NO = 10
count = 0

# pre-allocating the memory
Feature_all =  numpy.zeros(shape=(400000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
Targets = numpy.zeros( shape=(400000, STATE_NO*20+1), dtype=numpy.uint8)
# Access to each sample
for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;        
    if file_count<650: 
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()

        # Iterate for each action in this sample
        for gesture in gesturesList:
            # Get the gesture ID, and start and end frames for the gesture
            gestureID,startFrame,endFrame=gesture
            Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame)           
            # to see we actually detect a skeleton:
            if not valid_skel:
                print "No detected Skeleton: ", gestureID
            else:                            
                ### extract the features according to the CVPR2014 paper
                Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
                Target = numpy.zeros( shape=(Feature.shape[0], STATE_NO*20+1))
                #hmm中20个动作 每个都有10个隐含状态。
                fr_no =  Feature.shape[0]
                for i in range(STATE_NO):  #HMM states force alignment 这里是强制将动作帧分成10份，每个作为一个状态
                        begin_fr = numpy.round(fr_no* i /STATE_NO) + 1
                        end_fr = numpy.round( fr_no*(i+1) /STATE_NO) 
                        #print "begin: %d, end: %d"%(begin_fr-1, end_fr)
                        seg_length=end_fr-begin_fr + 1
                        targets = numpy.zeros( shape =(STATE_NO*20+1,1))#201 行，1列 的结果矩阵
                        targets[ i + STATE_NO*(gestureID-1)] = 1#对应这个动作*20+隐含状态序号==》标签的全局序号赋值为1
                        begin_frame = count
                        end_frame = count+seg_length
                        Feature_all[begin_frame:end_frame,:] = Feature[begin_fr-1:end_fr,:]#将特征拼接起来
                        Targets[begin_frame:end_frame, :]= numpy.tile(targets.T,(seg_length, 1))
                        #targets原本是 201行，1列 转置后就是1行201列， ==1的列就是状态的序号值
                        #tile是扩展函数，将标签 行扩展到帧长度， 列不变
                        count=count+seg_length
        # ###############################################
        ## delete the sample
        del smp

# save the skeleton file:

#最终，Feature_all 保存了特征矩阵， 每一行都是一个特征， 去除了没有动作的帧
#Targets 是对应特征的隐藏状态，【帧序号，201维】 列不为0 就是他的隐藏状态序号值
import cPickle as pickle
f = open('Feature_train_realtime.pkl','wb')
pickle.dump( {"Feature_all": Feature_all[0:end_frame, :], "Targets_all": Targets[0:end_frame, :] },f)
f.close()



import scipy.io as sio
sio.savemat('Feature_all_train__realtime.mat', { "Feature_all": Feature_all[0:end_frame, :], "Targets_all": Targets[0:end_frame, :] })




