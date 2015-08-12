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
        gesturesList=smp.getGestures()#获取手势的链表
        # Iterate for each action in this sample
        # Then we also choose 5 frame before and after the ground true data:
        seg_length = 5
        for gesture in gesturesList:
                # Get the gesture ID, and start and end frames for the gesture
                gestureID,startFrame,endFrame=gesture#得到手势的id ，开始结束位置帧
                # This part is to extract action data

                Skeleton_matrix = numpy.zeros(shape=(5, len(used_joints)*3))
                HipCentre_matrix = numpy.zeros(shape=(5, 3))
                frame_num = 0 
                
                ## extract first 5 frames
                if startFrame-seg_length > 0:
                    Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, startFrame-seg_length+1, startFrame)  
                    #提取开始帧的前5帧  到开始帧的骨架坐标，保存到矩阵中，返回是否归0 中心化   
                    #一个动作的前5帧，基本上是没有动作的。 这就当做自然的样本数据用。
                             
                    if not valid_skel:
                        print "No detected Skeleton: ", gestureID
                    else:
                        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
                        #提取特征，这里只有当前帧的坐标差， 和前后帧坐标差， 维度扩大

                        #下面是将这5帧特征保存到 all 的链表里
                        begin_frame = count
                        end_frame = count+seg_length-1
                        Feature_all[begin_frame:end_frame,:] = Feature#前5帧的特征矩阵
                        Targets[begin_frame:end_frame, -1] = 1#最后一位是1，表示没有动作
                        count=count+seg_length-1

                ## extract last 5 frames
                #提取结束动作的后5帧
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
#从原空间中分离出 得到的数据 。因为最开始申请了一个超大空间
Feature_all_new = Feature_all[0:end_frame, :]
Targets_all_new = Targets[0:end_frame, :]

import cPickle as pickle
#这里是当做词典保存了。
Feature_train = { "Feature_all_neutral": Feature_all_new, "Targets_all_new": Targets_all_new }
#将特征矩阵保存为pkl
pickle.dump( Feature_train, open( "Feature_all_neutral_realtime.pkl", "wb" ) )

import scipy.io as sio
sio.savemat('Feature_all_neutral_realtime.mat', Feature_train)
#保存为mat格式









