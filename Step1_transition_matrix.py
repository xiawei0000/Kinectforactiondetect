#ecoding=gbk
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
import cv2
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_normalized
from utils import Extract_feature
import time
import cPickle
""" Main script. Show how to perform all competition steps
    Access the sample information to learn a model. """
# Data folder (Training data)
print("Extracting the training files")
#data=os.path.join("I:\Kaggle_multimodal\Training\\")  
#data=os.path.join("M:\\ALP14\\trainingall\\");


##########################################################################################################测试修改
data=os.path.join("D:\\360\\down\\");


# Get the list of training samples
samples=os.listdir(data)
used_joints = ['ElbowLeft', 'WristLeft', 'ElbowRight', 'WristRight']
njoints = len(used_joints)
STATE_NO = 10
batch_num = 13

# pre-allocating the memory
Prior = numpy.zeros(shape=(201))
Transition_matrix = numpy.zeros(shape=(201,201))

for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;  
    time_tic = time.time()      
    if (file_count<651):
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()
        ''' 对于每个动作文件，获取开始结束帧
       
        '''

        for gesture in gesturesList:
            gestureID,startFrame,endFrame=gesture

            for frame in range(endFrame-startFrame+1-4):
                #对于一个动作的每一帧，
                #获取当前帧的动作小id（201个）
                state_no_1 = numpy.floor(frame*(STATE_NO*1.0/(endFrame-startFrame+1-3)))
                state_no_1 = state_no_1+STATE_NO*(gestureID-1)
                #获取下一阵的 小id
                state_no_2 = numpy.floor((frame+1)*(STATE_NO*1.0/(endFrame-startFrame+1-3)))
                state_no_2 = state_no_2+STATE_NO*(gestureID-1)

                ## we allow first two states add together:
                Prior [state_no_1] += 1             #状态 的次数+1
                Transition_matrix[state_no_1, state_no_2] += 1  #转移概率，由一个动作id 转另一个 次数+1
               
                
                if frame<2:#动作的前两帧
                    Transition_matrix[-1, state_no_1] += 1 ##转移概率最后一行， 动作小id一列+1 。
                    #表示 从无动作，转移到了这个动作第一个起始id
                    #最后一行 的 每一列 表示 那个大的动作的次数 +2
                    Prior[-1] += 1  #大的动作 样本个数 总数+2

                if frame> (endFrame-startFrame+1-4-2):#结束时的两帧
                    Transition_matrix[state_no_2, -1] += 1  #结束状态行 转移到最后一列+1 。表示结束的小id 转到无动作了。
                    Prior[-1] += 1          #大的动作 样本个数 总数+2
        del smp        

        '''最终，Transition_matrix 得到的是所有样本中， 
        每个小动作id-》下一个小动作id。 无动作-》到某个动作开始。 一个动作结束=》到无动作。 的次数

        Prior 是每个小动作id的次数，最后一个是所有无动作的次数。
        注意， 每个开始都加了两次，表明 开始的第一帧，第二帧都可以进入这个动作。
        结束也加2次，表明可以在最后第二帧结束退出，也可以标准的在最后一帧退出
        '''

'''保存前还差了 归一化数据， prior要除以总帧数，
matrix，  M[i,j]= M[i.j]/ 累加(M[i,x])
'''

Prior=Prior*1.0 /(Prior.sum())
for i in range(0,201): 
    tempsum=Transition_matrix[i,:].sum()
    for j in range(0,201):       
        Transition_matrix[i][j]=Transition_matrix[i][j]*1.0 / tempsum;



import scipy.io as sio
#sio.savemat('Transition_matrix.mat', {'Transition_matrix':Transition_matrix})
sio.savemat('Prior.mat', {Prior:'Prior'})
sio.savemat('Prior_Transition_matrix.mat', {'Transition_matrix':Transition_matrix, 'Prior': Prior})

img = Transition_matrix*1.0*255/Transition_matrix.max()
fig, ax = plt.subplots()
cax = ax.imshow(temp2, interpolation='nearest', cmap=cm.coolwarm)
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

''' 实际上，取看转移矩阵，99的数据都是自己转自己 。或者转下一个id。
 自己转自己概率最大。'''    