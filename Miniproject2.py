#libraries that are gonna be used 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import norm
#loading of the training data 

TrainDigits=pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainDigits.npy'))
TrainLabels=pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainLabels.npy'))
TestDigits =pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TestDigits.npy'))
TestLabels =pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TestLabels.npy'))
#Training
Atrain_matrices={}
for i in range(10):                                                           #Creating a dictionary which seperates the training data into [A0, A1.. A9] matrices 
    Atrain_matrices.update({"A"+str(i):TrainDigits.loc[:,TrainLabels.loc[0,:]==i]})
#Test
Atest_matrices={}
for i in range(10):                                                           
    Atest_matrices.update({"Atest"+str(i):TestDigits.loc[:,TestLabels.loc[0,:]==i]})
A0 =Atrain_matrices['A0'].to_numpy()[:,0:500]
A1 =Atrain_matrices['A1'].to_numpy()[:,0:500]
A2 =Atrain_matrices['A2'].to_numpy()[:,0:500]
A3 =Atrain_matrices['A3'].to_numpy()[:,0:500]
A4 =Atrain_matrices['A4'].to_numpy()[:,0:500]
A5 =Atrain_matrices['A5'].to_numpy()[:,0:500]
A6 =Atrain_matrices['A6'].to_numpy()[:,0:500]
A7 =Atrain_matrices['A7'].to_numpy()[:,0:500]
A8 =Atrain_matrices['A8'].to_numpy()[:,0:500]
A9 =Atrain_matrices['A9'].to_numpy()[:,0:500]
A0test =Atest_matrices['Atest0'].to_numpy()[:,0:100]                                     
A1test =Atest_matrices['Atest1'].to_numpy()[:,0:100]                                        
A2test =Atest_matrices['Atest2'].to_numpy()[:,0:100]                                        
A3test =Atest_matrices['Atest3'].to_numpy()[:,0:100]                                        
A4test =Atest_matrices['Atest4'].to_numpy()[:,0:100]                                        
A5test =Atest_matrices['Atest5'].to_numpy()[:,0:100]                                        
A6test =Atest_matrices['Atest6'].to_numpy()[:,0:100]                                        
A7test =Atest_matrices['Atest7'].to_numpy()[:,0:100]                                        
A8test =Atest_matrices['Atest8'].to_numpy()[:,0:100]                                        
A9test =Atest_matrices['Atest9'].to_numpy()[:,0:100]                                       

(u0,s0,v0)=np.linalg.svd(A0)
(u1,s1,v1)=np.linalg.svd(A1)
(u2,s2,v2)=np.linalg.svd(A2)
(u3,s3,v3)=np.linalg.svd(A3)
(u4,s4,v4)=np.linalg.svd(A4)
(u5,s5,v5)=np.linalg.svd(A5)
(u6,s6,v7)=np.linalg.svd(A6)
(u7,s7,v8)=np.linalg.svd(A7)
(u8,s8,v8)=np.linalg.svd(A8)
(u9,s9,v9)=np.linalg.svd(A9)
print(np.shape(u0))
# print(len(A0test[0,:]))

# L=[u0,u1,u2,u3,u4,u5,u6,u7,u8,u9]
# def classifyUnknownDigit(newDigit):
#     L=[u0,u1,u2,u3,u4,u5,u6,u7,u8,u9]
#     values = []
#     for U in L:
#         values.append(np.linalg.norm((np.identity(len(U))-np.matrix(U)*np.matrix(U.T)).dot(newDigit),ord=2)/np.linalg.norm(newDigit,ord=2))
#     return values.index(min(values))
# zero_pred = []
I = np.eye(TrainDigits.shape[0])
print(np.shape(I))
for i in range(len(A0test[0,:])):
    residuals = []
    for j in range(10):
        res=norm( np.dot(I-np.dot(u0,u0.T), A0test[:,:10]  ))
        residuals.append(res)
print(residuals)
#For classifying a specific number
# ResidualU=np.linalg.norm((np.identity(len(u0))-np.matrix(u0)*np.matrix(u0.T)).dot(),ord=2)/np.linalg.norm(A0test,ord=2)
# zeroprediction=[]
# for i in range(len(A0test)):
#     zeroprediction.append()
# classes = [u0,u1,u2,u3,u4]
# values = []
# for U in classes:
#     values.append(np.linalg.norm((np.identity(len(U))-np.matrix(U)*np.matrix(U.T)).dot(newDigit),ord=2)/np.linalg.norm(newDigit,ord=2))
# k=values.index(min(values))
# zero_pred = []
# for i in range(len(TestDigits)):
#     zero_pred.append(classifyUnknownDigit(zero_test.iloc[i]))
# for i in L:
#     ResidualU.append(np.linalg.norm((np.identity(len(i))-np.matrix(i)*np.matrix(i.T)).dot(),ord=2)/np.linalg.norm(A0test,ord=2))
# ResidualU.index(min(ResidualU))


#Part of code to display as images the first 15 left singular vectors [u1,.....u15] in this case the value zero 
# A=Atrain_matrices['A0'].to_numpy()[:,0:2000]
# (u,s,v)=np.linalg.svd(A)
# plt.figure(figsize=(20,10))
# columns = 5
# for i in range(15):
#    plt.subplot(columns + 1, columns, i + 1)
#    plt.imshow(u[:,i].reshape(28,28),cmap='binary')
# plt.show()

# Part of code to visualize one of the training images

# A = np.reshape(A_matrices['A0'].to_numpy()[:,1], (28, 28)).T
# print(np.shape(A))
# plt.imshow(A, cmap ='gray')
# plt.show()



