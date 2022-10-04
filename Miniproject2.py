
#libraries that are gonna be used 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import norm
#loading of the training data 

TrainDigits=pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainDigits.npy'))
TrainLabels=pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainLabels.npy'))
TestDigits =pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TestLabels.npy'))
TestLabels =pd.DataFrame(np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TestLabels.npy'))
A_matrices={}
for i in range(10):                                                           #Creating a dictionary which seperates the training data into [A0, A1.. A9] matrices 
    A_matrices.update({"A"+str(i):TrainDigits.loc[:,TrainLabels.loc[0,:]==i]})
A0 =A_matrices['A0'].to_numpy()[:,0:500]
A1 =A_matrices['A1'].to_numpy()[:,0:500]
A2 =A_matrices['A2'].to_numpy()[:,0:500]
A3 =A_matrices['A3'].to_numpy()[:,0:500]
A4 =A_matrices['A4'].to_numpy()[:,0:500]
A5 =A_matrices['A5'].to_numpy()[:,0:500]
A6 =A_matrices['A6'].to_numpy()[:,0:500]
A7 =A_matrices['A7'].to_numpy()[:,0:500]
A8 =A_matrices['A8'].to_numpy()[:,0:500]
A9 =A_matrices['A9'].to_numpy()[:,0:500]
# (u0,s0,v0)=np.linalg.svd(A)


#Part of code to display as images the first 15 left singular vectors [u1,.....u15] in this case the value zero 
# A=A_matrices['A0'].to_numpy()[:,0:200]
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
