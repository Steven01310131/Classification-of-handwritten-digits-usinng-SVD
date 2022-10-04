import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
TrainDigits=np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainDigits.npy')
data=pd.DataFrame(TrainDigits)
TrainLabels = pd.DataFrame((np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainLabels.npy')),index=None,columns=None)


vTrainDigits=np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainDigits.npy')
data=pd.DataFrame(TrainDigits)
TrainLabels = pd.DataFrame((np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainLabels.npy')),index=None,columns=None)

A_matrices={}
for i in range(10):                                                           #Creating a dictionary which seperates the training data into [A0, A1.. A9] matrices 
    A_matrices.update({"A"+str(i):data.loc[:,TrainLabels.loc[0,:]==i]})

A = np.reshape(A_matrices['A0'].to_numpy()[:,1], (28, 28)).T
print(np.shape(A))
plt.imshow(A, cmap ='gray')
plt.show()



