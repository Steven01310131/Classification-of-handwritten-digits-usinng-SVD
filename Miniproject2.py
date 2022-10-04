import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
TrainDigits=np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainDigits.npy')
data=pd.DataFrame(TrainDigits)
TrainLabels = pd.DataFrame((np.load('X:\Documents\Python Scripts\ScientificComp\HandwrittenDigits\TrainLabels.npy')),index=None,columns=None)


# print(df)
# print(np.shape(TrainLabels))
# (num_rows,num_cols)=np.shape(TrainDigits)
# d = TrainDigits[:,14]    # The first digit in the training set
# A = np.reshape(d, (28, 28)).T
# print(np.shape(A))
# plt.imshow(A, cmap ='gray')
# plt.show()
alpha_matrices={}
A0=data.loc[:,TrainLabels.loc[0,:]==8]
A0=A0.to_numpy()
print(A0)
print(np.shape(A0))
for i in range(10):
    print(alpha_matrices)
    alpha_matrices={"A"+str(i):data.loc[:,TrainLabels.loc[0,:]==i].to_numpy}

# # A = np.reshape(alpha_matrices['A[0]'][:,0], (28, 28)).T
# # print(np.shape(A))
# # plt.imshow(A, cmap ='gray')
# # plt.show()


