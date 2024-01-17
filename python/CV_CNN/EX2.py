'''
Experiment 2:  DoA Estimation Results for Varying Geometry
Creator: Shulin Hu
Date: 08/15/2023
run steps:
1. Generate Experiment 2 test data by running matlab/GENER_Test_Data_Ex2.m (except for DNN).
2. Select EX2.py under the corresponding method (DNN\DCN\CNN\CV-CNN).
3. Run the EX2.py.
4. The program will automatically save the ture DoAs, base estimates, bias estimates, and generalized estimates
to result/EX2/Result_EX2_Geometry_Generalization_Alpha_0.5.h5.
5. Run the matlab/Plot_Experiment_2.m to plot the results.
'''
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from model import *
import os
import random
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
# Load the DL model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_CV_CNN = torch.load('../../model/CV_CNN/Model_CV_CNN.pth',map_location='cpu')

#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
d_base = 0.5
alpha = 0.5
d_bias = d_base/alpha
# Load the test data.
f_data_root ='../../data/EX2/'
f_data = f_data_root+'EX2_Alpha_0.5.h5'
#save path
f_result_root = '../../result/EX2/'
f_result = f_result_root+'Result_EX2_Geometry_Generalization_Alpha_0.5.h5'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)

#read test data
f2 = h5py.File(f_data, 'r')
GT_angles = np.array(f2['angle'])
DOA_num = len(GT_angles.reshape(-1))

RX_sam_test_gamma_base = np.array(f2['sam_gamma_base'])
RX_sam_test_gamma_base = torch.tensor(RX_sam_test_gamma_base)

#Reorganize the data structure to match the model input (complex values).
X_test_data_sam_gamma_base = (RX_sam_test_gamma_base[:,0,:,:].type(torch.complex64)+1j*RX_sam_test_gamma_base[:,1,:,:].type(torch.complex64)).unsqueeze(1).to(device)

RX_sam_test_gamma_bias = np.array(f2['sam_gamma_bias'])
RX_sam_test_gamma_bias = torch.tensor(RX_sam_test_gamma_bias)

#Reorganize the data structure to match the model input (complex values).
X_test_data_sam_gamma_bias = (RX_sam_test_gamma_bias[:,0,:,:].type(torch.complex64)+1j*RX_sam_test_gamma_bias[:,1,:,:].type(torch.complex64)).unsqueeze(1).to(device)

#Temporary variable to save results
DOA_base = []
DOA_bias = []
DOA_hat = []

# testing
model_CV_CNN.to(device)
for doa_index in range(DOA_num):     #Traverse the EFOV
    x_pred_sam = model_CV_CNN(X_test_data_sam_gamma_base[doa_index,:,:,:]).detach().cpu().numpy()
    doa_base = np.argmax(x_pred_sam) - 60  #The base estimation i.e. \theta_{base}
    DOA_base.append(doa_base)

    x_pred_sam = model_CV_CNN(X_test_data_sam_gamma_bias[doa_index,:,:,:]).detach().cpu().numpy()
    doa_bias = np.argmax(x_pred_sam) - 60  #The bias estimation i.e. \theta_{bias}
    DOA_bias.append(doa_bias)

    doa_hat = asind(alpha * sind(doa_bias))              #The generalized estimation i.e. \hat_{\theta}
    DOA_hat.append(doa_hat)


# plot the generalization estimate, bias estimate and ture DoA.
plt.figure()
plt.plot(GT_angles,DOA_base,'x',color='b',label='Base estimate')
plt.plot(GT_angles,DOA_bias,'*',color='g',label='Bias estimate')
plt.plot(GT_angles,DOA_hat,'+',color='r',label='Generalization estimate')
plt.xlabel("Direction (\u00b0)")
plt.ylabel("Estimated DoA (\u00b0)")
plt.legend()
plt.grid()
plt.show()

# Save the results.
with h5py.File(f_result, 'a') as hf:
        if 'DOA_ture' in hf:
                del hf['DOA_ture']
        if 'DOA_base_CV_CNN' in hf:
            del hf['DOA_base_CV_CNN']
        if 'DOA_bias_CV_CNN' in hf:
                del hf['DOA_bias_CV_CNN']
        if 'DOA_hat_CV_CNN' in hf:
            del hf['DOA_hat_CV_CNN']
        hf.create_dataset('DOA_ture', data=GT_angles)
        hf.create_dataset('DOA_base_CV_CNN', data=DOA_base)
        hf.create_dataset('DOA_bias_CV_CNN', data=DOA_bias)
        hf.create_dataset('DOA_hat_CV_CNN', data=DOA_hat)
        hf.close()

