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
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import random
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
# Load the DL model
model_CNN = load_model('../../model/2D_CNN/Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5')

#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
gamma_base = 0.5
alpha = 0.5
gamma_bias = gamma_base/alpha
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
test_num = RX_sam_test_gamma_base.shape[0]
X_test_data_sam_gamma_base = RX_sam_test_gamma_base.swapaxes(1,3)

RX_sam_test_gamma_bias = np.array(f2['sam_gamma_bias'])
X_test_data_sam_gamma_bias = RX_sam_test_gamma_bias.swapaxes(1,3)

#Temporary variable to save results
DOA_base = []
DOA_bias = []
DOA_hat = []


# testing
for doa_index in range(DOA_num):    #Traverse the EFOV
    x_pred_sam = model_CNN.predict(np.expand_dims(X_test_data_sam_gamma_base[doa_index, :, :, :], 0))
    doa_base = np.argmax(x_pred_sam) - 60  #The base estimation i.e. \theta_{base}
    DOA_base.append(doa_base)

    x_pred_sam = model_CNN.predict(np.expand_dims(X_test_data_sam_gamma_bias[doa_index, :, :, :], 0))
    doa_bias = np.argmax(x_pred_sam) - 60  #The bias estimation i.e. \theta_{bias}
    DOA_bias.append(doa_bias)

    doa_hat = asind(alpha * sind(doa_bias))             #The generalized estimation i.e. \hat_{\theta}
    DOA_hat.append(doa_hat)


# plot the ture DoA, generalization estimate, bias estimate and base estimate.
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
        if 'DOA_base_2DCNN' in hf:
            del hf['DOA_base_2DCNN']
        if 'DOA_bias_2DCNN' in hf:
                del hf['DOA_bias_2DCNN']
        if 'DOA_hat_2DCNN' in hf:
            del hf['DOA_hat_2DCNN']
        hf.create_dataset('DOA_ture', data=GT_angles)
        hf.create_dataset('DOA_base_2DCNN', data=DOA_base)
        hf.create_dataset('DOA_bias_2DCNN', data=DOA_bias)
        hf.create_dataset('DOA_hat_2DCNN', data=DOA_hat)
        hf.close()

