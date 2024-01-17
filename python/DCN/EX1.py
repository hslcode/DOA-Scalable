'''
Experiment 1: The Validation of the Mapping in DL-based DoA Estimator
Creator: Shulin Hu
Date: 08/15/2023
run steps:
1. Generate Experiment 1 test data by running GENER_Test_Data_Ex1.m (except for DNN).
2. Select EX1.py under the corresponding method (DNN\DCN\CNN\CV-CNN).
3. Set the alpha value in EX1.py.
4. Run the EX1.py.
5. Record the printed Phase average RMSE to the "result/EX1/Phase average RMSE.xlsx" file,
and The program will automatically save the average estimated phase difference and true phase
difference of 1000 Monte Carlo tests corresponding to each DoA to result/EX1/Result_EX1_Alpha_{alpha}.h5.
6. Replace the value of alpha and return to step 2,  until traversing the entire set([0.1,1]).
7. Run the matlab/Plot_Experiment_1.m to plot the results.
'''
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import h5py
import os
from sklearn.metrics import mean_squared_error
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
# Load the DL model
model_1DCNN = keras.models.load_model('../../model/1D_CNN/1DCNN_16.h5')
#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
#########################################################################
# !!! please replace the value For different geometries!!!
# You need to manually set it up
alpha = 0.1
#########################################################################
gamma_base = 0.5
gamma_bias = gamma_base/alpha
# Load the test data.
f_data_root ='../../data/EX1/'
f_data = f_data_root+'EX1_Alpha_' +str(alpha)+'.mat'
#save path
f_result_root = '../../result/EX1/'
f_result = f_result_root+'Result_EX1_Phase_Each_DoA_Alpha_' +str(alpha)+'.h5'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)

#read test data
read_temp= scipy.io.loadmat(f_data)
S_est = read_temp['S_est']
GT_angles = read_temp['DOA_ture'].reshape(-1)
[test_num,DOA_num,grids,_] = np.shape(S_est)

#Temporary variable to save results
DOA_ture = []
DOA_bias = []
RMSE_phase = []
Ture_phase = np.zeros((DOA_num,test_num))       # The ture Spatial phase difference
Est_phase = np.zeros((DOA_num,test_num))        # The estimated spatial phase difference correspond to the estimated DOAs

# testing
for doa_index in range(DOA_num):        #Traverse the EFOV
        for test_index in range(test_num):      # Monte Carlo test
            DF_T_cnn = model_1DCNN.predict(S_est[test_index, doa_index, :, :].reshape(1, grids, 2))
            DF_T_cnn = np.array(DF_T_cnn)
            DF_T_cnn = np.reshape(DF_T_cnn, -1)
            doa_bias = np.argmax(DF_T_cnn) - 60                 #The bias estimation i.e. \theta_{bias}
            phase_esti = gamma_base * np.sin(doa_bias / 180 * np.pi)     #The estimated spatial phase difference corresponds to the bias estimation
            Est_phase[doa_index,test_index] = phase_esti

            doa_ture = GT_angles[doa_index]                      #The ture DoA i.e. \theta_{bias}
            DOA_ture.append(doa_ture)
            phase_ture = gamma_bias * np.sin(doa_ture / 180 * np.pi)     #The ture spatial phase difference corresponds to the ture DoA
            Ture_phase[doa_index,test_index] = phase_ture

        RMSE_phase.append(np.sqrt(mean_squared_error(Est_phase[doa_index,:], Ture_phase[doa_index,:])))      # RMSE between two spatial phase differences  of 1000 Monte Carlo tests

print("Phase RMSE at each DoA: ", RMSE_phase)
print(" Phase average RMSE within FoV: ", np.sum(RMSE_phase)/DOA_num)

# plot the last test Est_phase and Ture_phase.
Est_phase_average = np.sum(Est_phase,axis=-1)/test_num   # average Est_phase at each DoA with a number of test_mum test26


plt.figure()
plt.plot(GT_angles,Est_phase_average)
plt.plot(GT_angles,Ture_phase[:,0])
plt.grid()
plt.show()

# Save the results.
with h5py.File(f_result, 'a') as hf:  
        if 'Est_phase_1D_CNN' in hf:
                del hf['Est_phase_1D_CNN']
        if 'Ture_phase_1D_CNN' in hf:
                del hf['Ture_phase_1D_CNN']
        hf.create_dataset('Est_phase_1D_CNN', data=Est_phase_average)
        hf.create_dataset('Ture_phase_1D_CNN', data=Ture_phase[:,0])
        hf.close()

