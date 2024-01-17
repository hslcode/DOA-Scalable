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
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from model import *
import os
# Load the DL model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_CV_CNN = torch.load('../../model/CV_CNN/Model_CV_CNN.pth',map_location='cpu')
#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
#########################################################################
# !!! please replace the value For different geometries!!!
# You need to manually set it up
alpha = 0.1
#########################################################################
d_base = 0.5
d_bias = d_base/alpha
# Load the test data.
f_data_root ='../../data/EX1/'
f_data = f_data_root+'EX1_Alpha_' +str(alpha)+'.h5'
#save path
f_result_root = '../../result/EX1/'
f_result = f_result_root+'Result_EX1_Phase_Each_DoA_Alpha_' +str(alpha)+'.h5'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)
# Read the test data.
f2 = h5py.File(f_data, 'r')
GT_angles = np.array(f2['angle'])
DOA_num = len(GT_angles.reshape(-1))
RX_sam_test = np.array(f2['sam'])
test_num = RX_sam_test.shape[0]
RX_sam_test = torch.tensor(RX_sam_test)

#Reorganize the data structure to match the model input (complex values).
X_test_data_sam = (RX_sam_test[:,:,0,:,:].type(torch.complex64)+1j*RX_sam_test[:,:,1,:,:].type(torch.complex64)).unsqueeze(2).to(device)

#Temporary variable to save results
DOA_ture = []
DOA_bias = []
RMSE_phase = []
Ture_phase = np.zeros((DOA_num,test_num))       # The ture Spatial phase difference
Est_phase = np.zeros((DOA_num,test_num))        #The estimated spatial phase difference correspond to the estimated DOAs


# testing
model_CV_CNN.to(device)
for doa_index in range(DOA_num):         #Traverse the EFOV
        for test_index in range(test_num):      # Monte Carlo test
                x_pred_sam = model_CV_CNN(X_test_data_sam[test_index,doa_index,:,:,:]).detach().cpu().numpy()
                doa_bias = np.argmax(x_pred_sam) - 60                    #The bias estimation i.e. \theta_{bias}
                DOA_bias.append(doa_bias)
                phase_esti = d_base * np.sin(doa_bias / 180 * np.pi)    #The estimated spatial phase difference corresponds to the bias estimation
                Est_phase[doa_index,test_index] = phase_esti

                doa_ture = GT_angles[doa_index]                      #The ture spatial phase difference corresponds to the ture DoA
                DOA_ture.append(doa_ture)
                phase_ture = d_bias * np.sin(doa_ture / 180 * np.pi)    #The ture spatial phase difference corresponds to the ture DoA
                Ture_phase[doa_index,test_index] = phase_ture

        RMSE_phase.append(np.sqrt(mean_squared_error(Est_phase[doa_index,:], Ture_phase[doa_index,:]))) # RMSE between two spatial phase differences  of 1000 Monte Carlo tests



print("Phase RMSE at each DoA: ", RMSE_phase)
print(" Phase average RMSE within FoV: ", np.sum(RMSE_phase)/DOA_num)  

# plot the last test Est_phase and Ture_phase.
Est_phase_average = np.sum(Est_phase,axis=-1)/test_num   # average Est_phase at each DoA with a number of test_mum test
plt.figure()
plt.plot(GT_angles,Est_phase_average)
plt.plot(GT_angles,Ture_phase[:,0])
plt.grid()
plt.show()

# Save the results.
with h5py.File(f_result, 'a') as hf:  # 注意这里的模式是'a'，代表追加模式
        if 'Est_phase_CV_CNN' in hf:
                del hf['Est_phase_CV_CNN']
        if 'Ture_phase_CV_CNN' in hf:
                del hf['Ture_phase_CV_CNN']
        hf.create_dataset('Est_phase_CV_CNN', data=Est_phase_average)
        hf.create_dataset('Ture_phase_CV_CNN', data=Ture_phase[:,0])
        hf.close()
