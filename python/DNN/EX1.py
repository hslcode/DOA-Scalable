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
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from ensemble_model import *
from utils import *
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x

# model path
model_path_nn = '../../model/DNN/initial_model_AI.npy'
model_path_sf = '../../model/DNN/spatialfilter_model_AI.npy'
model_path_ss = '../../model/DNN/spatialspectrum_model_AI.npy'


## base array (traning model) parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 16        # array sensor number
wavelength = c / fc  # signal wavelength
gamma_base = 0.5
d_base = gamma_base * wavelength  # inter-sensor distance

#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
#########################################################################
# !!! please replace the value For different geometries!!!
# You need to manually set it up
alpha = 0.1
#########################################################################
gamma_bias = gamma_base/alpha
d_bias = gamma_bias * wavelength

# results save path
f_result_root = '../../result/EX1/'
f_result = f_result_root+'Result_EX1_Phase_Each_DoA_Alpha_' +str(alpha)+'.h5'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)

#  spatial filter training parameters
theta_min = -60      # minimal DOA (degree)
theta_max = 60       # maximal DOA (degree)
grid_sf = 1         # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((theta_max - theta_min) / grid_sf)
SF_NUM = 6       # number of spatial filters

# autoencoder parameters
input_size_sf = M * (M-1)
hidden_size_sf = int(1/2 * input_size_sf)
output_size_sf = input_size_sf
learning_rate_sf = 0.001

# DNN parameters
grid_ss = 1    # inter-grid angle in spatial spectrum
NUM_GRID_SS = int((theta_max - theta_min + 0.5 * grid_ss) / grid_ss)   # spectrum grids
input_size_ss = M * (M-1)
hidden_size_ss = [int(2/3* input_size_ss), int(4/9* input_size_ss), int(1/3* input_size_ss)]
output_size_ss = int(NUM_GRID_SS / SF_NUM)
learning_rate_ss = 0.001

# array imperfection parameters
# We do not consider array imperfections
MC_mtx = np.identity(M)
AP_mtx = np.identity(M)
pos_para = np.zeros([M, 1])

# Load the DL model
tf.reset_default_graph()
enmod_3 = Ensemble_Model(input_size_sf=input_size_sf,
                         hidden_size_sf=hidden_size_sf,
                         output_size_sf=output_size_sf,
                         SF_NUM=SF_NUM,
                         learning_rate_sf=learning_rate_sf,
                         input_size_ss=input_size_ss,
                         hidden_size_ss=hidden_size_ss,
                         output_size_ss=output_size_ss,
                         learning_rate_ss=learning_rate_ss,
                         reconstruct_nn_flag=False,
                         train_sf_flag=False,
                         train_ss_flag=False,
                         model_path_nn=model_path_nn,
                         model_path_sf=model_path_sf,
                         model_path_ss=model_path_ss)
# test data parameters
test_SNR = 20
T = 1000
test_num = 1000
theta_hat_max = np.floor(asind(alpha * sind(theta_max)))
test_DOA_set = np.arange(-theta_hat_max,theta_hat_max).astype(float)
GT_angles = test_DOA_set
DOA_num = len(test_DOA_set)

#Temporary variable to save results
DOA_ture = []
DOA_bias = []
RMSE_phase = []
Ture_phase = np.zeros((DOA_num,test_num))   # The ture Spatial phase difference
Est_phase = np.zeros((DOA_num,test_num))    #The estimated spatial phase difference correspond to the estimated DOAs

# testing
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('testing...')
    for doa_index in range(DOA_num):    #Traverse the EFOV
            for test_index in range(test_num):      # Monte Carlo test

                # Generate data for bias geometry
                test_cov_vector = generate_array_cov_vector_AI(M,T, d_bias, wavelength, test_DOA_set[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
                data_batch = np.expand_dims(test_cov_vector, axis=-1)
                feed_dict = {enmod_3.data_train: data_batch}
                ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
                ss_min = np.min(ss_output)
                ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
                doa_bias = get_DOA_estimate(ss_output_regularized, asind(1/alpha*sind(test_DOA_set[doa_index])), theta_min, grid_ss)          #The bias estimation i.e. \theta_{bias}
                phase_esti = gamma_base * np.sin(doa_bias / 180 * np.pi)            #The estimated spatial phase difference corresponds to the bias estimation
                Est_phase[doa_index,test_index] = phase_esti

                doa_ture = GT_angles[doa_index]                      #The ture DoA i.e. \theta_{bias}
                DOA_ture.append(doa_ture)
                phase_ture = gamma_bias * np.sin(doa_ture / 180 * np.pi)    #The ture spatial phase difference corresponds to the ture DoA
                Ture_phase[doa_index,test_index] = phase_ture

            RMSE_phase.append(np.sqrt(mean_squared_error(Est_phase[doa_index,:], Ture_phase[doa_index,:])))     # RMSE between two spatial phase differences  of 1000 Monte Carlo tests

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
with h5py.File(f_result, 'a') as hf:
        if 'Est_phase_DNN' in hf:
                del hf['Est_phase_DNN']
        if 'Ture_phase_DNN' in hf:
                del hf['Ture_phase_DNN']
        hf.create_dataset('Est_phase_DNN', data=Est_phase_average)
        hf.create_dataset('Ture_phase_DNN', data=Ture_phase[:,0])
        hf.close()


