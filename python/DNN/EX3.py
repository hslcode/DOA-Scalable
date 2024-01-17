'''
Experiment 3: Statistical Performance Analysis for Fine-Grained Grid
Creator: Shulin Hu
Date: 08/15/2023
run steps:
1. Generate Experiment 3 test data by running matlab/GENER_Test_Data_Ex3.m (except for DNN).
2. Select EX3.py under the corresponding method (DNN\DCN\CNN\CV-CNN).
3. Run the EX3.py (this process will run for a relatively long time as it will automatically traverse each alpha).
4. For each alpha, the program will automatically save the average RMSE of base estimates,
the average RMSE of generalized estimates, and the ratio of the two RMSEs to result/EX3/Result_EX3_RMSE.h5.
Attention: The average RMSE refers to the average RMSE corresponding to all DOAs within the EFOV.
5. Run the matlab/Plot_Experiment_3.m to plot the results.
'''
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import scipy.linalg as la
from ensemble_model import *
from utils import *
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import random
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
## array signal parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 16        # array sensor number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# spatial filter training parameters
theta_min = -60      # minimal DOA (degree)
theta_max = 60       # maximal DOA (degree)
grid_sf = 1         # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((theta_max - theta_min) / grid_sf)
SF_NUM = 6       # number of spatial filters
SF_SCOPE = (theta_max - theta_min) / SF_NUM   # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 10    # number of repeated sampling with random noise
noise_flag_sf = 1    # 0: noise-free; 1: noise-present
amp_or_phase = 0   # show filter amplitude or phase: 0-amplitude; 1-phase

# # autoencoder parameters
input_size_sf = M * (M-1)
hidden_size_sf = int(1/2 * input_size_sf)
output_size_sf = input_size_sf
batch_size_sf = 32
num_epoch_sf = 1000
learning_rate_sf = 0.001

# DNN parameters
grid_ss = 1    # inter-grid angle in spatial spectrum
NUM_GRID_SS = int((theta_max - theta_min + 0.5 * grid_ss) / grid_ss)   # spectrum grids
L = 2    # number of hidden layer
input_size_ss = M * (M-1)
hidden_size_ss = [int(2/3* input_size_ss), int(4/9* input_size_ss), int(1/3* input_size_ss)]
output_size_ss = int(NUM_GRID_SS / SF_NUM)
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 300

# array imperfection parameters
# We do not consider array imperfections
MC_mtx = np.identity(M)
AP_mtx = np.identity(M)
pos_para = np.zeros([M, 1])

# model path
model_path_nn = '../../model/DNN/initial_model_AI.npy'
model_path_sf = '../../model/DNN/spatialfilter_model_AI.npy'
model_path_ss = '../../model/DNN/spatialspectrum_model_AI.npy'

# Load the test data.
f_data_root ='../../data/EX3/'

# results save path
f_result_root = '../../result/EX3/'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)
f_result = f_result_root + 'Result_EX3_RMSE.h5'
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


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('testing...')
    np.random.seed(123)
    gamma_base = 0.5
    test_num = 1000
    test_batch = 100
    test_SNR = 20
    T = 1000
    alpha_set = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    # variables to save results
    Ave_RSME_gamma_base = []
    Ave_RSME_gamma_bias = []
    Ave_RSME_ratio = []
    # testing
    for alpha in alpha_set: #Traverse the alpha set
        gamma_bias = gamma_base / alpha
        wavelength_bias = d/gamma_bias

        theta_hat_max = np.floor(asind(alpha * sind(theta_max)))         # The on-grid DoAs within EFoV;
        DOA_set = np.arange(-theta_hat_max, theta_hat_max-2).astype(float)  # The random off-grid value;
        DOA_shift = np.round(np.random.rand()*0.5,2)
        DOA_set = DOA_set+DOA_shift         # The off-grid DoAs within EFoV;
        DOA_num = len(DOA_set)

        # Temporary variable to save results
        RMSE_gamma_base = []
        RMSE_gamma_bias = []

        for doa_index  in range(DOA_num):   #Traverse the EFOV
            # Temporary variable
            est_DOA_gamma_base = []
            est_DOA_gamma_bias = []
            for test_index in range(test_num):  # Monte Carlo test

                # Generate data for base geometry with wavelength
                test_cov_vector = generate_array_cov_vector_AI(M, T, d, wavelength, DOA_set[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
                data_batch = np.expand_dims(test_cov_vector, axis=-1)
                feed_dict = {enmod_3.data_train: data_batch}
                ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
                ss_min = np.min(ss_output)
                ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
                doa_base = get_DOA_estimate(ss_output_regularized, DOA_set[doa_index], theta_min, grid_ss)  #The base estimation i.e. \theta_{base}
                est_DOA_gamma_base.append(doa_base)

                # Generate data for base geometry with wavelength_bias
                test_cov_vector = generate_array_cov_vector_AI(M, T, d, wavelength_bias, DOA_set[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
                data_batch = np.expand_dims(test_cov_vector, axis=-1)
                feed_dict = {enmod_3.data_train: data_batch}
                ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
                ss_min = np.min(ss_output)
                ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
                doa_bias = get_DOA_estimate(ss_output_regularized, asind(1/alpha*sind(DOA_set[doa_index])), theta_min, grid_ss) #The bias estimation i.e. \theta_{bias}
                doa_hat = asind(alpha*sind(doa_bias))   #The generalized estimation i.e. \hat_{\theta}
                est_DOA_gamma_bias.append(doa_hat)

            RMSE_gamma_base.append(np.sqrt(mean_squared_error(est_DOA_gamma_base, np.repeat(DOA_set[doa_index],test_num,axis=0))))  #The RMSE of base estimation
            RMSE_gamma_bias.append(np.sqrt(mean_squared_error(est_DOA_gamma_bias, np.repeat(DOA_set[doa_index],test_num,axis=0))))  #The RMSE of generalized estimation
            est_DOA_gamma_base.clear()
            est_DOA_gamma_bias.clear()

        Ave_RSME_gamma_bias_index = np.mean(RMSE_gamma_bias)    #The average RMSE of base estimation corresponding to the current alpha
        Ave_RSME_gamma_base_index = np.mean(RMSE_gamma_base)    #The average RMSE of generalized estimation corresponding to the current alpha
        Ave_RMSE_ratio_index = Ave_RSME_gamma_bias_index/Ave_RSME_gamma_base_index  #The ratio of average RMSE corresponding to the current alpha
        print("alpha={} ".format(alpha), Ave_RSME_gamma_base_index,Ave_RSME_gamma_bias_index,Ave_RMSE_ratio_index)

        Ave_RSME_gamma_base.append(Ave_RSME_gamma_base_index)   #Record the result corresponding to the current alpha
        Ave_RSME_gamma_bias.append(Ave_RSME_gamma_bias_index)   #Record the result corresponding to the current alpha
        Ave_RSME_ratio.append(Ave_RMSE_ratio_index)             #Record the result corresponding to the current alpha
        RMSE_gamma_base.clear()
        RMSE_gamma_bias.clear()

    # Save the results.
    with h5py.File(f_result, 'a') as hf:
            if 'Ave_RSME_gamma_base_DNN' in hf:
                    del hf['Ave_RSME_gamma_base_DNN']
            if 'Ave_RSME_gamma_bias_DNN' in hf:
                    del hf['Ave_RSME_gamma_bias_DNN']
            if 'Ave_RSME_ratio_DNN' in hf:
                del hf['Ave_RSME_ratio_DNN']
            hf.create_dataset('Ave_RSME_gamma_base_DNN', data=Ave_RSME_gamma_base)
            hf.create_dataset('Ave_RSME_gamma_bias_DNN', data=Ave_RSME_gamma_bias)
            hf.create_dataset('Ave_RSME_ratio_DNN', data=Ave_RSME_ratio)
            hf.close()


