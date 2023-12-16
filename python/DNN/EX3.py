import numpy as np
import h5py
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
N = 400       # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # spatial filter training parameters
doa_min = -60      # minimal DOA (degree)
doa_max = 60       # maximal DOA (degree)
grid_sf = 1         # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6       # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
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

# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1         # DOA step (degree) for generating different scenarios
K_ss = 2            # signal number
doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE   # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 10    # number of repeated sampling with random noise

noise_flag_ss = 1    # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1    # inter-grid angle in spatial spectrum
NUM_GRID_SS = int((doa_max - doa_min + 0.5 * grid_ss) / grid_ss)   # spectrum grids
L = 2    # number of hidden layer
input_size_ss = M * (M-1)
hidden_size_ss = [int(2/3* input_size_ss), int(4/9* input_size_ss), int(1/3* input_size_ss)]
output_size_ss = int(NUM_GRID_SS / SF_NUM)
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 300

# # test data parameters
num_epoch_test = 1000
RMSE = []

MC_mtx = np.identity(M)
AP_mtx = np.identity(M)
pos_para = np.zeros([M, 1])

model_path_nn = '../../../model/DNN/EX3/initial_model_AI.npy'
model_path_sf = '../../../model/DNN/EX3/spatialfilter_model_AI.npy'
model_path_ss = '../../../model/DNN/EX3/spatialspectrum_model_AI.npy'

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

    # # test
    d1 = 0.5
    alpha = 0.9
    d2 = d / alpha

    L__ = np.floor(asind(alpha * sind(doa_max)))
    test_DOA = np.arange(-L__,L__-2).astype(float)
    # np.random.seed(123)
    off_grid = np.random.rand(len(test_DOA.reshape(-1)))
    test_DOA = test_DOA+off_grid
    doa_num = len(test_DOA)
    test_SNR = np.array([10,10])
    test_num = 1000
    est_DOA_d1 = []
    est_DOA_d2 = []
    est_phase_d1 = []
    est_phase_d2 = []
    RMSE_d1 = 0
    RMSE_d2 = 0
    for test_index in range(test_num):
        print(test_index)

        for doa_index  in range(doa_num):
            test_cov_vector = generate_array_cov_vector_AI(M, N, d2, wavelength, test_DOA[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
            data_batch = np.expand_dims(test_cov_vector, axis=-1)
            feed_dict = {enmod_3.data_train: data_batch}
            ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
            ss_min = np.min(ss_output)
            ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
            est_DOA_ii = get_DOA_estimate(ss_output_regularized, test_DOA[doa_index], doa_min, grid_ss)
            est_DOA_d2.append(asind(alpha*sind(est_DOA_ii)))
            phase_d2 = d1 * np.sin(est_DOA_ii / 180 * np.pi)
            est_phase_d2.append(phase_d2)


            test_cov_vector = generate_array_cov_vector_AI(M, N, d, wavelength, test_DOA[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
            data_batch = np.expand_dims(test_cov_vector, axis=-1)
            feed_dict = {enmod_3.data_train: data_batch}
            ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
            ss_min = np.min(ss_output)
            ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
            est_DOA_ii = get_DOA_estimate(ss_output_regularized, test_DOA[doa_index], doa_min, grid_ss)
            est_DOA_d1.append(est_DOA_ii)
            phase_d1 = d1/alpha * np.sin(est_DOA_ii / 180 * np.pi)
            est_phase_d1.append(phase_d1)
        RMSE_d1 = RMSE_d1+np.sqrt(mean_squared_error(est_DOA_d1, test_DOA))
        RMSE_d2 = RMSE_d2+np.sqrt(mean_squared_error(est_DOA_d2, test_DOA))
        est_DOA_d1.clear()
        est_DOA_d2.clear()


print(RMSE_d1/test_num)
print(RMSE_d2/test_num)
plt.figure()
plt.plot(test_DOA,est_DOA_d1)
plt.plot(test_DOA,est_DOA_d2)
plt.show()
plt.grid()

plt.figure()
plt.plot(test_DOA,est_phase_d1)
plt.plot(test_DOA,est_phase_d2)
plt.show()
plt.grid()

# f_result_root = '../../../result/EX1/'
# f_result = f_result_root+'DNN_phase_alpha_0.5.h5'
# hf = h5py.File(f_result, 'w')
# hf.create_dataset('est_phase_d1', data=est_phase_d1)
# hf.create_dataset('est_phase_d2', data=est_phase_d2)
# hf.close()
#
# f_result_root = '../../../result/EX2/'
# f_result = f_result_root+'EX2_result_DNN_d10.5_d21.0.h5'
# hf = h5py.File(f_result, 'w')
# hf.create_dataset('est_DOA_d1', data=est_DOA_d1)
# hf.create_dataset('est_DOA_d2', data=est_DOA_d2)
# hf.close()

