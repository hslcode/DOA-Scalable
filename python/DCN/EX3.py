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
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from sklearn import preprocessing
# from sklearn.externals import joblib
import h5py
from sklearn.metrics import mean_squared_error
import os
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x

# Load the DL model
model_1DCNN = keras.models.load_model('../../model/1D_CNN/1DCNN_16.h5')

# Load the test data.
f_data_root ='../../data/EX3/'
#save path
f_result_root = '../../result/EX3/'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)
f_result = f_result_root + 'Result_EX3_RMSE.h5'


#Information about base array Geometry (for traning model)
gamma_base = 0.5
test_num = 1000
test_batch = 100    #Large test_batch can accelerate the testing process

alpha_set = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# variables to save results
Ave_RSME_gamma_base = []
Ave_RSME_gamma_bias = []
Ave_RSME_ratio = []

# testing
for alpha in alpha_set: #Traverse the alpha set
    gamma_bias = gamma_base / alpha

    f_data = f_data_root + 'EX3_Alpha_'+str(alpha)+'.h5'
    # read test data
    f2 = h5py.File(f_data, 'r')
    DOA_ture = np.array(f2['angle']).reshape(-1)
    S_est_gamma_base = np.array(f2['S_est_gamma_base'])
    S_est_gamma_bias = np.array(f2['S_est_gamma_bias'])
    DOA_num = DOA_ture.shape[-1]
    [_, _, grids, _] = np.shape(S_est_gamma_base)

    # Temporary variable to save results
    RMSE_gamma_base = []
    RMSE_gamma_bias = []
    
    for doa_index in range(DOA_num):    #Traverse the EFOV
        # Temporary variable
        est_DOA_gamma_base = []
        est_DOA_gamma_bias = []

        for test_index in np.arange(0,test_num,test_batch): # Monte Carlo test
            DF_T_cnn = model_1DCNN.predict(S_est_gamma_base[test_index:test_index+test_batch, doa_index, :, :].reshape(test_batch, grids, 2),verbose=0)
            DF_T_cnn = np.array(DF_T_cnn)
            DF_T_cnn = np.reshape(DF_T_cnn, (test_batch,-1))
            doa_base = np.argmax(DF_T_cnn,axis=-1) - 60         #The base estimation i.e. \theta_{base}
            est_DOA_gamma_base.append(doa_base)

            DF_T_cnn = model_1DCNN.predict(S_est_gamma_bias[test_index:test_index+test_batch, doa_index, :, :].reshape(test_batch, grids, 2),verbose=0)
            DF_T_cnn = np.array(DF_T_cnn)
            DF_T_cnn = np.reshape(DF_T_cnn, (test_batch,-1))
            doa_bias = np.argmax(DF_T_cnn,axis=-1) - 60                      #The bias estimation i.e. \theta_{bias}
            doa_hat = asind(alpha*sind(doa_bias))                            #The generalized estimation i.e. \hat_{\theta}
            est_DOA_gamma_bias.append(doa_hat)     
            
        est_DOA_gamma_base = np.reshape(est_DOA_gamma_base,-1)
        est_DOA_gamma_bias = np.reshape(est_DOA_gamma_bias, -1)

        #The RMSE of base estimation
        RMSE_gamma_base.append(
            np.sqrt(mean_squared_error(est_DOA_gamma_base, np.repeat(DOA_ture[doa_index], test_num, axis=0))))
        
        #The RMSE of generalized estimation
        RMSE_gamma_bias.append(
            np.sqrt(mean_squared_error(est_DOA_gamma_bias, np.repeat(DOA_ture[doa_index], test_num, axis=0))))

    Ave_RSME_gamma_base_index = np.mean(RMSE_gamma_base)    #The average RMSE of base estimation corresponding to the current alpha
    Ave_RSME_gamma_bias_index = np.mean(RMSE_gamma_bias)    #The average RMSE of generalized estimation corresponding to the current alpha
    Ave_RMSE_ratio_index = Ave_RSME_gamma_bias_index / Ave_RSME_gamma_base_index    #The ratio of average RMSE corresponding to the current alpha
    print("alpha={} ".format(alpha), Ave_RSME_gamma_base_index, Ave_RSME_gamma_bias_index, Ave_RMSE_ratio_index)

    #Record the result corresponding to the current alpha
    Ave_RSME_gamma_base.append(Ave_RSME_gamma_base_index)
    Ave_RSME_gamma_bias.append(Ave_RSME_gamma_bias_index)
    Ave_RSME_ratio.append(Ave_RMSE_ratio_index)

# Save the results.
with h5py.File(f_result, 'a') as hf:  # 注意这里的模式是'a'，代表追加模式
        if 'Ave_RSME_gamma_base_1DCNN' in hf:
                del hf['Ave_RSME_gamma_base_1DCNN']
        if 'Ave_RSME_gamma_bias_1DCNN' in hf:
                del hf['Ave_RSME_gamma_bias_1DCNN']
        if 'Ave_RSME_ratio_1DCNN' in hf:
            del hf['Ave_RSME_ratio_1DCNN']
        hf.create_dataset('Ave_RSME_gamma_base_1DCNN', data=Ave_RSME_gamma_base)
        hf.create_dataset('Ave_RSME_gamma_bias_1DCNN', data=Ave_RSME_gamma_bias)
        hf.create_dataset('Ave_RSME_ratio_1DCNN', data=Ave_RSME_ratio)
        hf.close()
