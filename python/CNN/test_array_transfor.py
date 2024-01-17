import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, Softmax
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.initializers import glorot_normal
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x

model_CNN = load_model('../../../model/CNN/Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5')


# Load the Test Data1
f_data_root ='../../../data/EX3/'
f_data = f_data_root+'EX1_Alpha_1.h5'
d1 = 0.5
alpha = 1
d2 = d1/alpha

f2 = h5py.File(f_data, 'r')
GT_angles = np.array(f2['angle'])
DOA_num_d2 = len(GT_angles.reshape(-1))
RX_sam_test_d1 = np.array(f2['sam_d1'])
X_test_data_sam_d1 = RX_sam_test_d1.swapaxes(1,3)
RX_sam_test_d2 = np.array(f2['sam_d2'])
X_test_data_sam_d2 = RX_sam_test_d2.swapaxes(1,3)

est_DOA_d1 = []
est_DOA_d2 = []
est_phase_d1 = []
est_phase_d2 = []


for doa_index in range(DOA_num_d2):
        x_pred_sam_d2 = model_CNN.predict(np.expand_dims(X_test_data_sam_d2[doa_index,:,:,:],0))
        doa_d2 = np.argmax(x_pred_sam_d2)-60
        est_DOA_d2.append(asind(alpha*sind(doa_d2)))

        x_pred_sam_d1 = model_CNN.predict(np.expand_dims(X_test_data_sam_d1[doa_index,:,:,:],0))
        doa_d1 = np.argmax(x_pred_sam_d1)-60
        est_DOA_d1.append(doa_d1)


RMSE_d1 = np.sqrt(mean_squared_error(est_DOA_d1, GT_angles))
RMSE_d2 = np.sqrt(mean_squared_error(est_DOA_d2, GT_angles))
print(RMSE_d1)
print(RMSE_d2)
plt.figure()
plt.plot(GT_angles,est_DOA_d1)
plt.grid()
plt.show()


plt.figure()
plt.plot(GT_angles,est_DOA_d2)
plt.grid()
plt.show()


# f_result_root = '../../../result/EX1/'
# f_result = f_result_root+'phase_alpha_0.5.h5'
# hf = h5py.File(f_result, 'w')
# hf.create_dataset('est_phase_d1', data=est_phase_d1)
# hf.create_dataset('est_phase_d2', data=est_phase_d2)
# hf.close()



