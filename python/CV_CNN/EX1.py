import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from model import *
from data import MyDataset,My_Train_DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('../../../model/CV_CNN/Model_CV_CNN.pth',map_location='cpu')
# Load the Test Data1
f_data_root ='../../../data/EX1/'
f_data = f_data_root+'EX1_Alpha_0.5.h5'


f2 = h5py.File(f_data, 'r')
GT_angles_d1 = np.array(f2['angle'])
DOA_num_d1 = len(GT_angles_d1.reshape(-1))
RX_sam_test_d1 = np.array(f2['sam'])
RX_sam_test_d1 = torch.tensor(RX_sam_test_d1)
X_test_data_sam = (RX_sam_test_d1[:,0,:,:].type(torch.complex64)+1j*RX_sam_test_d1[:,1,:,:].type(torch.complex64)).unsqueeze(1).to(device)

DOA_d1 = []
Phase_d1 = []
est_DOA_d2 = []
est_phase_d2 = []
phase_dff = []
d1 = 0.5
alpha = 0.5
d2 = d1/alpha
#save path
f_result_root = '../../../result/EX1/'
f_result = f_result_root+'Result_EX1_Alpha_0.5.h5'
for doa_index in range(DOA_num_d1):

        x_pred_sam = model(X_test_data_sam[doa_index,:,:,:]).detach().cpu().numpy()
        doa_d2 = np.argmax(x_pred_sam) - 60
        est_DOA_d2.append(doa_d2)
        phase_d2 = d1 * np.sin(doa_d2 / 180 * np.pi)
        est_phase_d2.append(phase_d2)

        doa_d1 = GT_angles_d1[doa_index]
        DOA_d1.append(doa_d1)
        phase_d1 = d2 * np.sin(doa_d1 / 180 * np.pi)
        Phase_d1.append(phase_d1)



RMSE = np.sqrt(mean_squared_error(Phase_d1, est_phase_d2))
print(RMSE)
plt.figure()
plt.plot(GT_angles_d1,est_DOA_d2)
plt.show()
plt.grid()

plt.figure()
plt.plot(GT_angles_d1,est_phase_d2)
plt.plot(GT_angles_d1,Phase_d1)
plt.show()
plt.grid()

f_result_root = '../../../result/EX1/'
f_result = f_result_root+'CV_CNN_phase_alpha_0.5.h5'
hf = h5py.File(f_result, 'w')
hf.create_dataset('est_phase_d1', data=Phase_d1)
hf.create_dataset('est_phase_d2', data=est_phase_d2)
hf.close()
# filename_root ='E:/code/array_tansfor/data/EX1/'
# filename1 = filename_root+'DOA_set_K_1_d0.5.h5'
# f2 = h5py.File(filename1, 'r')
# GT_angles_d1 = np.array(f2['angle'])
# DOA_num_d1 = len(GT_angles_d1.reshape(-1))
# RX_sam_test_d1 = np.array(f2['sam'])
# RX_sam_test_d1 = torch.tensor(RX_sam_test_d1)
# X_test_data_sam_d1 = (RX_sam_test_d1[:,0,:,:].type(torch.complex64)+1j*RX_sam_test_d1[:,1,:,:].type(torch.complex64)).unsqueeze(1).to(device)
#
# filename2 = filename_root+'DOA_set_K_1_d1.h5'
# f2 = h5py.File(filename2, 'r')
# GT_angles_d2 = np.array(f2['angle'])
# DOA_num_d2 = len(GT_angles_d2.reshape(-1))
# RX_sam_test_d2 = np.array(f2['sam'])
# RX_sam_test_d2 = torch.tensor(RX_sam_test_d2)
# X_test_data_sam_d2 = (RX_sam_test_d2[:,0,:,:].type(torch.complex64)+1j*RX_sam_test_d2[:,1,:,:].type(torch.complex64)).unsqueeze(1).to(device)
#
# f_result_root = 'E:/code/array_tansfor/result\EX1/'
# f_result = f_result_root+'EX1_result_CV_CNN_d10.5_d21.0.h5'
#
# est_DOA_d1 = []
# est_DOA_d2 = []
# est_phase_d1 = []
# est_phase_d2 = []
# d1 = 0.5
# d2 = 1
#
# for doa_index in range(DOA_num_d1):
#         x_pred_sam_d1 = model(X_test_data_sam_d1[doa_index,:,:,:]).detach().cpu().numpy()
#         doa_d1 = np.argmax(x_pred_sam_d1)-60
#         est_DOA_d1.append(doa_d1)
#         phase_d1= d2 * np.sin(doa_d1/ 180 * np.pi)
#         est_phase_d1.append(phase_d1)
#
#         x_pred_sam_d2 = model(X_test_data_sam_d2[doa_index,:,:,:]).detach().cpu().numpy()
#         doa_d2 = np.argmax(x_pred_sam_d2) - 60
#         est_DOA_d2.append(doa_d2)
#         phase_d2 = d1 * np.sin(doa_d2 / 180 * np.pi)
#         est_phase_d2.append(phase_d2)
#
#
#
# plt.figure()
# plt.plot(GT_angles_d2,est_DOA_d1)
# plt.plot(GT_angles_d2,est_DOA_d2)
# plt.show()
# plt.grid()
#
# plt.figure()
# plt.plot(GT_angles_d2,est_phase_d1)
# plt.plot(GT_angles_d2,est_phase_d2)
# plt.show()
# plt.grid()
#
#
# hf = h5py.File(f_result, 'w')
# hf.create_dataset('est_DOA_d1', data=est_DOA_d1)
# hf.create_dataset('est_DOA_d2', data=est_DOA_d2)
# hf.create_dataset('est_phase_d1', data=est_phase_d1)
# hf.create_dataset('est_phase_d2', data=est_phase_d2)
#
# pass
#
#
