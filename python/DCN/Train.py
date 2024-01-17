import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.optimizers import RMSprop,Adam


# file path to save neural network parameters
f_model_root = '../../model/1D CNN/'
if not os.path.exists(f_model_root):
        os.makedirs(f_model_root)

# Load the training data.
f_data_root ='../../data/Train/'
f_data = f_data_root+'train_data.mat'

read_temp=scipy.io.loadmat(f_data)
S_est=read_temp['S_est']
S_label=read_temp['S_label']
S_label1 = np.expand_dims(S_label, 2)
[Sample,L,dim]=np.shape(S_est)
nb_epoch=600
batch_size=64

optimizer=Adam(lr=0.001)

#
cnn = Sequential()
cnn.add(Convolution1D(12,25,  input_shape=(L,dim), activation='relu',name="cnn_1", padding='same'))
cnn.add(Convolution1D(6,15, activation='relu',name="cnn_2", padding='same'))
cnn.add(Convolution1D(3,5, activation='relu',name="cnn_4", padding='same'))
cnn.add(Convolution1D(1,3,activation='relu',name="cnn_5", padding='same'))
cnn.compile(loss='mse', optimizer=optimizer)
cnn.summary()
history_cnn=cnn.fit(S_est, S_label1,epochs=nb_epoch, batch_size=batch_size,shuffle=True
                ,verbose=2,validation_split=0.2)
cnn.save(f_model_root+'1DCNN_16.h5')

figsize = 8,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(np.array(history_cnn.history['val_loss'])*1000)
plt.legend(['DCN+relu'])
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 13,}
plt.xlabel('Epoch',font2)
plt.ylabel('Test MSE(*1e$^-$$^3$)',font2)
plt.show()

