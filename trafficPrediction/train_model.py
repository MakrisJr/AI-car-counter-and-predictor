# how to run: python train_model.py or python3 train_model.py



from model.cnn1D import create_1Dcnn,create_lstm,create_gru
from data.simple_preprocessor import process_data
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import pandas as pd

import pickle

#Constants
LOOK_BACK = 12 #how many steps to look back
LEARNING_RATE = 0.01
BATCH_SIZE = 256 

#Data
X_train,Y_train,Scaler_train,df_train = process_data(file_name="data/train.csv",look_back=LOOK_BACK)

#Learning rate scheduler
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)

callbacks = [early_stop]

#create Neural Network
#uncomment if you want to change model
#model = create_1Dcnn(look_back=LOOK_BACK)
#model = create_lstm(look_back=LOOK_BACK)
model = create_gru(look_back=LOOK_BACK)

#compile Neural Network
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#run Neural Network and record history
history = model.fit(X_train, Y_train, epochs=100,shuffle=True, validation_split=0.1,batch_size=BATCH_SIZE, callbacks=callbacks,verbose=1)

#save the trained cnn
model.save('weights/' + 'trained_cnn' + '.h5')

#save the scaler for use in prediction
with open('weights/' + 'scaler.pkl', 'wb') as handle:
    pickle.dump(Scaler_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Second subplot-> errors graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout() # Ensures a nice layout
plt.savefig('images/training_mae.png', dpi=300)  # Specify the filename and DPI (optional)
plt.show()
