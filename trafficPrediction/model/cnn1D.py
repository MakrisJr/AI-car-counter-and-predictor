from keras.layers import Dense, Dropout, Flatten,Conv1D, BatchNormalization
from keras.models import Sequential
from keras import regularizers


#look_back(int): Number of past steps to use for prediction
def create_1Dcnn(look_back):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=3, activation='elu', input_shape=(look_back, 1), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=128, kernel_size=3, activation='elu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(50, activation='elu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


from keras.models import Sequential
from keras.layers import Dense, LSTM,GRU

# look_back: the number of previous time steps to use as input variables to predict the next time period
def create_lstm(look_back):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(64))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='elu'))
    return model

#Best of 3
# look_back: the number of previous time steps to use as input variables to predict the next time period
def create_gru(look_back):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=(look_back, 1)))
    model.add(GRU(64))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='elu'))
    return model