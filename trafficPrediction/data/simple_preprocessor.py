import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



look_back = 14

# Creating sliding window
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


def process_data(file_name,look_back):
    # Load  data
    df = pd.read_csv(file_name)
    df["5 Minutes"] = pd.to_datetime(df["5 Minutes"], format="%d/%m/%Y %H:%M")
    # Sort by time
    df = df.sort_values("5 Minutes")
    # Normalize the dataset
    scaler = MinMaxScaler()
    df["Lane 1 Flow (Veh/5 Minutes)"] = scaler.fit_transform(df["Lane 1 Flow (Veh/5 Minutes)"].values.reshape(-1,1))
    X, Y = create_dataset(df["Lane 1 Flow (Veh/5 Minutes)"].values, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X,Y,scaler,df