# how to run: python predict_traffic.py or python3 predict_traffic.py



from model.cnn1D import create_gru
from data.simple_preprocessor import process_data
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pickle

#how many steps to look back
LOOK_BACK = 12

#split dataset
X_test,Y_test,Scaler_test,df_test = process_data(file_name="data/test.csv",look_back=LOOK_BACK)

# load the trained cnn
model = load_model('weights/' + 'trained_cnn' + '.h5')

#draw the model's architecture (not working on powershell, only in wsl like ubuntu) (problem with graphiz)
#cheat solution: uncomment, run in linux shell, comment, run in powershell.
#tf.keras.utils.plot_model(model, to_file="model_architecture.png", show_shapes=True)


# load the scaler used during training
with open('weights/' + 'scaler.pkl', 'rb') as handle:
    Scaler_train = pickle.load(handle)

# Generate predictions for the test set
pred_test = model.predict(X_test)

# Reshape the predictions into a 2D array
pred_test = pred_test.reshape(pred_test.shape[0], -1)

# Inverse transform the predicted and actual values
pred_test = Scaler_train.inverse_transform(pred_test)
Y_test = Scaler_train.inverse_transform(Y_test.reshape(-1, 1))

# Get the corresponding timestamps from the test dataset
timestamps = df_test["5 Minutes"].values[-len(pred_test):]

# Convert the timestamps to a readable format (if necessary)
timestamps = pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M")

# Plot the last day of data
plt.figure(figsize=(10, 6))
plt.plot(timestamps[-12*24:], Y_test[-12*24:], label='True Data')
plt.plot(timestamps[-12*24:], pred_test[-12*24:], label='Predicted Data')
plt.title('True Data vs Predicted Data')
plt.ylabel('Traffic')
plt.xlabel('Time')
plt.xticks(rotation=45)  # Rotate X-axis labels for better readability
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('images/data.png', dpi=300)  # Specify the filename and DPI (optional)
plt.show()

#print final errors to recognize (under, good, over) fit
mse = mean_squared_error(Y_test, pred_test)
print("Mean Squared Error (MSE):", mse)
