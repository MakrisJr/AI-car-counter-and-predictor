# Folder Structure

## dataCollector/data:

- **coco.txt**: Text file with several object classes in string form.
- **dataGenerator.py**: Algorithm to expand our dataset.
- **main.py**: Algorithm to count objects from IP webcam (Android only).
- **mySample.mp4**: Small video sample for test usage.
- **resampled_data.csv**: Data after expansion.
- **sample2.mp4**: Another video sample for test usage.
- **test.py**: Test file to see if IP webcam is working.
- **tracker.py**: `Tracker()` class for object tracking (do NOT modify).
- **traffic_data_6_2023.csv**: Data we collected from a real-time video.
- **yolov8s.pt**: Pre-trained weights of YOLO model to recognize objects (do NOT delete or modify).

## trafficPrediction: Use of our AI model

### data:

- **simple_preprocessor.py**: Preprocess data for our AI model.
- **train.csv**: Training dataset.
- **test.csv**: Testing dataset.

### images:

- **data.png**: Graph that shows the predicted data vs the actual data.
- **training_mae.png**: Graph that shows training and validation error.

### model:

- **cnn1D.py**: Contains 3 functions (one for each model including CNN, RNN).
  - *Hint: If you want to change the model, just change the function name in `train_model` and that's it.*

### weights:

- **trained_cnn.h5**: Saved binary form of our model's trained weights to use them in prediction and avoid training every time.

- **model_architecture.png**

- **predict_traffic.py**: Algorithm to predict the traffic for 1 month.
- **train_model.py**: Algorithm to train our model using preprocessed data of 2 months.

---

For directions on how to run and more explanations, please look inside the files.
