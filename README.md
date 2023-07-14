folder structure:
|   dataCollector:
|   | coco.txt: text file with several object classes in string form 
|   | dataGenerator.py: algorithm to expand our dataset
|   | main.py: algorithm to count objects from IP webcam (android only)
|   | mySample.mp4: small video sample for test usage
|   | resampled_data.csv: data after expansion
|   | sample2.mp4: another cideo sample for test usage
|   | test.py: test file to see if IP webcam is working
|   | tracker.py: Tracker() class for object tracking (do NOT modify)
|   | traffic_data_6_2023.csv: some data we collected from a real time video
|   | yolov8s.pt: pre-trained weights of yolo model to recognize objects (do NOT delete or modify)
|   trafficPrediction: use of our AI model
|   | data: 
|   | |  simple_preprocessor.py: preprocess data for our AI model
|   | |  train.csv: training dataset
|   | |  test.csv: testing dataset
|   | images:
|   | | data.png: graph that shows the predicted data vs the actual data
|   | | training_mae.png: graph that shows training and validation error
|   | model:
|   | | cnn1D.py: contains 3 functions(one for each model including cnn,rnn)
|   | | hint: if you want to change model just change the function name in train_model and thats' it.
|   | weights:
|   | | trained_cnn.h5:we saved in binary form our model's trained weights in order to use them in prediction and avoid training every time
|   | model_architecture.png 
|   | predict_traffic.py: algorithm to predict the traffic for 1 month
|   | train_model.py: algorithm to train our model using preprocessed data of 2 months.

directions for how to run more explanations please look inside the files.