# how to run: python main.py or python3 main.py


import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist
import datetime
import os
import ssl
from urllib.request import urlopen

#this is to open hostname
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#to get this follow the below instructions:
#1. download IP Webcam app on your android phone
#2. open app
#3. scroll to the bottom
#4. start server (NOTE: your laptop and phone must be connected to the same wifi)
#5. type the http url on your browser.
#6  select "javascript" on "video renderer"
#7. right click on the video 
#8. copy image link 
#9. paste it here
#10. you will get something like this :http://192.168.0.153:8080/shot.jpg?rnd=280544
#11. remove ?rnd=280544
url = 'http://192.168.0.153:8080/shot.jpg'

# Get the current month and year
current_date = datetime.datetime.now()
current_month = current_date.month
current_year = current_date.year

# Create an empty DataFrame to store the data
data = pd.DataFrame(columns=['5 Minutes', 'Lane 1 Flow (Veh/5 Minutes)', '# Lane Points', '% Observed'])

# import pretrained model
model = YOLO('yolov8s.pt')

#uncomment to process video
# change based on video
#video_name = "sample2.mp4"

#uncoment if you want to find camera index
#find the index of the camera
#for i in range(10):
  #  cap = cv2.VideoCapture(i)
   # if cap.isOpened():
    #    print(f"Camera index {i} is available")
     #   cap.release()
    #else:
     #   print(f"Camera index {i} is not available")

#uncomment to process video
#cap = cv2.VideoCapture(video_name) #replace 0(integrated camera) with video name or path if you want to analyze the recorded video

#uncomment if you want to process video files.
# find width and height of the video
#if cap.isOpened(): 
 #   width = int(cap.get(3))  # float `width`
  #  height = int(cap.get(4))  # float `height
   # print(width)
    #print(height)
#else: 
 #   print("Video not openned")

# open classifiers file
my_file = open("coco.txt", "r")
file_data = my_file.read()
class_list = file_data.split("\n") 

# initialize count
count = 0

# import tracker
tracker = Tracker()

# height (y) points of the 2 lines
#tips:
#video res = 1080,1920 => car_height1 = 1060,car_height2=1100
#video res = 1920,1080 => car_height1 = 460,car_height2= 500
car_height1 = 760  # 1st line - change based on the video
car_height2 = 860  # 2nd line - change based on the video (always keep 40 difference with above for better results)

offset = 6

# list to store vehicles' id that are going down
vh_down = {}
# counter for the down vehicles
counter = []

# list to store vehicles' id that are going up
vh_up = {}
# counter for the up vehicles
counter1 = []

# variables to track elapsed time
start_time = time.time()
reset_interval = 300  # in seconds. Leave it 10 for experiments, switch to 300(5mins*60sec) for actual data 
current_time = datetime.datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)



while True:
    #open url
    imgResp = urlopen(url)
    # read img
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # set frame
    frame = cv2.imdecode(imgNp, -1)
    # Check if the image is empty
    if frame is None:
        print("Image is empty or couldn't be retrieved. Check the URL or the server.")
        continue

    #get h,w from frame
    height, width, _ = frame.shape  # get the dimensions from the frame itself
    print(width)    
    print(height)

    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (width, height))

    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        car_x = int(x3 + x4) // 2
        car_y = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        # going DOWN
        if car_height1 < (car_y + offset) and car_height1 > (car_y - offset):
            vh_down[id] = time.time()
        if id in vh_down:
            if car_height2 < (car_y + offset) and car_height2 > (car_y - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (car_x, car_y), 4, (0, 0, 255), -1)
                    cv2.line(frame, (177, car_height2), (927, car_height2), (0, 0, 255), 10)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # going UP
        if car_height2 < (car_y + offset) and car_height2 > (car_y - offset):
            vh_up[id] = time.time()
        if id in vh_up:
            if car_height1 < (car_y + offset) and car_height1 > (car_y - offset):
                elapsed1_time = time.time() - vh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (car_x, car_y), 4, (0, 0, 255), -1)
                    cv2.line(frame, (0, car_height2), (227, car_height2), (0, 0, 255), 10)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Calculate flow every 10 seconds
    if time.time() - start_time >= reset_interval:
        flow = len(counter) + len(counter1)  # Calculate flow as the sum of vehicles going up and down
        current_time += datetime.timedelta(minutes=5)  # Increment current time by 5 minutes
        current_time_str = current_time.strftime('%m/%d/%Y %H:%M')
        lane_points = 1  # Replace with your actual lane points value
        observed = 100  # Replace with your actual observed value
        
        #  Append the data to the DataFrame
        new_row = pd.DataFrame({'5 Minutes': [current_time_str],
                                'Lane 1 Flow (Veh/5 Minutes)': [flow],
                                '# Lane Points': [lane_points],
                                '% Observed': [observed]})
        data = pd.concat([data, new_row], ignore_index=True)

        # Reset counters
        counter = []
        counter1 = []
        start_time = time.time()

    
    d = len(counter)  # counter for down
    u = len(counter1)  # counter for up

    cv2.putText(frame, ('goingup:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, ('UP'), (0, car_height1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(frame, (0, car_height1), (width, car_height1), (0, 255, 0), 2)

    cv2.putText(frame, ('goingdown:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, ('DOWN'), (0, car_height2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (0, car_height2), (width, car_height2), (0, 255, 255), 2)

    cv2.imshow("VEHICLE COUNTER APP", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

frame.release()
cv2.destroyAllWindows()

# Save the data to an Excel file
filename = f"traffic_data_{current_month}_{current_year}.csv"
data.to_csv(filename, index=False)
