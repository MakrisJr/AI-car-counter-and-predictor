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
from camera import Camera
from serializer import Serializer

#this is to open hostname
#ctx = ssl.create_default_context()
#ctx.check_hostname = False
#ctx.verify_mode = ssl.CERT_NONE

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
#url = 'http://192.168.0.153:8080/shot.jpg'

#initialize Serializer
#Stores data into dataframe with columns=['5 Minutes', 'Lane 1 Flow (Veh/5 Minutes)', '# Lane Points', '% Observed']
data = Serializer()

# import pretrained model
print('[INFO] Importing YOLOv8s.')
model = YOLO('data/yolov8s.pt')

#uncomment to process video
#change based on video
#video_name = "sample2.mp4"

#uncomment to process video
print('[INFO] Connecting to camera.')
camera = Camera(1)

# initialize count
count = 0

# import tracker
tracker = Tracker()

# height (y) points of the 2 lines
#tips:
#video res = 1080,1920 => line_up = 1060,car_height2=1100
#video res = 1920,1080 => line_up = 460,car_height2= 500
line_up = 300  # 1st line - change based on the video
car_height2 = line_up + 40  # 2nd line - change based on the video (always keep 40 difference with above for better results)
#distance threshold from lines:
offset = 6

# box_list to store vehicles' id that are going down
vh_down = {}
# counter_d for the down vehicles
counter_d = []

# box_list to store vehicles' id that are going up
vh_up = {}
# counter_d for the up vehicles
counter_u = []

# variables to track elapsed time
start_time = time.time()
reset_interval = 10 #300 # in seconds. Leave it 10 for experiments, switch to 300(5mins*60sec) for actual data 
current_time = datetime.datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)



while True:
    #open url
    #imgResp = urlopen(url)
    # read img
    #imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # set frame
    #frame = cv2.imdecode(imgNp, -1)
    # Check if the image is empty
    #if frame is None:
    #    print("Image is empty or couldn't be retrieved. Check the URL or the server.")
    #    continue

    frame = camera.getFrame()
    #predict from frame 
    results = model.predict(frame)
    #predict from camera
    #results = model.predict(source=1)

    a = results[0].boxes.data
    boxes_data = pd.DataFrame(a).astype("float")

    bbox_id = tracker.update(boxes_data)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        car_x = int(x3 + x4) // 2
        car_y = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        # going DOWN
        #if car passed through UP line, add to vh_down
        if line_up < (car_y + offset) and line_up > (car_y - offset):
            vh_down[id] = time.time()
        #
        if id in vh_down:
            #if car is close to DOWN line, check elapsed time
            if car_height2 < (car_y + offset) and car_height2 > (car_y - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter_d.count(id) == 0:
                    counter_d.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (car_x, car_y), 4, (0, 0, 255), -1)
                    cv2.line(frame, (177, car_height2), (927, car_height2), (0, 0, 255), 10)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # going UP
        #if car passed through DOWN line, add to vh_up
        if car_height2 < (car_y + offset) and car_height2 > (car_y - offset):
            vh_up[id] = time.time()
        if id in vh_up:
            if line_up < (car_y + offset) and line_up > (car_y - offset):
                elapsed1_time = time.time() - vh_up[id]
                if counter_u.count(id) == 0:
                    counter_u.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (car_x, car_y), 4, (0, 0, 255), -1)
                    cv2.line(frame, (0, car_height2), (227, car_height2), (0, 0, 255), 10)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Calculate flow every 10 seconds
    if time.time() - start_time >= reset_interval:
        flow = len(counter_d) + len(counter_u)  # Calculate flow as the sum of vehicles going up and down
        current_time += datetime.timedelta(minutes=5)  # Increment current time by 5 minutes
        current_time_str = current_time.strftime('%Y/%m/%d %H:%M')
        lane_points = 1  # Replace with your actual lane points value
        observed = 100  # Replace with your actual observed value
        
        # add data to serializer
        data.add_data(current_time_str, flow, lane_points, observed)

        # Reset counters
        counter_d = []
        counter_u = []
        start_time = time.time()

    
    d = len(counter_d)  # counter_d for down
    u = len(counter_u)  # counter_u for up

    cv2.putText(frame, ('goingup:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, ('UP'), (0, line_up), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(frame, (0, line_up), (camera.width, line_up), (0, 255, 0), 2)

    cv2.putText(frame, ('goingdown:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, ('DOWN'), (0, car_height2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (0, car_height2), (camera.width, car_height2), (0, 255, 255), 2)

    cv2.imshow("VEHICLE counter_d APP", frame)
    #exit when ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

print('[INFO] Closing.')
cv2.destroyAllWindows()
camera.close()

# Save data to csv file
print('[INFO] Saving data.')
data.save()
