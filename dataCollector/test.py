# how to run: python test.py or dataGenerator test.py


import cv2
import numpy as np
import ssl
from urllib.request import urlopen

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.0.153:8080/shot.jpg'

while True:
    imgResp = urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    # Check if the image is empty
    if img is None:
        print("Image is empty or couldn't be retrieved. Check the URL or the server.")
        continue

    cv2.imshow('temp',cv2.resize(img,(600,400)))
    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cv2.destroyAllWindows()
