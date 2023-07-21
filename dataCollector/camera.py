import cv2

class Camera:
    def __init__(self, cameraNo = 1):
        self.camera = cv2.VideoCapture(cameraNo)
        if self.camera.isOpened():
            self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raise ValueError(f"Camera '{1}' could not be opened.")
        
    def getFrame(self):
        _, frame = self.camera.read()
        return frame
    
    def close(self):
        self.camera.release()

#code to find available camera index
#for i in range(10):
#    cap = cv2.VideoCapture(i)
#    if cap.isOpened():
#        print(f"Camera index {i} is available")
#        cap.release()
#    else:
#        print(f"Camera index {i} is not available")