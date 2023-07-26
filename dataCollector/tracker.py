import math
import numpy as np
import supervision as sv

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        # open classifiers file
        my_file = open("data/coco.txt", "r")
        file_data = my_file.read()
        self.class_list = file_data.split("\n") 


    def update(self, boxes_data):
        box_list = []
        for _, row in boxes_data.iterrows():
            d = int(row[5])
            c = self.class_list[d]
            if 'car' in c:
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                box_list.append([x1, y1, x2, y2])


        # Objects bounding boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in box_list:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                #if distance from other object is less than dist, then it is the same object
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    #update record of id and coordinates
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
class ZoneTracker():
    def __init__(self, model, video_info):
        self.model = model
        # Store the center positions of the objects
        self.id_count = 0

        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.center_points = {}

        self.count = [0, 0]
        # open classifiers file
        my_file = open("data/coco.txt", "r")
        file_data = my_file.read()
        self.class_list = file_data.split("\n") 

        self.video_info = video_info
        colors = sv.ColorPalette.default()
        polygons = [
            np.array([
            [894, 1060],[78, 1056],[194, 724],[938, 740]
            ]),np.array([
            [1026, 1032],[1866, 1032],[1826, 724],[978, 756]
            ])
        ]

        # initialize our zones
        self.zones = [
            sv.PolygonZone(
                polygon=polygon, 
                frame_resolution_wh=video_info.resolution_wh
            ) for polygon in polygons
        ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone, 
                color=colors.by_idx(index), 
                thickness=4,
                text_thickness=8,
                text_scale=4
            ) for index, zone in enumerate(self.zones)
        ]

        self.box_annotators = [
            sv.BoxAnnotator(
                color=colors.by_idx(index), 
                thickness=4, 
                text_thickness=4, 
                text_scale=2
            ) for index in range(len(polygons))
        ]
        

        self.frame = None

    def update(self, frame, detections):
        ids_seen = []
        for i, (zone, zone_annotator, box_annotator) in enumerate(zip(self.zones, self.zone_annotators, self.box_annotators)):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            labels = [
                f"{self.model.model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ 
                in detections_filtered
            ]
            
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, labels=labels)
            frame = zone_annotator.annotate(scene=frame)

            for x1,y1,x2,y2 in detections_filtered.xyxy:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                same_object_detected = False
                for id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 100:
                        same_object_detected = True
                        #update record of id and coordinates
                        self.center_points.update({self.id_count : (cx, cy)})
                        #print(self.center_points)
                        ids_seen.append(id)
                        break
  
                if same_object_detected is False:
                    self.count[i] += 1
                    self.center_points.update({self.id_count : (cx, cy)})
                    ids_seen.append(self.id_count)
                    self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        # Update dictionary with IDs not used removed
        self.center_points = {k : self.center_points[k] for k in ids_seen}
        self.frame = frame
        print("ID: " + str(self.id_count))


    def getAnnotatedFrame(self):
        return self.frame
    
    def getCount(self):
        return self.count
    

