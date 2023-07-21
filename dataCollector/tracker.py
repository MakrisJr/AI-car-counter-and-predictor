import math


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