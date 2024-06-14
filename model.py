import math
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

class LiftDetection:
    def __init__(self, class_ids, conf_level, thr_centers, frame_max, patience, alpha):
        self.class_ids = class_ids
        self.conf_level = conf_level
        self.thr_centers = thr_centers
        self.frame_max = frame_max
        self.patience = patience
        self.alpha = alpha
        self.centers_old = {}
        self.obj_id = 0
        self.count_p = 0
        self.lastKey = ''

    def update_tracking(self, obj_center, thr_centers, lastKey, frame, frame_max):
        is_new = 0
        lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in self.centers_old.items()]
        lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
        previous_pos = [(k, centers) for k, centers in lastpos if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]

        if previous_pos:
            id_obj = previous_pos[0][0]
            self.centers_old[id_obj][frame] = obj_center
        else:
            if lastKey:
                last = int(lastKey.split('ID')[1])
                id_obj = 'ID' + str(last + 1)
            else:
                id_obj = 'ID0'
            is_new = 1
            self.centers_old[id_obj] = {frame: obj_center}
            lastKey = id_obj

        return self.centers_old, id_obj, is_new, lastKey

    def filter_tracks(self, centers, patience):
        filter_dict = {}
        for k, i in centers.items():
            d_frames = i.items()
            filter_dict[k] = dict(list(d_frames)[-patience:])
        return filter_dict

    def process_frame(self, frame):
        cnt =0
        class_names = open("coco.names", "r")
        class_names = class_names.read()
        class_names = class_names.split("\n")

        scale_percent = 100
        ROI = frame[243:666, 97:774]
        area_ROI = [np.array([(97, 243), (774, 243), (97, 666), (774, 666)], np.int32)]

        # Check if the frame is not empty and the dimensions are greater than zero
        if ROI is not None and ROI.shape[0] > 0 and ROI.shape[1] > 0:
            ROI = cv2.resize(ROI, (ROI.shape[1] * scale_percent // 100, ROI.shape[0] * scale_percent // 100), interpolation=cv2.INTER_AREA)
            y_hat = model.predict(ROI, conf=self.conf_level, classes=self.class_ids, device="cpu", verbose=False)

            x, y, width, height = 243,97, 666, 774
            mask = ROI.copy()
            #cv2.rectangle(mask, (0, 0), (ROI.shape[1], ROI.shape[0]), (0, 0, 0), -1)
            #cv2.rectangle(mask, (x, y), (x + width, y + height), (0, 0, 255), -1)

            overlay = frame.copy()
            cv2.polylines(overlay, pts = area_ROI, isClosed=True, color=(255,0,0), thickness= 2)
            cv2.fillPoly(overlay, area_ROI, (255,0,0))
            frame_with_mask = cv2.addWeighted(mask, 0.5, ROI, 0.5, 0)

            boxes = y_hat[0].boxes.xyxy.cpu().numpy()
            conf = y_hat[0].boxes.conf.cpu().numpy()
            result = y_hat[0].boxes.data
            result = pd.DataFrame(result)

            positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

            for ix, row in enumerate(positions_frame.iterrows()):
                xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')
                center_x, center_y = int((xmax + xmin) / 2), int((ymax + ymin) / 2)
                self.centers_old, id_obj, is_new, self.lastKey = self.update_tracking((center_x, center_y), self.thr_centers, self.lastKey, 0, self.frame_max)
                self.count_p += is_new

                cv2.rectangle(frame_with_mask, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame_with_mask, id_obj + ':' + str(np.round(confidence, 2)), (xmin, ymin - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 1)

            for index, rows in result.iterrows():
                d = int(rows[5])
                c = class_names[d]
                cnt = c.count("person")
            if cnt>0:
                result_message = True
            else:
                result_message = False
        else:
            result_message = False

        return frame_with_mask, mask, result_message

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Create an instance of the LiftDetection class
class_ids = [0]
conf_level = 0.4
thr_centers = 20
frame_max = 10
patience = 100
alpha = 0.1
lift_detection = LiftDetection(class_ids, conf_level, thr_centers, frame_max, patience, alpha)


class model_class:
    
    def model_yolo(self,video_path):
        
        cap = cv2.VideoCapture(video_path)
        count = 0

        while cap.isOpened():

            while count%40 != 0:
                count+=1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame and get the result message, mask, and frame with mask
            frame_with_mask, mask, result_message = lift_detection.process_frame(frame)

            # Save the frames as images
            cv2.imwrite(f"frame_{count}.png", frame)
            cv2.imwrite(f"mask_{count}.png", mask)
            cv2.imwrite(f"frame_with_mask_{count}.png", frame_with_mask)

            # Display the frame with the result message
            #cv2.putText(frame_with_mask, result_message, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow("Lift Detection", frame_with_mask)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1

        cap.release()
        cv2.destroyAllWindows()
        return result_message