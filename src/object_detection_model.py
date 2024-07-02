import cv2
import logging
import time
import random
import requests
import numpy as np

from utils.renkKontrol import UAP_renk, UAI_renk
from utils.auto_model import AutoDetectionModel
from utils.predict import get_prediction

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation


class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.current_frame = None
        self.last_frame = None
        self.model_path = "data/best.pt"
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov5',
            model_path=self.model_path,
            confidence_threshold=0.6,
            device="cuda:0"
        )
        self.old_class_names = ['insan','motor','arac','ismakinesi','kamyon','otobus','uap','uai','gemi','tren']
        self.class_names = ['Tasit', 'Insan', 'UAP', 'UAI']
        self.tasit_classes = [1, 2, 3, 4, 5, 8, 9]
        
    @staticmethod
    def download_image(img_url, images_folder, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time

        for attempt in range(retries):
            try:
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()
                img_bytes = response.content
                image_name = img_url.split("/")[-1]

                with open(images_folder + image_name, 'wb') as img_file:
                    img_file.write(img_bytes)

                t2 = time.perf_counter()
                logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                return
            except requests.exceptions.RequestException as e:
                logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

        logging.error(f"Failed to download image from {img_url} after {retries} attempts.")

    def process(self, prediction,evaluation_server_url,health_status):
        self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images/")
        current_frame_path = "./_images/" + prediction.image_url.split("/")[-1]
        self.current_frame = cv2.imread(current_frame_path)
        if self.last_frame is not None: self.last_frame = self.current_frame # Eğer frame ilk frame ise last_frame'e atama yapma
        if self.current_frame is None:
            logging.error(f"Image at path {current_frame_path} could not be loaded.")
            return None
        
        # Modelin Tahmin Yapma Fonksiyonu
        frame_results = self.detect(prediction,health_status)
        return frame_results

    def detect(self, prediction,health_status):
        image = self.current_frame.copy()
        result = get_prediction(image, self.detection_model)
        object_prediction_list = result.object_prediction_list
        detected_objects = []
        
        for object_prediction in object_prediction_list:
            box = object_prediction.bbox.to_xyxy()
            conf = object_prediction.score.value
            cls = object_prediction.category.id
            
            if cls in self.tasit_classes: new_cls = 0   # Taşıt
            elif cls == 0:                new_cls = 1   # İnsan
            elif cls == 6:                new_cls = 2   # UAP
            elif cls == 7:                new_cls = 3   # UAI
            else:                         new_cls = -1  # Hiçbiri
            
            detected_objects.append({
                "box": box,
                "class_id": new_cls,
                "label": f"{self.class_names[new_cls]} ({conf:.2f})"
            })
            
        for obj in detected_objects:
            box = obj["box"]
            class_id = obj["class_id"]
            label = obj["label"]
            
            if class_id == 2: # UAP ise   
                roi = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (15, 15), 0)
                
                circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=40, minRadius=50, maxRadius=100)
                if circles is not None:
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        cv2.circle(image, (int(box[0]) + x, int(box[1]) + y), r, (255, 0, 0), 4)
                        circle_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                        cv2.circle(circle_mask, (x, y), r, 255)
                        masked_roi = cv2.bitwise_and(roi, roi, mask=circle_mask)

                        pixel_count = 0
                        for i in range(masked_roi.shape[0]):
                            for j in range(masked_roi.shape[1]):
                                if circle_mask[i, j] == 255 and not UAP_renk(masked_roi[i, j]):
                                    pixel_count += 1

                        if pixel_count > 16500:
                            status = "Inilemez"
                            cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            status = "Inilebilir"
                else:
                    status = "Inilemez"
                    cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if status != "Inilemez":
                    for other_obj in detected_objects:
                        if other_obj == obj:
                            status = "Inilebilir"
                            continue

                        other_box = other_obj['box']
                        
                        if (box[0] <= other_box[0] <= box[2] and box[1] <= other_box[1] <= box[3]) and \
                        (box[0] <= other_box[2] <= box[2] and box[1] <= other_box[3] <= box[3]):
                            status = "Inilemez"
                            
                    cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
            
            elif class_id == 3: # UAI ise
                roi = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (15, 15), 0)

                circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=35, param2=35, minRadius=40, maxRadius=100)

                if circles is not None:
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        cv2.circle(image, (int(box[0]) + x, int(box[1]) + y), r, (255, 0, 0), 4)
                        circle_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                        cv2.circle(circle_mask, (x, y), r, 255, thickness=-1)
                        masked_roi = cv2.bitwise_and(roi, roi, mask=circle_mask)

                        pixel_count = 0
                        for i in range(masked_roi.shape[0]):
                            for j in range(masked_roi.shape[1]):
                                if circle_mask[i, j] == 255 and not UAI_renk(masked_roi[i, j]):
                                    pixel_count += 1

                        if pixel_count > 16500:
                            status = "Inilemez"
                            cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            status = "Inilebilir"
                else:
                    status = "Inilemez"
                    cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if status != "Inilemez":
                    for other_obj in detected_objects:
                        if other_obj == obj:
                            status = "Inilebilir"
                            continue

                        other_box = other_obj['box']
                        
                        if (box[0] <= other_box[0] <= box[2] and box[1] <= other_box[1] <= box[3]) and \
                        (box[0] <= other_box[2] <= box[2] and box[1] <= other_box[3] <= box[3]):
                            status = "Inilemez"
                            
                    cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                status = "Inis Alani Degil"
                cv2.putText(image, str(status), (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if class_id == 0:
                    cls = classes["Tasit"]
                elif class_id == 1:
                    cls = classes["Insan"]

            landing_status = landing_statuses[status]
            top_left_x = box[0]
            top_left_y = box[1]
            bottom_right_x = box[2]
            bottom_right_y = box[3]

            d_obj = DetectedObject(cls, landing_status, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            prediction.add_detected_object(d_obj)
            
            
        # -- > LOKASYON TAHMINI < --
        if health_status == '0':
            if self.last_frame is not None:
                logging.info("İlk frame olduğu için lokasyon tahmini yapılamadı.")
                pred_translation_x = 0
                pred_translation_y = 0
            else:
                predicted_x, predicted_y = location_prediction(image, self.last_frame)
                kalman_x, kalman_y = kalman_filter(predicted_x, predicted_y)
                pred_translation_x = kalman_x * pixel2meter
                pred_translation_y = kalman_y * pixel2meter
            self.last_frame = self.current_frame
        else:
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y

        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y)
        prediction.add_translation_object(trans_obj)
        
        return prediction
    
    
    

# Kalman Filter and location prediction methods
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

def location_prediction(current_frame, last_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.undistort(current_frame_gray, cameraMatrix, distCoeffs)
    last_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    last_frame_gray = cv2.undistort(last_frame_gray, cameraMatrix, distCoeffs)
    keypoints1, descriptors1 = orb.detectAndCompute(current_frame_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(last_frame_gray, None)
    if descriptors1 is None or descriptors2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            matches_mask = mask.ravel().tolist()
            inliers = np.sum(matches_mask)
            if inliers > 0:
                dx = np.mean([keypoints2[m.trainIdx].pt[0] - keypoints1[m.queryIdx].pt[0] for i, m in enumerate(matches) if matches_mask[i]])
                dy = np.mean([keypoints2[m.trainIdx].pt[1] - keypoints1[m.queryIdx].pt[1] for i, m in enumerate(matches) if matches_mask[i]])
            else: dx = dy = 0
        else: dx = dy = 0
    else: dx = dy = 0
    return dx, dy

cameraParams = {"IntrinsicMatrix": np.array([[1.4133e+03, 0, 950.0639], [0, 1.4188e+03, 543.3796], [0, 0, 1]]),
    "RadialDistortion": np.array([-0.0091, 0.0666, 0, 0])}
cameraMatrix = cameraParams["IntrinsicMatrix"]
distCoeffs = cameraParams["RadialDistortion"].reshape(1, 4)
orb = cv2.ORB_create()

dt = 1.0
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.F = np.array([[1, dt], [0, 1]])
kf.H = np.array([[1, 0], [0, 1]])
kf.Q = np.array([[1, 0], [0, 1]])
kf.R = np.array([[10, 0], [0, 10]])
kf.x = np.array([[0], [0]])
kf.P = np.array([[1, 0], [0, 1]])

def kalman_filter(x, y):
    kf.predict()
    kf.update(np.array([[x], [y]]))
    kalman_x, kalman_y = kf.x.flatten().tolist()
    return kalman_x, kalman_y

pixel2meter = 0.2