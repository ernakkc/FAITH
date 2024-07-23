import os
import cv2
import logging
import time
import random
import requests

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation

from utils.detection.object_detection import Detection
from utils.location.predict_location import LocationPredictor

class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.current_frame = None
        self.last_frame = None
        self.detection_model = Detection(model_path="data/best.pt")
        self.location_predictor = LocationPredictor()
        self.image_path = None
        
    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        # Indirmek istedigimiz frame frames.json dosyasinda mevcut mu kontrol edelim
        image_name = img_url.split("/")[-1]
        # Eger indirecegimiz frame'i daha once indirmediysek indirme islemine gecelim
        if image_name not in images_files:
            for attempt in range(retries):
                    try:
                        response = requests.get(img_url, timeout=60)
                        response.raise_for_status()
                        
                        img_bytes = response.content
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
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction,evaluation_server_url,health_status, images_folder ,images_files):
        self.download_image(evaluation_server_url + "media" + prediction.image_url, images_folder, images_files)
        current_frame_path = images_folder + prediction.image_url.split("/")[-1]
        if cv2.imread(current_frame_path) is None:
            logging.error(f"Frame not found in {current_frame_path}")
            return None
                
        self.current_frame = cv2.imread(current_frame_path)       
    
        frame_results = self.detect(prediction,health_status)
        return frame_results

    def detect(self, prediction,health_status):
        # -----------> Object Detection <-----------
        results = self.detection_model.prediction(self.current_frame)
        for result in results:
            cls = result["cls"]
            landing_status = result["landing_status"]
            top_left_x = result["top_left_x"]
            top_left_y = result["top_left_y"]
            bottom_right_x = result["bottom_right_x"]
            bottom_right_y = result["bottom_right_y"]
            
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y
                                        )
            prediction.add_detected_object(d_obj)


        # -----------> Translation <-----------



        # Health Status biti hava aracinin uydu haberlesmesinin saglikli olup olmadigini gosterir.
        # Health Status 0 ise sistem calismali 1 ise gelen verinin aynisini gondermelidir.

        if health_status == '0':
            x , y = self.location_predictor.predict(self.current_frame, self.last_frame, health_status, prediction)
            pred_translation_x, pred_translation_y = x, y
        else :
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y

        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y)
        prediction.add_translation_object(trans_obj)

        self.last_frame = self.current_frame
        
        return prediction