import cv2
import time
import random
import logging
import numpy as np
from typing import List

import pathlib
import platform

from sahi.utils.file import import_model_class
from sahi.utils.cv import read_image_as_pil
from sahi.prediction import ObjectPrediction
from sahi.prediction import PredictionResult


class Detection:
    def __init__(self, model_path):
        model_path = self.modelPath(model_path)
        self.detection_model = self.initModel(model_path)
        
    
    def modelPath(self, path):
        if platform.system() == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
        else: pass
        return path

    def initModel(self, model_path):
        model_type = "yolov5"
        model_class_name = "Yolov5DetectionModel"
        model_path= model_path
        confidence_threshold= 0.6
        device= "cuda:0"
        load_at_init= True
        
        DetectionModel = import_model_class(model_type, model_class_name)
        DetectionModel = DetectionModel(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            load_at_init=load_at_init
        )
        return DetectionModel
        
    def isLanding(self, class_id, frame, top_left_x, top_left_y, bottom_right_x, bottom_right_y,detected_objects,obj):
        if class_id == 2:
            roi = frame[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred_roi = cv2.GaussianBlur(gray_roi, (15, 15), 0)
                
            circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=40, minRadius=50, maxRadius=100)
            if circles is not None:
                status = "1"
            else:
                status = "0"
                
            if status != "0":
                for other_obj in detected_objects:
                    if other_obj == obj:
                        status = "1"
                        continue

                    other_box = other_obj['box']
                        
                    if (top_left_x <= other_box[0] <= bottom_right_x and top_left_y <= other_box[1] <= bottom_right_y) and \
                    (top_left_x <= other_box[2] <= bottom_right_x and top_left_y <= other_box[3] <= bottom_right_y):
                        status = "0"  

        elif class_id == 3:
            roi = frame[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred_roi = cv2.GaussianBlur(gray_roi, (15, 15), 0)
                
            circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=40, minRadius=50, maxRadius=100)
            if circles is not None:
                status = "1"
            else:
                status = "0"
                
            if status != "0":
                for other_obj in detected_objects:
                    if other_obj == obj:
                        status = "1"
                        continue

                    other_box = other_obj['box']
                        
                    if (top_left_x <= other_box[0] <= bottom_right_x and top_left_y <= other_box[1] <= bottom_right_y) and \
                    (top_left_x <= other_box[2] <= bottom_right_x and top_left_y <= other_box[3] <= bottom_right_y):
                        status = "0"  

        else:
            status = "-1"

        return status

    
    def prediction(self, frame):
        detection_model = self.detection_model
        shift_amount: list = [0, 0]
        full_shape = None
        postprocess = None
        verbose = 0
        
        durations_in_seconds = dict()
        image_as_pil = read_image_as_pil(frame)
        
        # Get prediction
        time_start = time.time()
        detection_model.perform_inference(np.ascontiguousarray(image_as_pil))
        time_end = time.time() - time_start
        durations_in_seconds["prediction"] = time_end
        
        # process prediction
        time_start = time.time()
        # works only with 1 batch
        detection_model.convert_original_predictions(
            shift_amount=shift_amount,
            full_shape=full_shape,
        )
        object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list
        
        # postprocess matching predictions
        if postprocess is not None:
            object_prediction_list = postprocess(object_prediction_list)
            
        time_end = time.time() - time_start
        durations_in_seconds["postprocess"] = time_end
        
        if verbose == 1:
            print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.")
        
        result = PredictionResult(image=frame, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds)   
        object_prediction_list = result.object_prediction_list
        detected_objects = []
        sended_objects = []
        
        for object_prediction in object_prediction_list:
            cls = object_prediction.category.id # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            tasit_classes = [1,2,3,4,5,8,9]
            if cls in tasit_classes: new_cls = 0   # Taşıt
            elif cls == 0:                new_cls = 1   # İnsan
            elif cls == 6:                new_cls = 2   # UAP
            elif cls == 7:                new_cls = 3   # UAI
            box = object_prediction.bbox.to_xyxy()

            detected_objects.append({
                "box": box,
                "class_id": new_cls
            })

        for obj in detected_objects:
            box = obj["box"]
            cls = obj["class_id"]
            top_left_x = box[0]
            top_left_y = box[1]
            bottom_right_x = box[2]
            bottom_right_y = box[3]

            landing_status = self.isLanding(cls,frame,top_left_x,top_left_y,bottom_right_x,bottom_right_y,detected_objects,obj)
            
            sended_objects.append({
                "cls": cls,
                "landing_status": landing_status,
                "top_left_x": top_left_x,
                "top_left_y": top_left_y,
                "bottom_right_x": bottom_right_x,
                "bottom_right_y": bottom_right_y
                                                        })
            
        return sended_objects
