import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


class LocationPredictor:
    def __init__(self):
        cameraParams = {"IntrinsicMatrix": np.array([[1.4133e+03, 0, 950.0639], [0, 1.4188e+03, 543.3796], [0, 0, 1]]), "RadialDistortion": np.array([-0.0091, 0.0666, 0, 0])}
        self.cameraMatrix = cameraParams["IntrinsicMatrix"]
        self.distCoeffs = cameraParams["RadialDistortion"].reshape(1, 4)
        self.orb = cv2.ORB_create()
        dt = 1.0
        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        self.kf.F = np.array([[1, dt], [0, 1]])
        self.kf.H = np.array([[1, 0], [0, 1]])
        self.kf.Q = np.array([[1, 0], [0, 1]])
        self.kf.R = np.array([[10, 0], [0, 10]])
        self.kf.x = np.array([[0], [0]])
        self.kf.P = np.array([[1, 0], [0, 1]])
        
        self.pixel2meter = 0.001
        
    def pixel2world(self, x, y):
        # return np.dot(np.linalg.inv(self.cameraMatrix), np.array([[x], [y], [1]]))
        return np.array([[x * self.pixel2meter, y * self.pixel2meter]])
        
    def predict(self, current_frame, last_frame, health_status, prediction):
        # --> Veriler Sağlıklı İse <--
        if health_status == 1:
            return prediction.gt_translation_x, prediction.gt_translation_y
        
        # --> Veriler Sağlıksız İse <--
        if last_frame is None: # İlk frame'de hareket yok
            return 0, 0
        
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.undistort(current_frame_gray, self.cameraMatrix, self.distCoeffs)
        last_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        last_frame_gray = cv2.undistort(last_frame_gray, self.cameraMatrix, self.distCoeffs)
        keypoints1, descriptors1 = self.orb.detectAndCompute(current_frame_gray, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(last_frame_gray, None)
        if descriptors1 is None or descriptors2 is None: return 0, 0 # Eşleşme yok
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
        # Kalman Filtresi
        self.kf.predict()
        self.kf.update(np.array([[dx], [dy]]))
        predicted_x, predicted_y = self.kf.x.flatten().tolist()
        
        world_translation = self.pixel2world(predicted_x, predicted_y)
        world_x = world_translation[0, 0]
        world_y = world_translation[0, 1]
        
        return world_x, world_y
