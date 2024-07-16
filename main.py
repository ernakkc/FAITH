import logging
from datetime import datetime
from pathlib import Path
from decouple import config
from tqdm import tqdm
from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '%Y%m_%d_%H%M_%S_%f.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info(f"Logging configured. Log file: {log_filename}")



def run():
    print("Started...")
    # .env dosyasından konfigurasyon icin gereken bilgileri cek
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Logger'a ilgili konfigurasyon atamasini yap
    configure_logger(team_name)

    # Takimlar kendi kodlarini ObjectDetectionModel sinifi icerisine implemente edebilirler. (OPSIYONEL)
    detection_model = ObjectDetectionModel(evaluation_server_url)
    logging.info("ObjectDetectionModel initialized.")
    
    # Sunucuya baglan
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)
    logging.info("Connected to server.")

    # Mevcutta aktif olan oturumdan tum frameleri cek.
    frames_json = server.get_frames()
    logging.info(f"Retrieved {len(frames_json)} frames.")

    # Mevcutta aktif olan oturumdan tum translation verilerini cek.
    translations_json = server.get_translations()  
    logging.info(f"Retrieved {len(translations_json)} translations.")

    # Klasorun ve dosyanin yollarini cekelim
    images_files, images_folder = server.get_listdir()

    # Nesne tespiti modelini frame frame olacak sekilde calistir
    for frame, translation in tqdm(zip(frames_json, translations_json), total=len(frames_json)):
        # Prediction objesini frame ve translation bilgilerini tutacak sekilde olustur
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'],
                                       translation['translation_x'], translation['translation_y'])
        # Healt status kontrolu ikinci gorevde sistemin ne zaman devreye girmesi gerektigini gosterir
        health_status = translation['health_status']
        # Tespit modelini calistir
        predictions = detection_model.process(predictions,evaluation_server_url, health_status, images_folder, images_files)
        # Modelin o frame'e ait tahmin degerlerini sunucuya gonder
        result = server.send_prediction(predictions)
        # çizgi çizip ortasına FRAME ID yazdırıyoruz
        logging.info(f"Prediction sent.".center(85, "-"))

if __name__ == '__main__':
    run()
