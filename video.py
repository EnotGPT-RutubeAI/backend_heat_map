import cv2
import mediapipe as mp
from fer import FER
import pandas as pd

# Инициализация
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER()

# Открытие видео
video_path = 'your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Проверка открытия видео
if not cap.isOpened():
    print("Error opening video file")

# Список для хранения результатов
results = []

# Обработка видео
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Получение текущего времени в видео
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Преобразование изображения в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обнаружение лиц
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                # Получение координат лица
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                int(bboxC.width * iw), int(bboxC.height * ih))

                # Извлечение лица
                face = image[y:y + h, x:x + w]

                # Определение эмоций
                emotion, score = emotion_detector.top_emotion(face)

                # Сохранение результатов
                results.append({
                    'time': current_time,
                    'emotion': emotion,
                    'score': score
                })

# Закрытие видео
cap.release()

# Сохранение результатов в CSV
df = pd.DataFrame(results)
df.to_csv('emotion_results.csv', index=False)

print("Обработка завершена. Результаты сохранены в 'emotion_results.csv'")


