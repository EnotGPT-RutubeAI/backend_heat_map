from typing import List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel
from starlette.responses import JSONResponse
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import pipeline
from fastapi import FastAPI
import math
import time
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import io
import soundfile as sf
from urllib.request import urlopen
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

LABELS = ['neutral', 'joy', 'sadness', 'anger', 'enthusiasm', 'surprise', 'disgust', 'fear', 'guilt', 'shame']
LABELS_RU = ['нейтрально', 'радость', 'грусть', 'гнев', 'интерес', 'удивление', 'отвращение', 'страх', 'вина', 'стыд']

model = BertForSequenceClassification.from_pretrained('Djacon/rubert-tiny2-russian-emotion-detection')
tokenizer = AutoTokenizer.from_pretrained('Djacon/rubert-tiny2-russian-emotion-detection')

load_dotenv()


class Determinant:
    def __init__(self, text: str):
        self.text = text
        self.result = None
        self.parameters = (
            'neutral', 'joy', 'sadness', 'anger', 'enthusiasm', 'surprise', 'disgust', 'fear', 'guilt', 'shame')

    @torch.no_grad()
    def predict_emotions(self, text: str, labels: list = LABELS) -> dict:
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        pred = torch.nn.functional.sigmoid(outputs.logits)
        emotions_list = {}
        for i in range(len(pred[0].tolist())):
            emotions_list[labels[i]] = round(pred[0].tolist()[i], 3)
        self.result = emotions_list
        return emotions_list

    @torch.no_grad()
    def predict_emotion(self, text: str, labels: list = LABELS) -> str:
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        pred = torch.nn.functional.sigmoid(outputs.logits)
        pred = pred.argmax(dim=1)
        self.result = labels[pred[0]].title()
        return labels[pred[0]].title()

    def get_result_by_parameter(self, parameter: str):
        if type(self.result) == type(dict()):
            return self.result[parameter]
        return self.result


def find_most_frequent_emotion(emotions: list) -> str:
    emotion_counts = Counter(emotions)

    priorities = {
        "neutral": 1,
        "joy": 10,
        "sadness": 7,
        "anger": 9,
        "enthusiasm": 3,
        "surprise": 2,
        "disgust": 5,  # Отвращение
        "fear": 6,  # Страх
        "shame": 8,  # стыд
        "guilt": 4  # вина
    }

    most_frequent_emotion = None
    max_count = 0
    max_priority = 0

    for emotion, count in emotion_counts.items():
        if count > max_count or (count == max_count and priorities[emotion] > max_priority):
            most_frequent_emotion = emotion
            max_count = count
            max_priority = priorities[emotion]

    return most_frequent_emotion


class Modelivana:
    def __init__(self, segments: list):
        self.segments = segments
        self.map = []

    def plot_heatmap(self):
        data_array = np.array(self.map)
        data_array = data_array.reshape(1, -1)
        plt.imshow(data_array, cmap="hot", interpolation="nearest")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


class AudioEmotionText(Modelivana):
    def __init__(self, segments: list, parameter: str = None):
        super().__init__(segments)
        self.__emotions = (
            "neutral", "joy", "sadness", "anger", "enthusiasm", "surprise", "disgust", "fear", "shame", "guilt", None)
        if parameter not in self.__emotions:
            raise ValueError(f"Неверный аргумент. Возможные: {str(self.__emotions)}")
        self.parameter = parameter

    def start_with_param(self):
        for segment in self.segments:
            determinant = Determinant(segment)
            determinant.predict_emotions(segment)
            result = determinant.get_result_by_parameter(self.parameter)
            self.map.append(result)
        return self.map

    def start(self):
        if self.parameter is None:
            return self.parameter_is_none()
        return self.start_with_param()

    def parameter_is_none(self):
        moments = []
        text = ""
        for segment in self.segments:
            # determinant = Determinant(segment)
            # moments.append(determinant.predict_emotion(segment).lower())
            text += segment.lower()
        print(text)
        # print(moments)
        determinant = Determinant(text)
        self.parameter = determinant.predict_emotion(text).lower()
        # self.parameter = find_most_frequent_emotion(moments)
        return self.start_with_param()

    class AudioEmotionSchema(BaseModel):
        data: List[str]
        parameter: Optional[str] = None


def merge_text(segments: list):
    text = ""
    for segment in segments:
        text += segment
    return text


def round_timestamps(data):
    """Округляет временные метки в данных до целых чисел.

    Args:
      data: Словарь с данными, содержащий поле 'chunks' со списком словарей.

    Returns:
      Новый список списков с округленными временными метками.
    """

    new_timestamps = []
    for chunk in data.get('chunks'):
        if None in chunk['timestamp']:
            rounded_chunk = [int(chunk['timestamp'][0]),
                             int(chunk['timestamp'][0]) + 1]
        else:
            rounded_chunk = [int(math.ceil(timestamp))
                             for timestamp in chunk['timestamp']]
        new_timestamps.append(rounded_chunk)
    return np.array(new_timestamps)


def audio_to_loudness(audio_file, sr=22050):
    """
    Преобразует аудиофайл в массив NumPy с уровнями громкости в каждую секунду.

    Args:
      audio_file: Путь к аудиофайлу.
      sr: Частота дискретизации (по умолчанию 22050 Гц).

    Returns:
      Массив NumPy с уровнями громкости в каждой секунде.
    """

    # y, sr = sf.read(io.BytesIO(urlopen(url).read()))

    # Загрузка аудиофайла
    if audio_file.startswith('http://') or audio_file.startswith('https://'):
        # Загрузка с использованием urllib.request
        with io.BytesIO(urlopen(audio_file).read()) as f:
            y, sr = librosa.load(f, sr=sr)
    else:
        # Загрузка с использованием librosa.load
        y, sr = librosa.load(audio_file, sr=sr)

    # Длительность аудио в секундах
    duration = librosa.get_duration(y=y, sr=sr)

    # Разбиение аудио на 1-секундные фрагменты
    hop_length = sr
    n_fft = hop_length * 4
    C = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Вычисление средних значений MFCC для каждого фрагмента
    mean_mfcc = np.mean(C, axis=0)

    # Нормализация значений MFCC (для получения процентов)
    normalized_mfcc = (mean_mfcc - np.min(mean_mfcc)) / \
                      (np.max(mean_mfcc) - np.min(mean_mfcc)) * 100

    median_volume = np.median(normalized_mfcc)

    data = []

    for volume in normalized_mfcc:
        if (volume - median_volume) > 0:
            data.append((volume - median_volume) / median_volume)
        else:
            data.append(0)

    return np.array(data)


import os
import requests


def download_audio(url):
    """
  Функция получает аудио по ссылке, скачивает его в папку os.getenv(TEMP_FILES)
  и возвращает путь до этого файла.

  Args:
    url: URL-адрес аудиофайла.

  Returns:
    Путь до скачанного аудиофайла.
  """
    filename = os.path.basename(url)
    temp_files_dir = os.getenv("TEMP_FILES")
    file_path = os.path.join(temp_files_dir, filename)
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path


def audio_to_loudness_by_segments(audio_file, segments_time):
    segment_indices = []

    for start, end in segments_time:
        segment_indices.append(
            np.mean(audio_to_loudness(audio_file)[start:end]))

    return np.array(segment_indices)


data = [" Я задолбался делать это дебильное, я его ненавижу, это просто",
        " меня раздражает, я не могу.",
        " Я не понимаю, что происходит, я не устал.",
        " Ну не все страшно, главное, что мы победители",
        " чемпионата России и в этом есть какой-то праздник, какой-то",
        " свой экшен.",
        " Я так считаю."]

from fastapi import FastAPI


def get_segment_avg(timestamp: list, volumes: list):
    """
    Получаем среднее на сегменте от timestamp[0] до timestamp[1]
    :param timestamp:
    :param volumes:
    :return:
    """
    volumes = volumes[timestamp[0]:timestamp[1] + 1]
    if len(volumes) == 0:
        return 0
    avg = sum(volumes) / len(volumes) # деление на 0
    print(f"Среднее на сегменте {timestamp[0]}:{timestamp[1]} = {avg}")
    return avg


async def get_deviations(audio_url: str, data: dict):
    """
    получаем отклонения
    :param audio_url:
    :param segments:
    :return:
    """
    audio_data = audio_url
    audio_path = os.getenv("UPLOADS") + "/audios/" + extract_filename(audio_url)
    print("Отклонения пошли: ", audio_data)
    y, sr = librosa.load(audio_path)

    hop_length = sr
    avg_values = []
    for i in range(0, len(y), hop_length):
        avg_value = np.mean(y[i:i + hop_length])
        avg_values.append(avg_value)

    # Выведите массив усредненных значений
    print(avg_values)
    avg_volume = sum(avg_values) / len(avg_values)
    print(avg_volume)
    segments = data['chunks']
    result = list()
    for chunk in segments:
        timestamp: list = chunk['timestamp']
        timestamp[0], timestamp[1] = int(timestamp[0]), int(timestamp[1])
        segment_avg = get_segment_avg(timestamp, avg_values)
        result.append((segment_avg - avg_volume) / avg_volume)
    print(result)
    return result


async def analyze_video(video_path, frame_count, data: dict):
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open the video file.")
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the number of frames is valid
    if frame_count > length:
        print("The number of frames is too large for this video")
        cap.release()
        return

    results = []
    min_dr = min_dg = min_db = float('inf')
    max_dr = max_dg = max_db = 0

    previous_r_mean = previous_g_mean = previous_b_mean = None

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame
        if current_frame % frame_count == 0 or current_frame == 0:
            bgr = cv2.split(frame)
            r_values, g_values, b_values = bgr

            r_mean = np.mean(r_values)
            g_mean = np.mean(g_values)
            b_mean = np.mean(b_values)

            if previous_r_mean is not None:
                dr = abs(r_mean - previous_r_mean)
                dg = abs(g_mean - previous_g_mean)
                db = abs(b_mean - previous_b_mean)

                min_dr = min(min_dr, dr)
                min_dg = min(min_dg, dg)
                min_db = min(min_db, db)
                max_dr = max(max_dr, dr)
                max_dg = max(max_dg, dg)
                max_db = max(max_db, db)

                # Calculate avg_d
                avg_d = (dr + dg + db) / 3

                # Format timestamp
                seconds = int(current_frame / fps)
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60
                timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                results.append({
                    "timestamp": timestamp,
                    "dr": dr,
                    "dg": dg,
                    "db": db,
                    "avg_d": avg_d
                })

            previous_r_mean = r_mean
            previous_g_mean = g_mean
            previous_b_mean = b_mean

        current_frame += 1

    cap.release()

    # Normalize the deviations
    if results:
        for result in results:
            result["dr"] = (result["dr"] - min_dr) / (max_dr - min_dr) if max_dr - min_dr > 0 else result["dr"]
            result["dg"] = (result["dg"] - min_dg) / (max_dg - min_dg) if max_dg - min_dg > 0 else result["dg"]
            result["db"] = (result["db"] - min_db) / (max_db - min_db) if max_db - min_db > 0 else result["db"]
            result["avg_d"] = (result["avg_d"] - (min_dr + min_dg + min_db) / 3) / ((max_dr + max_dg + max_db) / 3 - (min_dr + min_dg + min_db) / 3) if (max_dr + max_dg + max_db) / 3 - (min_dr + min_dg + min_db) / 3 > 0 else result["avg_d"]

    segments = data['chunks']
    finish_results = []
    for result in results:
        finish_results.append(result['avg_d'])

    print(finish_results)
    avg_segments = []
    result_segments = []
    for chunk in segments:
        timestamp: list = chunk['timestamp']
        timestamp[0], timestamp[1] = int(timestamp[0]), int(timestamp[1])
        avg_segments = get_segment_avg(timestamp, finish_results)
        result_segments.append(avg_segments)
    return result_segments


def extract_filename(url) -> str:
    """
    Обрезает из ссылки в хеш
    :param url:
    :return:
    """
    last_slash_index = url.rfind('/')
    filename = url[last_slash_index + 1:]
    return filename


if __name__ == "__main__":
    app = FastAPI()


    @app.post("/audio/emotions")
    async def start(data: AudioEmotionText.AudioEmotionSchema):
        try:
            emotions = AudioEmotionText(data.data, data.parameter)
            emotions.start()
            emotion_all_text = AudioEmotionText([merge_text(data.data)])
            emotion_all_text.start()
        except ValueError as e:
            return JSONResponse(content={"message": e.__str__()}, status_code=400)
        return JSONResponse(content={"map": emotions.map, "parameter": emotions.parameter,
                                     "emotion_all_text": emotion_all_text.parameter})


    class VolumeData(BaseModel):
        data: dict
        audio_url: str


    @app.post("/audio/volume")
    async def modelivana_volume(data: VolumeData):
        url = download_audio(data.audio_url)
        result = await get_deviations(url, data.data)
        return JSONResponse(content={"status": True, "deviations": result})


    class VideoData(BaseModel):
        video_url: str
        data: dict


    @app.post("/video")
    async def modelivana_video(data: VideoData):
        url = data.video_url
        hash = extract_filename(url)
        video_path = os.getenv("UPLOADS") + "videos/" + hash
        return JSONResponse(content={"array":await analyze_video(video_path, 30, data.data)})



    class Data(BaseModel):
        audio_url: str


    # @app.post("/text_recognition")
    # async def text_recognition(data: Data):
    #     result = pipe(data.audio_url, generate_kwargs={"language": "russian"}, return_timestamps=True)
    #     return JSONResponse({"result": result})

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8005)
