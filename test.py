# import torch
# from transformers import BertForSequenceClassification, AutoTokenizer
#
# LABELS = ['neutral', 'joy', 'sadness', 'anger', 'enthusiasm', 'surprise', 'disgust', 'fear', 'guilt', 'shame']
# LABELS_RU = ['нейтрально', 'радость', 'грусть', 'гнев', 'интерес', 'удивление', 'отвращение', 'страх', 'вина', 'стыд']
#
# model = BertForSequenceClassification.from_pretrained('Djacon/rubert-tiny2-russian-emotion-detection')
# tokenizer = AutoTokenizer.from_pretrained('Djacon/rubert-tiny2-russian-emotion-detection')
#
#
# # Predicting emotion in text
# # Example: predict_emotion("Сегодня такой замечательный день!") -> Joy
# @torch.no_grad()
# def predict_emotion(text: str, labels: list = LABELS) -> str:
#     inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
#     inputs = inputs.to(model.device)
#
#     outputs = model(**inputs)
#
#     pred = torch.nn.functional.sigmoid(outputs.logits)
#     pred = pred.argmax(dim=1)
#
#     return labels[pred[0]].title()
#
#
# # Probabilistic prediction of emotion in a text
# # Example: predict_emotions("Сегодня такой замечательный день!") ->
# # -> {'neutral': 0.229, 'joy': 0.873, 'sadness': 0.045,...}
#
#
#
# def main():
#     try:
#         while True:
#             text = input('Enter Text (`q` for quit): ')
#             if not text:
#                 continue
#             elif text == 'q':
#                 return print('Bye 👋')
#             print('Your emotion is:', predict_emotions(text))
#     except KeyboardInterrupt:
#         print('\nBye 👋')
#
#
import os

import librosa
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

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


def get_audio_loudness_per_second_with_intonation_detection(audio_url):
  """
  Функция получает ссылку на аудио, скачивает его в os.getenv(TEMP_FILES) и выводит массив,
  где посекундно будет отображена громкость, с учетом интереса к отлову повышения интонаций.

  Args:
    audio_url: URL-адрес аудиофайла.

  Returns:
    Список значений громкости на каждой секунде аудио, с учетом повышения интонаций.
  """

  # Скачиваем аудиофайл
  audio_path = download_audio(audio_url)

  # Загружаем аудиофайл
  y, sr = librosa.load(audio_path)

  # Вычисляем громкость на каждой секунде
  loudness = librosa.feature.rms(y=y, frame_length=sr, hop_length=sr)

  # Вычисляем повышение интонаций
  pitch = librosa.feature.pitch(y=y, sr=sr)
  pitch_diff = np.diff(pitch)

  # Определяем повышение интонаций как значения pitch_diff, превышающие заданный порог
  intonation_threshold = 0.5
  intonation_flags = pitch_diff > intonation_threshold

  # Объединяем громкость и повышение интонаций
  loudness_with_intonation = []
  for i in range(len(loudness)):
    if intonation_flags[i]:
      loudness_with_intonation.append(loudness[i] * 1.5)  # Увеличиваем громкость для повышения интонаций
    else:
      loudness_with_intonation.append(loudness[i])

  # Возвращаем список значений громкости
  return loudness_with_intonation


def get_segment_avg(timestamp: list, volumes: list):
    """
    Получаем среднее на сегменте от timestamp[0] до timestamp[1]
    :param timestamp:
    :param volumes:
    :return:
    """
    volumes = volumes[timestamp[0]:timestamp[1]+1]
    avg = sum(volumes) / len(volumes)
    print(f"Среднее на сегменте {timestamp[0]}:{timestamp[1]} = {avg}")
    return avg


def get_deviations(audio_url: str, data: dict):
    """
    получаем отклонения
    :param audio_url:
    :param segments:
    :return:
    """
    audio_data = audio_url

    y, sr = librosa.load(audio_data)

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
    segments_avg = list()
    result = list()
    for chunk in segments:
        timestamp: list = chunk['timestamp']
        timestamp[0], timestamp[1] = int(timestamp[0]), int(timestamp[1])
        segment_avg = get_segment_avg(timestamp, avg_values)
        result.append((segment_avg - avg_volume) / avg_volume)
    print(result)
    return result


# [0, 5, 2, 1, 10, 0], среднее = 18/6 = 3
# [0, 5], среднее = 2.5, 2.5-3=-0.5, -0.5/3=-0.16
# [2, 1], среднее = 1.5, 1.5-3=-1.5, -1.5/3=-0.5
# [10, 0], среднее = 5, 5-3=2, 2/3=0.66


# [-0.16, -0.5, 0.66]



if __name__ == '__main__':
    url = os.getenv("TEMP_FILES") + "d4565c2e238ee6679befc0a7eb349e.mp3"
    import librosa
    import librosa.display
    import numpy as np
    get_deviations(url, {})
