import os
import subprocess

def convert_wav_to_ogg(input_wav_path: str, output_ogg_path: str, quality: int = 5) -> None:
    """
    Преобразует WAV-файл в OGG-файл с сохранением качества.

    Args:
        input_wav_path: Путь к входному WAV-файлу.
        output_ogg_path: Путь к выходному OGG-файлу.
        quality: Качество OGG-файла (от 0 до 10, где 0 - наихудшее, а 10 - наилучшее).
    """

    # Проверьте, существует ли входной WAV-файл.
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"Input WAV file '{input_wav_path}' not found.")

    # Проверьте, является ли качество допустимым значением.
    if quality < 0 or quality > 10:
        raise ValueError("Quality must be between 0 and 10.")

    # Выполните команду ffmpeg для преобразования WAV-файла в OGG-файл.
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_wav_path,
            "-c:a",
            "libvorbis",
            "-q:a",
            str(quality),
            output_ogg_path,
        ],
        check=True,
    )

convert_wav_to_ogg("0c7ee2f884ee8f4150820879cc8d4c.wav", "test.ogg", 5)