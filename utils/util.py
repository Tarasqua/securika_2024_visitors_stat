import logging
import asyncio
import os
from pathlib import Path
from fnmatch import fnmatch

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

KEY_POINTS = [  # наименования ключевых точек для YOLO по порядку
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
    'right_ankle'
]
LIMBS = (  # конечностей, заключенные между ключевых точек
    (KEY_POINTS.index('right_eye'), KEY_POINTS.index('nose')),
    (KEY_POINTS.index('right_eye'), KEY_POINTS.index('right_ear')),
    (KEY_POINTS.index('left_eye'), KEY_POINTS.index('nose')),
    (KEY_POINTS.index('left_eye'), KEY_POINTS.index('left_ear')),
    (KEY_POINTS.index('right_shoulder'), KEY_POINTS.index('right_elbow')),
    (KEY_POINTS.index('right_elbow'), KEY_POINTS.index('right_wrist')),
    (KEY_POINTS.index('left_shoulder'), KEY_POINTS.index('left_elbow')),
    (KEY_POINTS.index('left_elbow'), KEY_POINTS.index('left_wrist')),
    (KEY_POINTS.index('right_hip'), KEY_POINTS.index('right_knee')),
    (KEY_POINTS.index('right_knee'), KEY_POINTS.index('right_ankle')),
    (KEY_POINTS.index('left_hip'), KEY_POINTS.index('left_knee')),
    (KEY_POINTS.index('left_knee'), KEY_POINTS.index('left_ankle')),
    (KEY_POINTS.index('right_shoulder'), KEY_POINTS.index('left_shoulder')),
    (KEY_POINTS.index('right_hip'), KEY_POINTS.index('left_hip')),
    (KEY_POINTS.index('right_shoulder'), KEY_POINTS.index('right_hip')),
    (KEY_POINTS.index('left_shoulder'), KEY_POINTS.index('left_hip'))
)
PALETTE = np.array([  # цветовая палитра для ключевых точек и конечностей
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255], [153, 204, 255], [255, 102, 255],
    [255, 51, 255], [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153],
    [102, 255, 102], [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]])
LIMBS_COLORS = PALETTE[[16, 16, 16, 16, 9, 9, 9, 9, 0, 0, 0, 0, 7, 7, 7, 7]]  # цвета для конечностей
KPTS_COLORS = PALETTE[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]  # цвета для ключевых точек


def find_file(pattern: str, path: str) -> str:
    """
    Поиск файла по паттерну (первое вхождение).
    :param pattern: Паттерн для поиска.
    :param path: Корневой путь для поиска.
    :return: Путь из корневой папки до найденного файла (если файл не найден, возвращается пустая строка).
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                return os.path.join(root, name)
    return ''


def set_yolo_model(yolo_model: str, yolo_class: str, task: str = 'detect') -> YOLO:
    """
    Выполняет проверку путей и наличие модели:
        Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель
    :param yolo_model: n (nano), m (medium), etc.
    :param yolo_class: seg, pose, boxes
    :param task: detect, segment, classify, pose
    :return: Объект YOLO-pose
    """
    yolo_class = f'-{yolo_class}' if yolo_class != 'boxes' else ''
    yolo_models_path = Path.cwd().parents[1] / 'resources' / 'models' / 'yolo_models'
    if not os.path.exists(yolo_models_path):
        Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}{yolo_class}')
    if not os.path.exists(f'{model_path}.onnx'):
        YOLO(model_path).export(format='onnx')
    return YOLO(f'{model_path}.onnx', task=task, verbose=False)


async def plot_skeletons(frame: np.array, detections: Results, conf: float = 0.5) -> np.array:
    """
    Отрисовка скелетов людей в текущем кадре.
    :param frame: Кадр для отрисовки.
    :param detections: YOLO detections.
    :param conf: Порог по confidence.
    :return: Кадр c отрисованными скелетами людей.
    """

    async def plot_kpts(points: np.array) -> None:
        """
        Отрисовка ключевых точек человека на кадре.
        :param points: Пронумерованные координаты точек в формате [[i, x, y], [...]].
        :return: None.
        """
        circle_tasks = [asyncio.to_thread(  # отрисовываем точки
            cv2.circle, frame, (x, y), 8, tuple(map(int, KPTS_COLORS[i])), -1, 8, 0)
            for i, x, y in points.astype(int)]
        await asyncio.gather(*circle_tasks)

    async def plot_limbs(points: np.array) -> None:
        """
        Отрисовка конечностей человека на кадре.
        :param points: Пронумерованные координаты точек в формате [[i, x, y], [...]].
        :return: None.
        """
        # берем только те конечности, точки которых прошли фильтрацию по conf
        filtered_limbs = [limb for limb in LIMBS if np.all(np.in1d(limb, points[:, 0]))]
        limbs_tasks = [asyncio.to_thread(  # отрисовываем конечности
            cv2.line, frame,
            points[:, 1:][points[:, 0] == p1].astype(int)[0], points[:, 1:][points[:, 0] == p2].astype(int)[0],
            tuple(map(int, LIMBS_COLORS[i])), 4, 8, 0
        ) for i, (p1, p2) in enumerate(filtered_limbs)]
        await asyncio.gather(*limbs_tasks)

    if len(detections) == 0:
        return frame
    # номеруем и фильтруем по conf ключевые точки (нумерация нужна, чтобы после фильтрации не потерять порядок)
    people_kpts = [(points := np.c_[np.arange(17), kpt])[points[:, 3] >= conf][:, :-1]  # берем только i и точки
                   for kpt in detections.keypoints.data.numpy()]
    for kpts in people_kpts:  # отрисовываем по каждому человеку
        await asyncio.gather(plot_kpts(kpts), plot_limbs(kpts))
    return frame


class UltralyticsLogsFilter(logging.Filter):
    """
    Фильтрация для забагованных логов, приходящих из ultralytics.data.loaders LoadStreams.
    """
    def filter(self, record):
        """Фильтруем логи"""
        return not ('Waiting for stream' in record.getMessage())
