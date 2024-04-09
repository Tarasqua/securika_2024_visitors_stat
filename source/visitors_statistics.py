"""
@tarasqua

Подсчет статистики по людям, заходящим на стенд.
"""

import time
from typing import List

import cv2
from loguru import logger
import numpy as np
from ultralytics.engine.results import Results


class VisitorsStatistics:
    """
    Подсчет статистики людей на стенде за определенный промежуток времени.
    """

    def __init__(self):
        self.visitors_ids: np.array = np.array([])
        self.start_time: float = time.time()
        self.roi = np.array([])
        logger.success('VisitorsStatistics started successfully')

    def set_roi(self, roi: np.array) -> None:
        """
        Сеттер для ROI.
        :param roi: Полигон в формате [[x, y], [...], ...].
        :return: None.
        """
        self.roi = roi
        logger.success('ROI successfully set')

    def is_inside_roi(self, detection: Results) -> bool:
        """
        Проверка на то, что человек находится в области ROI по центроиду его ббокса.
        :param detection: YOLO detection.
        :return: Внутри или нет.
        """
        centroid = np.array([(bbox := detection.boxes.xyxy.numpy()[0])[:-2], bbox[-2:]]).sum(axis=0) / 2
        return cv2.pointPolygonTest(self.roi, centroid, False) >= 0

    def update_visitors(self, detections: Results) -> None:
        """
        Обновление данных по наблюдаемым людям.
        :param detections: YOLO detections.
        :return: None.
        """
        # находим id людей на текущем кадре, которые лежат в области ROI
        cur_frame_ids = np.array([
            id_.numpy()[0] for det in detections
            if (id_ := det.boxes.id) is not None and self.is_inside_roi(det)])
        self.visitors_ids = np.append(
            self.visitors_ids,  # добавляем к списку уже замеченных людей
            cur_frame_ids[~np.in1d(cur_frame_ids, self.visitors_ids)])  # тех, кого видим впервые

    def reset_statistics(self) -> List[int | float]:
        """
        Сброс накопления статистики и выдача текущих накопленных данных.
        :return: Список данных в формате:
            [количество пройденных людей за данный промежуток,
            время начала наблюдения, конца наблюдения (текущее время)].
        """
        logger.info(f'{self.visitors_ids.size} people were found in the time period from '
                    f'{time.strftime("%H:%M:%S", time.localtime(self.start_time))} to '
                    f'{time.strftime("%H:%M:%S", time.localtime(time.time()))}')
        summary_data = [self.visitors_ids.size, self.start_time, time.time()]
        self.visitors_ids = np.array([])
        self.start_time = time.time()
        return summary_data
