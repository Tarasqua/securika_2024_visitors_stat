"""
@tarasqua

Запуск демонстрации.
"""

import asyncio
import time
from typing import List
from collections import deque
from pathlib import Path

import screeninfo
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics.utils import LOGGER

from utils.util import plot_skeletons, set_yolo_model, UltralyticsLogsFilter
from visitors_statistics import VisitorsStatistics
from utils.roi_polygon import ROIPolygonSelector
from utils.config_loader import Config

LOGGER.addFilter(UltralyticsLogsFilter())  # фильтруем забагованные логи

MAX_HEIGHT = 478  # максимальная допустимая высота столбца в пикселях
COLUMN_WIDTH = 100  # ширина столбца в пикселях
BOTTOM_LEFT_POINTS = [(808, 1265), (808, 1415), (808, 1565), (808, 1715)]  # нижние левые точки столбцов гистограммы


class Main:

    def __init__(self, font_path: str, bg_image_path: str, ):
        self.config_ = Config()
        self.config_.initialize('main')
        self.background_img = cv2.imread(bg_image_path)
        self.time_font = ImageFont.truetype(font_path, 12)  # временные интервалы
        self.count_font = ImageFont.truetype(font_path, 24)  # количество людей
        self.statistical_data: (
            deque)[List[int | float]] = deque([], maxlen=4)  # данные по посетителям для построения гистограммы
        self.visitors_statistics = VisitorsStatistics()
        self.fullscreen_mode = False  # полноэкранный режим для стрима

    def click_event(self, event, x, y, flags, params) -> None:
        """
        Callback на кликер для определения координат, куда было совершено нажатие.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.fullscreen_mode:
                if 263 <= y <= 896 and 25 <= x <= 1145:
                    self.fullscreen_mode = True
            else:
                self.fullscreen_mode = False

    def plot_hist(self, frame: np.array) -> np.array:
        """
        Отрисовка гистограммы.
        :param frame: Кадр для отрисовки.
        :return: Кадр с отрисованной гистограммой на ней.
        """
        if not self.statistical_data:
            return frame
        # масштабируем высоты столбцов гистограммы
        col_heights = (
                (data_table := np.array(self.statistical_data))[:, 0] /  # делим на большее значение с проверкой на 0
                (m if (m := max(data_table[:, 0])) != 0 else 1) * MAX_HEIGHT  # и умножаем на допустимую высоту
        ).astype(int)
        for height, (y, x) in zip(col_heights, BOTTOM_LEFT_POINTS):
            cv2.rectangle(frame, (x, y - height), (x + COLUMN_WIDTH, y), (255, 255, 255), -1)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        for height, (count, start, end), (y, x) in zip(col_heights, self.statistical_data, BOTTOM_LEFT_POINTS):
            draw.text((x - 5, y + 10), time.strftime('%H:%M', time.localtime(start)),
                      font=self.time_font, fill=(255, 255, 255, 0))
            draw.text((x - 15 + COLUMN_WIDTH, y + 10), time.strftime('%H:%M', time.localtime(end)),
                      font=self.time_font, fill=(255, 255, 255, 0))
            draw.text((int(x + COLUMN_WIDTH / 2) - 12, y - height - 28), str(count),
                      font=self.count_font, fill=(255, 255, 255, 0))
        return np.array(img_pil)

    async def update_statistical_data(self) -> None:
        """
        Обновление статистических данных.
        :return: None.
        """
        while True:
            await asyncio.sleep(int(self.config_.get('UPDATE_STATISTICS_DELAY') * 60))
            self.statistical_data.append(self.visitors_statistics.reset_statistics())

    async def main(self, stream_source):
        """
        Запуск сборщика статистики.
        :param stream_source:
        :return:
        """
        cap = cv2.VideoCapture(stream_source)
        _, frame = cap.read()
        reshape_main_frame = tuple((np.array(frame.shape[:-1][::-1]) / 2.4).astype(int))
        reshape_full_screen = tuple((np.array(self.background_img.shape[:-1][::-1])))
        roi = (ROIPolygonSelector().get_roi(  # выделяем зону, в которой будем считать людей
            cv2.resize(frame, reshape_main_frame)  # для удобства выделения, урезаем картинку
        ) * 2.4).astype(int)  # и переводим готовый полигон в исходные координаты

        screen = screeninfo.get_monitors()[0]
        cv2.namedWindow('main', cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('main', screen.x - 1, screen.y - 1)
        cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('main', self.click_event)

        self.visitors_statistics.set_roi(roi)  # выставляем его
        yolo_detector = set_yolo_model('n', 'pose', 'pose')
        update_stat_task = asyncio.create_task(self.update_statistical_data())
        self.visitors_statistics.reset_statistics()  # сбрасываем статистику
        for detections in yolo_detector.track(
                stream_source, classes=[0], stream=True, conf=self.config_.get('YOLO_CONF'), verbose=False
        ):
            self.visitors_statistics.update_visitors(detections)
            frame = await plot_skeletons(detections.orig_img, detections)
            frame = cv2.polylines(frame, [roi], True, (128, 0, 64), 8)
            if self.fullscreen_mode:  # полноэкраснный режим
                cv2.imshow('main', cv2.resize(frame, reshape_full_screen))
            else:
                show_frame = self.background_img.copy()
                show_frame[263:reshape_main_frame[1] + 263, 25:reshape_main_frame[0] + 25] = (
                    cv2.resize(frame, reshape_main_frame))
                await asyncio.sleep(0)  # освобождаем поток, чтобы статистические данные успели обновиться
                cv2.imshow('main', self.plot_hist(show_frame))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        update_stat_task.cancel()


if __name__ == '__main__':
    main = Main((Path.cwd().parents[0] / 'resources' / 'fonts' / 'Gilroy-Regular.ttf').as_posix(),
                (Path.cwd().parents[0] / 'resources' / 'images' / 'background.png').as_posix())
    asyncio.run(main.main('rtsp://admin:Qwer123@192.168.0.108?subtype=1'))
