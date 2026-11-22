"""
Модуль ocam.py — Драйвер / обёртка для камеры Withrobot oCam-1MGN-U
(1MP Global Shutter Monochrome USB 3.0 UVC Camera).

Особенности:
- Автоматически настраивает захват 'сырого' монохромного формата Y800 через MSMF на Windows
  (отключение HW-ускорения и RGB-конвертации).
- Авто-пересборка плоского буфера в 2D-массив (H, W).
- Поддерживает многопоточный захват (Threaded Capture) для нулевой задержки (zero input lag).
- Автоматический fallback на обычную вебкамеру, если oCam не подключена.
- Встроенные фильтры: программный Gain, CLAHE (адаптивное улучшение контраста), гамма-коррекция.
- Контекстный менеджер (with OCam() as cam:).
"""

import time
import threading
import cv2
import numpy as np


class OCam:
    """
    Класс для удобной работы с промышленной камерой Withrobot oCam-1MGN-U.
    
    Пример использования:
        with OCam(index=1, width=1280, height=960) as cam:
            while True:
                ret, gray = cam.read()
                if not ret:
                    break
                cv2.imshow("oCam", gray)
                if cv2.waitKey(1) == 27:
                    break
    """
    def __init__(self, index=1, width=1280, height=960, fps=30, threaded=True, auto_fallback=True):
        self.index = index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.threaded = threaded
        self.auto_fallback = auto_fallback

        self.cap = None
        self.is_ocam = False
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

        # Потоковые переменные
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._last_frame = None
        self._last_ret = False
        self._new_frame_event = threading.Event()

        # Статистика
        self._measured_fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()

        # CLAHE для программного улучшения контраста
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.open()

    def open(self):
        """Инициализирует и открывает камеру."""
        if self.cap is not None and self.cap.isOpened():
            self.release()

        # 1. Сначала пробуем открыть как oCam-1MGN-U через MSMF (Y800 raw capture)
        opened = False
        try:
            self.cap = cv2.VideoCapture()
            params = [
                cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE,
                cv2.CAP_PROP_CONVERT_RGB, 0,
            ]
            self.cap.open(self.index, cv2.CAP_MSMF, params)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                if self.target_fps > 0:
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                # Проверка чтения тестового кадра
                for _ in range(5):
                    ret, frame = self._read_raw()
                    if ret and frame is not None:
                        self.is_ocam = True
                        opened = True
                        self._last_frame = frame
                        self._last_ret = True
                        break
                    time.sleep(0.04)
        except Exception:
            opened = False

        # 2. Если oCam на указанном индексе не ответила, пробуем fallback
        if not opened and self.auto_fallback:
            # Пробуем проверить другие индексы (0, 1, 2)
            for fallback_idx in [1, 0, 2]:
                if fallback_idx == self.index:
                    continue
                try:
                    cap = cv2.VideoCapture()
                    params = [
                        cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE,
                        cv2.CAP_PROP_CONVERT_RGB, 0,
                    ]
                    cap.open(fallback_idx, cv2.CAP_MSMF, params)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                        ret, raw = cap.read()
                        if ret and raw is not None:
                            self.cap = cap
                            self.index = fallback_idx
                            self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            self.is_ocam = True
                            opened = True
                            break
                        cap.release()
                except Exception:
                    pass

        # 3. Если всё ещё не открылась — открываем стандартным бэкендом (вебкамера)
        if not opened and self.auto_fallback:
            print("[!] oCam не обнаружена в сыром Y800 режиме. Подключение стандартной веб-камеры...")
            for idx in [0, 1]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.index = idx
                        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        self.is_ocam = False
                        opened = True
                        break
                    cap.release()

        if not opened:
            raise RuntimeError(
                f"Не удалось открыть камеру (индекс {self.index}).\n"
                "Проверьте подключение USB 3.0 и убедитесь, что камера не занята другим приложением."
            )

        # Запуск фонового потока захвата
        if self.threaded:
            self._running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()

        return True

    def _read_raw(self):
        """Низкоуровневое считывание кадра."""
        ret, raw = self.cap.read()
        if not ret or raw is None:
            return False, None

        if self.is_ocam:
            expected_size = self.actual_width * self.actual_height
            actual_size = raw.size
            if actual_size == expected_size:
                frame = raw.reshape(self.actual_height, self.actual_width)
            elif actual_size > expected_size:
                frame = raw.flatten()[:expected_size].reshape(self.actual_height, self.actual_width)
            else:
                return False, None
        else:
            # Обычная вебкамера (может быть BGR)
            if len(raw.shape) == 3:
                frame = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            else:
                frame = raw

        return True, frame

    def _capture_worker(self):
        """Фоновый поток: непрерывно считывает последний кадр."""
        while self._running:
            ret, frame = self._read_raw()
            if ret and frame is not None:
                with self._lock:
                    self._last_frame = frame
                    self._last_ret = True
                    self._frame_count += 1
                    t = time.time()
                    dt = t - self._fps_time
                    if dt >= 0.5:
                        self._measured_fps = self._frame_count / dt
                        self._frame_count = 0
                        self._fps_time = t
                self._new_frame_event.set()
            else:
                time.sleep(0.005)

    def read(self, timeout=1.0, enhance_clahe=False, gain=1.0):
        """
        Считывает текущий кадр в формате 2D grayscale (H, W), dtype=uint8.
        """
        if not self.threaded:
            ret, frame = self._read_raw()
            if ret and frame is not None:
                self._frame_count += 1
                t = time.time()
                dt = t - self._fps_time
                if dt >= 0.5:
                    self._measured_fps = self._frame_count / dt
                    self._frame_count = 0
                    self._fps_time = t
        else:
            if not self._new_frame_event.wait(timeout=timeout):
                with self._lock:
                    if self._last_frame is None:
                        return False, None
            self._new_frame_event.clear()
            with self._lock:
                ret = self._last_ret
                frame = self._last_frame.copy() if self._last_frame is not None else None

        if not ret or frame is None:
            return False, None

        if gain != 1.0 and gain > 0:
            frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        if enhance_clahe:
            frame = self._clahe.apply(frame)

        return True, frame

    def read_bgr(self, timeout=1.0, enhance_clahe=False, gain=1.0):
        """
        Считывает текущий кадр и конвертирует его в 3-канальный BGR (H, W, 3).
        Удобно для прямого отображения в OpenCV и подачи в нейросеть.
        """
        ret, gray = self.read(timeout=timeout, enhance_clahe=enhance_clahe, gain=gain)
        if not ret or gray is None:
            return False, None
        return True, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @property
    def fps(self):
        """Реальный измеренный FPS потока."""
        return self._measured_fps

    @property
    def resolution(self):
        """(width, height)."""
        return self.actual_width, self.actual_height

    def open_settings_dialog(self):
        """Открывает системный диалог настроек камеры (DirectShow / Windows)."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def release(self):
        """Освобождает ресурсы камеры."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()
