"""
Скрипт для проверки и тестирования камеры Withrobot oCam-1MGN-U
(1MP Global Shutter Monochrome USB 3.0 Camera).

Камера выдаёт «сырой» формат Y800 (8-bit mono), который стандартный бэкенд
OpenCV MSMF не умеет конвертировать в RGB. Для корректного захвата необходимо
отключить HW-ускорение и конвертацию в RGB, а полученный буфер вручную
пересобирать в 2D-массив (reshape).

Управление (горячие клавиши в окне просмотра):
  [P]       - Открыть окно настроек DirectShow (экспозиция, gain)
  [F]       - Режим Focus Peaking (подсветка резких контуров зелёным)
  [C]       - Вкл/Выкл перекрестие по центру
  [H]       - Вкл/Выкл гистограмму яркости
  [R]       - Старт / Стоп запись видео (AVI)
  [S]       - Сохранить снимок (PNG)
  [Q]/[ESC] - Выход
"""

import sys
import time
import cv2
import numpy as np


# === Индекс камеры oCam-1MGN-U в системе ===
# Обычно 1 (0 = встроенная веб-камера ноутбука).
# Можно переопределить аргументом: python test.py 2
OCAM_INDEX = 1


def open_ocam(index):
    """
    Открывает камеру oCam-1MGN-U через MSMF с отключённым HW-ускорением
    и конвертацией RGB. Это единственный способ захватить Y800 (монохром)
    в стандартном opencv-python на Windows.
    """
    cap = cv2.VideoCapture()
    params = [
        cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE,
        cv2.CAP_PROP_CONVERT_RGB, 0,
    ]
    cap.open(index, cv2.CAP_MSMF, params)

    if not cap.isOpened():
        return None

    # Запрашиваем нативное разрешение 1280x960
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    return cap


def read_mono_frame(cap, width, height):
    """
    Считывает один кадр из oCam.
    MSMF с CONVERT_RGB=0 возвращает плоский буфер (1, W*H).
    Пересобираем его в 2D-массив (H, W).
    """
    ret, raw = cap.read()
    if not ret or raw is None:
        return False, None

    expected_size = width * height
    actual_size = raw.size

    if actual_size == expected_size:
        frame = raw.reshape(height, width)
    elif actual_size > expected_size:
        # Камера отдаёт буфер большего размера (бывает при смене разрешения)
        frame = raw.flatten()[:expected_size].reshape(height, width)
    else:
        return False, None

    return True, frame


def draw_histogram(gray_frame, width=220, height=80):
    """Строит компактную гистограмму распределения яркости."""
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)

    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    scale_x = width / 256.0
    for x in range(1, 256):
        x1 = int((x - 1) * scale_x)
        x2 = int(x * scale_x)
        pt1 = (x1, height - int(hist[x - 1][0]))
        pt2 = (x2, height - int(hist[x][0]))
        cv2.line(hist_img, pt1, pt2, (0, 255, 0), 1)

    return hist_img


def main():
    # Определяем индекс камеры
    cam_index = OCAM_INDEX
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_index = int(sys.argv[1])

    print("=" * 65)
    print("  oCam-1MGN-U Diagnostic Viewer")
    print("  Withrobot Global Shutter Mono Camera | Y800 Raw Capture")
    print("=" * 65)
    print(f"[*] Открытие камеры на индексе [{cam_index}]...")
    print("    (MSMF + HW_ACCELERATION=NONE + CONVERT_RGB=0)")

    cap = open_ocam(cam_index)
    if cap is None:
        print(f"\n[!] Не удалось открыть камеру на индексе [{cam_index}].")
        print("Проверьте:")
        print("  1. Подключена ли oCam по USB 3.0.")
        print("  2. Не занята ли камера другой программой.")
        print(f"  3. Попробуйте другой индекс: python test.py 0  или  python test.py 2")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[+] Камера открыта!")
    print(f"    Разрешение: {actual_w}x{actual_h}")
    print(f"    Формат:     {fourcc_str} (8-bit monochrome)")
    print(f"    FPS (цель): {cam_fps}")
    print()
    print("Горячие клавиши:")
    print("  [P] Настройки DirectShow   [F] Focus Peaking")
    print("  [C] Перекрестие            [H] Гистограмма")
    print("  [R] Старт/Стоп записи     [S] Снимок (PNG)")
    print("  [Q] / [ESC] Выход")
    print("=" * 65)

    # --- Состояние интерфейса ---
    window_name = f"oCam-1MGN-U [{actual_w}x{actual_h}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 960))

    show_hist = True
    show_crosshair = False
    focus_peaking = False
    recording = False
    video_writer = None

    prev_time = time.time()
    fps_display = 0.0
    frame_count = 0
    max_sharpness = 0.0

    while True:
        ret, gray = read_mono_frame(cap, actual_w, actual_h)
        if not ret:
            time.sleep(0.01)
            continue

        # FPS
        frame_count += 1
        now = time.time()
        dt = now - prev_time
        if dt >= 0.5:
            fps_display = frame_count / dt
            frame_count = 0
            prev_time = now

        # Резкость (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        max_sharpness = max(max_sharpness, sharpness)

        # Яркость
        min_val = int(gray.min())
        max_val = int(gray.max())
        mean_val = gray.mean()

        # --- Подготовка кадра для отображения ---
        if focus_peaking:
            edges = cv2.Canny(gray, 40, 120)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            display[edges > 0] = [0, 255, 0]
        else:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Перекрестие
        if show_crosshair:
            cx, cy = actual_w // 2, actual_h // 2
            cv2.line(display, (cx - 50, cy), (cx + 50, cy), (0, 0, 255), 1)
            cv2.line(display, (cx, cy - 50), (cx, cy + 50), (0, 0, 255), 1)
            cv2.circle(display, (cx, cy), 20, (0, 0, 255), 1)

        # Запись видео
        if recording and video_writer is not None:
            video_writer.write(display)

        # --- HUD ---
        rec_tag = " [REC]" if recording else ""
        lines = [
            f"oCam-1MGN-U | {actual_w}x{actual_h} | {fps_display:.1f} FPS{rec_tag}",
            f"Sharpness: {sharpness:.1f} (Peak: {max_sharpness:.1f})",
            f"Brightness: min={min_val}  max={max_val}  avg={mean_val:.1f}",
        ]

        overlay = display.copy()
        cv2.rectangle(overlay, (5, 5), (520, 82), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        for i, text in enumerate(lines):
            y = 23 + i * 22
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            if recording and i == 0:
                color = (0, 0, 255)
            cv2.putText(display, text, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Предупреждения
        if mean_val < 5:
            cv2.putText(display, "(!!) OCHEN' TEMNO — prover'te ob'ektiv i [P] exposure",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        elif mean_val > 248:
            cv2.putText(display, "(!!) ZASVETKA — umen'shite exposure/gain [P]",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

        # Индикатор записи (мигающий красный кружок)
        if recording and int(time.time() * 2) % 2 == 0:
            cv2.circle(display, (actual_w - 25, 25), 10, (0, 0, 255), -1)

        # Гистограмма
        if show_hist:
            hist_img = draw_histogram(gray)
            hh, hw = hist_img.shape[:2]
            hy = display.shape[0] - hh - 10
            hx = display.shape[1] - hw - 10
            if hy > 0 and hx > 0:
                display[hy:hy + hh, hx:hx + hw] = hist_img
                cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh), (180, 180, 180), 1)

        cv2.imshow(window_name, display)

        # --- Клавиши ---
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("[*] Выход...")
            break

        elif key in (ord('s'), ord('S')):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"ocam_snapshot_{ts}.png"
            cv2.imwrite(fname, gray)
            print(f"[+] Снимок: {fname}")

        elif key in (ord('h'), ord('H')):
            show_hist = not show_hist

        elif key in (ord('c'), ord('C')):
            show_crosshair = not show_crosshair

        elif key in (ord('f'), ord('F')):
            focus_peaking = not focus_peaking
            state = "ВКЛ" if focus_peaking else "ВЫКЛ"
            print(f"[*] Focus Peaking: {state}")

        elif key in (ord('p'), ord('P')):
            print("[*] Настройки камеры DirectShow...")
            cap.set(cv2.CAP_PROP_SETTINGS, 1)

        elif key in (ord('r'), ord('R')):
            recording = not recording
            if recording:
                ts = time.strftime("%Y%m%d_%H%M%S")
                rec_name = f"ocam_rec_{ts}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                rec_fps = fps_display if fps_display > 5 else 30.0
                video_writer = cv2.VideoWriter(rec_name, fourcc, rec_fps,
                                               (actual_w, actual_h))
                print(f"[+] Запись: {rec_name} @ {rec_fps:.0f} fps")
            else:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("[-] Запись остановлена.")

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
