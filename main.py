"""
main.py
═══════════════════════════════════════════════════════════════════════════════
Главная точка входа в проект:
"Система автоматической оптической инспекции печатных плат (AOI PCB Inspection)
 с камерой Withrobot oCam-1MGN-U (Global Shutter Monochrome USB 3.0)"
═══════════════════════════════════════════════════════════════════════════════

Запуск:
  python main.py               — интерактивное меню (по умолчанию Живая Инспекция)
  python main.py --live        — прямой запуск живой инспекции с камеры oCam
  python main.py --demo        — запуск демо-инспекции синтетической платы
  python main.py --image <img  — инспекция платы из файла
  python main.py --focus       — режим настройки фокуса и оптики камеры
  python main.py --train       — переобучение нейросети PCBNet
"""

import os
import sys
import time
import argparse
from collections import Counter
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Пути проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCB_DIR = os.path.join(BASE_DIR, "pcb_inspection")
MODELS_DIR = os.path.join(PCB_DIR, "models")
RESULTS_DIR = os.path.join(PCB_DIR, "results")
MODEL_PATH = os.path.join(MODELS_DIR, "pcbnet_best.pt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Подключение модулей
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PCB_DIR)

from ocam import OCam
from model import PCBNet
from tiling import tile_image, visualize_grid, make_demo_board

# Параметры инспекции
TILE_SIZE = 64
OVERLAP = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_COLORS = {
    "resistor":  (255,  80,  80),   # Красный
    "capacitor": ( 80, 180, 255),   # Голубой
    "chip":      (255, 200,   0),   # Желтый
    "empty":     ( 80,  80,  80),   # Серый
}

NORMALIZE = transforms.Normalize(mean=[0.4, 0.5, 0.2], std=[0.2, 0.2, 0.1])
TO_TENSOR = transforms.ToTensor()


# ═══════════════════════════════════════════════════════════════
#  Проверка и авто-инициализация модели
# ═══════════════════════════════════════════════════════════════

def ensure_model_exists():
    """Проверяет наличие обученной модели; если нет — обучает автоматически."""
    if os.path.exists(MODEL_PATH):
        return True

    print("\n[!] Модель pcbnet_best.pt не найдена. Запуск автоматической подготовки...")
    try:
        from generate_dataset import main as gen_data
        print("  1/2 Генерация синтетического датасета...")
        gen_data()
    except Exception as e:
        print(f"  Ошибка генерации датасета: {e}")
        return False

    try:
        from train import main as train_model
        print("  2/2 Обучение PCBNet...")
        train_model()
    except Exception as e:
        print(f"  Ошибка обучения: {e}")
        return False

    return os.path.exists(MODEL_PATH)


def load_pcb_model():
    """Загружает модель PCBNet."""
    if not ensure_model_exists():
        raise FileNotFoundError(f"Не удалось загрузить или создать модель: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    model = PCBNet(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


# ═══════════════════════════════════════════════════════════════
#  Пакетная классификация (Batched Inference — до 40x быстрее)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def classify_tiles_batched(model, tiles, classes, device, batch_size=64):
    """
    Классифицирует список тайлов пакетами (батчами) для максимального FPS.
    """
    predictions = []
    tensors = []

    for t in tiles:
        tile = t["tile"]
        if len(tile.shape) == 2:
            rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb).resize((TILE_SIZE, TILE_SIZE))
        tensor = NORMALIZE(TO_TENSOR(pil))
        tensors.append(tensor)

        if len(tensors) >= batch_size:
            batch = torch.stack(tensors).to(device)
            preds = model(batch).argmax(dim=1).tolist()
            predictions.extend([classes[i] for i in preds])
            tensors = []

    if tensors:
        batch = torch.stack(tensors).to(device)
        preds = model(batch).argmax(dim=1).tolist()
        predictions.extend([classes[i] for i in preds])

    return predictions


def analyze_board(image_bgr, model, classes, device):
    """
    Выполняет полный цикл инспекции платы:
    1. Нарезка на тайлы.
    2. Пакетная классификация.
    3. Отрисовка разметки и подсчет статистики.
    """
    start_t = time.time()
    tiles = tile_image(image_bgr, tile_size=TILE_SIZE, overlap=OVERLAP)
    preds = classify_tiles_batched(model, tiles, classes, device)
    vis_rgb = visualize_grid(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), tiles, preds, CLASS_COLORS)
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    counts = Counter(preds)
    elapsed_ms = (time.time() - start_t) * 1000

    return vis_bgr, preds, counts, tiles, elapsed_ms


# ═══════════════════════════════════════════════════════════════
#  РЕЖИМ 1: Живая инспекция платы с oCam-1MGN-U
# ═══════════════════════════════════════════════════════════════

def run_live_inspection(cam_index=1):
    print("\n" + "=" * 68)
    print("  РЕЖИМ: Живая инспекция печатных плат (Live AOI PCB Inspection)")
    print("=" * 68)

    model, classes = load_pcb_model()
    print(f"  [+] Модель загружена ({DEVICE}). Классы: {classes}")

    print(f"  [*] Подключение камеры oCam-1MGN-U (индекс {cam_index})...")
    try:
        cam = OCam(index=cam_index, width=1280, height=960, threaded=True, auto_fallback=True)
    except Exception as e:
        print(f"  [!] Ошибка камеры: {e}")
        return

    w, h = cam.resolution
    print(f"  [+] Камера активна: {w}x{h} (oCam: {cam.is_ocam})")
    print("\nГорячие клавиши:")
    print("  [ПРОБЕЛ] - Заморозить кадр и выполнить детальную инспекцию платы")
    print("  [T]      - Вкл/Выкл непрерывную живую инспекцию в реальном времени")
    print("  [A]      - Вкл/Выкл CLAHE (авто-улучшение контраста дорожек)")
    print("  [F]      - Режим Focus Peaking (подсветка резких контуров)")
    print("  [+] / [-]- Регулировка яркости/усиления (Gain)")
    print("  [C]      - Вкл/Выкл сетку центровки")
    print("  [P]      - Аппаратные настройки драйвера (DirectShow)")
    print("  [S]      - Сохранить текущее изображение в results/")
    print("  [Q]/[ESC]- Выход в меню\n")

    win_name = "PCB AOI Inspector — oCam-1MGN-U"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(w, 1280), min(h, 960))

    continuous_inspect = False
    frozen_inspection = None
    show_crosshair = False
    focus_peaking = False
    clahe_enabled = False
    gain_level = 1.0

    fps_display = 0.0
    frame_count = 0
    prev_time = time.time()
    last_stats_str = "Status: Live Feed (Press SPACE to Inspect)"

    try:
        while True:
            if frozen_inspection is not None:
                # Режим просмотра результатов замороженного анализа
                display = frozen_inspection.copy()
                key = cv2.waitKey(30) & 0xFF
                if key in (32, ord(' ')):  # Пробел размораживает
                    frozen_inspection = None
                    last_stats_str = "Status: Live Feed (Press SPACE to Inspect)"
                    continue
                elif key in (ord('s'), ord('S')):
                    ts = int(time.time())
                    out_p = os.path.join(RESULTS_DIR, f"inspection_frozen_{ts}.png")
                    cv2.imwrite(out_p, frozen_inspection)
                    print(f"  [+] Сохранено: {out_p}")
                elif key in (ord('q'), ord('Q'), 27):
                    break
                cv2.imshow(win_name, display)
                continue

            ret, bgr_frame = cam.read_bgr(enhance_clahe=clahe_enabled, gain=gain_level)
            if not ret or bgr_frame is None:
                time.sleep(0.01)
                continue

            # Расчет FPS
            frame_count += 1
            now = time.time()
            if now - prev_time >= 0.5:
                fps_display = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            # Резкость
            gray_for_sharpness = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_for_sharpness, cv2.CV_64F).var()

            # Если включена непрерывная инспекция
            if continuous_inspect:
                vis_bgr, preds, counts, tiles, el_ms = analyze_board(bgr_frame, model, classes, DEVICE)
                display = vis_bgr
                last_stats_str = (
                    f"R:{counts.get('resistor', 0)} C:{counts.get('capacitor', 0)} "
                    f"U:{counts.get('chip', 0)} Emp:{counts.get('empty', 0)} ({el_ms:.0f}ms)"
                )
            else:
                if focus_peaking:
                    edges = cv2.Canny(gray_for_sharpness, 40, 120)
                    display = bgr_frame.copy()
                    display[edges > 0] = [0, 255, 0]
                else:
                    display = bgr_frame.copy()

            # Перекрестие
            if show_crosshair:
                cx, cy = w // 2, h // 2
                cv2.line(display, (cx - 40, cy), (cx + 40, cy), (0, 0, 255), 1)
                cv2.line(display, (cx, cy - 40), (cx, cy + 40), (0, 0, 255), 1)
                cv2.circle(display, (cx, cy), 18, (0, 0, 255), 1)

            # Наложение HUD панели
            overlay = display.copy()
            cv2.rectangle(overlay, (5, 5), (650, 85), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            mode_tag = "[CONTINUOUS AOI]" if continuous_inspect else "[LIVE PREVIEW]"
            clahe_tag = "CLAHE:ON" if clahe_enabled else "CLAHE:OFF"
            cv2.putText(
                display,
                f"oCam-1MGN-U {w}x{h} | {fps_display:.1f} FPS | {mode_tag} | Sharp:{sharpness:.0f}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                display,
                last_stats_str,
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                display,
                f"[SPACE]=Inspect  [T]=Toggle Live  [A]={clahe_tag}  [F]=Peak  [+/-]=Gain:{gain_level:.1f}x  [S]=Save",
                (15, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA
            )

            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (32, ord(' ')):
                # Одиночный полный анализ текущего кадра
                print("  [*] Анализ кадра платы...")
                vis_bgr, preds, counts, tiles, el_ms = analyze_board(bgr_frame, model, classes, DEVICE)
                
                # Добавляем инфо-панель отчета
                report_overlay = vis_bgr.copy()
                cv2.rectangle(report_overlay, (0, 0), (w, 40), (10, 10, 10), -1)
                cv2.addWeighted(report_overlay, 0.8, vis_bgr, 0.2, 0, vis_bgr)
                rep_text = (
                    f"INSPECTION RESULT: Resistors: {counts.get('resistor', 0)} | "
                    f"Capacitors: {counts.get('capacitor', 0)} | Chips: {counts.get('chip', 0)} | "
                    f"Empty: {counts.get('empty', 0)}  [SPACE to Resume, S to Save]"
                )
                cv2.putText(vis_bgr, rep_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                
                frozen_inspection = vis_bgr
                print(f"  [+] Найдено: {dict(counts)} (обработано {len(tiles)} зон за {el_ms:.1f} мс)")
            elif key in (ord('t'), ord('T')):
                continuous_inspect = not continuous_inspect
                print(f"  [*] Непрерывная инспекция: {'ВКЛ' if continuous_inspect else 'ВЫКЛ'}")
            elif key in (ord('a'), ord('A')):
                clahe_enabled = not clahe_enabled
            elif key in (ord('f'), ord('F')):
                focus_peaking = not focus_peaking
            elif key in (ord('+'), ord('=')):
                gain_level = min(10.0, gain_level + 0.5)
            elif key in (ord('-'), ord('_')):
                gain_level = max(0.5, gain_level - 0.5)
            elif key in (ord('c'), ord('C')):
                show_crosshair = not show_crosshair
            elif key in (ord('p'), ord('P')):
                cam.open_settings_dialog()
            elif key in (ord('s'), ord('S')):
                ts = int(time.time())
                out_p = os.path.join(RESULTS_DIR, f"live_snapshot_{ts}.png")
                cv2.imwrite(out_p, bgr_frame)
                print(f"  [+] Снимок сохранён: {out_p}")

    finally:
        cam.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  РЕЖИМ 2: Инспекция изображения из файла
# ═══════════════════════════════════════════════════════════════

def run_file_inspection(image_path=None):
    if not image_path:
        image_path = input("\nВведите путь к файлу изображения платы (или Enter для отмены): ").strip()
        if not image_path:
            return

    if not os.path.exists(image_path):
        print(f"  [!] Файл не найден: {image_path}")
        return

    model, classes = load_pcb_model()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [!] Не удалось загрузить изображение: {image_path}")
        return

    print(f"\n  [*] Анализ файла {image_path} ({frame.shape[1]}x{frame.shape[0]})...")
    vis_bgr, preds, counts, tiles, el_ms = analyze_board(frame, model, classes, DEVICE)

    print(f"\n  ════════════════ Результаты инспекции ({len(tiles)} тайлов) ════════════════")
    for cls, cnt in sorted(counts.items()):
        bar = "█" * cnt
        print(f"    {cls:12s}: {cnt:3d}  {bar}")
    print(f"  Время анализа: {el_ms:.1f} мс")

    out_file = os.path.join(RESULTS_DIR, "file_inspection_result.png")
    cv2.imwrite(out_file, vis_bgr)
    print(f"  [+] Размеченное изображение сохранено в: {out_file}")

    cv2.imshow("PCB File Inspection Result", vis_bgr)
    print("  Нажмите любую клавишу в окне для продолжения...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  РЕЖИМ 3: Демо-инспекция синтетической платы
# ═══════════════════════════════════════════════════════════════

def run_synthetic_demo():
    print("\n" + "=" * 68)
    print("  РЕЖИМ: Демонстрационная инспекция (Synthetic PCB Demo)")
    print("=" * 68)

    model, classes = load_pcb_model()

    print("  [*] Генерация тестовой печатной платы...")
    board_rgb = make_demo_board(512, 384)
    board_bgr = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR)

    vis_bgr, preds, counts, tiles, el_ms = analyze_board(board_bgr, model, classes, DEVICE)

    print(f"\n  Результаты классификации ({len(tiles)} тайлов):")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:12s}: {cnt:3d}  {'█' * cnt}")
    print(f"  Время обработки: {el_ms:.1f} мс")

    out_path = os.path.join(RESULTS_DIR, "demo_inspection_result.png")
    cv2.imwrite(out_path, vis_bgr)
    print(f"\n  [+] Результат сохранен: {out_path}")

    # Создаем сравнительное изображение: оригинал слева, результат справа
    combined = np.hstack([board_bgr, vis_bgr])
    cv2.putText(combined, "Original PCB", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "AOI Inspection Result", (512 + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("PCB Inspection — Demo Comparison", combined)
    print("  Нажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  РЕЖИМ 4: Настройка камеры, оптики и фокуса
# ═══════════════════════════════════════════════════════════════

def run_camera_tuning(cam_index=1):
    """Запускает специализированный режим фокусировки test.py."""
    import test
    print("\n[*] Запуск интерактивного режима юстировки oCam-1MGN-U...")
    test.main()


# ═══════════════════════════════════════════════════════════════
#  РЕЖИМ 5: Переобучение нейросети PCBNet
# ═══════════════════════════════════════════════════════════════

def run_retrain():
    print("\n" + "=" * 68)
    print("  РЕЖИМ: Обучение / Переобучение нейросети PCBNet")
    print("=" * 68)
    ans = input("Сгенерировать новый датасет перед обучением? [y/N]: ").strip().lower()
    if ans in ("y", "yes", "д", "да"):
        from generate_dataset import main as gen_data
        gen_data()

    from train import main as train_model
    train_model()
    print("\n[+] Обучение завершено. Новая модель сохранена в pcbnet_best.pt")


# ═══════════════════════════════════════════════════════════════
#  Главное интерактивное меню
# ═══════════════════════════════════════════════════════════════

def interactive_menu():
    while True:
        print("\n" + "═" * 70)
        print("   СИСТЕМА АВТОМАТИЧЕСКОЙ ОПТИЧЕСКОЙ ИНСПЕКЦИИ ПЛАТ (AOI PCB)")
        print("   Камера: Withrobot oCam-1MGN-U (1MP Global Shutter Mono)")
        print("   Нейросеть: PCBNet (Резисторы / Конденсаторы / Микросхемы)")
        print("═" * 70)
        print("  [1] Живая инспекция платы с камеры (Live AOI Inspection) — РЕКОМЕНДУЕТСЯ")
        print("  [2] Инспекция файла изображения (Image File Inspection)")
        print("  [3] Демо-инспекция синтетической платы (Synthetic Demo)")
        print("  [4] Настройка резкости и фокуса камеры oCam (Focus Peaking + Zoom)")
        print("  [5] Переобучить нейросеть PCBNet (Train Model)")
        print("  [0] Выход")
        print("═" * 70)

        choice = input("Выберите режим [1]: ").strip()
        if not choice:
            choice = "1"

        if choice == "1":
            run_live_inspection()
        elif choice == "2":
            run_file_inspection()
        elif choice == "3":
            run_synthetic_demo()
        elif choice == "4":
            run_camera_tuning()
        elif choice == "5":
            run_retrain()
        elif choice in ("0", "q", "exit"):
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор, попробуйте еще раз.")


def main():
    parser = argparse.ArgumentParser(description="PCB AOI Inspection System")
    parser.add_argument("--live", action="store_true", help="Запустить живую инспекцию с камеры")
    parser.add_argument("--demo", action="store_true", help="Запустить демо-инспекцию")
    parser.add_argument("--image", type=str, default=None, help="Путь к изображению для инспекции")
    parser.add_argument("--focus", action="store_true", help="Запустить режим фокусировки камеры")
    parser.add_argument("--train", action="store_true", help="Переобучить модель")
    parser.add_argument("--cam", type=int, default=1, help="Индекс камеры (по умолчанию 1)")
    args = parser.parse_args()

    if args.live:
        run_live_inspection(cam_index=args.cam)
    elif args.demo:
        run_synthetic_demo()
    elif args.image:
        run_file_inspection(image_path=args.image)
    elif args.focus:
        run_camera_tuning(cam_index=args.cam)
    elif args.train:
        run_retrain()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
