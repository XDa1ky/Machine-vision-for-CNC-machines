"""
inspect.py
----------
Режим инспекции платы — основной пайплайн:
  1. Захват изображения с камеры (или загрузка файла)
  2. Разбивка на тайлы (tiling.py)
  3. Классификация каждого тайла (PCBNet)
  4. Визуализация результата с цветными метками

Запуск с вебкамерой (live-режим):
    python inspect.py --source camera

Запуск на изображении:
    python inspect.py --source image --file path/to/pcb.jpg

Запуск демо (синтетическая плата):
    python inspect.py --source demo
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# ── Импорты из нашего проекта ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from model import PCBNet
from tiling import tile_image, visualize_grid, make_demo_board

# ─────────────────────── Параметры ───────────────────────────
TILE_SIZE   = 64
OVERLAP     = 8
MODEL_PATH  = os.path.join(BASE_DIR, "models", "pcbnet_best.pt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_COLORS = {
    "resistor":  (255,  80,  80),
    "capacitor": ( 80, 180, 255),
    "chip":      (255, 200,   0),
    "empty":     (100, 100, 100),
}

# Нормализация (должна совпадать с train.py)
NORMALIZE = transforms.Normalize(mean=[0.4, 0.5, 0.2], std=[0.2, 0.2, 0.1])
TO_TENSOR = transforms.ToTensor()


# ═══════════════════════════════════════════════════════════════
#  Загрузка модели
# ═══════════════════════════════════════════════════════════════

def load_model(path: str, device: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Модель не найдена: {path}\n"
            "Сначала запустите: python train.py"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = PCBNet(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Модель загружена (val_acc={ckpt['val_acc']:.3f}, эпоха {ckpt['epoch']})")
    print(f"  Классы: {classes}")
    return model, classes


# ═══════════════════════════════════════════════════════════════
#  Классификация тайлов
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def classify_tiles(
    model: PCBNet,
    tiles: list[dict],
    classes: list[str],
    device: str,
) -> list[str]:
    """Классифицирует каждый тайл, возвращает список предсказанных классов."""
    predictions = []
    for t in tiles:
        # Преобразуем BGR→RGB, PIL→тензор
        rgb = cv2.cvtColor(t["tile"], cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((TILE_SIZE, TILE_SIZE))
        tensor = NORMALIZE(TO_TENSOR(pil)).unsqueeze(0).to(device)
        logits = model(tensor)
        idx = logits.argmax(dim=1).item()
        predictions.append(classes[idx])
    return predictions


# ═══════════════════════════════════════════════════════════════
#  Режим: демо-синтетическая плата
# ═══════════════════════════════════════════════════════════════

def run_demo(model, classes, device):
    print("\n  [Демо] Генерируем синтетическую плату...")
    board_rgb = make_demo_board(384, 256)
    board_bgr = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR)

    tiles = tile_image(board_bgr, tile_size=TILE_SIZE, overlap=OVERLAP)
    preds = classify_tiles(model, tiles, classes, device)

    vis_rgb = visualize_grid(board_rgb, tiles, preds, CLASS_COLORS)

    # Статистика
    from collections import Counter
    counts = Counter(preds)
    print(f"\n  Результат инспекции ({len(tiles)} тайлов):")
    for cls, n in sorted(counts.items()):
        bar = "█" * n
        print(f"    {cls:12s}: {n:3d}  {bar}")

    # Сохраняем
    out_path = os.path.join(BASE_DIR, "results", "inspection_demo.png")
    cv2.imwrite(out_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    print(f"\n  Сохранено: {out_path}")

    cv2.imshow("PCB Inspection — Demo", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    print("  Нажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  Режим: live-камера
# ═══════════════════════════════════════════════════════════════

def run_camera(model, classes, device, cam_id: int = 0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"  Ошибка: не удалось открыть камеру (id={cam_id})")
        return

    print(f"\n  [Camera] Нажмите Q для выхода, S для сохранения скриншота")
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Инспекция каждые 10 кадров (экономим CPU)
        if frame_i % 10 == 0:
            tiles = tile_image(frame, tile_size=TILE_SIZE, overlap=OVERLAP)
            preds = classify_tiles(model, tiles, classes, device)
            vis   = visualize_grid(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                tiles, preds, CLASS_COLORS
            )
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        cv2.putText(
            vis_bgr, "PCB Inspection  [Q=quit  S=save]",
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        cv2.imshow("PCB Inspection — Live", vis_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            fname = os.path.join(BASE_DIR, "results", f"screenshot_{ts}.png")
            cv2.imwrite(fname, vis_bgr)
            print(f"  Скриншот: {fname}")

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  Режим: файл изображения
# ═══════════════════════════════════════════════════════════════

def run_image(model, classes, device, file_path: str):
    if not os.path.exists(file_path):
        print(f"  Файл не найден: {file_path}")
        return

    frame = cv2.imread(file_path)
    tiles = tile_image(frame, tile_size=TILE_SIZE, overlap=OVERLAP)
    preds = classify_tiles(model, tiles, classes, device)

    vis_rgb = visualize_grid(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        tiles, preds, CLASS_COLORS
    )
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    from collections import Counter
    counts = Counter(preds)
    print(f"\n  Результат ({len(tiles)} тайлов):")
    for cls, n in sorted(counts.items()):
        print(f"    {cls:12s}: {n}")

    out = os.path.join(BASE_DIR, "results", "inspection_result.png")
    cv2.imwrite(out, vis_bgr)
    print(f"  Сохранено: {out}")
    cv2.imshow("PCB Inspection", vis_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  Точка входа
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Tile Inspector")
    parser.add_argument(
        "--source", choices=["demo", "camera", "image"], default="demo",
        help="Источник: demo / camera / image"
    )
    parser.add_argument("--file",   default=None, help="Путь к изображению (для --source image)")
    parser.add_argument("--cam_id", type=int, default=0, help="ID камеры (по умолчанию 0)")
    args = parser.parse_args()

    print("=" * 55)
    print("  PCB Inspection System")
    print("=" * 55)
    print(f"  Устройство: {DEVICE}")

    model, classes = load_model(MODEL_PATH, DEVICE)

    if args.source == "demo":
        run_demo(model, classes, DEVICE)
    elif args.source == "camera":
        run_camera(model, classes, DEVICE, cam_id=args.cam_id)
    elif args.source == "image":
        if not args.file:
            print("  Укажите --file <путь>")
        else:
            run_image(model, classes, DEVICE, args.file)
