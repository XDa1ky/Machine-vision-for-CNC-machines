"""
inspect.py
----------
Режим инспекции платы — основной пайплайн:
  1. Захват изображения с камеры oCam-1MGN-U (или вебкамеры / файла)
  2. Разбивка на тайлы (tiling.py)
  3. Пакетная классификация каждого тайла (PCBNet)
  4. Визуализация результата с цветными метками

Запуск с камерой (live-режим):
    python pcb_inspect.py --source camera

Запуск на изображении:
    python pcb_inspect.py --source image --file path/to/pcb.jpg

Запуск демо (синтетическая плата):
    python pcb_inspect.py --source demo
"""

import os
import sys
import argparse
import time
from collections import Counter
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Импорты из нашего проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from model import PCBNet
from tiling import tile_image, visualize_grid, make_demo_board
try:
    from ocam import OCam
except ImportError:
    OCam = None

# Параметры
TILE_SIZE = 64
OVERLAP = 8
MODEL_PATH = os.path.join(BASE_DIR, "models", "pcbnet_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_COLORS = {
    "resistor":  (255,  80,  80),
    "capacitor": ( 80, 180, 255),
    "chip":      (255, 200,   0),
    "empty":     (100, 100, 100),
}

# Нормализация
NORMALIZE = transforms.Normalize(mean=[0.4, 0.5, 0.2], std=[0.2, 0.2, 0.1])
TO_TENSOR = transforms.ToTensor()


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


@torch.no_grad()
def classify_tiles(
    model: PCBNet,
    tiles: list[dict],
    classes: list[str],
    device: str,
    batch_size: int = 64
) -> list[str]:
    """Классифицирует тайлы пакетами (батчами) для высокой скорости."""
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


def run_demo(model, classes, device):
    print("\n  [Демо] Генерируем синтетическую плату...")
    board_rgb = make_demo_board(384, 256)
    board_bgr = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR)

    tiles = tile_image(board_bgr, tile_size=TILE_SIZE, overlap=OVERLAP)
    preds = classify_tiles(model, tiles, classes, device)

    vis_rgb = visualize_grid(board_rgb, tiles, preds, CLASS_COLORS)

    counts = Counter(preds)
    print(f"\n  Результат инспекции ({len(tiles)} тайлов):")
    for cls, n in sorted(counts.items()):
        bar = "█" * n
        print(f"    {cls:12s}: {n:3d}  {bar}")

    out_path = os.path.join(BASE_DIR, "results", "inspection_demo.png")
    cv2.imwrite(out_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    print(f"\n  Сохранено: {out_path}")

    cv2.imshow("PCB Inspection — Demo", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    print("  Нажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(model, classes, device, cam_id: int = 1):
    cam = None
    if OCam is not None:
        try:
            cam = OCam(index=cam_id, width=1280, height=960, threaded=True, auto_fallback=True)
            print(f"  [+] oCam подключена на индексе {cam.index} ({cam.resolution[0]}x{cam.resolution[1]})")
        except Exception as e:
            print(f"  [!] oCam не открылась ({e}), переход на standard VideoCapture...")

    if cam is None:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"  Ошибка: не удалось открыть камеру (id={cam_id})")
            return
    else:
        cap = None

    print(f"\n  [Camera] [SPACE]=Инспекция, [Q]=Выход, [S]=Сохранить")
    frame_i = 0
    vis_bgr = None

    while True:
        if cam is not None:
            ret, frame = cam.read_bgr()
        else:
            ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        if vis_bgr is None or frame_i % 15 == 0:
            tiles = tile_image(frame, tile_size=TILE_SIZE, overlap=OVERLAP)
            preds = classify_tiles(model, tiles, classes, device)
            vis = visualize_grid(
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

    if cam is not None:
        cam.release()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Tile Inspector")
    parser.add_argument(
        "--source", choices=["demo", "camera", "image"], default="demo",
        help="Источник: demo / camera / image"
    )
    parser.add_argument("--file",   default=None, help="Путь к изображению (для --source image)")
    parser.add_argument("--cam_id", type=int, default=1, help="ID камеры (по умолчанию 1)")
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
