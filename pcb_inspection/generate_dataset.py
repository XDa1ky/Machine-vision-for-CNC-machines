"""
generate_dataset.py
-------------------
Генерирует синтетический датасет тайлов платы (PCB).
Каждый тайл — изображение 64x64 пикселя одного из 4 классов:
  - resistor  (резистор)
  - capacitor (конденсатор)
  - chip      (микросхема)
  - empty     (пустая зона)

Запуск:
    python generate_dataset.py

Результат: папки data/tiles/train/<класс>/ и data/tiles/val/<класс>/
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

# ─────────────────────────── Параметры ───────────────────────────
TILE_SIZE      = 64          # размер одного тайла (пикселей)
TRAIN_PER_CLS  = 80          # тайлов на класс в train
VAL_PER_CLS    = 20          # тайлов на класс в val
SEED           = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "tiles")

CLASSES = ["resistor", "capacitor", "chip", "empty"]

random.seed(SEED)
np.random.seed(SEED)


# ─────────────── Функции рисования компонентов ───────────────────

def pcb_background(draw: ImageDraw.ImageDraw, size: int):
    """Рисует фон платы — зелёный с лёгким шумом."""
    r = random.randint(0, 20)
    g = random.randint(80, 120)
    b = random.randint(0, 20)
    # Заливаем фон
    draw.rectangle([0, 0, size, size], fill=(r, g, b))

    # Дорожки (линии)
    for _ in range(random.randint(1, 4)):
        x0 = random.randint(0, size)
        y0 = random.randint(0, size)
        x1 = random.randint(0, size)
        y1 = random.randint(0, size)
        color = (random.randint(150, 220), random.randint(100, 160), 0)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=2)


def draw_resistor(draw: ImageDraw.ImageDraw, size: int):
    """Резистор — вытянутый прямоугольник с полосками."""
    pcb_background(draw, size)
    cx, cy = size // 2, size // 2
    w, h = random.randint(20, 32), random.randint(8, 14)
    # Тело
    body_color = (random.randint(140, 200), random.randint(100, 150), random.randint(60, 100))
    draw.rectangle([cx - w, cy - h, cx + w, cy + h], fill=body_color, outline=(30, 30, 30), width=1)
    # Полоски
    colors = [(200, 50, 50), (50, 50, 200), (200, 200, 50), (50, 200, 50), (100, 100, 100)]
    stripe_x = cx - w + 5
    for c in random.sample(colors, 3):
        draw.line([(stripe_x, cy - h), (stripe_x, cy + h)], fill=c, width=2)
        stripe_x += 6
    # Выводы
    draw.line([(cx - w - 8, cy), (cx - w, cy)], fill=(210, 180, 0), width=2)
    draw.line([(cx + w, cy), (cx + w + 8, cy)], fill=(210, 180, 0), width=2)


def draw_capacitor(draw: ImageDraw.ImageDraw, size: int):
    """Конденсатор — цилиндр (вид сверху = эллипс или прямоугольник)."""
    pcb_background(draw, size)
    cx, cy = size // 2, size // 2
    r = random.randint(10, 18)
    # Тело (алюминиевый цилиндр)
    cap_color = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cap_color, outline=(30, 30, 30), width=1)
    # Крестик сверху (маркировка)
    draw.line([(cx - r + 2, cy), (cx + r - 2, cy)], fill=(20, 20, 20), width=1)
    draw.line([(cx, cy - r + 2), (cx, cy + r - 2)], fill=(20, 20, 20), width=1)
    # Полоска (полярность)
    stripe_color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
    draw.rectangle([cx + r // 2, cy - r, cx + r, cy + r], fill=stripe_color)
    # Выводы
    draw.line([(cx - r - 8, cy - 4), (cx - r, cy - 4)], fill=(210, 180, 0), width=2)
    draw.line([(cx - r - 8, cy + 4), (cx - r, cy + 4)], fill=(210, 180, 0), width=2)


def draw_chip(draw: ImageDraw.ImageDraw, size: int):
    """Микросхема (чип) — квадрат с ножками."""
    pcb_background(draw, size)
    cx, cy = size // 2, size // 2
    s = random.randint(14, 20)
    # Корпус
    chip_color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
    draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=chip_color, outline=(80, 80, 80), width=1)
    # Точка ориентации
    draw.ellipse([cx - s + 2, cy - s + 2, cx - s + 7, cy - s + 7], fill=(200, 200, 200))
    # Ножки (пины)
    pin_color = (200, 180, 50)
    for i in range(-2, 3):
        y = cy + i * 6
        # Левые ножки
        draw.rectangle([cx - s - 8, y - 2, cx - s, y + 2], fill=pin_color)
        # Правые ножки
        draw.rectangle([cx + s, y - 2, cx + s + 8, y + 2], fill=pin_color)
    for i in range(-1, 2):
        x = cx + i * 8
        # Верхние ножки
        draw.rectangle([x - 2, cy - s - 8, x + 2, cy - s], fill=pin_color)
        # Нижние ножки
        draw.rectangle([x - 2, cy + s, x + 2, cy + s + 8], fill=pin_color)


def draw_empty(draw: ImageDraw.ImageDraw, size: int):
    """Пустая зона платы — только фон с дорожками."""
    pcb_background(draw, size)
    # Иногда добавляем контактную площадку
    if random.random() < 0.4:
        x = random.randint(10, size - 10)
        y = random.randint(10, size - 10)
        pad_color = (210, 180, 0)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=pad_color)


# ─────────────────────── Генератор шума ──────────────────────────

def add_noise(img: Image.Image) -> Image.Image:
    """Добавляет небольшой случайный шум к изображению."""
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ───────────────────────── Основной цикл ─────────────────────────

DRAWERS = {
    "resistor":  draw_resistor,
    "capacitor": draw_capacitor,
    "chip":      draw_chip,
    "empty":     draw_empty,
}


def generate_split(split: str, count: int):
    for cls in CLASSES:
        out_dir = os.path.join(DATA_DIR, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        draw_fn = DRAWERS[cls]

        for i in range(count):
            img = Image.new("RGB", (TILE_SIZE, TILE_SIZE))
            draw = ImageDraw.Draw(img)
            draw_fn(draw, TILE_SIZE)
            img = add_noise(img)
            img.save(os.path.join(out_dir, f"{cls}_{i:04d}.png"))

        print(f"  [{split:5s}] {cls:12s}: {count} img -> {out_dir}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Генерация синтетического датасета PCB-тайлов")
    print("=" * 55)
    print(f"  Классы: {CLASSES}")
    print(f"  Размер тайла: {TILE_SIZE}x{TILE_SIZE}")
    print(f"  Train / Val: {TRAIN_PER_CLS} / {VAL_PER_CLS} на класс")
    print("-" * 55)

    generate_split("train", TRAIN_PER_CLS)
    generate_split("val",   VAL_PER_CLS)

    total = (TRAIN_PER_CLS + VAL_PER_CLS) * len(CLASSES)
    print("-" * 55)
    print(f"  Готово! Всего изображений: {total}")
    print("=" * 55)
