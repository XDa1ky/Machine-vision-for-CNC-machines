"""
tiling.py
---------
Алгоритм разбивки изображения платы на тайлы (зоны).

Функции:
  - tile_image()     — делит изображение на равные тайлы с перекрытием
  - visualize_grid() — рисует сетку тайлов поверх изображения

Запуск демо:
    python tiling.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
#  Основная функция тайлинга
# ═══════════════════════════════════════════════════════════════

def tile_image(
    image: np.ndarray,
    tile_size: int = 64,
    overlap: int = 8,
) -> list[dict]:
    """
    Разбивает изображение на тайлы с перекрытием (overlap).

    Параметры
    ----------
    image     : np.ndarray  — входное изображение (H×W×C, BGR или RGB)
    tile_size : int         — размер тайла в пикселях
    overlap   : int         — перекрытие соседних тайлов в пикселях

    Возвращает
    ----------
    List[dict] — список тайлов, каждый содержит:
        {
            "tile"   : np.ndarray  — вырезанный тайл,
            "row"    : int         — индекс строки,
            "col"    : int         — индекс столбца,
            "x"      : int         — левый пиксель в исходном изображении,
            "y"      : int         — верхний пиксель в исходном изображении,
            "w"      : int         — ширина тайла,
            "h"      : int         — высота тайла,
        }
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap   # шаг между тайлами

    tiles = []
    row = 0
    y = 0

    while y < H:
        col = 0
        x = 0
        while x < W:
            # Границы текущего тайла
            x_end = min(x + tile_size, W)
            y_end = min(y + tile_size, H)
            crop = image[y:y_end, x:x_end]

            # Если тайл меньше tile_size — дополняем нулями
            pad_h = tile_size - crop.shape[0]
            pad_w = tile_size - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(
                    crop,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            tiles.append({
                "tile": crop,
                "row":  row,
                "col":  col,
                "x":    x,
                "y":    y,
                "w":    x_end - x,
                "h":    y_end - y,
            })

            col += 1
            x += stride
            if x >= W:
                break

        row += 1
        y += stride
        if y >= H:
            break

    return tiles


# ═══════════════════════════════════════════════════════════════
#  Визуализация сетки тайлов
# ═══════════════════════════════════════════════════════════════

def visualize_grid(
    image: np.ndarray,
    tiles: list[dict],
    predictions: list[str] | None = None,
    class_colors: dict | None = None,
) -> np.ndarray:
    """
    Рисует сетку тайлов поверх изображения.

    Параметры
    ----------
    image       : исходное изображение (RGB)
    tiles       : список тайлов из tile_image()
    predictions : список предсказанных классов (по одному на тайл)
    class_colors: словарь {класс: (R,G,B)} для цветов рамок

    Возвращает
    ----------
    np.ndarray — изображение с сеткой
    """
    if class_colors is None:
        class_colors = {
            "resistor":  (255,  80,  80),
            "capacitor": ( 80, 180, 255),
            "chip":      (255, 200,   0),
            "empty":     (100, 100, 100),
            "?":         (255, 255, 255),
        }

    vis = image.copy()
    for i, t in enumerate(tiles):
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        label = predictions[i] if predictions else "?"
        color = class_colors.get(label, (255, 255, 255))
        # Рамка тайла
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), color, 2)
        # Текст класса
        cv2.putText(
            vis, label[:3].upper(),
            (x + 2, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, color, 1, cv2.LINE_AA,
        )
    return vis


# ═══════════════════════════════════════════════════════════════
#  Создание демо-изображения платы
# ═══════════════════════════════════════════════════════════════

def make_demo_board(width: int = 384, height: int = 256) -> np.ndarray:
    """Генерирует синтетическое изображение платы для демонстрации."""
    random.seed(0)
    img = Image.new("RGB", (width, height), color=(10, 100, 10))
    draw = ImageDraw.Draw(img)

    # Дорожки
    for _ in range(20):
        x0, y0 = random.randint(0, width), random.randint(0, height)
        x1, y1 = random.randint(0, width), random.randint(0, height)
        draw.line([(x0, y0), (x1, y1)], fill=(190, 155, 0), width=2)

    # Компоненты
    components = [
        ("R", (100, 80),  (160, 100)),
        ("C", (200, 60),  (230, 90)),
        ("U", (280, 100), (340, 160)),
        ("R", (50,  150), (110, 165)),
        ("C", (160, 170), (185, 195)),
    ]
    comp_colors = {
        "R": (160, 120, 70),   # резистор
        "C": (140, 140, 180),  # конденсатор
        "U": (30,  30,  30),   # чип
    }
    for sym, (x0, y0), (x1, y1) in components:
        draw.rectangle([x0, y0, x1, y1], fill=comp_colors[sym], outline=(200, 200, 200))
        draw.text((x0 + 2, y0 + 2), sym, fill=(240, 240, 240))

    return np.array(img)


# ═══════════════════════════════════════════════════════════════
#  Демонстрация
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    board = make_demo_board(384, 256)
    tiles = tile_image(board, tile_size=64, overlap=8)

    print(f"Размер платы  : {board.shape[1]}×{board.shape[0]} пикс.")
    print(f"Размер тайла  : 64×64 пикс. (перекрытие 8 пикс.)")
    print(f"Всего тайлов  : {len(tiles)}")
    print(f"Сетка         : {max(t['row'] for t in tiles)+1} строк × "
          f"{max(t['col'] for t in tiles)+1} столбцов")

    # Фиктивные предсказания для визуализации
    classes = ["resistor", "capacitor", "chip", "empty"]
    fake_preds = [random.choice(classes) for _ in tiles]

    vis = visualize_grid(board, tiles, predictions=fake_preds)

    # ── Сохраняем результат ──
    out_path = os.path.join(BASE_DIR, "results", "tiling_demo.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\nВизуализация сохранена: {out_path}")

    # ── Показываем ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(board)
    axes[0].set_title("Исходное изображение платы")
    axes[0].axis("off")

    axes[1].imshow(vis)
    axes[1].set_title(f"Сетка тайлов ({len(tiles)} шт.)")
    axes[1].axis("off")

    # Легенда
    colors_rgb = {
        "resistor":  (255/255,  80/255,  80/255),
        "capacitor": ( 80/255, 180/255, 255/255),
        "chip":      (255/255, 200/255,   0/255),
        "empty":     (100/255, 100/255, 100/255),
    }
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors_rgb.items()]
    axes[1].legend(handles=patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", "tiling_demo_plot.png"), dpi=120)
    plt.show()
    print("Готово!")
