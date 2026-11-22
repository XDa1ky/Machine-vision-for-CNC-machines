"""
model.py
--------
CNN-классификатор тайлов платы.

Архитектура PCBNet:
  Input (3×64×64)
    → Conv1 (32 фильтра, 3×3) + BN + ReLU + MaxPool
    → Conv2 (64 фильтра, 3×3) + BN + ReLU + MaxPool
    → Conv3 (128 фильтров, 3×3) + BN + ReLU + MaxPool
    → Flatten → FC(256) + Dropout → FC(num_classes)

Классы: resistor / capacitor / chip / empty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Архитектура сети
# ═══════════════════════════════════════════════════════════════

class PCBNet(nn.Module):
    """
    Лёгкая свёрточная нейронная сеть для классификации тайлов платы.

    Параметры
    ----------
    num_classes : int — количество классов (по умолчанию 4)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        # ── Сверточные блоки ──────────────────────────────────────
        # Блок 1: 3×64×64 → 32×32×32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 64 → 32
        )

        # Блок 2: 32×32×32 → 64×16×16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 32 → 16
        )

        # Блок 3: 64×16×16 → 128×8×8
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 16 → 8
        )

        # ── Классификатор ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Возвращает число обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (1, 3, 64, 64)) -> None:
    """Выводит краткое описание архитектуры и число параметров."""
    print("=" * 50)
    print("  PCBNet — архитектура")
    print("=" * 50)

    # Прогоняем фиктивный батч
    device = next(model.parameters()).device
    x = torch.zeros(input_size, device=device)
    hooks = []
    shapes = []

    def hook_fn(module, inp, out):
        shapes.append((type(module).__name__, out.shape))

    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and layer is not model:
            hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    for name, shape in shapes:
        print(f"  {name:20s}  → {tuple(shape)}")

    print("-" * 50)
    print(f"  Обучаемых параметров: {count_parameters(model):,}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════
#  Запуск как скрипта — проверка архитектуры
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = PCBNet(num_classes=4)
    model_summary(model)

    # Тест: прогоняем батч из 4 тайлов
    dummy = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Тест прямого прохода:")
    print(f"  Вход:  {tuple(dummy.shape)}")
    print(f"  Выход: {tuple(out.shape)}  (батч × классы)")
    print(f"  Logits (первый элемент): {out[0].numpy().round(3)}")
