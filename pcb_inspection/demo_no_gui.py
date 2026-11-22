"""
demo_no_gui.py  —  демо инспекции без окна OpenCV
Сохраняет результат в results/inspection_demo.png
"""
import os, sys
import cv2, torch
import numpy as np
from collections import Counter
from torchvision import transforms
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pcbnet_best.pt")
RES_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from model import PCBNet
from tiling import tile_image, visualize_grid, make_demo_board

TILE_SIZE = 64
OVERLAP   = 8
DEVICE    = "cpu"

CLASS_COLORS = {
    "resistor":  (255,  80,  80),
    "capacitor": ( 80, 180, 255),
    "chip":      (255, 200,   0),
    "empty":     (100, 100, 100),
}

NORMALIZE = transforms.Normalize(mean=[0.4, 0.5, 0.2], std=[0.2, 0.2, 0.1])
TO_TENSOR = transforms.ToTensor()

# --- load model ---
ckpt    = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = ckpt["classes"]
model   = PCBNet(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded  val_acc={ckpt['val_acc']:.3f}  classes={classes}")

# --- make board ---
board_rgb = make_demo_board(384, 256)
board_bgr = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR)

# --- tile ---
tiles = tile_image(board_bgr, tile_size=TILE_SIZE, overlap=OVERLAP)
print(f"Tiles: {len(tiles)}")

# --- classify ---
preds = []
with torch.no_grad():
    for t in tiles:
        rgb    = cv2.cvtColor(t["tile"], cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb).resize((TILE_SIZE, TILE_SIZE))
        tensor = NORMALIZE(TO_TENSOR(pil)).unsqueeze(0)
        idx    = model(tensor).argmax(dim=1).item()
        preds.append(classes[idx])

# --- stats ---
counts = Counter(preds)
print("\nInspection result:")
for cls, n in sorted(counts.items()):
    print(f"  {cls:12s}: {n}")

# --- visualize ---
vis_rgb = visualize_grid(board_rgb, tiles, preds, CLASS_COLORS)
out = os.path.join(RES_DIR, "inspection_demo.png")
Image.fromarray(vis_rgb).save(out)
print(f"\nSaved: {out}")
