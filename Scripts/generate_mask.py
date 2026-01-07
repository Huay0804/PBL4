from pathlib import Path 
import json
from PIL import Image, ImageDraw

root = Path("archive/Teeth Segmentation JSON/d2")
ann_dir = root / "ann"
out_dir = root / "masks_semantic"
out_dir.mkdir(parents=True, exist_ok=True)
vis_dir = root / "masks_semantic_vis"
vis_dir.mkdir(parents=True, exist_ok=True)

classes = set()
for ann in ann_dir.glob("*.json"):
    data = json.loads(ann.read_text())
    for obj in data.get("objects", []):
        cls = obj.get("classTitle")
        if cls is not None:
            classes.add(cls)

def sort_key(x):
    try:
        return (0, int(x))
    except Exception:
        return (1, str(x))
    
sorted_classes = sorted(classes, key = sort_key)
class_map = {cls: i + 1 for i, cls in enumerate(sorted_classes)}

(out_dir / "class_map.txt").write_text(
    "\n".join(f"{k}\t{v}" for k, v in class_map.items()) + "\n"
)

def make_palette(size = 256):
    palette = [0] * (size * 3)
    for i in range(size):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        palette[i * 3 + 0] = r
        palette[i * 3 + 1] = g
        palette[i * 3 + 2] = b
    return palette

palette = make_palette()

for ann in ann_dir.glob("*.json"):
    name = ann.name
    stem = name[:-9] if name.endswith(".jpg.json") else ann.stem
    data = json.loads(ann.read_text())
    size = data["size"]
    w, h = int(size["width"]), int(size["height"])
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for obj in data.get("objects", []):
        cls = obj.get("classTitle")
        if cls not in class_map:
            continue
        pts = obj.get("points", {}).get("exterior", [])
        if len(pts) < 3:
            continue
        poly = [(float(x), float(y)) for x, y in pts]
        draw.polygon(poly, fill = class_map[cls])

        for hole in obj.get("points", {}).get("interior", []) or []:
            if len(hole) >= 3:
                hpoly = [(float(x), float(y)) for x, y in hole]
                draw.polygon(hpoly, fill = 0)

    mask.save(out_dir / f"{stem}.png")
    vis = mask.convert("P")
    vis.putpalette(palette)
    vis.save(vis_dir / f"{stem}.png")
