"""
YOLOSaphire Inference
==================
Run detection on images or video.

Usage:
    python predict.py --weights runs/train/best.pt --source image.jpg
    python predict.py --weights runs/train/best.pt --source images/
    python predict.py --weights runs/train/best.pt --source video.mp4
"""

import argparse
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

from model import YOLOSaphire, yolosaphire_medium


def load_model(weights_path: str, device: str) -> YOLOSaphire:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        nc      = ckpt["num_classes"]
        variant = ckpt.get("model_variant", "medium")
        from model import yolosaphire_nano, yolosaphire_small, yolosaphire_medium, yolosaphire_large
        build = {"nano": yolosaphire_nano, "small": yolosaphire_small,
                 "medium": yolosaphire_medium, "large": yolosaphire_large}[variant]
        model = build(nc).to(device)
        model.load_state_dict(ckpt["model_state"])
    else:
        # bare state dict — default to medium, 80 classes
        model = yolosaphire_medium(80).to(device)
        model.load_state_dict(ckpt)
    return model.eval()


def preprocess(img_path: str, img_size: int = 640):
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size  # (W, H)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor, img, orig_size


def decode_predictions(preds, img_size: int, orig_size, conf_thresh: float, num_classes: int):
    """Decode raw head outputs into bounding boxes."""
    all_boxes, all_scores, all_cls = [], [], []
    strides = [8, 16, 32]

    for pred, stride in zip(preds, strides):
        # pred: (1, 5+nc, H, W)
        B, C, H, W = pred.shape
        pred = pred[0].permute(1, 2, 0)  # (H, W, 5+nc)

        boxes_raw = pred[..., :4]
        obj       = pred[..., 4].sigmoid()
        cls_logit = pred[..., 5:].sigmoid()

        # Build grid
        gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        gx = gx.float().to(pred.device)
        gy = gy.float().to(pred.device)

        # Decode cx, cy, w, h
        cx = (boxes_raw[..., 0].sigmoid() + gx) * stride
        cy = (boxes_raw[..., 1].sigmoid() + gy) * stride
        bw = boxes_raw[..., 2].exp() * stride
        bh = boxes_raw[..., 3].exp() * stride

        # Scale to original image
        sx = orig_size[0] / img_size
        sy = orig_size[1] / img_size
        x1 = ((cx - bw / 2) * sx).reshape(-1)
        y1 = ((cy - bh / 2) * sy).reshape(-1)
        x2 = ((cx + bw / 2) * sx).reshape(-1)
        y2 = ((cy + bh / 2) * sy).reshape(-1)

        scores, cls_ids = (obj.reshape(-1, 1) * cls_logit.reshape(-1, num_classes)).max(dim=-1)
        mask = scores > conf_thresh

        all_boxes.append(torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], dim=-1))
        all_scores.append(scores[mask])
        all_cls.append(cls_ids[mask])

    if not any(len(b) for b in all_boxes):
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)

    boxes  = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    cls    = torch.cat(all_cls)

    # NMS per class
    keep = ops.batched_nms(boxes.float(), scores.float(), cls, iou_threshold=0.45)
    return boxes[keep], scores[keep], cls[keep]


COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

def draw_boxes(img: Image.Image, boxes, scores, cls_ids, class_names):
    draw = ImageDraw.Draw(img)
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = box.tolist()
        cls_id = int(cls_id)
        color  = COLORS[cls_id % len(COLORS)]
        label  = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), label, fill="white")
    return img


@torch.no_grad()
def predict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.weights, device)
    nc     = model.num_classes
    names  = [str(i) for i in range(nc)]

    # Load class names if available
    if args.names:
        with open(args.names) as f:
            names = [l.strip() for l in f.readlines()]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = list(Path(args.source).glob("*")) if Path(args.source).is_dir() \
              else [Path(args.source)]
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    sources  = [s for s in sources if s.suffix.lower() in img_exts]

    print(f"\nRunning inference on {len(sources)} image(s)...\n")

    for src in sources:
        tensor, orig_img, orig_size = preprocess(str(src), args.imgsz)
        tensor = tensor.to(device)

        preds = model(tensor)
        boxes, scores, cls_ids = decode_predictions(
            preds, args.imgsz, orig_size, args.conf, nc
        )

        result_img = draw_boxes(orig_img.copy(), boxes, scores, cls_ids, names)
        save_path  = out_dir / src.name
        result_img.save(str(save_path))

        print(f"  {src.name} — {len(boxes)} detections → {save_path}")

    print(f"\n✓ Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOSaphire Inference")
    parser.add_argument("--weights", type=str, required=True,       help="Path to .pt weights")
    parser.add_argument("--source",  type=str, required=True,       help="Image or directory")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--conf",    type=float, default=0.25,      help="Confidence threshold")
    parser.add_argument("--names",   type=str, default=None,        help="Path to class names .txt")
    parser.add_argument("--output",  type=str, default="runs/predict")
    args = parser.parse_args()
    predict(args)
