"""
파인튜닝 모델 성능 평가 스크립트
================================
사용법:
  python finetune/evaluate.py --val_dir ./data/korean_food/val
  python finetune/evaluate.py --val_dir ./data/korean_food/val --top_k 5
"""
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn

MODEL_PATH = Path(__file__).parent.parent / "models" / "food_finetuned.pth"


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}\n먼저 train.py를 실행하세요.")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    labels    = ckpt["labels"]
    backbone  = ckpt.get("backbone", "mobilenet_v3_large")
    num_classes = len(labels)

    if backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, num_classes))
    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, num_classes))
    else:
        raise ValueError(f"알 수 없는 백본: {backbone}")

    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels = load_model(device)
    print(f"모델 로드 완료 | 클래스: {len(labels)}종 | Device: {device}")

    val_tf = transforms.Compose([
        transforms.Resize(345),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.val_dir, transform=val_tf)
    loader  = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    top1 = top5 = total = 0
    per_class_correct = {i: 0 for i in range(len(labels))}
    per_class_total   = {i: 0 for i in range(len(labels))}

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            # Top-1
            pred1 = out.argmax(1)
            top1 += (pred1 == lbls).sum().item()
            # Top-5
            _, pred5 = out.topk(min(5, len(labels)), dim=1)
            top5 += sum(lbls[i].item() in pred5[i].tolist() for i in range(len(lbls)))
            total += lbls.size(0)
            for lbl, pred in zip(lbls.tolist(), pred1.tolist()):
                per_class_total[lbl] += 1
                if lbl == pred:
                    per_class_correct[lbl] += 1

    print(f"\n{'='*45}")
    print(f"  전체 이미지: {total:,}장")
    print(f"  Top-1 정확도: {top1/total*100:.2f}%")
    print(f"  Top-5 정확도: {top5/total*100:.2f}%")
    print(f"{'='*45}")

    # 클래스별 정확도 (낮은 순 상위 10개)
    print("\n▼ 정확도 낮은 상위 10개 클래스 (개선 필요):")
    cls_acc = []
    for i, name in enumerate(labels):
        t = per_class_total[i]
        if t == 0:
            continue
        acc = per_class_correct[i] / t * 100
        cls_acc.append((name, acc, t))
    cls_acc.sort(key=lambda x: x[1])
    for name, acc, cnt in cls_acc[:10]:
        print(f"  {name:15s}  {acc:5.1f}%  ({cnt}장)")

    print("\n▲ 정확도 높은 상위 10개 클래스:")
    for name, acc, cnt in cls_acc[-10:][::-1]:
        print(f"  {name:15s}  {acc:5.1f}%  ({cnt}장)")

    # JSON 저장
    out_path = MODEL_PATH.parent / "eval_result.json"
    result = {
        "top1_acc": round(top1/total*100, 2),
        "top5_acc": round(top5/total*100, 2),
        "total_images": total,
        "per_class": {name: round(per_class_correct[i]/max(per_class_total[i],1)*100, 1)
                      for i, name in enumerate(labels)},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    main(p.parse_args())
