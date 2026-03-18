"""
한국 음식 분류 모델 파인튜닝 (AI Hub 데이터셋용)
================================================
백본: EfficientNet-B3 (MobileNetV3보다 음식 인식 정확도 높음)
전략: 2단계 학습 (헤드만 → 전체 레이어)

사용법:
  # AI Hub 데이터 전처리 먼저
  python finetune/prepare_aihub_data.py --src_dir "C:/Downloads/AIHub_Food" --out_dir ./data/korean_food

  # 파인튜닝 실행
  python train.py --train_dir ./data/korean_food/train --val_dir ./data/korean_food/val

  # 학습 중 실시간 로그 확인
  python train.py --train_dir ./data/korean_food/train --val_dir ./data/korean_food/val --epochs 40
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# ─── 경로 설정 ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
SAVE_PATH = MODEL_DIR / "food_finetuned.pth"


# ─── 데이터 증강 ────────────────────────────────────────────────────────────
def get_transforms(img_size: int = 300):
    """음식 이미지에 최적화된 데이터 증강"""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),        # 흑백 사진 대응
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # 랜덤 가림 (과적합 방지)
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ─── 모델 빌드 ──────────────────────────────────────────────────────────────
def build_model(num_classes: int, backbone: str = "efficientnet_b3") -> nn.Module:
    """
    백본별 분류 헤드 교체
    efficientnet_b3 : 정확도 우수, 메모리 ~50MB (기본값)
    mobilenet_v3_large: 속도 빠름, 메모리 ~20MB
    resnet50        : 안정적, 메모리 ~100MB
    """
    if backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_feat, num_classes),
        )
    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_feat, num_classes),
        )
    else:
        raise ValueError(f"지원하지 않는 백본: {backbone}")
    return model


def freeze_backbone(model: nn.Module, backbone: str):
    """백본 파라미터 동결 (분류 헤드만 학습)"""
    if backbone.startswith("efficientnet"):
        for param in model.features.parameters():
            param.requires_grad = False
    elif backbone == "mobilenet_v3_large":
        for param in model.features.parameters():
            param.requires_grad = False
    elif backbone == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ─── 학습 루프 ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = correct = total = 0
    t0 = time.time()
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            print(f"  [{epoch}/{total_epochs}] Step {i+1}/{len(loader)} "
                  f"| Loss {total_loss/(i+1):.4f} | Acc {correct/total*100:.1f}% "
                  f"| {elapsed:.0f}s", flush=True)

    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total * 100


def save_checkpoint(model, labels: list[str], backbone: str, acc: float):
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels":     labels,
        "num_classes": len(labels),
        "backbone":   backbone,
        "val_acc":    acc,
    }, SAVE_PATH)
    print(f"  💾 저장 (val_acc={acc:.2f}%): {SAVE_PATH}")


# ─── 메인 ───────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  건강신호등 AI 렌즈 파인튜닝")
    print(f"  백본: {args.backbone}  |  Device: {device}")
    print(f"{'='*55}\n")

    if device.type == "cpu":
        print("⚠️  GPU 없음 — CPU 학습은 매우 느립니다 (수 시간 소요).")
        print("   Google Colab 사용을 권장합니다.\n")

    # 데이터셋 로드
    train_tf, val_tf = get_transforms(args.img_size)
    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(args.val_dir,   transform=val_tf)

    labels = train_ds.classes          # 한국어 폴더명 = 레이블
    num_classes = len(labels)
    print(f"클래스({num_classes}종): {labels}")
    print(f"학습 이미지: {len(train_ds):,}장  |  검증: {len(val_ds):,}장\n")

    # 클래스 불균형 대응 (WeightedRandomSampler)
    class_counts = [0] * num_classes
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    weights = [1.0 / class_counts[lbl] for _, lbl in train_ds.samples]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=sampler, num_workers=args.workers,
                               pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.workers)

    # 모델
    model = build_model(num_classes, args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0

    # ── 1단계: 분류 헤드만 학습 (빠른 수렴) ──────────────────────────────────
    warmup_epochs = min(args.warmup_epochs, args.epochs // 3)
    if warmup_epochs > 0:
        print(f"── 1단계: 헤드만 학습 ({warmup_epochs} 에포크) ──")
        freeze_backbone(model, args.backbone)
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader), epochs=warmup_epochs,
        )

        for epoch in range(1, warmup_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup_epochs)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"  ▶ Epoch {epoch}/{warmup_epochs} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc:.1f}% | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.1f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, labels, args.backbone, best_acc)

    # ── 2단계: 전체 레이어 파인튜닝 ──────────────────────────────────────────
    remaining = args.epochs - warmup_epochs
    if remaining > 0:
        print(f"\n── 2단계: 전체 레이어 파인튜닝 ({remaining} 에포크) ──")
        unfreeze_all(model)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(remaining // 3, 5), T_mult=1,
        )

        for epoch in range(warmup_epochs + 1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"  ▶ Epoch {epoch}/{args.epochs} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc:.1f}% | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.1f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, labels, args.backbone, best_acc)

    print(f"\n{'='*55}")
    print(f"✅ 파인튜닝 완료!")
    print(f"   최고 검증 정확도: {best_acc:.2f}%")
    print(f"   모델 저장: {SAVE_PATH}")
    print(f"   서버 재시작 후 자동 적용됩니다.")
    print(f"{'='*55}\n")

    # 레이블 저장 (nutrition_db.py 자동 업데이트용)
    labels_out = MODEL_DIR / "trained_labels.json"
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"   레이블 목록 저장: {labels_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="한국 음식 AI 파인튜닝")
    p.add_argument("--train_dir", required=True, help="학습 데이터 폴더 (ImageFolder 구조)")
    p.add_argument("--val_dir",   required=True, help="검증 데이터 폴더 (ImageFolder 구조)")
    p.add_argument("--backbone",  default="efficientnet_b3",
                   choices=["efficientnet_b3", "mobilenet_v3_large", "resnet50"])
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--warmup_epochs",type=int,   default=8,  help="1단계(헤드만) 에포크 수")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--img_size",     type=int,   default=300)
    p.add_argument("--workers",      type=int,   default=4)
    main(p.parse_args())
