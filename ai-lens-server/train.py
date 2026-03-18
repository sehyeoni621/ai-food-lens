"""
음식 분류 모델 파인튜닝 스크립트

지원 데이터셋:
  1. Food-101 (서양 음식 101종, 75,750장)
     wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

  2. AI Hub 한국 음식 이미지 (한식 150종, 무료)
     https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=74

  3. 직접 수집 데이터
     data/
       김치찌개/  (이미지 20장 이상 권장)
       비빔밥/
       떡볶이/
       ...

사용법:
  python train.py --data_dir ./data --epochs 30 --batch_size 32

결과:
  models/food_finetuned.pth 저장
  (state_dict + labels + num_classes 포함)
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

MODEL_DIR = Path(__file__).parent / "models"
SAVE_PATH = MODEL_DIR / "food_finetuned.pth"


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")
    if device.type == "cpu":
        print("⚠️  GPU 없음 — CPU로 학습 시 매우 느립니다. GPU 권장.")

    train_tf, val_tf = get_transforms()
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_tf)
    labels_ko = full_dataset.classes  # 폴더 이름 = 음식 이름 (한국어 가능)
    num_classes = len(labels_ko)
    print(f"클래스({num_classes}): {labels_ko}")
    print(f"전체 이미지: {len(full_dataset)}")

    if len(full_dataset) < num_classes * 10:
        print(f"⚠️  클래스당 이미지가 너무 적습니다. 최소 {num_classes * 20}장 권장.")

    val_size = max(1, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # MobileNetV3-Large (ImageNet 사전학습) → 분류 헤드만 교체
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # 처음엔 백본 동결, 헤드만 학습 (빠른 수렴)
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    print("\n── 1단계: 분류 헤드만 학습 (5 에포크) ──")
    for epoch in range(1, min(6, args.epochs + 1)):
        _train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        acc = _validate(model, val_loader, device, epoch)
        if acc > best_acc:
            best_acc = acc
            _save(model, labels_ko, num_classes)

    # 백본 언프리즈 → 전체 파인튜닝
    if args.epochs > 5:
        print("\n── 2단계: 전체 레이어 파인튜닝 ──")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)

        for epoch in range(6, args.epochs + 1):
            _train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
            acc = _validate(model, val_loader, device, epoch)
            scheduler.step()
            if acc > best_acc:
                best_acc = acc
                _save(model, labels_ko, num_classes)

    print(f"\n✅ 학습 완료! 최고 검증 정확도: {best_acc:.2f}%")
    print(f"   모델 저장: {SAVE_PATH}")


def _train_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    total_loss = 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch}/{total} | Loss: {total_loss/len(loader):.4f}")


def _validate(model, loader, device, epoch) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"  Val Acc: {acc:.2f}%")
    return acc


def _save(model, labels: list[str], num_classes: int):
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels": labels,
        "num_classes": num_classes,
    }, SAVE_PATH)
    print(f"  💾 저장: {SAVE_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="ImageFolder 형식 데이터셋 경로")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())
