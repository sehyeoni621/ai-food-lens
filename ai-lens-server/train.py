"""
음식 분류 모델 파인튜닝 스크립트
Food-101 데이터셋으로 MobileNetV3-Small 파인튜닝

사용법:
  python train.py --data_dir ./food-101/images --epochs 20 --batch_size 32

Food-101 다운로드:
  wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
  tar -xzf food-101.tar.gz
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import FoodClassifier, MODEL_PATH, MODEL_DIR


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")

    train_tf, val_tf = get_transforms()
    dataset = datasets.ImageFolder(root=args.data_dir, transform=train_tf)
    num_classes = len(dataset.classes)
    print(f"클래스 수: {num_classes}, 전체 이미지: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = FoodClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{args.epochs} Step {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        # 검증
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch}/{args.epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.model.state_dict(), MODEL_PATH)
            print(f"  ✅ 최고 정확도 모델 저장: {MODEL_PATH} (Acc={acc:.2f}%)")

    print(f"\n학습 완료! 최고 검증 정확도: {best_acc:.2f}%")
    print(f"모델 저장 위치: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="이미지 데이터셋 경로 (ImageFolder 구조)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
