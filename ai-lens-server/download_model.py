"""
파인튜닝 모델 가중치 다운로드 스크립트
실행: python download_model.py
"""
import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "food_classifier.pth"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        print(f"✅ 모델 이미 존재: {MODEL_PATH}")
        return

    print("⚡ 초기 모델 생성 중 (ImageNet 사전학습 MobileNetV3-Small)...")
    print("   → 파인튜닝하려면 train.py를 실행하세요")

    import torch
    import torch.nn as nn
    import torchvision.models as models
    from model import FOOD101_CLASSES

    num_classes = len(FOOD101_CLASSES)
    base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)

    torch.save(base.state_dict(), MODEL_PATH)
    print(f"✅ 모델 저장 완료: {MODEL_PATH}")
    print(f"   클래스 수: {num_classes}")
    print("   ⚠️  ImageNet 가중치만 있으므로 음식 인식 정확도를 높이려면 train.py로 파인튜닝하세요")


if __name__ == "__main__":
    main()
