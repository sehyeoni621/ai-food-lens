"""
모델 초기화 확인 스크립트
실행: python download_model.py

파인튜닝 모델(food_finetuned.pth)이 없으면
ImageNet 사전학습 모드로 동작함을 안내합니다.
"""
import torch
import torchvision.models as models
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
FINETUNED_PATH = MODEL_DIR / "food_finetuned.pth"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    if FINETUNED_PATH.exists():
        print(f"✅ 파인튜닝 모델 존재: {FINETUNED_PATH}")
        ckpt = torch.load(FINETUNED_PATH, map_location="cpu")
        labels = ckpt.get("labels", [])
        print(f"   인식 가능 음식: {len(labels)}종 → {labels[:10]} ...")
        return

    print("=" * 60)
    print("⚠️  파인튜닝 모델 없음 — ImageNet 사전학습 모드로 동작")
    print("=" * 60)
    print()
    print("현재 인식 가능한 음식 (ImageNet 기반, 약 70종):")
    from model import IMAGENET_FOOD_MAP
    names = list(IMAGENET_FOOD_MAP.values())
    for i in range(0, len(names), 5):
        print(" ", " / ".join(names[i:i+5]))

    print()
    print("한식 인식을 위한 파인튜닝 방법:")
    print()
    print("  1. AI Hub 한국 음식 이미지 데이터셋 다운로드 (무료)")
    print("     → https://www.aihub.or.kr (회원가입 필요)")
    print("     → '음식 이미지' 검색 → 다운로드")
    print()
    print("  2. 직접 데이터 수집 (클래스당 최소 20장 권장)")
    print("     data/")
    print("       비빔밥/  (비빔밥 사진 20장+)")
    print("       김치찌개/")
    print("       떡볶이/")
    print("       ...")
    print()
    print("  3. 파인튜닝 실행:")
    print("     python train.py --data_dir ./data --epochs 30")
    print()
    print("파인튜닝 없이도 서버는 즉시 실행 가능합니다.")

    # ImageNet 모델 미리 다운로드 (캐시)
    print("\nImageNet 가중치 사전 다운로드 중...")
    _ = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    print("✅ ImageNet 가중치 다운로드 완료 (~21MB)")


if __name__ == "__main__":
    main()
