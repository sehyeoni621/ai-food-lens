"""
단일 이미지 추론 테스트
========================
사용법:
  python finetune/infer_test.py --image ./test_food.jpg
  python finetune/infer_test.py --image ./test_food.jpg --top_k 5
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import get_engine

TRANSFORM = transforms.Compose([
    transforms.Resize(345),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main(args):
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ 이미지 없음: {img_path}")
        return

    print(f"이미지: {img_path.name}")
    image = Image.open(img_path).convert("RGB")
    results = get_engine().predict(image, top_k=args.top_k)

    print(f"\n{'='*40}")
    print(f"  AI 렌즈 추론 결과")
    print(f"{'='*40}")
    for i, r in enumerate(results, 1):
        conf  = r["confidence"]
        bar   = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
        warn  = " ⚠️ 낮은 신뢰도" if r.get("low_confidence") else ""
        print(f"  #{i}  {r['food_name_ko']:15s}  [{bar}] {conf:5.1f}%{warn}")

    top = results[0]
    print(f"\n최종 인식: {top['food_name_ko']} ({top['confidence']:.1f}%)")

    # 영양 정보도 출력
    from nutrition_db import get_nutrition
    nutrition = get_nutrition(top["food_name_ko"], top["food_key"])
    if nutrition:
        print(f"\n영양성분 ({nutrition.get('serving_size','1인분')}):")
        print(f"  칼로리: {nutrition['kcal']} kcal")
        print(f"  탄수화물: {nutrition['carbohydrate']}g  단백질: {nutrition['protein']}g  지방: {nutrition['fat']}g")
        print(f"  당류: {nutrition['sugar']}g  나트륨: {nutrition['sodium']}mg")
    else:
        print("  (영양 DB에 해당 음식 없음 — nutrition_db.py에 추가하세요)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="테스트할 이미지 파일 경로")
    p.add_argument("--top_k", type=int, default=3)
    main(p.parse_args())
