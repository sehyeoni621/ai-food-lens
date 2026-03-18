"""
AI Hub 한국 음식 데이터셋 전처리 스크립트
=========================================
대상 데이터셋: AI Hub 「음식 이미지 및 영양정보 텍스트」
URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=74

AI Hub 다운로드 절차:
  1. https://www.aihub.or.kr 회원가입 (무료)
  2. 검색창에 "음식 이미지" 검색 → 신청 (즉시 승인)
  3. 압축 해제 → 이 스크립트 실행

사용법:
  python prepare_aihub_data.py --src_dir "C:/Downloads/AIHub_Food" --out_dir "../data/korean_food"

AI Hub 폴더 구조 예시 (여러 버전 자동 감지):
  버전 A (카테고리 폴더):
    Training/원천데이터/비빔밥/*.jpg
    Validation/원천데이터/비빔밥/*.jpg

  버전 B (라벨링 포함):
    Training/원천데이터/TRA_00001.jpg
    Training/라벨링데이터/TRA_00001.json
"""

import os
import json
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

# AI Hub 음식 데이터셋 카테고리 (한국어 이름 정규화)
# AI Hub는 숫자 코드를 쓰는 경우가 있어서 여기서 매핑
AIHUB_CATEGORY_MAP = {
    # AI Hub 코드 or 폴더명 → 정규화된 한국어 이름
    "001": "갈비구이", "갈비구이": "갈비구이",
    "002": "갈비찜", "갈비찜": "갈비찜",
    "003": "갈비탕", "갈비탕": "갈비탕",
    "004": "감자국", "감자국": "감자국",
    "005": "감자볶음", "감자볶음": "감자볶음",
    "006": "감자조림", "감자조림": "감자조림",
    "007": "고등어구이", "고등어구이": "고등어구이",
    "008": "고등어조림", "고등어조림": "고등어조림",
    "009": "계란국", "계란국": "계란국",
    "010": "계란말이", "계란말이": "계란말이",
    "011": "계란볶음밥", "계란볶음밥": "계란볶음밥",
    "012": "계란후라이", "계란후라이": "계란후라이",
    "013": "곰탕", "곰탕": "곰탕",
    "014": "깍두기", "깍두기": "깍두기",
    "015": "김밥", "김밥": "김밥",
    "016": "김치", "김치": "김치",
    "017": "김치볶음밥", "김치볶음밥": "김치볶음밥",
    "018": "김치전", "김치전": "김치전",
    "019": "김치찌개", "김치찌개": "김치찌개",
    "020": "나물", "나물": "나물",
    "021": "냉면", "냉면": "냉면",
    "022": "닭갈비", "닭갈비": "닭갈비",
    "023": "닭볶음탕", "닭볶음탕": "닭볶음탕",
    "024": "닭죽", "닭죽": "닭죽",
    "025": "된장국", "된장국": "된장국",
    "026": "된장찌개", "된장찌개": "된장찌개",
    "027": "두부조림", "두부조림": "두부조림",
    "028": "떡국", "떡국": "떡국",
    "029": "떡볶이", "떡볶이": "떡볶이",
    "030": "라면", "라면": "라면",
    "031": "만두", "만두": "만두",
    "032": "미역국", "미역국": "미역국",
    "033": "백반", "백반": "백반",
    "034": "보쌈", "보쌈": "보쌈",
    "035": "불고기", "불고기": "불고기",
    "036": "비빔국수", "비빔국수": "비빔국수",
    "037": "비빔밥", "비빔밥": "비빔밥",
    "038": "삼겹살", "삼겹살": "삼겹살",
    "039": "삼계탕", "삼계탕": "삼계탕",
    "040": "새우볶음밥", "새우볶음밥": "새우볶음밥",
    "041": "새우튀김", "새우튀김": "새우튀김",
    "042": "설렁탕", "설렁탕": "설렁탕",
    "043": "소불고기", "소불고기": "소불고기",
    "044": "수제비", "수제비": "수제비",
    "045": "순두부찌개", "순두부찌개": "순두부찌개",
    "046": "순대", "순대": "순대",
    "047": "시금치나물", "시금치나물": "시금치나물",
    "048": "잡채", "잡채": "잡채",
    "049": "잡탕", "잡탕": "잡탕",
    "050": "제육볶음", "제육볶음": "제육볶음",
    "051": "조기구이", "조기구이": "조기구이",
    "052": "찜닭", "찜닭": "찜닭",
    "053": "참치김밥", "참치김밥": "참치김밥",
    "054": "치킨", "치킨": "치킨",
    "055": "콩나물국", "콩나물국": "콩나물국",
    "056": "콩나물볶음", "콩나물볶음": "콩나물볶음",
    "057": "파전", "파전": "파전",
    "058": "팥죽", "팥죽": "팥죽",
    "059": "편육", "편육": "편육",
    "060": "해물파전", "해물파전": "해물파전",
    "061": "해물찜", "해물찜": "해물찜",
    "062": "해물탕", "해물탕": "해물탕",
    "063": "호박전", "호박전": "호박전",
    "064": "황태국", "황태국": "황태국",
}


def detect_dataset_format(src_dir: Path) -> str:
    """AI Hub 폴더 구조 자동 감지"""
    # 버전 A: Training/원천데이터/{카테고리}/*.jpg
    if (src_dir / "Training" / "원천데이터").exists():
        return "aihub_v1"
    # 버전 B: Training/원천데이터/*.jpg + Training/라벨링데이터/*.json
    if (src_dir / "Training" / "라벨링데이터").exists():
        return "aihub_v2"
    # 단순 ImageFolder: {카테고리}/*.jpg
    subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
    if subdirs and any((d / "train").exists() or len(list(d.glob("*.jpg"))) > 0 for d in subdirs[:3]):
        return "imagefolder"
    # train/val 분리 포함
    if (src_dir / "train").exists() and (src_dir / "val").exists():
        return "split_imagefolder"
    return "unknown"


def parse_aihub_v1(src_dir: Path) -> dict[str, list[Path]]:
    """버전 A: Training/원천데이터/{카테고리}/*.jpg"""
    data: dict[str, list[Path]] = defaultdict(list)
    for split in ["Training", "Validation", "Training/원천데이터", "Validation/원천데이터"]:
        split_dir = src_dir / split
        if not split_dir.exists():
            continue
        for cat_dir in split_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            cat_name = normalize_category(cat_dir.name)
            if not cat_name:
                continue
            imgs = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png"))
            data[cat_name].extend(imgs)
    return dict(data)


def parse_aihub_v2(src_dir: Path) -> dict[str, list[Path]]:
    """버전 B: JSON 라벨 파일로 카테고리 매핑"""
    data: dict[str, list[Path]] = defaultdict(list)
    for split in ["Training", "Validation"]:
        label_dir = src_dir / split / "라벨링데이터"
        img_dir = src_dir / split / "원천데이터"
        if not label_dir.exists():
            continue
        for json_file in label_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    meta = json.load(f)
                food_name = meta.get("food_name") or meta.get("category") or meta.get("label", "")
                food_name = normalize_category(food_name)
                if not food_name:
                    continue
                img_name = json_file.stem + ".jpg"
                img_path = img_dir / img_name
                if not img_path.exists():
                    for ext in [".jpeg", ".png", ".JPG"]:
                        img_path = img_dir / (json_file.stem + ext)
                        if img_path.exists():
                            break
                if img_path.exists():
                    data[food_name].append(img_path)
            except Exception:
                continue
    return dict(data)


def parse_imagefolder(src_dir: Path) -> dict[str, list[Path]]:
    """단순 ImageFolder 구조"""
    data: dict[str, list[Path]] = defaultdict(list)
    for cat_dir in src_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        cat_name = normalize_category(cat_dir.name)
        imgs = list(cat_dir.glob("**/*.jpg")) + list(cat_dir.glob("**/*.jpeg")) + list(cat_dir.glob("**/*.png"))
        if imgs:
            data[cat_name or cat_dir.name].extend(imgs)
    return dict(data)


def normalize_category(name: str) -> str:
    """카테고리 이름 정규화 (숫자 코드 → 한국어, 공백 제거 등)"""
    name = name.strip()
    if name in AIHUB_CATEGORY_MAP:
        return AIHUB_CATEGORY_MAP[name]
    # 숫자로 시작하는 경우: "001_비빔밥" → "비빔밥"
    if "_" in name:
        parts = name.split("_", 1)
        if parts[0].isdigit():
            return parts[1].strip()
    return name


def build_dataset(data: dict[str, list[Path]], out_dir: Path,
                   val_ratio: float = 0.15, min_per_class: int = 10):
    """ImageFolder 구조로 저장 (train/val 분리)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(exist_ok=True)
    (out_dir / "val").mkdir(exist_ok=True)

    stats = {}
    skipped = []
    for cat_name, img_paths in sorted(data.items()):
        if len(img_paths) < min_per_class:
            skipped.append(f"{cat_name} ({len(img_paths)}장, 최소 {min_per_class}장 필요)")
            continue

        random.shuffle(img_paths)
        n_val = max(1, int(len(img_paths) * val_ratio))
        val_imgs = img_paths[:n_val]
        train_imgs = img_paths[n_val:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest_dir = out_dir / split / cat_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(imgs):
                ext = src.suffix.lower()
                dest = dest_dir / f"{i:05d}{ext}"
                shutil.copy2(src, dest)

        stats[cat_name] = {"total": len(img_paths), "train": len(train_imgs), "val": n_val}

    # 통계 저장
    report = {
        "total_classes": len(stats),
        "total_images": sum(v["total"] for v in stats.values()),
        "classes": stats,
        "skipped": skipped,
    }
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="AI Hub 한국 음식 데이터셋 전처리")
    parser.add_argument("--src_dir", required=True, help="AI Hub 압축 해제 폴더")
    parser.add_argument("--out_dir", default="../data/korean_food", help="출력 폴더")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="검증셋 비율 (기본 15%)")
    parser.add_argument("--min_images", type=int, default=20, help="클래스당 최소 이미지 수")
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)

    print(f"소스 경로: {src}")
    fmt = detect_dataset_format(src)
    print(f"데이터셋 형식 감지: {fmt}")

    if fmt == "aihub_v1":
        data = parse_aihub_v1(src)
    elif fmt == "aihub_v2":
        data = parse_aihub_v2(src)
    elif fmt in ("imagefolder", "split_imagefolder"):
        data = parse_imagefolder(src)
    else:
        print("❌ 알 수 없는 폴더 구조입니다.")
        print("   지원 형식:")
        print("   - Training/원천데이터/{카테고리}/*.jpg")
        print("   - Training/라벨링데이터/*.json + Training/원천데이터/*.jpg")
        print("   - {카테고리}/*.jpg")
        return

    print(f"\n발견된 카테고리: {len(data)}종")
    total_imgs = sum(len(v) for v in data.values())
    print(f"전체 이미지: {total_imgs:,}장")
    print("\n데이터셋 구성 중...")

    report = build_dataset(data, out, args.val_ratio, args.min_images)

    print("\n" + "="*50)
    print(f"✅ 완료!")
    print(f"   학습용 클래스: {report['total_classes']}종")
    print(f"   전체 이미지:   {report['total_images']:,}장")
    print(f"   저장 위치:     {out}")
    if report["skipped"]:
        print(f"\n⚠️  제외된 클래스 (이미지 부족):")
        for s in report["skipped"]:
            print(f"   - {s}")
    print(f"\n다음 단계: python ../train.py --data_dir {out}/train --val_dir {out}/val")


if __name__ == "__main__":
    main()
