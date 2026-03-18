# AI Hub 한식 파인튜닝 가이드

## 전체 흐름

```
[1] AI Hub 데이터셋 다운로드  →  [2] 데이터 전처리  →  [3] 파인튜닝  →  [4] 평가  →  [5] 서버 적용
```

---

## Step 1. AI Hub 데이터셋 다운로드

1. **https://www.aihub.or.kr** 접속 → 회원가입 (무료)
2. 검색창에 **"음식 이미지"** 검색
3. **「음식 이미지 및 영양정보 텍스트」** 클릭 → 활용신청 (즉시 승인)
4. 다운로드 → 압축 해제

> 제공 규모: 한식 약 150종 / 총 200,000장 이상

---

## Step 2. 환경 설치

```bash
# Windows
finetune\setup_env.bat

# Linux / Mac / WSL
bash finetune/setup_env.sh

# 환경 활성화
conda activate ai-lens
```

---

## Step 3. 데이터 전처리

```bash
cd ai-lens-server

python finetune/prepare_aihub_data.py \
  --src_dir "C:/Downloads/AIHub_Food" \
  --out_dir ./data/korean_food \
  --min_images 20
```

완료 후 폴더 구조:
```
data/korean_food/
  train/
    비빔밥/      (이미지 N장)
    김치찌개/
    떡볶이/
    ...
  val/
    비빔밥/
    ...
  dataset_info.json  ← 클래스별 이미지 수 확인 가능
```

---

## Step 4. 파인튜닝

```bash
# GPU 있는 경우 (권장, ~2-3시간)
python train.py \
  --train_dir ./data/korean_food/train \
  --val_dir   ./data/korean_food/val \
  --backbone  efficientnet_b3 \
  --epochs    40

# CPU만 있는 경우 (매우 느림 → Colab 권장)
python train.py \
  --train_dir ./data/korean_food/train \
  --val_dir   ./data/korean_food/val \
  --backbone  mobilenet_v3_large \
  --epochs    20 \
  --batch_size 16
```

| 백본 | 예상 정확도 | 속도 | 모델 크기 |
|------|------------|------|---------|
| efficientnet_b3 (권장) | ~85% | 보통 | ~50MB |
| mobilenet_v3_large | ~78% | 빠름 | ~20MB |
| resnet50 | ~80% | 느림 | ~100MB |

---

## Step 5. 평가

```bash
python finetune/evaluate.py \
  --val_dir ./data/korean_food/val
```

출력 예시:
```
Top-1 정확도: 84.32%
Top-5 정확도: 96.51%
정확도 낮은 클래스: 나물(41%), 잡탕(55%) ...
```

---

## Step 6. 단일 이미지 테스트

```bash
python finetune/infer_test.py --image ./test_food.jpg
```

---

## Step 7. 서버 적용

파인튜닝이 완료되면 **서버를 재시작**하기만 하면 자동 적용됩니다.
`models/food_finetuned.pth` 파일이 있으면 서버가 파인튜닝 모드로 부팅합니다.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# 로그에 [AI 렌즈] 파인튜닝 모드 | 백본=efficientnet_b3 | 클래스=XX 표시
```

---

## Google Colab으로 학습하기 (GPU 없는 경우)

1. Colab 접속 → 런타임 → GPU 선택
2. AI Hub 데이터를 Google Drive에 업로드
3. Colab에서:
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/sehyeoni621/ai-food-lens
%cd ai-food-lens/ai-lens-server

!pip install torch torchvision

!python finetune/prepare_aihub_data.py \
    --src_dir "/content/drive/MyDrive/AIHub_Food" \
    --out_dir ./data/korean_food

!python train.py \
    --train_dir ./data/korean_food/train \
    --val_dir   ./data/korean_food/val \
    --backbone  efficientnet_b3 \
    --epochs    40
```
4. `models/food_finetuned.pth` → Drive에서 다운로드 → `ai-lens-server/models/` 에 복사
