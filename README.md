# 건강신호등 AI 렌즈 🤖

> 음식 사진을 촬영하면 자체 AI 모델이 음식을 인식하고 영양성분을 분석합니다.
> **외부 AI API(Google Vision, OpenAI 등) 미사용** — MobileNetV3 모델을 직접 운용합니다.

건강신호등 앱([Health Traffic Light](https://github.com/sehyeoni621/Heathl-Traffic)) 의 AI 렌즈 모듈입니다.

---

## 프로젝트 구조

```
C:\AI\
├── ai-lens-server/          # Python FastAPI 백엔드
│   ├── main.py              # API 서버 진입점
│   ├── model.py             # MobileNetV3 음식 인식 모델
│   ├── nutrition_db.py      # 한국 음식 영양성분 DB
│   ├── agents.py            # AI 에이전트 (추천/분석/알러지)
│   ├── train.py             # 모델 파인튜닝 스크립트
│   ├── download_model.py    # 초기 모델 생성
│   └── requirements.txt
└── mobile-integration/      # React Native 앱 통합 파일
    ├── AILensScreen.tsx     # AI 렌즈 화면
    └── aiLensService.ts     # API 서비스 레이어
```

---

## 백엔드 서버 실행

### 1. 환경 설정

```bash
cd ai-lens-server
pip install -r requirements.txt
cp .env.example .env
# .env 파일에서 NUTRITION_API_KEY 설정 (선택사항)
```

### 2. 초기 모델 생성

```bash
python download_model.py
```

### 3. 서버 시작

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 `http://localhost:8000` 에서 실행됩니다.
API 문서: `http://localhost:8000/docs`

---

## AI 에이전트 기능

| 에이전트 | 엔드포인트 | 설명 |
|---------|-----------|------|
| 음식 인식 | `POST /analyze` | 사진 → 음식명 + 영양성분 + 건강점수 |
| 알러지 체크 | `POST /agents/allergy-check` | 음식별 알러지 성분 경고 |
| 일일 영양 분석 | `POST /agents/daily-analysis` | 하루 섭취량 vs 권장량 비교 |
| 식사 추천 | `POST /agents/meal-recommend` | 남은 영양 예산 기반 추천 |
| 식습관 트렌드 | `POST /agents/trend-analysis` | 스캔 기록 분석 |
| 대체 음식 | `GET /agents/alternatives/{food}` | 건강한 대체 음식 추천 |

---

## 음식 분석 API 사용 예시

```bash
# 이미지 파일로 분석
curl -X POST http://localhost:8000/analyze \
  -F "file=@pizza.jpg" \
  -F "user_allergies=밀,우유"

# base64로 분석
curl -X POST http://localhost:8000/analyze \
  -F "image_base64=<base64_string>"
```

응답 예시:
```json
{
  "predictions": [
    {"food_key": "pizza", "food_name_ko": "피자", "confidence": 82.5},
    {"food_key": "bruschetta", "food_name_ko": "브루스케타", "confidence": 9.1}
  ],
  "top_food": "피자",
  "nutrition": {
    "kcal": 266, "carbohydrate": 33.0, "protein": 11.0,
    "fat": 10.0, "sugar": 3.5, "sodium": 640,
    "serving_size": "1조각(100g)"
  },
  "health": {"score": 71, "traffic_light": "yellow", "emoji": "🟡"},
  "alternatives": ["샐러드", "샌드위치"],
  "allergy": {
    "has_warning": true,
    "triggered_allergens": ["밀", "우유"],
    "message": "⚠️ 밀, 우유 알러지 주의!"
  }
}
```

---

## 모델 파인튜닝 (선택사항)

Food-101 데이터셋으로 정확도를 높일 수 있습니다:

```bash
# Food-101 다운로드
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz

# 파인튜닝 (GPU 권장)
python train.py --data_dir ./food-101/images --epochs 20 --batch_size 32
```

---

## 앱 통합

`mobile-integration/` 폴더의 파일들을 건강신호등 앱에 복사합니다:

```
AILensScreen.tsx  →  src/screens/AILensScreen.tsx
aiLensService.ts  →  src/services/aiLensService.ts
```

`src/navigation/AppNavigator.tsx`에 AILens 라우트가 이미 추가되어 있습니다.

---

## 기술 스택

- **AI 모델**: PyTorch + MobileNetV3-Small (ImageNet 사전학습)
- **백엔드**: FastAPI + Uvicorn
- **영양 DB**: 국가표준식품성분DB 9.2 (농촌진흥청) + 공공데이터포털 API
- **지원 음식**: 한식 30종 + 양식/패스트푸드 60종 이상

---

## 개발팀

건강신호등 팀 @ 2026 창업 프로젝트
