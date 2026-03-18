"""
건강신호등 AI 렌즈 서버
FastAPI 기반 음식 인식 + 영양성분 분석 API
외부 AI API 없이 자체 MobileNetV3 모델로 추론

실행: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import io
import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from model import analyze_image, get_engine
from nutrition_db import get_nutrition, get_nutrition_from_api, estimate_nutrition
from agents import (
    calculate_health_score,
    get_traffic_light,
    check_allergies,
    get_healthy_alternatives,
    analyze_daily_nutrition,
    generate_meal_recommendation,
    analyze_food_trend,
    DailyNutritionSummary,
    HealthGoal,
)


# ─── 앱 생명주기 ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AI 렌즈 서버 시작")
    get_engine()  # 모델 사전 로드
    yield
    print("🛑 AI 렌즈 서버 종료")


app = FastAPI(
    title="건강신호등 AI 렌즈 API",
    description="음식 사진 → 영양성분 분석 (자체 AI 모델, 외부 API 미사용)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (React Native 앱에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 요청/응답 모델 ────────────────────────────────────────────────────────

class FoodPrediction(BaseModel):
    food_key: str
    food_name_ko: str
    confidence: float


class NutritionResult(BaseModel):
    kcal: float
    carbohydrate: float
    protein: float
    fat: float
    sugar: float
    sodium: float
    fiber: float
    serving_size: str
    estimated: bool = False
    note: Optional[str] = None
    source: Optional[str] = "로컬 DB"


class HealthResult(BaseModel):
    score: int
    traffic_light: str
    emoji: str


class AllergyResult(BaseModel):
    has_warning: bool
    triggered_allergens: list[str]
    message: str


class AnalyzeResponse(BaseModel):
    predictions: list[FoodPrediction]
    top_food: str
    nutrition: NutritionResult
    health: HealthResult
    alternatives: list[str]
    allergy: Optional[AllergyResult] = None


class AllergyCheckRequest(BaseModel):
    food_name_ko: str
    user_allergies: list[str]


class DailyAnalysisRequest(BaseModel):
    summary: DailyNutritionSummary
    goal: Optional[HealthGoal] = None


class MealRecommendRequest(BaseModel):
    summary: DailyNutritionSummary
    meal_type: str = "dinner"
    preferences: Optional[list[str]] = None


class TrendAnalysisRequest(BaseModel):
    scan_history: list[dict]


# ─── 유틸 ──────────────────────────────────────────────────────────────────

def _make_nutrition_result(nutrition: dict) -> NutritionResult:
    return NutritionResult(
        kcal=float(nutrition.get("kcal", 0)),
        carbohydrate=float(nutrition.get("carbohydrate", 0)),
        protein=float(nutrition.get("protein", 0)),
        fat=float(nutrition.get("fat", 0)),
        sugar=float(nutrition.get("sugar", 0)),
        sodium=float(nutrition.get("sodium", 0)),
        fiber=float(nutrition.get("fiber", 0)),
        serving_size=nutrition.get("serving_size", "1인분"),
        estimated=nutrition.get("estimated", False),
        note=nutrition.get("note"),
        source=nutrition.get("source", "로컬 DB"),
    )


def _make_health_result(score: int) -> HealthResult:
    traffic_light = get_traffic_light(score)
    emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(traffic_light, "⚪")
    return HealthResult(score=score, traffic_light=traffic_light, emoji=emoji)


async def _get_nutrition_for_food(food_name_ko: str, food_key: str) -> dict:
    """로컬 DB → API 폴백 → 추정값 순서로 영양 정보 조회"""
    nutrition = get_nutrition(food_name_ko, food_key)
    if nutrition:
        return nutrition
    # 온라인 API 시도
    nutrition = await get_nutrition_from_api(food_name_ko)
    if nutrition:
        return nutrition
    # 추정값 반환
    return estimate_nutrition(food_name_ko)


# ─── 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "message": "AI 렌즈 서버 정상 작동 중 🟢"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_food(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    user_allergies: Optional[str] = Form(None),
    health_conditions: Optional[str] = Form(None),
):
    """
    음식 사진 분석 메인 엔드포인트
    - file: multipart/form-data 이미지 업로드
    - image_base64: base64 인코딩 이미지 (data:image/... 포함 가능)
    - user_allergies: 쉼표 구분 알러지 목록 (예: "우유,계란")
    - health_conditions: JSON 문자열 (예: '{"diabetes":true}')
    """
    # 이미지 로드
    try:
        if file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        elif image_base64:
            # data:image/jpeg;base64,... 접두사 제거
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            img_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            raise HTTPException(status_code=400, detail="이미지를 첨부해주세요 (file 또는 image_base64)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {str(e)}")

    # AI 음식 인식
    predictions_raw = analyze_image(image)
    if not predictions_raw:
        raise HTTPException(status_code=422, detail="음식을 인식하지 못했습니다")

    predictions = [FoodPrediction(**p) for p in predictions_raw]
    top = predictions[0]

    # 영양성분 조회
    nutrition_raw = await _get_nutrition_for_food(top.food_name_ko, top.food_key)

    # 건강 점수 계산
    conditions = {}
    if health_conditions:
        import json
        try:
            conditions = json.loads(health_conditions)
        except Exception:
            pass
    score = calculate_health_score(nutrition_raw, conditions or None)

    # 알러지 체크
    allergy_result = None
    if user_allergies:
        allergy_list = [a.strip() for a in user_allergies.split(",") if a.strip()]
        if allergy_list:
            allergy_raw = check_allergies(top.food_name_ko, allergy_list)
            allergy_result = AllergyResult(
                has_warning=allergy_raw["has_warning"],
                triggered_allergens=allergy_raw["triggered_allergens"],
                message=allergy_raw["message"],
            )

    return AnalyzeResponse(
        predictions=predictions,
        top_food=top.food_name_ko,
        nutrition=_make_nutrition_result(nutrition_raw),
        health=_make_health_result(score),
        alternatives=get_healthy_alternatives(top.food_name_ko),
        allergy=allergy_result,
    )


@app.get("/nutrition/{food_name}")
async def get_food_nutrition(food_name: str):
    """음식 이름으로 영양성분 직접 조회"""
    nutrition = get_nutrition(food_name)
    if not nutrition:
        nutrition = await get_nutrition_from_api(food_name)
    if not nutrition:
        nutrition = estimate_nutrition(food_name)
    score = calculate_health_score(nutrition)
    return {
        "food_name": food_name,
        "nutrition": _make_nutrition_result(nutrition),
        "health": _make_health_result(score),
    }


@app.post("/agents/allergy-check")
async def allergy_check(req: AllergyCheckRequest):
    """알러지 경고 에이전트"""
    return check_allergies(req.food_name_ko, req.user_allergies)


@app.post("/agents/daily-analysis")
async def daily_analysis(req: DailyAnalysisRequest):
    """일일 영양 분석 에이전트"""
    return analyze_daily_nutrition(req.summary, req.goal)


@app.post("/agents/meal-recommend")
async def meal_recommend(req: MealRecommendRequest):
    """식사 추천 에이전트"""
    return {"recommendations": generate_meal_recommendation(req.summary, req.meal_type, req.preferences)}


@app.post("/agents/trend-analysis")
async def trend_analysis(req: TrendAnalysisRequest):
    """식습관 트렌드 분석 에이전트"""
    return analyze_food_trend(req.scan_history)


@app.get("/agents/alternatives/{food_name}")
async def food_alternatives(food_name: str):
    """대체 음식 추천 에이전트"""
    alts = get_healthy_alternatives(food_name)
    return {
        "food_name": food_name,
        "alternatives": alts,
        "message": f"{food_name}의 건강한 대체 음식" if alts else "대체 음식 정보가 없습니다",
    }
