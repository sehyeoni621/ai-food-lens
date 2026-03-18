"""
AI 에이전트 모듈
- 식단 추천 에이전트
- 건강 모니터링 에이전트
- 대체 음식 추천 에이전트
- 알러지 경고 에이전트
- 영양 목표 달성 에이전트
"""
from typing import Optional
from pydantic import BaseModel


# ─── 일일 영양 권장량 (한국영양학회 기준, 성인 기준) ─────────────────────
DAILY_RECOMMENDED = {
    "kcal":        {"male": 2600, "female": 2100, "default": 2000},
    "carbohydrate": {"male": 325, "female": 263, "default": 300},
    "protein":     {"male": 65, "female": 55, "default": 55},
    "fat":         {"male": 58, "female": 47, "default": 54},
    "sugar":       {"male": 25, "female": 25, "default": 25},   # WHO 권장
    "sodium":      {"male": 2000, "female": 2000, "default": 2000},  # 목표섭취량
    "fiber":       {"male": 25, "female": 20, "default": 25},
}

# 건강 점수 가중치 (건강신호등 앱 기준)
HEALTH_SCORE_WEIGHTS = {
    "sugar": 0.30,
    "fat": 0.25,
    "protein": 0.25,
    "sodium": 0.20,
}

# 알러지 유발 성분 키워드
ALLERGY_KEYWORDS = {
    "우유": ["dairy", "milk", "cheese", "butter", "cream", "lactose", "유제품", "치즈", "버터"],
    "계란": ["egg", "계란", "달걀"],
    "땅콩": ["peanut", "땅콩"],
    "밀": ["wheat", "flour", "gluten", "밀", "밀가루"],
    "대두": ["soy", "tofu", "두부", "콩", "된장", "간장"],
    "새우": ["shrimp", "prawn", "새우"],
    "복숭아": ["peach", "복숭아"],
    "토마토": ["tomato", "토마토"],
}

# 음식별 알러지 정보
FOOD_ALLERGEN_MAP: dict[str, list[str]] = {
    "피자": ["밀", "우유"],
    "햄버거": ["밀", "계란"],
    "파스타": ["밀", "계란"],
    "짜장면": ["밀", "대두"],
    "짬뽕": ["밀", "새우"],
    "라면": ["밀", "대두"],
    "만두": ["밀", "계란", "대두"],
    "교자": ["밀", "계란"],
    "김밥": ["계란"],
    "초밥": [],
    "케이크": ["밀", "계란", "우유"],
    "도넛": ["밀", "계란", "우유"],
    "와플": ["밀", "계란", "우유"],
    "팬케이크": ["밀", "계란", "우유"],
    "샌드위치": ["밀", "계란"],
    "아이스크림": ["우유", "계란"],
    "된장찌개": ["대두"],
    "순두부찌개": ["대두"],
    "비빔밥": [],
    "불고기": ["대두"],
    "삼계탕": [],
    "삼겹살": [],
    "갈비": ["대두"],
    "잡채": ["대두"],
    "치킨": ["밀"],
    "감자튀김": [],
}

# 저칼로리 대체 음식 추천
HEALTHY_ALTERNATIVES: dict[str, list[str]] = {
    "라면":     ["미역국", "콩나물국", "계란국"],
    "짜장면":   ["비빔밥", "잡채밥"],
    "짬뽕":    ["된장찌개", "순두부찌개"],
    "피자":    ["샐러드", "샌드위치"],
    "햄버거":  ["비빔밥", "삼계탕"],
    "치킨":    ["삼계탕", "불고기"],
    "감자튀김": ["샐러드", "사과", "바나나"],
    "케이크":  ["과일", "요거트"],
    "도넛":    ["과일", "견과류"],
    "아이스크림": ["과일", "요거트"],
    "삼겹살":  ["불고기", "닭갈비"],
    "떡볶이":  ["김밥", "잡채"],
}


class DailyNutritionSummary(BaseModel):
    total_kcal: float
    total_carbohydrate: float
    total_protein: float
    total_fat: float
    total_sugar: float
    total_sodium: float
    meal_count: int


class HealthGoal(BaseModel):
    goal_type: str  # "diet", "muscle", "maintain", "health"
    gender: str = "default"
    age: int = 30
    weight_kg: float = 65.0
    height_cm: float = 170.0


# ─── 건강 점수 계산 ────────────────────────────────────────────────────────
def calculate_health_score(
    nutrition: dict,
    health_conditions: Optional[dict] = None,
) -> int:
    """
    건강신호등 앱 동일 알고리즘
    당류 30%, 지방 25%, 단백질 25%, 나트륨 20%
    건강 조건(당뇨/고지혈/고혈압)에 따라 가중치 조정
    """
    weights = HEALTH_SCORE_WEIGHTS.copy()
    if health_conditions:
        if health_conditions.get("diabetes"):
            weights["sugar"] = min(0.45, weights["sugar"] + 0.15)
        if health_conditions.get("hyperlipidemia"):
            weights["fat"] = min(0.40, weights["fat"] + 0.15)
        if health_conditions.get("hypertension"):
            weights["sodium"] = min(0.35, weights["sodium"] + 0.15)
        # 가중치 정규화
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    sugar = float(nutrition.get("sugar", 0))
    fat = float(nutrition.get("fat", 0))
    protein = float(nutrition.get("protein", 0))
    sodium = float(nutrition.get("sodium", 0))
    kcal = float(nutrition.get("kcal", 300))

    # 당류 점수 (WHO 권장 25g → 100점, 초과할수록 감점)
    sugar_score = max(0, 100 - (sugar / 25) * 100)
    # 지방 점수 (칼로리의 35% 이하 → 100점)
    fat_ratio = (fat * 9) / max(kcal, 1)
    fat_score = max(0, 100 - max(0, fat_ratio - 0.35) / 0.35 * 100)
    # 단백질 점수 (5g 이상 있으면 가점, 최대 100)
    protein_score = min(100, (protein / 5) * 50)
    # 나트륨 점수 (500mg 이하 → 100점)
    sodium_score = max(0, 100 - (sodium / 500) * 100)

    score = (
        sugar_score * weights["sugar"]
        + fat_score * weights["fat"]
        + protein_score * weights["protein"]
        + sodium_score * weights["sodium"]
    )
    return int(round(min(100, max(0, score))))


def get_traffic_light(score: int) -> str:
    if score >= 80:
        return "green"
    elif score >= 60:
        return "yellow"
    return "red"


# ─── 에이전트 함수들 ─────────────────────────────────────────────────────

def check_allergies(food_name_ko: str, user_allergies: list[str]) -> dict:
    """알러지 경고 에이전트"""
    food_allergens = FOOD_ALLERGEN_MAP.get(food_name_ko, [])
    triggered = [a for a in user_allergies if a in food_allergens]
    return {
        "has_warning": len(triggered) > 0,
        "triggered_allergens": triggered,
        "food_allergens": food_allergens,
        "message": f"⚠️ {', '.join(triggered)} 알러지 주의!" if triggered else "✅ 알러지 성분 없음",
    }


def get_healthy_alternatives(food_name_ko: str) -> list[str]:
    """대체 음식 추천 에이전트"""
    return HEALTHY_ALTERNATIVES.get(food_name_ko, [])


def analyze_daily_nutrition(
    daily_summary: DailyNutritionSummary,
    goal: Optional[HealthGoal] = None,
) -> dict:
    """일일 영양 분석 에이전트"""
    gender = goal.gender if goal else "default"
    recs = {k: v.get(gender, v["default"]) for k, v in DAILY_RECOMMENDED.items()}

    kcal_pct = round(daily_summary.total_kcal / recs["kcal"] * 100, 1)
    carb_pct = round(daily_summary.total_carbohydrate / recs["carbohydrate"] * 100, 1)
    prot_pct = round(daily_summary.total_protein / recs["protein"] * 100, 1)
    fat_pct = round(daily_summary.total_fat / recs["fat"] * 100, 1)
    sugar_pct = round(daily_summary.total_sugar / recs["sugar"] * 100, 1)
    sodium_pct = round(daily_summary.total_sodium / recs["sodium"] * 100, 1)

    warnings = []
    tips = []

    if sugar_pct > 100:
        warnings.append(f"당류 초과 ({sugar_pct}% 섭취) – 단 음식을 줄여보세요")
    if sodium_pct > 100:
        warnings.append(f"나트륨 초과 ({sodium_pct}% 섭취) – 물을 충분히 드세요")
    if fat_pct > 100:
        warnings.append(f"지방 초과 ({fat_pct}% 섭취) – 기름진 음식을 줄여보세요")
    if prot_pct < 50:
        tips.append("단백질이 부족합니다. 고기, 두부, 계란을 추가해보세요")
    if kcal_pct < 60:
        tips.append("칼로리가 부족합니다. 충분히 드세요!")
    elif kcal_pct > 120:
        warnings.append(f"칼로리 초과 ({kcal_pct}%) – 가벼운 운동을 추천합니다")

    return {
        "percentages": {
            "kcal": kcal_pct, "carbohydrate": carb_pct, "protein": prot_pct,
            "fat": fat_pct, "sugar": sugar_pct, "sodium": sodium_pct,
        },
        "recommended": recs,
        "warnings": warnings,
        "tips": tips,
        "status": "danger" if warnings else ("good" if not tips else "caution"),
    }


def generate_meal_recommendation(
    daily_summary: DailyNutritionSummary,
    meal_type: str = "dinner",  # breakfast / lunch / dinner / snack
    preferences: Optional[list[str]] = None,
) -> list[dict]:
    """
    남은 영양 예산 기반 식사 추천 에이전트
    meal_type: 아침/점심/저녁/간식
    """
    from nutrition_db import NUTRITION_DB, calculate_health_score_for_food

    remaining_kcal = 2000 - daily_summary.total_kcal
    remaining_protein = 55 - daily_summary.total_protein

    recommendations = []

    for food_name, nutrition in NUTRITION_DB.items():
        food_kcal = nutrition.get("kcal", 0)
        food_protein = nutrition.get("protein", 0)

        # 남은 칼로리 범위 내 음식
        if meal_type == "snack":
            if not (50 <= food_kcal <= 300):
                continue
        else:
            if not (200 <= food_kcal <= remaining_kcal + 100):
                continue

        score = calculate_health_score(nutrition)
        if score < 50:
            continue

        recommendations.append({
            "food_name": food_name,
            "kcal": food_kcal,
            "protein": food_protein,
            "health_score": score,
            "traffic_light": get_traffic_light(score),
            "reason": _get_recommendation_reason(food_name, nutrition, daily_summary),
        })

    # 건강 점수 내림차순 정렬
    recommendations.sort(key=lambda x: x["health_score"], reverse=True)
    return recommendations[:5]


def _get_recommendation_reason(food_name: str, nutrition: dict, daily: DailyNutritionSummary) -> str:
    reasons = []
    if float(nutrition.get("protein", 0)) >= 15 and daily.total_protein < 30:
        reasons.append("단백질 보충")
    if float(nutrition.get("kcal", 0)) < 250:
        reasons.append("저칼로리")
    if float(nutrition.get("sodium", 0)) < 400:
        reasons.append("나트륨 낮음")
    return ", ".join(reasons) if reasons else "균형잡힌 영양"


def analyze_food_trend(scan_history: list[dict]) -> dict:
    """스캔 기록 기반 식습관 분석 에이전트"""
    if not scan_history:
        return {"message": "아직 분석할 기록이 없습니다."}

    total = len(scan_history)
    green_count = sum(1 for s in scan_history if s.get("traffic_light") == "green")
    red_count = sum(1 for s in scan_history if s.get("traffic_light") == "red")
    green_rate = round(green_count / total * 100, 1)

    # 평균 영양
    avg_kcal = round(
        sum(s.get("kcal", 0) for s in scan_history) / total, 1
    )

    trend = "좋음 🟢" if green_rate >= 65 else ("보통 🟡" if green_rate >= 40 else "주의 🔴")

    return {
        "total_scans": total,
        "green_rate": green_rate,
        "red_count": red_count,
        "avg_kcal": avg_kcal,
        "trend": trend,
        "advice": _get_trend_advice(green_rate, avg_kcal),
    }


def _get_trend_advice(green_rate: float, avg_kcal: float) -> str:
    if green_rate >= 65:
        return "훌륭한 식습관을 유지하고 있어요! 계속 이 페이스로 가세요 💪"
    elif green_rate >= 40:
        return "전체적으로 나쁘지 않지만, 빨간 불 식품을 조금 줄여보세요"
    else:
        return "건강한 음식 선택이 필요합니다. 채소, 과일, 단백질 위주로 드세요"
