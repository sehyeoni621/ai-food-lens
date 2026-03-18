"""
한국 음식 영양성분 데이터베이스
국가표준식품성분DB 기반 + 공공데이터포털 API 연동
"""
import os
import httpx
from typing import Optional

# ─── 로컬 영양성분 DB (100인분 기준, 단위: g/mg/kcal) ─────────────────────
# 출처: 국가표준식품성분DB 9.2 (농촌진흥청)
NUTRITION_DB: dict[str, dict] = {
    "비빔밥": {
        "kcal": 570, "carbohydrate": 86.0, "protein": 18.5, "fat": 15.2,
        "sugar": 4.2, "sodium": 890, "fiber": 4.1, "serving_size": "1인분(420g)"
    },
    "김치찌개": {
        "kcal": 210, "carbohydrate": 12.0, "protein": 14.0, "fat": 10.5,
        "sugar": 3.1, "sodium": 1450, "fiber": 2.8, "serving_size": "1인분(300g)"
    },
    "된장찌개": {
        "kcal": 190, "carbohydrate": 14.0, "protein": 12.5, "fat": 8.0,
        "sugar": 2.5, "sodium": 1380, "fiber": 3.2, "serving_size": "1인분(300g)"
    },
    "불고기": {
        "kcal": 285, "carbohydrate": 12.0, "protein": 26.5, "fat": 14.2,
        "sugar": 8.5, "sodium": 680, "fiber": 0.8, "serving_size": "1인분(200g)"
    },
    "삼겹살": {
        "kcal": 480, "carbohydrate": 0.0, "protein": 22.0, "fat": 42.5,
        "sugar": 0.0, "sodium": 480, "fiber": 0.0, "serving_size": "1인분(200g)"
    },
    "갈비": {
        "kcal": 390, "carbohydrate": 8.0, "protein": 28.0, "fat": 26.0,
        "sugar": 5.0, "sodium": 720, "fiber": 0.5, "serving_size": "1인분(200g)"
    },
    "떡볶이": {
        "kcal": 320, "carbohydrate": 62.0, "protein": 7.5, "fat": 4.5,
        "sugar": 14.0, "sodium": 1250, "fiber": 2.0, "serving_size": "1인분(250g)"
    },
    "김밥": {
        "kcal": 380, "carbohydrate": 65.0, "protein": 12.0, "fat": 8.0,
        "sugar": 3.5, "sodium": 890, "fiber": 2.5, "serving_size": "1줄(250g)"
    },
    "라면": {
        "kcal": 500, "carbohydrate": 72.0, "protein": 10.5, "fat": 18.0,
        "sugar": 4.0, "sodium": 1900, "fiber": 2.0, "serving_size": "1개(120g 건면)"
    },
    "잡채": {
        "kcal": 290, "carbohydrate": 38.0, "protein": 10.5, "fat": 10.0,
        "sugar": 5.2, "sodium": 580, "fiber": 2.5, "serving_size": "1인분(200g)"
    },
    "순두부찌개": {
        "kcal": 165, "carbohydrate": 8.0, "protein": 12.0, "fat": 8.5,
        "sugar": 1.5, "sodium": 1200, "fiber": 1.8, "serving_size": "1인분(300g)"
    },
    "삼계탕": {
        "kcal": 460, "carbohydrate": 28.0, "protein": 38.0, "fat": 18.5,
        "sugar": 1.0, "sodium": 820, "fiber": 0.8, "serving_size": "1인분(550g)"
    },
    "냉면": {
        "kcal": 450, "carbohydrate": 82.0, "protein": 14.5, "fat": 6.0,
        "sugar": 5.0, "sodium": 1350, "fiber": 2.5, "serving_size": "1인분(450g)"
    },
    "짜장면": {
        "kcal": 650, "carbohydrate": 100.0, "protein": 18.0, "fat": 18.0,
        "sugar": 6.0, "sodium": 1100, "fiber": 3.5, "serving_size": "1인분(500g)"
    },
    "짬뽕": {
        "kcal": 590, "carbohydrate": 76.0, "protein": 25.0, "fat": 18.5,
        "sugar": 4.5, "sodium": 2100, "fiber": 3.0, "serving_size": "1인분(500g)"
    },
    "닭갈비": {
        "kcal": 350, "carbohydrate": 25.0, "protein": 30.0, "fat": 14.0,
        "sugar": 8.0, "sodium": 980, "fiber": 2.5, "serving_size": "1인분(300g)"
    },
    "제육볶음": {
        "kcal": 380, "carbohydrate": 15.0, "protein": 28.0, "fat": 22.0,
        "sugar": 7.5, "sodium": 1050, "fiber": 1.8, "serving_size": "1인분(200g)"
    },
    "파전": {
        "kcal": 320, "carbohydrate": 38.0, "protein": 8.5, "fat": 14.5,
        "sugar": 2.0, "sodium": 680, "fiber": 2.0, "serving_size": "1개(200g)"
    },
    "만두": {
        "kcal": 280, "carbohydrate": 34.0, "protein": 12.5, "fat": 10.0,
        "sugar": 2.5, "sodium": 750, "fiber": 1.5, "serving_size": "1인분(200g)"
    },
    "피자": {
        "kcal": 266, "carbohydrate": 33.0, "protein": 11.0, "fat": 10.0,
        "sugar": 3.5, "sodium": 640, "fiber": 2.2, "serving_size": "1조각(100g)"
    },
    "햄버거": {
        "kcal": 490, "carbohydrate": 45.0, "protein": 25.0, "fat": 22.5,
        "sugar": 8.0, "sodium": 980, "fiber": 2.0, "serving_size": "1개(200g)"
    },
    "핫도그": {
        "kcal": 280, "carbohydrate": 25.0, "protein": 11.0, "fat": 15.0,
        "sugar": 3.0, "sodium": 680, "fiber": 1.0, "serving_size": "1개(120g)"
    },
    "치킨": {
        "kcal": 290, "carbohydrate": 12.0, "protein": 26.0, "fat": 15.5,
        "sugar": 1.0, "sodium": 520, "fiber": 0.5, "serving_size": "1조각(100g)"
    },
    "초밥": {
        "kcal": 180, "carbohydrate": 32.0, "protein": 8.0, "fat": 2.5,
        "sugar": 4.0, "sodium": 420, "fiber": 0.8, "serving_size": "2개(80g)"
    },
    "볶음밥": {
        "kcal": 420, "carbohydrate": 58.0, "protein": 14.0, "fat": 14.5,
        "sugar": 2.5, "sodium": 890, "fiber": 1.5, "serving_size": "1인분(300g)"
    },
    "파스타": {
        "kcal": 380, "carbohydrate": 52.0, "protein": 14.0, "fat": 12.5,
        "sugar": 5.0, "sodium": 680, "fiber": 3.0, "serving_size": "1인분(300g)"
    },
    "샐러드": {
        "kcal": 120, "carbohydrate": 12.0, "protein": 4.5, "fat": 6.0,
        "sugar": 5.0, "sodium": 280, "fiber": 3.5, "serving_size": "1인분(200g)"
    },
    "샌드위치": {
        "kcal": 350, "carbohydrate": 38.0, "protein": 18.0, "fat": 13.0,
        "sugar": 4.0, "sodium": 780, "fiber": 2.5, "serving_size": "1개(180g)"
    },
    "스테이크": {
        "kcal": 420, "carbohydrate": 0.0, "protein": 42.0, "fat": 26.0,
        "sugar": 0.0, "sodium": 480, "fiber": 0.0, "serving_size": "1인분(200g)"
    },
    "사과": {
        "kcal": 52, "carbohydrate": 14.0, "protein": 0.3, "fat": 0.2,
        "sugar": 10.0, "sodium": 1, "fiber": 2.4, "serving_size": "중간(100g)"
    },
    "바나나": {
        "kcal": 89, "carbohydrate": 23.0, "protein": 1.1, "fat": 0.3,
        "sugar": 12.2, "sodium": 1, "fiber": 2.6, "serving_size": "1개(100g)"
    },
    "달걀": {
        "kcal": 155, "carbohydrate": 1.1, "protein": 13.0, "fat": 11.0,
        "sugar": 1.1, "sodium": 124, "fiber": 0.0, "serving_size": "2개(100g)"
    },
    "아이스크림": {
        "kcal": 207, "carbohydrate": 24.0, "protein": 3.5, "fat": 11.0,
        "sugar": 21.0, "sodium": 80, "fiber": 0.0, "serving_size": "1컵(100g)"
    },
    "케이크": {
        "kcal": 350, "carbohydrate": 45.0, "protein": 5.0, "fat": 16.0,
        "sugar": 28.0, "sodium": 230, "fiber": 1.0, "serving_size": "1조각(100g)"
    },
    "도넛": {
        "kcal": 380, "carbohydrate": 50.0, "protein": 5.0, "fat": 18.0,
        "sugar": 20.0, "sodium": 310, "fiber": 1.2, "serving_size": "1개(100g)"
    },
    "와플": {
        "kcal": 291, "carbohydrate": 37.0, "protein": 8.0, "fat": 12.0,
        "sugar": 8.0, "sodium": 450, "fiber": 1.5, "serving_size": "1개(100g)"
    },
    "감자튀김": {
        "kcal": 312, "carbohydrate": 41.0, "protein": 3.5, "fat": 15.0,
        "sugar": 0.5, "sodium": 210, "fiber": 3.5, "serving_size": "1인분(100g)"
    },
    "쌀국수": {
        "kcal": 350, "carbohydrate": 62.0, "protein": 18.0, "fat": 4.5,
        "sugar": 3.0, "sodium": 1100, "fiber": 1.5, "serving_size": "1인분(400g)"
    },
    "라멘": {
        "kcal": 520, "carbohydrate": 64.0, "protein": 22.0, "fat": 18.0,
        "sugar": 5.0, "sodium": 1850, "fiber": 2.0, "serving_size": "1인분(450g)"
    },
    "팟타이": {
        "kcal": 400, "carbohydrate": 55.0, "protein": 18.0, "fat": 12.0,
        "sugar": 8.0, "sodium": 950, "fiber": 2.5, "serving_size": "1인분(350g)"
    },
    "타코": {
        "kcal": 220, "carbohydrate": 20.0, "protein": 12.0, "fat": 9.0,
        "sugar": 2.0, "sodium": 460, "fiber": 2.5, "serving_size": "1개(100g)"
    },
}

# Food101 키 → 한국어 이름 역매핑 (model.py FOOD101_TO_KO 활용)
FOOD101_KEY_TO_DB_NAME: dict[str, str] = {
    "bibimbap": "비빔밥",
    "kimchi_jjigae": "김치찌개",
    "doenjang_jjigae": "된장찌개",
    "bulgogi": "불고기",
    "samgyeopsal": "삼겹살",
    "galbi": "갈비",
    "tteokbokki": "떡볶이",
    "gimbap": "김밥",
    "kimbap": "김밥",
    "ramen": "라면",
    "sundubu_jjigae": "순두부찌개",
    "samgyetang": "삼계탕",
    "naengmyeon": "냉면",
    "jajangmyeon": "짜장면",
    "jjamppong": "짬뽕",
    "dakgalbi": "닭갈비",
    "jeyuk_bokkeum": "제육볶음",
    "pajeon": "파전",
    "dumplings": "만두",
    "gyoza": "만두",
    "pizza": "피자",
    "hamburger": "햄버거",
    "hot_dog": "핫도그",
    "fried_chicken": "치킨",
    "chicken_wings": "치킨",
    "sushi": "초밥",
    "fried_rice": "볶음밥",
    "spaghetti_bolognese": "파스타",
    "spaghetti_carbonara": "파스타",
    "caesar_salad": "샐러드",
    "greek_salad": "샐러드",
    "club_sandwich": "샌드위치",
    "steak": "스테이크",
    "filet_mignon": "스테이크",
    "prime_rib": "스테이크",
    "apple_pie": "케이크",
    "chocolate_cake": "케이크",
    "cheesecake": "케이크",
    "cup_cakes": "케이크",
    "ice_cream": "아이스크림",
    "frozen_yogurt": "아이스크림",
    "donuts": "도넛",
    "waffles": "와플",
    "french_fries": "감자튀김",
    "pho": "쌀국수",
    "ramen_soup": "라멘",
    "pad_thai": "팟타이",
    "tacos": "타코",
    "nachos": "나초",
    "banana": "바나나",
    "apple": "사과",
    "edamame": "샐러드",
}


def get_nutrition(food_name_ko: str, food_key: str = "") -> Optional[dict]:
    """
    한국어 음식 이름 또는 food_key로 영양성분 조회
    Returns: nutrition dict 또는 None
    """
    # 1. 직접 DB 조회 (한국어 이름)
    if food_name_ko in NUTRITION_DB:
        return NUTRITION_DB[food_name_ko]

    # 2. food_key로 한국어 이름 찾기
    if food_key and food_key in FOOD101_KEY_TO_DB_NAME:
        mapped = FOOD101_KEY_TO_DB_NAME[food_key]
        if mapped in NUTRITION_DB:
            return NUTRITION_DB[mapped]

    # 3. 부분 매치 (이름에 키워드 포함)
    for db_name, nutrition in NUTRITION_DB.items():
        if db_name in food_name_ko or food_name_ko in db_name:
            return nutrition

    return None


async def get_nutrition_from_api(food_name: str) -> Optional[dict]:
    """
    공공데이터포털 식품영양성분DB API 조회 (온라인 폴백)
    API 키: .env의 NUTRITION_API_KEY
    """
    api_key = os.getenv("NUTRITION_API_KEY", "")
    if not api_key:
        return None

    url = "http://api.data.go.kr/openapi/tn_pubr_public_nutri_process_info_api"
    params = {
        "serviceKey": api_key,
        "pageNo": 1,
        "numOfRows": 1,
        "type": "JSON",
        "foodNm": food_name,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            items = data.get("response", {}).get("body", {}).get("items", [])
            if not items:
                return None
            item = items[0]
            return {
                "kcal": float(item.get("enerc", 0) or 0),
                "carbohydrate": float(item.get("chocdf", 0) or 0),
                "protein": float(item.get("prot", 0) or 0),
                "fat": float(item.get("fatce", 0) or 0),
                "sugar": float(item.get("sugar", 0) or 0),
                "sodium": float(item.get("nat", 0) or 0),
                "fiber": float(item.get("fibtg", 0) or 0),
                "serving_size": item.get("servSize", "1인분"),
                "source": "공공데이터포털",
            }
    except Exception as e:
        print(f"[영양DB API] 오류: {e}")
        return None


def estimate_nutrition(food_name_ko: str) -> dict:
    """
    DB에 없는 음식의 경우 기본값 반환
    """
    return {
        "kcal": 300,
        "carbohydrate": 40.0,
        "protein": 10.0,
        "fat": 10.0,
        "sugar": 5.0,
        "sodium": 500,
        "fiber": 2.0,
        "serving_size": "1인분",
        "estimated": True,
        "note": f"'{food_name_ko}'의 정확한 영양 데이터가 없어 평균값으로 추정했습니다.",
    }
