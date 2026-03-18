"""
AI 음식 인식 모델 모듈
MobileNetV3-Small (ImageNet 사전학습) 기반 음식 분류기
외부 AI API 없이 자체 추론 수행
"""
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "food_classifier.pth"
LABELS_PATH = Path(__file__).parent / "food_labels_ko.json"


# ─── 음식 레이블 (한국어 + 영어 매핑) ────────────────────────────────────
FOOD_LABELS_KO = {
    # 한식
    "bibimbap": "비빔밥",
    "kimchi_jjigae": "김치찌개",
    "doenjang_jjigae": "된장찌개",
    "japchae": "잡채",
    "galbi": "갈비",
    "bulgogi": "불고기",
    "samgyeopsal": "삼겹살",
    "tteokbokki": "떡볶이",
    "gimbap": "김밥",
    "ramen": "라면",
    "sundubu_jjigae": "순두부찌개",
    "samgyetang": "삼계탕",
    "naengmyeon": "냉면",
    "jajangmyeon": "짜장면",
    "jjamppong": "짬뽕",
    "dakgalbi": "닭갈비",
    "gamjatang": "감자탕",
    "seolleongtang": "설렁탕",
    "sundubu": "순두부",
    "haemultang": "해물탕",
    "bosintang": "잡탕",
    "gukbap": "국밥",
    "dosirak": "도시락",
    "kimbap": "김밥",
    "tteonbap": "떡밥",
    "hobak_jeon": "호박전",
    "pajeon": "파전",
    "bindaetteok": "빈대떡",
    "japagetti": "자파게티",
    "jeyuk_bokkeum": "제육볶음",
    # 양식/패스트푸드
    "pizza": "피자",
    "hamburger": "햄버거",
    "hot_dog": "핫도그",
    "sushi": "초밥",
    "fried_rice": "볶음밥",
    "noodles": "국수",
    "pasta": "파스타",
    "salad": "샐러드",
    "sandwich": "샌드위치",
    "steak": "스테이크",
    "fried_chicken": "치킨",
    "donut": "도넛",
    "ice_cream": "아이스크림",
    "cake": "케이크",
    "waffles": "와플",
    "pancakes": "팬케이크",
    "french_fries": "감자튀김",
    "onion_rings": "양파링",
    "spring_rolls": "춘권",
    "dumplings": "만두",
    "egg": "계란",
    "bacon": "베이컨",
    "apple_pie": "애플파이",
    "chocolate_cake": "초콜릿케이크",
    "cheesecake": "치즈케이크",
    "cup_cakes": "컵케이크",
    "macarons": "마카롱",
    "tiramisu": "티라미수",
    "bruschetta": "브루스케타",
    "caesar_salad": "시저샐러드",
    "greek_salad": "그릭샐러드",
    "caprese_salad": "카프레제",
    "spaghetti_bolognese": "볼로네제",
    "pad_thai": "팟타이",
    "chicken_curry": "치킨 카레",
    "beef_curry": "비프 카레",
    "ramen_soup": "라멘",
    "miso_soup": "미소국",
    "sashimi": "회",
    "tempura": "튀김",
    "takoyaki": "타코야키",
    "okonomiyaki": "오코노미야키",
    "gyoza": "교자",
    "dim_sum": "딤섬",
    "baozi": "만두",
    "pho": "쌀국수",
    "banh_mi": "반미",
    "tom_yum": "똠얌",
    "green_curry": "그린 카레",
    "naan": "난",
    "hummus": "훔무스",
    "falafel": "팔라펠",
    "shawarma": "샤와르마",
    "kebab": "케밥",
    "tacos": "타코",
    "burrito": "부리또",
    "nachos": "나초",
    "quesadilla": "케사디아",
    "guacamole": "과카몰리",
    # 과일/채소
    "apple": "사과",
    "banana": "바나나",
    "orange": "오렌지",
    "strawberry": "딸기",
    "watermelon": "수박",
    "grapes": "포도",
    "mango": "망고",
    "peach": "복숭아",
    "pear": "배",
    "kiwi": "키위",
}

# Food-101 → 한국어 레이블 직접 매핑
FOOD101_TO_KO = {
    "apple_pie": "애플파이",
    "baby_back_ribs": "베이비백 립",
    "baklava": "바클라바",
    "beef_carpaccio": "카르파치오",
    "beef_tartare": "육회",
    "beet_salad": "비트 샐러드",
    "beignets": "베이네",
    "bibimbap": "비빔밥",
    "bread_pudding": "빵 푸딩",
    "breakfast_burrito": "부리또",
    "bruschetta": "브루스케타",
    "caesar_salad": "시저 샐러드",
    "cannoli": "카놀리",
    "caprese_salad": "카프레제",
    "carrot_cake": "당근 케이크",
    "ceviche": "세비체",
    "cheese_plate": "치즈 플레이트",
    "cheesecake": "치즈케이크",
    "chicken_curry": "치킨 카레",
    "chicken_quesadilla": "케사디아",
    "chicken_wings": "치킨 윙",
    "chocolate_cake": "초콜릿 케이크",
    "chocolate_mousse": "초콜릿 무스",
    "churros": "츄러스",
    "clam_chowder": "클램 차우더",
    "club_sandwich": "클럽 샌드위치",
    "crab_cakes": "크랩 케이크",
    "creme_brulee": "크렘 브륄레",
    "croque_madame": "크로크 마담",
    "cup_cakes": "컵케이크",
    "deviled_eggs": "데빌드 에그",
    "donuts": "도넛",
    "dumplings": "만두",
    "edamame": "에다마메",
    "eggs_benedict": "에그 베네딕트",
    "escargots": "에스카르고",
    "falafel": "팔라펠",
    "filet_mignon": "필레 미뇽",
    "fish_and_chips": "피시 앤 칩스",
    "foie_gras": "푸아그라",
    "french_fries": "감자튀김",
    "french_onion_soup": "프렌치 어니언 수프",
    "french_toast": "프렌치 토스트",
    "fried_calamari": "오징어 튀김",
    "fried_rice": "볶음밥",
    "frozen_yogurt": "프로즌 요거트",
    "garlic_bread": "마늘빵",
    "gnocchi": "뇨키",
    "greek_salad": "그릭 샐러드",
    "grilled_cheese_sandwich": "그릴드 치즈",
    "grilled_salmon": "연어 구이",
    "guacamole": "과카몰리",
    "gyoza": "교자",
    "hamburger": "햄버거",
    "hot_and_sour_soup": "쏸라탕",
    "hot_dog": "핫도그",
    "huevos_rancheros": "우에보스 란체로스",
    "hummus": "훔무스",
    "ice_cream": "아이스크림",
    "lasagna": "라자냐",
    "lobster_bisque": "랍스터 비스크",
    "lobster_roll_sandwich": "랍스터 롤",
    "macaroni_and_cheese": "맥 앤 치즈",
    "macarons": "마카롱",
    "miso_soup": "미소국",
    "mussels": "홍합 요리",
    "nachos": "나초",
    "omelette": "오믈렛",
    "onion_rings": "양파링",
    "oysters": "굴",
    "pad_thai": "팟타이",
    "paella": "파에야",
    "pancakes": "팬케이크",
    "panna_cotta": "판나코타",
    "peking_duck": "베이징덕",
    "pho": "쌀국수",
    "pizza": "피자",
    "pork_chop": "포크 찹",
    "poutine": "푸틴",
    "prime_rib": "프라임 립",
    "pulled_pork_sandwich": "풀드 포크",
    "ramen": "라멘",
    "ravioli": "라비올리",
    "red_velvet_cake": "레드벨벳 케이크",
    "risotto": "리조또",
    "samosa": "사모사",
    "sashimi": "사시미",
    "scallops": "가리비",
    "seaweed_salad": "해초 샐러드",
    "shrimp_and_grits": "새우 그리츠",
    "spaghetti_bolognese": "볼로네제",
    "spaghetti_carbonara": "카르보나라",
    "spring_rolls": "춘권",
    "steak": "스테이크",
    "strawberry_shortcake": "딸기 쇼트케이크",
    "sushi": "초밥",
    "tacos": "타코",
    "takoyaki": "타코야키",
    "tiramisu": "티라미수",
    "tuna_tartare": "참치 타르타르",
    "waffles": "와플",
}


class FoodClassifier(nn.Module):
    """MobileNetV3-Small 기반 음식 분류 모델"""

    def __init__(self, num_classes: int = 101):
        super().__init__()
        # ImageNet 사전학습 백본 로드
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # 분류 헤드 교체 (음식 카테고리용)
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# ImageNet 표준화 전처리
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Food-101 클래스 순서 (torchvision 순서와 동일)
FOOD101_CLASSES = list(FOOD101_TO_KO.keys())


class FoodRecognitionEngine:
    """음식 인식 엔진: 이미지 → 음식명 + 신뢰도"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = FOOD101_CLASSES
        self._load_model()

    def _load_model(self):
        """모델 로드 (파인튜닝 가중치 있으면 로드, 없으면 ImageNet 기본값 사용)"""
        num_classes = len(self.classes)
        self.model = FoodClassifier(num_classes=num_classes)

        if MODEL_PATH.exists():
            print(f"[AI 렌즈] 파인튜닝 가중치 로드: {MODEL_PATH}")
            state = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            print("[AI 렌즈] ImageNet 사전학습 가중치로 초기화 (파인튜닝 권장)")

        self.model.to(self.device)
        self.model.eval()
        print(f"[AI 렌즈] 모델 준비 완료 (device={self.device}, classes={num_classes})")

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        """
        이미지 → 상위 K개 음식 예측
        Returns: [{"food_key": str, "food_name_ko": str, "confidence": float}]
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.classes)))

        results = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            food_key = self.classes[idx] if idx < len(self.classes) else "unknown"
            food_name_ko = FOOD101_TO_KO.get(food_key, food_key)
            results.append({
                "food_key": food_key,
                "food_name_ko": food_name_ko,
                "confidence": round(prob * 100, 1),
            })

        return results


# 싱글톤 엔진 인스턴스
_engine: FoodRecognitionEngine | None = None


def get_engine() -> FoodRecognitionEngine:
    global _engine
    if _engine is None:
        _engine = FoodRecognitionEngine()
    return _engine


def analyze_image(image: Image.Image) -> list[dict]:
    """외부에서 사용하는 분석 함수"""
    return get_engine().predict(image, top_k=3)
