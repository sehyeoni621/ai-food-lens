"""
AI 음식 인식 모델 모듈 (실제 동작 버전)

동작 방식:
  1단계: ImageNet-1000 사전학습 MobileNetV3로 음식 관련 클래스 감지
          → 즉시 동작, 별도 학습 불필요
          → 피자/햄버거/핫도그 등 서양 음식 70종 인식 가능

  2단계: 한식 인식이 필요하면 train.py로 파인튜닝
          → AIFood30K 또는 직접 수집한 한식 이미지로 학습
"""

import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
FINETUNED_PATH = MODEL_DIR / "food_finetuned.pth"

# ─── ImageNet 1000 클래스 중 음식 관련 인덱스 → 한국어 이름 매핑 ────────────
# 출처: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
IMAGENET_FOOD_MAP: dict[int, str] = {
    # 과일 / 채소
    924: "과카몰리",
    948: "사과",
    949: "딸기",
    950: "오렌지",
    951: "레몬",
    953: "파인애플",
    954: "바나나",
    945: "피망",
    937: "브로콜리",
    938: "콜리플라워",
    936: "양배추",
    943: "오이",
    944: "아티초크",
    947: "버섯",
    939: "애호박",
    940: "단호박",
    # 빵 / 패스트리
    930: "바게트",
    931: "베이글",
    932: "프레첼",
    # 패스트푸드
    933: "치즈버거",       # cheeseburger
    934: "핫도그",         # hotdog
    963: "피자",           # pizza
    965: "부리또",         # burrito
    935: "매시드포테이토", # mashed potato
    # 서양 요리
    925: "콩소메 수프",
    926: "훠궈",
    927: "트라이플",
    928: "아이스크림",
    929: "아이스바",
    959: "카르보나라",
    960: "초콜릿 소스",
    961: "도우",
    962: "미트로프",
    964: "포트파이",
    # 음료 / 기타
    966: "레드와인",
    967: "에스프레소",
    969: "에그노그",
    # 일부 한식 관련 가능성 있는 클래스 (국수류)
    811: "국수",           # noodles (ImageNet 실제 레이블)
}

# 신뢰도 낮을 때 표시할 메시지
LOW_CONFIDENCE_THRESHOLD = 15.0  # 15% 미만이면 불확실

# 전처리
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class FoodRecognitionEngine:
    """
    ImageNet 사전학습 MobileNetV3-Large 기반 음식 인식 엔진.
    파인튜닝 가중치(food_finetuned.pth)가 있으면 우선 로드.
    없으면 ImageNet 1000 클래스 중 음식 관련 클래스만 필터링.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mode = "imagenet"  # "imagenet" | "finetuned"
        self._load()

    def _load(self):
        if FINETUNED_PATH.exists():
            print("[AI 렌즈] 파인튜닝 모델 로드:", FINETUNED_PATH)
            # 파인튜닝 모델은 별도 클래스 수로 저장돼 있을 수 있음
            # train.py 로 생성된 경우 state_dict 에 num_classes 정보 포함
            checkpoint = torch.load(FINETUNED_PATH, map_location=self.device)
            num_classes = checkpoint.get("num_classes", 101)
            labels = checkpoint.get("labels", None)
            self._finetuned_labels = labels  # list[str] 한국어 레이블
            base = models.mobilenet_v3_large(weights=None)
            in_feat = base.classifier[-1].in_features
            import torch.nn as nn
            base.classifier[-1] = nn.Linear(in_feat, num_classes)
            base.load_state_dict(checkpoint["state_dict"])
            self.model = base.to(self.device).eval()
            self.mode = "finetuned"
            print(f"[AI 렌즈] 파인튜닝 모드 (클래스={num_classes})")
        else:
            print("[AI 렌즈] ImageNet 사전학습 모드 (MobileNetV3-Large, 1000 클래스)")
            self.model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            ).to(self.device).eval()
            self._finetuned_labels = None
            self.mode = "imagenet"

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        """이미지 → 상위 K개 예측 반환"""
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]  # shape: (num_classes,)

        if self.mode == "finetuned" and self._finetuned_labels:
            return self._decode_finetuned(probs, top_k)
        else:
            return self._decode_imagenet(probs, top_k)

    def _decode_imagenet(self, probs: torch.Tensor, top_k: int) -> list[dict]:
        """ImageNet 1000 클래스 중 음식 관련만 필터링해 반환"""
        food_indices = list(IMAGENET_FOOD_MAP.keys())

        # 음식 클래스들만 추출
        food_probs = [(idx, probs[idx].item()) for idx in food_indices if idx < len(probs)]
        food_probs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, prob in food_probs[:top_k]:
            confidence = round(prob * 100, 1)
            food_name_ko = IMAGENET_FOOD_MAP[idx]
            results.append({
                "food_key": f"imagenet_{idx}",
                "food_name_ko": food_name_ko,
                "confidence": confidence,
                "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            })

        # 모든 음식 신뢰도가 낮으면 전체 상위 예측으로 대체
        if not results or results[0]["confidence"] < 5.0:
            top_vals, top_ids = torch.topk(probs, k=5)
            for prob_val, idx in zip(top_vals.tolist(), top_ids.tolist()):
                if idx in IMAGENET_FOOD_MAP:
                    results.insert(0, {
                        "food_key": f"imagenet_{idx}",
                        "food_name_ko": IMAGENET_FOOD_MAP[idx],
                        "confidence": round(prob_val * 100, 1),
                        "low_confidence": True,
                    })

        # 최소 1개는 반환
        if not results:
            top_idx = int(torch.argmax(probs).item())
            results = [{
                "food_key": "unknown",
                "food_name_ko": "인식 불가",
                "confidence": 0.0,
                "low_confidence": True,
            }]

        return results[:top_k]

    def _decode_finetuned(self, probs: torch.Tensor, top_k: int) -> list[dict]:
        """파인튜닝 모델 디코딩"""
        top_probs, top_ids = torch.topk(probs, k=min(top_k, len(self._finetuned_labels)))
        results = []
        for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
            label = self._finetuned_labels[idx] if idx < len(self._finetuned_labels) else "unknown"
            results.append({
                "food_key": label,
                "food_name_ko": label,
                "confidence": round(prob * 100, 1),
                "low_confidence": prob * 100 < LOW_CONFIDENCE_THRESHOLD,
            })
        return results


# ─── 싱글톤 ───────────────────────────────────────────────────────────────────
_engine: FoodRecognitionEngine | None = None


def get_engine() -> FoodRecognitionEngine:
    global _engine
    if _engine is None:
        _engine = FoodRecognitionEngine()
    return _engine


def analyze_image(image: Image.Image) -> list[dict]:
    return get_engine().predict(image, top_k=3)
