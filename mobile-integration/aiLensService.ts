/**
 * AI 렌즈 API 서비스
 * 백엔드 FastAPI 서버와 통신
 */

// ─── 설정 ─────────────────────────────────────────────────────────────────
// 개발 시: 로컬 서버 주소 사용
// 배포 시: 실제 서버 주소로 변경
const AI_LENS_BASE_URL = "http://10.0.2.2:8000"; // Android 에뮬레이터 → 호스트 localhost
// const AI_LENS_BASE_URL = "http://localhost:8000"; // iOS 시뮬레이터
// const AI_LENS_BASE_URL = "https://your-server.com"; // 배포 서버

// ─── 타입 정의 ────────────────────────────────────────────────────────────

export interface FoodPrediction {
  food_key: string;
  food_name_ko: string;
  confidence: number; // 0~100%
}

export interface AINutritionResult {
  kcal: number;
  carbohydrate: number;
  protein: number;
  fat: number;
  sugar: number;
  sodium: number;
  fiber: number;
  serving_size: string;
  estimated: boolean;
  note?: string;
  source?: string;
}

export interface AIHealthResult {
  score: number;
  traffic_light: "green" | "yellow" | "red";
  emoji: string;
}

export interface AIAllergyResult {
  has_warning: boolean;
  triggered_allergens: string[];
  message: string;
}

export interface AIAnalyzeResponse {
  predictions: FoodPrediction[];
  top_food: string;
  nutrition: AINutritionResult;
  health: AIHealthResult;
  alternatives: string[];
  allergy?: AIAllergyResult;
}

export interface DailyNutritionSummary {
  total_kcal: number;
  total_carbohydrate: number;
  total_protein: number;
  total_fat: number;
  total_sugar: number;
  total_sodium: number;
  meal_count: number;
}

export interface MealRecommendation {
  food_name: string;
  kcal: number;
  protein: number;
  health_score: number;
  traffic_light: string;
  reason: string;
}

// ─── API 함수들 ───────────────────────────────────────────────────────────

/** 이미지 base64 → 음식 인식 + 영양 분석 */
export const analyzeFoodImage = async (
  imageBase64: string,
  options?: {
    userAllergies?: string[];
    healthConditions?: {
      diabetes?: boolean;
      hyperlipidemia?: boolean;
      hypertension?: boolean;
    };
  }
): Promise<AIAnalyzeResponse> => {
  const formData = new FormData();
  formData.append("image_base64", imageBase64);

  if (options?.userAllergies?.length) {
    formData.append("user_allergies", options.userAllergies.join(","));
  }
  if (options?.healthConditions) {
    formData.append("health_conditions", JSON.stringify(options.healthConditions));
  }

  const response = await fetch(`${AI_LENS_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `서버 오류 (${response.status})`);
  }

  return response.json();
};

/** 음식 이름으로 영양성분 직접 조회 */
export const getFoodNutrition = async (foodName: string) => {
  const response = await fetch(
    `${AI_LENS_BASE_URL}/nutrition/${encodeURIComponent(foodName)}`
  );
  if (!response.ok) throw new Error("영양 정보 조회 실패");
  return response.json();
};

/** 알러지 체크 */
export const checkFoodAllergy = async (
  foodNameKo: string,
  userAllergies: string[]
): Promise<AIAllergyResult> => {
  const response = await fetch(`${AI_LENS_BASE_URL}/agents/allergy-check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ food_name_ko: foodNameKo, user_allergies: userAllergies }),
  });
  if (!response.ok) throw new Error("알러지 체크 실패");
  return response.json();
};

/** 일일 영양 분석 */
export const analyzeDailyNutrition = async (summary: DailyNutritionSummary) => {
  const response = await fetch(`${AI_LENS_BASE_URL}/agents/daily-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ summary }),
  });
  if (!response.ok) throw new Error("일일 분석 실패");
  return response.json();
};

/** 식사 추천 */
export const getMealRecommendations = async (
  summary: DailyNutritionSummary,
  mealType: "breakfast" | "lunch" | "dinner" | "snack" = "dinner"
): Promise<{ recommendations: MealRecommendation[] }> => {
  const response = await fetch(`${AI_LENS_BASE_URL}/agents/meal-recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ summary, meal_type: mealType }),
  });
  if (!response.ok) throw new Error("식사 추천 실패");
  return response.json();
};

/** 식습관 트렌드 분석 */
export const analyzeFoodTrend = async (scanHistory: any[]) => {
  const formattedHistory = scanHistory.map((item) => ({
    traffic_light: item.trafficLight,
    kcal: parseFloat(item.nutritionInfo?.kcal || "0"),
    food_name: item.productInfo?.productName || "",
  }));

  const response = await fetch(`${AI_LENS_BASE_URL}/agents/trend-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scan_history: formattedHistory }),
  });
  if (!response.ok) throw new Error("트렌드 분석 실패");
  return response.json();
};

/** 대체 음식 추천 */
export const getAlternatives = async (foodName: string) => {
  const response = await fetch(
    `${AI_LENS_BASE_URL}/agents/alternatives/${encodeURIComponent(foodName)}`
  );
  if (!response.ok) throw new Error("대체 음식 조회 실패");
  return response.json();
};

/** 서버 상태 확인 */
export const checkServerHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${AI_LENS_BASE_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return response.ok;
  } catch {
    return false;
  }
};
