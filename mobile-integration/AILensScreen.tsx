/**
 * AI 렌즈 스크린
 * 음식 사진 촬영/선택 → 자체 AI 분석 → 영양성분 + 건강신호등 표시
 *
 * 이 파일을 C:\Heathl-Traffic-main\src\screens\AILensScreen.tsx 에 복사하세요.
 */
import React, { useState, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  ScrollView,
  Image,
  Platform,
  Dimensions,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { useAppContext } from "../context/AppContext";
import { POINTS_CONFIG, TRAFFIC_LIGHT_COLORS } from "../constants";
import { saveScanHistory } from "../utils/storage";
import {
  analyzeFoodImage,
  checkServerHealth,
  type AIAnalyzeResponse,
  type AINutritionResult,
} from "../services/aiLensService";
import { FoodItem } from "../types";

const { width } = Dimensions.get("window");

type ViewMode = "camera" | "preview" | "result";

const TRAFFIC_EMOJI: Record<string, string> = {
  green: "🟢",
  yellow: "🟡",
  red: "🔴",
};

const TRAFFIC_LABEL: Record<string, string> = {
  green: "건강해요",
  yellow: "보통이에요",
  red: "주의하세요",
};

// ─── 영양 정보 카드 컴포넌트 ────────────────────────────────────────────────
const NutritionCard: React.FC<{ label: string; value: string; unit: string; color?: string }> = ({
  label, value, unit, color = "#2D3436",
}) => (
  <View style={styles.nutritionCard}>
    <Text style={[styles.nutritionValue, { color }]}>{value}</Text>
    <Text style={styles.nutritionUnit}>{unit}</Text>
    <Text style={styles.nutritionLabel}>{label}</Text>
  </View>
);

// ─── 메인 컴포넌트 ───────────────────────────────────────────────────────────
const AILensScreen: React.FC = () => {
  const navigation = useNavigation();
  const { points, userPlan, deductPoints, userProfile, showToast } = useAppContext();
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const [mode, setMode] = useState<ViewMode>("camera");
  const [capturedUri, setCapturedUri] = useState<string | null>(null);
  const [capturedBase64, setCapturedBase64] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeMsg, setAnalyzeMsg] = useState("");
  const [result, setResult] = useState<AIAnalyzeResponse | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);

  // AI 렌즈 포인트 비용
  const aiCost =
    userPlan === "premium"
      ? POINTS_CONFIG.AI_LENS_COST_PREMIUM
      : POINTS_CONFIG.AI_LENS_COST_FREE;

  // 서버 상태 체크
  const checkServer = useCallback(async () => {
    const ok = await checkServerHealth();
    setServerOnline(ok);
    return ok;
  }, []);

  // 사진 촬영
  const takePicture = async () => {
    if (!cameraRef.current) return;

    const ok = await checkServer();
    if (!ok) {
      Alert.alert(
        "AI 서버 오프라인",
        "AI 렌즈 서버가 실행되지 않고 있습니다.\n\nai-lens-server 폴더에서:\nuvicorn main:app --host 0.0.0.0 --port 8000\n을 실행해주세요.",
        [{ text: "확인" }]
      );
      return;
    }

    if (points < aiCost) {
      Alert.alert(
        "포인트 부족",
        `AI 렌즈 사용에 ${aiCost}P가 필요합니다.\n현재 포인트: ${points}P`,
        [{ text: "확인" }]
      );
      return;
    }

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        base64: true,
      });
      if (!photo) return;
      setCapturedUri(photo.uri);
      setCapturedBase64(photo.base64 || null);
      setMode("preview");
    } catch (e) {
      Alert.alert("오류", "사진 촬영에 실패했습니다");
    }
  };

  // AI 분석 실행
  const runAnalysis = async () => {
    if (!capturedBase64) return;

    setIsAnalyzing(true);
    setAnalyzeMsg("🤖 AI가 음식을 인식하고 있어요...");

    try {
      // 포인트 차감
      await deductPoints(aiCost, "AI 렌즈 사용");

      setAnalyzeMsg("🔍 영양성분을 분석하고 있어요...");

      const response = await analyzeFoodImage(
        capturedBase64,
        {
          userAllergies: userProfile?.allergies,
          healthConditions: userProfile?.healthConditions,
        }
      );

      setResult(response);
      setMode("result");

      // 히스토리에 저장
      const foodItem: FoodItem = {
        id: `ai_${Date.now()}`,
        productInfo: {
          barcode: "AI_LENS",
          productName: response.top_food,
          manufacturer: "AI 렌즈 인식",
          reportNumber: "",
          category: "AI 촬영",
          expiryInfo: "",
        },
        nutritionInfo: {
          kcal: String(response.nutrition.kcal),
          carbohydrate: String(response.nutrition.carbohydrate),
          protein: String(response.nutrition.protein),
          fat: String(response.nutrition.fat),
          sugar: String(response.nutrition.sugar),
          sodium: String(response.nutrition.sodium),
          servingSize: response.nutrition.serving_size,
        },
        healthScore: response.health.score,
        trafficLight: response.health.traffic_light,
        scannedAt: new Date(),
      };
      await saveScanHistory(foodItem);

      if (response.health.traffic_light === "green") {
        showToast?.("🟢 건강한 선택이에요! 포인트 적립됩니다");
      }
    } catch (e: any) {
      Alert.alert("분석 실패", e.message || "AI 분석 중 오류가 발생했습니다", [
        { text: "다시 시도", onPress: () => setMode("camera") },
      ]);
    } finally {
      setIsAnalyzing(false);
      setAnalyzeMsg("");
    }
  };

  // 다시 찍기
  const retake = () => {
    setCapturedUri(null);
    setCapturedBase64(null);
    setResult(null);
    setMode("camera");
  };

  // ── 권한 화면 ─────────────────────────────────────────────────────────────
  if (!permission) return <View style={styles.center}><ActivityIndicator color="#8DB38D" /></View>;
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.emoji}>📸</Text>
        <Text style={styles.errorTitle}>카메라 권한 필요</Text>
        <Text style={styles.errorSub}>AI 렌즈를 사용하려면 카메라 권한이 필요합니다</Text>
        <TouchableOpacity style={styles.primaryBtn} onPress={requestPermission}>
          <Text style={styles.primaryBtnText}>권한 허용하기</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── 결과 화면 ─────────────────────────────────────────────────────────────
  if (mode === "result" && result) {
    const { top_food, predictions, nutrition, health, alternatives, allergy } = result;
    const trafficColor = TRAFFIC_LIGHT_COLORS[health.traffic_light as keyof typeof TRAFFIC_LIGHT_COLORS];

    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.resultContent}>
        {/* 헤더 */}
        <View style={[styles.resultHeader, { backgroundColor: trafficColor }]}>
          <Text style={styles.trafficEmoji}>{health.emoji}</Text>
          <Text style={styles.topFoodName}>{top_food}</Text>
          <Text style={styles.trafficLabel}>{TRAFFIC_LABEL[health.traffic_light]}</Text>
          <Text style={styles.healthScore}>{health.score}점</Text>
        </View>

        {/* 촬영 이미지 */}
        {capturedUri && (
          <Image source={{ uri: capturedUri }} style={styles.capturedImage} resizeMode="cover" />
        )}

        {/* AI 예측 결과 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🤖 AI 인식 결과</Text>
          {predictions.map((p, i) => (
            <View key={p.food_key} style={styles.predictionRow}>
              <Text style={styles.predictionRank}>#{i + 1}</Text>
              <Text style={styles.predictionName}>{p.food_name_ko}</Text>
              <View style={styles.confidenceBar}>
                <View style={[styles.confidenceFill, { width: `${p.confidence}%` }]} />
              </View>
              <Text style={styles.confidenceText}>{p.confidence.toFixed(1)}%</Text>
            </View>
          ))}
        </View>

        {/* 영양성분 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📊 영양성분 ({nutrition.serving_size})</Text>
          <View style={styles.nutritionGrid}>
            <NutritionCard label="칼로리" value={String(Math.round(nutrition.kcal))} unit="kcal" color="#E74C3C" />
            <NutritionCard label="탄수화물" value={nutrition.carbohydrate.toFixed(1)} unit="g" />
            <NutritionCard label="단백질" value={nutrition.protein.toFixed(1)} unit="g" color="#2ECC71" />
            <NutritionCard label="지방" value={nutrition.fat.toFixed(1)} unit="g" color="#F39C12" />
            <NutritionCard label="당류" value={nutrition.sugar.toFixed(1)} unit="g" color="#E74C3C" />
            <NutritionCard label="나트륨" value={String(Math.round(nutrition.sodium))} unit="mg" color="#9B59B6" />
          </View>
          {nutrition.estimated && (
            <Text style={styles.estimatedNote}>⚠️ {nutrition.note}</Text>
          )}
        </View>

        {/* 알러지 경고 */}
        {allergy?.has_warning && (
          <View style={[styles.section, styles.allergyWarning]}>
            <Text style={styles.sectionTitle}>⚠️ 알러지 주의</Text>
            <Text style={styles.allergyText}>{allergy.message}</Text>
          </View>
        )}

        {/* 대체 음식 추천 */}
        {alternatives.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>💡 건강한 대체 음식</Text>
            <View style={styles.altFoodRow}>
              {alternatives.map((alt) => (
                <View key={alt} style={styles.altFoodChip}>
                  <Text style={styles.altFoodText}>{alt}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* 버튼 */}
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.secondaryBtn} onPress={retake}>
            <Ionicons name="camera" size={20} color="#4CAF50" />
            <Text style={styles.secondaryBtnText}>다시 찍기</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.primaryBtn}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.primaryBtnText}>완료</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    );
  }

  // ── 미리보기 화면 ─────────────────────────────────────────────────────────
  if (mode === "preview" && capturedUri) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: capturedUri }} style={styles.previewImage} resizeMode="cover" />

        {isAnalyzing ? (
          <View style={styles.analyzingOverlay}>
            <ActivityIndicator size="large" color="#C1E1C1" />
            <Text style={styles.analyzingText}>{analyzeMsg}</Text>
          </View>
        ) : (
          <View style={styles.previewButtons}>
            <TouchableOpacity style={styles.retakeBtn} onPress={retake}>
              <Ionicons name="refresh" size={24} color="white" />
              <Text style={styles.retakeBtnText}>다시 찍기</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.analyzeBtn} onPress={runAnalysis}>
              <Text style={styles.analyzeBtnText}>
                🤖 AI 분석하기 ({aiCost}P)
              </Text>
            </TouchableOpacity>
          </View>
        )}

        <TouchableOpacity style={styles.closeBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close-circle" size={40} color="white" />
        </TouchableOpacity>
      </View>
    );
  }

  // ── 카메라 화면 ───────────────────────────────────────────────────────────
  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        {/* 오버레이 가이드 */}
        <View style={styles.cameraOverlay}>
          <View style={styles.guideBox}>
            <View style={[styles.corner, styles.tl]} />
            <View style={[styles.corner, styles.tr]} />
            <View style={[styles.corner, styles.bl]} />
            <View style={[styles.corner, styles.br]} />
            <Text style={styles.guideText}>음식을 화면 안에 맞춰주세요</Text>
          </View>
        </View>

        {/* 상단 정보 */}
        <View style={styles.topInfo}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="close-circle" size={40} color="white" />
          </TouchableOpacity>
          <View style={styles.pointsBadge}>
            <Text style={styles.pointsText}>보유 {points}P | AI 렌즈 {aiCost}P</Text>
          </View>
        </View>

        {/* 서버 상태 표시 */}
        {serverOnline === false && (
          <View style={styles.serverWarning}>
            <Text style={styles.serverWarningText}>⚠️ AI 서버 오프라인 – 서버를 먼저 실행해주세요</Text>
          </View>
        )}

        {/* 촬영 버튼 */}
        <View style={styles.shutterContainer}>
          <TouchableOpacity style={styles.shutterBtn} onPress={takePicture}>
            <View style={styles.shutterInner} />
          </TouchableOpacity>
          <Text style={styles.shutterLabel}>촬영</Text>
        </View>
      </CameraView>
    </View>
  );
};

// ─── 스타일 ────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#F7F9F7", padding: 30 },
  camera: { flex: 1 },

  // 카메라 오버레이
  cameraOverlay: { flex: 1, justifyContent: "center", alignItems: "center" },
  guideBox: {
    width: width * 0.85, height: width * 0.85,
    position: "relative", justifyContent: "flex-end", alignItems: "center", paddingBottom: 20,
  },
  guideText: { color: "rgba(255,255,255,0.9)", fontSize: 15, fontWeight: "600" },
  corner: { position: "absolute", width: 32, height: 32, borderColor: "#C1E1C1", borderWidth: 4 },
  tl: { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0, borderTopLeftRadius: 8 },
  tr: { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0, borderTopRightRadius: 8 },
  bl: { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0, borderBottomLeftRadius: 8 },
  br: { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0, borderBottomRightRadius: 8 },

  // 상단 정보
  topInfo: {
    position: "absolute", top: 50, left: 20, right: 20,
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
  },
  pointsBadge: {
    backgroundColor: "rgba(0,0,0,0.6)", paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20,
  },
  pointsText: { color: "#C1E1C1", fontSize: 13, fontWeight: "700" },

  serverWarning: {
    position: "absolute", top: 110, left: 20, right: 20,
    backgroundColor: "rgba(231,76,60,0.85)", borderRadius: 12, padding: 10, alignItems: "center",
  },
  serverWarningText: { color: "white", fontSize: 13, fontWeight: "600" },

  // 셔터
  shutterContainer: { position: "absolute", bottom: 60, width: "100%", alignItems: "center" },
  shutterBtn: {
    width: 80, height: 80, borderRadius: 40, backgroundColor: "rgba(255,255,255,0.3)",
    justifyContent: "center", alignItems: "center", borderWidth: 3, borderColor: "white",
  },
  shutterInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: "white" },
  shutterLabel: { color: "white", marginTop: 10, fontSize: 14, fontWeight: "600" },

  // 미리보기
  previewImage: { flex: 1 },
  analyzingOverlay: {
    position: "absolute", bottom: 0, left: 0, right: 0, height: 180,
    backgroundColor: "rgba(0,0,0,0.8)", justifyContent: "center", alignItems: "center",
  },
  analyzingText: { color: "#C1E1C1", fontSize: 18, fontWeight: "700", marginTop: 16 },
  previewButtons: {
    position: "absolute", bottom: 40, left: 20, right: 20,
    flexDirection: "row", gap: 12,
  },
  retakeBtn: {
    flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.2)", borderRadius: 16, paddingVertical: 16, gap: 8,
  },
  retakeBtnText: { color: "white", fontSize: 16, fontWeight: "700" },
  analyzeBtn: {
    flex: 2, backgroundColor: "#4CAF50", borderRadius: 16,
    paddingVertical: 16, alignItems: "center",
  },
  analyzeBtnText: { color: "white", fontSize: 16, fontWeight: "800" },
  closeBtn: { position: "absolute", top: 50, right: 20 },

  // 결과 화면
  resultContent: { paddingBottom: 40 },
  resultHeader: {
    alignItems: "center", paddingVertical: 32, paddingHorizontal: 20,
  },
  trafficEmoji: { fontSize: 60 },
  topFoodName: { fontSize: 28, fontWeight: "800", color: "white", marginTop: 8 },
  trafficLabel: { fontSize: 18, color: "rgba(255,255,255,0.9)", marginTop: 4 },
  healthScore: { fontSize: 42, fontWeight: "900", color: "white", marginTop: 8 },
  capturedImage: { width: "100%", height: 220 },

  section: { margin: 16, backgroundColor: "white", borderRadius: 20, padding: 20 },
  sectionTitle: { fontSize: 17, fontWeight: "700", color: "#2D3436", marginBottom: 16 },

  // AI 예측
  predictionRow: { flexDirection: "row", alignItems: "center", marginBottom: 10, gap: 8 },
  predictionRank: { fontSize: 14, fontWeight: "700", color: "#636E72", width: 24 },
  predictionName: { fontSize: 15, fontWeight: "600", color: "#2D3436", width: 90 },
  confidenceBar: {
    flex: 1, height: 8, backgroundColor: "#F0F0F0", borderRadius: 4, overflow: "hidden",
  },
  confidenceFill: { height: "100%", backgroundColor: "#4CAF50", borderRadius: 4 },
  confidenceText: { fontSize: 13, fontWeight: "600", color: "#636E72", width: 45, textAlign: "right" },

  // 영양성분 그리드
  nutritionGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  nutritionCard: {
    width: (width - 72) / 3, backgroundColor: "#F8F9FA", borderRadius: 14,
    padding: 12, alignItems: "center",
  },
  nutritionValue: { fontSize: 20, fontWeight: "800" },
  nutritionUnit: { fontSize: 11, color: "#636E72", marginTop: 2 },
  nutritionLabel: { fontSize: 12, fontWeight: "600", color: "#636E72", marginTop: 4 },
  estimatedNote: { fontSize: 12, color: "#E67E22", marginTop: 12, lineHeight: 18 },

  // 알러지
  allergyWarning: { borderWidth: 2, borderColor: "#E74C3C" },
  allergyText: { fontSize: 15, color: "#C0392B", fontWeight: "600" },

  // 대체 음식
  altFoodRow: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  altFoodChip: {
    backgroundColor: "#E8F5E9", paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20,
  },
  altFoodText: { color: "#2D6A2D", fontSize: 14, fontWeight: "600" },

  // 버튼
  buttonRow: { flexDirection: "row", gap: 12, margin: 16 },
  primaryBtn: {
    flex: 1, backgroundColor: "#4CAF50", borderRadius: 16,
    paddingVertical: 16, alignItems: "center",
  },
  primaryBtnText: { color: "white", fontSize: 17, fontWeight: "800" },
  secondaryBtn: {
    flex: 1, borderWidth: 2, borderColor: "#4CAF50", borderRadius: 16,
    paddingVertical: 16, alignItems: "center", flexDirection: "row",
    justifyContent: "center", gap: 8,
  },
  secondaryBtnText: { color: "#4CAF50", fontSize: 17, fontWeight: "700" },

  // 권한 에러
  emoji: { fontSize: 60, marginBottom: 16 },
  errorTitle: { fontSize: 22, fontWeight: "800", color: "#2D3436", marginBottom: 8 },
  errorSub: { fontSize: 15, color: "#636E72", textAlign: "center", marginBottom: 24, lineHeight: 22 },
});

export default AILensScreen;
