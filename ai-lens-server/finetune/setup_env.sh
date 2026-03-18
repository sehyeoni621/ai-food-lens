#!/bin/bash
# 건강신호등 AI 렌즈 환경 설치 (Linux/Mac/WSL)
set -e

echo "============================================"
echo " 건강신호등 AI 렌즈 환경 설치"
echo "============================================"

conda create -n ai-lens python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ai-lens

echo "[1/3] PyTorch 설치..."
# GPU 있으면 CUDA 버전 자동 선택
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo "  → CUDA 버전 설치됨"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo "  → CPU 버전 설치됨 (GPU 없음)"
fi

echo "[2/3] 기타 패키지..."
pip install fastapi "uvicorn[standard]" python-multipart Pillow numpy \
            httpx pydantic python-dotenv aiofiles scikit-learn \
            matplotlib tqdm

echo "[3/3] 설치 확인..."
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "완료! 이후: conda activate ai-lens"
