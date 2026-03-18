@echo off
:: 건강신호등 AI 렌즈 환경 설치 (Windows)
:: 실행: setup_env.bat

echo ============================================
echo  건강신호등 AI 렌즈 환경 설치
echo ============================================

:: conda 환경 생성 (Python 3.10)
conda create -n ai-lens python=3.10 -y
call conda activate ai-lens

:: PyTorch (CUDA 12.1 지원, GPU 없으면 CPU 버전)
echo.
echo [1/3] PyTorch 설치 중...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 나머지 패키지
echo.
echo [2/3] 기타 패키지 설치 중...
pip install fastapi uvicorn[standard] python-multipart Pillow numpy httpx pydantic python-dotenv aiofiles scikit-learn matplotlib tqdm

:: 설치 확인
echo.
echo [3/3] 설치 확인...
python -c "import torch; print('PyTorch:', torch.__version__, '/ CUDA:', torch.cuda.is_available())"

echo.
echo ============================================
echo  설치 완료!
echo  이후 실행: conda activate ai-lens
echo ============================================
pause
