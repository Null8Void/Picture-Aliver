# Picture-Aliver - Running Instructions

Complete guide to running the Picture-Aliver application on all supported platforms.

---

## Quick Start Commands

### PyQt5 Desktop App
```bash
python desktop/pyqt/main.py
```

### Electron Desktop App
```bash
cd desktop/electron
npm start
```

### Mobile App (React Native/Expo)
```bash
cd mobile_app
npm start
```

### API Server Only
```bash
python -m uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000
```

---

## Platform Details

### PyQt5 Desktop App

The PyQt5 desktop application provides a full GUI for image-to-video generation with:
- Image selection and preview
- Prompt input
- Video settings (duration, FPS, resolution)
- Generation progress
- Video playback

#### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install PyQt5 dependencies  
cd desktop/pyqt
pip install -r requirements.txt
```

#### Running
```bash
# From project root
python desktop/pyqt/main.py

# Or with validation first
python -c "from src.picture_aliver.validate import validate_early"
python desktop/pyqt/main.py
```

#### Building Executable
```bash
cd desktop/pyqt
pyinstaller build.spec --noconfirm
```

Output: `dist/Picture-Aliver/Picture-Aliver.exe`

---

### Electron Desktop App

The Electron app provides a web-based UI with a local backend server.

#### Prerequisites
```bash
# Install Node dependencies
cd desktop/electron
npm install
```

#### Running
```bash
cd desktop/electron
npm start
```

This will:
1. Start the FastAPI backend server on port 8000
2. Launch the Electron application window

#### Building
```bash
cd desktop/electron
npm run build:win    # Windows
npm run build:mac   # macOS
npm run build:linux # Linux
```

Output: `desktop/electron/release/`

---

### Mobile App

The React Native/Expo mobile app for Android and iOS.

#### Prerequisites
```bash
# Install Node dependencies
cd mobile_app
npm install

# Install Expo CLI (if needed)
npm install -g expo-cli
```

#### Running

**Development Server:**
```bash
cd mobile_app
npm start
```

**Android (Emulator):**
```bash
cd mobile_app
npm run android
```

**iOS (Simulator):**
```bash
cd mobile_app
npm run ios
```

**Physical Device:**
1. Ensure phone and computer on same WiFi
2. Update API URL in `mobile_app/lib/services/api.ts`:
   ```typescript
   physical: 'http://YOUR_PC_IP:8000'
   ```
3. Run `npm start`
4. Scan QR code with Expo Go app

#### Building APK
```bash
cd mobile_app
npm run prebuild:android
cd android && ./gradlew assembleRelease
```

---

## API Server

The backend API can run standalone for integration with other clients.

#### Running
```bash
# Default (all interfaces, port 8000)
python -m uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000

# Development with auto-reload
python -m uvicorn src.picture_aliver.api:app --reload --host 0.0.0.0 --port 8000

# Custom port
python -m uvicorn src.picture_aliver.api:app --port 8080
```

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Generate Video
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@your_image.jpg" \
  -F "prompt=gentle wave animation" \
  -F "duration=3" \
  -F "fps=8"
```

#### Check Task Status
```bash
curl http://localhost:8000/tasks/YOUR_TASK_ID
```

#### Download Video
```bash
curl -O http://localhost:8000/download/YOUR_TASK_ID
```

---

## Environment Setup

### Virtual Environment (Recommended)
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Setup

**CUDA (NVIDIA):**
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**ROCm (AMD):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm
```

### Environment Variables
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model cache
export HF_HOME=./models/cache
export TRANSFORMERS_CACHE=./models/cache/transformers
```

---

## Validation

Run startup validation to check for issues:
```bash
python -c "from src.picture_aliver.validate import validate_early; validate_early()"
```

This checks:
- Python version (3.9+)
- Required dependencies
- Directory structure
- GPU availability
- Pipeline imports

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| Import errors | Validate with `python -c "from src.picture_aliver.validate import validate_early"` |
| GPU not detected | Install CUDA version of PyTorch |
| Port in use | Use `--port 8080` or `--port 8001` |
| Electron not starting | Run `npm install` in `desktop/electron/` |
| Mobile app can't connect | Update IP in `mobile_app/lib/services/api.ts` |
| Image loading fails | Check file format (PNG, JPG, WEBP supported) |
| Out of memory | Reduce resolution or FPS |

---

## Common Commands Reference

| Command | Description |
|---------|-------------|
| `python desktop/pyqt/main.py` | Run PyQt5 app |
| `python -m uvicorn src.picture_aliver.api:app` | Run API server |
| `cd desktop/electron && npm start` | Run Electron app |
| `cd mobile_app && npm start` | Run mobile dev server |
| `python -m tests.testing_workflow` | Run test suite |
| `pip install -r requirements.txt` | Install dependencies |