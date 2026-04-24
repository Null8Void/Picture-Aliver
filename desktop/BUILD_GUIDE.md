# Picture-Aliver - Build Scripts and Shortcuts

## Quick Start

### PyQt5 Desktop App

```batch
# Run directly (development)
python desktop/pyqt/main.py

# Build executable
pyinstaller desktop/pyqt/build.spec --noconfirm

# Output: dist/Picture-Aliver/Picture-Aliver.exe
```

### Electron Desktop App

```batch
# Install dependencies
cd desktop/electron
npm install

# Run in development
npm run dev

# Build for Windows
npm run build:win

# Output: desktop/electron/release/Picture-Aliver Setup.exe
```

---

## Windows Shortcuts

### Create Desktop Shortcut (PyQt)

```batch
powershell -Command "$s = (New-Object -COM WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\Picture-Aliver.lnk'); $s.TargetPath = 'python'; $s.Arguments = 'desktop\pyqt\main.py'; $s.WorkingDirectory = '%CD%'; $s.Save()"
```

### Create Desktop Shortcut (Electron)

```batch
powershell -Command "$s = (New-Object -COM WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\Picture-Aliver-Electron.lnk'); $s.TargetPath = 'desktop\electron\release\Picture-Aliver Setup.exe'; $s.WorkingDirectory = 'desktop\electron'; $s.Save()"
```

---

## Build Commands Reference

### PyQt5

| Command | Description |
|---------|-------------|
| `pyinstaller desktop/pyqt/build.spec` | Build executable |
| `pyinstaller desktop/pyqt/build.spec --noconfirm` | Build without prompts |
| `pyinstaller desktop/pyqt/build.spec --clean` | Clean and rebuild |
| `python desktop/pyqt/main.py` | Run without building |

### Electron

| Command | Description |
|---------|-------------|
| `npm run start` | Run in development |
| `npm run build:win` | Build for Windows |
| `npm run build:mac` | Build for Mac |
| `npm run build:linux` | Build for Linux |
| `npx electron-builder --win portable` | Build portable exe |

---

## Running Both Apps

### Method 1: PyQt5 (Native Python)

```batch
# 1. Install dependencies
pip install PyQt5>=5.15.0

# 2. Run the app
python desktop/pyqt/main.py
```

### Method 2: Electron (Web-based)

```batch
# 1. Install Node.js dependencies
cd desktop/electron
npm install

# 2. Run in development
npm run dev

# 3. Or build and run the executable
npm run build:win
desktop\electron\release\win-unpacked\Picture-Aliver.exe
```

---

## Build Outputs

```
desktop/
├── pyqt/
│   ├── main.py              # Source code
│   ├── build.spec           # PyInstaller config
│   └── dist/
│       └── Picture-Aliver/  # Built executable
│           └── Picture-Aliver.exe
│
└── electron/
    ├── src/
    │   ├── main.js         # Electron main process
    │   ├── preload.js     # Preload script
    │   └── index.html      # Renderer UI
    └── release/
        └── win-unpacked/   # Built Electron app
            └── Picture-Aliver.exe
```

---

## Troubleshooting Builds

### PyQt5 Build Issues

**Problem:** PyInstaller can't find PyQt5
```batch
pip install pyinstaller
pyinstaller desktop/pyqt/build.spec --collect-all PyQt5
```

**Problem:** Missing DLLs on Windows
```batch
# Install Visual C++ Redistributable
winget install Microsoft.VCRedist.2015+.x64
```

### Electron Build Issues

**Problem:** npm install fails
```batch
# Clear cache and retry
npm cache clean --force
rm -rf node_modules
npm install
```

**Problem:** Build fails on Windows
```batch
# Install electron-builder globally
npm install -g electron-builder
npm run build:win
```

---

## Automated Build Script

Create `build_all.bat`:

```batch
@echo off
echo ========================================
echo Picture-Aliver Build Script
echo ========================================

echo.
echo [1/3] Building PyQt5 App...
pyinstaller desktop/pyqt/build.spec --noconfirm
if errorlevel 1 (
    echo PyQt5 build FAILED
    goto :end
)
echo PyQt5 build SUCCESS

echo.
echo [2/3] Building Electron App...
cd desktop\electron
call npm install
call npm run build:win
cd ..\..
if errorlevel 1 (
    echo Electron build FAILED
    goto :end
)
echo Electron build SUCCESS

echo.
echo [3/3] Copying executables...
if not exist "dist" mkdir dist
copy "desktop\pyqt\dist\Picture-Aliver\Picture-Aliver.exe" "dist\"
copy "desktop\electron\release\win-unpacked\Picture-Aliver.exe" "dist\"

echo.
echo ========================================
echo BUILD COMPLETE
echo ========================================
echo Output files in:
echo   - dist\Picture-Aliver.exe (PyQt5)
echo   - dist\Picture-Aliver.exe (Electron)
echo ========================================

:end
pause
```

Run with:
```batch
build_all.bat
```

---

## Version Information

- PyQt5 App: Python 3.9+ with PyQt5
- Electron App: Node.js 18+ with Electron 28

Both apps require the main Picture-Aliver dependencies from `requirements.txt`.