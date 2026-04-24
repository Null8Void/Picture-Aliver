# Picture-Aliver Mobile - Build Instructions

## Prerequisites

- Node.js 18+
- npm or yarn
- Expo CLI: `npm install -g expo-cli`
- EAS CLI (for building): `npm install -g eas-cli`
- Android Studio (for Android builds)
- Xcode (for iOS builds, Mac only)

---

## Development Setup

### 1. Install Dependencies

```bash
cd mobile_app
npm install
```

### 2. Run on Device/Emulator

```bash
# Start Metro bundler
npm start

# Android
npm run android

# iOS
npm run ios
```

### 3. Configure API URL

Edit `lib/services/api.ts` for your backend URL:

```typescript
export const API_CONFIG = {
  DEFAULT_URLS: {
    android: 'http://10.0.2.2:8000',      // Android emulator
    ios: 'http://localhost:8000',          // iOS simulator
    physical: 'http://192.168.1.x:8000', // Physical device (update IP)
  },
  // ...
};
```

---

## Building APK (Android)

### Method 1: Local Build (Recommended for Testing)

```bash
# 1. Generate native Android project
npx expo prebuild --platform android

# 2. Build debug APK
cd android
./gradlew assembleDebug

# Output: android/app/build/outputs/apk/debug/app-debug.apk
```

### Method 2: EAS Build (Cloud Build)

```bash
# 1. Login to EAS
eas login

# 2. Configure build
eas build:configure

# 3. Build for Android
eas build --platform android --profile preview

# Output: Download link from EAS
```

### Method 3: EAS Build with Local Credentials

```bash
# 1. Build locally with EAS
eas build --platform android --local

# This creates: android/app/build/outputs/apk/release/app-release.apk
```

---

## Building IPA (iOS)

### Prerequisites
- Apple Developer Account
- Mac with Xcode
- Proper provisioning profile

### Build Commands

```bash
# 1. Prebuild iOS project
npx expo prebuild --platform ios

# 2. Build with Xcode
xcodebuild -workspace ios/*.xcworkspace -scheme PictureAliver -configuration Release -archivePath build/PictureAliver.xcarchive archive
```

### EAS Build (Recommended)

```bash
# Build for App Store
eas build --platform ios --profile production

# Build for TestFlight
eas build --platform ios --profile preview
```

---

## Standalone APK (No Backend Required)

For fully offline operation, the APK bundles a local server:

### Current Architecture (WiFi Required)

```
Mobile App ──WiFi──> PC Backend ──GPU
```

### Future Architecture (Fully Offline)

```
Mobile App + Local Python Backend ──GPU on device
```

**Note:** True offline operation requires ONNX conversion of models,
which is beyond current scope. The current APK still requires
a backend server, but can connect to any device on the local network.

---

## APK Output Locations

After building, find your APK at:

| Method | Location |
|--------|----------|
| Local Debug | `android/app/build/outputs/apk/debug/app-debug.apk` |
| Local Release | `android/app/build/outputs/apk/release/app-release.apk` |
| EAS Build | Download link from EAS dashboard |

---

## Installing on Device

### Android

```bash
# Transfer APK to device
adb install android/app/build/outputs/apk/debug/app-debug.apk

# Or install via file manager on device
```

### iOS

```bash
# Open Xcode
open ios/*.xcworkspace

# Select your device and click Run
```

---

## Troubleshooting

### Metro Bundler Issues

```bash
# Clear cache
npm start -- --reset-cache

# Or
rm -rf node_modules/.cache
```

### Build Fails

```bash
# Clean and rebuild
rm -rf android/build android/app/build
npx expo prebuild --platform android --clean
cd android && ./gradlew clean
```

### Permissions Issues

```bash
# Check AndroidManifest.xml
cat android/app/src/main/AndroidManifest.xml
```

---

## EAS Build Configuration

The `eas.json` file configures builds:

```json
{
  "build": {
    "preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "android": {
        "buildType": "release"
      }
    }
  }
}
```

---

## Quick Build Commands

| Command | Description |
|---------|-------------|
| `npm start` | Start Metro dev server |
| `npm run android` | Run on Android |
| `npm run ios` | Run on iOS |
| `npx expo prebuild` | Generate native projects |
| `eas build:configure` | Setup EAS |
| `eas build --platform android` | Build Android APK |
| `eas build --platform ios` | Build iOS IPA |

---

## App Features

| Feature | Status |
|---------|--------|
| Image selection (gallery/camera) | Working |
| Image upload to backend | Working |
| Progress polling | Working |
| Video playback | Working |
| Offline operation | Requires backend |
| Local network discovery | Working |

---

## Need Help?

- Expo Documentation: https://docs.expo.dev
- EAS Build: https://docs.expo.dev/build/introduction
- React Native: https://reactnative.dev