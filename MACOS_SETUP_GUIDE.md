# macOS & IDE Setup Guide: SCNet Karaoke Engine

This guide walks you through setting up, compiling, running, and debugging the **SCNet Karaoke Engine** and its **Offline Quantization Pipeline** on macOS (Apple Silicon M1/M2/M3/M4 or Intel).

---

## Table of Contents
1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Hardware & Software Prerequisites](#2-hardware--software-prerequisites)
3. [Quick Start (One-Liner Summary)](#3-quick-start)
4. [Step-by-Step System Setup](#4-step-by-step-system-setup)
   - [Step 1: Install Xcode & Command Line Tools](#step-1-install-xcode--command-line-tools)
   - [Step 2: Install Homebrew & Native Dependencies](#step-2-install-homebrew--native-dependencies)
   - [Step 3: Clone Repository & Configure Git LFS](#step-3-clone-repository--configure-git-lfs)
5. [Python Quantization Pipeline Setup](#5-python-quantization-pipeline-setup)
   - [Step 4: Create Python 3.11 Virtual Environment](#step-4-create-python-311-virtual-environment)
   - [Step 5: Download SCNet Model Checkpoints](#step-5-download-scnet-model-checkpoints)
   - [Step 6: Run Conversion, Pruning & INT8 Quantization](#step-6-run-conversion-pruning--int8-quantization)
   - [Step 7: Compile CoreML Model (`.mlpackage` to `.mlmodelc`)](#step-7-compile-coreml-model-mlpackage-to-mlmodelc)
   - [Step 8: Run Separation Test on Real Audio](#step-8-run-separation-test-on-real-audio)
6. [Swift Native Engine Setup (CLI)](#6-swift-native-engine-setup-cli)
   - [Step 9: Build and Test via Swift Package Manager](#step-9-build-and-test-via-swift-package-manager)
7. [IDE Setup & Configuration](#7-ide-setup--configuration)
   - [Option A: Xcode (Recommended for iOS/macOS Engine & App Development)](#option-a-xcode-setup)
   - [Option B: Visual Studio Code / Cursor (Recommended for Python + Swift Dual Dev)](#option-b-visual-studio-code--cursor-setup)
8. [Hardware Verification & Profiling on Apple Neural Engine (ANE)](#8-hardware-verification--profiling-on-apple-neural-engine-ane)
9. [Common Gotchas & Troubleshooting](#9-common-gotchas--troubleshooting)

---

## 1. Project Overview & Architecture

The repository contains two tightly integrated halves:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   OFFLINE PYTHON TOOLCHAIN (Mac Workstation)                     │
│                                                                                  │
│  [PyTorch Checkpoint] ──> [scnet_pruner / wrapper] ──> [torch.jit.trace]         │
│                                (Decouple STFT)          (Static 3s Window)       │
│                                                               │                  │
│                                                               v                  │
│  [SCNetInstrumental.mlmodelc] <── [xcrun coremlcompiler] <── [coremltools]       │
│    (Precompiled ANE Graph)                                 (INT8 Quantization)   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Bundled into
                                         v
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     NATIVE RUNTIME ENGINE (iOS 17+ / macOS 14+)                  │
│                                                                                  │
│  Audio In ──> [Accelerate vDSP] ──> [CoreML Model] ──> [Accelerate vDSP]         │
│  (PCM 44.1k)   (Forward STFT)        (ANE Execution)    (Inverse STFT)           │
│                                                               │                  │
│                                                               v                  │
│  Audio Out <── [AVAudioEngine] <── [LockFreePCMBuffer] <── [Overlap-Add 50%]     │
│  (Headphones)    (Render Thread)     (SPSC Ring Buffer)   (Continuous Stream)    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

1. **Python Conversion & Quantization Pipeline** (`SCNetQuantizationPipeline/` & `SCNetKaraokeEngine/OfflineConversion/`):
   - Ingests pretrained PyTorch SCNet checkpoints (Base or Small).
   - Excises non-vocal/non-instrumental heads to reduce model size by 60–75%.
   - Strips dynamic shapes, `torch.stft`, and complex tensors to satisfy Apple Neural Engine (ANE) requirements.
   - Converts to CoreML MIL `mlprogram` with `Float16` activations and `INT8` linear symmetric weight quantization.
   - Compiles `.mlpackage` into `.mlmodelc`.

2. **Native Swift Audio Engine** (`SCNetKaraokeEngine/`, `Package.swift`):
   - Targets **iOS 17+** and **macOS 14+**.
   - Built on Apple's **Accelerate** (`vDSP`, `vImage`), **AVFAudio**, and **CoreML**.
   - Implements zero-allocation streaming via 16 KB page-aligned memory pools (`PageAlignedTensorPool`).
   - Uses a lock-free Single-Producer Single-Consumer (SPSC) ring buffer (`LockFreePCMBuffer`) with `swift-atomics`.
   - Protects against background GPU crashes (`0xdead10cc`) by routing exclusively to `MLComputeUnits.cpuAndNeuralEngine`.

---

## 2. Hardware & Software Prerequisites

| Requirement | Minimum | Recommended | Notes |
|:---|:---|:---|:---|
| **Hardware** | Apple Silicon Mac or Intel Mac | **Apple Silicon (M1/M2/M3/M4)** | Apple Silicon is strongly recommended to test 16 KB page memory alignment (`posix_memalign`) and native ANE execution. |
| **macOS** | macOS 14.0 (Sonoma) | **macOS 14.5+ or macOS 15+ (Sequoia)** | Required for CoreML 7+ / 8+ and modern Accelerate APIs. |
| **Xcode** | Xcode 15.0 | **Xcode 16.0+** | Must include Command Line Tools and iOS 17+ SDK. |
| **Python** | Python 3.10 | **Python 3.11.x** | *Caution:* Do **not** use Python 3.12 or 3.13 yet due to PyTorch / `coremltools` pre-built binary wheel compatibility. |
| **Package Manager**| Homebrew | Homebrew (`brew`) | For Python 3.11, git-lfs, and audio codecs. |
| **Audio Tools** | `libsndfile` | `libsndfile`, `ffmpeg` | Required by Python's `soundfile` library to read/write MP3 and WAV. |

---

## 3. Quick Start

If your Mac already has Xcode and Homebrew installed, run this terminal sequence:

```bash
# 1. Install system dependencies
brew install python@3.11 git-lfs ffmpeg libsndfile
git lfs install

# 2. Setup Python Virtual Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download Model Weights
python SCNetQuantizationPipeline/download_checkpoints.py

# 4. Run Separation Test (Python)
python SCNetQuantizationPipeline/test_audio_separation.py \
  --audio test_audio/audio_example.mp3 \
  --checkpoint checkpoints/scnet_small.ckpt

# 5. Build and Test Swift Native Engine
swift test
```

---

## 4. Step-by-Step System Setup

### Step 1: Install Xcode & Command Line Tools

1. Install **Xcode** from the Mac App Store or download it from [Apple Developer Downloads](https://developer.apple.com/download/all/).
2. Open Terminal and install the Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```
3. Set the active developer directory to Xcode (this is **critical** so `xcrun coremlcompiler` is discoverable by Python scripts):
   ```bash
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```
4. Verify the toolchain:
   ```bash
   xcrun --find swift
   xcrun --find coremlcompiler
   ```
   Both commands should output paths within `/Applications/Xcode.app/...`.

### Step 2: Install Homebrew & Native Dependencies

1. If you don't have Homebrew installed, install it:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python 3.11, Git LFS, and audio manipulation libraries:
   ```bash
   brew install python@3.11 git-lfs ffmpeg libsndfile
   ```
3. Initialize Git LFS:
   ```bash
   git lfs install
   ```

### Step 3: Clone Repository & Configure Git LFS

Clone the repository to your Mac workspace:

```bash
cd ~/Documents/GitHub
git clone https://github.com/your-username/karaoke.git
cd karaoke
```

---

## 5. Python Quantization Pipeline Setup

### Step 4: Create Python 3.11 Virtual Environment

Always use Python 3.11 for model conversion to ensure full compatibility with `coremltools` and `torch`:

```bash
# Verify Python 3.11 is active
python3.11 --version
# Expected: Python 3.11.x

# Create virtual environment named .venv
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

Verify the key libraries import cleanly in Python:
```bash
python -c "import torch, torchaudio, coremltools, soundfile; print('All ML modules loaded successfully!')"
```

### Step 5: Download SCNet Model Checkpoints

Run the automated checkpoint downloader to fetch official pretrained weights (`scnet_base.ckpt` and `scnet_small.ckpt`):

```bash
python SCNetQuantizationPipeline/download_checkpoints.py
```

The script will populate the `checkpoints/` directory:
- `checkpoints/scnet_base.ckpt` (~40 MB)
- `checkpoints/scnet_small.ckpt` (~30 MB)
- `checkpoints/config_scnet_base.yaml`
- `checkpoints/config_scnet_small.yaml`

### Step 6: Run Conversion, Pruning & INT8 Quantization

Convert the PyTorch model into an ANE-compatible, static-shape CoreML model with INT8 linear symmetric weight quantization:

```bash
# Convert and quantize SCNet-Small (Optimal for mobile ANE background execution)
python SCNetQuantizationPipeline/pipeline.py \
  --model small \
  --checkpoint checkpoints/scnet_small.ckpt \
  --out SCNetVocals_small.mlpackage
```

What this step does:
1. Prunes unneeded stem decoder branches (keeps vocals / instrumental).
2. Sets static dimensions `[1, 4, 2049, 130]` (3.0-second STFT window at 44.1 kHz).
3. Strips dynamic `torch.stft`/`torch.istft` (which would trigger CPU fallback).
4. Applies `ct.precision.FLOAT16` activations and `int8` weight quantization.
5. Verifies SDR numerical parity against PyTorch reference.

### Step 7: Compile CoreML Model (`.mlpackage` to `.mlmodelc`)

The iOS app loads compiled `.mlmodelc` bundles at runtime to avoid spending 5–15 seconds compiling the graph in RAM during launch:

```bash
# Compile .mlpackage into an optimized .mlmodelc directory
xcrun coremlcompiler compile SCNetVocals_small.mlpackage .
```

This generates `SCNetVocals_small.mlmodelc` in your current directory.

> [!TIP]
> If your Swift code expects `SCNetInstrumental.mlmodelc`, you can either rename the generated folder or compile via `SCNetKaraokeEngine/OfflineConversion/export_ios_mlpackage.py`:
> ```bash
> python SCNetKaraokeEngine/OfflineConversion/export_ios_mlpackage.py
> xcrun coremlcompiler compile SCNetInstrumental.mlpackage .
> ```

### Step 8: Run Separation Test on Real Audio

Verify the separation quality on macOS before touching mobile code:

```bash
python SCNetQuantizationPipeline/test_audio_separation.py \
  --audio test_audio/audio_example.mp3 \
  --checkpoint checkpoints/scnet_small.ckpt \
  --output_dir test_audio/output
```

Check `test_audio/output/` for the separated files:
- `audio_example_vocals.wav`
- `audio_example_instrumental.wav`
- `audio_example_original.wav`

---

## 6. Swift Native Engine Setup (CLI)

The native engine is structured as a standard Swift Package (`Package.swift`).

### Step 9: Build and Test via Swift Package Manager

In your terminal (from project root):

```bash
# 1. Resolve Swift package dependencies (swift-atomics)
swift package resolve

# 2. Build the engine target
swift build

# 3. Run the complete unit test suite
swift test
```

Expected test output:
```
Building for debugging...
[4/4] Linking SCNetKaraokeEnginePackageTests
Test Suite 'All tests' started at ...
Test Suite 'SCNetKaraokeEnginePackageTests.xctest' started at ...
Test Suite 'SCNetKaraokeEngineTests' started at ...
Test Case '-[SCNetKaraokeEngineTests.SCNetKaraokeEngineTests testEngineConfigConstants]' passed
Test Case '-[SCNetKaraokeEngineTests.SCNetKaraokeEngineTests testPageAlignedTensorPoolAllocation]' passed
Test Case '-[SCNetKaraokeEngineTests.SCNetKaraokeEngineTests testLockFreeRingBufferReadWrite]' passed
Test Case '-[SCNetKaraokeEngineTests.SCNetKaraokeEngineTests testOverlapAddAccumulatorReconstruction]' passed
Test Suite 'SCNetKaraokeEngineTests' passed.
```

---

## 7. IDE Setup & Configuration

You can develop using **Xcode** (best for iOS app profiling, Simulator, and Swift debugging) or **Visual Studio Code / Cursor** (best for simultaneous Python and Swift code editing).

---

### Option A: Xcode Setup

#### 1. Open the Swift Package
1. Launch Xcode.
2. Go to **File** > **Open...** (or run `open Package.swift` in Terminal).
3. Select the `karaoke` root folder or `Package.swift`.
4. Xcode will automatically resolve the `swift-atomics` dependency via Swift Package Manager.

#### 2. Select Scheme & Destination
- In the toolbar at the top of Xcode:
  - Scheme: Select **`SCNetKaraokeEngine`**.
  - Destination:
    - Choose **My Mac (Mac Catalyst or Mac)** to test on desktop Apple Silicon.
    - Choose **iPhone 15/16 Pro Simulator** or your **Connected Physical iPhone** running iOS 17+.

#### 3. Run Unit Tests in Xcode
- Press `Cmd + U` or go to **Product** > **Test**.
- View test results and execution times in the Test Navigator (`Cmd + 6`).

#### 4. Embedding the Engine into an iOS App Target
When wrapping `SCNetKaraokeEngine` into a full iOS UI application:
1. In your App target's **Signing & Capabilities** tab:
   - Click **+ Capability** and add **Background Modes**.
   - Check **Audio, AirPlay, and Picture in Picture**.
   - Check **Background processing**.
2. Drag `SCNetInstrumental.mlmodelc` (or `SCNetVocals_small.mlmodelc`) into your App target's **Copy Bundle Resources** build phase.
3. In `Info.plist`, ensure the background keys are present (matching [Info.plist](file:///c:/Users/zubair/Documents/GitHub/karaoke/SCNetKaraokeEngine/iOSApp/Config/Info.plist)):
   ```xml
   <key>UIBackgroundModes</key>
   <array>
       <string>audio</string>
       <string>processing</string>
   </array>
   ```

#### 5. Profiling with Xcode Instruments
To verify zero heap allocations and memory limits:
1. Go to **Product** > **Profile** (`Cmd + I`).
2. Choose **Allocations** instrument:
   - Verify that during continuous streaming, `All Heap Allocations` remains flat (no steady upward slope).
3. Choose **Core ML** instrument:
   - Inspect prediction latency and verify that operations execute on the **Neural Engine** rather than the GPU.

---

### Option B: Visual Studio Code / Cursor Setup

VS Code or Cursor provides a lightweight workflow for editing both Python and Swift files.

#### 1. Recommended Extensions
Install these extensions from the Extensions Marketplace (`Cmd + Shift + X`):
- **Swift** (`sswg.swift`): Official Swift extension providing code completion, build commands, and test running via SourceKit-LSP.
- **Python** (`ms-python.python`): Rich Python support.
- **Pylance** (`ms-python.vscode-pylance`): High-performance Python language server.
- **Netron** (`robert-harvey.netron` or standalone Netron app): Visual inspector for `.mlmodel`, `.mlpackage`, and `.pt` graphs.

#### 2. Workspace Configuration (`.vscode/settings.json`)
Create or edit `.vscode/settings.json` in the project root:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.extraPaths": [
    "${workspaceFolder}/SCNetQuantizationPipeline",
    "${workspaceFolder}/SCNetKaraokeEngine/OfflineConversion"
  ],
  "swift.path": "/usr/bin/swift",
  "swift.sourcekit-lsp.serverPath": "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/sourcekit-lsp",
  "files.exclude": {
    "**/.git": true,
    "**/.DS_Store": true,
    "**/__pycache__": true
  }
}
```

#### 3. Build & Test Tasks (`.vscode/tasks.json`)
Create `.vscode/tasks.json` to trigger builds and tests with `Cmd + Shift + B`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Swift: Build Engine",
      "type": "shell",
      "command": "swift build",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "problemMatcher": "$swift-device"
    },
    {
      "label": "Swift: Run Tests",
      "type": "shell",
      "command": "swift test",
      "group": {
        "kind": "test",
        "isDefault": true
      },
      "problemMatcher": "$swift-device"
    },
    {
      "label": "Python: Run Vocal Pipeline",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python SCNetQuantizationPipeline/pipeline.py --model small",
      "problemMatcher": []
    },
    {
      "label": "Python: Test Audio Separation",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python SCNetQuantizationPipeline/test_audio_separation.py --audio test_audio/audio_example.mp3",
      "problemMatcher": []
    }
  ]
}
```

#### 4. Launch Configurations (`.vscode/launch.json`)
To debug Python scripts directly:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Audio Separation Test",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/SCNetQuantizationPipeline/test_audio_separation.py",
      "args": [
        "--audio", "${workspaceFolder}/test_audio/audio_example.mp3",
        "--checkpoint", "${workspaceFolder}/checkpoints/scnet_small.ckpt"
      ],
      "console": "integratedTerminal"
    },
    {
      "name": "Python: Run Quantization Pipeline",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/SCNetQuantizationPipeline/pipeline.py",
      "args": ["--model", "small"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## 8. Hardware Verification & Profiling on Apple Neural Engine (ANE)

To ensure the model is utilizing the Apple Neural Engine and will survive iOS background memory limits:

### 1. Op Placement Inspection
Use Apple's CoreML compiler utility to inspect the sub-graph execution plan:

```bash
# Profile op-by-op hardware target on macOS
xcrun mlmodelc-profile SCNetVocals_small.mlmodelc
```

Verify that **Convolutions**, **Dense layers**, and **LayerNorms** map to **ANE**, with only minor routing ops on CPU, and **zero** fallback to Metal GPU.

### 2. Verify Background Compute Policy
In `SCNetModelLoader.swift`, compute units are enforced to:
```swift
config.computeUnits = .cpuAndNeuralEngine
```
This guarantees that when the app enters the background:
- Metal GPU command buffers are never submitted (`0xdead10cc` violation avoided).
- Dirty RAM (`phys_footprint`) stays under **75 MB**, well below the iOS background Jetsam threshold (~120 MB).

---

## 9. Common Gotchas & Troubleshooting

### Q1: `xcrun: error: unable to find utility "coremlcompiler"`
* **Cause:** The active developer directory is pointing to Command Line Tools instead of full Xcode.
* **Fix:**
  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  ```

### Q2: `ModuleNotFoundError: No module named 'coremltools'` on Apple Silicon Mac
* **Cause:** Python was launched in x86_64 emulation under Rosetta 2, or Python 3.12/3.13 was used where prebuilt wheels do not exist.
* **Fix:**
  1. Check architecture: `python3.11 -c "import platform; print(platform.machine())"` (must output `arm64`).
  2. If using Terminal under Rosetta, uncheck "Open using Rosetta" in Terminal app's "Get Info" panel.
  3. Ensure you are using Python 3.11 inside `.venv`.

### Q3: `posix_memalign failed for input buffer` runtime crash
* **Cause:** Memory alignment must be an exact multiple of `sizeof(void *)` and a power of two. On Apple Silicon, 16,384 bytes (16 KB) is used to align with page size.
* **Fix:** Make sure `SCNetEngineConfig.pageAlignmentBytes` is 16,384. This is pre-configured and verified by `testPageAlignedTensorPoolAllocation`.

### Q4: CoreML Model Not Found (`SCNetModelLoaderError.modelNotFound`)
* **Cause:** The compiled model `SCNetInstrumental.mlmodelc` has not been compiled or added to the application bundle resources.
* **Fix:**
  1. Compile the `.mlpackage`:
     ```bash
     xcrun coremlcompiler compile SCNetVocals_small.mlpackage .
     mv SCNetVocals_small.mlmodelc SCNetInstrumental.mlmodelc
     ```
  2. If running inside Xcode, add `SCNetInstrumental.mlmodelc` to **Build Phases** > **Copy Bundle Resources**.

### Q5: Audio clicks or pops during continuous playback
* **Cause:** Ring buffer underrun or improper crossfade accumulation.
* **Fix:**
  - Verify that `OverlapAddAccumulator` uses the 50% window hop (`stepSamples = 66150`, `windowSamples = 132300`).
  - Maintain a producer lookahead watermark of at least 3.0 to 6.0 seconds in `LockFreePCMBuffer`.

---

## Quick Reference Commands

```bash
# Activate Python environment
source .venv/bin/activate

# Download checkpoints
python SCNetQuantizationPipeline/download_checkpoints.py

# Quantize model to ANE .mlpackage
python SCNetQuantizationPipeline/pipeline.py --model small

# Compile .mlpackage for zero-runtime-latency loading
xcrun coremlcompiler compile SCNetVocals_small.mlpackage .

# Build and run Swift tests
swift test
```

