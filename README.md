# SCNet Karaoke Engine

Real-time audio source separation and karaoke engine for iOS and macOS, powered by pruned and quantized **SCNet** (Sparse Compression Network) models executing on the **Apple Neural Engine (ANE)**.

---

## Key Features

- **Apple Neural Engine (ANE) Acceleration**: Runs exclusively on `.cpuAndNeuralEngine` with static `[1, 4, 2049, 130]` tensors, completely bypassing the Metal GPU to prevent background GPU terminations (`0xdead10cc`) on iOS.
- **Decoupled Spectral Processing**: Forward and inverse STFT run in Swift via Apple's native `Accelerate` (`vDSP` / `vImage`), eliminating unsupported complex tensor ops from the CoreML graph.
- **Zero-Allocation Audio DSP**: Pre-allocated 16 KB page-aligned memory pools (`PageAlignedTensorPool`) ensure 0 bytes are allocated on the heap during active streaming.
- **Lock-Free Audio Streaming**: Single-Producer Single-Consumer (SPSC) ring buffer (`LockFreePCMBuffer`) using `swift-atomics` to isolate chunked model inference from real-time `AVAudioEngine` playback.
- **Offline Model Quantization Pipeline**: Ingests PyTorch weights, strips unneeded stems (down to single instrumental or vocal stem), and performs INT8 weight quantization via `coremltools`.

---

## Getting Started on macOS

For complete setup instructions, prerequisites, Xcode configuration, and VS Code/Cursor workflows, see:

👉 **[macOS & IDE Setup Guide (MACOS_SETUP_GUIDE.md)](MACOS_SETUP_GUIDE.md)**

### Quick Commands

```bash
# 1. System prerequisites
brew install python@3.11 git-lfs ffmpeg libsndfile

# 2. Python environment & weights
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python SCNetQuantizationPipeline/download_checkpoints.py

# 3. Model quantization & compilation
python SCNetQuantizationPipeline/pipeline.py --model small
xcrun coremlcompiler compile SCNetVocals_small.mlpackage .

# 4. Build and test Swift engine
swift test
```

---

## Documentation Links

- [macOS & IDE Setup Guide](MACOS_SETUP_GUIDE.md)
- [Architecture & Planning Specification](karaoke%20planning%20doc.md)
- [Mobile Quantization Framework Plan](SCNetQuantizationPipeline/QUANTIZATION_PLAN.md)
- [Additional System Architecture Notes](other%20architecture%20info.md)

