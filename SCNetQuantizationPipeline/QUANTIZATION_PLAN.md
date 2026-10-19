# SCNet Mobile Quantization Framework Plan

## 1. Online Landscape & Competitive Findings

A thorough investigation of publicly available model registries (Hugging Face, GitHub, UVR Model Garden, Apple Model Garden) yielded the following findings:

1. **`Ocean82/scnet-base` (Hugging Face / Schism Framework):**
   * **Target:** macOS desktop audio separation.
   * **Size / Memory:** ~10M parameters in FP32/FP16, requiring **1,024 MB peak RAM** and RTF 0.6.
   * **Limitation:** Completely unusable for iOS background audio execution where the dirty RAM ceiling is under 120 MB. It includes full 4-stem decoders and unoptimized desktop windowing.
2. **`Politrees/UVR_resources` & `Aname-Tommy/Huge-SCNet-4stems` (UVR Checkpoints):**
   * **Format:** Unquantized PyTorch `.ckpt` / `.pth` files.
   * **Size:** 160 MB to 500 MB+ weights.
   * **Limitation:** Contains embedded `torch.stft`/`torch.istft` operations and complex tensors that cannot run on the Apple Neural Engine (ANE), causing immediate fallback to CPU/GPU and crashing in the iOS background.
3. **Conclusion:** **No open, pre-quantized mobile-sized SCNet model (INT8, ≤8 MB weights, ≤75 MB runtime RAM, static ANE-compatible tensor shape, decoupled STFT) currently exists online.** Building a dedicated conversion and quantization framework is therefore strictly required.

---

## 2. Target Mobile Specifications

| Dimension | Desktop / Research Baseline | Target Mobile Quantized Variant |
| :--- | :--- | :--- |
| **Model Weights Size** | 40 MB (Base FP32) to 500 MB (Large) | **≤ 7.5 MB** (INT8 Quantized) |
| **Runtime Peak Dirty RAM** | 500 MB – 1,024 MB | **≤ 75.0 MB** (`phys_footprint`) |
| **Output Stems** | 4 Stems (Drums, Bass, Other, Vocals) | **1 Stem (Instrumental)** or **2 Stems (Instrumental + Guide Vocal)** |
| **Tensor Input Shape** | Variable / Dynamic ($B \times C \times F \times T$) | **Static [1, 4, 2049, 130]** (3.0s window, 44.1 kHz, hop 1024) |
| **Spectral Ops** | Embedded `torch.stft` / `torch.istft` | **Decoupled:** Native Apple `vDSP` on CPU |
| **Hardware Target** | GPU / CPU | **100% Apple Neural Engine (ANE)** via `.cpuAndNeuralEngine` |
| **SDR Quality Loss Target**| 0 dB (lossless reference) | **< 0.35 dB SDR loss** relative to FP32 reference |

---

## 3. Five-Phase Implementation Architecture

```
+-----------------------------------------------------------------------------------+
|               PHASE 1: CHECKPOINT INGESTION & ARCHITECTURE NORMALIZATION         |
|  - Fetch PyTorch weights (SCNet-Base, SCNet-Small, or UVR StarryTong)            |
|  - Load state_dict into standardized SCNet module topology                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               PHASE 2: STEM PRUNING & GRAPH DECOUPLING                            |
|  - Excise drums, bass, and other decoder branches (saving ~60% compute/RAM)       |
|  - Strip torch.stft, torch.istft, and complex64 types                             |
|  - Fix static 4D input [1, 4, 2049, 130] & trace via torch.jit.trace             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               PHASE 3: COREML CONVERSION & ADVANCED QUANTIZATION                  |
|  - Convert to CoreML MIL mlprogram (iOS 17/18 target)                             |
|  - Float16 activation precision                                                   |
|  - Apply INT8 linear symmetric weight quantization (coremltools.optimize)         |
|  - Optional: Data-driven calibration using representative audio snippets          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               PHASE 4: NUMERICAL PARITY & SDR QUALITY VERIFICATION                |
|  - Compare PyTorch FP32 vs. CoreML INT8 on benchmark audio mixtures              |
|  - Measure Signal-to-Distortion Ratio (SDR) and artifact leakage                  |
|  - Validate phase continuity across 50% overlap-add sliding windows               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               PHASE 5: ANE COMPILATION, PROFILING & PACKAGING                     |
|  - Run xcrun coremlcompiler compile -> SCNetInstrumental.mlmodelc                 |
|  - Profile op-by-op ANE placement with xcrun mlmodelc-profile                     |
|  - Package into SCNetKaraokeEngine/iOSApp bundle resources                        |
+-----------------------------------------------------------------------------------+
```

---

## 4. Detailed Component Specifications

### 4.1. Checkpoint Loader & Ingestion (`checkpoint_loader.py`)
* Supports loading from:
  1. Local `.pth` / `.ckpt` weights.
  2. Direct download from Hugging Face / UVR checkpoints via `huggingface_hub`.
  3. Automatic mapping of `state_dict` keys across different community implementations (e.g., `ZFTurbo` vs. `amanteur` vs. `SonyResearch`).

### 4.2. Model Pruning & ANE Adaptation (`scnet_pruner.py`)
* **Stem Extraction:** Extracts only the instrumental head or synthesizes the instrumental track via $x - \text{vocals}$ mask arithmetic.
* **Static Graph Enforcement:**
  - Replaces all dynamic shape queries (`x.shape[-1]`) with constant integers ($130$).
  - Eliminates dynamic padding/slicing in subband splitting layers.
  - Formats stereo spectrograms as planar real tensors: `[Batch, 4, Freq, Time]`.

### 4.3. Quantization Engine (`quantize_engine.py`)
* Leverages `coremltools 8.0+`:
  ```python
  from coremltools.optimize.coreml import (
      OpLinearQuantizerConfig,
      OptimizationConfig,
      linear_quantize_weights,
  )
  ```
* Evaluates two quantization schemes:
  1. **Post-Training Symmetric INT8 (`linear_symmetric`):** Default choice; zero calibration required, fast execution, compresses weights to $\sim$6.5 MB.
  2. **Calibration-Aware Block Quantization / Palettization:** Used if symmetric INT8 introduces high-frequency perceptual noise in vocal isolation.

### 4.4. Audio Quality & SDR Parity Suite (`evaluate_parity.py`)
* Runs comparative separation on a standardized 15-second audio track.
* Computes:
  - **SDR (Signal-to-Distortion Ratio)** between PyTorch FP32 output and CoreML INT8 output.
  - **SNR (Signal-to-Noise Ratio)** and Max Absolute Difference ($L_\infty$).
  - **Mel-Spectrogram Residual Plot** to ensure high-frequency harmonics are not blurred or clipped by quantization.

### 4.5. ANE Profiler & iOS Bundle Builder (`compile_and_package.py`)
* Executes `xcrun coremlcompiler compile` to generate the pre-compiled `.mlmodelc`.
* Executes `xcrun mlmodelc-profile` to verify **100% of subband convolutions and recurrent/transformer blocks run on the Apple Neural Engine** with 0 ops on the GPU.
* Copies the finalized `.mlmodelc` directly into `SCNetKaraokeEngine/iOSApp/`.

---

## 5. Execution Roadmap & Next Steps

1. **Setup Environment:** Install dependencies specified in `requirements.txt`.
2. **Execute Ingestion:** Run `python checkpoint_loader.py --checkpoint <path_or_repo>`.
3. **Run Quantization Pipeline:** Run `python pipeline.py --mode int8 --stem instrumental`.
4. **Parity Check:** Run `python evaluate_parity.py --input sample_track.wav`.
5. **Compile to `.mlmodelc`:** Compile and deploy into the iOS app bundle.

