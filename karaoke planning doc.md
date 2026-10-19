# Technical Architecture & Planning Document: SCNet CoreML Engine for iOS Background Execution

## 1. Executive Summary & Architectural Scope

This document defines the architecture, dependency stack, and file structure required to load, manage, and continuously execute the **SCNet** (Sparse Compression Network) audio source separation model on iOS devices in the background without triggering **Jetsam** (out-of-memory kills) or **Background GPU Execution Violations** (`MTLCommandBufferErrorNotPermitted` / `0xdead10cc`).

File selection and UI/playback layers are excluded from this scope. The architecture focuses strictly on:

1. **Model Adaptation & Loading:** Stripping unsupported spectral operations (`STFT`/`iSTFT`), quantizing weights, and loading the CoreML model with optimal compute unit routing (`Apple Neural Engine` vs. `CPU/GPU`).
2. **Zero-Allocation Streaming DSP & Inference:** Pre-allocating page-aligned memory buffers for `vDSP` Fast Fourier Transforms and `MLMultiArray` tensors so the continuous inference loop allocates **zero bytes** on the heap during runtime.
3. **iOS Background Survival:** Managing `AVAudioSession` keep-alive states, dynamic compute-unit switching on app backgrounding, and strict memory/thermal throttling.

---

## 2. System Architecture & Dataflow Pipeline

```
+-----------------------------------------------------------------------------------+
|                        iOS BACKGROUND LIFECYCLE CONTROLLER                        |
|  - AVAudioSession (.playback, .mixWithOthers) keep-alive                          |
|  - UIApplication.didEnterBackground -> Enforces .cpuAndNeuralEngine compute unit  |
|  - os_proc_available_memory() & ThermalState monitor -> Dynamic hop/chunk control |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     DEDICATED INFERENCE SERIAL DISPATCH QUEUE                     |
|                         (QoS: .userInitiated, Fixed Pool)                         |
|                                                                                   |
|  [1. Raw PCM Sliding Window Reader]                                               |
|      Reads 3.0s stereo Float32 (132,300 frames/ch) with 1.5s hop (66,150 frames)  |
|                                         |                                         |
|                                         v                                         |
|  [2. SpectralProcessorVDSP (Forward STFT)]                                        |
|      - Applies periodic Hann window (N=4096, Hop=1024) via vDSP_vmul              |
|      - Executes real-to-complex FFT via vDSP_DFT_zop                              |
|      - Packs [L_real, L_imag, R_real, R_imag] & converts FP32 -> FP16 (vImage)    |
|      - Writes directly into page-aligned Input MLMultiArray [1, 4, 2049, 130]     |
|                                         |                                         |
|                                         v                                         |
|  [3. SCNetModelRunner (CoreML / Apple Neural Engine)]                             |
|      - Executes quantized SCNet-Small (INT8 weights / FP16 activations)           |
|      - Writes directly into pre-allocated Output MLMultiArray [1, 4, 2049, 130]   |
|                                         |                                         |
|                                         v                                         |
|  [4. SpectralProcessorVDSP (Inverse STFT)]                                        |
|      - Converts FP16 -> FP32 and unpacks complex bins                             |
|      - Executes inverse DFT via vDSP_DFT_zop & synthesis windowing                |
|      - Reconstructs 3.0s time-domain stereo buffer [2, 132,300]                   |
|                                         |                                         |
|                                         v                                         |
|  [5. OverlapAddAccumulator (50% Crossfade)]                                       |
|      - Ramps up first 1.5s (0.0 -> 1.0) and sums with previous 1.5s Carry Tail    |
|      - Ramps down second 1.5s (1.0 -> 0.0) and saves as new Carry Tail            |
+-----------------------------------------------------------------------------------+
                                         |
                        1.5s Finalized PCM Chunk [2, 66,150]
                                         v
+-----------------------------------------------------------------------------------+
|                        LOCK-FREE SPSC AUDIO RING BUFFER                           |
|               (Pre-allocated 12-second circular buffer in flat RAM)               |
+-----------------------------------------------------------------------------------+

```

### Target Runtime Hardware Constraints

* **Peak Dirty RAM (`phys_footprint`) Budget:** Less than **120 MB** total across model weights, CoreML intermediate activations, and DSP buffers.
* **Inference Window Geometry:**
* Sample Rate: `44,100 Hz` Stereo (`2 channels`)
* Chunk Duration: `3.0 seconds` (`132,300 samples` per channel, padded to `133,120` for exact `130` STFT frames)
* Sliding Hop Size: `1.5 seconds` (`66,150 samples` per channel, 50% overlap)
* FFT Size (`n_fft`): `4096` (`2049` frequency bins)
* STFT Hop Length: `1024` (`130` time frames per 3.0s chunk)



---

## 3. Required Libraries & Frameworks

### Part A: Offline Model Preparation Toolchain (Python / macOS)

Used strictly offline to strip `torch.stft`/`torch.istft` from the macOS/PyTorch model, prune unneeded stems, and compile the ANE-compatible `.mlpackage`.

| Library | Version | Purpose |
| --- | --- | --- |
| **`torch` & `torchaudio**` | `2.1+` | Loading the pretrained SCNet checkpoint, stripping `stft`/`istft` nodes, and tracing/exporting the core separation graph. |
| **`coremltools`** | `8.0+` | Converting the traced PyTorch graph to CoreML MIL (`mlprogram`), applying `FLOAT16` activation precision, and executing `linear_quantize_weights` (INT8 weight compression). |
| **`einops`** | `0.7+` | Required by SCNet's internal subband split and dual-path tensor reshaping layers during tracing. |
| **`numpy`** | `1.26+` | Numerical parity verification between PyTorch STFT and Apple `vDSP` outputs. |

---

### Part B: iOS Native Runtime Frameworks (Xcode / Swift)

To minimize binary size and memory overhead, the iOS runtime relies almost entirely on Apple system frameworks with hardware SIMD/DMA acceleration.

| Framework / Package | Import Statement | Specific APIs Used & Purpose |
| --- | --- | --- |
| **CoreML** | `import CoreML` | `MLModel`, `MLModelConfiguration`, `MLComputeUnits`, `MLMultiArray`, `MLPredictionOptions` (specifically `outputBackings` to force zero-copy output buffers). |
| **Accelerate** | `import Accelerate` | **`vDSP`**: `vDSP_DFT_zop_CreateSetup` (forward/inverse DFT), `vDSP_vmul` (windowing), `vDSP_vadd` (overlap-add accumulation), `vDSP_vsmul` (scaling).<br>

<br>**`vImage`**: `vImageConvert_PlanarFtoPlanar16F` and `vImageConvert_Planar16FtoPlanarF` for hardware-accelerated vector conversion between `Float32` DSP buffers and `Float16` CoreML tensors. |
| **AVFAudio** | `import AVFAudio` | `AVAudioSession` (configuring `.playback` category to keep the background process alive and monitoring audio route/interruption notifications). |
| **UIKit** | `import UIKit` | `UIApplication.didEnterBackgroundNotification`, `willEnterForegroundNotification`, and `didReceiveMemoryWarningNotification` to switch compute policies and flush non-essential caches. |
| **Foundation / Darwin** | `import Foundation`<br>

<br>`import Darwin` | `posix_memalign` (allocating 16KB page-aligned raw memory for ANE Direct Memory Access), `os_proc_available_memory()` (monitoring exact remaining headroom before Jetsam kill), and `ProcessInfo.thermalState`. |
| **Swift Atomics** *(Swift Package)* | `import Atomics` | `ManagedAtomic<Int>` from `apple/swift-atomics` for lock-free read/write indices in the Single-Producer Single-Consumer (SPSC) output ring buffer. |

---

## 4. Directory & File Structure for the iOS App

Below is the complete file breakdown for the background inference engine, organized by subsystem.

```
SCNetKaraokeEngine/
├── OfflineConversion/                  # Run on Mac to prepare the iOS model
│   ├── scnet_core_wrapper.py
│   ├── export_ios_mlpackage.py
│   └── verify_ane_compatibility.py
└── iOSApp/
    ├── Config/
    │   ├── Info.plist
    │   └── SCNetEngineConfig.swift
    ├── LifecycleAndBackground/
    │   ├── BackgroundExecutionManager.swift
    │   └── SystemResourceMonitor.swift
    ├── MemoryAndBuffers/
    │   ├── PageAlignedTensorPool.swift
    │   └── LockFreePCMBuffer.swift
    ├── DSP/
    │   ├── SpectralProcessorVDSP.swift
    │   └── OverlapAddAccumulator.swift
    └── CoreMLRunner/
        ├── SCNetModelLoader.swift
        └── SCNetStreamingWorker.swift

```

---

## 5. Detailed File Specifications & Responsibilities

### Module 1: Offline Conversion Scripts (`OfflineConversion/`)

#### 1. `scnet_core_wrapper.py`

* **Purpose:** Wraps the original SCNet PyTorch architecture to make it 100% compatible with the Apple Neural Engine.
* **Key Operations:**
* Removes `torch.stft`, `torch.istft`, and all `torch.complex64` tensor operations (which force CoreML to fall back to the CPU).
* Accepts a static 4D real tensor of shape `[1, 4, 2049, 130]` representing `[Batch, Channels (L_real, L_imag, R_real, R_imag), FreqBins, TimeFrames]`.
* Prunes the 4-stem output decoder head down to a **single Instrumental stem** (or 2-stem `Instrumental + Vocal`), cutting decoder memory and compute in half.
* Replaces any dynamic shape slicing (`tensor[:, :, :T]`) with static dimensions locked to `130` frames (`3.0 seconds`).



#### 2. `export_ios_mlpackage.py`

* **Purpose:** Converts the wrapped PyTorch module into an iOS-optimized `.mlpackage` and pre-compiles it into `.mlmodelc`.
* **Key Operations:**
* Traces the model using `torch.jit.trace` with a fixed `[1, 4, 2049, 130]` dummy tensor.
* Runs `coremltools.convert` targeting `ct.target.iOS17` (or `iOS18`) with `compute_precision=ct.precision.FLOAT16`.
* Applies post-training weight quantization (`ct.optimize.coreml.linear_quantize_weights` with `mode="linear_symmetric"`, `dtype=int8`) to compress model weights to roughly 5–8 MB.
* Invokes `xcrun coremlcompiler compile` to generate `SCNetInstrumental.mlmodelc` so the iOS device does not have to compile the graph in RAM on first launch.



---

### Module 2: Configuration, Lifecycle & Background Survival (`Config/` & `LifecycleAndBackground/`)

#### 3. `Info.plist`

* **Purpose:** Declares system permissions required to prevent iOS from suspending the process when the screen locks or the user switches apps.
* **Required Keys:**
* `UIBackgroundModes`: Array containing `audio` (keeps the process alive continuously while active audio streaming is in progress) and `processing` (allows optional `BGProcessingTask` pre-caching).



#### 4. `SCNetEngineConfig.swift`

* **Purpose:** Single source of truth for compile-time and runtime DSP/tensor constants.
* **Key Constants Defined:**
* `sampleRate: Double = 44100.0`
* `fftSize: Int = 4096`
* `hopLength: Int = 1024`
* `freqBins: Int = 2049` (`fftSize / 2 + 1`)
* `stftFrames: Int = 130`
* `windowSamples: Int = 132300` (`3.0s` of audio)
* `stepSamples: Int = 66150` (`1.5s` hop / 50% overlap)
* `paddedWindowSamples: Int = 133120` (`(130 - 1) * 1024 + 4096` for exact center-padded STFT parity with PyTorch)



#### 5. `BackgroundExecutionManager.swift`

* **Purpose:** Coordinates `AVAudioSession` state and protects the app from iOS background GPU termination (`0xdead10cc` / `MTLCommandBufferErrorNotPermitted`).
* **Key Responsibilities:**
* Configures `AVAudioSession.sharedInstance()` with `.playback` category and `.default` mode.
* Subscribes to `UIApplication.didEnterBackgroundNotification` and `UIApplication.willEnterForegroundNotification`.
* **Critical Background Rule:** If the loaded CoreML model uses any Metal GPU fallback in the foreground, entering the background triggers `SCNetModelLoader` to ensure the inference session is strictly bound to `.cpuAndNeuralEngine` (since ANE and CPU execution are permitted in the background, whereas new Metal command buffer submissions from background apps without special entitlements can be rejected by iOS).
* Handles `AVAudioSession.interruptionNotification` (e.g., incoming phone call) by cleanly pausing the inference worker loop before the OS suspends the app.



#### 6. `SystemResourceMonitor.swift`

* **Purpose:** Monitors real-time available memory before Jetsam limit and device thermal state.
* **Key Responsibilities:**
* Calls `os_proc_available_memory()` (from `<os/proc.h>`) before every inference pass to check remaining byte headroom before the iOS Jetsam threshold.
* If available headroom drops below **40 MB**, triggers an emergency cache flush and pauses lookahead buffering until the ring buffer drains to its minimum threshold.
* Observes `ProcessInfo.didChangeThermalStateNotification`: if thermal state reaches `.serious` or `.critical`, reduces the lookahead buffer watermark from `6.0s` to `3.0s` to pace ANE/CPU duty cycles.



---

### Module 3: Zero-Copy Memory & Spectral DSP (`MemoryAndBuffers/` & `DSP/`)

#### 7. `PageAlignedTensorPool.swift`

* **Purpose:** Eliminates all heap allocations during continuous streaming by pre-allocating page-aligned memory blocks that are shared directly between `vDSP` and `CoreML`.
* **Key Responsibilities:**
* Uses `posix_memalign(&ptr, 16384, byteCount)` (16 KB page alignment on Apple Silicon) to allocate backing memory for:
1. `inputMultiArray: MLMultiArray` — Shape `[1, 4, 2049, 130]`, DataType `.float16` (`2,130,960 bytes` $\approx 2.03\text{ MB}$).
2. `outputMultiArray: MLMultiArray` — Shape `[1, 4, 2049, 130]`, DataType `.float16` (`2,130,960 bytes` $\approx 2.03\text{ MB}$).
3. Scratch `Float32` planar buffers for `vDSP` STFT/iSTFT (`[4, 2049, 130]` in `Float32` $\approx 4.06\text{ MB}$).


* Constructs `MLMultiArray(dataPointer:shape:dataType:strides:deallocator:)` wrapping the page-aligned pointers so the Apple Neural Engine can read and write tensors via Direct Memory Access (DMA) without copying buffers.



#### 8. `SpectralProcessorVDSP.swift`

* **Purpose:** Executes the Forward `STFT` and Inverse `iSTFT` on the CPU using `Accelerate`, completely replacing `torch.stft` and `torch.istft`.
* **Key Responsibilities:**
* **Initialization:**
* Pre-computes the periodic 4096-sample Hann window (`vDSP_hann_window`) and synthesis Cola normalization envelope once at startup.
* Creates reusable `vDSP_DFT_Setup` objects (`vDSP_DFT_zop_CreateSetup`) for length 4096 forward and inverse transforms.


* **Forward STFT (`encodeChunkToTensor`):**
* Applies reflect or zero center-padding (`n_fft / 2 = 2048` samples on each side) to match PyTorch `center=True`.
* Loops across the `130` frames (striding by `1024` samples), multiplies each 4096-sample frame by the Hann window (`vDSP_vmul`), and executes the forward DFT (`vDSP_DFT_zop`).
* Writes the resulting `Float32` real/imaginary frequency bins into the planar scratch buffer, then executes a single bulk `vImageConvert_PlanarFtoPlanar16F` call to populate `PageAlignedTensorPool.inputMultiArray`.


* **Inverse STFT (`decodeTensorToChunk`):**
* Converts `PageAlignedTensorPool.outputMultiArray` from `Float16` back to `Float32` via `vImageConvert_Planar16FtoPlanarF`.
* Runs the 4096-point inverse DFT (`vDSP_DFT_zop`) for each of the 130 frames, applies the synthesis window, and overlap-adds the 130 STFT frames into a continuous 3.0-second (`132,300` sample) stereo waveform, dividing by the squared-window normalization envelope (`vDSP_vdiv`).





#### 9. `OverlapAddAccumulator.swift`

* **Purpose:** Stitches consecutive 3.0-second separated chunks into a click-free continuous stream using 50% overlap-add crossfading.
* **Key Responsibilities:**
* Holds a pre-allocated stereo `carryOverTail` buffer of `66,150` `Float32` samples per channel (1.5 seconds) and pre-computed linear or $\sin^2$ fade-in (`0.0 -> 1.0`) and fade-out (`1.0 -> 0.0`) ramps of length `66,150`.
* On each completed 3.0s chunk from `SpectralProcessorVDSP`:
1. Multiplies the first `66,150` samples (`0.0s - 1.5s`) by the fade-in ramp (`vDSP_vmul`).
2. Adds `carryOverTail` (from the previous iteration) to those samples (`vDSP_vadd`), producing `1.5s` of finalized, phase-continuous audio.
3. Multiplies the second `66,150` samples (`1.5s - 3.0s`) by the fade-out ramp (`vDSP_vmul`) and overwrites `carryOverTail` for the next cycle.





#### 10. `LockFreePCMBuffer.swift`

* **Purpose:** Provides thread-safe, zero-lock storage for finalized separated audio between the background inference worker and downstream consumers.
* **Key Responsibilities:**
* Allocates a flat `Float32` circular array sized for `12.0 seconds` of stereo audio (`529,200` frames $\approx 4.2\text{ MB}$).
* Uses `ManagedAtomic<Int>` read and write pointers with `.acquiring` and `.releasing` memory ordering.
* Exposes `availableFramesToRead` so `SCNetStreamingWorker` knows when to sleep (above `6.0s` buffered) and when to wake up and process the next 1.5s hop.



---

### Module 4: CoreML Model Loader & Streaming Worker (`CoreMLRunner/`)

#### 11. `SCNetModelLoader.swift`

* **Purpose:** Manages the lifecycle, configuration, and memory footprint of the `MLModel` instance.
* **Key Responsibilities:**
* Loads the pre-compiled `SCNetInstrumental.mlmodelc` from the app bundle asynchronously.
* Sets `MLModelConfiguration`:
```swift
let config = MLModelConfiguration()
// Forces execution onto Apple Neural Engine + CPU, avoiding Metal GPU
// background command buffer crashes and minimizing dirty RAM footprint.
config.computeUnits = .cpuAndNeuralEngine
config.allowLowPrecisionAccumulationOnGPU = true

```


* Pre-warms the model at load time by running one dummy prediction using `PageAlignedTensorPool` so the ANE graph is compiled and mapped before streaming starts.
* Provides an `unloadModel()` method to release the `MLModel` if the app is paused for an extended period in the background.



#### 12. `SCNetStreamingWorker.swift`

* **Purpose:** The master orchestrator running on a dedicated serial `DispatchQueue(label: "com.app.scnet.inference", qos: .userInitiated)` that drives continuous background separation.
* **Key Execution Loop (`processNextChunk()`):**
1. **Watermark Check:** Queries `LockFreePCMBuffer.availableFramesToRead`. If greater than `6.0 seconds` (`264,600 frames`), schedules a timer/wakeup after `250 ms` and returns immediately.
2. **Resource Check:** Queries `SystemResourceMonitor` to verify memory headroom is healthy.
3. **Input Staging:** Copies the next `3.0s` window (advancing by `1.5s` hop) from the source audio buffer into the DSP working buffer.
4. **Forward STFT:** Calls `SpectralProcessorVDSP.encodeChunkToTensor()` to populate `tensorPool.inputMultiArray`.
5. **Zero-Copy CoreML Prediction:**
```swift
let options = MLPredictionOptions()
// Crucial: Binds the pre-allocated output MLMultiArray so CoreML
// writes directly into our page-aligned buffer without allocating a new tensor.
options.outputBackings = ["output_spectrogram": tensorPool.outputMultiArray]

try autoreleasepool {
    _ = try model.prediction(from: inputFeatureProvider, options: options)
}

```


*(Wrapping `prediction` in an explicit `autoreleasepool` ensures any small internal Objective-C wrapper objects created by CoreML are freed immediately on every 1.5s iteration rather than accumulating on the background dispatch queue).*
6. **Inverse STFT & Overlap-Add:** Calls `SpectralProcessorVDSP.decodeTensorToChunk()` followed by `OverlapAddAccumulator.process()` and writes the finalized `66,150` stereo frames into `LockFreePCMBuffer`.



---

## 6. Memory Footprint Budget Summary (iOS Background Target)

By following this architecture, the entire runtime memory footprint of the background separation engine stays flat and deterministic:

| Component | Allocation Strategy | Memory Size |
| --- | --- | --- |
| **`SCNet-Small` Weights (INT8 Quantized)** | Memory-mapped read-only from bundle (`mmap`) | `~6.5 MB` |
| **ANE Intermediate Activation Workspace** | Managed by `ANECompiler` (3.0s window, FP16) | `~45.0 MB` |
| **`PageAlignedTensorPool` (Input + Output `MLMultiArray`)** | Pre-allocated once via `posix_memalign` | `4.1 MB` |
| **`SpectralProcessorVDSP` Scratch & Setup Tables** | Pre-allocated once at startup | `5.2 MB` |
| **`OverlapAddAccumulator` Tail & Fade Windows** | Pre-allocated once at startup | `1.1 MB` |
| **`LockFreePCMBuffer` (12s Stereo Ring Buffer)** | Pre-allocated once at startup | `4.2 MB` |
| **Per-Iteration Heap Allocations** | Enforced via `outputBackings` + `autoreleasepool` | **`0 Bytes`** |
| **Total Peak Dirty RAM (`phys_footprint`)** | **Well below iOS background Jetsam limits** | **`~66.1 MB`** |