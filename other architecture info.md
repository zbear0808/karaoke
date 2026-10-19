Adapting SCNet from macOS to an iOS streaming karaoke app requires solving two tightly linked hardware constraints: **decoupling the Fourier transforms so the neural network can run on the Apple Neural Engine (ANE)**, and **isolating chunked inference from the real-time audio render thread using an Overlap-Add (OLA) ring buffer**.

Your hunch about `iSTFT` blocking the Neural Engine is right on the money—and fixing how the model handles `STFT` and `iSTFT` solves both the compute target problem and the iOS background RAM limit at the same time.

---

## Part 1: Porting macOS CoreML SCNet to iOS (Compute, ANE, & Memory Survival)

### 1. Decouple `STFT` and `iSTFT` from the CoreML Graph

On macOS Apple Silicon, unified memory is plentiful and the GPU/CPU can absorb the overhead of unsupported PyTorch spectral operators. On iOS, keeping `torch.stft` and `torch.istft` inside the CoreML model causes two problems:

1. **Compute Fallback:** Neither complex tensors nor `STFT`/`iSTFT` ops are supported on the ANE. Including them forces CoreML to partition the graph across CPU and GPU, creating costly memory copies between compute units.
2. **Jetsam Memory Spikes:** When CoreML falls back to the Metal GPU (`MLComputeUnits.cpuAndGPU`), Metal command buffers allocate working memory that counts directly against your app's `phys_footprint` (dirty RAM). On iOS, background audio apps face strict Jetsam limits and are quickly terminated if Metal buffers spike.

**The Fix: Spectrogram-In, Spectrogram-Out CoreML Export**

* **Strip STFT/iSTFT in PyTorch before conversion:** Wrap the core frequency-domain SCNet network (`Subband Split -> Sparse Compression Encoder -> Dual-Path RNN/Transformer -> Decoder`) so its input is a real-valued `Float16`/`Float32` tensor of shape `[1, 4, FreqBins, TimeFrames]` (stereo Left/Right $\times$ Real/Imaginary concatenated, typically `FreqBins = 2049` for `n_fft = 4096`) and its output is the separated spectrogram tensor.
* **Run STFT/iSTFT natively via Apple's `Accelerate` (`vDSP`):** In Swift/C++, use `vDSP_DFT_zop` or `vDSP_fft_zrip` with a periodic Hann window (`n_fft = 4096`, `hop_length = 1024`). On an iPhone CPU, a 4096-point FFT via `vDSP` takes **less than 0.05 milliseconds** and runs in-place on pre-allocated flat memory buffers.
* **Check Internal Dual-Path Layers for ANE Compatibility:** SCNet's separation core uses dual-path time-frequency blocks. If your checkpoint uses standard convolutions and attention/unidirectional layers, decoupling STFT allows the ANE (`MLComputeUnits.cpuAndNeuralEngine`) to take over. If it uses `biLSTM` layers that partially fall back from ANE, evaluate replacing or lowering them via `coremltools` MIL passes, or use `EnumeratedShapes` so the graph compiles cleanly without dynamic-shape bailouts.

---

### 2. Shrinking the Model & Context Window to Prevent iOS Jetsam Crashes

Even with STFT/iSTFT moved to `vDSP`, standard macOS SCNet configurations will use too much intermediate activation memory for an iPhone running in the background. Apply these four optimizations during your `coremltools` export:

| Optimization | macOS Default | iOS Target | Why It Matters on iOS |
| --- | --- | --- | --- |
| **Model Variant** | `SCNet-Base` (~10M params) | **`SCNet-Small`** or **2-Stem Variant** | Reduces channel dimensions in the high-resolution lower subbands where activation tensors are largest. |
| **Output Stems** | 4 Stems (`drums`, `bass`, `other`, `vocals`) | **1 or 2 Stems** (`instrumental` + `vocals`) | For karaoke, you only need the instrumental backing track (plus optional vocals for a "guide vocal" slider). Slicing off the `drums`/`bass`/`other` decoder heads in PyTorch before export cuts decoder compute and output memory by **50%–75%**. |
| **Inference Window** | `11.0 seconds` (~474 STFT frames) | **`3.0 seconds`** (~130 STFT frames) | Intermediate tensor RAM scales linearly with the time dimension. Shrinking the window from 11s to 3s cuts peak working RAM by **~72%** with negligible SDR loss. |
| **Weight & Activation Precision** | `Float32` | **`Float16` activations + `INT8` weights** | Use `ct.precision.FLOAT16` and `ct.optimize.coreml.linear_quantize_weights`. Shrinks model footprint to **< 8 MB** and keeps runtime peak memory under **80–120 MB**. |

---

### 3. Handling macOS vs. iOS System & Lifecycle Differences

* **Background GPU Prohibitions (`0xdead10cc` / `MTLCommandBufferErrorNotPermitted`):** Unlike macOS, iOS restricts apps from submitting new Metal GPU command buffers from the background unless audio session and compute unit rules are respected.
* **Mitigation 1 (ANE Routing):** When targeting `.cpuAndNeuralEngine`, execution bypasses the Metal GPU command queue, avoiding background GPU execution violations and keeping `phys_footprint` low.
* **Mitigation 2 (Lookahead Buffering):** If certain ops require the GPU, listen for `UIApplication.willResignActiveNotification` and either switch to a `.cpuAndNeuralEngine` model instance or maintain a deeper lookahead ring buffer (e.g., 15–30 seconds of pre-separated audio) so inference bursts are minimized.


* **Zero-Copy `MLMultiArray` Reuse:** Never initialize a new `MLMultiArray` inside your inference loop. Allocate a single input `MLMultiArray` and single output `MLMultiArray` backed by page-aligned raw pointers (`posix_memalign`) or `CVPixelBuffer` at song load time, and overwrite the contents in place on every chunk.

---

## Part 2: Designing the Constant Audio Streaming Architecture

Because SCNet operates on multi-second spectrogram windows rather than sample-by-sample causal frames, feeding a continuous audio stream requires decoupling **chunked inference** from **real-time playback** using a **50% Overlap-Add (OLA) sliding window** and a **lock-free ring buffer**.

### 1. Strict Two-Thread Architecture

Never call CoreML or allocate memory inside an `AVAudioEngine` render callback. Split your pipeline into two completely independent execution contexts communicating through a **Single-Producer, Single-Consumer (SPSC) lock-free ring buffer** (such as `TPCircularBuffer` or a custom C++/Swift `ManagedAtomic` ring buffer):

1. **Producer (Inference Worker Thread):** A dedicated background `DispatchQueue` (QoS `.userInitiated`) that reads raw PCM frames from the source song file (`AVAudioFile`), runs the `vDSP STFT -> CoreML SCNet -> vDSP iSTFT -> Overlap-Add` pipeline, and writes separated stereo `Float32` samples into the **Vocal Ring Buffer** and **Instrumental Ring Buffer**.
2. **Consumer (Real-Time Audio Render Thread):** An `AVAudioSourceNode` attached to `AVAudioEngine` (running every ~10 ms / 512 samples). It simply pulls `frameCount` samples from the Vocal and Instrumental ring buffers and mixes them according to the user's real-time Vocal Guide slider (`output = instrumental + vocalGain * vocals`).

---

### 2. The Overlap-Add (OLA) Sliding Window Math

If you simply stitch consecutive 3-second separated chunks end-to-end, slight phase and boundary differences between chunks will produce an audible "click" or "phasiness" every 3 seconds. You eliminate this with a **50% overlapping sliding window and triangular/Hann crossfading**:

* **Sample Rate:** $44,100\text{ Hz}$ stereo
* **Window Size ($W$):** $3.0\text{ seconds} = 132,300\text{ samples}$
* **Hop / Step Size ($S$):** $1.5\text{ seconds} = 66,150\text{ samples}$ ($50\%$ overlap)
* **Carry-Over Tail Buffer:** A pre-allocated stereo buffer of length $S$ ($66,150\text{ samples}$), initialized to zeros at the start of a song.

**For every step $k$ on the Producer Thread:**

1. **Check Buffer Watermark:** Wait until the output playback ring buffer has less than **6.0 seconds** of audio queued (keeps RAM flat while guaranteeing no underruns).
2. **Read Window:** Advance the file read pointer by $S$ ($1.5\text{s}$) and read a full window $W$ ($3.0\text{s}$) of stereo audio:

$$\text{Chunk}_k = \text{Audio}[k \cdot S \;\dots\; k \cdot S + W]$$


3. **Forward STFT (`vDSP`):** Convert $\text{Chunk}_k$ into a complex spectrogram (Hann window `4096`, hop `1024` $\rightarrow 130\text{ frames}$) and pack into your pre-allocated `MLMultiArray`.
4. **CoreML SCNet Inference:** Run `model.prediction(from:)` to produce the separated Instrumental (and optional Vocal) spectrogram.
5. **Inverse STFT (`vDSP`):** Reconstruct the $3.0\text{s}$ time-domain waveform $\hat{y}_k[0 \dots W-1]$.
6. **Windowed Crossfade & Overlap-Add (`vDSP`):**
* Apply a linear ramp-up (or $\sin^2(t)$ window) $w_{\text{in}} \in [0 \to 1]$ to the **first half** $\hat{y}_k[0 \dots S-1]$ using `vDSP_vmul`.
* Add the saved **Carry-Over Tail Buffer** from step $k-1$ to this ramped-up first half using `vDSP_vadd`:

$$\text{Out}_k[n] = \text{Tail}_{k-1}[n] + w_{\text{in}}[n] \cdot \hat{y}_k[n] \quad \text{for } 0 \le n < S$$


* **Push $\text{Out}_k$ ($1.5\text{s}$ of seamless audio)** into the SPSC playback ring buffer.
* Apply the complementary ramp-down $w_{\text{out}} = 1 - w_{\text{in}} \in [1 \to 0]$ to the **second half** $\hat{y}_k[S \dots W-1]$ and store it as $\text{Tail}_k$ for step $k+1$.



---

### 3. Achieving Instant "Tap-to-Play" Startup Latency

Because a $3.0\text{s}$ window on SCNet-Small takes roughly **120–220 ms** to infer on a modern iPhone (a ~15x real-time speed factor), you can begin playback in under **250 ms**:

1. For the very first chunk ($k=0$), pad the first $1.5\text{s}$ of the input window with silence (zeros) and load the first $1.5\text{s}$ of the song into the second half of the window.
2. Run inference once (~180 ms) to prime $\text{Tail}_0$, then immediately run chunk $k=1$ (~180 ms) to produce the first $1.5\text{s}$ of playable output and start `AVAudioEngine`.
3. From that point onward, the Producer thread only needs ~180 ms of compute every $1,500\text{ ms}$ of playback time, leaving the device cool and well below background memory limits.

---

## Interactive Architecture Explorer

Explore the end-to-end iOS audio streaming and CoreML inference pipeline below. Click on any stage or step through the pipeline to inspect buffer shapes, hardware execution targets, and threading rules: