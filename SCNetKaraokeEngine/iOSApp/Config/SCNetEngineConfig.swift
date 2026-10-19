import Foundation

/// Compile-time and runtime DSP/tensor configuration constants
/// derived strictly from target hardware constraints and SCNet architecture specifications.
public enum SCNetEngineConfig {
    /// Audio sampling rate in Hertz.
    public static let sampleRate: Double = 44100.0

    /// Number of audio channels (stereo).
    public static let channelCount: Int = 2

    /// FFT window size (N).
    public static let fftSize: Int = 4096

    /// STFT hop / step length in samples.
    public static let hopLength: Int = 1024

    /// Number of positive frequency bins (N / 2 + 1) for real-to-complex FFT.
    public static let freqBins: Int = 2049

    /// Number of time frames in the 3.0s spectrogram window.
    public static let stftFrames: Int = 130

    /// Exact 3.0 seconds duration in samples per channel (3.0 * 44100).
    public static let windowSamples: Int = 132300

    /// Exact 1.5 seconds sliding hop size (50% overlap) in samples per channel (1.5 * 44100).
    public static let stepSamples: Int = 66150

    /// Center-padded window samples: (stftFrames - 1) * hopLength + fftSize = (129 * 1024) + 4096 = 133120.
    /// Provides exact center-padded parity with PyTorch STFT (center=True, pad_mode='reflect' or zero).
    public static let paddedWindowSamples: Int = 133120

    /// Half of FFT size for symmetric center padding.
    public static let paddingSamples: Int = fftSize / 2 // 2048

    /// Input/Output tensor shape: [Batch, Channels (L_re, L_im, R_re, R_im), FreqBins, TimeFrames].
    public static let tensorShape: [Int] = [1, 4, 2049, 130]

    /// Strides for contiguous [1, 4, 2049, 130] row-major tensor layout.
    public static let tensorStrides: [Int] = [
        4 * 2049 * 130, // Batch stride
        2049 * 130,     // Channel stride: 266370
        130,            // FreqBin stride: 130
        1               // TimeFrame stride: 1
    ]

    /// Total number of scalar elements in input/output tensors: 1 * 4 * 2049 * 130 = 1,065,480.
    public static let tensorElementCount: Int = 1 * 4 * freqBins * stftFrames

    /// Total byte count of tensor in Float16: 1,065,480 * 2 = 2,130,960 bytes (~2.03 MB).
    public static let tensorFloat16ByteCount: Int = tensorElementCount * MemoryLayout<UInt16>.stride

    /// Total byte count of tensor in Float32: 1,065,480 * 4 = 4,261,920 bytes (~4.06 MB).
    public static let tensorFloat32ByteCount: Int = tensorElementCount * MemoryLayout<Float>.stride

    /// Memory page alignment for Apple Silicon Direct Memory Access (DMA) and ANE acceleration (16 KB).
    public static let pageAlignmentBytes: Int = 16384

    /// Maximum capacity of SPSC audio ring buffer in seconds (12.0s).
    public static let ringBufferDurationSeconds: Double = 12.0

    /// Ring buffer capacity in frames (12.0 * 44100 = 529,200 stereo frames).
    public static let ringBufferFrameCapacity: Int = Int(ringBufferDurationSeconds * sampleRate)

    /// Lookahead threshold in seconds to pace background inference worker (6.0s = 264,600 frames).
    public static let normalLookaheadSeconds: Double = 6.0
    public static let normalLookaheadFrames: Int = Int(normalLookaheadSeconds * sampleRate)

    /// Throttled lookahead threshold under thermal or low memory pressure (3.0s = 132,300 frames).
    public static let throttledLookaheadSeconds: Double = 3.0
    public static let throttledLookaheadFrames: Int = Int(throttledLookaheadSeconds * sampleRate)

    /// Emergency Jetsam headroom watermark in bytes (40 MB).
    public static let emergencyMemoryWatermarkBytes: UInt64 = 40 * 1024 * 1024
}

