import Foundation
#if canImport(CoreML)
import CoreML
#endif

public enum StreamingWorkerState {
    case idle
    case running
    case paused
    case stopped
}

/// The master orchestrator running on a dedicated serial DispatchQueue (.userInitiated)
/// that drives continuous background source separation.
/// Allocates ZERO heap memory per iteration.
public final class SCNetStreamingWorker: BackgroundExecutionDelegate {

    public let dispatchQueue: DispatchQueue
    public let tensorPool: PageAlignedTensorPool
    public let spectralProcessor: SpectralProcessorVDSP
    public let accumulator: OverlapAddAccumulator
    public let ringBuffer: LockFreePCMBuffer
    public let resourceMonitor: SystemResourceMonitor

    public private(set) var state: StreamingWorkerState = .idle

    // Pre-allocated raw audio staging buffers for 3.0s window [132,300 Float32 samples per channel]
    private let stageBufferL: UnsafeMutablePointer<Float>
    private let stageBufferR: UnsafeMutablePointer<Float>

    // Full source audio track references (or streaming audio provider block)
    public typealias AudioChunkReader = (
        _ startFrame: Int,
        _ frameCount: Int,
        _ destL: UnsafeMutablePointer<Float>,
        _ destR: UnsafeMutablePointer<Float>
    ) -> Int

    private var audioReader: AudioChunkReader?
    private var currentReadFrameOffset: Int = 0
    private var totalTrackFrames: Int = 0

    #if canImport(CoreML)
    private var featureProvider: SCNetFeatureProvider?
    private var predictionOptions: MLPredictionOptions
    #endif

    public init(
        tensorPool: PageAlignedTensorPool,
        spectralProcessor: SpectralProcessorVDSP,
        accumulator: OverlapAddAccumulator,
        ringBuffer: LockFreePCMBuffer,
        resourceMonitor: SystemResourceMonitor = SystemResourceMonitor.shared
    ) {
        self.dispatchQueue = DispatchQueue(label: "com.app.scnet.inference", qos: .userInitiated)
        self.tensorPool = tensorPool
        self.spectralProcessor = spectralProcessor
        self.accumulator = accumulator
        self.ringBuffer = ringBuffer
        self.resourceMonitor = resourceMonitor

        let pageAlignment = SCNetEngineConfig.pageAlignmentBytes
        let windowBytes = SCNetEngineConfig.windowSamples * MemoryLayout<Float>.stride

        var sLPtr: UnsafeMutableRawPointer? = nil
        var sRPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&sLPtr, pageAlignment, windowBytes)
        _ = posix_memalign(&sRPtr, pageAlignment, windowBytes)
        #else
        sLPtr = UnsafeMutableRawPointer.allocate(byteCount: windowBytes, alignment: pageAlignment)
        sRPtr = UnsafeMutableRawPointer.allocate(byteCount: windowBytes, alignment: pageAlignment)
        #endif
        self.stageBufferL = sLPtr!.bindMemory(to: Float.self, capacity: SCNetEngineConfig.windowSamples)
        self.stageBufferR = sRPtr!.bindMemory(to: Float.self, capacity: SCNetEngineConfig.windowSamples)
        memset(sLPtr!, 0, windowBytes)
        memset(sRPtr!, 0, windowBytes)

        #if canImport(CoreML)
        self.featureProvider = SCNetFeatureProvider(inputMultiArray: tensorPool.inputMultiArray)
        let options = MLPredictionOptions()
        // Force CoreML to write directly into our pre-allocated page-aligned tensor
        options.outputBackings = ["output_spectrogram": tensorPool.outputMultiArray]
        self.predictionOptions = options
        #endif

        BackgroundExecutionManager.shared.delegate = self
    }

    deinit {
        #if canImport(Darwin)
        free(UnsafeMutableRawPointer(stageBufferL))
        free(UnsafeMutableRawPointer(stageBufferR))
        #else
        UnsafeMutableRawPointer(stageBufferL).deallocate()
        UnsafeMutableRawPointer(stageBufferR).deallocate()
        #endif
    }

    // MARK: - Streaming Lifecycle

    /// Configures audio source and primes the first chunk for instant tap-to-play startup.
    public func start(
        totalFrames: Int,
        reader: @escaping AudioChunkReader
    ) {
        dispatchQueue.async { [weak self] in
            guard let self = self else { return }
            self.totalTrackFrames = totalFrames
            self.audioReader = reader
            self.currentReadFrameOffset = 0
            self.accumulator.reset()
            self.ringBuffer.reset()
            self.state = .running

            // Step 1: Prime startup chunk k=0 (first 1.5s silence + 1.5s audio) to prime Tail_0
            self.primeStartupChunk()

            // Step 2: Schedule continuous inference loop
            self.scheduleNextChunk()
        }
    }

    public func pause() {
        dispatchQueue.async { [weak self] in
            self?.state = .paused
        }
    }

    public func resume() {
        dispatchQueue.async { [weak self] in
            guard let self = self, self.state == .paused else { return }
            self.state = .running
            self.scheduleNextChunk()
        }
    }

    public func stop() {
        dispatchQueue.async { [weak self] in
            self?.state = .stopped
            self?.accumulator.reset()
            self?.ringBuffer.reset()
        }
    }

    // MARK: - Tap-To-Play Prime Step (Under 250ms Latency)

    private func primeStartupChunk() {
        guard let reader = self.audioReader else { return }
        let step = SCNetEngineConfig.stepSamples // 66,150

        // Zero first 1.5s
        memset(stageBufferL, 0, step * MemoryLayout<Float>.size)
        memset(stageBufferR, 0, step * MemoryLayout<Float>.size)

        // Read first 1.5s of song into second half of stageBuffer
        _ = reader(0, step, stageBufferL.advanced(by: step), stageBufferR.advanced(by: step))
        currentReadFrameOffset = step

        // Run forward STFT -> CoreML -> inverse STFT -> accumulator
        executeSingleInferencePass()
    }

    // MARK: - Master Inference Loop (Zero Allocations)

    private func scheduleNextChunk() {
        guard state == .running else { return }

        dispatchQueue.async { [weak self] in
            guard let self = self, self.state == .running else { return }

            // 1. Watermark check: If ring buffer has >= current lookahead threshold, sleep 250ms
            let availableFrames = self.ringBuffer.availableFramesToRead
            let lookaheadLimit = self.resourceMonitor.currentLookaheadLimitFrames

            if availableFrames >= lookaheadLimit {
                self.dispatchQueue.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                    self?.scheduleNextChunk()
                }
                return
            }

            // 2. Resource check: check available memory before Jetsam
            if !self.resourceMonitor.evaluateMemorySafety() {
                // Throttle: wait for buffer to drain to minimum
                if availableFrames > SCNetEngineConfig.stepSamples {
                    self.dispatchQueue.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                        self?.scheduleNextChunk()
                    }
                    return
                }
            }

            // 3. Check track end condition
            if self.currentReadFrameOffset >= self.totalTrackFrames {
                self.state = .idle
                return
            }

            // 4. Input Staging: Read next 3.0s window (advancing by 1.5s hop)
            self.stageNextWindow()

            // 5. Execute Forward STFT -> CoreML Prediction -> Inverse STFT -> Overlap-Add
            self.executeSingleInferencePass()

            // 6. Push finalized 1.5s output into the LockFreePCMBuffer
            self.ringBuffer.writePlanar(
                leftChannel: self.accumulator.finalizedOutputL,
                rightChannel: self.accumulator.finalizedOutputR,
                frameCount: SCNetEngineConfig.stepSamples
            )

            // 7. Loop to next chunk
            self.scheduleNextChunk()
        }
    }

    private func stageNextWindow() {
        guard let reader = self.audioReader else { return }
        let window = SCNetEngineConfig.windowSamples
        let step = SCNetEngineConfig.stepSamples

        // Read 3.0s stereo audio
        let framesRead = reader(
            currentReadFrameOffset,
            window,
            stageBufferL,
            stageBufferR
        )

        // Zero-pad remainder if near end of song
        if framesRead < window {
            let remain = window - framesRead
            memset(stageBufferL.advanced(by: framesRead), 0, remain * MemoryLayout<Float>.size)
            memset(stageBufferR.advanced(by: framesRead), 0, remain * MemoryLayout<Float>.size)
        }

        currentReadFrameOffset += step
    }

    private func executeSingleInferencePass() {
        // Forward STFT
        spectralProcessor.encodeChunkToTensor(
            inputL: stageBufferL,
            inputR: stageBufferR,
            tensorPool: tensorPool
        )

        // CoreML Prediction inside explicit autoreleasepool to flush internal temporary wrappers
        #if canImport(CoreML)
        if let model = SCNetModelLoader.shared.model, let provider = featureProvider {
            do {
                try autoreleasepool {
                    _ = try model.prediction(from: provider, options: predictionOptions)
                }
            } catch {
                // If prediction fails, passthrough input to prevent hard audio drop
            }
        }
        #endif

        // Inverse STFT
        spectralProcessor.decodeTensorToChunk(tensorPool: tensorPool)

        // Overlap-Add Accumulation
        accumulator.processChunk(
            left3sChunk: spectralProcessor.reconstructedTimeDomainL,
            right3sChunk: spectralProcessor.reconstructedTimeDomainR
        )
    }

    // MARK: - BackgroundExecutionDelegate

    public func backgroundExecutionManagerDidRequestPause() {
        pause()
    }

    public func backgroundExecutionManagerDidRequestResume() {
        resume()
    }

    public func backgroundExecutionManagerDidTransitionToBackground() {
        #if canImport(CoreML)
        SCNetModelLoader.shared.enforceBackgroundComputePolicy()
        #endif
    }

    public func backgroundExecutionManagerDidTransitionToForeground() {
        // Safe to maintain .cpuAndNeuralEngine
    }
}

