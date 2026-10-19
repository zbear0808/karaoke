import Foundation
import Atomics

/// A lock-free, Single-Producer Single-Consumer (SPSC) circular audio buffer
/// designed to bridge the background CoreML inference worker with the real-time audio render callback.
public final class LockFreePCMBuffer {

    public let capacityFrames: Int
    public let channelCount: Int
    private let totalScalarCapacity: Int

    /// Interleaved or split stereo raw buffer storage in flat RAM.
    /// Format: Stereo interleaved [L0, R0, L1, R1, ...]
    private let bufferPointer: UnsafeMutablePointer<Float>

    /// Atomic read and write indices (in audio frames, not scalars).
    private let atomicReadIndex: ManagedAtomic<Int>
    private let atomicWriteIndex: ManagedAtomic<Int>

    public init(capacityFrames: Int = SCNetEngineConfig.ringBufferFrameCapacity, channelCount: Int = SCNetEngineConfig.channelCount) {
        precondition(capacityFrames > 0 && channelCount > 0, "Capacity and channel count must be positive")
        self.capacityFrames = capacityFrames
        self.channelCount = channelCount
        self.totalScalarCapacity = capacityFrames * channelCount

        let pageAlignment = SCNetEngineConfig.pageAlignmentBytes
        let totalBytes = totalScalarCapacity * MemoryLayout<Float>.stride

        var rawPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        let status = posix_memalign(&rawPtr, pageAlignment, totalBytes)
        precondition(status == 0 && rawPtr != nil, "Failed to allocate aligned ring buffer")
        #else
        rawPtr = UnsafeMutableRawPointer.allocate(byteCount: totalBytes, alignment: pageAlignment)
        #endif

        self.bufferPointer = rawPtr!.bindMemory(to: Float.self, capacity: totalScalarCapacity)
        memset(rawPtr!, 0, totalBytes)

        self.atomicReadIndex = ManagedAtomic<Int>(0)
        self.atomicWriteIndex = ManagedAtomic<Int>(0)
    }

    deinit {
        #if canImport(Darwin)
        free(UnsafeMutableRawPointer(bufferPointer))
        #else
        UnsafeMutableRawPointer(bufferPointer).deallocate()
        #endif
    }

    // MARK: - Buffer Metrics

    /// Total frames currently available for the consumer to read.
    public var availableFramesToRead: Int {
        let writeIdx = atomicWriteIndex.load(ordering: .acquiring)
        let readIdx = atomicReadIndex.load(ordering: .acquiring)
        let delta = writeIdx - readIdx
        return delta >= 0 ? delta : (capacityFrames + delta)
    }

    /// Free frames available for the producer to write without overwriting unread data.
    public var availableFramesToWrite: Int {
        return capacityFrames - 1 - availableFramesToRead
    }

    /// Reset ring buffer indices and clear memory.
    public func reset() {
        atomicReadIndex.store(0, ordering: .releasing)
        atomicWriteIndex.store(0, ordering: .releasing)
        memset(bufferPointer, 0, totalScalarCapacity * MemoryLayout<Float>.stride)
    }

    // MARK: - Producer Operations (Inference Thread)

    /// Writes interleaved stereo Float32 audio frames into the ring buffer.
    /// - Returns: Actual number of frames successfully written.
    @discardableResult
    public func write(from source: UnsafePointer<Float>, frameCount: Int) -> Int {
        let framesAvailable = availableFramesToWrite
        let framesToWrite = min(frameCount, framesAvailable)
        guard framesToWrite > 0 else { return 0 }

        let currentWrite = atomicWriteIndex.load(ordering: .acquiring)
        let startScalar = currentWrite * channelCount
        let firstChunkFrames = min(framesToWrite, capacityFrames - currentWrite)
        let firstChunkScalars = firstChunkFrames * channelCount

        // First contiguous segment
        memcpy(bufferPointer.advanced(by: startScalar), source, firstChunkScalars * MemoryLayout<Float>.size)

        // Second wrap-around segment if needed
        let remainingFrames = framesToWrite - firstChunkFrames
        if remainingFrames > 0 {
            let remainingScalars = remainingFrames * channelCount
            memcpy(bufferPointer, source.advanced(by: firstChunkScalars), remainingScalars * MemoryLayout<Float>.size)
        }

        let nextWrite = (currentWrite + framesToWrite) % capacityFrames
        atomicWriteIndex.store(nextWrite, ordering: .releasing)
        return framesToWrite
    }

    /// Writes split planar stereo channels into interleaved ring buffer format.
    @discardableResult
    public func writePlanar(leftChannel: UnsafePointer<Float>, rightChannel: UnsafePointer<Float>, frameCount: Int) -> Int {
        let framesAvailable = availableFramesToWrite
        let framesToWrite = min(frameCount, framesAvailable)
        guard framesToWrite > 0 else { return 0 }

        let currentWrite = atomicWriteIndex.load(ordering: .acquiring)

        for i in 0..<framesToWrite {
            let destFrameIndex = (currentWrite + i) % capacityFrames
            let destScalarIndex = destFrameIndex * 2
            bufferPointer[destScalarIndex] = leftChannel[i]
            bufferPointer[destScalarIndex + 1] = rightChannel[i]
        }

        let nextWrite = (currentWrite + framesToWrite) % capacityFrames
        atomicWriteIndex.store(nextWrite, ordering: .releasing)
        return framesToWrite
    }

    // MARK: - Consumer Operations (Audio Render Thread)

    /// Reads interleaved stereo Float32 frames from the ring buffer into the destination buffer.
    /// If fewer frames than requested are available, remaining destination frames are zero-filled (underrun protection).
    /// - Returns: Actual number of valid frames read.
    @discardableResult
    public func read(into destination: UnsafeMutablePointer<Float>, frameCount: Int) -> Int {
        let framesAvailable = availableFramesToRead
        let framesToRead = min(frameCount, framesAvailable)

        if framesToRead > 0 {
            let currentRead = atomicReadIndex.load(ordering: .acquiring)
            let startScalar = currentRead * channelCount
            let firstChunkFrames = min(framesToRead, capacityFrames - currentRead)
            let firstChunkScalars = firstChunkFrames * channelCount

            memcpy(destination, bufferPointer.advanced(by: startScalar), firstChunkScalars * MemoryLayout<Float>.size)

            let remainingFrames = framesToRead - firstChunkFrames
            if remainingFrames > 0 {
                let remainingScalars = remainingFrames * channelCount
                memcpy(destination.advanced(by: firstChunkScalars), bufferPointer, remainingScalars * MemoryLayout<Float>.size)
            }

            let nextRead = (currentRead + framesToRead) % capacityFrames
            atomicReadIndex.store(nextRead, ordering: .releasing)
        }

        // Zero-fill underrun frames to prevent audible glitch/garbage noise
        let underrunFrames = frameCount - framesToRead
        if underrunFrames > 0 {
            let startUnderrunScalar = framesToRead * channelCount
            let underrunScalars = underrunFrames * channelCount
            memset(destination.advanced(by: startUnderrunScalar), 0, underrunScalars * MemoryLayout<Float>.size)
        }

        return framesToRead
    }
}

