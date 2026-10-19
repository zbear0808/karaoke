import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Stitches consecutive 3.0-second separated chunks into a click-free continuous
/// audio stream using 50% overlap-add (OLA) crossfading.
/// Pre-allocates all fade ramps and carry-over buffers to guarantee zero runtime heap allocations.
public final class OverlapAddAccumulator {

    public let stepSamples: Int
    public let windowSamples: Int
    public let channelCount: Int

    /// Pre-allocated carry-over tail buffers: [channelCount, stepSamples]
    private let carryOverTailL: UnsafeMutablePointer<Float>
    private let carryOverTailR: UnsafeMutablePointer<Float>

    /// Pre-computed linear or sin^2 fade ramps of length stepSamples (66,150)
    private let fadeInRamp: UnsafeMutablePointer<Float>
    private let fadeOutRamp: UnsafeMutablePointer<Float>

    /// Pre-allocated output buffer for the finalized 1.5s chunk
    public let finalizedOutputL: UnsafeMutablePointer<Float>
    public let finalizedOutputR: UnsafeMutablePointer<Float>

    public init(
        stepSamples: Int = SCNetEngineConfig.stepSamples,
        windowSamples: Int = SCNetEngineConfig.windowSamples,
        channelCount: Int = SCNetEngineConfig.channelCount
    ) {
        precondition(windowSamples == stepSamples * 2, "Window must be exactly double step size for 50% OLA")
        self.stepSamples = stepSamples
        self.windowSamples = windowSamples
        self.channelCount = channelCount

        let pageAlignment = SCNetEngineConfig.pageAlignmentBytes
        let tailBytes = stepSamples * MemoryLayout<Float>.stride

        // 1. Allocate aligned carry-over tails
        var tailLPtr: UnsafeMutableRawPointer? = nil
        var tailRPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&tailLPtr, pageAlignment, tailBytes)
        _ = posix_memalign(&tailRPtr, pageAlignment, tailBytes)
        #else
        tailLPtr = UnsafeMutableRawPointer.allocate(byteCount: tailBytes, alignment: pageAlignment)
        tailRPtr = UnsafeMutableRawPointer.allocate(byteCount: tailBytes, alignment: pageAlignment)
        #endif
        self.carryOverTailL = tailLPtr!.bindMemory(to: Float.self, capacity: stepSamples)
        self.carryOverTailR = tailRPtr!.bindMemory(to: Float.self, capacity: stepSamples)
        memset(tailLPtr!, 0, tailBytes)
        memset(tailRPtr!, 0, tailBytes)

        // 2. Allocate aligned fade ramps
        var fadeInPtr: UnsafeMutableRawPointer? = nil
        var fadeOutPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&fadeInPtr, pageAlignment, tailBytes)
        _ = posix_memalign(&fadeOutPtr, pageAlignment, tailBytes)
        #else
        fadeInPtr = UnsafeMutableRawPointer.allocate(byteCount: tailBytes, alignment: pageAlignment)
        fadeOutPtr = UnsafeMutableRawPointer.allocate(byteCount: tailBytes, alignment: pageAlignment)
        #endif
        self.fadeInRamp = fadeInPtr!.bindMemory(to: Float.self, capacity: stepSamples)
        self.fadeOutRamp = fadeOutPtr!.bindMemory(to: Float.self, capacity: stepSamples)

        // 3. Allocate finalized 1.5s output buffers
        var outLPtr: UnsafeMutableRawPointer? = nil
        var outRPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&outLPtr, pageAlignment, tailBytes)
        _ = posix_memalign(&outRPtr, pageAlignment, tailBytes)
        #else
        outLPtr = UnsafeMutableRawPointer.allocate(byteCount: tailBytes, alignment: pageAlignment)
        outRPtr = UnsafeMutableRawPointer.allocate(byteCount: tailBytes, alignment: pageAlignment)
        #endif
        self.finalizedOutputL = outLPtr!.bindMemory(to: Float.self, capacity: stepSamples)
        self.finalizedOutputR = outRPtr!.bindMemory(to: Float.self, capacity: stepSamples)
        memset(outLPtr!, 0, tailBytes)
        memset(outRPtr!, 0, tailBytes)

        // 4. Precompute linear crossfade ramps:
        // fadeIn:  0.0 -> 1.0
        // fadeOut: 1.0 -> 0.0
        // (Sum of ramps at any point = 1.0 for perfect amplitude reconstruction)
        let nFloat = Float(stepSamples)
        for i in 0..<stepSamples {
            let ramp = Float(i) / nFloat
            fadeInRamp[i] = ramp
            fadeOutRamp[i] = 1.0 - ramp
        }
    }

    deinit {
        #if canImport(Darwin)
        free(UnsafeMutableRawPointer(carryOverTailL))
        free(UnsafeMutableRawPointer(carryOverTailR))
        free(UnsafeMutableRawPointer(fadeInRamp))
        free(UnsafeMutableRawPointer(fadeOutRamp))
        free(UnsafeMutableRawPointer(finalizedOutputL))
        free(UnsafeMutableRawPointer(finalizedOutputR))
        #else
        UnsafeMutableRawPointer(carryOverTailL).deallocate()
        UnsafeMutableRawPointer(carryOverTailR).deallocate()
        UnsafeMutableRawPointer(fadeInRamp).deallocate()
        UnsafeMutableRawPointer(fadeOutRamp).deallocate()
        UnsafeMutableRawPointer(finalizedOutputL).deallocate()
        UnsafeMutableRawPointer(finalizedOutputR).deallocate()
        #endif
    }

    /// Resets the accumulator state for a new song or after a seek.
    public func reset() {
        let tailBytes = stepSamples * MemoryLayout<Float>.stride
        memset(carryOverTailL, 0, tailBytes)
        memset(carryOverTailR, 0, tailBytes)
        memset(finalizedOutputL, 0, tailBytes)
        memset(finalizedOutputR, 0, tailBytes)
    }

    /// Processes a full 3.0s (132,300 sample) stereo chunk from iSTFT.
    ///
    /// Pipeline:
    /// 1. First half (0.0s - 1.5s): multiply by fadeInRamp, add carryOverTail -> finalizedOutput.
    /// 2. Second half (1.5s - 3.0s): multiply by fadeOutRamp -> store into carryOverTail for next cycle.
    ///
    /// - Parameters:
    ///   - left3sChunk: Input 3.0s left channel buffer (windowSamples = 132,300)
    ///   - right3sChunk: Input 3.0s right channel buffer (windowSamples = 132,300)
    public func processChunk(
        left3sChunk: UnsafePointer<Float>,
        right3sChunk: UnsafePointer<Float>
    ) {
        #if canImport(Accelerate)
        let n = vDSP_Length(stepSamples)

        // --- LEFT CHANNEL ---
        // 1. Ramped first half = left3sChunk[0..<stepSamples] * fadeInRamp
        vDSP_vmul(left3sChunk, 1, fadeInRamp, 1, finalizedOutputL, 1, n)
        // 2. Finalized output = Ramped first half + carryOverTailL
        vDSP_vadd(finalizedOutputL, 1, carryOverTailL, 1, finalizedOutputL, 1, n)
        // 3. Next tail = left3sChunk[stepSamples..<windowSamples] * fadeOutRamp
        let leftSecondHalf = left3sChunk.advanced(by: stepSamples)
        vDSP_vmul(leftSecondHalf, 1, fadeOutRamp, 1, carryOverTailL, 1, n)

        // --- RIGHT CHANNEL ---
        // 1. Ramped first half = right3sChunk[0..<stepSamples] * fadeInRamp
        vDSP_vmul(right3sChunk, 1, fadeInRamp, 1, finalizedOutputR, 1, n)
        // 2. Finalized output = Ramped first half + carryOverTailR
        vDSP_vadd(finalizedOutputR, 1, carryOverTailR, 1, finalizedOutputR, 1, n)
        // 3. Next tail = right3sChunk[stepSamples..<windowSamples] * fadeOutRamp
        let rightSecondHalf = right3sChunk.advanced(by: stepSamples)
        vDSP_vmul(rightSecondHalf, 1, fadeOutRamp, 1, carryOverTailR, 1, n)

        #else
        // Fallback for non-Accelerate platforms / tests
        for i in 0..<stepSamples {
            finalizedOutputL[i] = (left3sChunk[i] * fadeInRamp[i]) + carryOverTailL[i]
            carryOverTailL[i] = left3sChunk[stepSamples + i] * fadeOutRamp[i]

            finalizedOutputR[i] = (right3sChunk[i] * fadeInRamp[i]) + carryOverTailR[i]
            carryOverTailR[i] = right3sChunk[stepSamples + i] * fadeOutRamp[i]
        }
        #endif
    }
}

