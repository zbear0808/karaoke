import Foundation
#if canImport(CoreML)
import CoreML
#endif
#if canImport(Darwin)
import Darwin
#endif

/// Pre-allocates page-aligned memory blocks (16 KB on Apple Silicon) shared directly
/// between vDSP transforms and CoreML to eliminate heap allocations during continuous streaming.
public final class PageAlignedTensorPool {

    // Raw memory pointers aligned to 16,384 bytes (16 KB page boundary)
    private let rawInputPointer: UnsafeMutableRawPointer
    private let rawOutputPointer: UnsafeMutableRawPointer
    private let rawScratchPointer: UnsafeMutableRawPointer

    /// Typed pointers for zero-copy access
    public let inputFloat16Pointer: UnsafeMutablePointer<UInt16>
    public let outputFloat16Pointer: UnsafeMutablePointer<UInt16>
    public let scratchFloat32Pointer: UnsafeMutablePointer<Float>

    #if canImport(CoreML)
    /// Input MLMultiArray [1, 4, 2049, 130] in Float16 wrapping page-aligned rawInputPointer
    public let inputMultiArray: MLMultiArray

    /// Output MLMultiArray [1, 4, 2049, 130] in Float16 wrapping page-aligned rawOutputPointer
    public let outputMultiArray: MLMultiArray
    #endif

    public init() {
        let pageAlignment = SCNetEngineConfig.pageAlignmentBytes
        let f16Bytes = SCNetEngineConfig.tensorFloat16ByteCount
        let f32Bytes = SCNetEngineConfig.tensorFloat32ByteCount

        // 1. Allocate 16KB page-aligned input buffer
        var inPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        let inStatus = posix_memalign(&inPtr, pageAlignment, f16Bytes)
        precondition(inStatus == 0 && inPtr != nil, "posix_memalign failed for input buffer")
        #else
        inPtr = UnsafeMutableRawPointer.allocate(byteCount: f16Bytes, alignment: pageAlignment)
        #endif
        self.rawInputPointer = inPtr!
        self.inputFloat16Pointer = rawInputPointer.bindMemory(to: UInt16.self, capacity: SCNetEngineConfig.tensorElementCount)
        memset(rawInputPointer, 0, f16Bytes)

        // 2. Allocate 16KB page-aligned output buffer
        var outPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        let outStatus = posix_memalign(&outPtr, pageAlignment, f16Bytes)
        precondition(outStatus == 0 && outPtr != nil, "posix_memalign failed for output buffer")
        #else
        outPtr = UnsafeMutableRawPointer.allocate(byteCount: f16Bytes, alignment: pageAlignment)
        #endif
        self.rawOutputPointer = outPtr!
        self.outputFloat16Pointer = rawOutputPointer.bindMemory(to: UInt16.self, capacity: SCNetEngineConfig.tensorElementCount)
        memset(rawOutputPointer, 0, f16Bytes)

        // 3. Allocate 16KB page-aligned scratch Float32 buffer for vDSP planar FFT operations
        var scratchPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        let scratchStatus = posix_memalign(&scratchPtr, pageAlignment, f32Bytes)
        precondition(scratchStatus == 0 && scratchPtr != nil, "posix_memalign failed for scratch buffer")
        #else
        scratchPtr = UnsafeMutableRawPointer.allocate(byteCount: f32Bytes, alignment: pageAlignment)
        #endif
        self.rawScratchPointer = scratchPtr!
        self.scratchFloat32Pointer = rawScratchPointer.bindMemory(to: Float.self, capacity: SCNetEngineConfig.tensorElementCount)
        memset(rawScratchPointer, 0, f32Bytes)

        // 4. Wrap pre-allocated page-aligned buffers into MLMultiArrays with zero copy
        #if canImport(CoreML)
        let shape = SCNetEngineConfig.tensorShape.map { NSNumber(value: $0) }
        let strides = SCNetEngineConfig.tensorStrides.map { NSNumber(value: $0) }

        do {
            self.inputMultiArray = try MLMultiArray(
                dataPointer: rawInputPointer,
                shape: shape,
                dataType: .float16,
                strides: strides,
                deallocator: { _ in } // Managed by PageAlignedTensorPool deinit
            )
            self.outputMultiArray = try MLMultiArray(
                dataPointer: rawOutputPointer,
                shape: shape,
                dataType: .float16,
                strides: strides,
                deallocator: { _ in } // Managed by PageAlignedTensorPool deinit
            )
        } catch {
            fatalError("Failed to initialize zero-copy MLMultiArray: \(error)")
        }
        #endif
    }

    deinit {
        #if canImport(Darwin)
        free(rawInputPointer)
        free(rawOutputPointer)
        free(rawScratchPointer)
        #else
        rawInputPointer.deallocate()
        rawOutputPointer.deallocate()
        rawScratchPointer.deallocate()
        #endif
    }

    /// Resets all tensor buffers to zero without deallocating or reallocating memory.
    public func zeroOut() {
        memset(rawInputPointer, 0, SCNetEngineConfig.tensorFloat16ByteCount)
        memset(rawOutputPointer, 0, SCNetEngineConfig.tensorFloat16ByteCount)
        memset(rawScratchPointer, 0, SCNetEngineConfig.tensorFloat32ByteCount)
    }
}

