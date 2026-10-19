import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Executes the Forward STFT and Inverse iSTFT on the CPU using Apple's Accelerate framework (vDSP/vImage),
/// completely replacing PyTorch's torch.stft and torch.istft.
/// Eliminates complex tensor / spectral fallback on the Apple Neural Engine.
public final class SpectralProcessorVDSP {

    public let fftSize: Int
    public let hopLength: Int
    public let freqBins: Int
    public let stftFrames: Int
    public let windowSamples: Int
    public let paddedWindowSamples: Int
    public let paddingSamples: Int

    #if canImport(Accelerate)
    private var dftSetupForward: vDSP_DFT_Setup?
    private var dftSetupInverse: vDSP_DFT_Setup?
    #endif

    // Pre-allocated analysis and synthesis window buffers
    private let hannWindow: UnsafeMutablePointer<Float>
    private let synthesisColaNorm: UnsafeMutablePointer<Float>

    // Pre-allocated padded time-domain scratch buffers (133,120 samples)
    private let paddedSignalL: UnsafeMutablePointer<Float>
    private let paddedSignalR: UnsafeMutablePointer<Float>

    // Pre-allocated split-complex DFT scratch buffers (4096 samples each)
    private let dftRealIn: UnsafeMutablePointer<Float>
    private let dftImagIn: UnsafeMutablePointer<Float>
    private let dftRealOut: UnsafeMutablePointer<Float>
    private let dftImagOut: UnsafeMutablePointer<Float>

    // Pre-allocated time-domain reconstruction buffer for 3.0s chunk (132,300 samples)
    public let reconstructedTimeDomainL: UnsafeMutablePointer<Float>
    public let reconstructedTimeDomainR: UnsafeMutablePointer<Float>
    private let istftAccumPaddedL: UnsafeMutablePointer<Float>
    private let istftAccumPaddedR: UnsafeMutablePointer<Float>

    public init(
        fftSize: Int = SCNetEngineConfig.fftSize,
        hopLength: Int = SCNetEngineConfig.hopLength,
        freqBins: Int = SCNetEngineConfig.freqBins,
        stftFrames: Int = SCNetEngineConfig.stftFrames,
        windowSamples: Int = SCNetEngineConfig.windowSamples,
        paddedWindowSamples: Int = SCNetEngineConfig.paddedWindowSamples
    ) {
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.freqBins = freqBins
        self.stftFrames = stftFrames
        self.windowSamples = windowSamples
        self.paddedWindowSamples = paddedWindowSamples
        self.paddingSamples = fftSize / 2

        let pageAlignment = SCNetEngineConfig.pageAlignmentBytes

        // 1. Setup Accelerate DFT setups
        #if canImport(Accelerate)
        self.dftSetupForward = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )
        self.dftSetupInverse = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.INVERSE
        )
        #endif

        // 2. Allocate Hann window & synthesis Cola normalization
        var hannPtr: UnsafeMutableRawPointer? = nil
        var colaPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&hannPtr, pageAlignment, fftSize * MemoryLayout<Float>.stride)
        _ = posix_memalign(&colaPtr, pageAlignment, paddedWindowSamples * MemoryLayout<Float>.stride)
        #else
        hannPtr = UnsafeMutableRawPointer.allocate(byteCount: fftSize * MemoryLayout<Float>.stride, alignment: pageAlignment)
        colaPtr = UnsafeMutableRawPointer.allocate(byteCount: paddedWindowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        #endif
        self.hannWindow = hannPtr!.bindMemory(to: Float.self, capacity: fftSize)
        self.synthesisColaNorm = colaPtr!.bindMemory(to: Float.self, capacity: paddedWindowSamples)

        // 3. Allocate padded time signals
        var padLPtr: UnsafeMutableRawPointer? = nil
        var padRPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&padLPtr, pageAlignment, paddedWindowSamples * MemoryLayout<Float>.stride)
        _ = posix_memalign(&padRPtr, pageAlignment, paddedWindowSamples * MemoryLayout<Float>.stride)
        #else
        padLPtr = UnsafeMutableRawPointer.allocate(byteCount: paddedWindowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        padRPtr = UnsafeMutableRawPointer.allocate(byteCount: paddedWindowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        #endif
        self.paddedSignalL = padLPtr!.bindMemory(to: Float.self, capacity: paddedWindowSamples)
        self.paddedSignalR = padRPtr!.bindMemory(to: Float.self, capacity: paddedWindowSamples)
        memset(padLPtr!, 0, paddedWindowSamples * MemoryLayout<Float>.stride)
        memset(padRPtr!, 0, paddedWindowSamples * MemoryLayout<Float>.stride)

        // 4. Allocate DFT split-complex scratch buffers
        var rInPtr: UnsafeMutableRawPointer? = nil
        var iInPtr: UnsafeMutableRawPointer? = nil
        var rOutPtr: UnsafeMutableRawPointer? = nil
        var iOutPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&rInPtr, pageAlignment, fftSize * MemoryLayout<Float>.stride)
        _ = posix_memalign(&iInPtr, pageAlignment, fftSize * MemoryLayout<Float>.stride)
        _ = posix_memalign(&rOutPtr, pageAlignment, fftSize * MemoryLayout<Float>.stride)
        _ = posix_memalign(&iOutPtr, pageAlignment, fftSize * MemoryLayout<Float>.stride)
        #else
        rInPtr = UnsafeMutableRawPointer.allocate(byteCount: fftSize * MemoryLayout<Float>.stride, alignment: pageAlignment)
        iInPtr = UnsafeMutableRawPointer.allocate(byteCount: fftSize * MemoryLayout<Float>.stride, alignment: pageAlignment)
        rOutPtr = UnsafeMutableRawPointer.allocate(byteCount: fftSize * MemoryLayout<Float>.stride, alignment: pageAlignment)
        iOutPtr = UnsafeMutableRawPointer.allocate(byteCount: fftSize * MemoryLayout<Float>.stride, alignment: pageAlignment)
        #endif
        self.dftRealIn = rInPtr!.bindMemory(to: Float.self, capacity: fftSize)
        self.dftImagIn = iInPtr!.bindMemory(to: Float.self, capacity: fftSize)
        self.dftRealOut = rOutPtr!.bindMemory(to: Float.self, capacity: fftSize)
        self.dftImagOut = iOutPtr!.bindMemory(to: Float.self, capacity: fftSize)
        memset(iInPtr!, 0, fftSize * MemoryLayout<Float>.stride)

        // 5. Allocate reconstructed output buffers
        var recLPtr: UnsafeMutableRawPointer? = nil
        var recRPtr: UnsafeMutableRawPointer? = nil
        var istftLPtr: UnsafeMutableRawPointer? = nil
        var istftRPtr: UnsafeMutableRawPointer? = nil
        #if canImport(Darwin)
        _ = posix_memalign(&recLPtr, pageAlignment, windowSamples * MemoryLayout<Float>.stride)
        _ = posix_memalign(&recRPtr, pageAlignment, windowSamples * MemoryLayout<Float>.stride)
        _ = posix_memalign(&istftLPtr, pageAlignment, paddedWindowSamples * MemoryLayout<Float>.stride)
        _ = posix_memalign(&istftRPtr, pageAlignment, paddedWindowSamples * MemoryLayout<Float>.stride)
        #else
        recLPtr = UnsafeMutableRawPointer.allocate(byteCount: windowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        recRPtr = UnsafeMutableRawPointer.allocate(byteCount: windowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        istftLPtr = UnsafeMutableRawPointer.allocate(byteCount: paddedWindowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        istftRPtr = UnsafeMutableRawPointer.allocate(byteCount: paddedWindowSamples * MemoryLayout<Float>.stride, alignment: pageAlignment)
        #endif
        self.reconstructedTimeDomainL = recLPtr!.bindMemory(to: Float.self, capacity: windowSamples)
        self.reconstructedTimeDomainR = recRPtr!.bindMemory(to: Float.self, capacity: windowSamples)
        self.istftAccumPaddedL = istftLPtr!.bindMemory(to: Float.self, capacity: paddedWindowSamples)
        self.istftAccumPaddedR = istftRPtr!.bindMemory(to: Float.self, capacity: paddedWindowSamples)

        // 6. Precompute periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N)) for n=0..N-1
        #if canImport(Accelerate)
        vDSP_hann_window(hannWindow, vDSP_Length(fftSize), Int32(vDSP_HANN_DENORM))
        #else
        for n in 0..<fftSize {
            let val = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(fftSize)))
            hannWindow[n] = Float(val)
        }
        #endif

        // 7. Precompute COLA (Constant Overlap-Add) normalization envelope: sum_m w^2[n - m*H]
        memset(colaPtr!, 0, paddedWindowSamples * MemoryLayout<Float>.stride)
        for frame in 0..<stftFrames {
            let offset = frame * hopLength
            for n in 0..<fftSize {
                let w = hannWindow[n]
                synthesisColaNorm[offset + n] += (w * w)
            }
        }
        // Protect against zero division at boundary tails
        for i in 0..<paddedWindowSamples {
            if synthesisColaNorm[i] < 1e-6 {
                synthesisColaNorm[i] = 1.0
            }
        }
    }

    deinit {
        #if canImport(Accelerate)
        if let f = dftSetupForward { vDSP_DFT_DestroySetup(f) }
        if let inv = dftSetupInverse { vDSP_DFT_DestroySetup(inv) }
        #endif
        #if canImport(Darwin)
        free(UnsafeMutableRawPointer(hannWindow))
        free(UnsafeMutableRawPointer(synthesisColaNorm))
        free(UnsafeMutableRawPointer(paddedSignalL))
        free(UnsafeMutableRawPointer(paddedSignalR))
        free(UnsafeMutableRawPointer(dftRealIn))
        free(UnsafeMutableRawPointer(dftImagIn))
        free(UnsafeMutableRawPointer(dftRealOut))
        free(UnsafeMutableRawPointer(dftImagOut))
        free(UnsafeMutableRawPointer(reconstructedTimeDomainL))
        free(UnsafeMutableRawPointer(reconstructedTimeDomainR))
        free(UnsafeMutableRawPointer(istftAccumPaddedL))
        free(UnsafeMutableRawPointer(istftAccumPaddedR))
        #else
        UnsafeMutableRawPointer(hannWindow).deallocate()
        UnsafeMutableRawPointer(synthesisColaNorm).deallocate()
        UnsafeMutableRawPointer(paddedSignalL).deallocate()
        UnsafeMutableRawPointer(paddedSignalR).deallocate()
        UnsafeMutableRawPointer(dftRealIn).deallocate()
        UnsafeMutableRawPointer(dftImagIn).deallocate()
        UnsafeMutableRawPointer(dftRealOut).deallocate()
        UnsafeMutableRawPointer(dftImagOut).deallocate()
        UnsafeMutableRawPointer(reconstructedTimeDomainL).deallocate()
        UnsafeMutableRawPointer(reconstructedTimeDomainR).deallocate()
        UnsafeMutableRawPointer(istftAccumPaddedL).deallocate()
        UnsafeMutableRawPointer(istftAccumPaddedR).deallocate()
        #endif
    }

    // MARK: - Forward STFT

    /// Center-pads raw 3.0s audio chunk and computes STFT for stereo Left & Right channels.
    /// Writes directly into tensorPool's scratch Float32 planar buffer, then bulk converts
    /// into Float16 inputMultiArray via hardware SIMD/vImage.
    public func encodeChunkToTensor(
        inputL: UnsafePointer<Float>,
        inputR: UnsafePointer<Float>,
        tensorPool: PageAlignedTensorPool
    ) {
        // 1. Center-padding with reflection (n_fft / 2 = 2048 samples on each side)
        padSignalReflect(source: inputL, destination: paddedSignalL)
        padSignalReflect(source: inputR, destination: paddedSignalR)

        // Channel offsets in the [1, 4, 2049, 130] layout:
        // Ch0: L_real (offset = 0)
        // Ch1: L_imag (offset = 2049 * 130)
        // Ch2: R_real (offset = 2 * 2049 * 130)
        // Ch3: R_imag (offset = 3 * 2049 * 130)
        let chStride = freqBins * stftFrames
        let scratch = tensorPool.scratchFloat32Pointer

        let lReal = scratch
        let lImag = scratch.advanced(by: chStride)
        let rReal = scratch.advanced(by: 2 * chStride)
        let rImag = scratch.advanced(by: 3 * chStride)

        // 2. Perform STFT for Left Channel
        computeSTFTChannel(signal: paddedSignalL, realDest: lReal, imagDest: lImag)

        // 3. Perform STFT for Right Channel
        computeSTFTChannel(signal: paddedSignalR, realDest: rReal, imagDest: rImag)

        // 4. Vector conversion Float32 -> Float16
        convertFloat32ToFloat16(
            source: scratch,
            destination: tensorPool.inputFloat16Pointer,
            elementCount: SCNetEngineConfig.tensorElementCount
        )
    }

    // MARK: - Inverse STFT

    /// Converts output Float16 tensor back to Float32, performs inverse DFT, synthesis windowing,
    /// and COLA overlap-add to reconstruct the 3.0s time-domain stereo audio.
    public func decodeTensorToChunk(
        tensorPool: PageAlignedTensorPool
    ) {
        // 1. Convert output Float16 tensor to scratch Float32
        convertFloat16ToFloat32(
            source: tensorPool.outputFloat16Pointer,
            destination: tensorPool.scratchFloat32Pointer,
            elementCount: SCNetEngineConfig.tensorElementCount
        )

        let chStride = freqBins * stftFrames
        let scratch = tensorPool.scratchFloat32Pointer

        let lReal = scratch
        let lImag = scratch.advanced(by: chStride)
        let rReal = scratch.advanced(by: 2 * chStride)
        let rImag = scratch.advanced(by: 3 * chStride)

        // 2. Perform iSTFT for Left Channel
        computeISTFTChannel(
            realSrc: lReal,
            imagSrc: lImag,
            accumPadded: istftAccumPaddedL,
            reconstructedDest: reconstructedTimeDomainL
        )

        // 3. Perform iSTFT for Right Channel
        computeISTFTChannel(
            realSrc: rReal,
            imagSrc: rImag,
            accumPadded: istftAccumPaddedR,
            reconstructedDest: reconstructedTimeDomainR
        )
    }

    // MARK: - Helper Methods

    private func padSignalReflect(source: UnsafePointer<Float>, destination: UnsafeMutablePointer<Float>) {
        let pad = paddingSamples // 2048
        let w = windowSamples    // 132300

        // Left reflection: source[pad], source[pad-1], ...
        for i in 0..<pad {
            destination[i] = source[pad - i]
        }
        // Center copy
        memcpy(destination.advanced(by: pad), source, w * MemoryLayout<Float>.size)
        // Right reflection
        for i in 0..<pad {
            destination[pad + w + i] = source[w - 2 - i]
        }
    }

    private func computeSTFTChannel(
        signal: UnsafePointer<Float>,
        realDest: UnsafeMutablePointer<Float>,
        imagDest: UnsafeMutablePointer<Float>
    ) {
        #if canImport(Accelerate)
        guard let setup = dftSetupForward else { return }

        for frame in 0..<stftFrames {
            let offset = frame * hopLength
            let frameSignal = signal.advanced(by: offset)

            // 1. Multiply frame by Hann window
            vDSP_vmul(frameSignal, 1, hannWindow, 1, dftRealIn, 1, vDSP_Length(fftSize))
            // Imaginary input is 0.0 (pre-cleared)

            // 2. Complex DFT
            vDSP_DFT_Execute(
                setup,
                dftRealIn, dftImagIn,
                dftRealOut, dftImagOut
            )

            // 3. Store the 2049 positive frequency bins into the [FreqBins, TimeFrames] matrix:
            // layout: element (f, frame) is at index (f * stftFrames + frame)
            for f in 0..<freqBins {
                let binIndex = f * stftFrames + frame
                realDest[binIndex] = dftRealOut[f]
                imagDest[binIndex] = dftImagOut[f]
            }
        }
        #endif
    }

    private func computeISTFTChannel(
        realSrc: UnsafePointer<Float>,
        imagSrc: UnsafePointer<Float>,
        accumPadded: UnsafeMutablePointer<Float>,
        reconstructedDest: UnsafeMutablePointer<Float>
    ) {
        #if canImport(Accelerate)
        guard let setup = dftSetupInverse else { return }
        memset(accumPadded, 0, paddedWindowSamples * MemoryLayout<Float>.stride)

        for frame in 0..<stftFrames {
            let offset = frame * hopLength

            // 1. Reconstruct full 4096 complex spectrum with conjugate symmetry
            for f in 0..<freqBins {
                let binIndex = f * stftFrames + frame
                let r = realSrc[binIndex]
                let im = imagSrc[binIndex]

                dftRealIn[f] = r
                dftImagIn[f] = im

                if f > 0 && f < (fftSize / 2) {
                    let symIndex = fftSize - f
                    dftRealIn[symIndex] = r
                    dftImagIn[symIndex] = -im
                }
            }

            // 2. Inverse DFT
            vDSP_DFT_Execute(
                setup,
                dftRealIn, dftImagIn,
                dftRealOut, dftImagOut
            )

            // 3. Scale by 1.0 / fftSize (Accelerate DFT does not automatically scale inverse)
            var scale = 1.0 / Float(fftSize)
            vDSP_vsmul(dftRealOut, 1, &scale, dftRealOut, 1, vDSP_Length(fftSize))

            // 4. Apply synthesis window
            vDSP_vmul(dftRealOut, 1, hannWindow, 1, dftRealOut, 1, vDSP_Length(fftSize))

            // 5. Overlap-add into accumulated padded buffer
            let destSlice = accumPadded.advanced(by: offset)
            vDSP_vadd(destSlice, 1, dftRealOut, 1, destSlice, 1, vDSP_Length(fftSize))
        }

        // 6. Divide by COLA normalization envelope
        vDSP_vdiv(synthesisColaNorm, 1, accumPadded, 1, accumPadded, 1, vDSP_Length(paddedWindowSamples))

        // 7. Strip padding (discard initial paddingSamples=2048)
        memcpy(reconstructedDest, accumPadded.advanced(by: paddingSamples), windowSamples * MemoryLayout<Float>.size)
        #endif
    }

    // MARK: - SIMD Precision Conversions

    private func convertFloat32ToFloat16(
        source: UnsafePointer<Float>,
        destination: UnsafeMutablePointer<UInt16>,
        elementCount: Int
    ) {
        #if canImport(Accelerate)
        var srcBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: source),
            height: 1,
            width: vImagePixelCount(elementCount),
            rowBytes: elementCount * MemoryLayout<Float>.stride
        )
        var dstBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(destination),
            height: 1,
            width: vImagePixelCount(elementCount),
            rowBytes: elementCount * MemoryLayout<UInt16>.stride
        )
        vImageConvert_PlanarFtoPlanar16F(&srcBuffer, &dstBuffer, 0)
        #else
        for i in 0..<elementCount {
            destination[i] = floatToFloat16Bits(source[i])
        }
        #endif
    }

    private func convertFloat16ToFloat32(
        source: UnsafePointer<UInt16>,
        destination: UnsafeMutablePointer<Float>,
        elementCount: Int
    ) {
        #if canImport(Accelerate)
        var srcBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: source),
            height: 1,
            width: vImagePixelCount(elementCount),
            rowBytes: elementCount * MemoryLayout<UInt16>.stride
        )
        var dstBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(destination),
            height: 1,
            width: vImagePixelCount(elementCount),
            rowBytes: elementCount * MemoryLayout<Float>.stride
        )
        vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
        #else
        for i in 0..<elementCount {
            destination[i] = float16BitsToFloat(source[i])
        }
        #endif
    }

    private func floatToFloat16Bits(_ value: Float) -> UInt16 {
        // Fallback bit conversion
        let u32 = value.bitPattern
        let sign = (u32 >> 16) & 0x8000
        let exp = Int((u32 >> 23) & 0xFF) - 127 + 15
        let mantissa = (u32 >> 13) & 0x03FF
        if exp <= 0 { return UInt16(sign) }
        if exp >= 31 { return UInt16(sign | 0x7C00) }
        return UInt16(sign | UInt32(exp << 10) | mantissa)
    }

    private func float16BitsToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32(h & 0x7C00) >> 10
        let mantissa = UInt32(h & 0x03FF)
        if exp == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }
            // Subnormal
            return Float(bitPattern: sign | ((127 - 14) << 23) | (mantissa << 13))
        } else if exp == 31 {
            return Float(bitPattern: sign | (0xFF << 23) | (mantissa << 13))
        }
        let fExp = (exp - 15 + 127) << 23
        return Float(bitPattern: sign | fExp | (mantissa << 13))
    }
}

