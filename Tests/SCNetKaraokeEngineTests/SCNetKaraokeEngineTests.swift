import XCTest
@testable import SCNetKaraokeEngine

final class SCNetKaraokeEngineTests: XCTestCase {

    func testEngineConfigConstants() {
        XCTAssertEqual(SCNetEngineConfig.sampleRate, 44100.0)
        XCTAssertEqual(SCNetEngineConfig.fftSize, 4096)
        XCTAssertEqual(SCNetEngineConfig.hopLength, 1024)
        XCTAssertEqual(SCNetEngineConfig.freqBins, 2049)
        XCTAssertEqual(SCNetEngineConfig.stftFrames, 130)
        XCTAssertEqual(SCNetEngineConfig.windowSamples, 132300)
        XCTAssertEqual(SCNetEngineConfig.stepSamples, 66150)
        XCTAssertEqual(SCNetEngineConfig.paddedWindowSamples, 133120)
        XCTAssertEqual(SCNetEngineConfig.tensorShape, [1, 4, 2049, 130])
    }

    func testPageAlignedTensorPoolAllocation() {
        let pool = PageAlignedTensorPool()
        XCTAssertNotNil(pool.inputFloat16Pointer)
        XCTAssertNotNil(pool.outputFloat16Pointer)
        XCTAssertNotNil(pool.scratchFloat32Pointer)
        pool.zeroOut()
    }

    func testLockFreeRingBufferReadWrite() {
        let capacity = 1000
        let ringBuffer = LockFreePCMBuffer(capacityFrames: capacity, channelCount: 2)

        XCTAssertEqual(ringBuffer.availableFramesToRead, 0)
        XCTAssertEqual(ringBuffer.availableFramesToWrite, capacity - 1)

        var testData: [Float] = Array(repeating: 0.5, count: 200 * 2)
        let written = ringBuffer.write(from: testData, frameCount: 200)
        XCTAssertEqual(written, 200)
        XCTAssertEqual(ringBuffer.availableFramesToRead, 200)

        var readData: [Float] = Array(repeating: 0.0, count: 100 * 2)
        let readCount = ringBuffer.read(into: &readData, frameCount: 100)
        XCTAssertEqual(readCount, 100)
        XCTAssertEqual(ringBuffer.availableFramesToRead, 100)
        XCTAssertEqual(readData[0], 0.5)
    }

    func testOverlapAddAccumulatorReconstruction() {
        let step = 100
        let window = 200
        let accumulator = OverlapAddAccumulator(stepSamples: step, windowSamples: window, channelCount: 2)

        let chunk1L = [Float](repeating: 1.0, count: window)
        let chunk1R = [Float](repeating: 1.0, count: window)

        accumulator.processChunk(left3sChunk: chunk1L, right3sChunk: chunk1R)

        let chunk2L = [Float](repeating: 1.0, count: window)
        let chunk2R = [Float](repeating: 1.0, count: window)

        accumulator.processChunk(left3sChunk: chunk2L, right3sChunk: chunk2R)

        // For constant input 1.0, linear fade-in + fade-out tail should equal 1.0
        for i in 0..<step {
            XCTAssertEqual(accumulator.finalizedOutputL[i], 1.0, accuracy: 1e-4)
            XCTAssertEqual(accumulator.finalizedOutputR[i], 1.0, accuracy: 1e-4)
        }
    }
}

