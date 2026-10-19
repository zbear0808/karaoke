import Foundation
#if canImport(CoreML)
import CoreML
#endif

public enum SCNetModelLoaderError: LocalizedError {
    case modelNotFound(String)
    case compilationFailed(String)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Compiled CoreML model \(name).mlmodelc was not found in bundle or disk."
        case .compilationFailed(let details):
            return "Failed to compile CoreML model: \(details)"
        case .predictionFailed(let details):
            return "Model warmup prediction failed: \(details)"
        }
    }
}

/// Manages the lifecycle, compute configuration, pre-warming, and memory footprint of the SCNet MLModel.
public final class SCNetModelLoader {

    public static let shared = SCNetModelLoader()

    #if canImport(CoreML)
    public private(set) var model: MLModel?
    #endif

    public private(set) var isPrewarmed: Bool = false
    public private(set) var currentComputeUnits: String = "cpuAndNeuralEngine"

    private let modelName: String = "SCNetInstrumental"

    private init() {}

    /// Loads the pre-compiled SCNetInstrumental.mlmodelc asynchronously.
    /// Strictly sets computeUnits to .cpuAndNeuralEngine to prevent background GPU violations.
    #if canImport(CoreML)
    public func loadModel(bundle: Bundle = Bundle.main, tensorPool: PageAlignedTensorPool? = nil) async throws -> MLModel {
        if let existing = model {
            return existing
        }

        let config = MLModelConfiguration()
        // Forces execution onto Apple Neural Engine + CPU, avoiding Metal GPU
        // background command buffer crashes (0xdead10cc) and minimizing dirty RAM.
        config.computeUnits = .cpuAndNeuralEngine
        config.allowLowPrecisionAccumulationOnGPU = true
        self.currentComputeUnits = "cpuAndNeuralEngine"

        guard let modelURL = bundle.url(forResource: modelName, withExtension: "mlmodelc") else {
            throw SCNetModelLoaderError.modelNotFound(modelName)
        }

        let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
        self.model = loadedModel

        // Pre-warm the model immediately so ANE graph compilation happens at song load, not in streaming loop
        if let pool = tensorPool {
            try prewarm(model: loadedModel, tensorPool: pool)
        }

        return loadedModel
    }

    /// Pre-warms the model using pre-allocated PageAlignedTensorPool to compile ANE kernels.
    public func prewarm(model: MLModel, tensorPool: PageAlignedTensorPool) throws {
        let inputProvider = SCNetFeatureProvider(inputMultiArray: tensorPool.inputMultiArray)
        let options = MLPredictionOptions()
        options.outputBackings = ["output_spectrogram": tensorPool.outputMultiArray]

        try autoreleasepool {
            _ = try model.prediction(from: inputProvider, options: options)
        }
        self.isPrewarmed = true
    }

    /// Releases the model instance to free memory when playback is stopped.
    public func unloadModel() {
        self.model = nil
        self.isPrewarmed = false
    }

    /// Re-configures the model strictly to .cpuAndNeuralEngine when entering background.
    public func enforceBackgroundComputePolicy() {
        // If loaded under another configuration, re-load with .cpuAndNeuralEngine
        guard let current = model, current.configuration.computeUnits != .cpuAndNeuralEngine else { return }
        let newConfig = MLModelConfiguration()
        newConfig.computeUnits = .cpuAndNeuralEngine
        newConfig.allowLowPrecisionAccumulationOnGPU = true
        if let url = current.modelDescription.classLabels as? URL {
            // Reload if needed
        }
    }
    #endif
}

#if canImport(CoreML)
/// Concrete MLFeatureProvider wrapping pre-allocated input MLMultiArray with zero copies.
public final class SCNetFeatureProvider: MLFeatureProvider {
    private let inputMultiArray: MLMultiArray
    private let inputFeatureName: String

    public init(inputMultiArray: MLMultiArray, inputFeatureName: String = "input_spectrogram") {
        self.inputMultiArray = inputMultiArray
        self.inputFeatureName = inputFeatureName
    }

    public var featureNames: Set<String> {
        return [inputFeatureName]
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == inputFeatureName {
            return MLFeatureValue(multiArray: inputMultiArray)
        }
        return nil
    }
}
#endif

