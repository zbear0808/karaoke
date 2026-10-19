import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Monitors real-time available memory headroom before iOS Jetsam kill
/// and observes system thermal states to pace inference duty cycles.
public final class SystemResourceMonitor {
    public static let shared = SystemResourceMonitor()

    public private(set) var isEmergencyMemoryThrottled: Bool = false
    public private(set) var isThermalThrottled: Bool = false

    private init() {
        setupThermalObservation()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    /// Queries remaining physical dirty memory headroom before iOS Jetsam triggers an OOM kill.
    /// Uses os_proc_available_memory() on Darwin/iOS, or simulated fallback on other platforms.
    public func getAvailableMemoryBytes() -> UInt64 {
        #if canImport(Darwin)
        return os_proc_available_memory()
        #else
        // Fallback for non-Darwin environments / testing
        var memoryFootprint: UInt64 = 0
        return 512 * 1024 * 1024 // 512 MB mock
        #endif
    }

    /// Inspects system memory headroom and updates throttling state.
    /// Returns true if memory is safely above the 40 MB emergency watermark.
    @discardableResult
    public func evaluateMemorySafety() -> Bool {
        let availableBytes = getAvailableMemoryBytes()
        if availableBytes < SCNetEngineConfig.emergencyMemoryWatermarkBytes {
            isEmergencyMemoryThrottled = true
            return false
        } else {
            // Add a 20MB hysteresis buffer before clearing emergency status
            if availableBytes > SCNetEngineConfig.emergencyMemoryWatermarkBytes + (20 * 1024 * 1024) {
                isEmergencyMemoryThrottled = false
            }
            return true
        }
    }

    /// Returns the target lookahead frame limit based on current thermal and memory pressure.
    /// Drops from 6.0s down to 3.0s under pressure to prevent thermal runaway and RAM spikes.
    public var currentLookaheadLimitFrames: Int {
        if isEmergencyMemoryThrottled || isThermalThrottled {
            return SCNetEngineConfig.throttledLookaheadFrames
        }
        return SCNetEngineConfig.normalLookaheadFrames
    }

    // MARK: - Thermal State Observation

    private func setupThermalObservation() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleThermalStateChange),
            name: ProcessInfo.thermalStateDidChangeNotification,
            object: nil
        )
        evaluateThermalState(ProcessInfo.processInfo.thermalState)
    }

    @objc private func handleThermalStateChange(_ notification: Notification) {
        evaluateThermalState(ProcessInfo.processInfo.thermalState)
    }

    private func evaluateThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .serious, .critical:
            isThermalThrottled = true
        case .nominal, .fair:
            isThermalThrottled = false
        @unknown default:
            isThermalThrottled = false
        }
    }
}

