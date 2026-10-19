import Foundation
#if canImport(AVFAudio)
import AVFAudio
#endif
#if canImport(UIKit)
import UIKit
#endif

public protocol BackgroundExecutionDelegate: AnyObject {
    func backgroundExecutionManagerDidRequestPause()
    func backgroundExecutionManagerDidRequestResume()
    func backgroundExecutionManagerDidTransitionToBackground()
    func backgroundExecutionManagerDidTransitionToForeground()
}

/// Coordinates AVAudioSession keep-alive states and background/foreground transitions.
/// Protects the application from iOS background GPU termination (0xdead10cc / MTLCommandBufferErrorNotPermitted).
public final class BackgroundExecutionManager {
    public static let shared = BackgroundExecutionManager()

    public weak var delegate: BackgroundExecutionDelegate?
    public private(set) var isInBackground: Bool = false
    public private(set) var isAudioSessionActive: Bool = false

    private init() {
        registerLifecycleNotifications()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    /// Configures AVAudioSession with .playback category and .default mode to allow continuous
    /// audio playback and background processing without suspension.
    public func configureAudioSession() throws {
        #if canImport(AVFAudio)
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try audioSession.setActive(true)
        isAudioSessionActive = true
        #endif
    }

    public func deactivateAudioSession() {
        #if canImport(AVFAudio)
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            isAudioSessionActive = false
        } catch {
            // Non-fatal deactivation log
        }
        #endif
    }

    // MARK: - Lifecycle Notifications

    private func registerLifecycleNotifications() {
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDidReceiveMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
        #endif

        #if canImport(AVFAudio)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioSessionInterruption),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
        #endif
    }

    #if canImport(UIKit)
    @objc private func handleDidEnterBackground() {
        isInBackground = true
        // CRITICAL iOS Background Rule:
        // Enforce that background inference is strictly routed to Apple Neural Engine and CPU,
        // avoiding background Metal GPU command buffer submissions that cause 0xdead10cc.
        delegate?.backgroundExecutionManagerDidTransitionToBackground()
    }

    @objc private func handleWillEnterForeground() {
        isInBackground = false
        delegate?.backgroundExecutionManagerDidTransitionToForeground()
    }

    @objc private func handleDidReceiveMemoryWarning() {
        // Evaluate memory safety immediately on system warning
        SystemResourceMonitor.shared.evaluateMemorySafety()
    }
    #endif

    #if canImport(AVFAudio)
    @objc private func handleAudioSessionInterruption(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let interruptionType = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }

        switch interruptionType {
        case .began:
            delegate?.backgroundExecutionManagerDidRequestPause()
        case .ended:
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                if options.contains(.shouldResume) {
                    delegate?.backgroundExecutionManagerDidRequestResume()
                }
            }
        @unknown default:
            break
        }
    }
    #endif
}

