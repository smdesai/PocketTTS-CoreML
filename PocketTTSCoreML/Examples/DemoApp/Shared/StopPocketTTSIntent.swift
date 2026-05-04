//
// StopPocketTTSIntent.swift
//
// App Intent surfaced on the expanded Dynamic Island via a Button in
// the Live Activity view. Conforms to LiveActivityIntent so tapping the
// button runs `perform()` in the widget extension's process WITHOUT
// launching the host app foreground.
//
// Implementation: post a Darwin notification that the host app listens
// for in TTSViewModel. Darwin notifications are cross-process and
// delivered on the listener's run loop, so this is the simplest route
// to signal "user wants to cancel the current run" from the widget.
//
// Kept in Shared/ so both targets compile the same intent type; the
// widget extension references it on the Button, the app references it
// only for the `init()` side-effect (subscribing to the notification)
// — though in practice only the widget needs the type itself.
//

import AppIntents
import Foundation

#if canImport(Darwin)
    import Darwin
#endif

@available(iOS 17.0, *)
public struct StopPocketTTSIntent: LiveActivityIntent {
    public static let title: LocalizedStringResource = "Stop"
    public static let description: IntentDescription? = IntentDescription(
        "Stops the current PocketTTS generation, playback, or cloning run."
    )

    /// No parameters — the host app only supports a single active run at
    /// a time, so "stop" is unambiguous.
    public init() {}

    public func perform() async throws -> some IntentResult {
        // Post a Darwin notification that the host app's TTSViewModel
        // listens for. Darwin notifications cross process boundaries
        // and the app registers an observer at init time.
        let name = PocketTTSLiveActivityNotifications.stopName as CFString
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(),
            CFNotificationName(name),
            nil,
            nil,
            true
        )
        return .result()
    }
}
