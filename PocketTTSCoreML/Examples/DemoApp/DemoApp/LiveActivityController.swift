//
// LiveActivityController.swift
//
// Thin wrapper around the ActivityKit lifecycle for PocketTTS's
// streaming / playing / cloning runs. Kept separate from
// TTSViewModel so the view-model file doesn't balloon with
// ActivityKit / Darwin-notification plumbing that is purely
// presentational.
//
// All methods are safe to call on the main actor. Activity updates
// are awaited asynchronously; failures (e.g. the user disabled Live
// Activities in Settings) are swallowed because the main app flow
// must continue regardless.
//
// The Stop Darwin notification observer is registered here via
// `observeStop(_:)`. The observer uses a raw Unmanaged pointer to
// the target callable — standard pattern for CFNotificationCenter.
//

import Foundation
import ActivityKit

@MainActor
final class LiveActivityController {
    private var current: Activity<PocketTTSActivityAttributes>? = nil

    /// Live Activities are a best-effort UX enhancement: if authorisation
    /// is off or iOS refuses the request, we silently skip.
    private var authorised: Bool {
        ActivityAuthorizationInfo().areActivitiesEnabled
    }

    /// Start a new Live Activity for `mode`. Ends any previous activity
    /// first so only one is ever on-screen.
    func start(
        mode: PocketTTSActivityAttributes.ContentState.Mode,
        voiceName: String,
        languageName: String,
        totalChunks: Int = 0,
        totalSeconds: Double = 0,
        statusLine: String = ""
    ) {
        guard authorised else { return }
        // End any leftover activity before requesting a new one.
        if current != nil {
            Task { await self.end() }
        }
        let attrs = PocketTTSActivityAttributes(sessionStartedAt: Date())
        let state = PocketTTSActivityAttributes.ContentState(
            mode: mode,
            voiceName: voiceName,
            languageName: languageName,
            currentChunk: 0,
            totalChunks: totalChunks,
            elapsedSeconds: 0,
            totalSeconds: totalSeconds,
            statusLine: statusLine
        )
        do {
            let content = ActivityContent(state: state, staleDate: nil)
            current = try Activity.request(
                attributes: attrs,
                content: content,
                pushType: nil
            )
        } catch {
            // Non-fatal: no activity means the Lock Screen / DI is
            // empty, app still works.
            current = nil
        }
    }

    /// Update the current activity's state. No-op if none is running.
    /// `Activity` / `ActivityContent` are not Sendable (ActivityKit
    /// predates strict concurrency) but are actually thread-safe in
    /// practice. Use UnsafeActivityBox to carry the reference through
    /// the await hop without tripping Swift 6's Sendable checks.
    func update(_ state: PocketTTSActivityAttributes.ContentState) {
        guard let activity = current else { return }
        let box = UnsafeActivityBox(activity)
        let content = UnsafeActivityContentBox(
            ActivityContent(state: state, staleDate: nil)
        )
        Task.detached {
            await box.value.update(content.value)
        }
    }

    /// End and dismiss the current activity immediately.
    func end() async {
        guard let activity = current else { return }
        current = nil
        let box = UnsafeActivityBox(activity)
        await box.value.end(nil, dismissalPolicy: .immediate)
    }

    /// Whether a Live Activity is currently on screen.
    var isActive: Bool { current != nil }
}

// MARK: - Darwin notification bridge

/// Wrap the Stop-Darwin-notification lifecycle. The widget extension
/// (StopPocketTTSIntent) posts a Darwin notification named
/// `PocketTTSLiveActivityNotifications.stopName`; this listener
/// invokes the supplied handler on the main actor when received.
///
/// Usage: construct once at app startup, retain it on the view model,
/// pass a closure that calls `TTSViewModel.stop()`. `deinit` removes
/// the observer so this is safe even if reconstructed (tests, etc).
@MainActor
final class StopNotificationListener {
    private let handler: @MainActor () -> Void
    // `nonisolated(unsafe)` because deinit must read it to unregister
    // the CFNotification observer, and deinit cannot be MainActor-
    // isolated. The token is set exactly once in `start()` (on the
    // main actor) and never mutated again, so single-writer-before-
    // reads ordering makes the unsafe access safe.
    private nonisolated(unsafe) var token: UnsafeMutableRawPointer? = nil

    init(handler: @escaping @MainActor () -> Void) {
        self.handler = handler
    }

    func start() {
        guard token == nil else { return }
        let ptr = Unmanaged.passRetained(self).toOpaque()
        token = ptr
        let name = PocketTTSLiveActivityNotifications.stopName as CFString
        CFNotificationCenterAddObserver(
            CFNotificationCenterGetDarwinNotifyCenter(),
            ptr,
            { _, observer, _, _, _ in
                guard let observer = observer else { return }
                let me = Unmanaged<StopNotificationListener>
                    .fromOpaque(observer)
                    .takeUnretainedValue()
                // Darwin observer callback is invoked on the thread
                // that processed the notification; hop to main.
                DispatchQueue.main.async {
                    me.handler()
                }
            },
            name,
            nil,
            .deliverImmediately
        )
    }

    deinit {
        if let token = token {
            CFNotificationCenterRemoveEveryObserver(
                CFNotificationCenterGetDarwinNotifyCenter(),
                token
            )
            Unmanaged<StopNotificationListener>.fromOpaque(token).release()
        }
    }
}

// MARK: - Unchecked-Sendable boxes for ActivityKit types
//
// `Activity<>` and `ActivityContent<>` aren't declared Sendable by
// ActivityKit (pre-Swift-6 framework) but are thread-safe in
// practice — all their mutation goes through the system's Live
// Activity daemon, not the struct state. These boxes let us pass
// them across a Task boundary without Swift 6 complaining.

@available(iOS 17.0, *)
private struct UnsafeActivityBox: @unchecked Sendable {
    let value: Activity<PocketTTSActivityAttributes>
    init(_ v: Activity<PocketTTSActivityAttributes>) { self.value = v }
}

@available(iOS 17.0, *)
private struct UnsafeActivityContentBox: @unchecked Sendable {
    let value: ActivityContent<PocketTTSActivityAttributes.ContentState>
    init(_ v: ActivityContent<PocketTTSActivityAttributes.ContentState>) { self.value = v }
}
