//
// PocketTTSActivityAttributes.swift
//
// Shared ActivityKit attributes + content state used by both the demo
// app (to start/update the Live Activity) and the widget extension (to
// render it). Kept in the Shared/ folder so xcodegen can compile the
// same source file into both targets.
//
// Mode enum covers the three PocketTTS flows we want to surface:
//   streaming — stream() with current/total chunk counter
//   playing   — play() of a pre-generated clip, with elapsed/total seconds
//   cloning   — cloneVoice() pipeline; indeterminate (no reliable progress)
//

import Foundation
import ActivityKit

public struct PocketTTSActivityAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        public enum Mode: String, Codable, Hashable {
            case streaming
            case playing
            case cloning
        }

        public var mode: Mode
        public var voiceName: String
        public var languageName: String
        /// 1-based chunk index currently being produced / played.
        /// Zero when not applicable (playing/cloning paths).
        public var currentChunk: Int
        /// Total chunks in the stream. Zero when not applicable.
        public var totalChunks: Int
        /// Elapsed seconds for the playing mode; zero otherwise.
        public var elapsedSeconds: Double
        /// Total duration in seconds for the playing mode; zero otherwise.
        public var totalSeconds: Double
        /// Mirror of `TTSViewModel.status` — shown verbatim in the Live
        /// Activity so the UI stays consistent with the main app card.
        public var statusLine: String

        public init(
            mode: Mode,
            voiceName: String,
            languageName: String,
            currentChunk: Int = 0,
            totalChunks: Int = 0,
            elapsedSeconds: Double = 0,
            totalSeconds: Double = 0,
            statusLine: String = ""
        ) {
            self.mode = mode
            self.voiceName = voiceName
            self.languageName = languageName
            self.currentChunk = currentChunk
            self.totalChunks = totalChunks
            self.elapsedSeconds = elapsedSeconds
            self.totalSeconds = totalSeconds
            self.statusLine = statusLine
        }

        /// Convenience: progress fraction in [0, 1] for the current mode.
        /// Cloning returns nil (indeterminate).
        public var progressFraction: Double? {
            switch mode {
            case .streaming:
                guard totalChunks > 0 else { return nil }
                return min(1.0, max(0.0, Double(currentChunk) / Double(totalChunks)))
            case .playing:
                guard totalSeconds > 0 else { return nil }
                return min(1.0, max(0.0, elapsedSeconds / totalSeconds))
            case .cloning:
                return nil
            }
        }
    }

    /// Fixed at Activity.request() time.
    public var sessionStartedAt: Date

    public init(sessionStartedAt: Date = Date()) {
        self.sessionStartedAt = sessionStartedAt
    }
}

/// Darwin notification name used by the widget-extension Stop button to
/// signal the host app. Centralised here so both targets agree on the
/// exact string. Kept as a plain String so it's usable from both sides
/// (wrap in `CFString` when posting).
public enum PocketTTSLiveActivityNotifications {
    public static let stopName: String = "com.sdesai.pockettts.demo.stop"
}
