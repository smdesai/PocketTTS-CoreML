//
// PocketTTSLiveActivity.swift
//
// Widget that renders the Lock Screen and Dynamic Island presentations
// for PocketTTS's three run modes (streaming / playing / cloning).
//
// Structure:
//   - LockScreenView: full Lock Screen banner. Scaled-down version of
//     the main app's StatusCard — tint by mode, icon, title + subtitle,
//     progress bar (or indeterminate spinner for cloning), Stop button.
//   - DI compact leading: waveform icon + first letter of voice name.
//   - DI compact trailing: chunk counter ("1/3"), elapsed seconds ("2.1s"),
//     or a pulsing dot for cloning.
//   - DI minimal: waveform icon only.
//   - DI expanded: full status + progress + Stop button.
//
// The Stop button uses StopPocketTTSIntent (LiveActivityIntent) which
// posts a Darwin notification picked up by TTSViewModel in the host app.
//

import ActivityKit
import AppIntents
import SwiftUI
import WidgetKit

@available(iOS 17.0, *)
struct PocketTTSLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: PocketTTSActivityAttributes.self) { context in
            LockScreenView(state: context.state)
                .activityBackgroundTint(Color.black.opacity(0.85))
                .activitySystemActionForegroundColor(.white)
        } dynamicIsland: { context in
            let state = context.state

            return DynamicIsland {
                // EXPANDED REGIONS
                DynamicIslandExpandedRegion(.leading) {
                    HStack(spacing: 8) {
                        ModeIcon(mode: state.mode)
                            .frame(width: 26, height: 26)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(state.voiceName)
                                .font(.caption.bold())
                                .lineLimit(1)
                            Text(state.languageName)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .padding(.leading, 4)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    TrailingBadge(state: state)
                        .padding(.trailing, 4)
                }
                DynamicIslandExpandedRegion(.center) {
                    // Intentionally empty: center of the expanded DI
                    // tends to be visually crowded. We let leading +
                    // trailing carry the top row.
                    EmptyView()
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(spacing: 8) {
                        Text(
                            state.statusLine.isEmpty
                                ? defaultStatus(for: state.mode)
                                : state.statusLine
                        )
                        .font(.footnote)
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .leading)

                        ProgressBar(state: state, tint: modeTint(for: state.mode))

                        HStack {
                            Spacer()
                            Button(intent: StopPocketTTSIntent()) {
                                Label("Stop", systemImage: "stop.fill")
                                    .labelStyle(.titleAndIcon)
                                    .font(.footnote.bold())
                            }
                            .tint(.red)
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }
                    }
                    .padding(.horizontal, 4)
                }
            } compactLeading: {
                HStack(spacing: 3) {
                    ModeIcon(mode: state.mode)
                        .frame(width: 14, height: 14)
                    Text(firstInitial(state.voiceName))
                        .font(.caption2.bold())
                }
                .foregroundStyle(modeTint(for: state.mode))
            } compactTrailing: {
                CompactTrailingLabel(state: state)
                    .foregroundStyle(modeTint(for: state.mode))
            } minimal: {
                ModeIcon(mode: state.mode)
                    .foregroundStyle(modeTint(for: state.mode))
            }
            .keylineTint(modeTint(for: state.mode))
        }
    }
}

// MARK: - Lock Screen

@available(iOS 17.0, *)
private struct LockScreenView: View {
    let state: PocketTTSActivityAttributes.ContentState

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(modeTint(for: state.mode).opacity(0.18))
                ModeIcon(mode: state.mode)
                    .foregroundStyle(modeTint(for: state.mode))
                    .frame(width: 22, height: 22)
            }
            .frame(width: 40, height: 40)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(title(for: state.mode))
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                    Spacer()
                    TrailingBadge(state: state)
                }

                Text("\(state.voiceName) • \(state.languageName)")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                    .lineLimit(1)

                if !state.statusLine.isEmpty {
                    Text(state.statusLine)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.85))
                        .lineLimit(1)
                }

                ProgressBar(state: state, tint: modeTint(for: state.mode))
                    .padding(.top, 2)

                HStack {
                    Spacer()
                    Button(intent: StopPocketTTSIntent()) {
                        Label("Stop", systemImage: "stop.fill")
                            .labelStyle(.titleAndIcon)
                            .font(.caption.bold())
                    }
                    .tint(.red)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
                .padding(.top, 2)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }
}

// MARK: - Shared pieces

@available(iOS 17.0, *)
private struct ProgressBar: View {
    let state: PocketTTSActivityAttributes.ContentState
    let tint: Color

    var body: some View {
        if let fraction = state.progressFraction {
            ProgressView(value: fraction)
                .progressViewStyle(.linear)
                .tint(tint)
        } else {
            // Indeterminate (cloning). iOS renders a thin animated
            // barberpole when value is nil on iOS 17+.
            ProgressView()
                .progressViewStyle(.linear)
                .tint(tint)
        }
    }
}

@available(iOS 17.0, *)
private struct ModeIcon: View {
    let mode: PocketTTSActivityAttributes.ContentState.Mode

    var body: some View {
        Image(systemName: systemName)
            .resizable()
            .scaledToFit()
    }

    private var systemName: String {
        switch mode {
        case .streaming: return "waveform"
        case .playing: return "play.circle.fill"
        case .cloning: return "person.wave.2.fill"
        }
    }
}

@available(iOS 17.0, *)
private struct CompactTrailingLabel: View {
    let state: PocketTTSActivityAttributes.ContentState

    var body: some View {
        switch state.mode {
        case .streaming:
            if state.totalChunks > 0 {
                Text("\(state.currentChunk)/\(state.totalChunks)")
                    .font(.caption2.monospacedDigit().bold())
            } else {
                Text("…").font(.caption2.bold())
            }
        case .playing:
            if state.totalSeconds > 0 {
                Text(String(format: "%.1fs", max(0, state.totalSeconds - state.elapsedSeconds)))
                    .font(.caption2.monospacedDigit().bold())
            } else {
                Text("▶︎").font(.caption2.bold())
            }
        case .cloning:
            PulsingDot()
        }
    }
}

/// Same content as CompactTrailingLabel but sized for the larger
/// expanded / lock-screen trailing slot.
@available(iOS 17.0, *)
private struct TrailingBadge: View {
    let state: PocketTTSActivityAttributes.ContentState

    var body: some View {
        Group {
            switch state.mode {
            case .streaming:
                if state.totalChunks > 0 {
                    Text("\(state.currentChunk)/\(state.totalChunks)")
                        .font(.caption.monospacedDigit().bold())
                } else {
                    Text("…").font(.caption.bold())
                }
            case .playing:
                if state.totalSeconds > 0 {
                    Text(
                        String(
                            format: "%.1fs",
                            max(0, state.totalSeconds - state.elapsedSeconds))
                    )
                    .font(.caption.monospacedDigit().bold())
                } else {
                    Text("▶︎").font(.caption.bold())
                }
            case .cloning:
                HStack(spacing: 4) {
                    PulsingDot()
                    Text("Cloning").font(.caption.bold())
                }
            }
        }
        .foregroundStyle(modeTint(for: state.mode))
    }
}

@available(iOS 17.0, *)
private struct PulsingDot: View {
    @State private var pulse = false
    var body: some View {
        Circle()
            .frame(width: 8, height: 8)
            .opacity(pulse ? 0.25 : 1.0)
            .animation(
                .easeInOut(duration: 0.7).repeatForever(autoreverses: true),
                value: pulse
            )
            .onAppear { pulse = true }
    }
}

// MARK: - Helpers

@available(iOS 17.0, *)
private func modeTint(for mode: PocketTTSActivityAttributes.ContentState.Mode) -> Color {
    switch mode {
    case .streaming: return .orange
    case .playing: return .blue
    case .cloning: return .gray
    }
}

@available(iOS 17.0, *)
private func title(for mode: PocketTTSActivityAttributes.ContentState.Mode) -> String {
    switch mode {
    case .streaming: return "PocketTTS — Streaming"
    case .playing: return "PocketTTS — Playing"
    case .cloning: return "PocketTTS — Cloning voice"
    }
}

@available(iOS 17.0, *)
private func defaultStatus(for mode: PocketTTSActivityAttributes.ContentState.Mode) -> String {
    switch mode {
    case .streaming: return "Streaming…"
    case .playing: return "Playing…"
    case .cloning: return "Cloning voice…"
    }
}

private func firstInitial(_ s: String) -> String {
    guard let ch = s.first else { return "•" }
    return String(ch).uppercased()
}
