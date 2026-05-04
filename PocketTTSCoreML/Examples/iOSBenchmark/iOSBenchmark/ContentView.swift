//
// ContentView.swift
//
// Single-screen SwiftUI front-end over BenchmarkRunner. Shows:
//  - live thermal state
//  - Run button
//  - per-mode results table (cold / warm RTF, ANE % per submodel, verdict)
//  - Share Report button (JSON + markdown)
//

import CoreML
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var runner: BenchmarkRunner = {
        do {
            return try BenchmarkRunner()
        } catch {
            // Construct an empty runner with errorMessage pre-populated so
            // the UI renders the message instead of crashing at launch.
            return BenchmarkRunner.failed(with: error.localizedDescription)
        }
    }()

    @State private var thermalState: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState
    @State private var shareItems: [Any] = []
    @State private var isSharing: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    headerSection
                    runSection
                    if let err = runner.errorMessage {
                        errorView(err)
                    }
                    if !runner.rows.isEmpty {
                        resultsSection
                    }
                    if runner.lastReport != nil {
                        shareSection
                    }
                    aboutSection
                }
                .padding()
            }
            .navigationTitle("PocketTTS RTF")
            .onReceive(
                NotificationCenter.default.publisher(
                    for: ProcessInfo.thermalStateDidChangeNotification
                )
            ) { _ in
                thermalState = ProcessInfo.processInfo.thermalState
            }
            .sheet(isPresented: $isSharing) {
                ShareSheet(items: shareItems)
            }
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Thermal state: \(Self.thermalDescription(thermalState))")
                .font(.subheadline)
                .foregroundStyle(thermalState == .nominal ? Color.primary : Color.orange)
            if thermalState != .nominal {
                Text("Cool device to .nominal before running for the ship-gate number.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            Text("Prompt: \"\(BenchmarkRunner.canonicalPrompt)\"")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var runSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                Task { await runner.run() }
            } label: {
                if runner.isRunning {
                    ProgressView()
                } else {
                    Label("Run benchmark", systemImage: "play.fill")
                        .font(.headline)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(runner.isRunning)

            if runner.isRunning {
                Text(runner.status).font(.caption).foregroundStyle(.secondary)
            } else if !runner.status.isEmpty {
                Text(runner.status).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Results").font(.headline)
            ForEach(runner.rows) { row in
                rowCard(row)
            }
        }
    }

    private func rowCard(_ row: BenchmarkRow) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(row.mode).font(.headline)
                Spacer()
                verdictBadge(row.verdict)
                if row.throttled {
                    Text("throttled").font(.caption)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.red.opacity(0.15)).clipShape(Capsule())
                }
            }
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow {
                    Text("audio").foregroundStyle(.secondary)
                    Text(String(format: "%.3f s", row.audioSeconds))
                    Text("peak thermal").foregroundStyle(.secondary)
                    Text(row.peakThermalState)
                }
                GridRow {
                    Text("cold").foregroundStyle(.secondary)
                    Text(String(format: "%.3f s", row.coldWallSeconds))
                    Text("cold RTF").foregroundStyle(.secondary)
                    Text(String(format: "%.3f", row.coldRTF))
                }
                GridRow {
                    Text("warm").foregroundStyle(.secondary)
                    Text(String(format: "%.3f s", row.warmWallSecondsMedian))
                    Text("warm RTF").foregroundStyle(.secondary)
                    Text(String(format: "%.3f", row.warmRTF))
                        .fontWeight(.semibold)
                }
            }.font(.subheadline.monospacedDigit())
            if !row.computePlan.isEmpty {
                Divider()
                Text("MLComputePlan (ops by device)").font(.caption).foregroundStyle(.secondary)
                ForEach(row.computePlan) { p in
                    HStack {
                        Text(p.modelName).font(.caption.monospaced())
                        Spacer()
                        Text("ops \(p.opCount)").font(.caption)
                        Text(String(format: "ANE %.0f%%", p.anePct))
                            .font(.caption.monospaced())
                            .foregroundStyle(
                                p.anePct >= 80 ? .green : (p.anePct >= 50 ? .orange : .red))
                        Text(String(format: "GPU %.0f%%", p.gpuPct))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                        Text(String(format: "CPU %.0f%%", p.cpuPct))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    if !p.note.isEmpty {
                        Text(p.note).font(.caption2).foregroundStyle(.red)
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func verdictBadge(_ verdict: String) -> some View {
        let color: Color
        switch verdict {
        case "GREEN": color = .green
        case "YELLOW": color = .orange
        default: color = .red
        }
        return Text(verdict)
            .font(.caption.weight(.bold))
            .padding(.horizontal, 8).padding(.vertical, 2)
            .background(color.opacity(0.2))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }

    private var shareSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                shareReport()
            } label: {
                Label("Share Report (JSON + markdown)", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
        }
    }

    private var aboutSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("About").font(.headline)
            Text(
                "Runs the PocketTTS CoreML pipeline with three MLComputeUnits configurations. Each row is cold + 3 warm runs; we report the median warm RTF. GREEN gate: warm RTF ≤ 0.5 and flow_lm_main ≥ 80% ANE."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.top, 24)
    }

    private func errorView(_ msg: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Error").font(.headline).foregroundStyle(.red)
            Text(msg).font(.caption.monospaced())
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Actions

    private func shareReport() {
        guard let report = runner.lastReport else { return }
        do {
            let jsonData = try report.jsonData()
            let md = report.markdownSummary()
            let tmp = FileManager.default.temporaryDirectory
            let ts = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            let jsonURL = tmp.appendingPathComponent("pockettts_rtf_\(ts).json")
            let mdURL = tmp.appendingPathComponent("pockettts_rtf_\(ts).md")
            try jsonData.write(to: jsonURL, options: .atomic)
            try md.data(using: .utf8)?.write(to: mdURL, options: .atomic)
            shareItems = [jsonURL, mdURL]
            isSharing = true
        } catch {
            runner.setError(error.localizedDescription)
        }
    }

    // MARK: - Utility

    private static func thermalDescription(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

// MARK: - Share Sheet wrapper

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

// MARK: - Runner error helper

extension BenchmarkRunner {
    /// UI-facing setter. Routes through the main actor because
    /// `errorMessage` drives `@Published` updates.
    public func setError(_ message: String) {
        Task { @MainActor in
            self.errorMessage = message
        }
    }
}
