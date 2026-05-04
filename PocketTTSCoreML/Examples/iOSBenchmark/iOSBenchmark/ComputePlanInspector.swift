//
// ComputePlanInspector.swift
//
// Wraps the iOS 17.4+ / macOS 14.4+ `MLComputePlan` API to answer:
// "for each op in each of the 6 mlpackages, which compute unit did CoreML
//  actually plan to use?"
//
// Aggregated by op-count per submodel (not FLOP-weighted — plan deliberately
// keeps this dependency-free; a FLOP-weighted view is Phase 5 follow-up).
//
// The API requires a compiled `.mlmodelc` directory. This class:
//   1. looks for a sibling `.mlmodelc` next to the `.mlpackage`; if absent
//   2. uses `MLModel.compileModel(at:)` to produce one in the tmp dir.
//
// On failure (simulator without ANE, CoreML declines to return a plan,
// malformed program), the row still renders with zeros + a note.
//

import CoreML
import Foundation

public struct ComputePlanReport: Codable, Sendable, Identifiable {
    public var id: String { modelName }
    public let modelName: String
    public let opCount: Int
    public let anePct: Double
    public let gpuPct: Double
    public let cpuPct: Double
    /// Short explanation if inspection failed (e.g. simulator, API not
    /// available). Empty string on success.
    public let note: String
}

@available(iOS 17.4, macOS 14.4, *)
public enum ComputePlanInspector {

    /// The six .mlpackage files in the Artifacts bundle.
    public static let submodelNames: [String] = [
        "text_conditioner",
        "flow_lm_prefill",
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
    ]

    public static func inspectAll(
        artifactsBundle: URL, computeUnits: MLComputeUnits
    ) async -> [ComputePlanReport] {
        var results: [ComputePlanReport] = []
        let fm = FileManager.default
        for name in submodelNames {
            // Prefer pre-compiled .mlmodelc (shipped form on iOS), fall back
            // to .mlpackage (dev-time / macOS).
            let mlmodelc = artifactsBundle.appendingPathComponent("\(name).mlmodelc")
            let mlpackage = artifactsBundle.appendingPathComponent("\(name).mlpackage")
            let url: URL
            if fm.fileExists(atPath: mlmodelc.path) {
                url = mlmodelc
            } else if fm.fileExists(atPath: mlpackage.path) {
                url = mlpackage
            } else {
                results.append(
                    .init(
                        modelName: name, opCount: 0, anePct: 0, gpuPct: 0, cpuPct: 0,
                        note: "not found (.mlmodelc or .mlpackage) in bundle"
                    ))
                continue
            }
            let report = await inspect(
                modelName: name, modelURL: url, computeUnits: computeUnits
            )
            results.append(report)
        }
        return results
    }

    public static func inspect(
        modelName: String, modelURL: URL, computeUnits: MLComputeUnits
    ) async -> ComputePlanReport {
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            return .init(
                modelName: modelName, opCount: 0,
                anePct: 0, gpuPct: 0, cpuPct: 0,
                note: "model not found: \(modelURL.lastPathComponent)"
            )
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits

        // Resolve to a compiled .mlmodelc. Prefer sibling that PocketTTS may
        // have already cached; otherwise compile to a temporary location.
        let compiledURL: URL
        do {
            compiledURL = try await resolveCompiled(modelURL: modelURL)
        } catch {
            return .init(
                modelName: modelName, opCount: 0,
                anePct: 0, gpuPct: 0, cpuPct: 0,
                note: "compile failed: \(error.localizedDescription)"
            )
        }

        do {
            let plan = try await MLComputePlan.load(
                contentsOf: compiledURL, configuration: cfg
            )
            return aggregate(modelName: modelName, plan: plan)
        } catch {
            return .init(
                modelName: modelName, opCount: 0,
                anePct: 0, gpuPct: 0, cpuPct: 0,
                note: "MLComputePlan.load failed: \(error.localizedDescription)"
            )
        }
    }

    private static func resolveCompiled(modelURL: URL) async throws -> URL {
        // Already compiled — pass through.
        if modelURL.pathExtension == "mlmodelc" {
            return modelURL
        }
        let sibling =
            modelURL
            .deletingPathExtension()
            .appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: sibling.path) {
            return sibling
        }
        // Fall back to compiling into tmp. Returned URL is valid until
        // app exit, which is fine here.
        return try await MLModel.compileModel(at: modelURL)
    }

    private static func aggregate(
        modelName: String, plan: MLComputePlan
    ) -> ComputePlanReport {
        // MLModelStructure is an enum with cases .program / .pipeline / .neuralNetwork / .unsupported
        guard case .program(let program) = plan.modelStructure,
            let main = program.functions["main"]
        else {
            return .init(
                modelName: modelName, opCount: 0,
                anePct: 0, gpuPct: 0, cpuPct: 0,
                note: "model is not an MIL program"
            )
        }

        var ane = 0
        var gpu = 0
        var cpu = 0
        var other = 0
        walk(
            block: main.block, plan: plan,
            ane: &ane, gpu: &gpu, cpu: &cpu, other: &other)

        let total = ane + gpu + cpu + other
        guard total > 0 else {
            return .init(
                modelName: modelName, opCount: 0,
                anePct: 0, gpuPct: 0, cpuPct: 0,
                note: "no operations enumerated"
            )
        }
        let denom = Double(total)
        return .init(
            modelName: modelName,
            opCount: total,
            anePct: Double(ane) * 100.0 / denom,
            gpuPct: Double(gpu) * 100.0 / denom,
            cpuPct: Double(cpu) * 100.0 / denom,
            note: ""
        )
    }

    private static func walk(
        block: MLModelStructure.Program.Block,
        plan: MLComputePlan,
        ane: inout Int, gpu: inout Int, cpu: inout Int, other: inout Int
    ) {
        for op in block.operations {
            // Actual API (iOS 17.4+): deviceUsage(for:), returns optional DeviceUsage
            // whose .preferred is the non-optional MLComputeDevice. Const ops etc.
            // may return nil because they aren't scheduled.
            let usage = plan.deviceUsage(for: op)
            classify(
                usage?.preferred,
                ane: &ane, gpu: &gpu, cpu: &cpu, other: &other)
            for nested in op.blocks {
                walk(
                    block: nested, plan: plan,
                    ane: &ane, gpu: &gpu, cpu: &cpu, other: &other)
            }
        }
    }

    /// `MLComputeDevice` is an enum with cases .cpu(...) / .gpu(...) / .neuralEngine(...).
    private static func classify(
        _ device: MLComputeDevice?,
        ane: inout Int, gpu: inout Int, cpu: inout Int, other: inout Int
    ) {
        guard let device else {
            other += 1
            return
        }
        switch device {
        case .neuralEngine: ane += 1
        case .gpu: gpu += 1
        case .cpu: cpu += 1
        @unknown default: other += 1
        }
    }
}
