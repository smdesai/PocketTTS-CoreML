import Foundation
import CoreML
import Accelerate

/// Drives the CoreML AR loop for a single utterance.
///
/// This mirrors the control flow in `pockettts_coreml.e2e.generator`'s
/// `CoreMLGenerator.generate`, but expects the caller to provide a
/// `VoiceHandle(.prefilled)` — i.e. voice+text prefill already baked in.
///
/// Why no in-Swift prefill? The currently-shipped `flow_lm_main.mlpackage`
/// takes only an AR step signature (`sequence=[1,1,32]`, no
/// `text_embeddings` input). Running step-by-step prefill through it is
/// not possible because the traced graph does not concat text embeddings
/// along the time axis. Until a dedicated prefill `.mlpackage` is added
/// (or a Swift port of the 6-layer transformer is written), the prefill
/// runs in Python — see `tools/export_full_prefill.py`.
public final class Orchestrator: @unchecked Sendable {
    public struct Models {
        public let flowMain: MLModel
        public let flowFlow: MLModel
        public let mimiDecoder: MLModel
        public let mimiLayout: MimiStateLayout
    }

    public let models: Models
    public let flowRope: RoPECache
    public let mimiRope: RoPECache

    public init(models: Models) {
        self.models = models
        self.flowRope = RoPECache(
            maxContext: PocketTTSArch.flowSCap,
            headDim: PocketTTSArch.flowHeadDim
        )
        self.mimiRope = RoPECache(
            maxContext: PocketTTSArch.mimiSCap,
            headDim: PocketTTSArch.mimiHeadDim
        )
    }

    public struct Frame {
        public let audio: MLMultiArray  // shape [1, 1, 1920]
        public let stepIndex: Int
        public let eosLogit: Float
    }

    /// Run the AR loop for one prefilled voice and stream audio frames.
    ///
    /// The returned async stream yields PCM16 little-endian frames of 1920
    /// samples each. Terminates either when max_gen_len is exhausted or
    /// `framesAfterEos` frames past the first EOS trigger.
    public func generate(
        voice: VoiceHandle,
        options: PocketTTS.GenerateOptions,
        maxGenLen: Int = 512
    ) -> AsyncThrowingStream<Data, Error> {
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task { [self] in
            do {
                try self.runLoop(
                    voice: voice, options: options,
                    maxGenLen: maxGenLen, continuation: continuation
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    private func runLoop(
        voice: VoiceHandle,
        options: PocketTTS.GenerateOptions,
        maxGenLen: Int,
        continuation: AsyncThrowingStream<Data, Error>.Continuation
    ) throws {
        guard case let .prefilled(kvBox, initialOffset, bosEmb, _, noiseSeq) = voice.kind else {
            throw OrchestratorError.prefillRequired
        }

        // Preallocate persistent buffers. flow_lm_* are fp16 end-to-end.
        let kvCache = kvBox.array  // rank-5 [12, 1, 256, 16, 64] fp16

        let sequence = try MLMultiArray(
            shape: [1, 1, NSNumber(value: PocketTTSArch.latentDim)], dataType: .float16
        )
        // Step 0 input = bos_emb (fp32 → fp16).
        writeFp32ToFp16MA(bosEmb, dst: sequence, count: PocketTTSArch.latentDim)

        // Preallocate flow step RoPE buffers (fp32 table → fp16 step slice).
        let flowRopeCosFp32 = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: PocketTTSArch.flowHalfDim)], dataType: .float32
        )
        let flowRopeSinFp32 = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: PocketTTSArch.flowHalfDim)], dataType: .float32
        )
        let flowRopeCos = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: PocketTTSArch.flowHalfDim)], dataType: .float16
        )
        let flowRopeSin = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: PocketTTSArch.flowHalfDim)], dataType: .float16
        )

        // Flow flow inputs (fp16).
        let flowSIn = try MLMultiArray(shape: [1, 1], dataType: .float16)
        let flowTIn = try MLMultiArray(shape: [1, 1], dataType: .float16)
        let flowNoiseIn = try MLMultiArray(
            shape: [1, NSNumber(value: PocketTTSArch.latentDim)], dataType: .float16
        )
        writeFp32ToFp16MA([0], dst: flowSIn, count: 1)
        writeFp32ToFp16MA([1], dst: flowTIn, count: 1)

        // Mimi state. `mimi_decoder.mlpackage` is fp32 end-to-end
        // (FP32 compute precision was chosen at conversion time to cover
        // the CPU-fallback ConvTranspose1d ops — see docs/phase2_3_notes.md).
        let mimiState = try MimiStateBuffer(layout: models.mimiLayout)
        let T = PocketTTSArch.mimiTStep
        let mimiRopeCos = try MLMultiArray(
            shape: [1, NSNumber(value: T), 1, NSNumber(value: PocketTTSArch.mimiHalfDim)],
            dataType: .float32
        )
        let mimiRopeSin = try MLMultiArray(
            shape: [1, NSNumber(value: T), 1, NSNumber(value: PocketTTSArch.mimiHalfDim)],
            dataType: .float32
        )

        var currentOffset = initialOffset
        var mimiOffset = 0
        var eosStep: Int? = nil

        let noiseSrc = NoiseSource(seed: options.seed ?? UInt64(bitPattern: Int64(clamping: 42)),
                                   std: sqrt(options.temperature))
        if let seq = noiseSeq { noiseSrc.setPrecomputed(seq) }

        // Intermediate buffers swapped into dictionaries per-step to avoid
        // per-step MLMultiArray allocation where possible. We still create
        // fresh attn/offset masks per step because those are position-dependent.
        let flowMainFeatures = try MLDictionaryFeatureProvider(dictionary: [:])
        _ = flowMainFeatures

        var stepLatent = [Float](repeating: 0, count: PocketTTSArch.latentDim)

        for gen in 0..<maxGenLen {
            try autoreleasepool {
                // ---- FLOW LM MAIN STEP (fp16 I/O) ----
                let offsetMask = try Masks.oneHotOffsetMaskFp16(
                    offset: currentOffset, sCapacity: PocketTTSArch.flowSCap
                )
                let attnMask = try Masks.additiveAttentionMaskStepFp16(
                    offset: currentOffset, sCapacity: PocketTTSArch.flowSCap
                )
                // Slice fp32 table then downcast to fp16.
                flowRope.fillStep(offset: currentOffset,
                                  cosOut: flowRopeCosFp32, sinOut: flowRopeSinFp32)
                let halfDim = PocketTTSArch.flowHalfDim
                Float16Ops.convertFp32ToFp16(
                    srcPtr: flowRopeCosFp32.dataPointer.bindMemory(to: Float.self, capacity: halfDim),
                    dstPtr: flowRopeCos.dataPointer.bindMemory(to: UInt16.self, capacity: halfDim),
                    count: halfDim
                )
                Float16Ops.convertFp32ToFp16(
                    srcPtr: flowRopeSinFp32.dataPointer.bindMemory(to: Float.self, capacity: halfDim),
                    dstPtr: flowRopeSin.dataPointer.bindMemory(to: UInt16.self, capacity: halfDim),
                    count: halfDim
                )

                let flowInputs: [String: MLFeatureValue] = [
                    "sequence":     MLFeatureValue(multiArray: sequence),
                    "kv_cache_in":  MLFeatureValue(multiArray: kvCache),
                    "offset_mask":  MLFeatureValue(multiArray: offsetMask),
                    "attn_mask":    MLFeatureValue(multiArray: attnMask),
                    "rope_cos":     MLFeatureValue(multiArray: flowRopeCos),
                    "rope_sin":     MLFeatureValue(multiArray: flowRopeSin),
                ]
                let provider = try MLDictionaryFeatureProvider(dictionary: flowInputs)
                let out = try models.flowMain.prediction(from: provider)

                guard
                    let ctxMA = out.featureValue(for: "ctx")?.multiArrayValue,
                    let eosMA = out.featureValue(for: "eos_logit")?.multiArrayValue,
                    let kvOutMA = out.featureValue(for: "kv_cache_out")?.multiArrayValue
                else { throw OrchestratorError.missingOutput("flow_lm_main") }

                // Copy kv_cache_out (fp16) back into our persistent kvCache buffer (fp16).
                copyFp16MLMultiArray(src: kvOutMA, dst: kvCache)
                currentOffset += 1

                // EOS check — eosMA is fp16[1, 1].
                let eosLogit = readFp16Scalar(eosMA)
                if eosLogit > options.eosThreshold, eosStep == nil {
                    eosStep = gen
                }
                if let eos = eosStep, gen >= eos + options.framesAfterEos {
                    return
                }

                // ---- FLOW LM FLOW (fp16 I/O) ----
                let noise = noiseSrc.sample(count: PocketTTSArch.latentDim)
                writeFp32ToFp16MA(noise, dst: flowNoiseIn, count: PocketTTSArch.latentDim)
                let flowFlowInputs: [String: MLFeatureValue] = [
                    "c": MLFeatureValue(multiArray: ctxMA),
                    "s": MLFeatureValue(multiArray: flowSIn),
                    "t": MLFeatureValue(multiArray: flowTIn),
                    "x": MLFeatureValue(multiArray: flowNoiseIn),
                ]
                let ffProvider = try MLDictionaryFeatureProvider(dictionary: flowFlowInputs)
                let ffOut = try models.flowFlow.prediction(from: ffProvider)
                guard let nextLatentMA = ffOut.featureValue(for: "next_latent")?.multiArrayValue
                else { throw OrchestratorError.missingOutput("flow_lm_flow") }

                readFp16MAToFp32(nextLatentMA, dst: &stepLatent, count: PocketTTSArch.latentDim)

                // ---- MIMI DECODER STEP (fp32 I/O — per artifact conversion) ----
                let mimiLatent = try MLMultiArray(
                    shape: [1, NSNumber(value: PocketTTSArch.latentDim), 1], dataType: .float32
                )
                mimiLatent.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
                    for i in 0..<PocketTTSArch.latentDim { buf[i] = stepLatent[i] }
                }
                let scatter = try Masks.scatterPrefillMask(
                    startOffset: mimiOffset, prefillLen: T, sCapacity: PocketTTSArch.mimiSCap
                )
                let mimiAttn = try Masks.additiveAttentionMaskPrefill(
                    startOffset: mimiOffset, prefillLen: T,
                    sCapacity: PocketTTSArch.mimiSCap,
                    context: PocketTTSArch.mimiTxContext
                )
                mimiRope.fillRange(offset: mimiOffset, length: T,
                                   cosOut: mimiRopeCos, sinOut: mimiRopeSin)

                let mimiInputs: [String: MLFeatureValue] = [
                    "latent":       MLFeatureValue(multiArray: mimiLatent),
                    "state_in":     MLFeatureValue(multiArray: mimiState.currentIn),
                    "scatter_mask": MLFeatureValue(multiArray: scatter),
                    "attn_mask":    MLFeatureValue(multiArray: mimiAttn),
                    "rope_cos":     MLFeatureValue(multiArray: mimiRopeCos),
                    "rope_sin":     MLFeatureValue(multiArray: mimiRopeSin),
                ]
                let mimiProvider = try MLDictionaryFeatureProvider(dictionary: mimiInputs)
                let mimiOut = try models.mimiDecoder.prediction(from: mimiProvider)
                guard
                    let audioMA = mimiOut.featureValue(for: "audio")?.multiArrayValue,
                    let stateOutMA = mimiOut.featureValue(for: "state_out")?.multiArrayValue
                else { throw OrchestratorError.missingOutput("mimi_decoder") }

                // Copy fp32 state_out back into the mimi state_in buffer.
                copyMLMultiArray(src: stateOutMA, dst: mimiState.currentIn)
                mimiOffset += T

                // audioMA is fp32[1,1,1920]. Convert to PCM16.
                let pcm = AudioStream.frameToPCM16(audioMA)
                continuation.yield(pcm)

                // Prepare next sequence input = current next_latent (fp16).
                writeFp32ToFp16MA(stepLatent, dst: sequence, count: PocketTTSArch.latentDim)
            }
            if let eos = eosStep, gen >= eos + options.framesAfterEos {
                break
            }
        }
    }

    @inline(__always)
    private func copyMLMultiArray(src: MLMultiArray, dst: MLMultiArray) {
        precondition(src.count == dst.count)
        src.withUnsafeBufferPointer(ofType: Float.self) { sp in
            dst.withUnsafeMutableBufferPointer(ofType: Float.self) { dp, _ in
                memcpy(dp.baseAddress!, sp.baseAddress!, src.count * MemoryLayout<Float>.stride)
            }
        }
    }

    @inline(__always)
    private func copyFp16MLMultiArray(src: MLMultiArray, dst: MLMultiArray) {
        precondition(src.count == dst.count)
        memcpy(dst.dataPointer, src.dataPointer, src.count * MemoryLayout<UInt16>.stride)
    }

    @inline(__always)
    private func writeFp32ToFp16MA(_ src: [Float], dst: MLMultiArray, count: Int) {
        let dstPtr = dst.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        src.withUnsafeBufferPointer { s in
            Float16Ops.convertFp32ToFp16(srcPtr: s.baseAddress!, dstPtr: dstPtr, count: count)
        }
    }

    @inline(__always)
    private func readFp16MAToFp32(_ src: MLMultiArray, dst: inout [Float], count: Int) {
        let srcPtr = src.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        dst.withUnsafeMutableBufferPointer { d in
            Float16Ops.convertFp16ToFp32(srcPtr: srcPtr, dstPtr: d.baseAddress!, count: count)
        }
    }

    @inline(__always)
    private func readFp16Scalar(_ ma: MLMultiArray) -> Float {
        let srcPtr = ma.dataPointer.bindMemory(to: UInt16.self, capacity: 1)
        var out: Float = 0
        withUnsafeMutablePointer(to: &out) { dst in
            Float16Ops.convertFp16ToFp32(srcPtr: srcPtr, dstPtr: dst, count: 1)
        }
        return out
    }

    @inline(__always)
    private func audioFp16ToPCM16(_ arr: MLMultiArray) -> Data {
        let n = arr.count
        var floats = [Float](repeating: 0, count: n)
        let src = arr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        floats.withUnsafeMutableBufferPointer { dst in
            Float16Ops.convertFp16ToFp32(srcPtr: src, dstPtr: dst.baseAddress!, count: n)
        }
        // Clip + scale + cast.
        var low: Float = -1, high: Float = 1
        var scale: Float = 32767
        vDSP_vclip(floats, 1, &low, &high, &floats, 1, vDSP_Length(n))
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(n))
        var ints = [Int16](repeating: 0, count: n)
        vDSP_vfixr16(floats, 1, &ints, 1, vDSP_Length(n))
        return ints.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

public enum OrchestratorError: Error, CustomStringConvertible {
    case prefillRequired
    case missingOutput(String)

    public var description: String {
        switch self {
        case .prefillRequired:
            return "Orchestrator needs a pre-prefilled VoiceHandle (Phase 4A: use tools/export_full_prefill.py)"
        case .missingOutput(let s):
            return "CoreML model \(s) did not return expected output"
        }
    }
}
