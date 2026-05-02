import Foundation
import CoreML
import Accelerate

/// Drives the CoreML AR loop for a single utterance.
///
/// Accepts either flavor of `VoiceHandle`:
///
/// - `.prefilled`: voice + text prefill already baked into the KV cache
///   (produced by the legacy `export_full_prefill.py`). Orchestrator goes
///   straight to the AR loop.
/// - `.voiceOnly`: voice-only KV prefix. Orchestrator runs
///   `text_conditioner.mlpackage` + `flow_lm_prefill.mlpackage` to bake
///   in the text prefill, then runs the AR loop. This is the normal path
///   — no Python helper required.
///
/// Models are optional to preserve compatibility with callers that only
/// hold AR-path artifacts (older bundles without `flow_lm_prefill.mlpackage`
/// or `text_conditioner.mlpackage`). In that case only `.prefilled`
/// handles work and `.voiceOnly` throws `prefillRequired`.
public final class Orchestrator: @unchecked Sendable {
    public struct Models {
        public let flowMain: MLModel
        public let flowFlow: MLModel
        public let mimiDecoder: MLModel
        public let mimiLayout: MimiStateLayout
        /// Optional — required only for in-Swift text prefill.
        public let flowPrefill: MLModel?
        /// Optional — required only for in-Swift text prefill.
        public let textConditioner: MLModel?
        /// Optional — used as a fallback when the `.voiceOnly` handle
        /// didn't carry its own bos_emb (read from
        /// `Artifacts/.../flow_lm_bos_emb.safetensors`).
        public let defaultBosEmb: [Float]?

        public init(
            flowMain: MLModel, flowFlow: MLModel, mimiDecoder: MLModel,
            mimiLayout: MimiStateLayout,
            flowPrefill: MLModel? = nil, textConditioner: MLModel? = nil,
            defaultBosEmb: [Float]? = nil
        ) {
            self.flowMain = flowMain
            self.flowFlow = flowFlow
            self.mimiDecoder = mimiDecoder
            self.mimiLayout = mimiLayout
            self.flowPrefill = flowPrefill
            self.textConditioner = textConditioner
            self.defaultBosEmb = defaultBosEmb
        }
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

    /// Run the AR loop for one voice and stream audio frames.
    ///
    /// - Parameter textTokens: int32 token ids for the prompt. Required
    ///   when `voice` is `.voiceOnly` (the orchestrator does the text
    ///   prefill in Swift). Ignored for `.prefilled` voices (text was
    ///   baked into the KV cache at export time).
    ///
    /// The returned async stream yields PCM16 little-endian frames of 1920
    /// samples each. Terminates either when max_gen_len is exhausted or
    /// `framesAfterEos` frames past the first EOS trigger.
    public func generate(
        voice: VoiceHandle,
        textTokens: [Int32] = [],
        options: PocketTTS.GenerateOptions,
        maxGenLen: Int = 512
    ) -> AsyncThrowingStream<Data, Error> {
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task { [self] in
            do {
                try self.runLoop(
                    voice: voice, textTokens: textTokens,
                    options: options,
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
        textTokens: [Int32],
        options: PocketTTS.GenerateOptions,
        maxGenLen: Int,
        continuation: AsyncThrowingStream<Data, Error>.Continuation
    ) throws {
        // Resolve to the AR-loop inputs: (kvCache, initialOffset, bosEmb, noiseSeq).
        let kvBox: MLMultiArrayBox
        let initialOffset: Int
        let bosEmb: [Float]
        let noiseSeq: [[Float]]?
        switch voice.kind {
        case let .prefilled(b, off, bos, _, ns):
            kvBox = b
            initialOffset = off
            bosEmb = bos
            noiseSeq = ns
        case let .voiceOnly(b, voiceOff, bos):
            guard textTokens.count > 0 else {
                throw OrchestratorError.emptyTextForVoiceOnly
            }
            let emb = bos ?? models.defaultBosEmb
            guard let resolvedBos = emb else {
                throw OrchestratorError.missingBosEmb
            }
            let newOffset = try runTextPrefill(
                voiceKV: b.array,
                voiceOffset: voiceOff,
                textTokens: textTokens
            )
            kvBox = b
            initialOffset = newOffset
            bosEmb = resolvedBos
            noiseSeq = nil  // no golden RNG trajectory for dynamic prompts
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
        var kvExhausted = false

        for gen in 0..<maxGenLen {
            // KV cache is S_cap=256 positions. When we've filled it, stop
            // cleanly with whatever audio we have rather than crashing in
            // Masks.additiveAttentionMaskStepFp16's precondition. Callers
            // wanting longer utterances must chunk at sentence boundaries
            // (TextChunker.splitIntoBestSentences) and run one generate()
            // per chunk.
            if currentOffset >= PocketTTSArch.flowSCap {
                kvExhausted = true
                break
            }
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
        if kvExhausted {
            // Emit one warning line to stderr; the caller gets clean finish
            // of the stream via AsyncThrowingStream.Continuation (we don't
            // throw here because we already produced audio). Apps that want
            // to generate long text must split at sentence boundaries; see
            // TextChunker.splitIntoBestSentences in the package.
            FileHandle.standardError.write(Data(
                "PocketTTS: KV cache S_cap=\(PocketTTSArch.flowSCap) exhausted at offset=\(currentOffset); stopped early. Split text at sentence boundaries for longer output.\n".utf8
            ))
        }
    }

    // ------------------------------------------------------------
    // Text prefill (Swift-native path)
    // ------------------------------------------------------------

    /// Run text_conditioner + flow_lm_prefill to update `voiceKV` in-place
    /// with the text KV at slots [voiceOffset, voiceOffset + S_text).
    /// Returns the new absolute offset (= voiceOffset + S_text).
    ///
    /// Inputs/outputs are fp16 end-to-end. The text_conditioner output is
    /// padded to S_TEXT_PAD=128 (the static input width baked into that
    /// .mlpackage). `scatter_mask` zeros out any column beyond S_text so
    /// the padded tokens don't touch the KV cache.
    public func runTextPrefill(
        voiceKV: MLMultiArray,
        voiceOffset: Int,
        textTokens: [Int32]
    ) throws -> Int {
        guard let prefill = models.flowPrefill else {
            throw OrchestratorError.prefillModelMissing
        }
        guard let textCond = models.textConditioner else {
            throw OrchestratorError.textConditionerMissing
        }

        let sTextPad = PocketTTSArch.sTextPad
        let sText = textTokens.count
        guard sText > 0 && sText <= sTextPad else {
            throw OrchestratorError.textTooLong(sText, max: sTextPad)
        }
        guard voiceOffset + sText <= PocketTTSArch.flowSCap else {
            throw OrchestratorError.prefillOverflow(
                voiceOffset: voiceOffset, sText: sText,
                sCap: PocketTTSArch.flowSCap
            )
        }

        // 1. text_conditioner: int32[1, 128] → fp16[1, 128, 1024].
        let tokensIn = try MLMultiArray(
            shape: [1, NSNumber(value: sTextPad)], dataType: .int32
        )
        tokensIn.withUnsafeMutableBufferPointer(ofType: Int32.self) { buf, _ in
            for i in 0..<sTextPad {
                buf[i] = i < sText ? textTokens[i] : 0
            }
        }
        let tcProvider = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokensIn)
        ])
        let tcOut = try prefill_predict_autorelease(model: textCond, provider: tcProvider)
        guard let embsMA = tcOut.featureValue(for: "embeddings")?.multiArrayValue else {
            throw OrchestratorError.missingOutput("text_conditioner")
        }
        // embsMA is fp16[1, 128, 1024].

        // 2. Build prefill inputs. scatter_mask covers only the real S_text
        //    columns; attn_mask rows 0..<S_text are causal, rows S_text..<128
        //    copy the last real row for softmax stability.
        let scatter = try Masks.scatterPrefillMaskFp16Padded(
            startOffset: voiceOffset, sText: sText, sTextPad: sTextPad,
            sCapacity: PocketTTSArch.flowSCap
        )
        let attn = try Masks.additiveAttentionMaskPrefillFp16Padded(
            startOffset: voiceOffset, sText: sText, sTextPad: sTextPad,
            sCapacity: PocketTTSArch.flowSCap
        )
        // RoPE table sliced at [voiceOffset, voiceOffset + 128), fp16.
        let rcFp32 = try MLMultiArray(
            shape: [1, NSNumber(value: sTextPad), 1,
                    NSNumber(value: PocketTTSArch.flowHalfDim)],
            dataType: .float32
        )
        let rsFp32 = try MLMultiArray(
            shape: [1, NSNumber(value: sTextPad), 1,
                    NSNumber(value: PocketTTSArch.flowHalfDim)],
            dataType: .float32
        )
        flowRope.fillRange(
            offset: voiceOffset, length: sTextPad,
            cosOut: rcFp32, sinOut: rsFp32
        )
        let rc = try MLMultiArray(
            shape: [1, NSNumber(value: sTextPad), 1,
                    NSNumber(value: PocketTTSArch.flowHalfDim)],
            dataType: .float16
        )
        let rs = try MLMultiArray(
            shape: [1, NSNumber(value: sTextPad), 1,
                    NSNumber(value: PocketTTSArch.flowHalfDim)],
            dataType: .float16
        )
        let ropeCount = sTextPad * PocketTTSArch.flowHalfDim
        Float16Ops.convertFp32ToFp16(
            srcPtr: rcFp32.dataPointer.bindMemory(to: Float.self, capacity: ropeCount),
            dstPtr: rc.dataPointer.bindMemory(to: UInt16.self, capacity: ropeCount),
            count: ropeCount
        )
        Float16Ops.convertFp32ToFp16(
            srcPtr: rsFp32.dataPointer.bindMemory(to: Float.self, capacity: ropeCount),
            dstPtr: rs.dataPointer.bindMemory(to: UInt16.self, capacity: ropeCount),
            count: ropeCount
        )

        // 3. Run flow_lm_prefill.
        let prefillFeatures: [String: MLFeatureValue] = [
            "text_embeddings": MLFeatureValue(multiArray: embsMA),
            "kv_cache_in":     MLFeatureValue(multiArray: voiceKV),
            "scatter_mask":    MLFeatureValue(multiArray: scatter),
            "attn_mask":       MLFeatureValue(multiArray: attn),
            "rope_cos":        MLFeatureValue(multiArray: rc),
            "rope_sin":        MLFeatureValue(multiArray: rs),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: prefillFeatures)
        let out = try prefill_predict_autorelease(model: prefill, provider: provider)
        guard let kvOut = out.featureValue(for: "kv_cache_out")?.multiArrayValue else {
            throw OrchestratorError.missingOutput("flow_lm_prefill")
        }
        // 4. Copy the updated KV back into the caller-owned buffer (both fp16).
        precondition(kvOut.count == voiceKV.count)
        memcpy(voiceKV.dataPointer, kvOut.dataPointer,
               voiceKV.count * MemoryLayout<UInt16>.stride)

        return voiceOffset + sText
    }

    @inline(__always)
    private func prefill_predict_autorelease(
        model: MLModel, provider: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        var result: MLFeatureProvider!
        var caught: Error?
        autoreleasepool {
            do { result = try model.prediction(from: provider) }
            catch { caught = error }
        }
        if let e = caught { throw e }
        return result
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
    case prefillModelMissing
    case textConditionerMissing
    case emptyTextForVoiceOnly
    case missingBosEmb
    case textTooLong(Int, max: Int)
    case prefillOverflow(voiceOffset: Int, sText: Int, sCap: Int)

    public var description: String {
        switch self {
        case .prefillRequired:
            return "Orchestrator needs a pre-prefilled VoiceHandle (load via VoiceLoader)"
        case .missingOutput(let s):
            return "CoreML model \(s) did not return expected output"
        case .prefillModelMissing:
            return "flow_lm_prefill.mlpackage not loaded; cannot run in-Swift text prefill"
        case .textConditionerMissing:
            return "text_conditioner.mlpackage not loaded; cannot run in-Swift text prefill"
        case .emptyTextForVoiceOnly:
            return "Voice-only handle requires text tokens (generate(text:voice:) with non-empty text)"
        case .missingBosEmb:
            return "bos_emb missing: no voice-provided value and no default sidecar loaded"
        case .textTooLong(let n, let m):
            return "text length \(n) exceeds S_TEXT_PAD=\(m); chunk the text first"
        case .prefillOverflow(let v, let s, let c):
            return "voice_offset(\(v)) + s_text(\(s)) > s_cap(\(c)); voice too long for cache"
        }
    }
}
