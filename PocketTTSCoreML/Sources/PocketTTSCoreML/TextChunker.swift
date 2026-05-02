import Foundation

/// Port of `pocket_tts.models.tts_model.split_into_best_sentences` — splits
/// long text into chunks ≤ `maxTokens` by walking end-of-sentence punctuation,
/// falling back to comma/colon if any single sentence is too long.
///
/// The default `maxTokens = 50` matches the reference's MAX_TOKEN_PER_CHUNK.
public enum TextChunker {
    public static let defaultMaxTokens: Int = 50

    public struct Preparation: Sendable {
        public let text: String
        public let framesAfterEosGuess: Int
    }

    /// Mirrors `prepare_text_prompt` — normalization + punctuation coercion.
    public static func prepareTextPrompt(
        _ input: String,
        padWithSpacesForShortInputs: Bool = true,
        removeSemicolons: Bool = false
    ) -> Preparation {
        var text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        precondition(!text.isEmpty, "Text prompt cannot be empty")
        text = text
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
        if removeSemicolons {
            text = text.replacingOccurrences(of: ";", with: ",")
        }
        let words = text.split(separator: " ", omittingEmptySubsequences: true)
        let wordCount = words.count
        let framesAfterEosGuess = wordCount <= 4 ? 3 : 1

        // Uppercase first char.
        if let first = text.first, !first.isUppercase, first.isLetter {
            text = first.uppercased() + text.dropFirst()
        }
        // Ensure trailing punctuation.
        if let last = text.last, last.isLetter || last.isNumber {
            text.append(".")
        }

        if padWithSpacesForShortInputs && wordCount < 5 {
            text = String(repeating: " ", count: 8) + text
        }
        return Preparation(text: text, framesAfterEosGuess: framesAfterEosGuess)
    }

    /// Port of `split_into_best_sentences`.
    public static func splitIntoBestSentences(
        _ text: String,
        tokenizer: Tokenizer,
        maxTokens: Int = defaultMaxTokens,
        padWithSpacesForShortInputs: Bool = true,
        removeSemicolons: Bool = false
    ) -> [String] {
        let prep = prepareTextPrompt(
            text,
            padWithSpacesForShortInputs: padWithSpacesForShortInputs,
            removeSemicolons: removeSemicolons
        ).text.trimmingCharacters(in: .whitespacesAndNewlines)

        let tokens = tokenizer.encodeAsInt(prep)
        // Matches reference: `_, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0]`
        // The leading element is dropped (usually the "▁" prefix token).
        let eosBoundaries = Array(tokenizer.encodeAsInt(".!...?").dropFirst())
        let fallback     = Array(tokenizer.encodeAsInt(",;:").dropFirst())

        let boundaries = findBoundaryIndices(tokens: tokens, boundaryTokens: Set(eosBoundaries))
        let segments = segmentsFromBoundaries(tokens: tokens, boundaries: boundaries, tokenizer: tokenizer)

        var refined: [(count: Int, text: String)] = []
        for seg in segments {
            if seg.count <= maxTokens {
                refined.append(seg)
                continue
            }
            let subTokens = tokenizer.encodeAsInt(seg.text.trimmingCharacters(in: .whitespacesAndNewlines))
            let subBounds = findBoundaryIndices(tokens: subTokens, boundaryTokens: Set(fallback))
            let subSegs = segmentsFromBoundaries(tokens: subTokens, boundaries: subBounds, tokenizer: tokenizer)
            if subSegs.count > 1 {
                refined.append(contentsOf: subSegs)
            } else {
                refined.append(seg)
            }
        }

        var chunks: [String] = []
        var current = ""
        var currentCount = 0
        for seg in refined {
            if current.isEmpty {
                current = seg.text
                currentCount = seg.count
                continue
            }
            if currentCount + seg.count > maxTokens {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = seg.text
                currentCount = seg.count
            } else {
                current += " " + seg.text
                currentCount += seg.count
            }
        }
        if !current.isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return chunks
    }

    static func findBoundaryIndices(tokens: [Int], boundaryTokens: Set<Int>) -> [Int] {
        var indices: [Int] = [0]
        var previousWasBoundary = false
        for (idx, tok) in tokens.enumerated() {
            if boundaryTokens.contains(tok) {
                previousWasBoundary = true
            } else {
                if previousWasBoundary {
                    indices.append(idx)
                }
                previousWasBoundary = false
            }
        }
        indices.append(tokens.count)
        return indices
    }

    static func segmentsFromBoundaries(
        tokens: [Int],
        boundaries: [Int],
        tokenizer: Tokenizer
    ) -> [(count: Int, text: String)] {
        var out: [(Int, String)] = []
        for i in 0..<(boundaries.count - 1) {
            let start = boundaries[i]
            let end = boundaries[i + 1]
            let slice = Array(tokens[start..<end]).map { Int32($0) }
            let text = tokenizer.decode(slice)
            out.append((end - start, text))
        }
        return out
    }
}
