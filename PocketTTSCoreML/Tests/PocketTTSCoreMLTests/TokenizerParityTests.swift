import XCTest

@testable import PocketTTSCoreML

/// BLOCKING: Swift SentencePiece must produce byte-for-byte-identical int32
/// token IDs to Python's `sentencepiece.SentencePieceProcessor.EncodeAsIds`
/// for the canonical prompt and a suite of secondary strings.
///
/// Values below were captured on 2026-05-01 from
///   pockettts_coreml/oracle/fixtures/english_alba_seed42/tokenizer.model
/// via `sp.Load(...) / sp.EncodeAsIds(text)`.
final class TokenizerParityTests: XCTestCase {

    // Python EncodeAsIds vectors. DO NOT edit without regenerating.
    static let goldens: [(text: String, ids: [Int32])] = [
        (
            "Pocket TTS is a lightweight text-to-speech model.",
            [
                1456, 603, 597, 602, 854, 640, 277, 267, 826, 1978, 2009, 337, 612, 337, 3476, 1836,
                263,
            ]
        ),
        ("Hello world!", [2994, 578, 682]),
        (
            "The quick brown fox jumps over the lazy dog.",
            [364, 976, 3683, 521, 1923, 1609, 261, 408, 265, 697, 690, 327, 1497, 263]
        ),
        ("Foo bar baz", [496, 1339, 1248, 891, 690]),
        ("A", [383]),
        ("   ", [260, 260, 260, 260]),
    ]

    func testVocabSize() throws {
        try XCTSkipUnless(
            FixturePaths.tokenizerAvailable,
            "tokenizer.model not available at \(FixturePaths.tokenizerURL.path)")
        let tok = try Tokenizer(modelURL: FixturePaths.tokenizerURL)
        XCTAssertEqual(tok.vocabSize, 4000)
    }

    func testEncodeParityAllGoldens() throws {
        try XCTSkipUnless(
            FixturePaths.tokenizerAvailable,
            "tokenizer.model not available at \(FixturePaths.tokenizerURL.path)")
        let tok = try Tokenizer(modelURL: FixturePaths.tokenizerURL)
        for (text, expected) in Self.goldens {
            let got = tok.encode(text)
            XCTAssertEqual(
                got, expected,
                "Tokenizer mismatch for \(String(reflecting: text))\n  expected=\(expected)\n  got=\(got)"
            )
        }
    }

    func testDecodeRoundtrip() throws {
        try XCTSkipUnless(
            FixturePaths.tokenizerAvailable,
            "tokenizer.model not available at \(FixturePaths.tokenizerURL.path)")
        let tok = try Tokenizer(modelURL: FixturePaths.tokenizerURL)
        let text = "Pocket TTS is a lightweight text-to-speech model."
        let ids = tok.encode(text)
        let decoded = tok.decode(ids)
        XCTAssertEqual(decoded, text)
    }
}
