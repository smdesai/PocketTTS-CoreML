// swift-tools-version:6.0
import PackageDescription

// PocketTTSCoreML — Phase 4A target (macOS-only).
// iOS support is explicitly listed in platforms so Phase 4B can enable it
// without restructuring the manifest.
let package = Package(
    name: "PocketTTSCoreML",
    platforms: [
        .macOS(.v14),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "PocketTTSCoreML",
            targets: ["PocketTTSCoreML"]
        ),
        .executable(
            name: "pocket-tts-cli",
            targets: ["PocketTTSCLI"]
        ),
    ],
    targets: [
        // SentencePiece C++ library slices packaged as a binary xcframework.
        .binaryTarget(
            name: "SentencePiece",
            path: "Frameworks/SentencePiece.xcframework"
        ),
        // Thin C bridge exposing the SentencePiece C++ API via extern "C".
        .target(
            name: "CSentencePieceBridge",
            dependencies: ["SentencePiece"],
            path: "Sources/PocketTTSCoreML/CSentencePieceBridge",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../../../Frameworks/SentencePiece.xcframework/macos-arm64_x86_64/SentencePiece.framework/Headers"),
                .define("SENTENCEPIECE_STATIC", to: "1"),
            ],
            linkerSettings: [
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "PocketTTSCoreML",
            dependencies: [
                "CSentencePieceBridge",
                "SentencePiece",
            ],
            path: "Sources/PocketTTSCoreML",
            exclude: [
                "CSentencePieceBridge",
            ],
            resources: [
                // No bundled resources by default; caller passes URLs.
            ],
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
            ],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedLibrary("c++"),
            ]
        ),
        .executableTarget(
            name: "PocketTTSCLI",
            dependencies: ["PocketTTSCoreML"],
            path: "Sources/PocketTTSCLI"
        ),
        .testTarget(
            name: "PocketTTSCoreMLTests",
            dependencies: ["PocketTTSCoreML"],
            path: "Tests/PocketTTSCoreMLTests"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
