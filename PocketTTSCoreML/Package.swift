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
        .target(
            name: "PocketTTSCoreML",
            dependencies: [
                "SentencePiece",
            ],
            path: "Sources/PocketTTSCoreML",
            resources: [
                // No bundled resources by default; caller passes URLs.
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
            path: "Tests/PocketTTSCoreMLTests",
            exclude: ["Fixtures/README.md"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
