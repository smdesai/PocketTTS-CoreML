// swift-tools-version:6.0
import PackageDescription

// PocketTTSCoreML Swift package.
// The manifest lives at the repository root so the package can be consumed as a
// git dependency (SwiftPM requires a root Package.swift); the Swift sources,
// tests and binary framework stay under PocketTTSCoreML/.
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
            path: "PocketTTSCoreML/Frameworks/SentencePiece.xcframework"
        ),
        .target(
            name: "PocketTTSCoreML",
            dependencies: [
                "SentencePiece",
            ],
            path: "PocketTTSCoreML/Sources/PocketTTSCoreML",
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
            path: "PocketTTSCoreML/Sources/PocketTTSCLI"
        ),
        .testTarget(
            name: "PocketTTSCoreMLTests",
            dependencies: ["PocketTTSCoreML"],
            path: "PocketTTSCoreML/Tests/PocketTTSCoreMLTests",
            exclude: ["Fixtures/README.md"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
