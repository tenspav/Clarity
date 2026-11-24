// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "ClarityANEBench",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(name: "ClarityANEBench", path: "Sources/ClarityANEBench",
                          resources: [.copy("ClarityBridge.metal")]),
    ],
    swiftLanguageModes: [.v5]
)
