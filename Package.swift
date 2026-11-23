// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Clarity",
    platforms: [.macOS("26.0"), .iOS("26.0")],
    products: [
        .library(name: "Clarity", targets: ["Clarity"]),
        .executable(name: "clarity-bench", targets: ["clarity-bench"]),
    ],
    targets: [
        .target(name: "Clarity", path: "Sources/Clarity"),
        .executableTarget(name: "clarity-bench", dependencies: ["Clarity"], path: "Sources/clarity-bench"),
    ],
    swiftLanguageModes: [.v5]
)
