// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SCNetKaraokeEngine",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "SCNetKaraokeEngine",
            targets: ["SCNetKaraokeEngine"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0")
    ],
    targets: [
        .target(
            name: "SCNetKaraokeEngine",
            dependencies: [
                .product(name: "Atomics", package: "swift-atomics")
            ],
            path: "SCNetKaraokeEngine/iOSApp",
            exclude: [
                "Config/Info.plist"
            ]
        ),
        .testTarget(
            name: "SCNetKaraokeEngineTests",
            dependencies: ["SCNetKaraokeEngine"],
            path: "Tests/SCNetKaraokeEngineTests"
        )
    ]
)

