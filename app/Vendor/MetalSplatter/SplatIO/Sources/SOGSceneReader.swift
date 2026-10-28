import CoreGraphics
import Foundation
import ImageIO
import simd
import ZIPFoundation

/// Reads a bundled PlayCanvas SOG v2 archive without changing the source file.
public final class SOGSceneReader: SplatSceneReader, @unchecked Sendable {
    public enum Error: Swift.Error, LocalizedError {
        case missingFile(String)
        case invalidArchive
        case unsupportedVersion(Int)
        case invalidMetadata
        case invalidImage(String)
        case invalidQuaternion
        case invalidPalette

        public var errorDescription: String? {
            switch self {
            case .missingFile(let name): "SOG archive is missing \(name)."
            case .invalidArchive: "SOG file is not a valid ZIP archive."
            case .unsupportedVersion(let version): "SOG v\(version) is not supported yet; export SOG v2."
            case .invalidMetadata: "SOG metadata is invalid."
            case .invalidImage(let name): "SOG image \(name) could not be read as raw RGBA."
            case .invalidQuaternion: "SOG rotation data is invalid."
            case .invalidPalette: "SOG spherical-harmonics palette index is invalid."
            }
        }
    }

    private struct Files: Decodable, Sendable {
        let files: [String]
    }

    private struct Codebook: Decodable, Sendable {
        let codebook: [Float]
        let files: [String]
    }

    private struct Means: Decodable, Sendable {
        let mins: [Float]
        let maxs: [Float]
        let files: [String]
    }

    private struct SHN: Decodable, Sendable {
        let count: Int
        let bands: Int
        let codebook: [Float]
        let files: [String]
    }

    private struct Metadata: Decodable, Sendable {
        let version: Int
        let count: Int
        let means: Means
        let scales: Codebook
        let quats: Files
        let sh0: Codebook
        let shN: SHN?
    }

    private struct Pixels: Sendable {
        let bytes: [UInt8]
        let width: Int
        let height: Int
        let stride: Int

        func rgba(at index: Int) -> SIMD4<UInt8> {
            let offset = (index / width) * stride + (index % width) * 4
            return SIMD4(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3])
        }

        func contains(_ count: Int) -> Bool { width * height >= count }
    }

    private let url: URL

    public init(_ url: URL) throws {
        guard url.isFileURL else { throw Error.invalidArchive }
        self.url = url
    }

    public func read() async throws -> AsyncThrowingStream<[SplatPoint], Swift.Error> {
        let url = url
        return AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    try Self.decode(url, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func data(_ name: String, from archive: Archive) throws -> Data {
        guard let entry = archive[name], entry.type == .file else { throw Error.missingFile(name) }
        var result = Data()
        _ = try archive.extract(entry) { result.append($0) }
        return result
    }

    private static func pixels(_ name: String, from archive: Archive) throws -> Pixels {
        let encoded = try data(name, from: archive)
        guard let source = CGImageSourceCreateWithData(encoded as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.width > 0, image.height > 0,
              image.bitsPerComponent == 8, image.bitsPerPixel == 32,
              image.alphaInfo == .last || image.alphaInfo == .noneSkipLast,
              let provider = image.dataProvider, let raw = provider.data as Data?,
              raw.count >= image.bytesPerRow * image.height else {
            throw Error.invalidImage(name)
        }
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        guard order == .byteOrderDefault || order == .byteOrder32Big else {
            throw Error.invalidImage(name)
        }
        return Pixels(bytes: [UInt8](raw), width: image.width,
                      height: image.height, stride: image.bytesPerRow)
    }

    private static func decode(_ url: URL,
                               continuation: AsyncThrowingStream<[SplatPoint], Swift.Error>.Continuation) throws {
        let archive: Archive
        do { archive = try Archive(url: url, accessMode: .read) }
        catch { throw Error.invalidArchive }
        let meta = try JSONDecoder().decode(Metadata.self, from: data("meta.json", from: archive))
        guard meta.version == 2 else { throw Error.unsupportedVersion(meta.version) }
        guard meta.count > 0, meta.means.mins.count == 3, meta.means.maxs.count == 3,
              meta.means.files.count == 2, meta.scales.files.count == 1,
              meta.quats.files.count == 1, meta.sh0.files.count == 1,
              meta.scales.codebook.count == 256, meta.sh0.codebook.count == 256 else {
            throw Error.invalidMetadata
        }
        let low = try pixels(meta.means.files[0], from: archive)
        let high = try pixels(meta.means.files[1], from: archive)
        let scales = try pixels(meta.scales.files[0], from: archive)
        let quats = try pixels(meta.quats.files[0], from: archive)
        let sh0 = try pixels(meta.sh0.files[0], from: archive)
        guard [low, high, scales, quats, sh0].allSatisfy({ $0.contains(meta.count) }),
              [high, scales, quats, sh0].allSatisfy({ $0.width == low.width && $0.height == low.height }) else {
            throw Error.invalidMetadata
        }

        var centroids: Pixels?
        var labels: Pixels?
        var shCoefficients = 0
        if let shN = meta.shN {
            guard (1...3).contains(shN.bands), (1...65_536).contains(shN.count),
                  shN.codebook.count == 256, shN.files.count == 2 else { throw Error.invalidMetadata }
            shCoefficients = [3, 8, 15][shN.bands - 1]
            centroids = try pixels(shN.files[0], from: archive)
            labels = try pixels(shN.files[1], from: archive)
            guard labels!.width == low.width, labels!.height == low.height,
                  centroids!.width >= 64 * shCoefficients,
                  centroids!.height >= (shN.count + 63) / 64 else { throw Error.invalidMetadata }
        }

        var batch = [SplatPoint]()
        batch.reserveCapacity(10_000)
        for index in 0..<meta.count {
            if index % 10_000 == 0 { try Task.checkCancellation() }
            let lo = low.rgba(at: index)
            let hi = high.rgba(at: index)
            let codeScale = scales.rgba(at: index)
            let encodedQuat = quats.rgba(at: index)
            let color = sh0.rgba(at: index)

            let position = SIMD3<Float>((0..<3).map { axis in
                let quantized = Float(UInt16(hi[axis]) << 8 | UInt16(lo[axis])) / 65_535
                let n = meta.means.mins[axis] +
                    (meta.means.maxs[axis] - meta.means.mins[axis]) * quantized
                return n.sign == .minus ? -expm1(-n) : expm1(n)
            })
            let scale = SIMD3<Float>(meta.scales.codebook[Int(codeScale.x)],
                                     meta.scales.codebook[Int(codeScale.y)],
                                     meta.scales.codebook[Int(codeScale.z)])
            guard encodedQuat.w >= 252 else { throw Error.invalidQuaternion }
            let components = [encodedQuat.x, encodedQuat.y, encodedQuat.z].map {
                (Float($0) / 255 - 0.5) * Float(2.0.squareRoot())
            }
            let omitted = Int(encodedQuat.w - 252)
            let missing = sqrt(max(0, 1 - components.reduce(0) { $0 + $1 * $1 }))
            var quaternion = [Float](repeating: 0, count: 4)
            var stored = 0
            for component in 0..<4 {
                if component == omitted { quaternion[component] = missing }
                else { quaternion[component] = components[stored]; stored += 1 }
            }

            var coefficients = [SIMD3<Float>(meta.sh0.codebook[Int(color.x)],
                                               meta.sh0.codebook[Int(color.y)],
                                               meta.sh0.codebook[Int(color.z)])]
            if let shN = meta.shN, let centroids, let labels {
                let label = labels.rgba(at: index)
                let palette = Int(label.x) | (Int(label.y) << 8)
                guard palette < shN.count else { throw Error.invalidPalette }
                for coefficient in 0..<shCoefficients {
                    let pixel = centroids.rgba(at: (palette / 64) * centroids.width +
                                               (palette % 64) * shCoefficients + coefficient)
                    coefficients.append(SIMD3<Float>(shN.codebook[Int(pixel.x)],
                                                     shN.codebook[Int(pixel.y)],
                                                     shN.codebook[Int(pixel.z)]))
                }
            }
            batch.append(SplatPoint(position: position,
                                    color: .sphericalHarmonicFloat(coefficients),
                                    opacity: .linearUInt8(color.w),
                                    scale: .exponent(scale),
                                    rotation: simd_quatf(ix: quaternion[1], iy: quaternion[2],
                                                         iz: quaternion[3], r: quaternion[0])))
            if batch.count == 10_000 {
                continuation.yield(batch)
                batch.removeAll(keepingCapacity: true)
            }
        }
        if !batch.isEmpty { continuation.yield(batch) }
    }
}
