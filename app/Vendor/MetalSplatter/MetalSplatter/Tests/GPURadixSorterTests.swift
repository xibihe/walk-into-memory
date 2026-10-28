import XCTest
import Metal
import simd
@testable import MetalSplatter

final class GPURadixSorterTests: XCTestCase {
    private func makeChunks(device: MTLDevice, sizes: [Int], seed: UInt64,
                            duplicates: Bool) throws -> [SplatSorter.ChunkReference] {
        var state = seed
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            // Multiples of 1/16 in [-10, 10): squared distances are exact in Float, so the CPU and the
            // GPU (which may fuse multiply-adds) compute bit-identical keys and the test isolates sort order.
            return Float(Int(state >> 40) % 320 - 160) / 16
        }
        return try sizes.enumerated().map { chunkIndex, size in
            let buffer = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: size)
            for i in 0..<size {
                // Duplicated positions exercise stability of equal keys.
                let p = duplicates && i % 3 == 0 ? SIMD3<Float>(1, 2, 3) : SIMD3(next(), next(), next())
                buffer.append(EncodedSplatPoint(position: p, colorSH0: .one, opacity: 1,
                                                scale: SIMD3(repeating: 0.01), rotation: simd_quatf(angle: 0, axis: SIMD3(0, 0, 1))))
            }
            return SplatSorter.ChunkReference(chunkIndex: UInt16(chunkIndex), buffer: buffer)
        }
    }

    private func cpuOrder(_ chunks: [SplatSorter.ChunkReference], camera: SIMD3<Float>) -> [SplatSorter.SplatReferenceAndDepth] {
        var values: [SplatSorter.SplatReferenceAndDepth] = []
        for chunk in chunks {
            for i in 0..<chunk.buffer.count {
                let p = chunk.buffer.values[i].position
                let d = SIMD3<Float>(p.x, p.y, p.z) - camera
                values.append(.init(chunkIndex: chunk.chunkIndex, splatIndex: UInt32(i),
                                    depth: d.x * d.x + d.y * d.y + d.z * d.z))
            }
        }
        var scratch: [SplatSorter.SplatReferenceAndDepth] = []
        SplatSorter.radixSortDescending(&values, scratch: &scratch)
        return values
    }

    private func gpuOrder(_ chunks: [SplatSorter.ChunkReference], camera: SIMD3<Float>,
                          passes: Int, device: MTLDevice) async throws -> [ChunkedSplatIndex] {
        let sorter = try GPURadixSorter(device: device)
        let count = chunks.reduce(0) { $0 + $1.buffer.count }
        let output = try MetalBuffer<ChunkedSplatIndex>(device: device, capacity: count)
        let ok = await sorter.sort(chunks: chunks, cameraPosition: camera, cameraForward: SIMD3(0, 0, -1),
                                   byDistance: true, passes: passes, output: output.buffer)
        XCTAssertTrue(ok)
        return Array(UnsafeBufferPointer(start: output.values, count: count))
    }

    func testFourPassGPUSortMatchesCPUExactly() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        for (sizes, duplicates) in [([1], false), ([2047, 1, 2049], true), ([100_003, 37_501], true), ([300_000], false)] {
            let chunks = try makeChunks(device: device, sizes: sizes, seed: UInt64(sizes.count * 977), duplicates: duplicates)
            let camera = SIMD3<Float>(0.25, -0.125, 0.0625)
            let expected = cpuOrder(chunks, camera: camera)
            let actual = try await gpuOrder(chunks, camera: camera, passes: 4, device: device)
            XCTAssertEqual(actual.map(\.chunkIndex), expected.map(\.chunkIndex), "sizes \(sizes)")
            XCTAssertEqual(actual.map(\.splatIndex), expected.map(\.splatIndex), "sizes \(sizes)")
        }
    }

    func testFewerPassesStayBackToFront() async throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let chunks = try makeChunks(device: device, sizes: [200_000], seed: 5, duplicates: false)
        let camera = SIMD3<Float>(0, 0, 0)
        let actual = try await gpuOrder(chunks, camera: camera, passes: 3, device: device)
        XCTAssertEqual(Set(actual.map(\.splatIndex)).count, 200_000, "every splat exactly once")
        var previous = Float.infinity
        for entry in actual {
            let p = chunks[0].buffer.values[Int(entry.splatIndex)].position
            let depth = p.x * p.x + p.y * p.y + p.z * p.z
            // Keys differ only below the ignored byte: relative error < 2^-15.
            XCTAssertLessThanOrEqual(depth, previous * (1 + 1.0 / 16384) + 1e-30)
            previous = min(previous, depth)
        }
    }
}
