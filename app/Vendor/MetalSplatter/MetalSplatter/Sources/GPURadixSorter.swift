import Foundation
@preconcurrency import Metal
import simd

/// Sorts splats back-to-front on the GPU with a stable LSD radix sort and writes
/// `ChunkedSplatIndex` entries straight into the renderer's index buffer.
///
/// With `passes == 4` the output order is identical to `SplatSorter.radixSortDescending`.
/// Fewer passes ignore the lowest key byte(s): 3 passes keep sign, exponent and 15 mantissa
/// bits (relative depth resolution about 3e-5); only near-equal depths keep their input order.
final class GPURadixSorter: @unchecked Sendable {
    // Keep in sync with GPUSort.metal
    private static let threadsPerGroup = 256
    private static let itemsPerThread = 8
    private static let blockSize = threadsPerGroup * itemsPerThread
    private static let radix = 256

    struct Params {
        var cameraX: Float, cameraY: Float, cameraZ: Float, _pad0: Float = 0
        var forwardX: Float, forwardY: Float, forwardZ: Float, _pad1: Float = 0
        var count: UInt32
        var byDistance: UInt32
        var shift: UInt32 = 0
        var blockCount: UInt32
        var base: UInt32 = 0
        var localCount: UInt32 = 0
    }

    struct ChunkBase {
        var base: UInt32
        var chunkIndex: UInt32
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let keysPipeline: MTLComputePipelineState
    private let histogramPipeline: MTLComputePipelineState
    private let scanPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let writePipeline: MTLComputePipelineState

    // Scratch buffers, reused while the splat count does not grow. Only touched by the sort task.
    private var capacity = 0
    private var keys: [MTLBuffer] = []
    private var values: [MTLBuffer] = []
    private var histogram: MTLBuffer?

    init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw MetalBuffer<UInt32>.Error.bufferCreationFailed }
        queue.label = "Splat GPU Sort"
        self.queue = queue
        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalBuffer<UInt32>.Error.bufferCreationFailed
            }
            return try device.makeComputePipelineState(function: function)
        }
        keysPipeline = try pipeline("gpuSortComputeKeys")
        histogramPipeline = try pipeline("gpuSortHistogram")
        scanPipeline = try pipeline("gpuSortScan")
        scatterPipeline = try pipeline("gpuSortScatter")
        writePipeline = try pipeline("gpuSortWriteIndices")
        guard histogramPipeline.maxTotalThreadsPerThreadgroup >= Self.threadsPerGroup,
              scatterPipeline.maxTotalThreadsPerThreadgroup >= Self.threadsPerGroup,
              scatterPipeline.threadExecutionWidth == 32 else {
            throw MetalBuffer<UInt32>.Error.bufferCreationFailed
        }
    }

    private func ensureCapacity(_ count: Int) -> Bool {
        let blockCount = (count + Self.blockSize - 1) / Self.blockSize
        if count <= capacity, let histogram, histogram.length >= blockCount * Self.radix * 4 { return true }
        let length = max(count, 1) * MemoryLayout<UInt32>.stride
        var newKeys: [MTLBuffer] = []
        var newValues: [MTLBuffer] = []
        for _ in 0..<2 {
            guard let k = device.makeBuffer(length: length, options: .storageModePrivate),
                  let v = device.makeBuffer(length: length, options: .storageModePrivate) else { return false }
            newKeys.append(k)
            newValues.append(v)
        }
        guard let h = device.makeBuffer(length: max(blockCount, 1) * Self.radix * 4, options: .storageModePrivate) else {
            return false
        }
        keys = newKeys
        values = newValues
        histogram = h
        capacity = count
        return true
    }

    /// Sorts all chunks into `output` (capacity >= total splat count). Returns false on GPU failure.
    func sort(chunks: [SplatSorter.ChunkReference],
              cameraPosition: SIMD3<Float>,
              cameraForward: SIMD3<Float>,
              byDistance: Bool,
              passes: Int,
              output: MTLBuffer) async -> Bool {
        let count = chunks.reduce(0) { $0 + $1.buffer.count }
        guard count > 0, count <= Int(UInt32.max), ensureCapacity(count),
              let histogram, let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        commandBuffer.label = "Splat GPU Sort"
        encoder.label = "Radix Sort"

        let blockCount = (count + Self.blockSize - 1) / Self.blockSize
        var params = Params(cameraX: cameraPosition.x, cameraY: cameraPosition.y, cameraZ: cameraPosition.z,
                            forwardX: cameraForward.x, forwardY: cameraForward.y, forwardZ: cameraForward.z,
                            count: UInt32(count), byDistance: byDistance ? 1 : 0,
                            blockCount: UInt32(blockCount))

        // 1. Keys and ordinals, chunk by chunk, in the same order as the CPU sorter.
        var bases: [ChunkBase] = []
        var base = 0
        encoder.setComputePipelineState(keysPipeline)
        encoder.setBuffer(keys[0], offset: 0, index: 1)
        encoder.setBuffer(values[0], offset: 0, index: 2)
        for chunk in chunks where chunk.buffer.count > 0 {
            bases.append(ChunkBase(base: UInt32(base), chunkIndex: UInt32(chunk.chunkIndex)))
            params.base = UInt32(base)
            params.localCount = UInt32(chunk.buffer.count)
            encoder.setBuffer(chunk.buffer.buffer, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: chunk.buffer.count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: keysPipeline.threadExecutionWidth * 4, height: 1, depth: 1))
            base += chunk.buffer.count
        }

        // 2. Radix passes over the most significant bytes, ping-ponging scratch buffers.
        let passCount = min(4, max(1, passes))
        var source = 0
        var histogramTotal = UInt32(blockCount * Self.radix)
        let scanThreads = min(1024, scanPipeline.maxTotalThreadsPerThreadgroup)
        for pass in (4 - passCount)..<4 {
            params.shift = UInt32(pass * 8)
            let destination = 1 - source

            encoder.setComputePipelineState(histogramPipeline)
            encoder.setBuffer(keys[source], offset: 0, index: 0)
            encoder.setBuffer(histogram, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: blockCount, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))

            encoder.setComputePipelineState(scanPipeline)
            encoder.setBuffer(histogram, offset: 0, index: 0)
            encoder.setBytes(&histogramTotal, length: 4, index: 1)
            encoder.setThreadgroupMemoryLength(scanThreads * 4, index: 0)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))

            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(keys[source], offset: 0, index: 0)
            encoder.setBuffer(values[source], offset: 0, index: 1)
            encoder.setBuffer(keys[destination], offset: 0, index: 2)
            encoder.setBuffer(values[destination], offset: 0, index: 3)
            encoder.setBuffer(histogram, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 5)
            encoder.dispatchThreadgroups(MTLSize(width: blockCount, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
            source = destination
        }

        // 3. Ordinals back to (chunk, local index).
        var chunkCount = UInt32(bases.count)
        var total = UInt32(count)
        encoder.setComputePipelineState(writePipeline)
        encoder.setBuffer(values[source], offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(bases, length: bases.count * MemoryLayout<ChunkBase>.stride, index: 2)
        encoder.setBytes(&chunkCount, length: 4, index: 3)
        encoder.setBytes(&total, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: writePipeline.threadExecutionWidth * 4, height: 1, depth: 1))
        encoder.endEncoding()

        return await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { completed in
                continuation.resume(returning: completed.status == .completed && completed.error == nil)
            }
            commandBuffer.commit()
        }
    }
}
