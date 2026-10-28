#include "ShaderCommon.h"

// GPU stable LSD radix sort of splat depth keys.
// Keys use the same float->uint mapping as the CPU sorter, so a 4-pass sort reproduces the CPU
// order exactly (stable ties). With fewer passes only the lowest key byte(s) are ignored.

constant constexpr uint kSortThreads = 256;
constant constexpr uint kSortItems = 8;
constant constexpr uint kSortBlock = kSortThreads * kSortItems;
constant constexpr uint kRadix = 256;
constant constexpr uint kSimdWidth = 32;
constant constexpr uint kSimdGroups = kSortThreads / kSimdWidth;

typedef struct
{
    packed_float3 cameraPosition;
    float _pad0;
    packed_float3 cameraForward;
    float _pad1;
    uint count;
    uint byDistance;
    uint shift;
    uint blockCount;
    uint base;        // first global ordinal of the chunk being keyed
    uint localCount;  // splats in that chunk
} SortParams;

typedef struct
{
    uint base;
    uint chunkIndex;
} SortChunkBase;

inline uint depthKey(float depth) {
    // Matches SplatSorter.radixSortDescending: +0 and -0 equal, far (larger) first.
    uint bits = depth == 0.0f ? 0u : as_type<uint>(depth);
    uint ascending = (bits & 0x80000000u) == 0 ? (bits ^ 0x80000000u) : ~bits;
    return ~ascending;
}

kernel void gpuSortComputeKeys(uint i [[thread_position_in_grid]],
                               device const Splat* splats [[buffer(0)]],
                               device uint* keys [[buffer(1)]],
                               device uint* values [[buffer(2)]],
                               constant SortParams& p [[buffer(3)]]) {
    if (i >= p.localCount) return;
    float3 position = float3(splats[i].position);
    float depth;
    if (p.byDistance != 0) {
        float3 d = position - float3(p.cameraPosition);
        depth = d.x * d.x + d.y * d.y + d.z * d.z;
    } else {
        depth = dot(position, float3(p.cameraForward));
    }
    keys[p.base + i] = depthKey(depth);
    values[p.base + i] = p.base + i;
}

kernel void gpuSortHistogram(uint tid [[thread_index_in_threadgroup]],
                             uint block [[threadgroup_position_in_grid]],
                             device const uint* keys [[buffer(0)]],
                             device uint* histogram [[buffer(1)]],
                             constant SortParams& p [[buffer(2)]]) {
    threadgroup atomic_uint counts[kRadix];
    atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = block * kSortBlock;
    for (uint k = 0; k < kSortItems; ++k) {
        uint index = start + k * kSortThreads + tid;
        if (index < p.count) {
            atomic_fetch_add_explicit(&counts[(keys[index] >> p.shift) & 0xFFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Digit-major layout: an exclusive scan yields each block's global output base per digit.
    histogram[tid * p.blockCount + block] = atomic_load_explicit(&counts[tid], memory_order_relaxed);
}

kernel void gpuSortScan(uint tid [[thread_index_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]],
                        device uint* histogram [[buffer(0)]],
                        constant uint& total [[buffer(1)]],
                        threadgroup uint* partial [[threadgroup(0)]]) {
    uint per = (total + threads - 1) / threads;
    uint begin = min(tid * per, total);
    uint end = min(begin + per, total);
    uint sum = 0;
    for (uint i = begin; i < end; ++i) sum += histogram[i];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint running = 0;
        for (uint i = 0; i < threads; ++i) {
            uint v = partial[i];
            partial[i] = running;
            running += v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint running = partial[tid];
    for (uint i = begin; i < end; ++i) {
        uint v = histogram[i];
        histogram[i] = running;
        running += v;
    }
}

// Exclusive scan across the threadgroup. Returns the prefix; writes the total to `total`.
inline uint threadgroupExclusiveSum(uint value, uint lane, uint simdGroup,
                                    threadgroup uint* simdTotals, threadgroup uint& total) {
    uint prefix = simd_prefix_exclusive_sum(value);
    uint sum = simd_sum(value);
    if (lane == 0) simdTotals[simdGroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0) {
        uint t = lane < kSimdGroups ? simdTotals[lane] : 0u;
        uint ex = simd_prefix_exclusive_sum(t);
        if (lane < kSimdGroups) simdTotals[lane] = ex;
        if (lane == kSimdGroups - 1) total = ex + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simdTotals[simdGroup] + prefix;
}

kernel void gpuSortScatter(uint tid [[thread_index_in_threadgroup]],
                           uint block [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simdGroup [[simdgroup_index_in_threadgroup]],
                           device const uint* keysIn [[buffer(0)]],
                           device const uint* valuesIn [[buffer(1)]],
                           device uint* keysOut [[buffer(2)]],
                           device uint* valuesOut [[buffer(3)]],
                           device const uint* histogram [[buffer(4)]],
                           constant SortParams& p [[buffer(5)]]) {
    threadgroup uint tgKeys[kSortBlock];
    threadgroup uint tgValues[kSortBlock];
    threadgroup uint simdTotals[kSimdGroups];
    threadgroup uint totalZeros;
    threadgroup uint digitStart[kRadix];
    threadgroup uint globalBase[kRadix];

    uint start = block * kSortBlock;
    uint validCount = min(kSortBlock, p.count - start);

    // Coalesced load. Padding keys are all ones (digit 255) and sit after every real item,
    // so the stable local sort keeps them last and they are never written out.
    for (uint k = 0; k < kSortItems; ++k) {
        uint local = k * kSortThreads + tid;
        uint index = start + local;
        bool valid = local < validCount;
        tgKeys[local] = valid ? keysIn[index] : 0xFFFFFFFFu;
        tgValues[local] = valid ? valuesIn[index] : 0u;
    }
    globalBase[tid] = histogram[tid * p.blockCount + block];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread owns a contiguous run, which keeps the 1-bit splits stable.
    uint keys[kSortItems];
    uint values[kSortItems];
    for (uint k = 0; k < kSortItems; ++k) {
        keys[k] = tgKeys[tid * kSortItems + k];
        values[k] = tgValues[tid * kSortItems + k];
    }

    for (uint bit = 0; bit < 8; ++bit) {
        uint shift = p.shift + bit;
        uint zeros = 0;
        for (uint k = 0; k < kSortItems; ++k) zeros += ((keys[k] >> shift) & 1u) == 0 ? 1u : 0u;
        uint zerosBefore = threadgroupExclusiveSum(zeros, lane, simdGroup, simdTotals, totalZeros);
        uint zeroPosition = zerosBefore;
        uint onePosition = totalZeros + (tid * kSortItems - zerosBefore);
        for (uint k = 0; k < kSortItems; ++k) {
            uint position = ((keys[k] >> shift) & 1u) == 0 ? zeroPosition++ : onePosition++;
            tgKeys[position] = keys[k];
            tgValues[position] = values[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < kSortItems; ++k) {
            keys[k] = tgKeys[tid * kSortItems + k];
            values[k] = tgValues[tid * kSortItems + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = 0; k < kSortItems; ++k) {
        uint position = tid * kSortItems + k;
        uint digit = (keys[k] >> p.shift) & 0xFFu;
        uint previous = position == 0 ? 0xFFFFFFFFu : (tgKeys[position - 1] >> p.shift) & 0xFFu;
        if (digit != previous) digitStart[digit] = position;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < kSortItems; ++k) {
        uint position = tid * kSortItems + k;
        if (position >= validCount) continue;
        uint digit = (keys[k] >> p.shift) & 0xFFu;
        uint destination = globalBase[digit] + (position - digitStart[digit]);
        keysOut[destination] = keys[k];
        valuesOut[destination] = values[k];
    }
}

kernel void gpuSortWriteIndices(uint i [[thread_position_in_grid]],
                                device const uint* sortedOrdinals [[buffer(0)]],
                                device ChunkedSplatIndex* output [[buffer(1)]],
                                constant SortChunkBase* chunkBases [[buffer(2)]],
                                constant uint& chunkCount [[buffer(3)]],
                                constant uint& count [[buffer(4)]]) {
    if (i >= count) return;
    uint ordinal = sortedOrdinals[i];
    uint low = 0, high = chunkCount - 1;
    while (low < high) {
        uint mid = (low + high + 1) / 2;
        if (chunkBases[mid].base <= ordinal) low = mid; else high = mid - 1;
    }
    ChunkedSplatIndex index;
    index.chunkIndex = uint16_t(chunkBases[low].chunkIndex);
    index._padding = 0;
    index.splatIndex = ordinal - chunkBases[low].base;
    output[i] = index;
}
