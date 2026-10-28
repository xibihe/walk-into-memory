import Foundation
import QuartzCore
import Synchronization
import simd

final class RenderDiagnostics: Sendable {
    struct TimedValue: Sendable { var time: Double; var value: Double }
    struct Snapshot: Sendable {
        var frames = 0
        var recentFrames = [Double]()
        var fps = 0.0
        var longestGapMS = 0.0
        var outputs = 1
        var gpuMS = 0.0
        var sortMS = 0.0
        var skipped = 0
        var untracked = 0
        var displayGPU = [TimedValue]()
        var captureGPU = [TimedValue]()
        var lateGPU = [TimedValue]()
        var sorts = [TimedValue]()
        var lastHeadTime: Double = 0
        var lastHeadPosition: SIMD3<Float>?
        var lastHeadForward: SIMD3<Float>?
        var headAngularSpeed = 0.0
        var headLinearSpeed = 0.0
        var lastSaved: Double = 0
        var rows = 0
        // Scene info, written once after loading, so slow runs can be traced to a scene and detail level.
        var sceneName = ""
        var points = 0
        var detail = ""
    }
    private let state = Mutex(Snapshot())
    private let writer = DispatchQueue(label: "WalkIntoMemory.Metrics", qos: .utility)
    private let file: URL
    private static let header = "time,uptime,capture_active,manual_recording,outputs,fps,max_gap_ms,display_gpu_p95_ms,capture_gpu_p95_ms,gpu_late_max_ms,deadline_samples_2s,head_degrees_per_s,head_meters_per_s,thermal_state,cpu_encode_ms,gpu_wait_ms,sorts_2s,sort_cpu_ms_2s,render_quality,in_flight,skipped,untracked,scene,points,detail\n"

    init() {
        let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appending(path: "Diagnostics")
        file = folder.appending(path: "metrics-\(UUID().uuidString).csv")
    }

    func scene(name: String, points: Int, detail: SceneDetail) {
        // Keep the CSV columns intact: no commas or newlines in the name.
        let safe = name.replacingOccurrences(of: ",", with: "_").replacingOccurrences(of: "\n", with: " ")
        state.withLock {
            $0.sceneName = safe
            $0.points = points
            $0.detail = detail.rawValue
        }
    }

    func frame(presented: Bool, outputs: Int, captureActive: Bool, recordingPriority: Bool,
               cpuMS: Double, gpuWaitMS: Double, quality: Float, inFlight: Int) {
        guard presented else { return }
        let now = CACurrentMediaTime()
        let row: String? = state.withLock {
            $0.frames += 1
            $0.outputs = outputs
            $0.recentFrames.append(now)
            $0.recentFrames.removeAll { $0 < now - 2 }
            let times = $0.recentFrames
            if times.count > 1, let first = times.first {
                $0.fps = Double(times.count - 1) / (now - first)
                $0.longestGapMS = zip(times, times.dropFirst()).map { ($0.1 - $0.0) * 1000 }.max() ?? 0
            } else { $0.fps = 0; $0.longestGapMS = 0 }
            $0.displayGPU.removeAll { $0.time < now - 2 }
            $0.captureGPU.removeAll { $0.time < now - 2 }
            $0.lateGPU.removeAll { $0.time < now - 2 }
            $0.sorts.removeAll { $0.time < now - 2 }
            guard now - $0.lastSaved >= 1, $0.rows < 1800 else { return nil }
            $0.lastSaved = now
            $0.rows += 1
            func p95(_ samples: [TimedValue]) -> Double {
                let values = samples.map(\.value).sorted()
                return values.isEmpty ? 0 : values[Int(Double(values.count - 1) * 0.95)]
            }
            let time = Date().timeIntervalSince1970
            return "\(time),\(now),\(captureActive ? 1 : 0),\(recordingPriority ? 1 : 0),\(outputs),\($0.fps),\($0.longestGapMS),\(p95($0.displayGPU)),\(p95($0.captureGPU)),\($0.lateGPU.map(\.value).max() ?? 0),\($0.lateGPU.count),\($0.headAngularSpeed),\($0.headLinearSpeed),\(ProcessInfo.processInfo.thermalState.rawValue),\(cpuMS),\(gpuWaitMS),\($0.sorts.count),\($0.sorts.map(\.value).reduce(0,+)),\(quality),\(inFlight),\($0.skipped),\($0.untracked),\($0.sceneName),\($0.points),\($0.detail)\n"
        }
        if let row {
            let file = file
            writer.async {
                do {
                    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                    if !FileManager.default.fileExists(atPath: file.path) {
                        try Data(Self.header.utf8).write(to: file)
                    }
                    let handle = try FileHandle(forWritingTo: file)
                    defer { try? handle.close() }
                    try handle.seekToEnd()
                    try handle.write(contentsOf: Data(row.utf8))
                } catch { /* Metrics must never interrupt rendering. */ }
            }
        }
    }

    func gpu(seconds: Double, capture: Bool, deadline: Double, completedAt: Double) {
        let now = CACurrentMediaTime()
        state.withLock {
            $0.gpuMS = seconds * 1000
            let sample = TimedValue(time: now, value: seconds * 1000)
            if capture { $0.captureGPU.append(sample) } else { $0.displayGPU.append(sample) }
            // The Metal and compositor times are Mach absolute time. Ignore unavailable values.
            let delta = completedAt - deadline
            if completedAt > 0, abs(delta) < 2 {
                $0.lateGPU.append(TimedValue(time: now, value: max(0, delta * 1000)))
            }
        }
    }
    func sorted(seconds: Double) {
        let now = CACurrentMediaTime()
        state.withLock {
            $0.sortMS = seconds * 1000
            $0.sorts.append(TimedValue(time: now, value: seconds * 1000))
        }
    }
    func motion(transform: simd_float4x4, at timestamp: Double) {
        let p = transform.columns.3
        let f = transform.columns.2
        let position = SIMD3<Float>(p.x, p.y, p.z)
        let forward = simd_normalize(SIMD3<Float>(f.x, f.y, f.z))
        state.withLock {
            let dt = timestamp - $0.lastHeadTime
            if dt > 0, dt < 0.2, let oldPosition = $0.lastHeadPosition, let oldForward = $0.lastHeadForward {
                let cosine = min(1, max(-1, simd_dot(oldForward, forward)))
                $0.headAngularSpeed = Double(acos(cosine)) * 180 / .pi / dt
                $0.headLinearSpeed = Double(simd_distance(position, oldPosition)) / dt
            } else { $0.headAngularSpeed = 0; $0.headLinearSpeed = 0 }
            $0.lastHeadTime = timestamp
            $0.lastHeadPosition = position
            $0.lastHeadForward = forward
        }
    }
    func skip() { state.withLock { $0.skipped += 1 } }
    func pose(tracked: Bool) { if !tracked { state.withLock { $0.untracked += 1 } } }
    var summary: String {
        let value = state.withLock { $0 }
        return String(format: "%.0f frames/s submitted · longest gap %.1f ms · GPU %.1f ms · sort %.1f ms · %d outputs · %d skipped · %d untracked",
                      value.fps, value.longestGapMS, value.gpuMS, value.sortMS,
                      value.outputs, value.skipped, value.untracked)
    }
}
