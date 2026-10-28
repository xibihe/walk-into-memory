// CompositorServices frame lifecycle follows the MIT-licensed MetalSplatter sample.
import ARKit
import CompositorServices
import Metal
import MetalSplatter
import SplatIO
import Synchronization
import SwiftUI
import QuartzCore
import simd

struct ArtworkLayerConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities,
                           configuration: inout LayerRenderer.Configuration) {
        configuration.colorFormat = .bgra8Unorm_srgb
        configuration.depthFormat = .depth32Float
        configuration.isFoveationEnabled = capabilities.supportsFoveation
        if capabilities.supportsFoveation {
            configuration.maxRenderQuality = .init(CFG.renderQuality)
        }
        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions =
            capabilities.supportsFoveation ? [.foveationEnabled] : []
        configuration.layout = capabilities.supportedLayouts(options: options).contains(.layered)
            ? .layered : .dedicated
    }
}

private struct TwoHandGrab {
    var distance: Float
    var localPivot: SIMD3<Float>
    var startScale: Float
    var startDirection: SIMD3<Float>
    var startOrientation: simd_quatf
    var filteredMidpoint: SIMD3<Float>
    var filteredScale: Float
    var filteredOrientation: simd_quatf
    var time: Double
}

final class ArtworkRenderer: @unchecked Sendable {
    let layer: LayerRenderer
    let queue: MTLCommandQueue
    let bridge: RenderBridge
    let session = ARKitSession()
    let tracking = WorldTrackingProvider()
    let hands = HandTrackingProvider()
    let renderer = Mutex<SplatRenderer?>(nil)
    let inFlight = RenderInFlightGate()
    var handTrackingAllowed = false
    var leftPinching = false
    var rightPinching = false
    private var twoHandGrab: TwoHandGrab?
    private var lastEditing = true
    private var pinchArmed = false
    // Right-hand double pinch (immersion)
    private var rightTapDown = false
    private var rightTapFirst: Double?
    private var rightSinglePending = false
    var lastPlacementRevision: UInt64 = 0
    var clockStart = CACurrentMediaTime()
    private var appliedRenderQuality = CFG.renderQuality
    // Reach dissolve, one slot per hand (0 left, 1 right); render thread only.
    private var touchCenter: [SIMD3<Float>] = [.zero, .zero]
    private var touchAmountValue: [Float] = [0, 0]
    private var touchGrowth: [Float] = [0, 0]
    private var touchHold: [Float] = [0, 0]
    private var touchRadius: [Float] = [0, 0]
    private var touchLastPinch: [Double] = [-.infinity, -.infinity]
    private var touchClock: Double?
    // Fist wind; render thread only.
    private var fistHeld: [Bool] = [false, false]
    private var fistHold: [Double] = [0, 0]
    private var fistArmed: [Bool] = [true, true]
    private var lastWindRequest: UInt64 = 0
    private var windOn = false
    private var windOnSince: Double = 0
    private var windProgress: Float = 0
    private var windClock: Double?

    init(layer: LayerRenderer, bridge: RenderBridge) {
        self.layer = layer
        self.bridge = bridge
        queue = layer.device.makeCommandQueue()!
    }

    static func start(layer: LayerRenderer, url: URL, bridge: RenderBridge, detail: SceneDetail) {
        let host = ArtworkRenderer(layer: layer, bridge: bridge)
        Task(executorPreference: ArtworkRenderExecutor.shared) {
            if host.layer.configuration.isFoveationEnabled {
                host.layer.renderQuality = .init(CFG.renderQuality)
            }
            defer {
                host.session.stop()
                bridge.state.withLock { $0.finished = true }
            }
            if HandTrackingProvider.isSupported {
                let authorization = await host.session.requestAuthorization(for: [.handTracking])
                host.handTrackingAllowed = authorization[.handTracking] == .allowed
            }
            bridge.state.withLock {
                $0.handTrackingAvailable = host.handTrackingAllowed
                $0.gestureMessage = host.handTrackingAllowed
                    ? "Edit mode · pinch with both hands to move, rotate and scale."
                    : "Hand tracking unavailable · use the window to adjust and show/hide."
            }
            do {
                if host.handTrackingAllowed {
                    do { try await host.session.run([host.tracking, host.hands]) }
                    catch {
                        host.handTrackingAllowed = false
                        bridge.state.withLock {
                            $0.handTrackingAvailable = false
                            $0.gestureMessage = "Hand tracking failed to start · use the window to adjust and show/hide."
                        }
                        try await host.session.run([host.tracking])
                    }
                } else {
                    try await host.session.run([host.tracking])
                }
            } catch {
                bridge.state.withLock { $0.message = "World tracking failed to start: \(error.localizedDescription)" }
                host.loop()
                return
            }
            let loadTask = Task.detached(priority: .userInitiated) {
                do { try await host.load(url, detail: detail) }
                catch is CancellationError { }
                catch { bridge.state.withLock { $0.message = "Loading failed: \(error.localizedDescription)" } }
            }
            host.loop()
            loadTask.cancel()
        }
    }

    /// Visual contribution: opacity × projected area (two largest axes). Tiny, transparent points barely show but cost as much to sort and blend.
    static func importance(_ point: SplatPoint) -> Float {
        let s = point.scale.asLinearFloat
        let largest = max(s.x, s.y, s.z)
        let second = s.x + s.y + s.z - largest - min(s.x, s.y, s.z)
        return point.opacity.asLinearFloat * largest * second
    }

    /// Drop invisible points, then keep the top `fraction` by contribution, preserving the original order for locality.
    static func prune(_ points: [SplatPoint], fraction: Double) -> [SplatPoint] {
        let visible = points.filter { $0.opacity.asLinearFloat >= CFG.minimumOpacity }
        let target = Int((Double(points.count) * fraction).rounded())
        guard target < visible.count, target > 0 else { return visible.isEmpty ? points : visible }
        var scores = visible.map(importance)
        // Use the (count - target)-th smallest value as the threshold: O(n) selection, no full sort.
        let cutIndex = visible.count - target
        var sample = scores
        let threshold = Self.select(&sample, cutIndex)
        var kept = [SplatPoint]()
        kept.reserveCapacity(target)
        var ties = target - scores.lazy.filter { $0 > threshold }.count
        for (index, point) in visible.enumerated() {
            let score = scores[index]
            if score > threshold || (score == threshold && ties > 0) {
                if score == threshold { ties -= 1 }
                kept.append(point)
            }
        }
        scores.removeAll()
        return kept
    }

    /// Quickselect: the k-th smallest value (0-based).
    private static func select(_ values: inout [Float], _ k: Int) -> Float {
        var low = 0, high = values.count - 1
        var generator = SystemRandomNumberGenerator()
        while low < high {
            let pivot = values[Int.random(in: low...high, using: &generator)]
            var i = low, j = high
            while i <= j {
                while values[i] < pivot { i += 1 }
                while values[j] > pivot { j -= 1 }
                if i <= j { values.swapAt(i, j); i += 1; j -= 1 }
            }
            if k <= j { high = j } else if k >= i { low = i } else { return values[k] }
        }
        return values[k]
    }

    func load(_ url: URL, detail: SceneDetail) async throws {
        let reader = try AutodetectSceneReader(url)
        var points = [SplatPoint]()
        var sourceCount = 0
        for try await batch in try await reader.read() {
            try Task.checkCancellation()
            sourceCount += batch.count
            guard sourceCount <= CFG.maxPoints else { throw ArtworkLoadError.tooManyPoints }
            for point in batch {
                let p = point.position
                guard p.x.isFinite && p.y.isFinite && p.z.isFinite else {
                    throw ArtworkLoadError.invalidPosition
                }
                points.append(point)
            }
            bridge.state.withLock { $0.message = "Read \(sourceCount.formatted()) Gaussians…" }
        }
        guard !points.isEmpty else { throw ArtworkLoadError.empty }
        bridge.state.withLock { $0.message = "Ranking Gaussians by visual contribution…" }
        points = Self.prune(points, fraction: detail.fraction)
        guard !points.isEmpty else { throw ArtworkLoadError.empty }
        try Task.checkCancellation()
        let splat = try SplatRenderer(device: layer.device,
                                      colorFormat: layer.configuration.colorFormat,
                                      depthFormat: layer.configuration.depthFormat,
                                      sampleCount: 1,
                                      maxViewCount: layer.properties.viewCount,
                                      maxSimultaneousRenders: 8,
                                      highQualityDepth: true,
                                      clearColor: MTLClearColor(red: 0.012, green: 0.018, blue: 0.025, alpha: 1))
        splat.useGPUSort = CFG.gpuSort
        splat.gpuMinimumSortInterval = CFG.gpuSortInterval
        let firstSortReady = Mutex(false)
        let diagnostics = bridge.diagnostics
        splat.onSortComplete = { seconds in
            diagnostics.sorted(seconds: seconds)
            firstSortReady.withLock { $0 = true }
        }
        let chunk = try SplatChunk(device: layer.device, from: points)
        await splat.addChunk(chunk)
        while !firstSortReady.withLock({ $0 }) {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(2))
        }
        try Task.checkCancellation()
        bridge.diagnostics.scene(name: url.lastPathComponent, points: points.count, detail: detail)
        renderer.withLock { $0 = splat }
        bridge.state.withLock {
            let sampleCount = min(points.count, CFG.revealSampleCount)
            $0.revealSamples = (0..<sampleCount).map {
                points[($0 * points.count + points.count / 2) / sampleCount].position
            }
            $0.pointCount = points.count
            $0.sourceCount = sourceCount
            $0.message = "Loaded \(points.count.formatted()) Gaussians."
        }
    }

    private func pinchSample(_ anchor: HandAnchor?) -> (position: SIMD3<Float>, gap: Float)? {
        guard let anchor, anchor.isTracked, let skeleton = anchor.handSkeleton else { return nil }
        let thumb = skeleton.joint(.thumbTip)
        let index = skeleton.joint(.indexFingerTip)
        let a = (anchor.originFromAnchorTransform * thumb.anchorFromJointTransform).columns.3
        let b = (anchor.originFromAnchorTransform * index.anchorFromJointTransform).columns.3
        let thumbPosition = SIMD3<Float>(a.x, a.y, a.z)
        let indexPosition = SIMD3<Float>(b.x, b.y, b.z)
        return ((thumbPosition + indexPosition) / 2,
                simd_distance(thumbPosition, indexPosition))
    }

    /// The reach "contact point": middle knuckle blended with the index and middle tips; also returns the thumb–index gap (pinches do not count).
    private func reachSample(_ anchor: HandAnchor?) -> (position: SIMD3<Float>, gap: Float)? {
        guard let anchor, anchor.isTracked, let skeleton = anchor.handSkeleton else { return nil }
        func p(_ name: HandSkeleton.JointName) -> SIMD3<Float> {
            let c = (anchor.originFromAnchorTransform * skeleton.joint(name).anchorFromJointTransform).columns.3
            return SIMD3<Float>(c.x, c.y, c.z)
        }
        let knuckle = p(.middleFingerKnuckle), index = p(.indexFingerTip), middle = p(.middleFingerTip)
        return (knuckle * 0.5 + index * 0.25 + middle * 0.25, simd_distance(p(.thumbTip), index))
    }

    /// Curl of the four fingers: tip-to-knuckle distance ÷ chain length, taking the straightest finger. Open hand ≈ 0.95, fist ≈ 0.3–0.5.
    private func handCurl(_ anchor: HandAnchor?) -> Float? {
        guard let anchor, anchor.isTracked, let skeleton = anchor.handSkeleton else { return nil }
        func p(_ name: HandSkeleton.JointName) -> SIMD3<Float> {
            let c = skeleton.joint(name).anchorFromJointTransform.columns.3
            return SIMD3<Float>(c.x, c.y, c.z)
        }
        let fingers: [[HandSkeleton.JointName]] = [
            [.indexFingerKnuckle, .indexFingerIntermediateBase, .indexFingerIntermediateTip, .indexFingerTip],
            [.middleFingerKnuckle, .middleFingerIntermediateBase, .middleFingerIntermediateTip, .middleFingerTip],
            [.ringFingerKnuckle, .ringFingerIntermediateBase, .ringFingerIntermediateTip, .ringFingerTip],
            [.littleFingerKnuckle, .littleFingerIntermediateBase, .littleFingerIntermediateTip, .littleFingerTip]]
        var worst: Float = 0
        for f in fingers {
            let a = p(f[0]), b = p(f[1]), c = p(f[2]), d = p(f[3])
            let chain = simd_distance(a, b) + simd_distance(b, c) + simd_distance(c, d)
            guard chain > 1e-4 else { return nil }
            worst = max(worst, simd_distance(a, d) / chain)
        }
        return worst
    }

    /// Wind: hold a fist with either hand for ~0.3 s to toggle it. It ramps up over 1.5 s with a gust, calms over 3 s, and stops by itself after 14 s.
    private func updateWind(state: inout RenderState, curls: [Float?], now: Double) {
        let dt = Float(max(0, min(0.1, now - (windClock ?? now))))
        windClock = now
        let available = state.photoSpace && state.windConfig != nil
        var toggle = false
        if state.windRequest != lastWindRequest {
            lastWindRequest = state.windRequest
            if available { toggle = true }
        }
        for i in 0..<2 {
            guard let c = curls[i] else { fistHeld[i] = false; fistHold[i] = 0; continue }
            fistHeld[i] = c < (fistHeld[i] ? 0.7 : 0.55)
            if fistHeld[i] {
                fistHold[i] += Double(dt)
                if fistArmed[i], fistHold[i] >= 0.3, available, state.windEnabled, state.sceneRevealed {
                    fistArmed[i] = false
                    toggle = true
                }
            } else {
                fistHold[i] = 0
                fistArmed[i] = true
            }
        }
        if !available { windOn = false; windProgress = 0 }
        if toggle {
            windOn.toggle()
            if windOn { windOnSince = now }
            state.message = windOn ? "The wind picks up." : "The wind settles."
        }
        if windOn, now - windOnSince > 14 { windOn = false }
        windProgress = windOn ? min(1, windProgress + dt / 1.5) : max(0, windProgress - dt / 3)
        let p = windProgress
        var level = p * p * (3 - 2 * p)
        if windOn {
            // An opening gust: about +35% around 1.2 s.
            let x = Float(now - windOnSince - 1.2) / 0.6
            level *= 1 + 0.35 * exp(-x * x)
        }
        state.windLevel = level
        state.windOn = windOn
    }

    /// Reach dissolve: when a hand reaches into the scene (not pinching), the area around it slowly drifts apart like dust and gathers back a few seconds after the hand withdraws.
    private func updateTouch(state: inout RenderState, anchors: (leftHand: HandAnchor?, rightHand: HandAnchor?),
                             headPose: simd_float4x4?, now: Double) {
        let dt = Float(max(0, min(0.1, now - (touchClock ?? now))))
        touchClock = now
        let active = state.touchDissolveEnabled && state.photoSpace && state.sceneRevealed
            && !state.revealSamples.isEmpty && headPose != nil
        let scale = max(state.placement.scale, 1e-3)
        let inverse = state.placement.matrix.inverse
        let samples = state.revealSamples
        let hands = [anchors.leftHand, anchors.rightHand]
        for i in 0..<2 {
            var target: SIMD3<Float>?
            var depth: Float = 0
            if active, let head = headPose, let reach = reachSample(hands[i]), (handCurl(hands[i]) ?? 1) > 0.62 {
                if reach.gap < 0.045 { touchLastPinch[i] = now }
                let headPos = SIMD3<Float>(head.columns.3.x, head.columns.3.y, head.columns.3.z)
                let fwd = simd_normalize(-SIMD3<Float>(head.columns.2.x, head.columns.2.y, head.columns.2.z))
                let toHand = reach.position - headPos
                let along = simd_dot(toHand, fwd)
                let pinchedRecently = now - touchLastPinch[i] < 0.8
                if reach.gap > 0.06, along > 0.33, !pinchedRecently {
                    let l4 = inverse * SIMD4<Float>(reach.position, 1)
                    let local = SIMD3<Float>(l4.x, l4.y, l4.z)
                    let d4 = inverse * SIMD4<Float>(simd_normalize(toHand), 0)
                    let dir = simd_normalize(SIMD3<Float>(d4.x, d4.y, d4.z))
                    let h4 = inverse * SIMD4<Float>(headPos, 1)
                    let headLocal = SIMD3<Float>(h4.x, h4.y, h4.z)
                    let near = 0.3 / scale
                    var bestT = Float.infinity
                    var touching = false
                    for q in samples {
                        let v = q - local
                        if simd_length_squared(v) < near * near { touching = true; break }
                        let t = simd_dot(v, dir)
                        if t > 0, t < bestT, simd_length(v - dir * t) < 0.08 * t + 0.05 / scale { bestT = t }
                    }
                    if touching {
                        target = local
                    } else if bestT.isFinite {
                        target = local + dir * bestT
                    }
                    if let target { depth = simd_distance(headLocal, target) }
                }
            }
            if let target {
                touchHold[i] += dt
                if touchAmountValue[i] < 0.003 { touchCenter[i] = target }
                else { touchCenter[i] += (target - touchCenter[i]) * (1 - exp(-dt / 0.25)) }
                if touchHold[i] > 0.3 {
                    touchAmountValue[i] += (1 - touchAmountValue[i]) * (1 - exp(-dt / 1.0))
                    touchGrowth[i] = min(1, touchGrowth[i] + dt / 6)
                }
                let g = touchGrowth[i]
                touchRadius[i] = max((0.15 + 0.2 * g) / scale, depth * (0.04 + 0.08 * g))
            } else {
                touchHold[i] = 0
                touchAmountValue[i] *= exp(-dt / 1.0)
                touchGrowth[i] *= exp(-dt / 2.0)
                if touchAmountValue[i] < 0.003 { touchAmountValue[i] = 0; touchGrowth[i] = 0 }
            }
        }
        state.touchA = SIMD4<Float>(touchCenter[0], touchRadius[0])
        state.touchB = SIMD4<Float>(touchCenter[1], touchRadius[1])
        state.touchAmount = SIMD2<Float>(touchAmountValue[0], touchAmountValue[1])
        let up4 = inverse * SIMD4<Float>(0, 1, 0, 0)
        let up = SIMD3<Float>(up4.x, up4.y, up4.z)
        state.touchUp = simd_length(up) > 1e-5 ? simd_normalize(up) : SIMD3<Float>(0, 1, 0)
        state.touchDrift = 0.05 / scale
    }

    private func updateInteraction(at timestamp: Double, headPose: simd_float4x4?) -> RenderState {
        // Query ARKit hand anchors once per frame, on the render thread only.
        let anchors = handTrackingAllowed && hands.state == .running
            ? hands.handAnchors(at: timestamp) : (leftHand: nil, rightHand: nil)
        // In a fist the thumb often touches the index finger: never treat that as a pinch
        // (no show/hide, no double-pinch immersion, no reach dissolve).
        let curls = [handCurl(anchors.leftHand), handCurl(anchors.rightHand)]
        let left = (curls[0] ?? 1) < 0.62 ? nil : pinchSample(anchors.leftHand)
        let right = (curls[1] ?? 1) < 0.62 ? nil : pinchSample(anchors.rightHand)
        let now = CACurrentMediaTime()
        return bridge.state.withLock { state in
            if state.photoSpace, state.photoNeedsPlacement, let headPose {
                var forward = SIMD3<Float>(-headPose.columns.2.x, 0, -headPose.columns.2.z)
                if simd_length_squared(forward) > 0.0001 {
                    forward = simd_normalize(forward)
                    let head = headPose.columns.3
                    // Stand exactly where the photo was taken: the immersive-space origin is on the floor,
                    // so the scene origin (the original camera) goes to the recorded camera height.
                    // Without a height, fall back to head height with a short entry distance.
                    let entry: Float = state.photoFloorHeight != nil ? 0 : CFG.photoEntryDistance
                    state.placement.position = SIMD3<Float>(head.x, state.photoFloorHeight ?? head.y, head.z)
                        + forward * entry
                    state.placement.orientation = simd_quatf(
                        angle: atan2(-forward.x, -forward.z), axis: SIMD3<Float>(0, 1, 0))
                    state.photoNeedsPlacement = false
                    state.placementRevision &+= 1
                }
            }
            if state.placementRevision != lastPlacementRevision {
                lastPlacementRevision = state.placementRevision
                twoHandGrab = nil
            }
            leftPinching = left.map {
                $0.gap < (leftPinching ? CFG.pinchRelease : CFG.pinchEngage)
            } ?? false
            rightPinching = right.map {
                $0.gap < (rightPinching ? CFG.pinchRelease : CFG.pinchEngage)
            } ?? false

            if state.editing != lastEditing {
                lastEditing = state.editing
                twoHandGrab = nil
                pinchArmed = false
            }
            if !leftPinching && !rightPinching { pinchArmed = true }
            if !state.editing, state.visibilityStart.isFinite,
               now - state.visibilityStart >= state.transitionDuration {
                state.visibilityStart = -.infinity
                state.visibilityFrom = state.sceneRevealed ? 1 : 0
                state.message = state.sceneRevealed ? "Scene is here." : "Scene is gathered away."
            }

            // Immersion: in a scene with an edge frame, two right-hand pinches within 0.45 s toggle it.
            // A single right pinch (show/hide) waits until it is clearly not a double pinch;
            // a single left pinch acts immediately.
            let memoryTaps = state.edgeFrame != nil
            var singleDue = false
            if let first = rightTapFirst, now - first > CFG.doublePinchWindow {
                rightTapFirst = nil
                singleDue = rightSinglePending
                rightSinglePending = false
            }
            let tapDown = right.map { $0.gap < (rightTapDown ? CFG.tapRelease : CFG.pinchEngage) } ?? false
            if memoryTaps, tapDown, !rightTapDown, !leftPinching, state.pointCount > 0 {
                if rightTapFirst != nil {
                    rightTapFirst = nil
                    rightSinglePending = false
                    state.toggleImmersion(at: now)
                    state.message = state.immersive
                        ? "Immersion: the room fades to black, the edge softens."
                        : "Immersion off: back to the original edge."
                } else {
                    rightTapFirst = now
                    rightSinglePending = !state.editing && pinchArmed
                }
            }
            rightTapDown = tapDown
            let immediate = memoryTaps ? (pinchArmed && leftPinching)
                                       : (pinchArmed && (leftPinching || rightPinching))
            if !state.editing, state.pointCount > 0, immediate || singleDue {
                state.setVisibility(!state.sceneRevealed, at: now)
                state.message = state.sceneRevealed ? "Scene appearing…" : "Scene gathering away…"
                pinchArmed = false
            }

            if state.editing, state.pointCount > 0, leftPinching, rightPinching,
               let left, let right {
                let midpoint = (left.position + right.position) / 2
                let distance = simd_distance(left.position, right.position)
                if distance >= CFG.minimumHandSpan {
                    let direction = (right.position - left.position) / distance
                    if var grab = twoHandGrab {
                        let dt = now - grab.time
                        if dt > 0 && dt < CFG.maximumGestureGap {
                            let factor = Float(1 - exp(-dt / CFG.gestureDamping))
                            let targetScale = min(CFG.maximumScale,
                                                  max(CFG.minimumScale,
                                                      grab.startScale * distance / grab.distance))
                            grab.filteredMidpoint += (midpoint - grab.filteredMidpoint) * factor
                            grab.filteredScale += (targetScale - grab.filteredScale) * factor
                            let targetOrientation =
                                simd_quatf(from: grab.startDirection, to: direction) * grab.startOrientation
                            grab.filteredOrientation = simd_slerp(
                                grab.filteredOrientation, targetOrientation, factor)
                            grab.time = now
                            state.placement.scale = grab.filteredScale
                            state.placement.orientation = grab.filteredOrientation
                            let offset = state.placement.rotation *
                                SIMD4<Float>(grab.localPivot * grab.filteredScale, 0)
                            state.placement.position = grab.filteredMidpoint -
                                SIMD3<Float>(offset.x, offset.y, offset.z)
                            twoHandGrab = grab
                        } else {
                            twoHandGrab = nil
                        }
                    } else {
                        let local = state.placement.matrix.inverse * SIMD4<Float>(midpoint, 1)
                        twoHandGrab = TwoHandGrab(
                            distance: distance,
                            localPivot: SIMD3<Float>(local.x, local.y, local.z),
                            startScale: state.placement.scale,
                            startDirection: direction,
                            startOrientation: state.placement.orientation,
                            filteredMidpoint: midpoint, filteredScale: state.placement.scale,
                            filteredOrientation: state.placement.orientation,
                            time: now)
                    }
                    state.gestureMessage = "Holding · move, rotate and scale with both hands; release to place."
                } else {
                    twoHandGrab = nil
                    state.gestureMessage = "Move your hands a little further apart, then pinch."
                }
            } else {
                twoHandGrab = nil
                if state.editing {
                    state.gestureMessage = "Edit mode · pinch with both hands to move, rotate and scale."
                } else {
                    state.gestureMessage = state.sceneRevealed
                        ? "Viewing · pinch once to gather the scene away."
                        : "Viewing · pinch once to bring the scene in."
                }
            }
            updateTouch(state: &state, anchors: anchors, headPose: headPose, now: now)
            updateWind(state: &state, curls: curls, now: now)
            return state
        }
    }

    func loop() {
        while layer.state != .invalidated && !bridge.state.withLock({ $0.stopRequested }) {
            autoreleasepool {
                if layer.state == .paused { layer.waitUntilRunning() }
                else if layer.state == .running { renderFrame() }
            }
        }
    }

    func renderFrame() {
        let recording = bridge.state.withLock { $0.recordingPriority || $0.systemCaptureActive }
        if layer.configuration.isFoveationEnabled {
            let quality = recording ? CFG.recordingRenderQuality : CFG.renderQuality
            if quality != appliedRenderQuality {
                layer.renderQuality = .init(quality)
                appliedRenderQuality = quality
            }
        }
        let waitStart = CACurrentMediaTime()
        inFlight.acquire(limit: recording ? CFG.recordingFramesInFlight : CFG.maxFramesInFlight)
        let gpuWaitMS = (CACurrentMediaTime() - waitStart) * 1000
        var completionOwnsPermit = false
        defer { if !completionOwnsPermit { inFlight.release() } }
        guard layer.state == .running, !bridge.state.withLock({ $0.stopRequested }),
              let frame = layer.queryNextFrame() else { return }
        frame.startUpdate()
        frame.endUpdate()
        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)
        let drawables = frame.queryDrawables().sorted {
            $0.target == .builtIn && $1.target != .builtIn
        }
        guard !drawables.isEmpty else { return }
        frame.startSubmission()
        let encodeStart = CACurrentMediaTime()
        defer { frame.endSubmission() }
        var state = bridge.state.withLock { $0 }
        let splat = renderer.withLock { $0 }
        var presented = false
        for (drawableIndex, drawable) in drawables.enumerated() {
            guard let buffer = queue.makeCommandBuffer() else { continue }
            buffer.label = drawable.target == .capture ? "WalkIntoMemory.capture" : "WalkIntoMemory.display"
            let duration = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime)
            let timestamp = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
            let anchor = tracking.state == .running ? tracking.queryDeviceAnchor(atTimestamp: timestamp) : nil
            let tracked = anchor?.isTracked == true
            drawable.deviceAnchor = anchor
            bridge.state.withLock { $0.headPose = tracked ? anchor?.originFromAnchorTransform : nil }
            bridge.diagnostics.pose(tracked: tracked)
            if drawableIndex == 0 {
                if tracked, let transform = anchor?.originFromAnchorTransform {
                    bridge.diagnostics.motion(transform: transform, at: timestamp)
                }
                state = updateInteraction(at: timestamp,
                                          headPose: tracked ? anchor?.originFromAnchorTransform : nil)
            }
            let model = state.placement.matrix
            let viewports = drawable.views.enumerated().map { index, view in
                SplatRenderer.ViewportDescriptor(
                    viewport: view.textureMap.viewport,
                    projectionMatrix: drawable.computeProjection(viewIndex: index),
                    viewMatrix: ((anchor?.originFromAnchorTransform ?? matrix_identity_float4x4)
                                 * view.transform).inverse * model,
                    screenSize: SIMD2(Int(view.textureMap.viewport.width),
                                      Int(view.textureMap.viewport.height)))
            }
            let visibility = state.visibilityProgress(at: CACurrentMediaTime())
            let wantsContent = splat != nil && !state.photoNeedsPlacement && visibility > 0
            var ready = !wantsContent || tracked
            // Immersion: the surroundings (your room in Mixed, dark blue in Full) fade to black in 1.5 s;
            // the edge goes from hard to the soft breathing border in 2 s.
            let immersion = state.photoSpace && state.edgeFrame != nil
                ? state.immersionLevels(at: CACurrentMediaTime()) : (surround: Float(0), edge: Float(0))
            let dimRest = Double(1 - immersion.surround)
            let clearColor = state.viewingMode == .mixed
                ? MTLClearColor(red: 0, green: 0, blue: 0, alpha: Double(immersion.surround))
                : MTLClearColor(red: 0.012 * dimRest, green: 0.018 * dimRest, blue: 0.025 * dimRest, alpha: 1)
            let layered = layer.configuration.layout == .layered
            let groups = layered ? [Array(viewports.indices)] : viewports.indices.map { [$0] }
            for indices in groups where ready {
                let textureIndex = layered ? 0 : indices[0]
                let arrayLength = layered ? drawable.views.count : 1
                let rateMap = drawable.rasterizationRateMaps.isEmpty ? nil
                    : drawable.rasterizationRateMaps[min(textureIndex, drawable.rasterizationRateMaps.count - 1)]
                if let splat, wantsContent {
                    splat.clearColor = clearColor
                    let elapsed = CACurrentMediaTime() - clockStart
                    splat.edgeBreath = state.photoSpace ? state.edgeFrame.map {
                        SplatRenderer.EdgeBreath(frame: $0, time: Float(elapsed.truncatingRemainder(dividingBy: 3600)),
                                                 mix: immersion.edge)
                    } : nil
                    let edgeTime = Float(elapsed.truncatingRemainder(dividingBy: 3600))
                    splat.touchDissolve = state.photoSpace && (state.touchAmount.x > 0 || state.touchAmount.y > 0)
                        ? SplatRenderer.TouchDissolve(a: state.touchA, b: state.touchB, amount: state.touchAmount,
                                                      feather: 0.45, time: edgeTime, up: state.touchUp,
                                                      drift: state.touchDrift)
                        : nil
                    if state.photoSpace, state.windLevel > 0.001, let w = state.windConfig, w.count >= 11 {
                        splat.wind = SplatRenderer.Wind(time: edgeTime, strength: state.windLevel, ground: w[0], up: w[1],
                                                        shimmer: state.windShimmer,
                                                        box: SIMD4<Float>(w[2], w[3], w[4], w[5]),
                                                        water: SIMD4<Float>(w[6], w[7], w[8], w[9]), boxDepth: w[10])
                    } else {
                        splat.wind = nil
                    }
                    let reveal = SplatRenderer.RevealEffect(
                        progress: visibility,
                        noiseFrequency: CFG.revealNoiseFrequency * state.placement.scale,
                        frontWidth: CFG.revealFrontWidth,
                        seedRadius: CFG.revealSeedRadius / state.placement.scale,
                        particleTravel: CFG.revealParticleTravel / state.placement.scale,
                        origin: state.revealOrigin, direction: state.revealDirection,
                        nearDepth: state.revealNearDepth, farDepth: state.revealFarDepth,
                        style: state.activeRevealStyle.rawValue, up: state.revealUp,
                        low: state.revealLow, high: state.revealHigh, radius: state.revealRadius)
                    do {
                        ready = try splat.render(viewports: indices.map { viewports[$0] },
                            colorTexture: drawable.colorTextures[textureIndex], colorStoreAction: .store,
                            depthTexture: drawable.depthTextures[textureIndex], rasterizationRateMap: rateMap,
                            renderTargetArrayLength: arrayLength, revealEffect: reveal,
                            sortWorkFraction: recording ? CFG.recordingSortWorkFraction : 1,
                            updatesSortPose: drawable.target != .capture || drawables.count == 1,
                            accessTimeout: 0, sortTimeout: 0, to: buffer)
                    } catch {
                        ready = false
                        bridge.state.withLock { $0.message = "Rendering failed: \(error.localizedDescription)" }
                    }
                } else {
                    let pass = MTLRenderPassDescriptor()
                    pass.colorAttachments[0].texture = drawable.colorTextures[textureIndex]
                    pass.colorAttachments[0].loadAction = .clear
                    pass.colorAttachments[0].storeAction = .store
                    pass.colorAttachments[0].clearColor = clearColor
                    pass.depthAttachment.texture = drawable.depthTextures[textureIndex]
                    pass.depthAttachment.loadAction = .clear
                    pass.depthAttachment.storeAction = .store
                    pass.depthAttachment.clearDepth = 0
                    pass.renderTargetArrayLength = arrayLength
                    pass.rasterizationRateMap = rateMap
                    if let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) {
                        encoder.endEncoding()
                    } else { ready = false }
                }
            }
            if ready {
                drawable.encodePresent(commandBuffer: buffer)
                presented = presented || wantsContent
            } else {
                bridge.diagnostics.skip()
            }
            let recordGPU = wantsContent && ready
            let capture = drawable.target == .capture
            let deadlineDuration = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.renderingDeadline)
            let deadline = Double(deadlineDuration.components.seconds) + Double(deadlineDuration.components.attoseconds) / 1e18
            let lastDrawable = drawableIndex == drawables.count - 1
            let semaphore = inFlight
            let diagnostics = bridge.diagnostics
            buffer.addCompletedHandler { completed in
                if recordGPU {
                    diagnostics.gpu(seconds: completed.gpuEndTime - completed.gpuStartTime,
                                    capture: capture, deadline: deadline, completedAt: completed.gpuEndTime)
                }
                if lastDrawable { semaphore.release() }
            }
            if lastDrawable { completionOwnsPermit = true }
            buffer.commit()
        }
        bridge.diagnostics.frame(presented: presented, outputs: drawables.count,
            captureActive: state.systemCaptureActive, recordingPriority: state.recordingPriority,
            cpuMS: (CACurrentMediaTime() - encodeStart) * 1000, gpuWaitMS: gpuWaitMS,
            quality: appliedRenderQuality, inFlight: inFlight.activeCount)
    }
}

enum ArtworkLoadError: LocalizedError {
    case tooManyPoints, empty, invalidPosition
    var errorDescription: String? {
        switch self {
        case .tooManyPoints: "At most \(CFG.maxPoints.formatted()) Gaussians can be loaded."
        case .empty: "The file contains no Gaussians."
        case .invalidPosition: "The file contains invalid positions."
        }
    }
}

final class ArtworkRenderExecutor: TaskExecutor {
    static let shared = ArtworkRenderExecutor()
    private let queue = DispatchQueue(label: "WalkIntoMemory.Render", qos: .userInteractive)
    func enqueue(_ job: UnownedJob) {
        queue.async { job.runSynchronously(on: self.asUnownedTaskExecutor()) }
    }
}
