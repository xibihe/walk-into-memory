import Foundation
import Observation
import QuartzCore
import Synchronization
import simd

/// Load-time detail: keeps the Gaussians with the largest opacity × projected area instead of sampling uniformly.
enum SceneDetail: String, CaseIterable, Identifiable, Sendable {
    case full, balanced, light
    var id: Self { self }
    /// Fraction of the source Gaussians that is kept.
    var fraction: Double {
        switch self { case .full: 1; case .balanced: 0.6; case .light: 0.35 }
    }
    var title: String {
        switch self { case .full: "Full"; case .balanced: "Balanced 60%"; case .light: "Light 35%" }
    }
}

// All tunable values live here. Distances are in metres (the units of the scene file).
enum CFG {
    static let sceneAssetsFolder = "SceneAssets" // Bundled scenes: put one <name>.spz (+ <name>.json) here.
    static let positionStep: Float = 0.10 // Window nudge buttons move 10 cm.
    static let maxPoints = 4_000_000 // Upper limit when reading a file.
    static let minimumOpacity: Float = 1.0 / 255 // Invisible at 8-bit precision; dropped while loading.
    static let maxFramesInFlight = 1 // Normal frame pacing.
    static let recordingFramesInFlight = 2 // While recording, overlap CPU/GPU by at most two frames.
    static let recordingSortWorkFraction: Double = 0.5 // While recording, leave headroom for the encoder.
    static let gpuSort = true // Sort on the GPU (bit-identical to the CPU sort).
    static let gpuSortInterval: Double = 1.0 / 45 // Minimum time between two GPU sorts.
    static let renderQuality: Float = 1.0 // Full foveated quality.
    static let recordingRenderQuality: Float = 0.72 // Recording priority: lower resolution, full scene.
    static let pinchEngage: Float = 0.035 // Metres between thumb and index tip to start a pinch.
    static let pinchRelease: Float = 0.055 // Metres to release it again.
    static let tapRelease: Float = 0.045 // Smaller release distance for double-pinch detection.
    static let doublePinchWindow: Double = 0.45 // Seconds between two right-hand pinches.
    static let immersionSurroundDuration: Double = 1.5 // Seconds for the surroundings to fade to black.
    static let immersionEdgeDuration: Double = 2.0 // Seconds for the hard edge to become the soft breathing edge.
    static let minimumHandSpan: Float = 0.12 // Hands closer than this do not start a two-hand grab.
    static let minimumScale: Float = 0.1
    static let maximumScale: Float = 10
    static let gestureDamping: Double = 0.025 // Exponential smoothing, no overshoot.
    static let maximumGestureGap: Double = 0.15 // Re-grab after a long tracking gap.
    static let defaultRevealDuration: Double = 1.6 // Seconds for show/hide.
    static let revealNoiseFrequency: Float = 1.3 // Noise cycles per metre; scaled with the scene.
    static let revealFrontWidth: Float = 0.24 // The particle front takes 24% of the progress.
    static let revealSeedRadius: Float = 0.003 // Newborn particle radius, 3 mm.
    static let revealParticleTravel: Float = 0.045 // Front particles travel up to 4.5 cm before settling.
    static let revealSampleCount = 4096 // Scene samples used for reveal depth and reach targets.
    static let photoEntryDistance: Float = 0.65 // Only used if a sidecar has no camera height.
    static let savedScenePathKey = "LastScenePath" // Last imported scene.
}

/// The JSON written next to each .spz by the pipeline (`python -m wim run ...`).
struct SceneSidecar: Codable, Sendable {
    struct Wind: Codable, Sendable {
        var ground: Float // Height of the scene origin above the floor (metres).
        var up: Float // Sign that turns local y into height: h = ground + up * y.
        var stillBox: [Float] // Wind-free screen box (x0 x1 y0 y1 in [-1, 1]), usually the person.
        var stillDepth: Float // The still box only protects points closer than this (metres).
        var waterBox: [Float]? // Optional water shimmer box.
    }

    var format: String?
    var version: Int
    var cameraHeight: Float? // Original camera height above the floor, metres.
    var edgeFrame: [Float]? // 24 floats: the original camera, used by the edge, wind and immersion effects.
    var wind: Wind?

    static let noBox: [Float] = [2, 2, 2, 2]

    var validEdgeFrame: [Float]? {
        guard let edgeFrame, edgeFrame.count == 24, edgeFrame.allSatisfy(\.isFinite) else { return nil }
        return edgeFrame
    }

    /// [ground, up, still x0 x1 y0 y1, water x0 x1 y0 y1, stillDepth] as the renderer expects it.
    var windConfig: [Float]? {
        guard validEdgeFrame != nil, let wind, wind.stillBox.count == 4 else { return nil }
        let water = wind.waterBox?.count == 4 ? wind.waterBox! : Self.noBox
        let values = [wind.ground, wind.up] + wind.stillBox + water + [wind.stillDepth]
        return values.allSatisfy(\.isFinite) ? values : nil
    }

    var hasWater: Bool { wind?.waterBox?.count == 4 }

    static func load(_ url: URL) -> SceneSidecar? {
        guard let data = try? Data(contentsOf: url),
              let value = try? JSONDecoder().decode(SceneSidecar.self, from: data),
              value.version == 1 else { return nil }
        return value
    }
}

enum ViewingMode: String, CaseIterable, Identifiable, Sendable {
    case mixed = "Mixed · your room"
    case virtual = "Full · virtual"
    var id: Self { self }
}

enum RevealStyle: UInt32, CaseIterable, Identifiable, Sendable {
    case spread = 1, magic, unroll, twister
    var id: Self { self }
    var title: String {
        switch self {
        case .spread: "Spread"
        case .magic: "Magic"
        case .unroll: "Unroll"
        case .twister: "Twister"
        }
    }
    var detail: String {
        switch self {
        case .spread: "Particles grow from near to far, layer by layer."
        case .magic: "Small particles light up around you, then gather outward."
        case .unroll: "The scene unrolls around its height axis."
        case .twister: "Particles swirl up from below into the full space."
        }
    }
}

struct ScenePlacement: Equatable, Sendable {
    var position = SIMD3<Float>.zero
    var scale: Float = 1
    var flipVertical = false
    var orientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))

    /// Generic scene files: placed at the immersive-space origin.
    static let imported = ScenePlacement(position: .zero, scale: 1, flipVertical: false)
    /// Pipeline scenes are y-down (camera convention); flip them upright.
    static let photo = ScenePlacement(position: .zero, scale: 1, flipVertical: true)

    var rotation: simd_float4x4 {
        simd_float4x4(orientation * simd_quatf(angle: flipVertical ? .pi : 0,
                                               axis: SIMD3<Float>(1, 0, 0)))
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.position == rhs.position && lhs.scale == rhs.scale &&
        lhs.flipVertical == rhs.flipVertical && lhs.orientation.vector == rhs.orientation.vector
    }

    var matrix: simd_float4x4 {
        var result = rotation
        result.columns.0 *= scale
        result.columns.1 *= scale
        result.columns.2 *= scale
        result.columns.3 = SIMD4<Float>(position, 1)
        return result
    }
}

struct RenderState: Sendable {
    var viewingMode: ViewingMode = .mixed
    var recordingPriority = false
    var systemCaptureActive = false
    var photoSpace = false // A pipeline scene with a sidecar: placed at the original camera, floor aligned.
    var photoNeedsPlacement = false
    var photoFloorHeight: Float? = nil // Original camera height above the floor.
    var edgeFrame: [Float]? = nil // Soft edge / immersion parameters.
    var windConfig: [Float]? = nil
    var windShimmer: Float = 0 // > 0 only if the scene marks a water area.
    var windEnabled = true // Allow the fist gesture (the window button always works).
    var windRequest: UInt64 = 0 // Window button counter: each press toggles the wind.
    var windLevel: Float = 0 // Written by the render thread: 0 calm, 1 full wind.
    var windOn = false // Written by the render thread.
    // Reach dissolve, written by the render thread every frame (scene-local coordinates).
    var touchDissolveEnabled = true
    var touchA = SIMD4<Float>.zero
    var touchB = SIMD4<Float>.zero
    var touchAmount = SIMD2<Float>.zero
    var touchUp = SIMD3<Float>(0, 1, 0)
    var touchDrift: Float = 0
    // Immersion: off = original hard edge in your room; right-hand double pinch darkens the surroundings
    // and turns the edge into a soft, breathing border.
    var immersive = false
    var immersionStart: Double = -.infinity
    var immersionFromSurround: Float = 0
    var immersionFromEdge: Float = 0
    var placement = ScenePlacement()
    var placementRevision: UInt64 = 0
    var message = "Preparing scene…"
    var gestureMessage = "Preparing hand tracking…"
    var editing = true
    var handTrackingAvailable = false
    var pointCount = 0
    var sourceCount = 0
    var headPose: simd_float4x4?
    var stopRequested = false
    var finished = false
    var selectedRevealStyle: RevealStyle = .spread
    var activeRevealStyle: RevealStyle = .spread
    var revealUp = SIMD3<Float>(0, 1, 0)
    var revealLow: Float = -1
    var revealHigh: Float = 1
    var revealRadius: Float = 5
    var revealSamples = [SIMD3<Float>]()
    var revealOrigin = SIMD3<Float>.zero
    var revealDirection = SIMD3<Float>(0, 0, -1)
    var revealNearDepth: Float = 0
    var revealFarDepth: Float = 10
    var sceneRevealed = true
    var transitionDuration = CFG.defaultRevealDuration
    var visibilityStart: Double = -.infinity
    var visibilityFrom: Float = 1

    func visibilityProgress(at now: Double) -> Float {
        guard visibilityStart.isFinite else { return sceneRevealed ? 1 : 0 }
        let elapsed = Float(min(1, max(0, (now - visibilityStart) / transitionDuration)))
        let eased = elapsed * elapsed * (3 - 2 * elapsed)
        return visibilityFrom + ((sceneRevealed ? 1 : 0) - visibilityFrom) * eased
    }

    /// (surround darkness, edge softness), 0…1 with smootherstep easing; reversing mid-way continues from the current value.
    func immersionLevels(at now: Double) -> (surround: Float, edge: Float) {
        let target: Float = immersive ? 1 : 0
        guard immersionStart.isFinite else { return (target, target) }
        func ease(_ x: Double) -> Float {
            let s = Float(max(0, min(1, x))); return s * s * s * (s * (s * 6 - 15) + 10)
        }
        let dt = now - immersionStart
        return (immersionFromSurround + (target - immersionFromSurround) * ease(dt / CFG.immersionSurroundDuration),
                immersionFromEdge + (target - immersionFromEdge) * ease(dt / CFG.immersionEdgeDuration))
    }

    mutating func toggleImmersion(at now: Double) {
        let current = immersionLevels(at: now)
        immersionFromSurround = current.surround
        immersionFromEdge = current.edge
        immersive.toggle()
        immersionStart = now
    }

    mutating func setVisibility(_ visible: Bool, at now: Double) {
        visibilityFrom = visibilityProgress(at: now)
        // Only re-aim at an endpoint. Reversing mid-wave preserves every particle's phase.
        if visibilityFrom <= 0.0001 || visibilityFrom >= 0.9999 {
            activeRevealStyle = selectedRevealStyle
            captureRevealDirection()
        }
        visibilityStart = now
        sceneRevealed = visible
    }

    private mutating func captureRevealDirection() {
        guard let headPose else { return }
        let inverseModel = placement.matrix.inverse
        let origin = inverseModel * headPose.columns.3
        let forward = inverseModel * SIMD4<Float>(-headPose.columns.2.x,
                                                  -headPose.columns.2.y,
                                                  -headPose.columns.2.z, 0)
        let direction = SIMD3<Float>(forward.x, forward.y, forward.z)
        guard simd_length_squared(direction) > 0.000001 else { return }
        revealOrigin = SIMD3<Float>(origin.x, origin.y, origin.z)
        revealDirection = simd_normalize(direction)
        let up = inverseModel * SIMD4<Float>(0, 1, 0, 0)
        revealUp = simd_normalize(SIMD3<Float>(up.x, up.y, up.z))
        let depths = revealSamples.map { simd_dot($0 - revealOrigin, revealDirection) }
            .filter { $0.isFinite && $0 >= 0 }.sorted()
        let minimumSpan = 0.25 / placement.scale
        if !depths.isEmpty {
            revealNearDepth = depths[Int(Float(depths.count - 1) * 0.02)]
            revealFarDepth = max(revealNearDepth + minimumSpan,
                                 depths[Int(Float(depths.count - 1) * 0.98)])
        } else {
            revealNearDepth = 0
            revealFarDepth = 10 / placement.scale
        }
        let heights = revealSamples.map { simd_dot($0 - revealOrigin, revealUp) }.sorted()
        if !heights.isEmpty {
            revealLow = heights[Int(Float(heights.count - 1) * 0.02)]
            revealHigh = max(revealLow + minimumSpan,
                             heights[Int(Float(heights.count - 1) * 0.98)])
        }
        let center = revealOrigin + revealDirection * ((revealNearDepth + revealFarDepth) * 0.5)
            + revealUp * ((revealLow + revealHigh) * 0.5)
        let radii = revealSamples.map { point in
            let q = point - center
            return simd_length(q - revealUp * simd_dot(q, revealUp))
        }.sorted()
        if !radii.isEmpty {
            revealRadius = max(minimumSpan, radii[Int(Float(radii.count - 1) * 0.98)])
        }
    }
}

final class RenderBridge: Sendable {
    let state = Mutex(RenderState())
    let diagnostics = RenderDiagnostics()
}

@MainActor @Observable
final class ArtworkSession {
    var bridge = RenderBridge()
    var asset: URL?
    var status = "No scene yet. Import a .spz (and its .json) from the pipeline."
    var isImmersive = false
    var transitioning = false
    var importing = false
    var detail = SceneDetail.full
    private var windowCaptured = false
    private var immersiveCaptured = false
    private(set) var systemCaptureActive = false {
        didSet { bridge.state.withLock { $0.systemCaptureActive = systemCaptureActive } }
    }
    var recordingPriority = false {
        didSet { bridge.state.withLock { $0.recordingPriority = recordingPriority } }
    }
    var pointCount = 0
    var performanceText = "Render statistics appear after entering."
    var gestureMessage = "Edit mode · pinch with both hands to move, rotate and scale."
    private(set) var photoSpace = false
    private(set) var editing = true
    private(set) var handTrackingAvailable = false
    private(set) var sceneRevealed = true
    private(set) var sceneName: String?

    var viewingMode: ViewingMode = .mixed {
        didSet { bridge.state.withLock { $0.viewingMode = viewingMode } }
    }
    /// Immersion has switched the viewing space to full (VR). Entering: the room fades to black in
    /// mixed for 1.5 s, then the space switches to full. Leaving: back to mixed at once, and the black
    /// fades away over the room.
    private var immersionSynced = false
    var revealStyle: RevealStyle = RevealStyle(rawValue:
        UInt32(clamping: UserDefaults.standard.integer(forKey: "RevealStyle"))) ?? .spread {
        didSet {
            bridge.state.withLock { $0.selectedRevealStyle = revealStyle }
            UserDefaults.standard.set(Int(revealStyle.rawValue), forKey: "RevealStyle")
        }
    }
    var revealDuration = CFG.defaultRevealDuration {
        didSet { bridge.state.withLock { $0.transitionDuration = revealDuration } }
    }
    private var readingPlacement = false
    private var sidecar: SceneSidecar?
    var edgeFrame: [Float]? { sidecar?.validEdgeFrame }
    var windConfig: [Float]? { sidecar?.windConfig }
    var windEnabled = UserDefaults.standard.object(forKey: "WindFistEnabled") as? Bool ?? true {
        didSet {
            bridge.state.withLock { $0.windEnabled = windEnabled }
            UserDefaults.standard.set(windEnabled, forKey: "WindFistEnabled")
        }
    }
    /// Window fallback for the fist gesture: toggles the wind.
    func toggleWind() { bridge.state.withLock { $0.windRequest &+= 1 } }
    var touchDissolveEnabled = UserDefaults.standard.object(forKey: "TouchDissolveEnabled") as? Bool ?? true {
        didSet {
            bridge.state.withLock { $0.touchDissolveEnabled = touchDissolveEnabled }
            UserDefaults.standard.set(touchDissolveEnabled, forKey: "TouchDissolveEnabled")
        }
    }
    var placement = ScenePlacement() {
        didSet {
            guard !readingPlacement else { return }
            guard editing else { return }
            let value = placement
            bridge.state.withLock {
                $0.placement = value
                $0.placementRevision &+= 1
            }
        }
    }

    init() {
        if let saved = UserDefaults.standard.string(forKey: CFG.savedScenePathKey) {
            let url = URL(fileURLWithPath: saved)
            if FileManager.default.fileExists(atPath: url.path) {
                use(url)
                status = "Last imported scene is ready."
                return
            }
            UserDefaults.standard.removeObject(forKey: CFG.savedScenePathKey)
        }
        if let bundled = Self.bundledScene() {
            use(bundled)
            status = "Bundled scene \(bundled.lastPathComponent) is ready."
        }
    }

    /// The first .spz in the bundled SceneAssets folder, if any.
    static func bundledScene() -> URL? {
        (Bundle.main.urls(forResourcesWithExtension: "spz", subdirectory: CFG.sceneAssetsFolder) ?? [])
            .sorted { $0.lastPathComponent < $1.lastPathComponent }.first
    }

    /// Point the session at a scene file; reads `<name>.json` next to it if present.
    private func use(_ url: URL) {
        asset = url
        sceneName = url.deletingPathExtension().lastPathComponent
        sidecar = SceneSidecar.load(url.deletingPathExtension().appendingPathExtension("json"))
        photoSpace = sidecar != nil
        if photoSpace { viewingMode = .mixed }
        immersionSynced = false
        if !editing { beginEditing() }
        resetPlacement()
    }

    func updateCaptureState(_ captured: Bool, immersive: Bool) {
        if immersive { immersiveCaptured = captured } else { windowCaptured = captured }
        systemCaptureActive = windowCaptured || immersiveCaptured
    }

    func prepare() {
        bridge = RenderBridge()
        let position = placement
        let floorHeight = sidecar?.cameraHeight
        let edge = edgeFrame
        let wind = windConfig
        let shimmer: Float = sidecar?.hasWater == true ? 0.08 : 0
        bridge.state.withLock {
            $0.viewingMode = viewingMode
            $0.recordingPriority = recordingPriority
            $0.systemCaptureActive = systemCaptureActive
            $0.photoSpace = photoSpace
            $0.photoNeedsPlacement = photoSpace
            $0.photoFloorHeight = floorHeight
            $0.edgeFrame = edge
            $0.windConfig = wind
            $0.windShimmer = shimmer
            $0.windEnabled = windEnabled
            $0.touchDissolveEnabled = touchDissolveEnabled
            $0.placement = position
            $0.transitionDuration = revealDuration
            $0.selectedRevealStyle = revealStyle
            $0.activeRevealStyle = revealStyle
            $0.editing = editing
            $0.sceneRevealed = editing
            $0.visibilityFrom = editing ? 1 : 0
            $0.message = "Loading scene…"
        }
    }

    func stop() { bridge.state.withLock { $0.stopRequested = true } }

    func finishEditing() {
        guard editing else { return }
        editing = false
        sceneRevealed = false
        bridge.state.withLock {
            $0.editing = false
            $0.setVisibility(false, at: CACurrentMediaTime())
            $0.message = "Placement locked · pinch once to bring the scene in."
        }
    }

    func beginEditing() {
        guard !editing else { return }
        readingPlacement = true
        placement = bridge.state.withLock { $0.placement }
        readingPlacement = false
        editing = true
        sceneRevealed = true
        bridge.state.withLock {
            $0.editing = true
            $0.sceneRevealed = true
            $0.visibilityStart = -.infinity
            $0.visibilityFrom = 1
            $0.message = "Edit mode · pinch with both hands to adjust the scene."
        }
    }

    /// Window button and gesture share this: toggle immersion.
    func toggleImmersion() {
        bridge.state.withLock { $0.toggleImmersion(at: CACurrentMediaTime()) }
    }

    /// Fallback when hand tracking is unavailable.
    func toggleVisibility() {
        guard !editing, !handTrackingAvailable else { return }
        sceneRevealed.toggle()
        bridge.state.withLock { state in
            state.setVisibility(!state.sceneRevealed, at: CACurrentMediaTime())
            state.message = state.sceneRevealed ? "Scene appearing…" : "Scene gathering away…"
        }
    }

    func refresh() {
        guard isImmersive else { return }
        let snapshot = bridge.state.withLock { $0 }
        status = snapshot.message
        gestureMessage = snapshot.gestureMessage
        pointCount = snapshot.pointCount
        handTrackingAvailable = snapshot.handTrackingAvailable
        sceneRevealed = snapshot.sceneRevealed
        syncImmersionViewingMode(snapshot)
        performanceText = bridge.diagnostics.summary
        readingPlacement = true
        placement = snapshot.placement
        readingPlacement = false
        if snapshot.finished { isImmersive = false }
    }

    /// A right-hand double pinch (or the window button) also toggles the viewing space mixed <-> full.
    /// The panel picker is bound to `viewingMode`, so it stays in sync.
    private func syncImmersionViewingMode(_ snapshot: RenderState) {
        guard photoSpace, edgeFrame != nil else { immersionSynced = false; return }
        if snapshot.immersive, !immersionSynced {
            guard snapshot.immersionLevels(at: CACurrentMediaTime()).surround >= 0.999 else { return }
            immersionSynced = true
            if viewingMode != .virtual { viewingMode = .virtual }
        } else if !snapshot.immersive, immersionSynced {
            immersionSynced = false
            if viewingMode != .mixed { viewingMode = .mixed }
        }
    }

    func resetPlacement() {
        if photoSpace {
            placement = .photo
            let floorHeight = sidecar?.cameraHeight
            bridge.state.withLock {
                $0.photoFloorHeight = floorHeight
                $0.photoNeedsPlacement = true
            }
        } else {
            placement = .imported
        }
    }

    /// Import a scene: one splat file (.spz/.ply/.splat/.sog) and optionally its .json sidecar.
    func importFiles(_ sources: [URL]) async -> Bool {
        let splatTypes = ["spz", "ply", "splat", "sog"]
        guard let scene = sources.first(where: { splatTypes.contains($0.pathExtension.lowercased()) }) else {
            status = "Pick a .spz, .ply, .splat or .sog file (plus its .json, if you have one)."
            return false
        }
        let base = scene.deletingPathExtension().lastPathComponent
        let jsons = sources.filter { $0.pathExtension.lowercased() == "json" }
        let json = jsons.first { $0.deletingPathExtension().lastPathComponent == base } ?? jsons.first
        importing = true
        status = "Copying to this device…"
        defer { importing = false }
        do {
            let copied = try await Task.detached(priority: .userInitiated) {
                func copy(_ source: URL, to target: URL) throws {
                    let scoped = source.startAccessingSecurityScopedResource()
                    defer { if scoped { source.stopAccessingSecurityScopedResource() } }
                    try FileManager.default.copyItem(at: source, to: target)
                }
                let folder = URL.documentsDirectory.appending(path: "Scenes", directoryHint: .isDirectory)
                try? FileManager.default.removeItem(at: folder) // keep only the current import
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let target = folder.appending(path: base + "." + scene.pathExtension.lowercased())
                try copy(scene, to: target)
                if let json { try copy(json, to: folder.appending(path: base + ".json")) }
                return target
            }.value
            UserDefaults.standard.set(copied.path, forKey: CFG.savedScenePathKey)
            use(copied)
            status = photoSpace
                ? "Imported \(scene.lastPathComponent) with its sidecar. Opening…"
                : "Imported \(scene.lastPathComponent) (no sidecar: edge, immersion and wind are off). Opening…"
            return true
        } catch {
            status = "Import failed: \(error.localizedDescription)"
            return false
        }
    }
}
