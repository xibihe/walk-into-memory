import CompositorServices
import SwiftUI
import UniformTypeIdentifiers

/// Scene files plus the JSON sidecar written by the pipeline.
private let importTypes: [UTType] = ["spz", "ply", "splat", "sog"]
    .compactMap { UTType(filenameExtension: $0, conformingTo: .data) } + [.json]

@main
struct WalkIntoMemoryApp: App {
    @State private var artwork = ArtworkSession()
    @State private var immersionStyle: any ImmersionStyle = .mixed

    var body: some Scene {
        WindowGroup { ArtworkControls(artwork: artwork, immersionStyle: $immersionStyle) }
            .defaultSize(width: 460, height: 780)

        ImmersiveSpace(id: "WalkIntoMemory") {
            ArtworkImmersiveLayer(artwork: artwork)
        }
        .immersionStyle(selection: $immersionStyle, in: .mixed, .full)
    }
}

private struct ArtworkImmersiveLayer: CompositorContent {
    let artwork: ArtworkSession
    @Environment(\.isSceneCaptured) private var isSceneCaptured

    var body: some CompositorContent {
        CompositorLayer(configuration: ArtworkLayerConfiguration()) { layer in
            if let url = artwork.asset {
                ArtworkRenderer.start(layer: layer, url: url, bridge: artwork.bridge,
                                      detail: artwork.detail)
            }
        }
        .onChange(of: isSceneCaptured, initial: true) { _, captured in
            artwork.updateCaptureState(captured, immersive: true)
        }
        .onDisappear { artwork.updateCaptureState(false, immersive: true) }
    }
}

struct ArtworkControls: View {
    @Bindable var artwork: ArtworkSession
    @Binding var immersionStyle: any ImmersionStyle
    @State private var picking = false
    @State private var autoOpenAttempted = false
    @Environment(\.isSceneCaptured) private var isSceneCaptured
    @Environment(\.openImmersiveSpace) private var openSpace
    @Environment(\.dismissImmersiveSpace) private var dismissSpace

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Walk Into Memory")
                    .font(.system(size: 38, weight: .light))
                Text("Step into a photo, from the spot where it was taken.")
                    .foregroundStyle(.secondary)

                Text(artwork.editing
                     ? "Pinch with both hands to hold and adjust the scene, then finish editing."
                     : "Pinch once with either hand to show or hide the scene.")
                    .foregroundStyle(.secondary)

                Picker("Viewing space", selection: $artwork.viewingMode) {
                    ForEach(ViewingMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .onChange(of: artwork.viewingMode, initial: true) { _, mode in
                    immersionStyle = mode == .mixed ? .mixed : .full
                }

                if artwork.editing {
                    Button("Finish editing · lock placement") { artwork.finishEditing() }
                        .buttonStyle(.borderedProminent)
                        .disabled(!artwork.isImmersive || artwork.pointCount == 0)
                } else {
                    HStack {
                        Text("Viewing · placement locked").font(.callout)
                        Button("Edit again") { artwork.beginEditing() }
                    }
                    if !artwork.handTrackingAvailable {
                        Button(artwork.sceneRevealed ? "Hide scene" : "Show scene") {
                            artwork.toggleVisibility()
                        }
                    }
                }
                Picker("Reveal style", selection: $artwork.revealStyle) {
                    ForEach(RevealStyle.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.menu)
                Text(artwork.revealStyle.detail)
                    .font(.caption).foregroundStyle(.secondary)
                parameter("Reveal time", value: $artwork.revealDuration, range: 1...2, unit: " s")

                HStack {
                    Button("Import scene (.spz + .json)") { picking = true }
                        .disabled(artwork.transitioning || artwork.importing)
                    if artwork.isImmersive {
                        Button("Leave space") {
                            Task { @MainActor in
                                artwork.transitioning = true
                                defer { artwork.transitioning = false }
                                artwork.stop()
                                await dismissSpace()
                                artwork.isImmersive = false
                                artwork.status = "Left the space."
                            }
                        }
                        .disabled(artwork.transitioning)
                    } else if autoOpenAttempted && !artwork.transitioning {
                        Button("Open space") {
                            Task { await openArtworkSpace() }
                        }
                        .disabled(artwork.asset == nil || artwork.importing)
                    }
                }

                Text(artwork.status).font(.callout).textSelection(.enabled)
                if artwork.photoSpace {
                    Text("You start with the photo's original edge and your room around it. Pinch twice quickly with your right hand for immersion (the room fades to black and the edge turns soft and breathing); pinch twice again to return.")
                        .font(.callout).foregroundStyle(.secondary)
                    Button("Toggle immersion") { artwork.toggleImmersion() }
                        .disabled(!artwork.isImmersive || artwork.edgeFrame == nil)
                    Toggle("Reach dissolve (reach toward the memory and it slowly drifts apart)",
                           isOn: $artwork.touchDissolveEnabled)
                    if artwork.windConfig != nil {
                        Button("Wind (start / stop)") { artwork.toggleWind() }
                        Toggle("Fist wind (hold a fist ~0.3 s to start; again, or wait ~14 s, to stop)",
                               isOn: $artwork.windEnabled)
                    }
                } else if artwork.asset != nil {
                    Text("No sidecar .json was found, so the edge, immersion, reach dissolve and wind effects are off. Run the pipeline to create one.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Placement").font(.headline)
                    if artwork.editing {
                        Toggle("Flip upside down", isOn: $artwork.placement.flipVertical)
                        HStack {
                            Button("← Left") { artwork.placement.position.x -= CFG.positionStep }
                            Button("Right →") { artwork.placement.position.x += CFG.positionStep }
                            Button("↑ Up") { artwork.placement.position.y += CFG.positionStep }
                            Button("Down ↓") { artwork.placement.position.y -= CFG.positionStep }
                        }
                        HStack {
                            Button("Closer") { artwork.placement.position.z += CFG.positionStep }
                            Button("Farther") { artwork.placement.position.z -= CFG.positionStep }
                            Button("Smaller") { artwork.placement.scale = max(CFG.minimumScale, artwork.placement.scale / 1.1) }
                            Button("Larger") { artwork.placement.scale = min(CFG.maximumScale, artwork.placement.scale * 1.1) }
                        }
                    }
                    Text(String(format: "Position %.2f / %.2f / %.2f m · scale %.2f×",
                                artwork.placement.position.x, artwork.placement.position.y,
                                artwork.placement.position.z, artwork.placement.scale))
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    if artwork.editing {
                        Button(artwork.photoSpace ? "Stand at the original camera again" : "Reset placement") {
                            artwork.resetPlacement()
                        }
                    }
                    Text(artwork.gestureMessage)
                        .font(.callout).foregroundStyle(.secondary)
                }
                Toggle("Recording priority", isOn: $artwork.recordingPriority)
                Text("Applied automatically while recording; turn it on early if you like.")
                    .font(.caption).foregroundStyle(.secondary)
                Picker("Scene detail", selection: $artwork.detail) {
                    ForEach(SceneDetail.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(artwork.isImmersive || artwork.transitioning)
                Text("Keeps the Gaussians that contribute most to the image. Pick Balanced if a large scene stutters. Takes effect the next time you enter.")
                    .font(.caption).foregroundStyle(.secondary)
                if artwork.isImmersive {
                    Text("\(artwork.pointCount.formatted()) Gaussians · \(artwork.performanceText)")
                        .font(.caption).monospacedDigit()
                    Text("Submission rate is a debugging aid, not the headset refresh rate.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: isSceneCaptured, initial: true) { _, captured in
            artwork.updateCaptureState(captured, immersive: false)
        }
        .fileImporter(isPresented: $picking, allowedContentTypes: importTypes,
                      allowsMultipleSelection: true) {
            if case .success(let files) = $0, !files.isEmpty {
                Task { await importAndOpen(files) }
            }
        }
        .task {
            guard !autoOpenAttempted else { return }
            autoOpenAttempted = true
            await openArtworkSpace()
        }
        .task {
            while !Task.isCancelled {
                artwork.refresh()
                try? await Task.sleep(for: .milliseconds(400))
            }
        }
    }

    @MainActor
    private func importAndOpen(_ files: [URL]) async {
        guard !artwork.transitioning else { return }
        artwork.transitioning = true
        if artwork.isImmersive {
            artwork.stop()
            await dismissSpace()
            artwork.isImmersive = false
        }
        let imported = await artwork.importFiles(files)
        immersionStyle = artwork.viewingMode == .mixed ? .mixed : .full
        artwork.transitioning = false
        if imported { await openArtworkSpace() }
    }

    @MainActor
    private func openArtworkSpace() async {
        guard artwork.asset != nil, !artwork.isImmersive, !artwork.transitioning else { return }
        artwork.transitioning = true
        defer { artwork.transitioning = false }
        artwork.prepare()
        switch await openSpace(id: "WalkIntoMemory") {
        case .opened: artwork.isImmersive = true
        case .userCancelled: artwork.status = "Opening the space was cancelled."
        case .error: artwork.status = "The space could not be opened."
        @unknown default: artwork.status = "The space is not available."
        }
    }

    private func parameter(_ title: String, value: Binding<Double>, range: ClosedRange<Double>, unit: String) -> some View {
        HStack {
            Text(title).frame(width: 90, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.2f%@", value.wrappedValue, unit))
                .monospacedDigit().frame(width: 76, alignment: .trailing)
        }
    }
}
