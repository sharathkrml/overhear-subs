import AVKit
import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ContentViewModel()
    @State private var showChapters = false

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            playerPanel
        }
        .navigationTitle("OverhearSubs")
        .onAppear {
            if let p = ProcessInfo.processInfo.environment["LT_TEST_VIDEO"] {
                vm.openPath(p)
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button("Open Video…", systemImage: "folder") { vm.openVideo() }
                    .keyboardShortcut("o", modifiers: .command)
                Button("Export SRT", systemImage: "square.and.arrow.up") { vm.export(fmt: "srt") }
                    .disabled(vm.cues.isEmpty)
                Toggle(isOn: Binding(get: { vm.ttsEnabled }, set: { _ in vm.toggleTTS() })) {
                    Label("Read-along", systemImage: "speaker.wave.2")
                }
                Button("Chapters", systemImage: "list.bullet") { showChapters.toggle() }
                    .disabled(vm.chapters.isEmpty)
            }
        }
        .onKeyPress(.space) {
            togglePlay()
            return .handled
        }
        .sheet(isPresented: $showChapters) {
            ChaptersSheet(chapters: vm.chapters) { ch in
                vm.player?.seek(to: CMTime(seconds: ch.start, preferredTimescale: 600))
                showChapters = false
            }
        }
    }

    private var sidebar: some View {
        List {
            Section("Transcript") {
                ForEach(vm.cues) { cue in
                    Button {
                        vm.player?.seek(to: CMTime(seconds: cue.start, preferredTimescale: 600))
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(format(cue.start)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            Text(cue.text).font(.body)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(minWidth: 260)
    }

    private var playerPanel: some View {
        VStack(spacing: 0) {
            ZStack {
                if let player = vm.player {
                    VideoPlayer(player: player)
                } else {
                    Color.black.overlay(Text("Open a video").foregroundStyle(.white))
                }
                VStack {
                    Spacer()
                    if let cue = currentCue {
                        Text(cue.text)
                            .font(.title3.bold())
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                            .padding(8)
                            .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
                            .padding(.bottom, 40)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            HStack {
                Text(vm.status).font(.caption).foregroundStyle(.secondary)
                if vm.isTranscribing { ProgressView(value: vm.progress).frame(width: 120) }
            }
            .padding(8)
        }
    }

    private var currentCue: Cue? {
        guard let t = vm.player?.currentTime().seconds else { return nil }
        return vm.cues.first { $0.start <= t && t <= $0.end }
    }

    private func togglePlay() {
        guard let p = vm.player else { return }
        if p.rate == 0 { p.play() } else { p.pause() }
    }

    private func format(_ t: Double) -> String {
        String(format: "%02d:%02d", Int(t/60), Int(t.truncatingRemainder(dividingBy: 60)))
    }
}

struct ChaptersSheet: View {
    let chapters: [Chapter]
    let onSelect: (Chapter) -> Void
    var body: some View {
        NavigationStack {
            List(chapters) { ch in
                Button {
                    onSelect(ch)
                } label: {
                    VStack(alignment: .leading) {
                        Text(ch.title).font(.headline)
                        Text(String(format: "%02d:%02d", Int(ch.start/60), Int(ch.start.truncatingRemainder(dividingBy: 60))))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
            }
            .navigationTitle("Chapters")
        }
        .frame(minWidth: 400, minHeight: 300)
    }
}