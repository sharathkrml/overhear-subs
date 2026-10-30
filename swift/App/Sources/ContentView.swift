import AVKit
import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ContentViewModel()

    var body: some View {
        NavigationSplitView {
            transcriptPanel
        } detail: {
            playerPanel
        }
        .navigationTitle("OverhearSubs")
        .onAppear {
            if let p = ProcessInfo.processInfo.environment["LT_TEST_VIDEO"] {
                vm.load(url: URL(fileURLWithPath: p))
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button("Open Video…", systemImage: "folder") { vm.openVideo() }
                Button("Export SRT", systemImage: "square.and.arrow.up") { vm.exportSRT() }
                    .disabled(vm.cues.isEmpty)
            }
        }
    }

    private var playerPanel: some View {
        VStack(spacing: 0) {
            ZStack {
                if let player = vm.player {
                    VideoPlayer(player: player)
                } else {
                    Color.black.overlay(Text("Open a video").foregroundStyle(.white))
                }
                // Caption overlay
                VStack {
                    Spacer()
                    if let cue = currentCue {
                        Text(cue.text)
                            .font(.title3.bold())
                            .foregroundStyle(.white)
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

    private var transcriptPanel: some View {
        List(vm.cues) { cue in
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
        .frame(minWidth: 260)
    }

    private var currentCue: Cue? {
        guard let t = vm.player?.currentTime().seconds else { return nil }
        return vm.cues.first { $0.start <= t && t <= $0.end }
    }

    private func format(_ t: Double) -> String {
        String(format: "%02d:%02d", Int(t/60), Int(t.truncatingRemainder(dividingBy: 60)))
    }
}