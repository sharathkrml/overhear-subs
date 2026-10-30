import AppKit
import AVFoundation
import Foundation
import SwiftUI
import Transcript
import WhisperKit

/// A subtitle cue: text + time span.
struct Cue: Identifiable, Equatable {
    let id = UUID()
    let start: Double
    let end: Double
    let text: String
}

@MainActor
final class ContentViewModel: ObservableObject {
    @Published var videoURL: URL?
    @Published var player: AVPlayer?
    @Published var cues: [Cue] = []
    @Published var isTranscribing = false
    @Published var progress: Double = 0
    @Published var status = "Open a video to begin."

    private var whisper: WhisperKit?
    private var transcriptionTask: Task<Void, Never>?

    func openVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .audio]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url: url)
    }

    func load(url: URL) {
        transcriptionTask?.cancel()
        videoURL = url
        player = AVPlayer(url: url)
        cues = []
        isTranscribing = true
        progress = 0
        status = "Preparing transcription…"
        transcriptionTask = Task { await transcribe(url: url) }
    }

    private func transcribe(url: URL) async {
        do {
            if whisper == nil {
                status = "Loading Whisper model…"
                let config = WhisperKitConfig(model: "large-v3-v20240930_626MB")
                whisper = try await WhisperKit(config)
            }
            status = "Transcribing…"
            let results = try await whisper!.transcribe(
                audioPaths: [url.path(percentEncoded: false)],
                decodeOptions: DecodingOptions(task: .transcribe, wordTimestamps: true)
            )
            let words = (results.first ?? nil)?.flatMap { $0.allWords }.map {
                TimedWord(text: $0.word, start: Double($0.start), end: Double($0.end))
            } ?? []
            let sentences = Sentence.sentences(from: words)
            // Reflow each sentence into subtitle cues (<=2 lines, ~42 chars).
            var out: [Cue] = []
            for s in sentences {
                let chunks = Reflow.cues(from: s.text)
                // split time proportionally
                let totalChars = max(1, s.text.count)
                var cursor = s.start
                for chunk in chunks {
                    let dur = s.duration * Double(chunk.count) / Double(totalChars)
                    out.append(Cue(start: cursor, end: min(s.end, cursor + dur), text: chunk))
                    cursor += dur
                }
            }
            cues = out.sorted { $0.start < $1.start }
            isTranscribing = false
            status = "Transcribed \(cues.count) cues."
        } catch {
            isTranscribing = false
            status = "Transcription failed: \(error.localizedDescription)"
        }
    }

    func exportSRT() {
        guard let url = videoURL else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = url.deletingPathExtension().lastPathComponent + ".srt"
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        var s = ""
        for (i, c) in cues.enumerated() {
            s += "\(i+1)\n\(formatTime(c.start)) --> \(formatTime(c.end))\n\(c.text)\n\n"
        }
        try? s.write(to: dest, atomically: true, encoding: .utf8)
    }

    private func formatTime(_ t: Double) -> String {
        let h = Int(t/3600); let m = Int((t.truncatingRemainder(dividingBy: 3600))/60)
        let s = t.truncatingRemainder(dividingBy: 60)
        return String(format: "%02d:%02d:%06.3f", h, m, s).replacingOccurrences(of: ".", with: ",")
    }
}