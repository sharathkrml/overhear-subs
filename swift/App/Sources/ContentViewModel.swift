import AppKit
import AVFoundation
import Foundation
import SwiftUI

struct Cue: Identifiable, Equatable {
    let id = UUID()
    let start: Double
    let end: Double
    let text: String
}

struct Chapter: Identifiable, Equatable {
    let id = UUID()
    let start: Double
    let title: String
}

@MainActor
final class ContentViewModel: ObservableObject {
    @Published var player: AVPlayer?
    @Published var cues: [Cue] = []
    @Published var chapters: [Chapter] = []
    @Published var isTranscribing = false
    @Published var progress: Double = 0
    @Published var status = "Starting backend…"
    @Published var ttsEnabled = false
    @Published var duration: Double = 0

    private let port = 8765
    private var backendProcess: Process?
    private var ws: URLSessionWebSocketTask?
    private var pollTask: Task<Void, Never>?
    private var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }
    private var mediaURL: URL { URL(string: "http://127.0.0.1:\(port)/media")! }

    init() {
        Task { await startBackend() }
    }

    deinit {
        backendProcess?.terminate()
    }

    private func backendDir() -> URL {
        // Bundled backend first, then dev repo root.
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("backend"),
           FileManager.default.fileExists(atPath: bundled.appendingPathComponent("app.py").path) {
            return bundled
        }
        // dev: try a few candidate repo roots
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            URL(fileURLWithPath: "/Users/sharath/Documents/Github/live-transcribe"),
            home.appendingPathComponent("Documents/Github/live-transcribe"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).deletingLastPathComponent(),
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c.appendingPathComponent("app.py").path) {
                return c
            }
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private func pythonPath() -> String {
        if let env = ProcessInfo.processInfo.environment["LT_BACKEND_PYTHON"] { return env }
        let venv = backendDir().appendingPathComponent(".venv/bin/python3")
        if FileManager.default.isExecutableFile(atPath: venv.path) { return venv.path }
        let bundled = backendDir().appendingPathComponent("venv/bin/python3")
        if FileManager.default.isExecutableFile(atPath: bundled.path) { return bundled.path }
        // system
        return "/usr/bin/python3"
    }

    private func startBackend() async {
        // If already healthy, reuse.
        if await healthy() {
            status = "Backend ready."
            connect()
            return
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath())
        proc.arguments = ["-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "\(port)"]
        proc.currentDirectoryURL = backendDir()
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            backendProcess = proc
        } catch {
            status = "Failed to launch backend: \(error.localizedDescription)"
            return
        }
        // Wait for readiness
        for _ in 0..<60 {
            try? await Task.sleep(nanoseconds: 500_000_000)
            if await healthy() { break }
        }
        if await healthy() {
            status = "Backend ready."
            connect()
        } else {
            status = "Backend did not start."
        }
    }

    private func healthy() async -> Bool {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/state")!)
        req.timeoutInterval = 2
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            return (resp as? HTTPURLResponse)?.statusCode == 200
        } catch { return false }
    }

    private func connect() {
        let url = URL(string: "ws://127.0.0.1:\(port)/ws")!
        let task = URLSession.shared.webSocketTask(with: url)
        ws = task
        task.resume()
        receiveLoop()
        pollTask = Task { await poll() }
        startPlayheadTimer()
    }

    private func startPlayheadTimer() {
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let p = self.player else { return }
                let t = p.currentTime().seconds
                if t.isFinite { self.sendPlayhead(t) }
            }
        }
    }

    private func receiveLoop() {
        guard let ws else { return }
        ws.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let msg):
                    if case .string(let s) = msg, let data = s.data(using: .utf8),
                       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        self.handleState(obj)
                    }
                    self.receiveLoop()
                case .failure:
                    break
                }
            }
        }
    }

    private func handleState(_ obj: [String: Any]) {
        if let sched = obj["playhead"] as? [String: Any] {} // no
        // state payload is flat-ish: playhead etc at top level after unwrap? WS sends {"type":"state", ..._state_payload()}
        if let chunksTotal = obj["chunks_total"] as? Int {
            let done = obj["chunks_done"] as? Int ?? 0
            if chunksTotal > 0 { progress = Double(done) / Double(chunksTotal) }
            isTranscribing = !(obj["finished"] as? Bool ?? false)
        }
        if let ahead = obj["ahead"] as? Double {}
        if let dur = obj["duration"] as? Double { duration = dur }
        if let chaptersObj = obj["chapters"] as? [String: Any],
           let list = chaptersObj["chapters"] as? [[String: Any]] {
            chapters = list.compactMap { c in
                guard let start = c["start"] as? Double else { return nil }
                let title = c["title"] as? String ?? "Chapter"
                let end = c["end"] as? Double ?? start
                return Chapter(start: start, title: title)
            }
        }
        if let tts = obj["tts"] as? [String: Any], let en = tts["enabled"] as? Bool {
            ttsEnabled = en
        }
        // media duration from playback? 
        if let media = obj["media"] as? [String: Any], let d = media["duration"] as? Double {
            duration = d
        }
        if let prep = obj["prep"] as? [String: Any] {
            if let stage = prep["stage"] as? String { status = stage }
        } else if let err = obj["error"] as? String {
            status = err
        } else {
            status = isTranscribing ? "Transcribing…" : "Ready"
        }
    }

    private func poll() async {
        while true {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            await fetchCues()
            // also poll state as fallback
            await fetchState()
        }
    }

    private func fetchState() async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/state") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                await MainActor.run { self.handleState(obj) }
            }
        } catch {}
    }

    private func fetchCues() async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/cues") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let items = obj["items"] as? [[String: Any]] {
                let newCues = items.compactMap { d -> Cue? in
                    guard let s = d["start"] as? Double, let e = d["end"] as? Double else { return nil }
                    let text = (d["target"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? (d["source"] as? String ?? "")
                    return Cue(start: s, end: e, text: text)
                }
                await MainActor.run {
                    self.cues = newCues.sorted { $0.start < $1.start }
                }
            }
        } catch {}
    }

    // MARK: - Actions

    func openVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .audio]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        openPath(url.path)
    }

    func openPath(_ path: String) {
        // Wait for backend readiness before opening.
        Task {
            for _ in 0..<60 {
                if await healthy() { break }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            await MainActor.run {
                self.isTranscribing = true
                self.cues = []
            }
            var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/open")!)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try? JSONSerialization.data(withJSONObject: ["path": path])
            do {
                let (data, _) = try await URLSession.shared.data(for: req)
                if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    print("open result: \(obj)")
                }
                await MainActor.run {
                    self.player = AVPlayer(url: self.mediaURL)
                    self.player?.play()
                }
            } catch {
                await MainActor.run { self.status = "Open failed: \(error.localizedDescription)" }
            }
        }
        return
    }

    func seek(_ t: Double) {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/seek")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["time": t])
        Task { _ = try? await URLSession.shared.data(for: req) }
        player?.seek(to: CMTime(seconds: t, preferredTimescale: 600))
    }

    func toggleTTS() {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/tts")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["enabled": !ttsEnabled])
        Task { _ = try? await URLSession.shared.data(for: req) }
    }

    func export(fmt: String = "srt") {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/export?fmt=\(fmt)") else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "transcript.\(fmt)"
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        Task {
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                try data.write(to: dest)
            } catch {}
        }
    }

    func sendPlayhead(_ t: Double) {
        ws?.send(.string("{\"type\":\"playhead\",\"time\":\(t)}"), completionHandler: { _ in })
    }
}