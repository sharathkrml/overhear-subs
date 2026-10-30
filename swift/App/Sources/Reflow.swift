import Foundation

/// Pure helpers shared by the view model and tests.
enum Reflow {
    /// Split text into subtitle cues of at most `limit` chars per line,
    /// two lines per cue, reflowing on word boundaries.
    static func cues(from text: String, limit: Int = 42) -> [String] {
        let words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var lines: [String] = []
        var current = ""
        for w in words {
            if current.isEmpty { current = w }
            else if current.count + 1 + w.count <= limit { current += " " + w }
            else { lines.append(current); current = w }
        }
        if !current.isEmpty { lines.append(current) }
        var out: [String] = []
        var i = 0
        while i < lines.count {
            if i + 1 < lines.count {
                out.append(lines[i] + "\n" + lines[i + 1])
                i += 2
            } else {
                out.append(lines[i])
                i += 1
            }
        }
        if out.isEmpty { out = [text] }
        return out
    }
}