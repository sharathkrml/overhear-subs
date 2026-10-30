import XCTest
@testable import OverhearSubs

final class ReflowTests: XCTestCase {
    func testShortTextSingleCue() {
        XCTAssertEqual(Reflow.cues(from: "hello world"), ["hello world"])
    }
    func testWrapsAtLimit() {
        let text = "this is a fairly long subtitle line that should wrap into two lines"
        let out = Reflow.cues(from: text)
        XCTAssertFalse(out.isEmpty)
        for cue in out {
            for line in cue.split(separator: "\n") {
                XCTAssertLessThanOrEqual(line.count, 42)
            }
        }
    }
    func testEmptyFallback() {
        XCTAssertEqual(Reflow.cues(from: ""), [""])
    }
}