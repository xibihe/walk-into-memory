import Foundation

// Bound CPU/GPU overlap without accumulating an unbounded queue of old head poses.
final class RenderInFlightGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var count = 0

    func acquire(limit: Int) {
        condition.lock()
        defer { condition.unlock() }
        while count >= max(1, limit) { condition.wait() }
        count += 1
    }

    func release() {
        condition.lock()
        precondition(count > 0)
        count -= 1
        condition.broadcast()
        condition.unlock()
    }

    var activeCount: Int {
        condition.lock()
        defer { condition.unlock() }
        return count
    }
}
