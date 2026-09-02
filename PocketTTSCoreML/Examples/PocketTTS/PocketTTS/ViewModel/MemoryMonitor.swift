//
//  MemoryMonitor.swift
//  PocketTTS
//
//  Created by Sachin Desai on 5/3/26.
//

//
// Samples the process's physical footprint (the same metric iOS Jetsam
// uses to decide which app to kill under memory pressure) via
// `task_info(..., TASK_VM_INFO, ...)`. Used to report peak RAM across
// a generate / stream run in the stats card.
//

import Darwin
import Foundation

public enum MemoryMonitor {
    // Current process physical footprint in bytes. Returns 0 if the
    // kernel call fails (shouldn't happen in practice on iOS/macOS).
    public static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    $0,
                    &count
                )
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }

    public static func physFootprintMB() -> Double {
        Double(physFootprintBytes()) / (1024.0 * 1024.0)
    }
}

// Samples memory on a background Task at a fixed interval and tracks
// the peak observed. Start with `start()`, stop with `stop()` which
// returns the peak in MB.
@MainActor
public final class PeakMemoryTracker {
    private var task: Task<Void, Never>? = nil
    private(set) var peakMB: Double = 0

    public init() {}

    // Reset peak to the current footprint and begin polling every
    // `intervalMS` milliseconds.
    public func start(intervalMS: UInt64 = 100) {
        stop()
        peakMB = MemoryMonitor.physFootprintMB()
        task = Task { [weak self] in
            while !Task.isCancelled {
                let mb = MemoryMonitor.physFootprintMB()
                await MainActor.run {
                    guard let self = self else { return }
                    if mb > self.peakMB { self.peakMB = mb }
                }
                try? await Task.sleep(nanoseconds: intervalMS * 1_000_000)
            }
        }
    }

    // Cancel the polling task and return the peak observed since start().
    // Also samples one final time in case a burst happened between the
    // last poll and stop().
    @discardableResult
    public func stop() -> Double {
        task?.cancel()
        task = nil
        let final = MemoryMonitor.physFootprintMB()
        if final > peakMB { peakMB = final }
        return peakMB
    }
}
