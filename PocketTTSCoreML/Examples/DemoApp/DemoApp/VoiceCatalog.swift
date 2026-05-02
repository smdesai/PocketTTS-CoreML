//
// VoiceCatalog.swift
//
// Enumerates voice .safetensors files that were copied into the app bundle
// by prepare_resources.sh. A voice file's filename stem is used as the id;
// the display name is computed by splitting on '_' and title-casing each
// part ("peter_yearsley" -> "Peter Yearsley").
//

import Foundation

public struct VoiceEntry: Identifiable, Hashable, Sendable {
    /// Filename stem (e.g. "bill_boerst").
    public let id: String
    /// Human-readable title-cased name (e.g. "Bill Boerst").
    public let displayName: String
    /// Absolute URL inside the app bundle.
    public let url: URL
}

public enum VoiceCatalog {

    /// Return all `.safetensors` voices bundled in `Resources/Voices/`,
    /// sorted alphabetically by displayName. Falls back to an empty array
    /// if the folder was not seeded (prepare_resources.sh not run).
    public static func bundled() -> [VoiceEntry] {
        guard let voicesDir = resolveVoicesDir() else { return [] }
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: voicesDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ) else { return [] }

        var entries: [VoiceEntry] = []
        entries.reserveCapacity(items.count)
        for url in items where url.pathExtension == "safetensors" {
            let stem = url.deletingPathExtension().lastPathComponent
            entries.append(VoiceEntry(
                id: stem,
                displayName: titleCase(stem),
                url: url
            ))
        }
        entries.sort { $0.displayName < $1.displayName }
        return entries
    }

    /// "peter_yearsley" -> "Peter Yearsley", "alba" -> "Alba".
    static func titleCase(_ stem: String) -> String {
        stem.split(separator: "_")
            .map { part -> String in
                guard let first = part.first else { return String(part) }
                return first.uppercased() + part.dropFirst().lowercased()
            }
            .joined(separator: " ")
    }

    /// Try a few common layouts. xcodegen folder-reference resources land
    /// in `Bundle.main.resourceURL/Voices/`; stray file-by-file copies land
    /// at the bundle root. Try both.
    private static func resolveVoicesDir() -> URL? {
        guard let base = Bundle.main.resourceURL else { return nil }
        let fm = FileManager.default
        let candidate = base.appendingPathComponent("Voices", isDirectory: true)
        if fm.fileExists(atPath: candidate.path) { return candidate }
        // Fallback: bundle root if .safetensors landed there directly.
        if let items = try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil
        ), items.contains(where: { $0.pathExtension == "safetensors" }) {
            return base
        }
        return nil
    }
}
