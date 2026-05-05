//
// ModelDownloader.swift
//
// Downloads a language's CoreML artifacts + tokenizer + voices from the
// Hugging Face Hub into the app's Application Support directory on first
// use. Subsequent launches detect the on-disk cache and skip the download.
//
// On-device layout (writable):
//
//   Application Support/pocket-tts-coreml/<configName>/
//       ├── flow_lm_main.mlmodelc/...
//       ├── flow_lm_flow.mlmodelc/...
//       ├── flow_lm_prefill.mlmodelc/...
//       ├── mimi_decoder.mlmodelc/...
//       ├── mimi_encoder.mlmodelc/...
//       ├── text_conditioner.mlmodelc/...
//       ├── flow_lm_bos_emb.safetensors
//       ├── speaker_proj.safetensors         (optional — only if cloning supported)
//       ├── mimi_decoder.state_layout.json
//       ├── tokenizer.model
//       ├── voices/*.safetensors
//       └── .complete                         (sentinel — written last)
//
// The `.complete` sentinel gates "is this language installed?" checks so a
// partial/interrupted download is treated as "not installed" on the next
// launch (we start over rather than resuming — HF's LFS redirects make
// robust resume harder than a fresh retry).
//

import Foundation

/// Repository-level configuration for the Hugging Face download source.
public enum HFRepo {
    /// Owner + repo name on huggingface.co. Files are resolved from
    /// `https://huggingface.co/<id>/resolve/main/<path>`.
    public static let id: String = "smdesai/pocket-tts-coreml"
    /// Git revision. "main" tracks the latest commit on the default branch.
    public static let revision: String = "main"
}

/// Download progress snapshot emitted on the main actor during a language
/// download. Consumers render a progress bar + byte counter from this.
public struct DownloadProgress: Sendable, Equatable {
    /// Bytes completed so far (sum across all files).
    public let bytesCompleted: Int64
    /// Total bytes expected (sum across all files). Zero while the file
    /// list is still being fetched, since HF's tree API doesn't include
    /// sizes for LFS-pointered files cheaply — we only know totals once
    /// the first range response comes back, so we report total as
    /// "currently observed max" which may grow mid-download. UI should
    /// tolerate this by showing a percentage only when total > 0.
    public let bytesTotal: Int64
    /// 1-indexed position of the file currently downloading. Zero while
    /// the file list is still being built.
    public let currentFileIndex: Int
    /// Total number of files to download for this language.
    public let totalFiles: Int
    /// Human-readable name of the file currently downloading (relative
    /// path under the language directory). Empty while the list is being
    /// built.
    public let currentFileName: String

    public static let zero = DownloadProgress(
        bytesCompleted: 0, bytesTotal: 0,
        currentFileIndex: 0, totalFiles: 0,
        currentFileName: ""
    )
}

/// Errors surfaced by the downloader. All cases include a user-facing
/// description via `LocalizedError`.
public enum ModelDownloadError: LocalizedError {
    case networkError(String)
    case httpStatus(Int, String)
    case invalidResponse(String)
    case noFilesFound(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .networkError(let m): return "Network error: \(m)"
        case .httpStatus(let code, let m): return "HTTP \(code): \(m)"
        case .invalidResponse(let m): return "Invalid response: \(m)"
        case .noFilesFound(let m): return "No files found: \(m)"
        case .cancelled: return "Download cancelled"
        }
    }
}

/// Downloads a language bundle from Hugging Face into Application Support.
///
/// Usage:
///   let dl = ModelDownloader()
///   try await dl.ensureLanguageInstalled("english") { progress in
///       // update UI
///   }
public final class ModelDownloader: Sendable {

    public init() {}

    // MARK: - Paths

    /// Root directory where all language bundles live on-device. Created
    /// on demand; survives app restarts; excluded from iCloud backup.
    public static func rootDirectory() -> URL {
        let fm = FileManager.default
        let base: URL
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            base = appSupport
        } else {
            base = fm.temporaryDirectory
        }
        let root = base.appendingPathComponent("pocket-tts-coreml", isDirectory: true)
        if !fm.fileExists(atPath: root.path) {
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
            // Mark the whole tree as "do not back up" — these files are
            // large, derived, and re-downloadable on demand.
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var rootMutable = root
            try? rootMutable.setResourceValues(values)
        }
        return root
    }

    /// Directory for `configName` (e.g. "english", "french_24l"). Created
    /// on demand. Matches the folder layout in the Hugging Face repo.
    public static func languageDirectory(configName: String) -> URL {
        let dir = rootDirectory().appendingPathComponent(configName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true
            )
        }
        return dir
    }

    /// True if `configName` has been fully downloaded previously (sentinel
    /// file exists). Does NOT re-validate every artifact — the sentinel is
    /// only written after every file has been successfully written.
    public static func isInstalled(configName: String) -> Bool {
        let sentinel = languageDirectory(configName: configName)
            .appendingPathComponent(".complete")
        return FileManager.default.fileExists(atPath: sentinel.path)
    }

    // MARK: - Public API

    /// Ensure the bundle for `configName` is installed locally. If it
    /// already is (sentinel present), returns immediately. Otherwise
    /// fetches the file list from the HF tree API and downloads each file
    /// in sequence. Progress is reported via `onProgress` on the main
    /// actor.
    @MainActor
    public func ensureLanguageInstalled(
        configName: String,
        onProgress: @escaping @Sendable @MainActor (DownloadProgress) -> Void = { _ in }
    ) async throws {
        if Self.isInstalled(configName: configName) { return }

        onProgress(DownloadProgress.zero)

        let files = try await fetchFileList(configName: configName)
        guard !files.isEmpty else {
            throw ModelDownloadError.noFilesFound(
                "HF repo \(HFRepo.id) returned no files under \(configName)/"
            )
        }

        // Sum of tree-API sizes. LFS entries carry their resolved size at
        // the top level of the tree response so this is usually accurate.
        // If a file reports size=0, the progress bar stays indeterminate
        // until the download finishes — acceptable since it's rare.
        let totalBytes = files.reduce(Int64(0)) { $0 + max($1.size, 0) }
        var bytesCompleted: Int64 = 0
        let destRoot = Self.languageDirectory(configName: configName)
        let fm = FileManager.default
        let totalFiles = files.count

        for (idx, file) in files.enumerated() {
            try Task.checkCancellation()
            let destURL = destRoot.appendingPathComponent(file.relativePath)
            let fileIndex = idx + 1
            let fileName = file.relativePath

            // If this file is already on-disk at the expected size, skip
            // it. Lets us resume a partial install after the user killed
            // the app mid-download.
            if let attrs = try? fm.attributesOfItem(atPath: destURL.path),
                let size = attrs[.size] as? Int64, file.size > 0, size == file.size
            {
                bytesCompleted += size
                onProgress(
                    DownloadProgress(
                        bytesCompleted: bytesCompleted, bytesTotal: totalBytes,
                        currentFileIndex: fileIndex, totalFiles: totalFiles,
                        currentFileName: fileName
                    ))
                continue
            }

            try fm.createDirectory(
                at: destURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let url = Self.downloadURL(
                configName: configName, relativePath: file.relativePath
            )
            let fileStart = bytesCompleted
            try await downloadFile(
                from: url, to: destURL,
                onBytes: { @Sendable written in
                    Task { @MainActor in
                        onProgress(
                            DownloadProgress(
                                bytesCompleted: fileStart + written,
                                bytesTotal: totalBytes,
                                currentFileIndex: fileIndex, totalFiles: totalFiles,
                                currentFileName: fileName
                            ))
                    }
                }
            )
            let finalSize =
                (try? fm.attributesOfItem(atPath: destURL.path)[.size] as? Int64) ?? file.size
            bytesCompleted = fileStart + max(finalSize, 0)
            onProgress(
                DownloadProgress(
                    bytesCompleted: bytesCompleted, bytesTotal: totalBytes,
                    currentFileIndex: fileIndex, totalFiles: totalFiles,
                    currentFileName: fileName
                ))
        }

        // Write the sentinel — the ONLY signal used to decide "is this
        // language installed" on subsequent launches.
        try Data().write(
            to: destRoot.appendingPathComponent(".complete"), options: .atomic
        )
    }

    // MARK: - Internal

    fileprivate struct RemoteFile: Sendable {
        /// Path relative to the language directory, e.g.
        /// "flow_lm_main.mlmodelc/weights/weight.bin" or
        /// "voices/alba.safetensors". Never starts with `/`.
        let relativePath: String
        /// Size in bytes as reported by the HF tree API. For LFS files
        /// this is already the resolved size (not a pointer size). 0 if
        /// the API didn't report it for some reason.
        let size: Int64
    }

    /// Build the `resolve` URL for a single file. Pattern:
    /// https://huggingface.co/<repo>/resolve/<rev>/<configName>/<relPath>
    fileprivate static func downloadURL(
        configName: String, relativePath: String
    ) -> URL {
        var u = URL(string: "https://huggingface.co")!
            .appendingPathComponent(HFRepo.id)
            .appendingPathComponent("resolve")
            .appendingPathComponent(HFRepo.revision)
            .appendingPathComponent(configName)
        for segment in relativePath.split(separator: "/") {
            u.appendPathComponent(String(segment))
        }
        return u
    }

    /// Recursively enumerate every file under `<repo>/<configName>/` via
    /// the HF tree API. Returns flat list of file paths relative to the
    /// language directory.
    fileprivate func fetchFileList(configName: String) async throws -> [RemoteFile] {
        var out: [RemoteFile] = []
        try await walkTree(configName: configName, subpath: "", into: &out)
        out.sort { $0.relativePath < $1.relativePath }
        return out
    }

    /// HF tree API entry shape. We only decode the fields we use.
    private struct TreeEntry: Decodable {
        let type: String  // "file" or "directory"
        let path: String  // full path from repo root, e.g. "english/voices/alba.safetensors"
        let size: Int64?  // bytes; present for files, absent for directories
    }

    private func walkTree(
        configName: String, subpath: String, into out: inout [RemoteFile]
    ) async throws {
        var u = URL(string: "https://huggingface.co/api/models")!
            .appendingPathComponent(HFRepo.id)
            .appendingPathComponent("tree")
            .appendingPathComponent(HFRepo.revision)
            .appendingPathComponent(configName)
        for segment in subpath.split(separator: "/") {
            u.appendPathComponent(String(segment))
        }
        var comps = URLComponents(url: u, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        let apiURL = comps.url!

        var req = URLRequest(url: apiURL)
        req.httpMethod = "GET"
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await urlSession().data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw ModelDownloadError.invalidResponse("Non-HTTP response from \(apiURL)")
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw ModelDownloadError.httpStatus(
                http.statusCode, "Failed to list files at \(apiURL.path)")
        }

        let entries: [TreeEntry]
        do {
            entries = try JSONDecoder().decode([TreeEntry].self, from: data)
        } catch {
            throw ModelDownloadError.invalidResponse(
                "Could not parse tree response: \(error.localizedDescription)"
            )
        }

        // `path` is repo-rooted (e.g. "english/voices/alba.safetensors").
        // Strip the `configName/` prefix to get a language-relative path.
        let prefix = configName + "/"
        for entry in entries where entry.type == "file" {
            guard entry.path.hasPrefix(prefix) else { continue }
            let rel = String(entry.path.dropFirst(prefix.count))
            if rel.isEmpty { continue }
            out.append(RemoteFile(relativePath: rel, size: entry.size ?? 0))
        }
    }

    /// URLSession configured for large downloads: no caching (we stream
    /// straight to disk) and a generous resource timeout.
    private func urlSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60 * 60
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    /// Stream a single URL to `destURL`, reporting incremental written
    /// byte counts via `onBytes`. Uses `URLSession.bytes(for:)` so we can
    /// control the write loop (needed for cancellation + progress without
    /// relying on delegates).
    private func downloadFile(
        from url: URL,
        to destURL: URL,
        onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let tmpURL = destURL.appendingPathExtension("tmp")
        let fm = FileManager.default
        if fm.fileExists(atPath: tmpURL.path) {
            try? fm.removeItem(at: tmpURL)
        }
        if fm.fileExists(atPath: destURL.path) {
            try? fm.removeItem(at: destURL)
        }
        fm.createFile(atPath: tmpURL.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: tmpURL) else {
            throw ModelDownloadError.networkError(
                "Could not open \(tmpURL.path) for writing"
            )
        }
        defer { try? handle.close() }

        var req = URLRequest(url: url)
        req.httpMethod = "GET"

        let (bytes, response) = try await urlSession().bytes(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw ModelDownloadError.invalidResponse(
                "Non-HTTP response from \(url.absoluteString)"
            )
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw ModelDownloadError.httpStatus(
                http.statusCode,
                "Failed to download \(url.lastPathComponent)"
            )
        }

        // 256 KB write buffer: decent memory/throughput trade-off on
        // device. Flush on buffer fill; emit progress at ~10 Hz so
        // SwiftUI isn't hammered on fast networks.
        var buffer = Data()
        buffer.reserveCapacity(256 * 1024)
        var written: Int64 = 0
        var lastEmit = Date.distantPast
        let emitInterval: TimeInterval = 0.1

        for try await byte in bytes {
            try Task.checkCancellation()
            buffer.append(byte)
            if buffer.count >= 256 * 1024 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                let now = Date()
                if now.timeIntervalSince(lastEmit) >= emitInterval {
                    onBytes(written)
                    lastEmit = now
                }
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            written += Int64(buffer.count)
        }
        try handle.close()
        onBytes(written)

        try fm.moveItem(at: tmpURL, to: destURL)
    }
}
