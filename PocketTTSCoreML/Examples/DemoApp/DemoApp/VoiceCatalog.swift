//
// VoiceCatalog.swift
//
// Enumerates voice .safetensors files that were copied into the app bundle
// by prepare_resources.sh, AND (new) any user-cloned voices stored under
// Documents/ClonedVoices/<languageID>/. A voice file's filename stem is
// used as the id; the display name is computed by splitting on '_' and
// title-casing each part ("peter_yearsley" -> "Peter Yearsley").
//
// Per-language layout (expected on disk):
//
//   Bundle:    Resources/Languages/<id>/Voices/*.safetensors    (read-only, bundled)
//   Bundle:    Resources/Languages/<id>/Artifacts/...
//   Bundle:    Resources/Languages/<id>/tokenizer.model
//   Writable:  Documents/ClonedVoices/<id>/*.safetensors        (user clones)
//
// A legacy single-language layout (Resources/Voices/...) is still honored
// as a fallback by `VoiceCatalog.bundled()` (no language id) so older demo
// bundles keep working until prepare_resources.sh is re-run.
//

import Foundation

public struct VoiceEntry: Identifiable, Hashable, Sendable {
    /// Filename stem (e.g. "bill_boerst").
    public let id: String
    /// Human-readable title-cased name (e.g. "Bill Boerst").
    public let displayName: String
    /// Absolute URL inside the app bundle or Documents dir.
    public let url: URL
    /// Language code ("en", "es", "de", "it", "pt", "fr"). Empty for the
    /// legacy single-language bundle.
    public let language: String
    /// True for user-cloned voices stored under Documents/ClonedVoices/.
    /// Bundled voices set this to false. Cloned voices are rendered with
    /// a "(cloned)" suffix + custom icon in VoiceListSheet and can be
    /// deleted via swipe-to-delete.
    public let isCloned: Bool
}

public enum VoiceCatalog {

    /// Combined list of bundled + cloned voices for the given language,
    /// sorted alphabetically by displayName. Cloned voices appear inline
    /// with the bundled ones; use `VoiceEntry.isCloned` to distinguish.
    public static func all(for language: String) -> [VoiceEntry] {
        let bundled = self.bundled(for: language)
        let cloned = self.cloned(for: language)
        return (bundled + cloned).sorted { $0.displayName < $1.displayName }
    }

    /// Return `.safetensors` voices bundled under `Resources/Languages/<id>/Voices/`,
    /// sorted alphabetically by displayName. Falls back to an empty array
    /// if the folder was not seeded (prepare_resources.sh not run for that
    /// language).
    public static func bundled(for language: String) -> [VoiceEntry] {
        guard let voicesDir = resolveVoicesDir(language: language) else { return [] }
        return scanVoices(at: voicesDir, language: language, isCloned: false)
    }

    /// Return user-cloned voices saved under
    /// `Documents/ClonedVoices/<languageID>/*.safetensors`. Empty if the
    /// user hasn't cloned anything for this language yet.
    public static func cloned(for language: String) -> [VoiceEntry] {
        let dir = clonedVoicesDir(for: language)
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return [] }
        return scanVoices(at: dir, language: language, isCloned: true)
    }

    /// Legacy flat-layout enumeration (pre language-picker demos). Returns
    /// any `.safetensors` found directly under `Resources/Voices/` or the
    /// bundle root, with an empty `language` field.
    public static func bundled() -> [VoiceEntry] {
        guard let voicesDir = resolveFlatVoicesDir() else { return [] }
        return scanVoices(at: voicesDir, language: "", isCloned: false)
    }

    // MARK: - Cloned voice persistence

    /// Writable directory where cloned voices for `language` are stored.
    /// Always creates the directory on demand — callers don't need to
    /// guard against it being missing.
    public static func clonedVoicesDir(for language: String) -> URL {
        let fm = FileManager.default
        let docs =
            fm.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? fm.temporaryDirectory
        let dir =
            docs
            .appendingPathComponent("ClonedVoices", isDirectory: true)
            .appendingPathComponent(language, isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// Build the destination URL for a named clone in the given language.
    /// Name is sanitized to alphanumerics + underscore; empty names fall
    /// back to a timestamp. Does NOT check for collisions — caller is
    /// expected to add a "(2)"-style disambiguator if they want one.
    public static func clonedVoiceURL(language: String, rawName: String) -> URL {
        let sanitized = sanitizeName(rawName)
        let stem = sanitized.isEmpty ? "clone_\(Int(Date().timeIntervalSince1970))" : sanitized
        return clonedVoicesDir(for: language)
            .appendingPathComponent("\(stem).safetensors")
    }

    /// Delete a cloned voice from disk. Returns true if removed. Throws
    /// an error if the entry is bundled (callers should guard but this is
    /// a belt-and-braces safety check against accidental deletion of
    /// read-only bundle content).
    public static func deleteCloned(_ entry: VoiceEntry) throws {
        guard entry.isCloned else {
            throw NSError(
                domain: "VoiceCatalog", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "Refusing to delete bundled voice \(entry.id)"
                ])
        }
        try FileManager.default.removeItem(at: entry.url)
    }

    /// Check whether voice cloning is available for `language`. Cloning
    /// requires `Resources/Languages/<id>/Artifacts/speaker_proj.safetensors`
    /// to be present (it's an optional sidecar emitted by
    /// `pockettts_coreml.convert.export_speaker_proj`).
    public static func cloningAvailable(for language: String) -> Bool {
        guard let base = Bundle.main.resourceURL else { return false }
        let proj =
            base
            .appendingPathComponent("Languages", isDirectory: true)
            .appendingPathComponent(language, isDirectory: true)
            .appendingPathComponent("Artifacts", isDirectory: true)
            .appendingPathComponent("speaker_proj.safetensors")
        return FileManager.default.fileExists(atPath: proj.path)
    }

    // MARK: - Internals

    private static func sanitizeName(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        return String(trimmed.map { allowed.contains($0) ? $0 : "_" })
    }

    private static func scanVoices(
        at voicesDir: URL, language: String, isCloned: Bool
    ) -> [VoiceEntry] {
        let fm = FileManager.default
        guard
            let items = try? fm.contentsOfDirectory(
                at: voicesDir,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
        else { return [] }

        var entries: [VoiceEntry] = []
        entries.reserveCapacity(items.count)
        for url in items where url.pathExtension == "safetensors" {
            let stem = url.deletingPathExtension().lastPathComponent
            entries.append(
                VoiceEntry(
                    id: stem,
                    displayName: titleCase(stem),
                    url: url,
                    language: language,
                    isCloned: isCloned
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

    /// Per-language: `Bundle.main.resourceURL/Languages/<id>/Voices/`.
    private static func resolveVoicesDir(language: String) -> URL? {
        guard let base = Bundle.main.resourceURL else { return nil }
        let fm = FileManager.default
        let candidate =
            base
            .appendingPathComponent("Languages", isDirectory: true)
            .appendingPathComponent(language, isDirectory: true)
            .appendingPathComponent("Voices", isDirectory: true)
        if fm.fileExists(atPath: candidate.path) { return candidate }
        return nil
    }

    /// Legacy flat-layout resolver (single-language builds).
    private static func resolveFlatVoicesDir() -> URL? {
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

// MARK: - Language model

/// Tiny value type describing a demo-app language. Not a `Locale` because
/// the language id here is the folder name on disk under `Resources/Languages/`,
/// which is a project-internal convention (matches the reference config file
/// names: `english`, `spanish`) rather than a standard locale code.
public struct Language: Identifiable, Hashable, Sendable {
    /// Folder id ("en", "es", "de", "it", "pt", "fr"). Used as the
    /// `Resources/Languages/<id>/` path.
    public let id: String
    /// Reference config name used by conversion scripts ("english",
    /// "spanish", "german", "italian", "portuguese", "french_24l").
    /// Note: French only ships as the 24-layer variant (`french_24l`);
    /// there is no 6-layer French config.
    public let configName: String
    /// Human-readable display name ("English", "Spanish", "German",
    /// "Italian", "Portuguese", "French").
    public let displayName: String
    /// Default demo paragraph in this language. Used as initial text at
    /// launch and when the user switches languages (only overwrites the
    /// editor if the current text matches a known default, so custom user
    /// input is preserved).
    public let defaultPrompt: String

    public init(
        id: String, configName: String, displayName: String, defaultPrompt: String
    ) {
        self.id = id
        self.configName = configName
        self.displayName = displayName
        self.defaultPrompt = defaultPrompt
    }

    /// All shipped languages. Order is the picker order.
    public static let all: [Language] = [
        Language(
            id: "en", configName: "english", displayName: "English",
            defaultPrompt: """
                Pocket TTS is a lightweight text-to-speech model. \
                It runs entirely on your iPhone, streaming audio as the voice speaks. \
                Tap Stream to hear it break a longer paragraph into sentences.
                """
        ),
        Language(
            id: "es", configName: "spanish", displayName: "Spanish",
            defaultPrompt: """
                Pocket TTS es un modelo ligero de conversión de texto a voz. \
                Se ejecuta por completo en tu iPhone, transmitiendo audio mientras la voz habla. \
                Toca Stream para oírlo dividir un párrafo largo en frases.
                """
        ),
        Language(
            id: "de", configName: "german", displayName: "German",
            defaultPrompt: """
                Pocket TTS ist ein leichtgewichtiges Text-zu-Sprache-Modell. \
                Es läuft vollständig auf deinem iPhone und überträgt Audio, während die Stimme spricht. \
                Tippe auf Stream, um zu hören, wie ein langer Absatz in Sätze aufgeteilt wird.
                """
        ),
        Language(
            id: "it", configName: "italian", displayName: "Italian",
            defaultPrompt: """
                Pocket TTS è un modello leggero di sintesi vocale. \
                Funziona interamente sul tuo iPhone, trasmettendo l'audio mentre la voce parla. \
                Tocca Stream per sentirlo dividere un paragrafo lungo in frasi.
                """
        ),
        Language(
            id: "pt", configName: "portuguese", displayName: "Portuguese",
            defaultPrompt: """
                Pocket TTS é um modelo leve de conversão de texto em fala. \
                Funciona inteiramente no seu iPhone, transmitindo áudio enquanto a voz fala. \
                Toque em Stream para ouvir como ele divide um parágrafo longo em frases.
                """
        ),
        Language(
            id: "fr", configName: "french_24l", displayName: "French",
            defaultPrompt: """
                Pocket TTS est un modèle léger de synthèse vocale. \
                Il fonctionne entièrement sur votre iPhone, diffusant l'audio pendant que la voix parle. \
                Appuyez sur Stream pour l'entendre diviser un long paragraphe en phrases.
                """
        ),
    ]

    /// The set of texts that count as "default demo prompts" — used to
    /// decide whether switching languages should overwrite the editor.
    public static var allDefaultPrompts: Set<String> {
        Set(all.map(\.defaultPrompt))
    }

    /// Convenience: the first language that has any bundled voices. Used
    /// as the initial selection when the app boots.
    public static var defaultBundled: Language {
        for lang in all {
            if !VoiceCatalog.bundled(for: lang.id).isEmpty { return lang }
        }
        // Fallback: first in the static list (still valid for identification
        // even if its resources are missing — TTSViewModel.load() surfaces
        // the error).
        return all[0]
    }
}
