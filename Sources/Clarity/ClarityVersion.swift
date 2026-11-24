import Foundation

/// Clarity's own version, separate from the app version (like Motion). v1.0 is the final pure-Metal release (CLARITY_PROGRESS.md › v1.0); earlier versions kept `-beta` until the live A/B on real
/// sessions is done. Every bump is logged in CLARITY_PROGRESS.md.
public enum ClarityVersion {
    public static let current = "1.0"
    /// "Clarity (v1.0)"
    public static var label: String { "Clarity (v\(current))" }
}
