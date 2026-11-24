import Foundation

/// Clarity's own version, separate from the app version (like Motion). Stays `-beta` until the live A/B on real
/// sessions is done. Every bump is logged in CLARITY_PROGRESS.md.
public enum ClarityVersion {
    public static let current = "0.3-beta"
    /// "Clarity (v0.3-beta)"
    public static var label: String { "Clarity (v\(current))" }
}
