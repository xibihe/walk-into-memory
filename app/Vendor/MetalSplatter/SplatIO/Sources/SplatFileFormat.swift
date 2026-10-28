import Foundation

public enum SplatFileFormat {
    case ply
    case dotSplat
    case spz
    case sog

    public init?(for url: URL) {
        switch url.pathExtension.lowercased() {
        case "ply": self = .ply
        case "splat": self = .dotSplat
        case "spz": self = .spz
        case "sog": self = .sog
        default: return nil
        }
    }
}
