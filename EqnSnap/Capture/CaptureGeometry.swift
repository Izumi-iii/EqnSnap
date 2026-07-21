import CoreGraphics
import Foundation

// Adapted from SnapioCore/Shared/GeometryTypes.swift and
// SnapioCore/Capture/GeometryMapper.swift.
enum BottomLeftGlobalPoints: Sendable {}
enum DisplayLocalPoints: Sendable {}

struct Point2<Space>: Equatable, Sendable {
    let rawValue: CGPoint
}

struct PointRect<Space>: Equatable, Sendable {
    let rawValue: CGRect
}

struct PixelSize: Equatable, Sendable {
    let width: Int
    let height: Int
}

struct PixelRect: Equatable, Sendable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
}

struct CaptureSessionID: Hashable, Sendable {
    let rawValue: UUID

    init(rawValue: UUID = UUID()) {
        self.rawValue = rawValue
    }
}

struct CaptureDisplayGeometry: Equatable, Sendable {
    let displayID: UInt32
    let bottomLeftGlobalFrame: PointRect<BottomLeftGlobalPoints>
    let localBounds: PointRect<DisplayLocalPoints>
    let pixelSize: PixelSize
}

enum CaptureGeometryError: Error, Equatable, Sendable {
    case emptyRect
    case outsideDisplay
    case invalidPixelSize
}

struct CaptureGeometryMapper: Sendable {
    func clamp(
        _ rect: PointRect<DisplayLocalPoints>,
        to geometry: CaptureDisplayGeometry
    ) -> PointRect<DisplayLocalPoints> {
        let bounds = geometry.localBounds.rawValue
        let input = rect.rawValue
        guard bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0,
              input.isFinite
        else {
            return PointRect(rawValue: .zero)
        }

        let normalized = input.standardized
        let minX = min(max(normalized.minX, 0), bounds.width)
        let minY = min(max(normalized.minY, 0), bounds.height)
        let maxX = min(max(normalized.maxX, 0), bounds.width)
        let maxY = min(max(normalized.maxY, 0), bounds.height)
        return PointRect(
            rawValue: CGRect(
                x: minX,
                y: minY,
                width: max(0, maxX - minX),
                height: max(0, maxY - minY)
            )
        )
    }

    func pixelRect(
        from rect: PointRect<DisplayLocalPoints>,
        geometry: CaptureDisplayGeometry
    ) throws -> PixelRect {
        let bounds = geometry.localBounds.rawValue
        guard bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0,
              geometry.pixelSize.width > 0,
              geometry.pixelSize.height > 0
        else {
            throw CaptureGeometryError.invalidPixelSize
        }

        let normalized = rect.rawValue.standardized
        guard normalized.isFinite,
              normalized.width > 0,
              normalized.height > 0
        else {
            throw CaptureGeometryError.emptyRect
        }
        guard normalized.maxX > 0,
              normalized.maxY > 0,
              normalized.minX < bounds.width,
              normalized.minY < bounds.height
        else {
            throw CaptureGeometryError.outsideDisplay
        }

        let clipped = clamp(
            PointRect(rawValue: normalized),
            to: geometry
        ).rawValue
        guard clipped.width > 0, clipped.height > 0 else {
            throw CaptureGeometryError.outsideDisplay
        }

        let scaleX = CGFloat(geometry.pixelSize.width) / bounds.width
        let scaleY = CGFloat(geometry.pixelSize.height) / bounds.height
        let minX = max(0, Int(floor(clipped.minX * scaleX)))
        let minY = max(0, Int(floor(clipped.minY * scaleY)))
        let maxX = min(
            geometry.pixelSize.width,
            Int(ceil(clipped.maxX * scaleX))
        )
        let maxY = min(
            geometry.pixelSize.height,
            Int(ceil(clipped.maxY * scaleY))
        )
        guard maxX > minX, maxY > minY else {
            throw CaptureGeometryError.emptyRect
        }
        return PixelRect(
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        )
    }

    func pixelSize(
        of rect: PointRect<DisplayLocalPoints>,
        geometry: CaptureDisplayGeometry
    ) throws -> PixelSize {
        let rect = try pixelRect(from: rect, geometry: geometry)
        return PixelSize(width: rect.width, height: rect.height)
    }
}

private extension CGRect {
    var isFinite: Bool {
        origin.x.isFinite
            && origin.y.isFinite
            && size.width.isFinite
            && size.height.isFinite
    }
}
