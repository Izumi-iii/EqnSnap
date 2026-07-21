import AppKit
import CoreGraphics
import CoreVideo
import Foundation
import ScreenCaptureKit

enum FormulaCaptureError: Error, Equatable, Sendable {
    case screenCapturePermissionDenied
    case displayUnavailable
    case invalidGeometry
    case emptyImage
    case captureFailed
}

struct CapturedDisplayFrame: @unchecked Sendable {
    let image: CGImage
    let geometry: CaptureDisplayGeometry

    func crop(to rect: PixelRect) throws -> CGImage {
        guard rect.x >= 0,
              rect.y >= 0,
              rect.width > 0,
              rect.height > 0,
              rect.x <= image.width - rect.width,
              rect.y <= image.height - rect.height
        else {
            throw FormulaCaptureError.invalidGeometry
        }
        let cropRect = CGRect(
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height
        )
        guard let cropped = image.cropping(to: cropRect),
              cropped.width == rect.width,
              cropped.height == rect.height
        else {
            throw FormulaCaptureError.emptyImage
        }
        return cropped
    }
}

@MainActor
final class MacScreenCapturePermissionClient {
    var isAuthorized: Bool {
        CGPreflightScreenCaptureAccess()
    }

    func requestAccess() -> Bool {
        CGRequestScreenCaptureAccess()
    }

    func openSystemSettings() {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
        ) else {
            return
        }
        NSWorkspace.shared.open(url)
    }
}

@MainActor
final class CaptureDisplayGeometryAdapter {
    func displayContainingPointer() throws -> CaptureDisplayGeometry {
        let pointer = NSEvent.mouseLocation
        guard let screen = NSScreen.screens.first(where: {
            NSMouseInRect(pointer, $0.frame, false)
        }),
        let displayNumber = screen.deviceDescription[
            NSDeviceDescriptionKey("NSScreenNumber")
        ] as? NSNumber else {
            throw FormulaCaptureError.displayUnavailable
        }

        let displayID = displayNumber.uint32Value
        guard CGDisplayIsOnline(displayID) != 0,
              CGDisplayIsActive(displayID) != 0 else {
            throw FormulaCaptureError.displayUnavailable
        }

        let pixelWidth = CGDisplayPixelsWide(displayID)
        let pixelHeight = CGDisplayPixelsHigh(displayID)
        guard !screen.frame.isNull,
              !screen.frame.isEmpty,
              pixelWidth > 0,
              pixelHeight > 0 else {
            throw FormulaCaptureError.invalidGeometry
        }
        return CaptureDisplayGeometry(
            displayID: displayID,
            bottomLeftGlobalFrame: PointRect(rawValue: screen.frame),
            localBounds: PointRect(
                rawValue: CGRect(origin: .zero, size: screen.frame.size)
            ),
            pixelSize: PixelSize(
                width: pixelWidth,
                height: pixelHeight
            )
        )
    }
}

@MainActor
final class MacScreenCaptureService {
    private let displayAdapter = CaptureDisplayGeometryAdapter()

    func captureDisplayContainingPointer() async throws
        -> CapturedDisplayFrame
    {
        let initialGeometry = try displayAdapter.displayContainingPointer()
        let image: CGImage
        if #available(macOS 14.0, *) {
            image = try await ScreenCaptureKitDisplayBackend().capture(
                displayID: initialGeometry.displayID
            )
        } else {
            image = try await LegacyDisplayCaptureBackend().capture(
                displayID: initialGeometry.displayID
            )
        }
        guard image.width > 0, image.height > 0 else {
            throw FormulaCaptureError.emptyImage
        }
        return CapturedDisplayFrame(
            image: image,
            geometry: CaptureDisplayGeometry(
                displayID: initialGeometry.displayID,
                bottomLeftGlobalFrame: initialGeometry.bottomLeftGlobalFrame,
                localBounds: initialGeometry.localBounds,
                pixelSize: PixelSize(
                    width: image.width,
                    height: image.height
                )
            )
        )
    }
}

@available(macOS 14.0, *)
private actor ScreenCaptureKitDisplayBackend {
    func capture(displayID: UInt32) async throws -> CGImage {
        guard CGPreflightScreenCaptureAccess() else {
            throw FormulaCaptureError.screenCapturePermissionDenied
        }
        do {
            let content = try await SCShareableContent
                .excludingDesktopWindows(
                    true,
                    onScreenWindowsOnly: true
                )
            guard let display = content.displays.first(where: {
                $0.displayID == displayID
            }) else {
                throw FormulaCaptureError.displayUnavailable
            }

            let filter = SCContentFilter(
                display: display,
                excludingWindows: []
            )
            let scale = CGFloat(filter.pointPixelScale)
            let width = filter.contentRect.width * scale
            let height = filter.contentRect.height * scale
            guard width.isFinite,
                  height.isFinite,
                  width > 0,
                  height > 0,
                  width <= CGFloat(Int.max),
                  height <= CGFloat(Int.max)
            else {
                throw FormulaCaptureError.invalidGeometry
            }

            let configuration = SCStreamConfiguration()
            configuration.width = Int(ceil(width))
            configuration.height = Int(ceil(height))
            configuration.showsCursor = false
            configuration.pixelFormat = kCVPixelFormatType_32BGRA
            configuration.captureResolution = .best
            return try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: configuration
            )
        } catch let error as FormulaCaptureError {
            throw error
        } catch {
            if !CGPreflightScreenCaptureAccess() {
                throw FormulaCaptureError.screenCapturePermissionDenied
            }
            throw FormulaCaptureError.captureFailed
        }
    }
}

@available(macOS, introduced: 13.0, obsoleted: 14.0)
private actor LegacyDisplayCaptureBackend {
    func capture(displayID: UInt32) async throws -> CGImage {
        guard CGPreflightScreenCaptureAccess() else {
            throw FormulaCaptureError.screenCapturePermissionDenied
        }
        guard CGDisplayIsOnline(displayID) != 0,
              CGDisplayIsActive(displayID) != 0 else {
            throw FormulaCaptureError.displayUnavailable
        }
        guard let image = CGDisplayCreateImage(displayID) else {
            if !CGPreflightScreenCaptureAccess() {
                throw FormulaCaptureError.screenCapturePermissionDenied
            }
            throw FormulaCaptureError.captureFailed
        }
        return image
    }
}
