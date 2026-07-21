import Carbon
import Foundation

private let eqnSnapHotKeySignature: OSType = 0x4551_4E53

private let eqnSnapHotKeyEventHandler: EventHandlerUPP = {
    _, event, userData in
    guard let event, let userData else {
        return OSStatus(eventNotHandledErr)
    }
    let client = Unmanaged<CarbonHotKeyClient>
        .fromOpaque(userData)
        .takeUnretainedValue()
    return client.receive(event)
}

// Adapted from SnapioPlatform/Shortcuts/CarbonHotKeyClient.swift.
@MainActor
final class CarbonHotKeyClient {
    private var eventHandler: EventHandlerRef?
    private var hotKeyReference: EventHotKeyRef?
    private var onTrigger: (() -> Void)?

    deinit {
        if let hotKeyReference {
            UnregisterEventHotKey(hotKeyReference)
        }
        if let eventHandler {
            RemoveEventHandler(eventHandler)
        }
    }

    func registerDefault(onTrigger: @escaping () -> Void) throws {
        unregister()
        try ensureEventHandler()

        let hotKeyID = EventHotKeyID(
            signature: eqnSnapHotKeySignature,
            id: 1
        )
        var reference: EventHotKeyRef?
        let status = RegisterEventHotKey(
            14, // E
            UInt32(cmdKey | controlKey),
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &reference
        )
        guard status == noErr, let reference else {
            throw CarbonHotKeyError.registrationFailed(status)
        }
        self.onTrigger = onTrigger
        hotKeyReference = reference
    }

    func unregister() {
        if let hotKeyReference {
            UnregisterEventHotKey(hotKeyReference)
        }
        hotKeyReference = nil
        onTrigger = nil
    }

    nonisolated fileprivate func receive(_ event: EventRef) -> OSStatus {
        var hotKeyID = EventHotKeyID()
        let status = GetEventParameter(
            event,
            EventParamName(kEventParamDirectObject),
            EventParamType(typeEventHotKeyID),
            nil,
            MemoryLayout<EventHotKeyID>.size,
            nil,
            &hotKeyID
        )
        guard status == noErr,
              hotKeyID.signature == 0x4551_4E53,
              hotKeyID.id == 1 else {
            return OSStatus(eventNotHandledErr)
        }
        Task { @MainActor [weak self] in
            self?.onTrigger?()
        }
        return noErr
    }

    private func ensureEventHandler() throws {
        guard eventHandler == nil else { return }
        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        var installedHandler: EventHandlerRef?
        let status = InstallEventHandler(
            GetApplicationEventTarget(),
            eqnSnapHotKeyEventHandler,
            1,
            &eventType,
            Unmanaged.passUnretained(self).toOpaque(),
            &installedHandler
        )
        guard status == noErr, let installedHandler else {
            throw CarbonHotKeyError.registrationFailed(status)
        }
        eventHandler = installedHandler
    }
}

enum CarbonHotKeyError: Error, Equatable {
    case registrationFailed(OSStatus)
}
