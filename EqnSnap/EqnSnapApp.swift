//
//  EqnSnapApp.swift
//  EqnSnap
//
//  Created by wangjie on 2026/7/16.
//

import SwiftUI

@main
struct EqnSnapApp: App {
    @NSApplicationDelegateAdaptor(EqnSnapAppDelegate.self)
    private var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
final class EqnSnapAppDelegate: NSObject, NSApplicationDelegate {
    private var workflow: FormulaCaptureWorkflow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        let workflow = FormulaCaptureWorkflow()
        workflow.start()
        self.workflow = workflow
    }

    func applicationWillTerminate(_ notification: Notification) {
        workflow?.stop()
    }

    func applicationSupportsSecureRestorableState(
        _ app: NSApplication
    ) -> Bool {
        true
    }
}
