//
// PocketTTSLiveActivityBundle.swift
//
// @main entry point for the widget extension. Registers the single
// Live Activity widget. WidgetBundle is the mandated entry type for
// extensions that can ship multiple widgets (we only ship one).
//

import SwiftUI
import WidgetKit

@main
struct PocketTTSLiveActivityBundle: WidgetBundle {
    var body: some Widget {
        PocketTTSLiveActivity()
    }
}
