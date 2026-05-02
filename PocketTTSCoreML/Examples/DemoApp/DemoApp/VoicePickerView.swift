//
// VoicePickerView.swift
//
// A Menu-backed picker that keeps the UI compact even with 21 voices.
//

import SwiftUI

struct VoicePickerView: View {
    let voices: [VoiceEntry]
    @Binding var selection: VoiceEntry?
    let disabled: Bool

    var body: some View {
        Menu {
            ForEach(voices) { voice in
                Button {
                    selection = voice
                } label: {
                    HStack {
                        Text(voice.displayName)
                        if selection?.id == voice.id {
                            Spacer()
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack {
                Label(
                    selection?.displayName ?? "Pick a voice",
                    systemImage: "person.wave.2.fill"
                )
                Spacer()
                Image(systemName: "chevron.up.chevron.down")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .disabled(disabled || voices.isEmpty)
    }
}
