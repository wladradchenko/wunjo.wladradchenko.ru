/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QAudioInput>
#include <QMediaCaptureSession>
#include <QMediaRecorder>
#include <QObject>
#include <QProcess>

#include <memory>

/** @class ChatVoiceInput
    @brief Voice input for the chat: records from the default microphone into
    a temporary file and transcribes it with the project's Whisper speech
    engine (the same venv/models used by the Speech Editor).
 */
class ChatVoiceInput : public QObject
{
    Q_OBJECT
public:
    explicit ChatVoiceInput(QObject *parent = nullptr);

    bool isRecording() const;

public Q_SLOTS:
    /** @brief Start recording, or stop and transcribe when recording. */
    void toggle();

Q_SIGNALS:
    void recordingChanged(bool recording);
    /** @brief True while Whisper is processing the recording. */
    void busyChanged(bool busy);
    void transcribed(const QString &text);
    void errorOccurred(const QString &message);

private:
    void startRecording();
    void transcribe(const QString &fileName);

    QMediaCaptureSession m_session;
    std::unique_ptr<QAudioInput> m_audioInput;
    std::unique_ptr<QMediaRecorder> m_recorder;
    std::unique_ptr<QProcess> m_process;
};
