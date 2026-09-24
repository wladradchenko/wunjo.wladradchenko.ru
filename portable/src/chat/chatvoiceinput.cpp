/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chatvoiceinput.h"
#include "pythoninterfaces/speechtotextwhisper.h"
#include "wunjosettings.h"

#include <KLocalizedString>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QMediaFormat>
#include <QRegularExpression>
#include <QUrl>

ChatVoiceInput::ChatVoiceInput(QObject *parent)
    : QObject(parent)
    , m_audioInput(std::make_unique<QAudioInput>())
    , m_recorder(std::make_unique<QMediaRecorder>())
{
    m_session.setAudioInput(m_audioInput.get());
    m_session.setRecorder(m_recorder.get());
    QMediaFormat format(QMediaFormat::Wave);
    m_recorder->setMediaFormat(format);
    m_recorder->setQuality(QMediaRecorder::HighQuality);

    connect(m_recorder.get(), &QMediaRecorder::recorderStateChanged, this, [this](QMediaRecorder::RecorderState state) {
        if (state == QMediaRecorder::StoppedState) {
            Q_EMIT recordingChanged(false);
            const QString file = m_recorder->actualLocation().toLocalFile();
            if (!file.isEmpty() && QFile::exists(file)) {
                transcribe(file);
            }
        } else if (state == QMediaRecorder::RecordingState) {
            Q_EMIT recordingChanged(true);
        }
    });
    connect(m_recorder.get(), &QMediaRecorder::errorOccurred, this,
            [this](QMediaRecorder::Error, const QString &message) { Q_EMIT errorOccurred(i18n("Recording failed: %1", message)); });
}

bool ChatVoiceInput::isRecording() const
{
    return m_recorder->recorderState() == QMediaRecorder::RecordingState;
}

void ChatVoiceInput::toggle()
{
    if (isRecording()) {
        m_recorder->stop();
        return;
    }
    if (m_process && m_process->state() != QProcess::NotRunning) {
        // still transcribing the previous take
        return;
    }
    startRecording();
}

void ChatVoiceInput::startRecording()
{
    const QString file =
        QDir::temp().absoluteFilePath(QStringLiteral("wunjo-voice-%1.wav").arg(QDateTime::currentDateTime().toString(QStringLiteral("hhmmsszzz"))));
    m_recorder->setOutputLocation(QUrl::fromLocalFile(file));
    m_recorder->record();
}

void ChatVoiceInput::transcribe(const QString &fileName)
{
    SpeechToTextWhisper stt;
    const QString python = stt.venvPythonExecs().python;
    if (python.isEmpty() || !QFile::exists(python)) {
        Q_EMIT errorOccurred(i18n("The Whisper speech engine is not installed. Configure it in Settings > Configure Wunjo > Speech to text."));
        return;
    }
    QString model = WunjoSettings::whisperModel();
    if (model.isEmpty()) {
        model = QStringLiteral("base");
    }

    QStringList args = {stt.speechScript(),
                        QStringLiteral("--src=\"%1\"").arg(fileName),
                        QStringLiteral("--model=%1").arg(model),
                        QStringLiteral("--task=transcribe"),
                        QStringLiteral("--ffmpeg=%1").arg(WunjoSettings::ffmpegpath()),
                        QStringLiteral("--language=%1").arg(WunjoSettings::whisperLanguage())};
    if (!WunjoSettings::whisperDevice().isEmpty()) {
        args << QStringLiteral("--device=%1").arg(WunjoSettings::whisperDevice());
    }

    m_process = std::make_unique<QProcess>();
    Q_EMIT busyChanged(true);
    connect(m_process.get(), QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this, fileName](int exitCode, QProcess::ExitStatus status) {
                Q_EMIT busyChanged(false);
                const QString output = QString::fromUtf8(m_process->readAllStandardOutput());
                QFile::remove(fileName);
                if (status != QProcess::NormalExit || exitCode != 0) {
                    Q_EMIT errorOccurred(i18n("Speech recognition failed: %1", QString::fromUtf8(m_process->readAllStandardError()).right(300)));
                    return;
                }
                // whispertotext.py emits "[start>end]word" per word; collect the words
                static const QRegularExpression wordLine(QStringLiteral("^\\[[\\d.]+>[\\d.]+\\](.+)$"));
                QString text;
                const QStringList lines = output.split(QLatin1Char('\n'));
                for (const QString &line : lines) {
                    const QRegularExpressionMatch match = wordLine.match(line);
                    if (match.hasMatch()) {
                        text += match.captured(1);
                    }
                }
                text = text.simplified();
                if (text.isEmpty()) {
                    Q_EMIT errorOccurred(i18n("No speech detected."));
                } else {
                    Q_EMIT transcribed(text);
                }
            });
    m_process->start(python, args);
}
