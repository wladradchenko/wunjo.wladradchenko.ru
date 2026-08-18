/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginchatbackend.h"

#include "plugins/pluginmanager.h"

#include <KLocalizedString>

#include <QJsonObject>

PluginChatBackend::PluginChatBackend(const QString &pluginId, QObject *parent)
    : AbstractChatBackend(parent)
    , m_pluginId(pluginId)
{
}

PluginChatBackend::~PluginChatBackend()
{
    // Only stop what is running. Starting a new process here would mean asking
    // the editor for the open project while it is being torn down, and the
    // model lets go by itself once the conversation has gone quiet.
    cancel();
}

void PluginChatBackend::setSession(const QString &sessionId)
{
    m_sessionId = sessionId;
}

void PluginChatBackend::sendMessage(const QString &text)
{
    if (!m_job.isEmpty()) {
        // One turn at a time — a second model beside the first would only take
        // its memory — but the message is kept rather than refused, and goes as
        // soon as this turn ends. What somebody adds while waiting is usually
        // the correction that matters most.
        m_pending << text;
        return;
    }
    const QString blocker = PluginManager::instance().runBlocker(m_pluginId);
    if (!blocker.isEmpty()) {
        Q_EMIT errorOccurred(blocker);
        return;
    }

    QJsonObject input;
    input.insert(QStringLiteral("action"), QStringLiteral("chat"));
    input.insert(QStringLiteral("message"), text);
    input.insert(QStringLiteral("session"), m_sessionId);

    Q_EMIT busyChanged(true);
    m_stopped = false;
    m_job = PluginManager::instance().runPluginJob(
        m_pluginId, input, this, nullptr, [this](const QJsonObject &, const QString &error) {
            m_job.clear();
            Q_EMIT busyChanged(false);
            // A turn the user stopped ends with the process being killed, and
            // the last thing it printed is not an error to report back at them:
            // they know why it stopped, they stopped it.
            if (!error.isEmpty() && !m_stopped) {
                Q_EMIT errorOccurred(error);
            }
            sendPending();
        });
    if (m_job.isEmpty()) {
        // startProcess already reported why through the finished callback
        Q_EMIT busyChanged(false);
    }
}

void PluginChatBackend::sendPending()
{
    if (m_pending.isEmpty() || !m_job.isEmpty()) {
        return;
    }
    // Everything said while the assistant was busy goes as one message: it was
    // one train of thought, and answering each line on its own would lose what
    // the later ones were correcting.
    const QString text = m_pending.join(QLatin1Char('\n'));
    m_pending.clear();
    sendMessage(text);
}

void PluginChatBackend::cancel()
{
    m_stopped = true;
    if (m_job.isEmpty()) {
        return;
    }
    PluginManager::instance().cancelPluginJob(m_job);
    m_job.clear();
    Q_EMIT busyChanged(false);
}

void PluginChatBackend::release()
{
    cancel();
    QJsonObject input;
    input.insert(QStringLiteral("action"), QStringLiteral("stop"));
    // Nothing to follow: it either lets the model go or it was not holding one.
    PluginManager::instance().runPluginJob(m_pluginId, input, this, nullptr, nullptr);
}
