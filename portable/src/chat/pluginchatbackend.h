/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "chatbackend.h"

#include <QStringList>

/** @class PluginChatBackend
    @brief The chat, answered by an assistant plugin.

    It carries the user's message to the plugin and reports whether the turn is
    still running — and that is all it carries. The reply, the thinking state
    and the progress cards do not come back through here: the plugin puts them
    in the chat the same way Claude Code does, over the scripting socket into
    ChatWidget::external*. So the panel cannot tell the two apart, which is the
    point — one path to keep working, not two that drift.
 */
class PluginChatBackend : public AbstractChatBackend
{
    Q_OBJECT
public:
    explicit PluginChatBackend(const QString &pluginId, QObject *parent = nullptr);
    ~PluginChatBackend() override;

    void sendMessage(const QString &text) override;
    void cancel() override;

    /** @brief Which conversation the next message belongs to, so the assistant
     *  can pick up where it left off instead of meeting the user anew. */
    void setSession(const QString &sessionId);

    /** @brief Ask the plugin to unload the model. Called when the user picks a
     *  different way of talking, and when the app is closing: a few gigabytes
     *  of weights should not outlive the reason they were loaded. */
    void release();

private:
    QString m_pluginId;
    QString m_sessionId;
    /** @brief The turn was stopped on purpose, so its death is not news. */
    bool m_stopped{false};
    QString m_job; ///< the turn in flight, for cancel()
    /** @brief What the user said while the assistant was still answering.
     *
     *  Somebody who has just watched the assistant misread them types the
     *  correction immediately — that is when they think of it, not a minute
     *  later. Turning it away means it has to be typed again; keeping it means
     *  it is the next thing asked. Several such lines are one message, because
     *  they are one thought.
     */
    QStringList m_pending;
    /** @brief Send what was said while busy, if anything. */
    void sendPending();
};
