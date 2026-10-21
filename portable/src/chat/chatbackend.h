/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QJsonArray>
#include <QObject>
#include <QString>

/** @class AbstractChatBackend
    @brief Seam between the chat UI and a future assistant implementation
    (Claude via the MCP server in mcp/). The event vocabulary mirrors the
    web prototype (assistant_text / tool_start / tool_progress / tool_end),
    so a backend can be plugged in without touching ChatWidget.
 */
class AbstractChatBackend : public QObject
{
    Q_OBJECT
public:
    explicit AbstractChatBackend(QObject *parent = nullptr)
        : QObject(parent)
    {
    }

    /** @brief Send a user message to the assistant. */
    virtual void sendMessage(const QString &text) = 0;
    /** @brief Cancel the in-flight exchange, if any. */
    virtual void cancel() = 0;
    /** @brief Which conversation the next message belongs to, so the assistant
     *  can pick up where it left off instead of meeting the user anew. */
    virtual void setSession(const QString &sessionId) { Q_UNUSED(sessionId) }
    /** @brief The user picked @p optionId on the question @p questionId. */
    virtual void answerQuestion(const QString &questionId, const QString &optionId)
    {
        Q_UNUSED(questionId) Q_UNUSED(optionId)
    }
    /** @brief Let go of what the backend holds (a model, a process) before the
     *  user switches to another way of talking or the app closes. */
    virtual void release() { }
    /** @brief How freely the assistant may act, as the assistant itself names
     *  its modes: [{"id", "name", "description"}]. Empty when it has none. */
    virtual QJsonArray modes() const { return {}; }
    virtual QString currentMode() const { return {}; }
    /** @brief Switch to the mode @p id, now and for the assistant's next sessions. */
    virtual void setMode(const QString &id) { Q_UNUSED(id) }

Q_SIGNALS:
    /** @brief A chunk (or the whole) of the assistant's text reply. */
    void assistantText(const QString &text);
    /** @brief The reply so far, as it is written: replaces the bubble the
     *  previous call made instead of adding one. */
    void assistantStream(const QString &textSoFar);
    /** @brief The assistant asks something only the user can answer — run this
     *  command? sign in how? — with @p options as [{"id", "label"}]. */
    void questionAsked(const QString &questionId, const QString &text, const QJsonArray &options);
    /** @brief The assistant started running a tool (an MCP call). */
    void toolStarted(const QString &id, const QString &name);
    /** @brief Progress update for a running tool. */
    void toolProgress(const QString &id, int percent, const QString &message);
    /** @brief A tool finished; @p result is a short human-readable summary. */
    void toolFinished(const QString &id, bool isError, const QString &result);
    /** @brief The exchange failed with an error. */
    void errorOccurred(const QString &message);
    /** @brief True while a request is in flight (disables the send button). */
    void busyChanged(bool busy);
};
