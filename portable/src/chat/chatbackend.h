/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

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

Q_SIGNALS:
    /** @brief A chunk (or the whole) of the assistant's text reply. */
    void assistantText(const QString &text);
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
