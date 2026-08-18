/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QDateTime>
#include <QJsonArray>
#include <QList>
#include <QObject>
#include <QString>

/** @brief Summary of one stored chat session. */
struct ChatSessionInfo
{
    QString id;
    QString title;
    QDateTime updatedAt;
};

/** @class ChatHistoryStore
    @brief Per-project persistence for chat sessions. Sessions are stored as
    individual JSON files in <projectDataFolder>/chats/.
 */
class ChatHistoryStore : public QObject
{
    Q_OBJECT
public:
    explicit ChatHistoryStore(QObject *parent = nullptr);

    /** @brief Point the store at a project's data folder ("" disables it). */
    void setProjectFolder(const QString &projectDataFolder);
    bool isValid() const;

    /** @brief All stored sessions, newest first. */
    QList<ChatSessionInfo> sessions() const;
    QString createSessionId() const;
    /** @brief Load a session's messages; returns an empty array if missing. */
    QJsonArray loadSession(const QString &id, QString *title = nullptr) const;
    void saveSession(const QString &id, const QString &title, const QJsonArray &messages) const;
    void deleteSession(const QString &id) const;

private:
    QString sessionPath(const QString &id) const;
    QString m_folder;
};
