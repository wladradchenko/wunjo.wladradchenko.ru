/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QJsonArray>
#include <QVector>

/** @brief One entry of a chat session: a text bubble or a tool-run card. */
struct ChatMessage
{
    enum class Author { User, Assistant, Error };
    enum class Kind { Text, ToolCard };
    enum class ToolStatus { Running, Done, Failed };

    Author author{Author::User};
    Kind kind{Kind::Text};
    QString text;
    QString toolId;
    QString toolName;
    ToolStatus toolStatus{ToolStatus::Running};
    int toolProgress{-1};
    QDateTime timestamp;
};

/** @class ChatMessageModel
    @brief Source of truth for the messages of the current chat session.
    The view (ChatWidget) appends widgets on rowsInserted/dataChanged.
 */
class ChatMessageModel : public QAbstractListModel
{
    Q_OBJECT
public:
    enum Roles {
        TextRole = Qt::UserRole + 1,
        AuthorRole,
        KindRole,
        ToolIdRole,
        ToolNameRole,
        ToolStatusRole,
        ToolProgressRole,
        TimestampRole
    };

    explicit ChatMessageModel(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    void appendText(ChatMessage::Author author, const QString &text);
    /** @brief Replace the last bubble if it is text by @p author (streaming
        updates), otherwise append a new one. */
    void updateLastText(ChatMessage::Author author, const QString &text);
    void appendToolCard(const QString &id, const QString &name);
    /** @brief End every card still shown as running, with @p statusText as its
     *  last word — except those whose id starts with one of @p keptPrefixes.
     *
     *  A card is closed by whoever opened it, and an agent that stops mid-turn
     *  never does. The editor's own cards are the exception: they are prefixed,
     *  and they outlive a turn on purpose — a render goes on long after the
     *  assistant has finished answering. */
    void endRunningTools(const QString &statusText, const QStringList &keptPrefixes = {});
    /** @brief Update a tool card by id; @p progress -1 keeps the current value. */
    void updateTool(const QString &id, ChatMessage::ToolStatus status, int progress, const QString &statusText);
    void clear();

    QJsonArray toJson() const;
    void loadJson(const QJsonArray &array);

private:
    QVector<ChatMessage> m_messages;
};
