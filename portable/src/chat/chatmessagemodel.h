/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QJsonArray>
#include <QJsonObject>
#include <QVector>

/** @brief One entry of a chat session: a text bubble, a tool-run card, or the
 *  form of a generator (what to make, and a button that makes it). */
struct ChatMessage
{
    enum class Author { User, Assistant, Error };
    enum class Kind { Text, ToolCard, Generator };
    enum class ToolStatus { Running, Done, Failed };

    Author author{Author::User};
    Kind kind{Kind::Text};
    QString text;
    QString toolId;
    QString toolName;
    ToolStatus toolStatus{ToolStatus::Running};
    int toolProgress{-1};
    QDateTime timestamp;
    /** @brief For a generator: {"plugin", "values": {key: value}, "collapsed",
     *  "author": "user" | "assistant"}. The form's state lives here and not in
     *  its widgets, which are rebuilt whenever the session is redrawn. */
    QJsonObject payload;
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
        TimestampRole,
        PayloadRole
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
    /** @brief Add the form of a generator plugin, under the card id @p id. */
    void appendGenerator(const QString &id, const QString &name, const QJsonObject &payload);
    /** @brief Replace a generator's state. @p notify false when the change came
     *  from its own widgets, which already show it. */
    void updatePayload(const QString &id, const QJsonObject &payload, bool notify = true);
    /** @brief The state of the generator @p id, empty when there is none. */
    QJsonObject payload(const QString &id) const;
    /** @brief Every generator in the session: {"id", "name", payload…}. */
    QJsonArray generators() const;
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
