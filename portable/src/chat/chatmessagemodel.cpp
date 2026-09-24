/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chatmessagemodel.h"

#include <QJsonObject>

#include <algorithm>

ChatMessageModel::ChatMessageModel(QObject *parent)
    : QAbstractListModel(parent)
{
}

int ChatMessageModel::rowCount(const QModelIndex &parent) const
{
    return parent.isValid() ? 0 : m_messages.size();
}

QVariant ChatMessageModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_messages.size()) {
        return {};
    }
    const ChatMessage &m = m_messages.at(index.row());
    switch (role) {
    case TextRole:
    case Qt::DisplayRole:
        return m.text;
    case AuthorRole:
        return int(m.author);
    case KindRole:
        return int(m.kind);
    case ToolIdRole:
        return m.toolId;
    case ToolNameRole:
        return m.toolName;
    case ToolStatusRole:
        return int(m.toolStatus);
    case ToolProgressRole:
        return m.toolProgress;
    case TimestampRole:
        return m.timestamp;
    }
    return {};
}

QHash<int, QByteArray> ChatMessageModel::roleNames() const
{
    return {{TextRole, "text"},         {AuthorRole, "author"},         {KindRole, "kind"},
            {ToolIdRole, "toolId"},     {ToolNameRole, "toolName"},     {ToolStatusRole, "toolStatus"},
            {ToolProgressRole, "toolProgress"}, {TimestampRole, "timestamp"}};
}

void ChatMessageModel::appendText(ChatMessage::Author author, const QString &text)
{
    ChatMessage m;
    m.author = author;
    m.kind = ChatMessage::Kind::Text;
    m.text = text;
    m.timestamp = QDateTime::currentDateTime();
    beginInsertRows(QModelIndex(), m_messages.size(), m_messages.size());
    m_messages.append(m);
    endInsertRows();
}

void ChatMessageModel::updateLastText(ChatMessage::Author author, const QString &text)
{
    if (!m_messages.isEmpty()) {
        ChatMessage &m = m_messages.last();
        if (m.kind == ChatMessage::Kind::Text && m.author == author) {
            m.text = text;
            const QModelIndex ix = index(m_messages.size() - 1);
            Q_EMIT dataChanged(ix, ix, {TextRole});
            return;
        }
    }
    appendText(author, text);
}

void ChatMessageModel::appendToolCard(const QString &id, const QString &name)
{
    ChatMessage m;
    m.author = ChatMessage::Author::Assistant;
    m.kind = ChatMessage::Kind::ToolCard;
    m.toolId = id;
    m.toolName = name;
    m.timestamp = QDateTime::currentDateTime();
    beginInsertRows(QModelIndex(), m_messages.size(), m_messages.size());
    m_messages.append(m);
    endInsertRows();
}

void ChatMessageModel::endRunningTools(const QString &statusText, const QStringList &keptPrefixes)
{
    for (int row = 0; row < m_messages.size(); ++row) {
        ChatMessage &m = m_messages[row];
        if (m.kind != ChatMessage::Kind::ToolCard || m.toolStatus != ChatMessage::ToolStatus::Running) {
            continue;
        }
        const bool kept = std::any_of(keptPrefixes.cbegin(), keptPrefixes.cend(),
                                      [&m](const QString &prefix) { return !prefix.isEmpty() && m.toolId.startsWith(prefix); });
        if (kept) {
            continue;
        }
        m.toolStatus = ChatMessage::ToolStatus::Failed;
        m.text = statusText;
        const QModelIndex ix = index(row);
        Q_EMIT dataChanged(ix, ix, {ToolStatusRole, TextRole});
    }
}

void ChatMessageModel::updateTool(const QString &id, ChatMessage::ToolStatus status, int progress, const QString &statusText)
{
    for (int row = m_messages.size() - 1; row >= 0; --row) {
        ChatMessage &m = m_messages[row];
        if (m.kind == ChatMessage::Kind::ToolCard && m.toolId == id) {
            m.toolStatus = status;
            if (progress >= 0) {
                m.toolProgress = progress;
            }
            if (!statusText.isEmpty()) {
                m.text = statusText;
            }
            const QModelIndex ix = index(row);
            Q_EMIT dataChanged(ix, ix, {ToolStatusRole, ToolProgressRole, TextRole});
            return;
        }
    }
}

void ChatMessageModel::clear()
{
    beginResetModel();
    m_messages.clear();
    endResetModel();
}

QJsonArray ChatMessageModel::toJson() const
{
    QJsonArray array;
    for (const ChatMessage &m : m_messages) {
        QJsonObject o;
        o.insert(QStringLiteral("author"), int(m.author));
        o.insert(QStringLiteral("kind"), int(m.kind));
        o.insert(QStringLiteral("text"), m.text);
        if (m.kind == ChatMessage::Kind::ToolCard) {
            o.insert(QStringLiteral("toolId"), m.toolId);
            o.insert(QStringLiteral("toolName"), m.toolName);
            o.insert(QStringLiteral("toolStatus"), int(m.toolStatus));
            o.insert(QStringLiteral("toolProgress"), m.toolProgress);
        }
        o.insert(QStringLiteral("timestamp"), m.timestamp.toString(Qt::ISODate));
        array.append(o);
    }
    return array;
}

void ChatMessageModel::loadJson(const QJsonArray &array)
{
    beginResetModel();
    m_messages.clear();
    for (const QJsonValue &value : array) {
        const QJsonObject o = value.toObject();
        ChatMessage m;
        m.author = ChatMessage::Author(o.value(QStringLiteral("author")).toInt());
        m.kind = ChatMessage::Kind(o.value(QStringLiteral("kind")).toInt());
        m.text = o.value(QStringLiteral("text")).toString();
        m.toolId = o.value(QStringLiteral("toolId")).toString();
        m.toolName = o.value(QStringLiteral("toolName")).toString();
        m.toolStatus = ChatMessage::ToolStatus(o.value(QStringLiteral("toolStatus")).toInt());
        m.toolProgress = o.value(QStringLiteral("toolProgress")).toInt(-1);
        m.timestamp = QDateTime::fromString(o.value(QStringLiteral("timestamp")).toString(), Qt::ISODate);
        m_messages.append(m);
    }
    endResetModel();
}
