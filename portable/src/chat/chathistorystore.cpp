/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chathistorystore.h"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QUuid>

#include <algorithm>

ChatHistoryStore::ChatHistoryStore(QObject *parent)
    : QObject(parent)
{
}

void ChatHistoryStore::setProjectFolder(const QString &projectDataFolder)
{
    if (projectDataFolder.isEmpty()) {
        m_folder.clear();
        return;
    }
    m_folder = projectDataFolder + QStringLiteral("/chats");
}

bool ChatHistoryStore::isValid() const
{
    return !m_folder.isEmpty();
}

QString ChatHistoryStore::sessionPath(const QString &id) const
{
    return m_folder + QLatin1Char('/') + id + QStringLiteral(".json");
}

QList<ChatSessionInfo> ChatHistoryStore::sessions() const
{
    QList<ChatSessionInfo> result;
    if (!isValid()) {
        return result;
    }
    const QDir dir(m_folder);
    const QStringList files = dir.entryList({QStringLiteral("*.json")}, QDir::Files);
    for (const QString &file : files) {
        QFile f(dir.absoluteFilePath(file));
        if (!f.open(QIODevice::ReadOnly)) {
            continue;
        }
        const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
        ChatSessionInfo info;
        info.id = o.value(QStringLiteral("id")).toString(file.chopped(5));
        info.title = o.value(QStringLiteral("title")).toString();
        info.updatedAt = QDateTime::fromString(o.value(QStringLiteral("updatedAt")).toString(), Qt::ISODate);
        result.append(info);
    }
    std::sort(result.begin(), result.end(), [](const ChatSessionInfo &a, const ChatSessionInfo &b) { return a.updatedAt > b.updatedAt; });
    return result;
}

QString ChatHistoryStore::createSessionId() const
{
    return QUuid::createUuid().toString(QUuid::WithoutBraces).left(8) + QLatin1Char('-') +
           QString::number(QDateTime::currentSecsSinceEpoch());
}

QJsonArray ChatHistoryStore::loadSession(const QString &id, QString *title) const
{
    if (!isValid()) {
        return {};
    }
    QFile f(sessionPath(id));
    if (!f.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
    if (title) {
        *title = o.value(QStringLiteral("title")).toString();
    }
    return o.value(QStringLiteral("messages")).toArray();
}

void ChatHistoryStore::saveSession(const QString &id, const QString &title, const QJsonArray &messages) const
{
    if (!isValid() || id.isEmpty() || messages.isEmpty()) {
        return;
    }
    QDir().mkpath(m_folder);
    QJsonObject o;
    o.insert(QStringLiteral("id"), id);
    o.insert(QStringLiteral("title"), title);
    o.insert(QStringLiteral("updatedAt"), QDateTime::currentDateTime().toString(Qt::ISODate));
    o.insert(QStringLiteral("messages"), messages);
    QFile f(sessionPath(id));
    if (f.open(QIODevice::WriteOnly)) {
        f.write(QJsonDocument(o).toJson(QJsonDocument::Compact));
    }
}

void ChatHistoryStore::deleteSession(const QString &id) const
{
    if (isValid() && !id.isEmpty()) {
        QFile::remove(sessionPath(id));
    }
}
