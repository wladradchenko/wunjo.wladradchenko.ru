/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "plugincatalog.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QStandardPaths>
#include <QUrl>

PluginCatalog &PluginCatalog::instance()
{
    static PluginCatalog catalog;
    return catalog;
}

PluginCatalog::PluginCatalog()
    : m_net(new QNetworkAccessManager(this))
{
    // what the site said last time, until it says otherwise
    QFile cache(cacheFile());
    if (cache.open(QIODevice::ReadOnly)) {
        parse(cache.readAll());
    }
}

QString PluginCatalog::siteUrl()
{
    const QString site = QString::fromLocal8Bit(qgetenv("WUNJO_SITE")).trimmed();
    return site.isEmpty() ? QStringLiteral("https://wunjo.online") : site;
}

QString PluginCatalog::cacheFile() const
{
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dir);
    return dir + QStringLiteral("/plugin-catalog.json");
}

void PluginCatalog::refresh()
{
    if (m_fetching) {
        return;
    }
    m_fetching = true;
    QNetworkRequest request(QUrl(siteUrl() + QStringLiteral("/product/plugins.json")));
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setTransferTimeout(15000);
    QNetworkReply *reply = m_net->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        m_fetching = false;
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            return; // offline, or the site is down: the kept list stands
        }
        const QByteArray json = reply->readAll();
        if (!parse(json)) {
            return;
        }
        QFile cache(cacheFile());
        if (cache.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            cache.write(json);
        }
        Q_EMIT changed();
    });
}

bool PluginCatalog::parse(const QByteArray &json)
{
    const QJsonObject root = QJsonDocument::fromJson(json).object();
    const QJsonArray plugins = root.value(QStringLiteral("plugins")).toArray();
    if (plugins.isEmpty()) {
        return false;
    }
    QList<Entry> entries;
    for (const QJsonValue &value : plugins) {
        const QJsonObject obj = value.toObject();
        Entry entry;
        entry.id = obj.value(QStringLiteral("id")).toString();
        entry.name = obj.value(QStringLiteral("name")).toString(entry.id);
        entry.description = obj.value(QStringLiteral("description")).toString();
        entry.url = obj.value(QStringLiteral("url")).toString();
        entry.version = obj.value(QStringLiteral("version")).toString();
        const QJsonArray os = obj.value(QStringLiteral("os")).toArray();
        for (const QJsonValue &name : os) {
            entry.os << name.toString();
        }
        const QJsonArray topics = obj.value(QStringLiteral("topics")).toArray();
        for (const QJsonValue &topic : topics) {
            entry.topics << topic.toString();
        }
        if (!entry.id.isEmpty() && !entry.url.isEmpty()) {
            entries.append(entry);
        }
    }
    m_entries = entries;
    return true;
}
