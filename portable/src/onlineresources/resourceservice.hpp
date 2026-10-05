/*
    SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "providermodel.hpp"

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QPair>
#include <QUrl>

#include <functional>

class FileDownloadJob;

/** @class ResourceService
    @brief The Online Resources services without the tab: what the assistant
    searches and imports through, and the one place a file of these services is
    downloaded into the project.

    The searches have objects of their own. The tab's objects keep the page the
    user is looking at, and a plugin's list keeps its paging cursors in them; a
    search sent from here through those would change what the user sees under
    their hands.

    Searching and importing take a while, so each call answers with a request
    number at once and the outcome is read later with that number — the same
    way a card's price is asked for and read.
 */
class ResourceService : public QObject
{
    Q_OBJECT
public:
    explicit ResourceService(QObject *parent = nullptr);

    /** @brief Every service: id, name, kind, whether it is a plugin's own files,
     *  whether it can be used now and what to know about it. */
    QJsonArray services();
    /** @brief Start a search. @p from and @p to (yyyy-MM-dd, empty for no limit)
     *  only apply to a plugin's files, @p query only to a stock library. */
    int search(const QString &service, const QString &query, int page, const QString &from, const QString &to);
    QJsonObject searchAnswer(int request);
    /** @brief Download an item of a previous search into the project and add it
     *  to the bin. @p version picks one of a stock video's sizes. */
    int importItem(const QString &service, const QString &itemId, const QString &version);
    QJsonObject importAnswer(int request);

    using Fetched = std::function<void(const QString &path, const QString &error)>;
    using Progress = std::function<void(int percent)>;
    /** @brief Download @p url to @p dest unless it is there already. One file is
     *  downloaded once however many ask for it; each asker hears the outcome:
     *  the path, or an empty path with the reason (empty when the user
     *  cancelled). */
    void fetch(const QUrl &url, const QString &dest, const Fetched &then, const Progress &progress = {});
    /** @brief Where a file of a plugin's list is kept in the project. */
    static QString libraryTarget(const QString &fileName, const QString &id, const QString &contentType, const QString &url);

Q_SIGNALS:
    void addLicenseInfo(const QString &text);

private:
    struct Service {
        QString id;
        QString key;
        QString name;
        bool library = false;
        bool isValid() const { return !key.isEmpty(); }
    };
    struct FilesRequest {
        int request = 0;
        QString itemId;
        QString version;
    };

    QList<Service> list() const;
    Service find(const QString &id) const;
    ProviderModel *model(const Service &service);
    void dropModel(const QString &id);
    void dropAll();
    void onSearchDone(const QString &id, const QList<ResourceItemInfo> &list, int pageCount);
    void onSearchError(const QString &id, const QString &message);
    void onFetchedFiles(const QString &id, const QStringList &urls, const QStringList &labels);
    void startImport(int request, const Service &service, const ResourceItemInfo &item, const QString &url, const QString &version, bool preview);
    void fail(QHash<int, QJsonObject> &answers, int request, const QString &message);
    QJsonObject itemJson(const Service &service, const ResourceItemInfo &item);
    static int pickVersion(const QStringList &labels, const QString &version, QString *error);

    int m_next = 0;
    QHash<QString, ProviderModel *> m_models;
    /** @brief Service id → the search running on it. One at a time per service. */
    QHash<QString, int> m_searching;
    QHash<QString, FilesRequest> m_filesRequests;
    /** @brief Every item a search answered with, for importing it later. */
    QHash<QString, QHash<QString, ResourceItemInfo>> m_items;
    /** @brief The dates a plugin's list was last asked with: further pages must
     *  be asked with the same. */
    QHash<QString, QPair<QString, QString>> m_lastDates;
    QHash<int, QJsonObject> m_searchAnswers;
    QHash<int, QJsonObject> m_importAnswers;
    QHash<QString, FileDownloadJob *> m_downloads;
    QHash<QString, QList<Fetched>> m_waiting;
    QHash<QString, QList<Progress>> m_progress;
};
