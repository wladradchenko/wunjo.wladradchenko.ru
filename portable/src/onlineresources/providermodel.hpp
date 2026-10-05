/*
    SPDX-FileCopyrightText: 2021 Julius Künzel <julius.kuenzel@kde.org>
    SPDX-FileCopyrightText: 2011 Jean-Baptiste Mardelle <jb@kdenlive.org>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QDate>
#include <QHash>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QOAuth2AuthorizationCodeFlow>
#include <QOAuthHttpServerReplyHandler>
#include <QObject>
#include <QPixmap>
#include <QTemporaryFile>

struct ResourceItemInfo
{
    QString fileType;
    QString name;
    QString description;
    QString id;
    QString infoUrl;
    QString license;
    QString author;
    QString authorUrl;
    int width;
    int height;
    int duration;
    QString downloadUrl;
    QString filetype;
    QStringList downloadUrls;
    QStringList downloadLabels;
    QString imageUrl;
    QString previewUrl;
    // only filled for a plugin's library: when it was made, what it is, the
    // name the file is saved under, which tool made it and whether it is ready
    QString date;
    QString contentType;
    QString fileName;
    QString group;
    QString status;
    /** @brief What it cost, -1 when the list does not say. */
    int price = -1;
    // int filesize;
};

class ProviderModel : public QObject
{
    Q_OBJECT
public:
    enum SERVICETYPE { UNKNOWN = 0, AUDIO = 1, VIDEO = 2, IMAGE = 3 };
    ProviderModel() = delete;
    /** @param pluginId set when the description comes from an installed
     *  plugin: the list is then that plugin's own files, the key and the address
     *  are read from the plugin's settings. */
    ProviderModel(const QString &path, const QString &pluginId = QString());

    void authorize();
    void refreshAccessToken();
    bool is_valid() const;
    QString name() const;
    QString homepage() const;
    ProviderModel::SERVICETYPE type() const;
    QString attribution() const;
    bool downloadOAuth2() const;
    bool requiresLogin() const;
    /** @brief The list is a plugin's files in its own cloud, not a stock library. */
    bool isLibrary() const;
    QString pluginId() const;
    /** @brief Limits the next requests to these days; an invalid date is no limit. */
    void setDateRange(const QDate &from, const QDate &to);
    /** @brief The plugin's tools by id, with the names the list shows. */
    QMap<QString, QString> groups() const;
    /** @brief Whether the plugin's key has been entered. */
    bool hasKey() const;
    /** @brief The server's host name, for messages. */
    QString host() const;
    /** @brief Whether @p page can be asked for now: a list paged by the date of
     *  its last item reaches page N only through page N-1. */
    bool canRequestPage(int page) const;
    /** @brief The search gives no file links and a second request does. */
    bool hasFilesRequest() const;

public Q_SLOTS:
    void slotStartSearch(const QString &searchText, int page);
    void slotFetchFiles(const QString &id);
    // void slotShowResults(QNetworkReply *reply);

protected:
    QOAuthHttpServerReplyHandler *m_replyHandler;
    QOAuth2AuthorizationCodeFlow m_oauth2;
    QString m_path;
    QString m_name;
    QString m_homepage;
    SERVICETYPE m_type;
    QString m_clientkey;
    QString m_attribution;
    bool m_invalid;
    QJsonDocument m_doc;
    QString m_apiroot;
    QJsonObject m_search;
    QJsonObject m_download;
    QNetworkAccessManager *m_networkManager;

private:
    bool m_oauth2InitDone = false;

    void validate();
    void initOAuth2();
    QUrl getSearchUrl(const QString &searchText, const int page = 1);
    QUrl getFilesUrl(const QString &id);
    QJsonValue objectGetValue(QJsonObject item, QString key);
    QString objectGetString(QJsonObject item, const QString &key, const QString &id = QString(), const QString &parentKey = QString());
    QString replacePlaceholders(QString string, const QString &query = QString(), const int page = 0, const QString &id = QString());
    /** @brief api.root, read again for every request of a plugin's library. */
    QString apiRoot() const;
    QString pluginKey() const;
    std::pair<QList<ResourceItemInfo>, const int> parseSearchResponse(const QByteArray &res, int page = 1);
    std::pair<QStringList, QStringList> parseFilesResponse(const QByteArray &res, const QString &id);
    QTemporaryFile *m_tmpThumbFile;
    int m_perPage = 15;
    QString m_pluginId;
    QDate m_from;
    QDate m_to;
    /** @brief For lists paged by the date of the last item: page → the value
     *  that asks for it. Page 1 needs none. */
    QHash<int, QString> m_cursors;
    int m_requestedPage = 1;

Q_SIGNALS:
    void searchDone(QList<ResourceItemInfo> &list, const int pageCount);
    void searchError(const QString &msg = QString());
    void fetchedFiles(QStringList, QStringList, const QString &token = QString());
    void authenticated(const QString &token);
    void usePreview();
    void authorizeWithBrowser(const QUrl &url);
};
