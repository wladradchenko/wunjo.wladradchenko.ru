/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QList>
#include <QObject>
#include <QString>
#include <QStringList>

class PluginManifest;
class QNetworkAccessManager;

/** @class PluginCatalog
    @brief The plugins wunjo.online offers, as the site lists them.

    The site publishes `product/plugins.json`; this reads it so the settings
    page can show what could be added next to what is installed. Only names
    and descriptions travel: what a plugin costs and how it is bought stay on
    the site, where the page a row opens says so. The list is kept from the
    last time the site answered, so it is there without a connection too.
 */
class PluginCatalog : public QObject
{
    Q_OBJECT
public:
    struct Entry
    {
        QString id;
        QString name;
        QString description;
        QString url; ///< the plugin's page on the site
        QString version;
        QStringList os;     ///< "linux", "windows", "macos"; empty = all
        QStringList topics; ///< what it works on, in the site's words
        /** @brief The application versions the plugin's current release runs
         *  on, as its manifest's `min_app_version` / `max_app_version`. Empty
         *  means no bound on that side. */
        QString minAppVersion;
        QString maxAppVersion;
        /** @brief True when this application is inside that range: a plugin
         *  that would refuse to install here is not worth recommending. */
        bool fitsThisApp() const;
    };

    static PluginCatalog &instance();
    /** @brief The site the catalogue and the pages come from. The address is
     *  overridden by WUNJO_SITE, so a copy of the site on this machine can be
     *  tried without touching the code. */
    static QString siteUrl();
    QList<Entry> entries() const { return m_entries; }
    /** @brief The site's entry for @p plugin when it offers a newer release
     *  than the one installed, one that runs on this system and this
     *  application; otherwise an entry with an empty id. Read from the list
     *  kept since the settings last asked the site: nothing is fetched here. */
    Entry updateFor(const PluginManifest &plugin) const;
    /** @brief Ask the site again. @ref changed follows when it answers. */
    void refresh();

Q_SIGNALS:
    void changed();

private:
    PluginCatalog();
    bool parse(const QByteArray &json);
    QString cacheFile() const;

    QNetworkAccessManager *m_net;
    QList<Entry> m_entries;
    bool m_fetching = false;
};
