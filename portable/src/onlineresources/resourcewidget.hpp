/*
    SPDX-FileCopyrightText: 2021 Julius Künzel <julius.kuenzel@kde.org>
    SPDX-FileCopyrightText: 2011 Jean-Baptiste Mardelle <jb@kdenlive.org>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "providersrepository.hpp"
#include "ui_resourcewidget_ui.h"

#include <KJob>
#include <QDate>
#include <QElapsedTimer>
#include <QHash>
#include <QListWidgetItem>
#include <QMutex>
#include <QNetworkReply>
#include <QProcess>
#include <QSet>
#include <QSlider>
#include <QTimer>
#include <QUrl>
#include <QWidget>

#include <functional>

class FileDownloadJob;
class KDateComboBox;
class QComboBox;
class QToolButton;

const int imageRole = Qt::UserRole;
const int urlRole = Qt::UserRole + 1;
const int downloadRole = Qt::UserRole + 2;
const int durationRole = Qt::UserRole + 3;
const int previewRole = Qt::UserRole + 4;
const int authorRole = Qt::UserRole + 5;
const int authorUrl = Qt::UserRole + 6;
const int infoUrl = Qt::UserRole + 7;
const int infoData = Qt::UserRole + 8;
const int idRole = Qt::UserRole + 9;
const int licenseRole = Qt::UserRole + 10;
const int descriptionRole = Qt::UserRole + 11;
const int widthRole = Qt::UserRole + 12;
const int heightRole = Qt::UserRole + 13;
const int nameRole = Qt::UserRole + 14;
const int singleDownloadRole = Qt::UserRole + 15;
const int filetypeRole = Qt::UserRole + 16;
const int downloadLabelRole = Qt::UserRole + 17;
// a plugin's library only
const int dateRole = Qt::UserRole + 18;
const int contentTypeRole = Qt::UserRole + 19;
const int fileNameRole = Qt::UserRole + 20;
const int groupRole = Qt::UserRole + 21;
const int statusRole = Qt::UserRole + 22;

class ResourceWidget : public QWidget, public Ui::ResourceWidget_UI
{
    Q_OBJECT

public:
    explicit ResourceWidget(QWidget *parent = nullptr);
    ~ResourceWidget() override;
    /** @brief The editor has finished starting: from now on showing the tab
     *  loads a plugin's list. Before, the tab may be on screen without anyone
     *  having asked for it, and nothing goes to the network unasked. */
    void started();

protected:
    void showEvent(QShowEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;

private Q_SLOTS:
    void slotChangeProvider();
    void slotOpenUrl(const QString &url);
    void slotStartSearch();
    void slotLoadImages();
    void slotShowPixmap(const QString &url, const QPixmap &pixmap);
    void slotSearchFinished(const QList<ResourceItemInfo> &list, const int pageCount);
    void slotDisplayError(const QString &message);
    void slotUpdateCurrentItem();
    void slotSetIconSize(int size);
    void slotPreviewItem();
    void slotChooseVersion(const QStringList &urls, const QStringList &labels, const QString &accessToken = QString());
    void slotSaveItem(const QString &originalUrl = QString(), const QString &accessToken = QString());
    void slotGotFile(KJob *job);
    void slotAccessTokenReceived(const QString &accessToken);
    void abortDownload();
    /** @brief Rebuild the services after a plugin was installed or removed. */
    void slotPluginsChanged();

private:
    std::unique_ptr<ProviderModel> *m_currentProvider{nullptr};
    QListWidgetItem *m_currentItem{nullptr};
    QStringList m_imagesUrl;
    QMutex m_imageLock;
    /** @brief Default icon size for the views. */
    QSize m_iconSize;
    int wheelAccumulatedDelta;
    bool m_showloadingWarning;
    QSet<QNetworkReply *> m_activeImageReplies;
    QSet<QTimer *> m_imageBackoffTimers;
    int m_backoff = 0;
    QElapsedTimer m_backoffCooldownTimer;
    QAction *m_stopAction;
    QNetworkAccessManager *m_networkManager{nullptr};
    ResourceItemInfo getItemById(const QString &id);
    void loadConfig();
    void saveConfig();
    void blockUI(bool block);
    QString licenseNameFromUrl(const QString &licenseUrl, const bool shortName);
    void downloadImage(const QString &url, QSharedPointer<QMap<QString, int>> retryCount);
    void fillServices();

    /* A plugin's library: the user's own files in the plugin's cloud. The list
       comes with links only; a file is downloaded into the project when it is
       watched, imported or dragged, and never twice. */
    bool isLibrary() const;
    /** @brief Ask for page 1 again, with the dates on the bar. */
    void reloadLibrary();
    /** @brief Ask for the list when it was never loaded for this service. */
    void loadIfEmpty();
    void showLibraryList(const QList<ResourceItemInfo> &list, int pageCount);
    QString libraryRowText(const QListWidgetItem *item) const;
    QString toolName(const QString &group) const;
    /** @brief Hide the rows the tool filter and the search text leave out. */
    void applyFilter();
    QListWidgetItem *itemById(const QString &id) const;
    /** @brief Where the item's file is kept in the project. */
    QString libraryTarget(const QListWidgetItem *item) const;
    /** @brief Download the item's file unless it is already there, then @p then. */
    void fetchLibraryItem(QListWidgetItem *item, const std::function<void(const QString &)> &then);
    void previewLibraryItem(QListWidgetItem *item);
    void importLibraryItem(QListWidgetItem *item);
    void dragLibraryItem(QListWidgetItem *item);
    void showNote(const QString &text, KMessageWidget::MessageType type);
    /** @brief First frames of the videos, taken by ffmpeg from the links. */
    void requestThumbnails();
    void nextThumbnail();
    void stopThumbnails();
    QString thumbnailPath(const QString &id) const;

    QWidget *m_libraryBar{nullptr};
    KDateComboBox *m_from{nullptr};
    KDateComboBox *m_to{nullptr};
    QComboBox *m_group{nullptr};
    QToolButton *m_refresh{nullptr};
    QDate m_rangeFrom;
    QDate m_rangeTo;
    /** @brief Picture size of the stock libraries and of a plugin's list, and
     *  which of the two the slider shows now. */
    int m_stockZoom{7};
    int m_libraryZoom{2};
    bool m_zoomForLibrary{false};
    bool m_started{false};
    bool m_searching{false};
    bool m_loaded{false};
    QStringList m_thumbQueue;
    QHash<QString, QString> m_thumbUrls;
    QList<QProcess *> m_thumbProcesses;
    QHash<QString, FileDownloadJob *> m_fetching;
    QHash<QString, std::function<void(const QString &)>> m_afterFetch;
    QString m_pressedId;
    QPoint m_pressPos;

Q_SIGNALS:
    void addClip(const QUrl &, const QString &);
    void addLicenseInfo(const QString &);
    void previewClip(const QString &path, const QString &title);
    void gotPixmap(const QString &url, const QPixmap &pix);
};
