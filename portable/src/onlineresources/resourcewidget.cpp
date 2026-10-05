/*
    SPDX-FileCopyrightText: 2021 Julius Künzel <julius.kuenzel@kde.org>
    SPDX-FileCopyrightText: 2011 Jean-Baptiste Mardelle <jb@kdenlive.org>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "resourcewidget.hpp"
#include "bin/bin.h"
#include "bin/projectitemmodel.h"
#include "core.h"
#include "doc/wunjodoc.h"
#include "plugins/filedownloadjob.h"
#include "plugins/pluginmanager.h"
#include "wunjosettings.h"

#include <KConfigGroup>
#include <KDateComboBox>
#include <KFileItem>

#include <KIO/JobTracker>
#include <KJobTrackerInterface>
#include <KIO/Global>
#include <KLocalizedString>
#include <KMessageBox>
#include <KRecentDirs>
#include <KSelectAction>
#include <KSharedConfig>
#include <KSqueezedTextLabel>
#include <QComboBox>
#include <QApplication>
#include <QDesktopServices>
#include <QDir>
#include <QDrag>
#include <QFileDialog>
#include <QFontDatabase>
#include <QGridLayout>
#include <QIcon>
#include <QInputDialog>
#include <QMenu>
#include <QMimeData>
#include <QMouseEvent>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>

ResourceWidget::ResourceWidget(QWidget *parent)
    : QWidget(parent)
    , m_showloadingWarning(true)
{
    setFont(QFontDatabase::systemFont(QFontDatabase::SmallestReadableFont));
    setupUi(this);

    int iconHeight = int(QFontInfo(font()).pixelSize() * 3.5);
    m_iconSize = QSize(int(iconHeight * pCore->getCurrentDar()), iconHeight);

    slider_zoom->setRange(0, 15);
    connect(slider_zoom, &QAbstractSlider::valueChanged, this, &ResourceWidget::slotSetIconSize);
    connect(button_zoomin, &QToolButton::clicked, this, [&]() { slider_zoom->setValue(qMin(slider_zoom->value() + 1, slider_zoom->maximum())); });
    connect(button_zoomout, &QToolButton::clicked, this, [&]() { slider_zoom->setValue(qMax(slider_zoom->value() - 1, slider_zoom->minimum())); });

    message_line->hide();

    m_stopAction = new QAction(i18n("Abort"), this);
    connect(m_stopAction, &QAction::triggered, this, &ResourceWidget::abortDownload);

    // The bar a plugin's library adds over the list: which days, which tool
    m_libraryBar = new QWidget(this);
    auto *bar = new QGridLayout(m_libraryBar);
    bar->setContentsMargins(0, QFontInfo(font()).pixelSize() / 2, 0, 0);
    m_from = new KDateComboBox(m_libraryBar);
    m_from->setToolTip(i18nc("@info:tooltip first day of a date range", "From"));
    m_from->setDate(QDate());
    m_to = new KDateComboBox(m_libraryBar);
    m_to->setToolTip(i18nc("@info:tooltip last day of a date range", "To"));
    m_to->setDate(QDate());
    m_group = new QComboBox(m_libraryBar);
    m_refresh = new QToolButton(m_libraryBar);
    m_refresh->setIcon(QIcon::fromTheme(QStringLiteral("view-refresh")));
    m_refresh->setToolTip(i18n("Refresh"));
    m_refresh->setAutoRaise(true);
    // A date field squeezed by a narrow tab showed only its arrow: it keeps room
    // for a whole date, and the dates get a row of their own
    for (KDateComboBox *field : {m_from, m_to}) {
        field->setMinimumContentsLength(10);
        field->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    }
    bar->addWidget(m_from, 0, 0);
    bar->addWidget(m_to, 0, 1, 1, 2);
    bar->addWidget(m_group, 1, 0, 1, 2);
    bar->addWidget(m_refresh, 1, 2);
    bar->setColumnStretch(0, 1);
    bar->setColumnStretch(1, 1);
    left_box->insertWidget(1, m_libraryBar);
    m_libraryBar->hide();
    // a date is "entered" also when the field only loses focus: ask again only
    // when the days really changed
    auto datesEntered = [this]() {
        if (m_from->date() != m_rangeFrom || m_to->date() != m_rangeTo) {
            reloadLibrary();
        }
    };
    connect(m_from, &KDateComboBox::dateEntered, this, datesEntered);
    connect(m_to, &KDateComboBox::dateEntered, this, datesEntered);
    connect(m_refresh, &QToolButton::clicked, this, &ResourceWidget::reloadLibrary);
    connect(m_group, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &ResourceWidget::applyFilter);

    fillServices();
    connect(service_list, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &ResourceWidget::slotChangeProvider);
    // picked by the user, not restored from the settings: that may go online
    connect(service_list, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated), this, &ResourceWidget::loadIfEmpty);
    connect(&PluginManager::instance(), &PluginManager::pluginsChanged, this, &ResourceWidget::slotPluginsChanged);
    loadConfig();
    connect(provider_info, &KUrlLabel::leftClickedUrl, this, [&]() { slotOpenUrl(provider_info->url()); });
    connect(label_license, &KUrlLabel::leftClickedUrl, this, [&]() { slotOpenUrl(label_license->url()); });
    connect(search_text, &KLineEdit::returnKeyPressed, this, [this]() {
        if (isLibrary()) {
            applyFilter();
        } else {
            slotStartSearch();
        }
    });
    connect(search_text, &QLineEdit::textChanged, this, &ResourceWidget::applyFilter);
    connect(search_results, &QListWidget::currentRowChanged, this, &ResourceWidget::slotUpdateCurrentItem);
    connect(search_results, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *item) {
        if (isLibrary()) {
            importLibraryItem(item);
        }
    });
    search_results->viewport()->installEventFilter(this);
    connect(this, &ResourceWidget::gotPixmap, this, &ResourceWidget::slotShowPixmap);
    connect(button_preview, &QAbstractButton::clicked, this, [&]() {
        if (!m_currentProvider) {
            return;
        }
        if (isLibrary()) {
            previewLibraryItem(m_currentItem);
            return;
        }

        slotPreviewItem();
    });

    connect(button_import, &QAbstractButton::clicked, this, [&]() {
        if (!m_currentProvider) {
            return;
        }
        if (isLibrary()) {
            importLibraryItem(m_currentItem);
            return;
        }
        if (m_currentProvider->get()->downloadOAuth2()) {
            if (m_currentProvider->get()->requiresLogin()) {
                KMessageBox::information(this, i18n("Login is required to download this item.\nYou will be redirected to the login page now."));
            }
            m_currentProvider->get()->authorize();
        } else {
            if (m_currentItem->data(singleDownloadRole).toBool()) {
                if (m_currentItem->data(downloadRole).toString().isEmpty()) {
                    m_currentProvider->get()->slotFetchFiles(m_currentItem->data(idRole).toString());
                    return;
                } else {
                    slotSaveItem();
                }
            } else {
                slotChooseVersion(m_currentItem->data(downloadRole).toStringList(), m_currentItem->data(downloadLabelRole).toStringList());
            }
        }
    });

    page_number->setEnabled(false);

    connect(page_number, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &ResourceWidget::slotStartSearch);
    adjustSize();
}

ResourceWidget::~ResourceWidget()
{
    stopThumbnails();
    saveConfig();
}

void ResourceWidget::fillServices()
{
    const QString previous = service_list->currentData().toString();
    QSignalBlocker blocker(service_list);
    service_list->clear();
    const QVector<QPair<QString, QString>> providers = ProvidersRepository::get()->getAllProviers();
    // the files of the user's own plugins come before the stock libraries
    for (const bool libraries : {true, false}) {
        for (const QPair<QString, QString> &provider : providers) {
            ProviderModel *model = ProvidersRepository::get()->getProvider(provider.second).get();
            if (model->isLibrary() != libraries) {
                continue;
            }
            QIcon icon;
            if (model->isLibrary()) {
                icon = PluginManager::instance().plugin(model->pluginId()).icon();
            } else {
                switch (model->type()) {
                case ProviderModel::AUDIO:
                    icon = QIcon::fromTheme(QStringLiteral("player-volume"));
                    break;
                case ProviderModel::VIDEO:
                    icon = QIcon::fromTheme(QStringLiteral("camera-video"));
                    break;
                case ProviderModel::IMAGE:
                    icon = QIcon::fromTheme(QStringLiteral("camera-photo"));
                    break;
                default:
                    icon = QIcon();
                }
            }
            service_list->addItem(icon, provider.first, provider.second);
        }
    }
    const int index = previous.isEmpty() ? -1 : service_list->findData(previous);
    if (index >= 0) {
        service_list->setCurrentIndex(index);
    }
}

void ResourceWidget::slotPluginsChanged()
{
    // The repository is about to rebuild every service: nothing may point into it
    stopThumbnails();
    if (m_currentProvider != nullptr) {
        m_currentProvider->get()->disconnect(this);
        m_currentProvider = nullptr;
    }
    m_searching = false;
    blockUI(false);
    ProvidersRepository::get()->refresh();
    fillServices();
    slotChangeProvider();
}

void ResourceWidget::started()
{
    m_started = true;
}

void ResourceWidget::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    if (m_started) {
        QTimer::singleShot(0, this, &ResourceWidget::loadIfEmpty);
    }
}

/**
 * @brief ResourceWidget::saveConfig
 * Load current provider and zoom value from config file
 */
void ResourceWidget::loadConfig()
{
    KSharedConfigPtr config = KSharedConfig::openConfig();
    KConfigGroup resourceConfig(config, "OnlineResources");
    m_stockZoom = resourceConfig.readEntry("zoom", 7);
    m_libraryZoom = resourceConfig.readEntry("libraryZoom", 2);
    slider_zoom->setValue(m_stockZoom);
    if (resourceConfig.readEntry("provider", service_list->itemText(0)).isEmpty()) {
        service_list->setCurrentIndex(0);
    } else {
        service_list->setCurrentItem(resourceConfig.readEntry("provider", service_list->itemText(0)));
    }
    slotChangeProvider();
}

/**
 * @brief ResourceWidget::saveConfig
 * Save current provider and zoom value to config file
 */
void ResourceWidget::saveConfig()
{
    KSharedConfigPtr config = KSharedConfig::openConfig();
    KConfigGroup resourceConfig(config, "OnlineResources");
    resourceConfig.writeEntry(QStringLiteral("provider"), service_list->currentText());
    (m_zoomForLibrary ? m_libraryZoom : m_stockZoom) = slider_zoom->value();
    resourceConfig.writeEntry(QStringLiteral("zoom"), m_stockZoom);
    resourceConfig.writeEntry(QStringLiteral("libraryZoom"), m_libraryZoom);
    config->sync();
}

/**
 * @brief ResourceWidget::blockUI
 * @param block
 * Block or unblock the online resource ui
 */
void ResourceWidget::blockUI(bool block)
{
    m_libraryBar->setEnabled(!block);
    buildin_box->setEnabled(!block);
    search_text->setEnabled(!block);
    service_list->setEnabled(!block);
    setCursor(block ? Qt::WaitCursor : Qt::ArrowCursor);
}

/**
 * @brief ResourceWidget::slotChangeProvider
 * Set m_currentProvider to the current selected provider of the service_list and update ui
 */
void ResourceWidget::slotChangeProvider()
{
    if (m_currentProvider != nullptr) {
        m_currentProvider->get()->disconnect(this);
    }
    stopThumbnails();
    m_searching = false;
    m_loaded = false;

    // Reset backoff when provider changes assuming the new has not put us under rate limit (yet)
    m_backoff = 0;
    m_backoffCooldownTimer.invalidate();

    details_box->setEnabled(false);
    button_import->setEnabled(false);
    button_preview->setEnabled(false);
    info_browser->clear();
    search_results->clear();
    page_number->blockSignals(true);
    page_number->setValue(1);
    page_number->setMaximum(1);
    page_number->blockSignals(false);

    if (service_list->currentData().toString().isEmpty()) {
        provider_info->clear();
        buildin_box->setEnabled(false);
        return;
    } else {
        buildin_box->setEnabled(true);
        message_line->hide();
    }

    m_currentProvider = &ProvidersRepository::get()->getProvider(service_list->currentData().toString());

    const bool library = m_currentProvider->get()->isLibrary();
    // A plugin's list is read by its text, a stock library by its pictures:
    // each keeps a size of its own
    if (library != m_zoomForLibrary) {
        (m_zoomForLibrary ? m_libraryZoom : m_stockZoom) = slider_zoom->value();
        m_zoomForLibrary = library;
        slider_zoom->setValue(library ? m_libraryZoom : m_stockZoom);
    }
    m_libraryBar->setVisible(library);
    label_license->setVisible(!library);
    if (library) {
        provider_info->setText(i18n("Your files from %1", m_currentProvider->get()->name()));
        QSignalBlocker blocker(m_group);
        m_group->clear();
        m_group->addItem(i18n("All"), QString());
        const QMap<QString, QString> groups = m_currentProvider->get()->groups();
        for (auto it = groups.constBegin(); it != groups.constEnd(); ++it) {
            m_group->addItem(i18n(it.value().toUtf8().constData()), it.key());
        }
        if (m_currentProvider->get()->hasKey()) {
            showNote(i18n("Press Refresh to load the list"), KMessageWidget::Information);
        } else {
            showNote(i18n("Add the key in the plugin's settings"), KMessageWidget::Warning);
        }
    } else {
        provider_info->setText(i18n("Media provided by %1", m_currentProvider->get()->name()));
    }
    provider_info->setUrl(m_currentProvider->get()->homepage());
    connect(m_currentProvider->get(), &ProviderModel::searchDone, this, &ResourceWidget::slotSearchFinished);
    connect(m_currentProvider->get(), &ProviderModel::searchError, this, &ResourceWidget::slotDisplayError);

    connect(m_currentProvider->get(), &ProviderModel::fetchedFiles, this, &ResourceWidget::slotChooseVersion);
    connect(m_currentProvider->get(), &ProviderModel::authenticated, this, &ResourceWidget::slotAccessTokenReceived);

    // automatically kick of a search if we have search text and we switch services.
    if (!library && !search_text->text().isEmpty()) {
        slotStartSearch();
    }
}

/**
 * @brief ResourceWidget::slotOpenUrl
 * @param url link to open in external browser
 * Open a url in a external browser
 */
void ResourceWidget::slotOpenUrl(const QString &url)
{
    QDesktopServices::openUrl(QUrl(url));
}

/**
 * @brief ResourceWidget::slotStartSearch
 * Calls slotStartSearch on the object for the currently selected service.
 */
void ResourceWidget::slotStartSearch()
{
    if (m_currentProvider == nullptr) {
        return;
    }
    if (isLibrary()) {
        stopThumbnails();
        if (!m_currentProvider->get()->hasKey()) {
            search_results->clear();
            showNote(i18n("Add the key in the plugin's settings"), KMessageWidget::Warning);
            return;
        }
        m_rangeFrom = m_from->date();
        m_rangeTo = m_to->date();
        m_currentProvider->get()->setDateRange(m_rangeFrom, m_rangeTo);
    }
    m_searching = true;
    // Abort and clear all active image downloads from previous searches
    for (QNetworkReply *reply : std::as_const(m_activeImageReplies)) {
        reply->abort();
        reply->deleteLater();
    }
    m_activeImageReplies.clear();
    // Stop and delete all active delay timers from previous searches
    for (QTimer *timer : std::as_const(m_imageBackoffTimers)) {
        timer->stop();
        timer->deleteLater();
    }
    m_imageBackoffTimers.clear();
    message_line->clearActions();
    message_line->setText(i18nc("@info:status", "Search pending…"));
    message_line->setMessageType(KMessageWidget::Information);
    message_line->addAction(m_stopAction);
    message_line->show();

    blockUI(true);
    details_box->setEnabled(false);
    button_import->setEnabled(false);
    button_preview->setEnabled(false);
    info_browser->clear();
    search_results->clear();
    m_currentProvider->get()->slotStartSearch(search_text->text(), page_number->value());
}

/**
 * @brief ResourceWidget::slotDisplayError
 * @param message the error text
 * Displays the items of list in the search_results ListView
 */
void ResourceWidget::slotDisplayError(const QString &message)
{
    m_searching = false;
    message_line->clearActions();
    // a plugin's library says in a whole sentence what went wrong
    message_line->setText(isLibrary() ? message : i18n("Search failed! %1", message));
    message_line->setMessageType(KMessageWidget::Error);
    message_line->show();
    page_number->setEnabled(isLibrary() && page_number->maximum() > 1);
    service_list->setEnabled(true);
    buildin_box->setEnabled(true);
    search_text->setEnabled(true);
    m_libraryBar->setEnabled(true);
    setCursor(Qt::ArrowCursor);
}

/**
 * @brief ResourceWidget::slotSearchFinished
 * @param list list of the found items
 * @param pageCount number of found pages
 * Displays the items of list in the search_results ListView
 */
void ResourceWidget::slotSearchFinished(const QList<ResourceItemInfo> &list, int pageCount)
{
    if (isLibrary()) {
        if (m_searching) {
            showLibraryList(list, pageCount);
        }
        return;
    }
    m_searching = false;
    QMutexLocker lock(&m_imageLock);
    m_imagesUrl.clear();
    if (list.isEmpty()) {
        message_line->setText(i18nc("@info", "No items found."));
        message_line->setMessageType(KMessageWidget::Error);
        message_line->show();
        blockUI(false);
        return;
    }

    message_line->setMessageType(KMessageWidget::Information);
    message_line->show();
    int count = 0;
    for (const ResourceItemInfo &item : std::as_const(list)) {
        message_line->setText(i18nc("@info:progress", "Parsing item %1 of %2…", count, list.count()));
        // if item has no name use "Created by Author", if item even has no author use "Unnamed"
        QListWidgetItem *listItem = new QListWidgetItem(
            item.name.isEmpty() ? (item.author.isEmpty() ? i18n("Unnamed") : i18nc("Created by author name", "Created by %1", item.author)) : item.name);
        if (!item.imageUrl.isEmpty()) {
            m_imagesUrl << item.imageUrl;
        }

        listItem->setData(idRole, item.id);
        listItem->setData(nameRole, item.name);
        listItem->setData(filetypeRole, item.filetype);
        listItem->setData(descriptionRole, item.description);
        listItem->setData(imageRole, item.imageUrl);
        listItem->setData(previewRole, item.previewUrl);
        listItem->setData(authorUrl, item.authorUrl);
        listItem->setData(authorRole, item.author);
        listItem->setData(widthRole, item.width);
        listItem->setData(heightRole, item.height);
        listItem->setData(durationRole, item.duration);
        listItem->setData(urlRole, item.infoUrl);
        listItem->setData(licenseRole, item.license);
        if (item.downloadUrl.isEmpty() && item.downloadUrls.length() > 0) {
            listItem->setData(singleDownloadRole, false);
            listItem->setData(downloadRole, item.downloadUrls);
            listItem->setData(downloadLabelRole, item.downloadLabels);
        } else {
            listItem->setData(singleDownloadRole, true);
            listItem->setData(downloadRole, item.downloadUrl);
        }
        search_results->addItem(listItem);
        count++;
    }
    m_imagesUrl.removeDuplicates();
    message_line->hide();
    page_number->setMaximum(pageCount);
    page_number->setEnabled(true);
    blockUI(false);
    lock.unlock();
    slotLoadImages();
}

void ResourceWidget::slotShowPixmap(const QString &url, const QPixmap &pixmap)
{
    for (int i = 0; i < search_results->count(); i++) {
        auto item = search_results->item(i);
        if (item->data(imageRole).toString() == url) {
            item->setIcon(pixmap);
            break;
        }
    }
}

void ResourceWidget::abortDownload()
{
    message_line->clearActions();
    if (isLibrary()) {
        // the answer that still comes is thrown away
        m_searching = false;
        message_line->hide();
        blockUI(false);
        return;
    }
    slotSearchFinished({}, 1);
    delete m_networkManager;
    m_networkManager = nullptr;
}

void ResourceWidget::slotLoadImages()
{
    if (!m_imageLock.tryLock()) {
        // Another request is loading, wait
        return;
    }
    if (m_imagesUrl.isEmpty()) {
        // No images to load
        m_imageLock.unlock();
        return;
    }
    delete m_networkManager;
    m_networkManager = new QNetworkAccessManager(this);
    qDebug() << "Starting image download for" << m_imagesUrl.size() << "URLs";
    QSharedPointer<QMap<QString, int>> retryCount(new QMap<QString, int>());

    // Make a copy of the URLs before we start processing
    QStringList urls = m_imagesUrl;
    // Clear the original list to avoid reprocessing
    m_imagesUrl.clear();
    m_imageLock.unlock();

    // Start downloads for all URLs
    for (const QString &url : urls) {
        retryCount->insert(url, 0);
        downloadImage(url, retryCount);
    }
}

void ResourceWidget::downloadImage(const QString &url, QSharedPointer<QMap<QString, int>> retryCount)
{
    // Apply delay if needed (for handling rate limiting)
    if (m_backoff > 0) {
        QTimer *timer = new QTimer(this);
        timer->setSingleShot(true);
        connect(timer, &QTimer::timeout, this, [this, url, retryCount, timer]() {
            m_imageBackoffTimers.remove(timer);
            timer->deleteLater();
            downloadImage(url, retryCount);
        });
        m_imageBackoffTimers.insert(timer);
        timer->start(m_backoff);
        return;
    }

    QNetworkRequest request(QUrl::fromUserInput(url));
    QNetworkReply *reply = m_networkManager->get(request);
    m_activeImageReplies.insert(reply);

    connect(
        reply, &QNetworkReply::finished, this,
        [this, reply, url, retryCount]() {
            m_activeImageReplies.remove(reply);
            if (reply->error() == QNetworkReply::NoError) {
                QByteArray imageData = reply->readAll();
                QPixmap pixmap;
                if (pixmap.loadFromData(imageData)) {
                    Q_EMIT gotPixmap(url, pixmap);
                } else {
                    qDebug() << "Failed to load image from URL:" << url << "- Invalid image format";
                }
                // Decay backoff after successful download
                if (m_backoff > 0) {
                    m_backoff = m_backoff / 2;
                    if (m_backoff < 1000) m_backoff = 0;
                    qDebug() << "Backoff decreased to" << m_backoff << "ms after successful download";
                }
            } else if (reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 429) {
                // Rate limited, retry with exponential backoff
                int attempts = retryCount->value(url);
                if (attempts < 3) {
                    retryCount->insert(url, attempts + 1);
                    // Exponential backoff: 1s, 3s, 7s (2**attempts - 1)
                    int backoffSingleImageCurrentAttempt = 1000 * ((1 << (attempts + 1)) - 1);
                    int newBackoff = qMin(qMax(m_backoff * 2, backoffSingleImageCurrentAttempt), 15000); // cap at 15s
                    // Only increase backoff if the last backoff period has expired
                    if (!m_backoffCooldownTimer.isValid() || m_backoffCooldownTimer.elapsed() > m_backoff) {
                        if (newBackoff > m_backoff) {
                            m_backoff = newBackoff;
                            m_backoffCooldownTimer.restart();
                            qDebug() << "Backoff increased to" << m_backoff << "ms due to 429 on URL:" << url;
                        }
                    } else {
                        qDebug() << "Backoff not increased due to recent increase, current backoff:" << m_backoff;
                    }
                    qDebug() << "Rate limited, retrying URL:" << url << "attempt:" << attempts + 1 << "with delay:" << m_backoff;
                    downloadImage(url, retryCount);
                } else {
                    qDebug() << "Maximum retry attempts reached for URL:" << url << "giving up";
                }
            } else {
                qDebug() << "Error downloading image from URL:" << url << "- Error:" << reply->errorString();
            }
            reply->deleteLater();
        },
        Qt::DirectConnection);
}

/**
 * @brief ResourceWidget::slotUpdateCurrentItem
 * Set m_currentItem to the current selected item of the search_results ListView and
 * show its details within the details_box
 */
void ResourceWidget::slotUpdateCurrentItem()
{
    details_box->setEnabled(false);
    button_import->setEnabled(false);
    button_preview->setEnabled(false);

    // get the item the user selected
    m_currentItem = search_results->currentItem();
    if (!m_currentItem) {
        return;
    }

    if (isLibrary()) {
        const bool ready = m_currentItem->data(statusRole).toString() == QLatin1String("done") && !m_currentItem->data(downloadRole).toString().isEmpty();
        const QString tool = toolName(m_currentItem->data(groupRole).toString());
        const QStringList lines = libraryRowText(m_currentItem).split(QLatin1Char('\n'));
        QString details = QStringLiteral("<h3>%1</h3>").arg(tool.toHtmlEscaped());
        details.append(lines.value(1).toHtmlEscaped() + QStringLiteral("<br /><br />"));
        details.append(m_currentItem->data(descriptionRole).toString().toHtmlEscaped().replace(QLatin1Char('\n'), QStringLiteral("<br />")));
        info_browser->setHtml(details);
        button_preview->show();
        details_box->setEnabled(true);
        button_import->setEnabled(ready);
        button_preview->setEnabled(ready);
        return;
    }

    if (m_currentProvider->get()->type() != ProviderModel::IMAGE && !m_currentItem->data(previewRole).toString().isEmpty()) {
        button_preview->show();
    } else {
        button_preview->hide();
    }

    QString details = "<h3>" + m_currentItem->text();

    if (!m_currentItem->data(urlRole).toString().isEmpty()) {
        details += QStringLiteral(" <a href=\"%1\">%2</a>").arg(m_currentItem->data(urlRole).toString(), i18nc("the url link pointing to a web page", "link"));
    }

    details.append(QStringLiteral("</h3>"));

    if (!m_currentItem->data(authorUrl).toString().isEmpty()) {
        details += i18n("Created by <a href=\"%1\">", m_currentItem->data(authorUrl).toString());
        if (!m_currentItem->data(authorRole).toString().isEmpty()) {
            details.append(m_currentItem->data(authorRole).toString());
        } else {
            details.append(i18n("Author"));
        }
        details.append(QStringLiteral("</a><br />"));
    } else if (!m_currentItem->data(authorRole).toString().isEmpty()) {
        details.append(i18n("Created by %1", m_currentItem->data(authorRole).toString()) + QStringLiteral("<br />"));
    } else {
        details.append(QStringLiteral("<br />"));
    }

    if (m_currentProvider->get()->type() != ProviderModel::AUDIO && m_currentItem->data(widthRole).toInt() != 0) {
        details.append(i18n("Size: %1 x %2", m_currentItem->data(widthRole).toInt(), m_currentItem->data(heightRole).toInt()) + QStringLiteral("<br />"));
    }
    if (m_currentItem->data(durationRole).toInt() != 0) {
        details.append(i18n("Duration: %1 sec", m_currentItem->data(durationRole).toInt()) + QStringLiteral("<br />"));
    }
    details.append(m_currentItem->data(descriptionRole).toString());
    info_browser->setHtml(details);

    label_license->setText(licenseNameFromUrl(m_currentItem->data(licenseRole).toString(), true));
    label_license->setTipText(licenseNameFromUrl(m_currentItem->data(licenseRole).toString(), false));
    label_license->setUseTips(true);
    label_license->setUrl(m_currentItem->data(licenseRole).toString());

    details_box->setEnabled(true);
    button_import->setEnabled(true);
    button_preview->setEnabled(true);
}

/**
 * @brief ResourceWidget::licenseNameFromUrl
 * @param licenseUrl
 * @param shortName Whether the long name like "Attribution-NonCommercial-ShareAlike 3.0" or the short name like "CC BY-NC-SA 3.0" should be returned
 * @return the license name "Unnamed License" if url is not known.
 */
QString ResourceWidget::licenseNameFromUrl(const QString &licenseUrl, const bool shortName)
{
    QString licenseName;
    QString licenseShortName;

    if (licenseUrl.contains("creativecommons.org")) {
        if (licenseUrl.contains(QStringLiteral("/sampling+/"))) {
            licenseName = i18nc("Creative Commons License", "CC Sampling+");
        } else if (licenseUrl.contains(QStringLiteral("/by/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons Attribution");
            licenseShortName = i18nc("Creative Commons License (short)", "CC BY");
        } else if (licenseUrl.contains(QStringLiteral("/by-nd/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons Attribution-NoDerivs");
            licenseShortName = i18nc("Creative Commons License (short)", "CC BY-ND");
        } else if (licenseUrl.contains(QStringLiteral("/by-nc-sa/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons Attribution-NonCommercial-ShareAlike");
            licenseShortName = i18nc("Creative Commons License (short)", "CC BY-NC-SA");
        } else if (licenseUrl.contains(QStringLiteral("/by-sa/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons Attribution-ShareAlike");
            licenseShortName = i18nc("Creative Commons License (short)", "CC BY-SA");
        } else if (licenseUrl.contains(QStringLiteral("/by-nc/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons Attribution-NonCommercial");
            licenseShortName = i18nc("Creative Commons License (short)", "CC BY-NC");
        } else if (licenseUrl.contains(QStringLiteral("/by-nc-nd/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons Attribution-NonCommercial-NoDerivs");
            licenseShortName = i18nc("Creative Commons License (short)", "CC BY-NC-ND");
        } else if (licenseUrl.contains(QLatin1String("/publicdomain/zero/"))) {
            licenseName = i18nc("Creative Commons License", "Creative Commons 0");
            licenseShortName = i18nc("Creative Commons License (short)", "CC 0");
        } else if (licenseUrl.endsWith(QLatin1String("/publicdomain")) || licenseUrl.contains(QLatin1String("openclipart.org/share"))) {
            licenseName = i18nc("License", "Public Domain");
        } else {
            licenseShortName = i18nc("Short for: Unknown Creative Commons License", "Unknown CC License");
            licenseName = i18n("Unknown Creative Commons License");
        }

        if (licenseUrl.contains(QStringLiteral("/1.0"))) {
            licenseName.append(QStringLiteral(" 1.0"));
            licenseShortName.append(QStringLiteral(" 1.0"));
        } else if (licenseUrl.contains(QStringLiteral("/2.0"))) {
            licenseName.append(QStringLiteral(" 2.0"));
            licenseShortName.append(QStringLiteral(" 2.0"));
        } else if (licenseUrl.contains(QStringLiteral("/2.5"))) {
            licenseName.append(QStringLiteral(" 2.5"));
            licenseShortName.append(QStringLiteral(" 2.5"));
        } else if (licenseUrl.contains(QStringLiteral("/3.0"))) {
            licenseName.append(QStringLiteral(" 3.0"));
            licenseShortName.append(QStringLiteral(" 3.0"));
        } else if (licenseUrl.contains(QStringLiteral("/4.0"))) {
            licenseName.append(QStringLiteral(" 4.0"));
            licenseShortName.append(QStringLiteral(" 4.0"));
        }
    } else if (licenseUrl.contains("pexels.com/license/")) {
        licenseName = i18n("Pexels License");
    } else if (licenseUrl.contains("pixabay.com/service/license/")) {
        licenseName = i18n("Pixabay License");
        ;
    } else {
        licenseName = i18n("Unknown License");
    }

    if (shortName && !licenseShortName.isEmpty()) {
        return licenseShortName;
    } else {
        return licenseName;
    }
}

/**
 * @brief ResourceWidget::slotSetIconSize
 * @param size
 * Set the icon size for the search_results ListView
 */
void ResourceWidget::slotSetIconSize(int size)
{
    if (!search_results) {
        return;
    }
    QSize zoom = m_iconSize;
    zoom = zoom * (size / 4.0);
    search_results->setIconSize(zoom);
}

/**
 * @brief ResourceWidget::slotPreviewItem
 * Emits the previewClip signal if m_currentItem is valid to display a preview in the Clip Monitor
 */
void ResourceWidget::slotPreviewItem()
{
    if (!m_currentItem) {
        return;
    }
    blockUI(true);
    const QString path = m_currentItem->data(previewRole).toString();
    if (m_showloadingWarning && !QUrl::fromUserInput(path).isLocalFile()) {
        message_line->setText(i18n("It maybe takes a while until the preview is loaded"));
        message_line->setMessageType(KMessageWidget::Warning);
        message_line->show();
        QTimer::singleShot(6000, message_line, &KMessageWidget::animatedHide);
        repaint();
        // Only show this warning once
        m_showloadingWarning = false;
    }
    Q_EMIT previewClip(path, i18n("Online Resources Preview"));
    blockUI(false);
}

/**
 * @brief ResourceWidget::slotChooseVersion
 * @param urls list of download urls pointing to the certain file version
 * @param labels list of labels for the certain file version (needs to have the same order than urls)
 * @param accessToken access token to pass through to slotSaveItem
 * Displays a dialog to let the user choose a file version (e.g. filetype, quality) if there are multiple versions
 * available
 */
void ResourceWidget::slotChooseVersion(const QStringList &urls, const QStringList &labels, const QString &accessToken)
{
    if (urls.isEmpty() || labels.isEmpty()) {
        return;
    }
    if (urls.length() == 1) {
        slotSaveItem(urls.first(), accessToken);
        return;
    }
    bool ok;
    QString name = QInputDialog::getItem(this, i18nc("@title:window", "Choose File Version"), i18n("Please choose the version you want to download"), labels, 0,
                                         false, &ok);
    if (!ok || name.isEmpty()) {
        return;
    }
    slotSaveItem(urls.at(labels.indexOf(name)), accessToken);
}

/**
 * @brief ResourceWidget::slotSaveItem
 * @param originalUrl url pointing to the download file
 * @param accessToken the access token (optional)
 * Opens a dialog for user to choose a save location and start the download of the file
 */
void ResourceWidget::slotSaveItem(const QString &originalUrl, const QString &accessToken)
{
    QUrl saveUrl;

    if (!m_currentItem) {
        return;
    }

    QUrl srcUrl(originalUrl.isEmpty() ? m_currentItem->data(downloadRole).toString() : originalUrl);
    if (srcUrl.isEmpty()) {
        return;
    }

    QString path = KRecentDirs::dir(QStringLiteral(":WunjoOnlineResourceFolder"));
    QString ext;

    if (path.isEmpty()) {
        path = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    }
    if (!path.endsWith(QLatin1Char('/'))) {
        path.append(QLatin1Char('/'));
    }
    if (!srcUrl.fileName().isEmpty()) {
        path.append(srcUrl.fileName());
        ext = "*." + srcUrl.fileName().section(QLatin1Char('.'), -1);
    } else if (!m_currentItem->data(filetypeRole).toString().isEmpty()) {
        ext = "*." + m_currentItem->data(filetypeRole).toString();
    } else {
        if (m_currentProvider->get()->type() == ProviderModel::AUDIO) {
            ext = i18n("Audio") + QStringLiteral(" (*.wav *.mp3 *.ogg *.aif *.aiff *.m4a *.flac)") + QStringLiteral(";;");
        } else if (m_currentProvider->get()->type() == ProviderModel::VIDEO) {
            ext = i18n("Video") + QStringLiteral(" (*.mp4 *.webm *.mpg *.mov *.avi *.mkv)") + QStringLiteral(";;");
        } else if (m_currentProvider->get()->type() == ProviderModel::ProviderModel::IMAGE) {
            ext = i18n("Images") + QStringLiteral(" (*.png *.jpg *.jpeg *.svg") + QStringLiteral(";;");
        }
        ext.append(i18n("All Files") + QStringLiteral(" (*)"));
    }
    if (path.endsWith(QLatin1Char('/'))) {
        if (m_currentItem->data(nameRole).toString().isEmpty()) {
            path.append(i18n("Unnamed"));
        } else {
            path.append(m_currentItem->data(nameRole).toString());
        }
        path.append(srcUrl.fileName().section(QLatin1Char('.'), -1));
    }

    QString attribution;
    if (KMessageBox::questionTwoActions(this,
                                        i18n("Be aware that the usage of the resource is maybe restricted by license terms or law!\n"
                                             "Do you want to add license attribution to your Project Notes?"),
                                        QString(), KStandardGuiItem::add(), KGuiItem(i18nc("@action:button", "Continue without")),
                                        i18n("Remember this decision")) == KMessageBox::PrimaryAction) {
        attribution = i18nc("item name, item url, author name, license name, license url",
                            "This video uses \"%1\" (%2) by \"%3\" licensed under %4. To view a copy of this license, visit %5",
                            m_currentItem->data(nameRole).toString().isEmpty() ? i18n("Unnamed") : m_currentItem->data(nameRole).toString(),
                            m_currentItem->data(urlRole).toString(), m_currentItem->data(authorRole).toString(),
                            ResourceWidget::licenseNameFromUrl(m_currentItem->data(licenseRole).toString(), true), m_currentItem->data(licenseRole).toString());
        attribution.append(QStringLiteral("<br/> "));
    }

    QString saveUrlstring = QFileDialog::getSaveFileName(this, QString(), path, ext);

    // if user cancels save
    if (saveUrlstring.isEmpty()) {
        return;
    }

    saveUrl = QUrl::fromLocalFile(saveUrlstring);

    // Not KIO: its http worker is a kio-extras package the Windows and macOS
    // builds do not have, and a stock clip of several hundred megabytes deserves
    // a transfer that picks itself up after a broken line.
    auto *getJob = new FileDownloadJob(srcUrl, saveUrl.toLocalFile(), this);
    if (!accessToken.isEmpty()) {
        getJob->setHeader("Authorization", QStringLiteral("Bearer %1").arg(accessToken).toUtf8());
    }
    getJob->setProperty("attribution", attribution);
    getJob->setProperty("usedOAuth2", !accessToken.isEmpty());

    connect(getJob, &KJob::result, this, &ResourceWidget::slotGotFile);
    KIO::getJobTracker()->registerJob(getJob);
    getJob->start();
}

/**
 * @brief ResourceWidget::slotGotFile
 * @param job
 * Finish the download by emitting addClip and if necessary addLicenseInfo
 * Enables the import button
 */
void ResourceWidget::slotGotFile(KJob *job)

{
    if (job->error() != 0) {
        const QString errTxt = job->errorString();
        if (job->property("usedOAuth2").toBool()) {
            KMessageBox::error(this, i18n("%1 Try again.", errTxt), i18n("Error Loading Data"));
        } else {
            KMessageBox::error(this, errTxt, i18n("Error Loading Data"));
        }
        qCDebug(WUNJO_LOG) << "//file import job errored: " << errTxt;
        return;
    }
    auto *copyJob = qobject_cast<FileDownloadJob *>(job);
    if (copyJob == nullptr) {
        return;
    }
    const QUrl filePath = QUrl::fromLocalFile(copyJob->destination());
    KRecentDirs::add(QStringLiteral(":WunjoOnlineResourceFolder"), filePath.adjusted(QUrl::RemoveFilename).toLocalFile());

    KMessageBox::information(this, i18n("Resource saved to %1", filePath.toLocalFile()), i18n("Data Imported"));
    Q_EMIT addClip(filePath, QString());

    if (!copyJob->property("attribution").toString().isEmpty()) {
        Q_EMIT addLicenseInfo(copyJob->property("attribution").toString());
    }
}

/**
 * @brief ResourceWidget::slotAccessTokenReceived
 * @param accessToken - the access token obtained from the provider
 * Calls slotSaveItem or slotChooseVersion for auth protected files
 */
void ResourceWidget::slotAccessTokenReceived(const QString &accessToken)
{
    if (!accessToken.isEmpty()) {
        if (m_currentItem->data(singleDownloadRole).toBool()) {
            if (m_currentItem->data(downloadRole).toString().isEmpty()) {
                m_currentProvider->get()->slotFetchFiles(m_currentItem->data(idRole).toString());
                return;
            } else {
                slotSaveItem(QString(), accessToken);
            }
        } else {
            slotChooseVersion(m_currentItem->data(downloadRole).toStringList(), m_currentItem->data(downloadLabelRole).toStringList(), accessToken);
        }

    } else {
        KMessageBox::error(this, i18n("Try importing again to obtain a new connection"),
                           i18n("Error Getting Access Token from %1.", m_currentProvider->get()->name()));
    }
}

/*
 * A plugin's library
 */

bool ResourceWidget::isLibrary() const
{
    return m_currentProvider != nullptr && m_currentProvider->get() != nullptr && m_currentProvider->get()->isLibrary();
}

void ResourceWidget::showNote(const QString &text, KMessageWidget::MessageType type)
{
    message_line->clearActions();
    message_line->setText(text);
    message_line->setMessageType(type);
    message_line->show();
}

void ResourceWidget::reloadLibrary()
{
    if (!isLibrary()) {
        return;
    }
    page_number->blockSignals(true);
    page_number->setValue(1);
    page_number->blockSignals(false);
    slotStartSearch();
}

void ResourceWidget::loadIfEmpty()
{
    if (isLibrary() && !m_loaded && !m_searching) {
        reloadLibrary();
    }
}

QString ResourceWidget::toolName(const QString &group) const
{
    if (!isLibrary()) {
        return group;
    }
    const QString name = m_currentProvider->get()->groups().value(group);
    return name.isEmpty() ? group : i18n(name.toUtf8().constData());
}

QString ResourceWidget::libraryRowText(const QListWidgetItem *item) const
{
    const QString tool = toolName(item->data(groupRole).toString());
    QString title = item->data(nameRole).toString().simplified();
    if (title.isEmpty()) {
        title = tool.isEmpty() ? i18n("Unnamed") : tool;
    }
    if (title.size() > 60) {
        title = title.left(59).trimmed() + QChar(0x2026);
    }
    QStringList second;
    const QString stamp = item->data(dateRole).toString();
    QDateTime made = QDateTime::fromString(stamp, Qt::ISODateWithMs);
    if (!made.isValid()) {
        // the server writes microseconds
        QString shorter = stamp;
        shorter.remove(QRegularExpression(QStringLiteral("\\.\\d+")));
        made = QDateTime::fromString(shorter, Qt::ISODate);
    }
    if (made.isValid()) {
        second << QLocale().toString(made.toLocalTime(), QLocale::ShortFormat);
    }
    if (!tool.isEmpty()) {
        second << tool;
    }
    if (item->data(statusRole).toString() != QLatin1String("done") || item->data(downloadRole).toString().isEmpty()) {
        second << i18n("in progress");
    } else if (QFile::exists(libraryTarget(item))) {
        second << i18n("downloaded");
    }
    return title + QLatin1Char('\n') + second.join(QStringLiteral(", "));
}

void ResourceWidget::showLibraryList(const QList<ResourceItemInfo> &list, int pageCount)
{
    m_searching = false;
    m_loaded = true;
    blockUI(false);
    message_line->clearActions();
    search_results->clear();
    for (const ResourceItemInfo &item : list) {
        const bool ready = item.status == QLatin1String("done") && !item.downloadUrl.isEmpty();
        const bool audio = item.contentType.startsWith(QLatin1String("audio/"));
        auto *row = new QListWidgetItem(QIcon::fromTheme(audio ? QStringLiteral("audio-x-generic") : QStringLiteral("video-x-generic")), QString());
        row->setData(idRole, item.id);
        row->setData(nameRole, item.name);
        row->setData(descriptionRole, item.name);
        row->setData(downloadRole, item.downloadUrl);
        row->setData(previewRole, item.downloadUrl);
        row->setData(singleDownloadRole, true);
        row->setData(dateRole, item.date);
        row->setData(contentTypeRole, item.contentType);
        row->setData(fileNameRole, item.fileName);
        row->setData(groupRole, item.group);
        row->setData(statusRole, ready ? QStringLiteral("done") : item.status);
        row->setText(libraryRowText(row));
        search_results->addItem(row);
    }
    if (list.isEmpty()) {
        showNote(i18n("No files yet"), KMessageWidget::Information);
    } else {
        message_line->hide();
    }
    page_number->setMaximum(qMax(1, pageCount));
    page_number->setEnabled(page_number->maximum() > 1);
    applyFilter();
    requestThumbnails();
}

void ResourceWidget::applyFilter()
{
    if (!isLibrary()) {
        return;
    }
    const QString text = search_text->text().trimmed();
    const QString group = m_group->currentData().toString();
    for (int i = 0; i < search_results->count(); ++i) {
        QListWidgetItem *row = search_results->item(i);
        const bool otherTool = !group.isEmpty() && row->data(groupRole).toString() != group;
        const bool otherText = !text.isEmpty() && !row->data(nameRole).toString().contains(text, Qt::CaseInsensitive);
        row->setHidden(otherTool || otherText);
    }
}

QListWidgetItem *ResourceWidget::itemById(const QString &id) const
{
    for (int i = 0; i < search_results->count(); ++i) {
        if (search_results->item(i)->data(idRole).toString() == id) {
            return search_results->item(i);
        }
    }
    return nullptr;
}

QString ResourceWidget::libraryTarget(const QListWidgetItem *item) const
{
    WunjoDoc *doc = pCore->currentDoc();
    if (doc == nullptr || item == nullptr) {
        return QString();
    }
    QString name = QFileInfo(item->data(fileNameRole).toString()).fileName();
    if (name.isEmpty()) {
        const QString type = item->data(contentTypeRole).toString();
        QString suffix = QFileInfo(QUrl(item->data(downloadRole).toString()).path()).suffix();
        if (suffix.isEmpty()) {
            suffix = type.startsWith(QLatin1String("audio/")) ? QStringLiteral("mp3") : QStringLiteral("mp4");
        }
        name = item->data(idRole).toString() + QLatin1Char('.') + suffix;
    }
    name.replace(QRegularExpression(QStringLiteral("[/\\\\:*?\"<>|]")), QStringLiteral("-"));
    return doc->projectDataFolder() + QStringLiteral("/plugin-results/") + name;
}

void ResourceWidget::fetchLibraryItem(QListWidgetItem *item, const std::function<void(const QString &)> &then)
{
    if (item == nullptr || item->data(statusRole).toString() != QLatin1String("done")) {
        return;
    }
    const QString id = item->data(idRole).toString();
    const QString url = item->data(downloadRole).toString();
    const QString dest = libraryTarget(item);
    if (url.isEmpty() || dest.isEmpty()) {
        return;
    }
    if (QFile::exists(dest)) {
        if (then) {
            then(dest);
        }
        return;
    }
    if (then) {
        m_afterFetch.insert(id, then);
    }
    if (m_fetching.contains(id)) {
        return;
    }
    QDir().mkpath(QFileInfo(dest).absolutePath());
    // The link leads to the provider's storage, not to the plugin's server: it
    // takes no key, and none is sent there.
    auto *job = new FileDownloadJob(QUrl(url), dest, this);
    m_fetching.insert(id, job);
    connect(job, &KJob::result, this, [this, id, dest](KJob *finished) {
        m_fetching.remove(id);
        const std::function<void(const QString &)> next = m_afterFetch.take(id);
        if (finished->error() != 0 || !QFile::exists(dest)) {
            if (finished->error() != KJob::KilledJobError) {
                showNote(i18n("The file could not be downloaded. The service may no longer keep it"), KMessageWidget::Error);
            }
            return;
        }
        if (QListWidgetItem *row = itemById(id)) {
            row->setText(libraryRowText(row));
        }
        if (next) {
            next(dest);
        }
    });
    KIO::getJobTracker()->registerJob(job);
    job->start();
}

void ResourceWidget::previewLibraryItem(QListWidgetItem *item)
{
    if (item == nullptr) {
        return;
    }
    // Played from the disk: the monitor opens a link in the interface's thread
    // and the editor stands still for as long as that takes
    const QString title = libraryRowText(item).section(QLatin1Char('\n'), 0, 0);
    fetchLibraryItem(item, [this, title](const QString &path) { Q_EMIT previewClip(path, title); });
}

void ResourceWidget::importLibraryItem(QListWidgetItem *item)
{
    if (item == nullptr || !isLibrary()) {
        return;
    }
    // the same bin folder a run of that tool puts its result into
    QString folder = m_currentProvider->get()->groups().value(item->data(groupRole).toString());
    if (folder.isEmpty()) {
        folder = m_currentProvider->get()->name();
    }
    fetchLibraryItem(item, [this, folder](const QString &path) {
        std::shared_ptr<ProjectItemModel> model = pCore->projectItemModel();
        const QStringList known = model ? model->getClipByUrl(QFileInfo(path)) : QStringList();
        if (!known.isEmpty() && pCore->activeBin() != nullptr) {
            pCore->activeBin()->selectClipById(known.first());
            return;
        }
        Q_EMIT addClip(QUrl::fromLocalFile(path), PluginManager::resultsFolder(folder));
    });
}

void ResourceWidget::dragLibraryItem(QListWidgetItem *item)
{
    if (item == nullptr || item->data(statusRole).toString() != QLatin1String("done")) {
        return;
    }
    const QString dest = libraryTarget(item);
    if (dest.isEmpty()) {
        return;
    }
    if (!QFile::exists(dest)) {
        // nothing to drop yet: get it, and say so
        fetchLibraryItem(item, {});
        showNote(i18n("The file is downloading. Drag it when it is ready"), KMessageWidget::Information);
        QTimer::singleShot(6000, message_line, &KMessageWidget::animatedHide);
        return;
    }
    auto *mime = new QMimeData;
    mime->setUrls({QUrl::fromLocalFile(dest)});
    auto *drag = new QDrag(search_results);
    drag->setMimeData(mime);
    drag->setPixmap(item->icon().pixmap(search_results->iconSize()));
    drag->exec(Qt::CopyAction);
}

bool ResourceWidget::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == search_results->viewport() && isLibrary()) {
        if (event->type() == QEvent::MouseButtonPress) {
            auto *mouse = static_cast<QMouseEvent *>(event);
            if (mouse->button() == Qt::LeftButton) {
                m_pressPos = mouse->position().toPoint();
                QListWidgetItem *row = search_results->itemAt(m_pressPos);
                m_pressedId = row ? row->data(idRole).toString() : QString();
            }
        } else if (event->type() == QEvent::MouseMove) {
            auto *mouse = static_cast<QMouseEvent *>(event);
            if ((mouse->buttons() & Qt::LeftButton) && !m_pressedId.isEmpty() &&
                (mouse->position().toPoint() - m_pressPos).manhattanLength() >= QApplication::startDragDistance()) {
                const QString id = m_pressedId;
                m_pressedId.clear();
                dragLibraryItem(itemById(id));
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonRelease) {
            m_pressedId.clear();
        }
    }
    return QWidget::eventFilter(watched, event);
}

QString ResourceWidget::thumbnailPath(const QString &id) const
{
    QString safe = id;
    safe.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9_-]")), QStringLiteral("_"));
    return QStandardPaths::writableLocation(QStandardPaths::CacheLocation) + QStringLiteral("/library-thumbs/") + m_currentProvider->get()->pluginId() +
           QLatin1Char('/') + safe + QStringLiteral(".jpg");
}

void ResourceWidget::requestThumbnails()
{
    stopThumbnails();
    if (!isLibrary()) {
        return;
    }
    for (int i = 0; i < search_results->count(); ++i) {
        QListWidgetItem *row = search_results->item(i);
        if (!row->data(contentTypeRole).toString().startsWith(QLatin1String("video/")) || row->data(statusRole).toString() != QLatin1String("done")) {
            continue;
        }
        const QString id = row->data(idRole).toString();
        const QString cached = thumbnailPath(id);
        if (QFile::exists(cached)) {
            row->setIcon(QIcon(cached));
        } else {
            m_thumbQueue << id;
            m_thumbUrls.insert(id, row->data(downloadRole).toString());
        }
    }
    // two at a time: each one is a connection to the provider's storage
    nextThumbnail();
    nextThumbnail();
}

void ResourceWidget::nextThumbnail()
{
    if (m_thumbQueue.isEmpty() || m_thumbProcesses.size() >= 2 || !isLibrary()) {
        return;
    }
    const QString id = m_thumbQueue.takeFirst();
    const QString url = m_thumbUrls.take(id);
    const QString target = thumbnailPath(id);
    const QString part = target + QStringLiteral(".part.jpg");
    QDir().mkpath(QFileInfo(target).absolutePath());

    auto *process = new QProcess(this);
    m_thumbProcesses << process;
    // ffmpeg reads the start of the file only, enough for one frame; a link
    // that does not answer is given up after a while
    QTimer::singleShot(20000, process, [process]() { process->kill(); });
    auto finish = [this, process, id, target, part](bool ok) {
        if (!m_thumbProcesses.removeOne(process)) {
            return;
        }
        process->deleteLater();
        if (ok && QFileInfo(part).size() > 0) {
            QFile::remove(target);
            QFile::rename(part, target);
            if (QListWidgetItem *row = itemById(id)) {
                row->setIcon(QIcon(target));
            }
        } else {
            QFile::remove(part);
        }
        nextThumbnail();
    };
    connect(process, &QProcess::finished, this,
            [finish](int code, QProcess::ExitStatus status) { finish(status == QProcess::NormalExit && code == 0); });
    connect(process, &QProcess::errorOccurred, this, [finish](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) {
            finish(false);
        }
    });
    process->start(WunjoSettings::ffmpegpath(), {QStringLiteral("-hide_banner"), QStringLiteral("-loglevel"), QStringLiteral("error"), QStringLiteral("-y"),
                                                 QStringLiteral("-ss"), QStringLiteral("0"), QStringLiteral("-i"), url, QStringLiteral("-frames:v"),
                                                 QStringLiteral("1"), QStringLiteral("-vf"), QStringLiteral("scale=320:-2"), QStringLiteral("-update"),
                                                 QStringLiteral("1"), part});
}

void ResourceWidget::stopThumbnails()
{
    m_thumbQueue.clear();
    m_thumbUrls.clear();
    const QList<QProcess *> running = m_thumbProcesses;
    m_thumbProcesses.clear();
    for (QProcess *process : running) {
        process->disconnect(this);
        process->kill();
        process->waitForFinished(1000);
        process->deleteLater();
    }
}
