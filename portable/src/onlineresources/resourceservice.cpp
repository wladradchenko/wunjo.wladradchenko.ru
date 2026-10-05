/*
    SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "resourceservice.hpp"
#include "bin/clipcreator.hpp"
#include "bin/projectitemmodel.h"
#include "core.h"
#include "doc/wunjodoc.h"
#include "plugins/filedownloadjob.h"
#include "plugins/pluginmanager.h"
#include "providersrepository.hpp"
#include "resourcewidget.hpp"

#include <KIO/JobTracker>
#include <KJobTrackerInterface>
#include <KLocalizedString>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTimer>

namespace {
/** How long a search or a request for a file's links may take. */
constexpr int kAnswerTimeout = 45000;
/** Items kept per service for importing; a long session starts over past it. */
constexpr int kMaxItems = 1000;

QString safeName(QString name)
{
    name.replace(QRegularExpression(QStringLiteral("[/\\\\:*?\"<>|]")), QStringLiteral("-"));
    return name;
}
} // namespace

ResourceService::ResourceService(QObject *parent)
    : QObject(parent)
{
    // Installing or removing a plugin changes which lists exist
    connect(&PluginManager::instance(), &PluginManager::pluginsChanged, this, &ResourceService::dropAll);
}

QList<ResourceService::Service> ResourceService::list() const
{
    QList<Service> libraries;
    QList<Service> stock;
    const QVector<QPair<QString, QString>> providers = ProvidersRepository::get()->getAllProviers();
    for (const QPair<QString, QString> &provider : providers) {
        Service service;
        service.key = provider.second;
        service.name = provider.first;
        if (provider.second.startsWith(QLatin1String("plugin:"))) {
            service.id = provider.second.mid(7);
            service.library = true;
            libraries << service;
        } else {
            service.id = QFileInfo(provider.second).completeBaseName();
            stock << service;
        }
    }
    return libraries + stock;
}

ResourceService::Service ResourceService::find(const QString &id) const
{
    const QList<Service> services = list();
    for (const Service &service : services) {
        if (service.id == id) {
            return service;
        }
    }
    return Service();
}

ProviderModel *ResourceService::model(const Service &service)
{
    if (ProviderModel *known = m_models.value(service.id)) {
        return known;
    }
    ProviderModel *created = nullptr;
    if (service.library) {
        const PluginManifest plugin = PluginManager::instance().plugin(service.id);
        if (plugin.libraryFile().isEmpty()) {
            return nullptr;
        }
        created = new ProviderModel(QDir(plugin.rootDir()).absoluteFilePath(plugin.libraryFile()), service.id);
    } else {
        created = new ProviderModel(service.key);
    }
    if (!created->is_valid()) {
        delete created;
        return nullptr;
    }
    created->setParent(this);
    const QString id = service.id;
    connect(created, &ProviderModel::searchDone, this, [this, id](const QList<ResourceItemInfo> &found, int pageCount) { onSearchDone(id, found, pageCount); });
    connect(created, &ProviderModel::searchError, this, [this, id](const QString &message) { onSearchError(id, message); });
    connect(created, &ProviderModel::fetchedFiles, this, [this, id](const QStringList &urls, const QStringList &labels) { onFetchedFiles(id, urls, labels); });
    m_models.insert(id, created);
    return created;
}

void ResourceService::dropModel(const QString &id)
{
    if (ProviderModel *gone = m_models.take(id)) {
        gone->disconnect(this);
        gone->deleteLater();
    }
    m_lastDates.remove(id);
}

void ResourceService::dropAll()
{
    for (auto it = m_searching.constBegin(); it != m_searching.constEnd(); ++it) {
        fail(m_searchAnswers, it.value(), QStringLiteral("the plugins changed while searching; search again"));
    }
    m_searching.clear();
    for (auto it = m_filesRequests.constBegin(); it != m_filesRequests.constEnd(); ++it) {
        fail(m_importAnswers, it.value().request, QStringLiteral("the plugins changed while importing; import again"));
    }
    m_filesRequests.clear();
    const QStringList ids = m_models.keys();
    for (const QString &id : ids) {
        dropModel(id);
    }
    m_items.clear();
}

void ResourceService::fail(QHash<int, QJsonObject> &answers, int request, const QString &message)
{
    QJsonObject answer = answers.value(request);
    answer.insert(QStringLiteral("request"), request);
    answer.insert(QStringLiteral("done"), true);
    answer.insert(QStringLiteral("ok"), false);
    answer.insert(QStringLiteral("message"), message);
    answers.insert(request, answer);
}

QJsonArray ResourceService::services()
{
    QJsonArray found;
    const QList<Service> services = list();
    for (const Service &service : services) {
        ProviderModel *provider = model(service);
        if (provider == nullptr) {
            continue;
        }
        QJsonObject entry{{QStringLiteral("id"), service.id}, {QStringLiteral("name"), service.name}, {QStringLiteral("library"), service.library}};
        QString kind = QStringLiteral("mixed");
        if (!service.library) {
            switch (provider->type()) {
            case ProviderModel::AUDIO:
                kind = QStringLiteral("audio");
                break;
            case ProviderModel::VIDEO:
                kind = QStringLiteral("video");
                break;
            case ProviderModel::IMAGE:
                kind = QStringLiteral("image");
                break;
            default:
                break;
            }
        }
        entry.insert(QStringLiteral("kind"), kind);
        bool ready = true;
        QString note;
        if (service.library) {
            ready = provider->hasKey();
            note = ready ? QStringLiteral("the user's own files made by this plugin; filter by dates, tool or text")
                         : QStringLiteral("the plugin's key is missing: the user adds it on the plugin's settings tab");
            QJsonObject tools;
            const QMap<QString, QString> groups = provider->groups();
            for (auto it = groups.constBegin(); it != groups.constEnd(); ++it) {
                tools.insert(it.key(), it.value());
            }
            entry.insert(QStringLiteral("tools"), tools);
        } else if (provider->downloadOAuth2()) {
            note = QStringLiteral("files come as the service's mp3 preview; the full file needs the user's own login on the Online Resources tab");
        } else {
            note = QStringLiteral("stock library; tell the user the author and the license of what you import");
        }
        entry.insert(QStringLiteral("ready"), ready);
        entry.insert(QStringLiteral("note"), note);
        found.append(entry);
    }
    return found;
}

int ResourceService::search(const QString &serviceId, const QString &query, int page, const QString &from, const QString &to)
{
    const int request = ++m_next;
    page = qMax(1, page);
    m_searchAnswers.insert(request, QJsonObject{{QStringLiteral("request"), request}, {QStringLiteral("service"), serviceId}, {QStringLiteral("page"), page}});
    auto refuse = [this, request](const QString &message) {
        fail(m_searchAnswers, request, message);
        return request;
    };
    const Service service = find(serviceId);
    if (!service.isValid()) {
        return refuse(QStringLiteral("no service '%1'; online_services lists them").arg(serviceId));
    }
    m_searchAnswers[request].insert(QStringLiteral("library"), service.library);
    if (m_searching.contains(service.id)) {
        return refuse(QStringLiteral("a search of this service is still running; wait for it"));
    }
    ProviderModel *provider = model(service);
    if (provider == nullptr) {
        return refuse(QStringLiteral("the description of this service could not be read"));
    }
    if (service.library) {
        if (!provider->hasKey()) {
            return refuse(QStringLiteral("the plugin's key is missing: the user adds it on the plugin's settings tab"));
        }
        const QPair<QString, QString> dates{from.trimmed(), to.trimmed()};
        const QDate first = QDate::fromString(dates.first, Qt::ISODate);
        const QDate last = QDate::fromString(dates.second, Qt::ISODate);
        if ((!dates.first.isEmpty() && !first.isValid()) || (!dates.second.isEmpty() && !last.isValid())) {
            return refuse(QStringLiteral("dates are written yyyy-MM-dd"));
        }
        if (page > 1 && m_lastDates.value(service.id) != dates) {
            return refuse(QStringLiteral("the dates changed; start from page 1"));
        }
        if (!provider->canRequestPage(page)) {
            return refuse(QStringLiteral("ask for page %1 first").arg(page - 1));
        }
        provider->setDateRange(first, last);
        m_lastDates.insert(service.id, dates);
    } else if (query.trimmed().isEmpty()) {
        return refuse(QStringLiteral("a stock library needs a query"));
    }

    m_searchAnswers[request].insert(QStringLiteral("done"), false);
    m_searching.insert(service.id, request);
    const QString id = service.id;
    QTimer::singleShot(kAnswerTimeout, this, [this, id, request]() {
        if (m_searching.value(id) != request) {
            return;
        }
        m_searching.remove(id);
        // a late answer must not be taken for the next search's
        dropModel(id);
        fail(m_searchAnswers, request, QStringLiteral("no answer from the service; try again"));
    });
    provider->slotStartSearch(query.trimmed(), page);
    return request;
}

void ResourceService::onSearchDone(const QString &id, const QList<ResourceItemInfo> &found, int pageCount)
{
    const int request = m_searching.take(id);
    if (request == 0) {
        return;
    }
    const Service service = find(id);
    QHash<QString, ResourceItemInfo> &cache = m_items[id];
    if (cache.size() > kMaxItems) {
        cache.clear();
    }
    QJsonArray items;
    for (const ResourceItemInfo &item : found) {
        cache.insert(item.id, item);
        items.append(itemJson(service, item));
    }
    QJsonObject answer = m_searchAnswers.value(request);
    const int page = answer.value(QStringLiteral("page")).toInt(1);
    answer.insert(QStringLiteral("done"), true);
    answer.insert(QStringLiteral("ok"), true);
    answer.insert(QStringLiteral("message"), QString());
    answer.insert(QStringLiteral("pages"), qMax(page, pageCount));
    answer.insert(QStringLiteral("items"), items);
    m_searchAnswers.insert(request, answer);
}

void ResourceService::onSearchError(const QString &id, const QString &message)
{
    const int request = m_searching.take(id);
    if (request != 0) {
        fail(m_searchAnswers, request, message);
    }
}

QJsonObject ResourceService::itemJson(const Service &service, const ResourceItemInfo &item)
{
    QJsonObject entry{{QStringLiteral("id"), item.id}};
    QString name = item.name.simplified();
    if (service.library) {
        QString kind = QStringLiteral("video");
        if (item.contentType.startsWith(QLatin1String("audio/"))) {
            kind = QStringLiteral("audio");
        } else if (item.contentType.startsWith(QLatin1String("image/"))) {
            kind = QStringLiteral("image");
        }
        entry.insert(QStringLiteral("kind"), kind);
        QDateTime made = QDateTime::fromString(item.date, Qt::ISODateWithMs);
        if (!made.isValid()) {
            // the server writes microseconds
            QString shorter = item.date;
            shorter.remove(QRegularExpression(QStringLiteral("\\.\\d+")));
            made = QDateTime::fromString(shorter, Qt::ISODate);
        }
        entry.insert(QStringLiteral("made"), made.isValid() ? made.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm")) : item.date);
        entry.insert(QStringLiteral("tool"), item.group);
        ProviderModel *provider = model(service);
        entry.insert(QStringLiteral("tool_name"), provider ? provider->groups().value(item.group) : QString());
        const bool ready = item.status == QLatin1String("done") && !item.downloadUrl.isEmpty();
        entry.insert(QStringLiteral("status"), ready ? QStringLiteral("done") : QStringLiteral("in progress"));
        entry.insert(QStringLiteral("downloaded"), ready && QFile::exists(libraryTarget(item.fileName, item.id, item.contentType, item.downloadUrl)));
        if (item.price >= 0) {
            entry.insert(QStringLiteral("price"), item.price);
        }
    } else {
        ProviderModel *provider = model(service);
        QString kind = QStringLiteral("video");
        if (provider && provider->type() == ProviderModel::AUDIO) {
            kind = QStringLiteral("audio");
        } else if (provider && provider->type() == ProviderModel::IMAGE) {
            kind = QStringLiteral("image");
        }
        entry.insert(QStringLiteral("kind"), kind);
        if (name.isEmpty() && !item.author.isEmpty()) {
            name = QStringLiteral("by %1").arg(item.author);
        }
        entry.insert(QStringLiteral("author"), item.author);
        entry.insert(QStringLiteral("license"), item.license.isEmpty() ? QString() : ResourceWidget::licenseNameFromUrl(item.license, true));
        entry.insert(QStringLiteral("page_url"), item.infoUrl);
        entry.insert(QStringLiteral("duration"), item.duration);
        entry.insert(QStringLiteral("width"), item.width);
        entry.insert(QStringLiteral("height"), item.height);
        if (item.downloadLabels.size() > 1) {
            entry.insert(QStringLiteral("versions"), QJsonArray::fromStringList(item.downloadLabels));
        }
    }
    entry.insert(QStringLiteral("name"), name);
    return entry;
}

QJsonObject ResourceService::searchAnswer(int request)
{
    if (!m_searchAnswers.contains(request)) {
        return QJsonObject{{QStringLiteral("request"), request}, {QStringLiteral("done"), true}, {QStringLiteral("ok"), false},
                           {QStringLiteral("message"), QStringLiteral("no such request")}};
    }
    const QJsonObject answer = m_searchAnswers.value(request);
    if (answer.value(QStringLiteral("done")).toBool()) {
        m_searchAnswers.remove(request);
    }
    return answer;
}

int ResourceService::pickVersion(const QStringList &labels, const QString &version, QString *error)
{
    if (labels.size() <= 1) {
        return 0;
    }
    if (!version.trimmed().isEmpty()) {
        for (int i = 0; i < labels.size(); ++i) {
            if (labels.at(i).compare(version.trimmed(), Qt::CaseInsensitive) == 0) {
                return i;
            }
        }
        for (int i = 0; i < labels.size(); ++i) {
            if (labels.at(i).contains(version.trimmed(), Qt::CaseInsensitive)) {
                return i;
            }
        }
        *error = QStringLiteral("no version '%1'; there are: %2").arg(version, labels.join(QStringLiteral(", ")));
        return -1;
    }
    // The largest that the project's picture does not have to shrink
    const int frame = pCore->getCurrentFrameSize().height();
    static const QRegularExpression size(QStringLiteral("(\\d+)x(\\d+)"));
    int best = -1;
    int bestHeight = 0;
    int lowest = -1;
    int lowestHeight = 0;
    for (int i = 0; i < labels.size(); ++i) {
        const QRegularExpressionMatch match = size.match(labels.at(i));
        if (!match.hasMatch()) {
            continue;
        }
        const int height = match.captured(2).toInt();
        if (height <= frame && height > bestHeight) {
            best = i;
            bestHeight = height;
        }
        if (lowest < 0 || height < lowestHeight) {
            lowest = i;
            lowestHeight = height;
        }
    }
    if (best >= 0) {
        return best;
    }
    return lowest >= 0 ? lowest : 0;
}

int ResourceService::importItem(const QString &serviceId, const QString &itemId, const QString &version)
{
    const int request = ++m_next;
    m_importAnswers.insert(request, QJsonObject{{QStringLiteral("request"), request}, {QStringLiteral("service"), serviceId}, {QStringLiteral("item"), itemId}});
    auto refuse = [this, request](const QString &message) {
        fail(m_importAnswers, request, message);
        return request;
    };
    const Service service = find(serviceId);
    if (!service.isValid()) {
        return refuse(QStringLiteral("no service '%1'; online_services lists them").arg(serviceId));
    }
    if (!m_items.value(service.id).contains(itemId)) {
        return refuse(QStringLiteral("search first: item '%1' is not among the results seen").arg(itemId));
    }
    const ResourceItemInfo item = m_items.value(service.id).value(itemId);
    if (service.library && (item.status != QLatin1String("done") || item.downloadUrl.isEmpty())) {
        return refuse(QStringLiteral("the file is still being made; search again later"));
    }
    ProviderModel *provider = model(service);
    if (provider == nullptr) {
        return refuse(QStringLiteral("the description of this service could not be read"));
    }
    if (!service.library && provider->downloadOAuth2()) {
        // The full file asks the user to log in through the browser: not
        // something to open behind their back. The preview needs no login.
        if (item.previewUrl.isEmpty()) {
            return refuse(QStringLiteral("this item has no preview to take; the full file needs the user's login on the Online Resources tab"));
        }
        startImport(request, service, item, item.previewUrl, QString(), true);
        return request;
    }
    if (!item.downloadUrl.isEmpty()) {
        startImport(request, service, item, item.downloadUrl, QString(), false);
        return request;
    }
    if (!item.downloadUrls.isEmpty()) {
        QString error;
        const int index = pickVersion(item.downloadLabels, version, &error);
        if (index < 0) {
            return refuse(error);
        }
        startImport(request, service, item, item.downloadUrls.value(index), item.downloadLabels.value(index), false);
        return request;
    }
    if (provider->hasFilesRequest()) {
        // The search did not say where the files are: ask for them first
        if (m_filesRequests.contains(service.id)) {
            return refuse(QStringLiteral("this service is busy with another import; try again in a moment"));
        }
        m_filesRequests.insert(service.id, FilesRequest{request, itemId, version});
        m_importAnswers[request].insert(QStringLiteral("done"), false);
        m_importAnswers[request].insert(QStringLiteral("percent"), 0);
        const QString id = service.id;
        QTimer::singleShot(kAnswerTimeout, this, [this, id, request]() {
            if (m_filesRequests.value(id).request != request) {
                return;
            }
            m_filesRequests.remove(id);
            dropModel(id);
            fail(m_importAnswers, request, QStringLiteral("no answer from the service; try again"));
        });
        provider->slotFetchFiles(itemId);
        return request;
    }
    return refuse(QStringLiteral("the service gives no link to this file"));
}

void ResourceService::onFetchedFiles(const QString &id, const QStringList &urls, const QStringList &labels)
{
    if (!m_filesRequests.contains(id)) {
        return;
    }
    const FilesRequest pending = m_filesRequests.take(id);
    if (urls.isEmpty()) {
        fail(m_importAnswers, pending.request, QStringLiteral("the service lists no file for this item"));
        return;
    }
    QString error;
    const int index = pickVersion(labels, pending.version, &error);
    if (index < 0) {
        fail(m_importAnswers, pending.request, error);
        return;
    }
    const Service service = find(id);
    startImport(pending.request, service, m_items.value(id).value(pending.itemId), urls.value(index), labels.value(index), false);
}

void ResourceService::startImport(int request, const Service &service, const ResourceItemInfo &item, const QString &url, const QString &version, bool preview)
{
    WunjoDoc *doc = pCore->currentDoc();
    if (doc == nullptr || url.isEmpty()) {
        fail(m_importAnswers, request, doc == nullptr ? QStringLiteral("no project is open") : QStringLiteral("the service gives no link to this file"));
        return;
    }
    QString dest;
    QString folder;
    ProviderModel *provider = model(service);
    if (service.library) {
        dest = libraryTarget(item.fileName, item.id, item.contentType, url);
        // the same bin folder a run of that tool puts its result into
        folder = provider ? provider->groups().value(item.group) : QString();
    } else {
        QString name = QFileInfo(QUrl(url).path()).fileName();
        if (name.isEmpty() || QFileInfo(name).suffix().isEmpty()) {
            QString suffix = item.filetype;
            if (suffix.isEmpty()) {
                const ProviderModel::SERVICETYPE type = provider ? provider->type() : ProviderModel::VIDEO;
                suffix = type == ProviderModel::AUDIO ? QStringLiteral("mp3") : (type == ProviderModel::IMAGE ? QStringLiteral("jpg") : QStringLiteral("mp4"));
            }
            name = item.id + QLatin1Char('.') + suffix;
        } else {
            name = item.id + QLatin1Char('_') + name;
        }
        dest = doc->projectDataFolder() + QStringLiteral("/online-resources/") + safeName(service.id) + QLatin1Char('/') + safeName(name);
    }
    if (folder.isEmpty()) {
        folder = service.name;
    }
    QString title = item.name.simplified();
    if (title.isEmpty()) {
        title = item.author.isEmpty() ? item.id : QStringLiteral("by %1").arg(item.author);
    }

    QJsonObject answer = m_importAnswers.value(request);
    answer.insert(QStringLiteral("done"), false);
    answer.insert(QStringLiteral("percent"), 0);
    answer.insert(QStringLiteral("name"), title);
    answer.insert(QStringLiteral("version"), version);
    answer.insert(QStringLiteral("preview"), preview);
    if (!service.library) {
        answer.insert(QStringLiteral("author"), item.author);
        answer.insert(QStringLiteral("license"), item.license.isEmpty() ? QString() : ResourceWidget::licenseNameFromUrl(item.license, true));
        answer.insert(QStringLiteral("page_url"), item.infoUrl);
    }
    m_importAnswers.insert(request, answer);

    const bool stock = !service.library;
    fetch(
        QUrl(url), dest,
        [this, request, folder, item, stock, title](const QString &path, const QString &error) {
            if (path.isEmpty()) {
                fail(m_importAnswers, request, error.isEmpty() ? QStringLiteral("the download was cancelled") : error);
                return;
            }
            std::shared_ptr<ProjectItemModel> bin = pCore->projectItemModel();
            if (!bin || pCore->currentDoc() == nullptr) {
                fail(m_importAnswers, request, QStringLiteral("no project is open"));
                return;
            }
            QString binId;
            const QStringList known = bin->getClipByUrl(QFileInfo(path));
            if (!known.isEmpty()) {
                binId = known.constFirst();
            } else {
                Fun undo = []() { return true; };
                Fun redo = []() { return true; };
                binId = ClipCreator::createClipFromFile(path, PluginManager::resultsFolder(folder), bin, undo, redo);
                if (binId.isEmpty() || binId == QLatin1String("-1")) {
                    fail(m_importAnswers, request, QStringLiteral("the file was downloaded to %1 but could not be added to the bin").arg(path));
                    return;
                }
                pCore->pushUndo(undo, redo, i18nc("@action", "Add clip"));
                if (stock && !item.license.isEmpty()) {
                    // The user is asked about this when importing by hand; the
                    // assistant has nobody to ask, and credit is owed either way
                    Q_EMIT addLicenseInfo(ResourceWidget::attributionText(title, item.infoUrl, item.author, item.license));
                }
            }
            QJsonObject answer = m_importAnswers.value(request);
            answer.insert(QStringLiteral("done"), true);
            answer.insert(QStringLiteral("ok"), true);
            answer.insert(QStringLiteral("message"), QString());
            answer.insert(QStringLiteral("percent"), 100);
            answer.insert(QStringLiteral("bin_id"), binId);
            answer.insert(QStringLiteral("folder"), folder);
            answer.insert(QStringLiteral("path"), path);
            m_importAnswers.insert(request, answer);
        },
        [this, request](int percent) {
            if (m_importAnswers.contains(request)) {
                m_importAnswers[request].insert(QStringLiteral("percent"), percent);
            }
        });
}

QJsonObject ResourceService::importAnswer(int request)
{
    if (!m_importAnswers.contains(request)) {
        return QJsonObject{{QStringLiteral("request"), request}, {QStringLiteral("done"), true}, {QStringLiteral("ok"), false},
                           {QStringLiteral("message"), QStringLiteral("no such request")}};
    }
    const QJsonObject answer = m_importAnswers.value(request);
    if (answer.value(QStringLiteral("done")).toBool()) {
        m_importAnswers.remove(request);
    }
    return answer;
}

QString ResourceService::libraryTarget(const QString &fileName, const QString &id, const QString &contentType, const QString &url)
{
    WunjoDoc *doc = pCore->currentDoc();
    if (doc == nullptr) {
        return QString();
    }
    QString name = QFileInfo(fileName).fileName();
    if (name.isEmpty()) {
        QString suffix = QFileInfo(QUrl(url).path()).suffix();
        if (suffix.isEmpty()) {
            suffix = contentType.startsWith(QLatin1String("audio/")) ? QStringLiteral("mp3") : QStringLiteral("mp4");
        }
        name = id + QLatin1Char('.') + suffix;
    }
    return doc->projectDataFolder() + QStringLiteral("/plugin-results/") + safeName(name);
}

void ResourceService::fetch(const QUrl &url, const QString &dest, const Fetched &then, const Progress &progress)
{
    if (QFile::exists(dest)) {
        if (then) {
            then(dest, QString());
        }
        return;
    }
    if (then) {
        m_waiting[dest].append(then);
    }
    if (progress) {
        m_progress[dest].append(progress);
    }
    if (m_downloads.contains(dest)) {
        return;
    }
    QDir().mkpath(QFileInfo(dest).absolutePath());
    // The links lead to the providers' storage, not to a plugin's server: they
    // take no key, and none is sent there.
    auto *job = new FileDownloadJob(url, dest, this);
    m_downloads.insert(dest, job);
    connect(job, &KJob::percentChanged, this, [this, dest](KJob *, unsigned long percent) {
        const QList<Progress> listeners = m_progress.value(dest);
        for (const Progress &listener : listeners) {
            listener(int(percent));
        }
    });
    connect(job, &KJob::result, this, [this, dest](KJob *finished) {
        m_downloads.remove(dest);
        m_progress.remove(dest);
        const QList<Fetched> waiting = m_waiting.take(dest);
        QString path;
        QString error;
        if (finished->error() == 0 && QFile::exists(dest)) {
            path = dest;
        } else if (finished->error() != KJob::KilledJobError) {
            error = i18n("The file could not be downloaded. The service may no longer keep it");
        }
        for (const Fetched &listener : waiting) {
            listener(path, error);
        }
    });
    KIO::getJobTracker()->registerJob(job);
    job->start();
}
