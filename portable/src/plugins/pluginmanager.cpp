/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginmanager.h"

#include "filedownloader.h"
#include "pluginpythonenv.h"
#include "pluginsetstore.h"

#include "bin/clipcreator.hpp"
#include "bin/projectclip.h"
#include "bin/projectfolder.h"
#include "bin/projectitemmodel.h"
#include "core.h"
#include "doc/wunjodoc.h"
#include "effects/effectstack/model/effectitemmodel.hpp"
#include "effects/effectstack/model/effectstackmodel.hpp"
#include "macros.hpp"
#include "mainwindow.h"
#include "timeline2/model/timelineitemmodel.hpp"
#include "wunjosettings.h"
#include "xml/xml.hpp"

#include <KArchive>
#include <KArchiveDirectory>
#include <KConfig>
#include <KConfigGroup>
#include <KIO/Global>
#include <KLocalizedString>
#include <KTar>
#include <KZip>

#include <QCryptographicHash>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QDirIterator>
#include <QDomDocument>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLibrary>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QStorageInfo>
#include <QUuid>

#include <algorithm>
#include <memory>

PluginManager &PluginManager::instance()
{
    static PluginManager manager;
    return manager;
}

PluginManager::PluginManager()
    : QObject(nullptr)
{
    rescan();
}

QString PluginManager::userPluginsDir() const
{
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/plugins");
}

void PluginManager::rescan()
{
    m_plugins.clear();
    m_bundledIds.clear();
    // Read-only bundled locations are every AppLocalData root except the
    // writable one; the writable "plugins" dir holds user imports and is
    // scanned last so a user copy shadows a bundled plugin of the same id.
    // Compare by cleaned path — standardLocations() may list the writable root
    // in a different textual form, and treating it as "bundled" would wrongly
    // mark imported plugins built-in (and hide their Uninstall button).
    const QString userDir = QDir::cleanPath(userPluginsDir());
    QStringList scanOrder;
    for (const QString &root : QStandardPaths::standardLocations(QStandardPaths::AppLocalDataLocation)) {
        const QString dir = QDir::cleanPath(root + QStringLiteral("/plugins"));
        if (dir != userDir && !scanOrder.contains(dir)) {
            scanOrder << dir;
        }
    }
    scanOrder << userDir;
    for (const QString &base : std::as_const(scanOrder)) {
        const bool bundled = base != userDir;
        QDir dir(base);
        if (!dir.exists()) {
            continue;
        }
        const QStringList entries = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const QString &entry : entries) {
            const QString path = dir.absoluteFilePath(entry);
            if (!QFileInfo::exists(path + QStringLiteral("/plugin.json"))) {
                continue;
            }
            const PluginManifest manifest = PluginManifest::fromDir(path, true);
            if (!manifest.isValid()) {
                continue;
            }
            if (bundled) {
                m_bundledIds.insert(manifest.id());
            }
            m_plugins.insert(manifest.id(), manifest);
        }
    }
    // A private environment used to be called venv-plugin-<id> and is now
    // venv-<id>; the word said nothing the folder did not already say. One
    // built under the old name is renamed rather than abandoned — it is
    // gigabytes of wheels that would otherwise be downloaded a second time.
    // (pip is always run as `python -m pip` and plugins are started by absolute
    // interpreter path, so the stale shebangs inside bin/ are nobody's road.)
    const QString dataRoot = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    for (const PluginManifest &manifest : std::as_const(m_plugins)) {
        const QString venv = manifest.venvName();
        if (venv.isEmpty() || venv == QLatin1String("venv")) {
            continue;
        }
        const QString legacy = dataRoot + QStringLiteral("/venv-plugin-") + manifest.id();
        if (QFileInfo::exists(legacy) && !QFileInfo::exists(dataRoot + QLatin1Char('/') + venv)) {
            QDir().rename(legacy, dataRoot + QLatin1Char('/') + venv);
        }
    }

    // Effect ownership is answered from the manifests, so it is known before the
    // effect files are written and stays right if writing them fails.
    m_effectOwners.clear();
    for (const PluginManifest &manifest : std::as_const(m_plugins)) {
        const QList<PluginEffect> effects = manifest.effects();
        for (const PluginEffect &effect : effects) {
            m_effectOwners.insert(effect.id, manifest.id());
        }
    }
}

QList<PluginManifest> PluginManager::installedPlugins() const
{
    QList<PluginManifest> list = m_plugins.values();
    std::sort(list.begin(), list.end(), [](const PluginManifest &a, const PluginManifest &b) { return a.name().localeAwareCompare(b.name()) < 0; });
    return list;
}

QList<PluginManifest> PluginManager::pluginsForTarget(const QString &target) const
{
    QList<PluginManifest> list;
    const auto all = installedPlugins();
    for (const PluginManifest &m : all) {
        // A plugin may work on more than one kind of clip, so ask whether the
        // asked-for kind is among them rather than whether it is the only one.
        if (m.hasTarget(target)) {
            list << m;
        }
    }
    return list;
}

PluginManifest PluginManager::plugin(const QString &id) const
{
    return m_plugins.value(id);
}

bool PluginManager::isBundled(const QString &id) const
{
    return m_bundledIds.contains(id);
}

QString PluginManager::modelsDir(const QString &id) const
{
    const PluginManifest manifest = m_plugins.value(id);
    if (manifest.rootDir().isEmpty()) {
        return {};
    }
    // Weights are fetched on this machine, so they cannot live where a plugin
    // that ships with the app does: that copy sits in the read-only install
    // location. Only plugins the user imported keep their models beside
    // themselves; for the rest the writable plugins folder stands in, under the
    // same id, so everything else — the state checks, "Delete all models",
    // uninstall — goes on addressing one place.
    QString base = manifest.rootDir();
    if (!QDir::cleanPath(base).startsWith(QDir::cleanPath(userPluginsDir()) + QLatin1Char('/'))) {
        base = userPluginsDir() + QLatin1Char('/') + id;
    }
    const QString dir = base + QStringLiteral("/models");
    QDir().mkpath(dir);
    return dir;
}

QList<PluginModel> PluginManager::applicableModels(const PluginManifest &manifest)
{
    return manifest.modelsFor(gpuVramGb(), gpuBackend());
}

QString PluginManager::modelPath(const QString &id, const PluginModel &model) const
{
    return modelsDir(id) + QLatin1Char('/') + model.name;
}

void PluginManager::recordJob(const QString &jobId, const QString &state, int percent, const QString &message)
{
    QJsonObject entry = m_jobOutcomes.value(jobId);
    entry.insert(QStringLiteral("state"), state);
    entry.insert(QStringLiteral("percent"), percent);
    entry.insert(QStringLiteral("message"), message);
    if (!m_jobOutcomes.contains(jobId)) {
        m_jobOrder.append(jobId);
        while (m_jobOrder.size() > kRememberedJobs) {
            m_jobOutcomes.remove(m_jobOrder.takeFirst());
        }
    }
    m_jobOutcomes.insert(jobId, entry);
}

void PluginManager::noteJobQueued(const QString &jobId, const QString &name)
{
    // Waiting for the graphics card is running as far as anybody asking is
    // concerned — it has not failed and it has not finished. No card yet: the
    // one @ref noteJobStarted opens when it really begins would be a second.
    recordJob(jobId, QStringLiteral("running"), 0, name);
}

void PluginManager::noteJobStarted(const QString &jobId, const QString &name)
{
    recordJob(jobId, QStringLiteral("running"), 0, name);
    if (auto *window = pCore->window()) {
        window->scriptChatToolStart(jobId, name);
    }
}

void PluginManager::noteJobProgress(const QString &jobId, int percent)
{
    auto it = m_jobOutcomes.find(jobId);
    if (it != m_jobOutcomes.end()) {
        it->insert(QStringLiteral("percent"), percent);
    }

    if (auto *window = pCore->window()) {
        window->scriptChatToolProgress(jobId, percent, QString());
    }
}

void PluginManager::noteJobEnded(const QString &jobId, bool isError, const QString &message)
{
    const int percent = isError ? m_jobOutcomes.value(jobId).value(QStringLiteral("percent")).toInt() : 100;
    recordJob(jobId, isError ? QStringLiteral("failed") : QStringLiteral("done"), percent, message);
    if (auto *window = pCore->window()) {
        window->scriptChatToolEnd(jobId, isError, message);
    }
}

QJsonObject PluginManager::jobOutcome(const QString &jobId) const
{
    return m_jobOutcomes.value(jobId);
}

QString PluginManager::downloadTarget(const QString &id, const PluginModel &model) const
{
    const QString path = modelPath(id, model);
    // ".archive" and not ".part": an unfinished download is already called
    // "<target>.part", and an archive that borrowed the same suffix produced a
    // "<name>.part.part" nobody was looking for and nothing cleaned up.
    return model.unpack.isEmpty() ? path : path + QStringLiteral(".archive");
}

PluginManager::ModelState PluginManager::modelState(const QString &id, const PluginModel &model, bool verifyChecksum) const
{
    const QString path = modelPath(id, model);
    // An archived weight is judged by what came out of it, not by the archive:
    // the download is a means, the unpacked runtime is the thing the plugin runs.
    if (!model.unpack.isEmpty()) {
        QDir dir(path);
        return dir.exists() && !dir.isEmpty() ? ModelReady : ModelMissing;
    }
    const QFileInfo info(path);
    if (!info.exists() || info.size() == 0) {
        return ModelMissing;
    }
    // size_mb is what the plugin author wrote down, not an exact figure, so it
    // can only answer "is this a fraction of the real thing" — which is exactly
    // what an aborted download leaves behind. It cannot be tightened towards the
    // declared size: the numbers in a manifest are rounded by hand and are
    // sometimes half again as large as the file, so a stricter rule condemns
    // weights that are perfectly whole. Telling a damaged file from a complete
    // one needs the checksum, not the size.
    if (model.sizeMb > 0 && info.size() < model.sizeMb * 1024 * 1024 / 2) {
        return ModelIncomplete;
    }
    if (verifyChecksum && !model.sha256.isEmpty()) {
        QFile file(path);
        QCryptographicHash hash(QCryptographicHash::Sha256);
        if (!file.open(QIODevice::ReadOnly) || !hash.addData(&file)) {
            return ModelIncomplete;
        }
        if (QString::fromLatin1(hash.result().toHex()).compare(model.sha256, Qt::CaseInsensitive) != 0) {
            return ModelIncomplete;
        }
    }
    return ModelReady;
}

bool PluginManager::unpackModel(const QString &id, const PluginModel &model, const QString &archivePath, QString *errorOut)
{
    const QString dest = modelPath(id, model);
    QDir(dest).removeRecursively(); // a retry must not merge with the leftovers of a failed one
    if (!QDir().mkpath(dest)) {
        if (errorOut) {
            *errorOut = i18n("Could not create the folder for %1.", model.name);
        }
        return false;
    }
    // The tools these weights come from publish tarballs on Linux and zips on
    // Windows, so both have to be understood or half the platforms need a
    // different manifest for the same runtime.
    std::unique_ptr<KArchive> archive;
    if (model.unpack == QLatin1String("zip")) {
        archive = std::make_unique<KZip>(archivePath);
    } else {
        archive = std::make_unique<KTar>(archivePath, QStringLiteral("application/x-gzip"));
    }
    if (!archive->open(QIODevice::ReadOnly) || !archive->directory()->copyTo(dest)) {
        QDir(dest).removeRecursively();
        if (errorOut) {
            *errorOut = i18n("%1 could not be unpacked.", model.name);
        }
        return false;
    }
    archive->close();
    // A runtime arrives without the executable bit — the zip format keeps
    // permissions, but the release archives of these tools routinely lose them
    // in transit, and a llama-server that cannot be started fails much later
    // with something unreadable. Anything without a suffix, plus the usual
    // library suffixes, gets it back.
    QDirIterator it(dest, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QFileInfo info(it.next());
        const QString suffix = info.suffix();
        if (!suffix.isEmpty() && suffix != QLatin1String("so") && !suffix.startsWith(QLatin1String("so."))) {
            continue;
        }
        QFile file(info.absoluteFilePath());
        file.setPermissions(file.permissions() | QFileDevice::ExeOwner | QFileDevice::ExeUser);
    }
    return true;
}

QString PluginManager::pythonFor(const QString &id) const
{
    const auto it = m_plugins.constFind(id);
    return it == m_plugins.constEnd() ? QString() : interpreterFor(it.value());
}

QString PluginManager::mcpServerDir()
{
    return QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("mcp"), QStandardPaths::LocateDirectory);
}

QString PluginManager::effectsDir()
{
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/effects");
}

QString PluginManager::pluginForEffect(const QString &effectId) const
{
    return m_effectOwners.value(effectId);
}

QByteArray PluginManager::stampedEffect(const PluginManifest &manifest, const PluginEffect &effect)
{
    QDomDocument doc;
    if (!Xml::docContentFromFile(doc, manifest.rootDir() + QLatin1Char('/') + effect.file, false)) {
        return {};
    }
    QDomElement base = doc.documentElement();
    base.setAttribute(QStringLiteral("wunjo_plugin"), manifest.id());
    return doc.toByteArray(4);
}

QString PluginManager::effectStamp(const QString &path, QString *effectId)
{
    QDomDocument doc;
    if (!Xml::docContentFromFile(doc, path, false)) {
        return {};
    }
    const QDomElement base = doc.documentElement();
    if (effectId) {
        *effectId = base.attribute(QStringLiteral("id"));
    }
    return base.attribute(QStringLiteral("wunjo_plugin"));
}

void PluginManager::syncEffects()
{
    const QString folder = effectsDir();
    QDir dir(folder);
    if (!dir.exists() && !QDir().mkpath(folder)) {
        qWarning() << "Could not create the effects folder" << folder;
        return;
    }
    QStringList addedFiles;
    QStringList removedIds;
    // Drop what plugins left behind. Only stamped files are ours: the same
    // folder holds the effects the user saved from the effect stack.
    const QStringList files = dir.entryList({QStringLiteral("*.xml")}, QDir::Files);
    for (const QString &file : files) {
        const QString path = dir.absoluteFilePath(file);
        QString effectId;
        const QString owner = effectStamp(path, &effectId);
        if (owner.isEmpty() || effectId.isEmpty()) {
            continue;
        }
        if (m_effectOwners.value(effectId) != owner && QFile::remove(path)) {
            removedIds << effectId;
        }
    }
    // Write what the installed plugins bring, skipping the ones already in place
    // so a normal start does not touch the folder at all.
    const QList<PluginManifest> plugins = installedPlugins();
    for (const PluginManifest &manifest : plugins) {
        const QList<PluginEffect> effects = manifest.effects();
        for (const PluginEffect &effect : effects) {
            const QByteArray content = stampedEffect(manifest, effect);
            if (content.isEmpty()) {
                qWarning() << "Could not read effect" << effect.file << "of plugin" << manifest.id();
                continue;
            }
            const QString dest = dir.absoluteFilePath(effect.id + QStringLiteral(".xml"));
            QFile target(dest);
            if (target.exists()) {
                if (target.open(QIODevice::ReadOnly) && target.readAll() == content) {
                    target.close();
                    continue;
                }
                target.close();
            }
            if (!target.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                qWarning() << "Could not write effect" << dest;
                continue;
            }
            target.write(content);
            target.close();
            addedFiles << dest;
        }
    }
    if (!addedFiles.isEmpty() || !removedIds.isEmpty()) {
        Q_EMIT pluginEffectsChanged(addedFiles, removedIds);
    }
}

QString PluginManager::apiKey(const QString &provider) const
{
    if (provider.isEmpty()) {
        return {};
    }
    KConfig config(QStringLiteral("wunjoapikeysrc"), KConfig::SimpleConfig);
    return KConfigGroup(&config, QStringLiteral("keys")).readEntry(provider, QString());
}

void PluginManager::setApiKey(const QString &provider, const QString &key)
{
    if (provider.isEmpty()) {
        return;
    }
    KConfig config(QStringLiteral("wunjoapikeysrc"), KConfig::SimpleConfig);
    KConfigGroup group(&config, QStringLiteral("keys"));
    if (key.isEmpty()) {
        group.deleteEntry(provider);
    } else {
        group.writeEntry(provider, key);
    }
    config.sync();
    // keys are secrets — keep the file readable only by the user
    QFile::setPermissions(config.name(), QFileDevice::ReadOwner | QFileDevice::WriteOwner);
}

PluginManager::ImportCandidate PluginManager::inspect(const QString &path) const
{
    ImportCandidate candidate;
    const QFileInfo info(path);
    if (!info.exists()) {
        candidate.manifest = PluginManifest();
        return candidate;
    }
    if (info.isDir()) {
        candidate.sourceDir = info.absoluteFilePath();
        candidate.manifest = PluginManifest::fromDir(candidate.sourceDir, true);
        return candidate;
    }
    // treat as a .wunjoplugin archive
    KZip zip(info.absoluteFilePath());
    if (!zip.open(QIODevice::ReadOnly)) {
        return candidate; // invalid: empty sourceDir
    }
    auto temp = QSharedPointer<QTemporaryDir>::create();
    if (!temp->isValid()) {
        return candidate;
    }
    zip.directory()->copyTo(temp->path());
    zip.close();
    QString src = temp->path();
    if (!QFileInfo::exists(src + QStringLiteral("/plugin.json"))) {
        // archive may wrap the plugin in a single top-level folder
        const QStringList subdirs = QDir(src).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const QString &sub : subdirs) {
            if (QFileInfo::exists(src + QLatin1Char('/') + sub + QStringLiteral("/plugin.json"))) {
                src = src + QLatin1Char('/') + sub;
                break;
            }
        }
    }
    candidate.temp = temp;
    candidate.sourceDir = src;
    // an extracted archive lives in a temp dir, so do not require id==folder
    candidate.manifest = PluginManifest::fromDir(src, false);
    return candidate;
}

/** @brief The bin folder an effect's results belong in, made on first use.
 *
 * A plugin that renders puts a new clip in the bin every time it is run, and a
 * project that uses three of them ends up with the results of all three loose
 * among the footage. One folder per effect keeps each plugin's output where its
 * user will look for it. Returns "-1" — the bin root — if there is no project
 * yet or the folder could not be made, which is where clips landed before.
 */
static QString effectResultsFolder(const QString &effectName)
{
    std::shared_ptr<ProjectItemModel> model = pCore->projectItemModel();
    if (!model || effectName.isEmpty()) {
        return QStringLiteral("-1");
    }
    std::shared_ptr<ProjectFolder> root = model->getRootFolder();
    if (!root) {
        return QStringLiteral("-1");
    }
    for (int i = 0; i < root->childCount(); ++i) {
        auto child = std::static_pointer_cast<AbstractProjectItem>(root->child(i));
        if (child && child->itemType() == AbstractProjectItem::FolderItem && child->name() == effectName) {
            return child->clipId();
        }
    }
    QString folderId;
    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    if (!model->requestAddFolder(folderId, effectName, root->clipId(), undo, redo)) {
        return QStringLiteral("-1");
    }
    return folderId.isEmpty() ? QStringLiteral("-1") : folderId;
}

/** @brief Put a finished render on its own track, above the clip it came from.
 *
 * The bin keeps the file — that is its catalogue — but someone who just pressed
 * Generate is looking at the timeline, not at a folder. A track of its own
 * directly above the source, starting on the same frame, puts the answer where
 * the question was asked, and a track of its own means the render never covers
 * anything that was already there.
 *
 * Video and audio go in as two separate insertions (the "V" and "A" prefixes
 * the timeline understands) rather than as one clip: dropping an A/V clip asks
 * the timeline to find an audio target track and to invent one if it cannot,
 * which is a conversation the user did not start.
 */
static void placeResultOnTimeline(const ObjectId &owner, const QString &binId)
{
    // An effect sitting on a bin clip has no position to be placed at; the bin
    // is the whole answer there.
    if (binId.isEmpty() || owner.type != WunjoObjectType::TimelineClip) {
        return;
    }
    WunjoDoc *doc = pCore->currentDoc();
    if (!doc) {
        return;
    }
    std::shared_ptr<TimelineItemModel> timeline = doc->getTimeline(owner.uuid);
    if (!timeline || !timeline->isClip(owner.itemId)) {
        return;
    }
    const int sourceTrack = timeline->getClipTrackId(owner.itemId);
    if (sourceTrack <= -1) {
        return;
    }
    const int position = timeline->getClipPosition(owner.itemId);
    std::shared_ptr<ProjectClip> master = pCore->projectItemModel()->getClipByBinID(binId);
    if (!master) {
        return;
    }
    const ClipType::ProducerType type = master->clipType();
    const bool wantsVideo = type != ClipType::Audio;
    const bool wantsAudio = type == ClipType::Audio || master->hasAudioAndVideo();

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    bool placed = false;

    if (wantsVideo) {
        int videoTrack = -1;
        if (timeline->requestTrackInsertion(timeline->getTrackPosition(sourceTrack) + 1, videoTrack, QString(), false, undo, redo)) {
            int clipId = -1;
            placed = timeline->requestClipInsertion(QStringLiteral("V") + binId, videoTrack, position, clipId, true, true, false, undo, redo);
        }
    }
    if (wantsAudio) {
        // Audio tracks live below the video ones, so the new one belongs on top
        // of the audio stack rather than above the source's video track.
        int audioTrack = -1;
        if (timeline->requestTrackInsertion(int(timeline->getTracksIds(true).count()), audioTrack, QString(), true, undo, redo)) {
            int clipId = -1;
            const bool ok = timeline->requestClipInsertion(QStringLiteral("A") + binId, audioTrack, position, clipId, true, true, false, undo, redo);
            placed = placed || ok;
        }
    }
    if (placed) {
        pCore->pushUndo(undo, redo, i18n("Add the rendered result to the timeline"));
    } else {
        // half a placement is worse than none: take the tracks back out again
        undo();
    }
}

static bool copyRecursively(const QString &src, const QString &dst, QString *errorOut)
{
    QDir().mkpath(dst);
    QDir srcDir(src);
    const QFileInfoList entries = srcDir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
    for (const QFileInfo &entry : entries) {
        const QString target = dst + QLatin1Char('/') + entry.fileName();
        if (entry.isDir()) {
            if (!copyRecursively(entry.absoluteFilePath(), target, errorOut)) {
                return false;
            }
        } else {
            QFile::remove(target);
            if (!QFile::copy(entry.absoluteFilePath(), target)) {
                if (errorOut) {
                    *errorOut = QStringLiteral("cannot copy %1").arg(entry.fileName());
                }
                return false;
            }
        }
    }
    return true;
}

bool PluginManager::install(const ImportCandidate &candidate, QString *errorOut)
{
    if (!candidate.valid()) {
        if (errorOut) {
            *errorOut = i18n("the plugin is not valid");
        }
        return false;
    }
    if (!candidate.manifest.osSupported()) {
        if (errorOut) {
            *errorOut = i18n("this plugin does not support %1", PluginManifest::currentOs());
        }
        return false;
    }
    const QString dest = userPluginsDir() + QLatin1Char('/') + candidate.manifest.id();
    if (QFileInfo::exists(dest)) {
        QDir(dest).removeRecursively();
    }
    if (!copyRecursively(candidate.sourceDir, dest, errorOut)) {
        QDir(dest).removeRecursively();
        return false;
    }
    rescan();
    syncEffects();
    Q_EMIT pluginsChanged();
    return true;
}

bool PluginManager::uninstall(const QString &id, QString *errorOut)
{
    if (isBundled(id) && !QFileInfo::exists(userPluginsDir() + QLatin1Char('/') + id)) {
        if (errorOut) {
            *errorOut = i18n("built-in plugins cannot be removed");
        }
        return false;
    }
    const QString dir = userPluginsDir() + QLatin1Char('/') + id;
    if (QFileInfo::exists(dir) && !QDir(dir).removeRecursively()) {
        if (errorOut) {
            *errorOut = i18n("could not delete the plugin folder");
        }
        return false;
    }
    // drop the private venv too (never the shared one)
    const QString venv = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/venv-") + id;
    if (QFileInfo::exists(venv)) {
        QDir(venv).removeRecursively();
    }
    rescan();
    // the plugin no longer claims its effects, so this takes them off the list
    syncEffects();
    Q_EMIT pluginsChanged();
    return true;
}

QString PluginManager::interpreterFor(const PluginManifest &m) const
{
    if (m.hasDependencies()) {
        const QString py = venvPython(m.venvName());
        if (!py.isEmpty()) {
            return py;
        }
        // environment not built yet — fall back so the stub still runs; the
        // real venv build lands with the plugin-task work
    }
    QString py = QStandardPaths::findExecutable(QStringLiteral("python3"));
    if (py.isEmpty()) {
        py = QStandardPaths::findExecutable(QStringLiteral("python"));
    }
    return py;
}

namespace {

/** @brief What the plugin has told us so far over its stdout protocol. */
struct PluginJobState
{
    QString buffer; ///< bytes read but not yet a complete line
    QJsonObject result;
    QString message; ///< last human-readable line, used to explain a failure
    bool failed{false};
    /** @brief A `result:` line was seen. Not the same as a non-empty result: a
     *  plugin may legitimately finish with nothing to hand back, and one that
     *  finishes without saying so at all did not finish its work. */
    bool reported{false};
};

/** @brief Keep what a plugin printed next to the job it printed it for. */
void appendJobLog(const QString &workDir, const QString &text)
{
    if (workDir.isEmpty() || text.isEmpty()) {
        return;
    }
    QFile file(workDir + QStringLiteral("/plugin.log"));
    if (file.open(QIODevice::Append | QIODevice::WriteOnly)) {
        file.write(text.toUtf8());
    }
}

/** @brief The line a failed run is most likely to be explained by. */
QString lastMeaningfulLine(const QString &workDir)
{
    QFile file(workDir + QStringLiteral("/plugin.log"));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    const QStringList lines = QString::fromUtf8(file.readAll()).split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (auto line = lines.crbegin(); line != lines.crend(); ++line) {
        const QString trimmed = line->trimmed();
        if (!trimmed.isEmpty() && !trimmed.startsWith(QLatin1String("progress:"))) {
            return trimmed.left(200);
        }
    }
    return {};
}

/** @brief Turn a plugin's `need:` payload into a sentence that says what to do.
 *  The protocol is deliberately tiny (kind + name), so every plugin gets the
 *  same wording instead of showing the user raw json. */
QString describeNeed(const QString &payload)
{
    const QJsonObject need = QJsonDocument::fromJson(payload.toUtf8()).object();
    const QString kind = need.value(QStringLiteral("kind")).toString();
    const QString name = need.value(QStringLiteral("name")).toString();
    if (kind == QLatin1String("model")) {
        return name.isEmpty() ? i18n("A model is missing — download it in the plugin's settings.")
                              : i18n("The model %1 is missing — download it in the plugin's settings.", name);
    }
    if (kind == QLatin1String("api_key")) {
        const QString provider = need.value(QStringLiteral("provider")).toString();
        return provider.isEmpty() ? i18n("An API key is missing — add it in the plugin's settings.")
                                  : i18n("The %1 API key is missing — add it in the plugin's settings.", provider);
    }
    // unknown kind: better the plugin's own words than nothing
    return payload.simplified();
}

void parsePluginLine(const QString &line, PluginJobState &state, const std::function<void(int)> &onProgress)
{
    if (line.isEmpty()) {
        return;
    }
    if (line.startsWith(QLatin1String("progress:"))) {
        if (onProgress) {
            onProgress(line.mid(9).trimmed().toInt());
        }
    } else if (line.startsWith(QLatin1String("result:"))) {
        state.result = QJsonDocument::fromJson(line.mid(7).toUtf8()).object();
        state.reported = true;
    } else if (line.startsWith(QLatin1String("need:"))) {
        state.failed = true;
        state.message = describeNeed(line.mid(5));
    } else if (line.startsWith(QLatin1String("info:"))) {
        state.message = line.mid(5);
    }
}

/** @brief Make every produced path absolute so callers never need the work dir. */
QJsonObject resolveOutputs(QJsonObject result, const QString &workDir)
{
    const QJsonArray outputs = result.value(QStringLiteral("outputs")).toArray();
    if (outputs.isEmpty()) {
        return result;
    }
    QJsonArray resolved;
    for (const QJsonValue &value : outputs) {
        QJsonObject output = value.toObject();
        QString path = output.value(QStringLiteral("path")).toString();
        if (!path.isEmpty() && !QDir::isAbsolutePath(path)) {
            output.insert(QStringLiteral("path"), QString(workDir + QLatin1Char('/') + path));
        }
        resolved.append(output);
    }
    result.insert(QStringLiteral("outputs"), resolved);
    return result;
}

} // namespace

QProcess *PluginManager::startProcess(const PluginManifest &manifest, const QJsonObject &input, QString *workDirOut, QString *errorOut)
{
    const QString id = manifest.id();
    const QString python = interpreterFor(manifest);
    if (python.isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("Python was not found — cannot run '%1'.", manifest.name());
        }
        return nullptr;
    }

    const QString jobId = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    QString base = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    if (auto *doc = pCore->currentDoc()) {
        const QString folder = doc->projectDataFolder();
        if (!folder.isEmpty()) {
            base = folder;
        }
    }
    const QString workDir = base + QStringLiteral("/plugins-work/") + jobId;
    QDir().mkpath(workDir);

    QJsonObject project;
    project.insert(QStringLiteral("fps"), pCore->getCurrentFps());
    const QSize size = pCore->getCurrentFrameSize();
    project.insert(QStringLiteral("width"), size.width());
    project.insert(QStringLiteral("height"), size.height());
    // Where a plugin may keep something that belongs to this project rather
    // than to this run — the assistant keeps its conversations there, so they
    // survive the process that answered.
    if (auto *doc = pCore->currentDoc()) {
        project.insert(QStringLiteral("data_folder"), doc->projectDataFolder());
    }

    // parameter values chosen on the plugin's settings tab
    QJsonObject params;
    QString device;
    {
        KConfig config(QStringLiteral("wunjopluginsrc"), KConfig::SimpleConfig);
        KConfigGroup group(&config, id);
        const QList<PluginParam> declared = manifest.params();
        for (const PluginParam &param : declared) {
            params.insert(param.key, QJsonValue::fromVariant(group.readEntry(param.key, param.defaultValue)));
        }
        // Not a declared parameter but a property of the machine: which device
        // the user pointed this plugin at. Empty means they left it to decide.
        device = group.readEntry("device", QString());
    }
    // Per-call params from the caller (e.g. the AI over run_plugin) override the
    // saved settings-tab defaults, so a prompt/model can be passed at call time.
    const QJsonObject callerParams = input.value(QStringLiteral("params")).toObject();
    for (auto pit = callerParams.constBegin(); pit != callerParams.constEnd(); ++pit) {
        params.insert(pit.key(), pit.value());
    }

    QJsonObject job;
    job.insert(QStringLiteral("job_id"), jobId);
    job.insert(QStringLiteral("plugin_id"), id);
    job.insert(QStringLiteral("input"), input);
    job.insert(QStringLiteral("params"), params);
    job.insert(QStringLiteral("output_dir"), workDir);
    job.insert(QStringLiteral("project"), project);
    job.insert(QStringLiteral("ffmpeg"), WunjoSettings::ffmpegpath());
    // "cuda:0", "cpu", or absent — the plugin picks for itself when absent,
    // which is what the Automatic entry on its settings tab means.
    if (!device.isEmpty()) {
        job.insert(QStringLiteral("device"), device);
    }

    const QString jobFile = workDir + QStringLiteral("/job.json");
    QFile file(jobFile);
    if (!file.open(QIODevice::WriteOnly)) {
        if (errorOut) {
            *errorOut = i18n("Could not prepare the job for '%1'.", manifest.name());
        }
        return nullptr;
    }
    file.write(QJsonDocument(job).toJson(QJsonDocument::Compact));
    file.close();

    auto *process = new QProcess(this);
    process->setProcessChannelMode(QProcess::MergedChannels);
    const QString entry = manifest.rootDir() + QLatin1Char('/') + manifest.entry();

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    bool envChanged = false;

    // Where its weights actually are. A plugin cannot work this out from its own
    // location any more: a bundled one runs from the read-only install dir while
    // its models were downloaded elsewhere.
    const QString models = modelsDir(id);
    if (!models.isEmpty()) {
        env.insert(QStringLiteral("WUNJO_MODELS_DIR"), models);
        envChanged = true;
    }

    // The device chosen on the plugin's settings tab, for plugins that read the
    // environment rather than job.json. Absent means Automatic, exactly as in
    // the job file.
    {
        KConfig config(QStringLiteral("wunjopluginsrc"), KConfig::SimpleConfig);
        const QString device = KConfigGroup(&config, manifest.id()).readEntry("device", QString());
        if (!device.isEmpty()) {
            env.insert(QStringLiteral("WUNJO_DEVICE"), device);
            envChanged = true;
        }
    }

    // Hand the API key to the plugin through the environment (never argv, never
    // into any model context). The var name defaults to WUNJO_KEY_<PROVIDER>.
    if (manifest.kind() == QLatin1String("api") && !manifest.providerName().isEmpty()) {
        const QString key = apiKey(manifest.providerName());
        if (!key.isEmpty()) {
            QString var = manifest.providerKeySetting();
            if (var.isEmpty()) {
                var = QStringLiteral("WUNJO_KEY_") + manifest.providerName().toUpper();
            }
            env.insert(var, key);
            envChanged = true;
        }
    }

    // How to talk back to the editor, for every plugin and not only for the
    // assistant: the client that speaks to us — the same copy an outside agent
    // is given — and the exact address to reach *this* copy on.
    //
    // An assistant is not the only plugin with something to say. The cut finder
    // places its cuts on the timeline; anything that analyses a clip may want
    // to leave a marker on it. They were all left to find the editor by
    // themselves, which meant D-Bus, which is gone — the cut finder found
    // thirty-five cuts and placed none of them, reporting only that "the editor
    // did not answer". Guessing does not work anyway: a socket file can only be
    // looked for on Unix, Windows named pipes cannot be enumerated at all, and
    // with two copies of the application open the wrong one is found.
    const QString mcpDir = mcpServerDir();
    if (!mcpDir.isEmpty()) {
        env.insert(QStringLiteral("WUNJO_MCP_DIR"), mcpDir);
        envChanged = true;
    }
    if (pCore->window() != nullptr) {
        const QString socket = pCore->window()->scriptingSocketName();
        if (!socket.isEmpty()) {
            env.insert(QStringLiteral("WUNJO_SOCKET"), socket);
            envChanged = true;
        }
    }
    if (manifest.isAgent()) {
        // What this machine can run the model on — only the assistant loads one
        // that cares.
        env.insert(QStringLiteral("WUNJO_GPU_BACKEND"), gpuBackend());
        envChanged = true;
    }

    if (envChanged) {
        process->setProcessEnvironment(env);
    }

    if (workDirOut) {
        *workDirOut = workDir;
    }
    process->start(python, {entry, QStringLiteral("--job"), jobFile});
    return process;
}

QString PluginManager::runPluginJob(const QString &id, const QJsonObject &input, QObject *context, const std::function<void(int)> &onProgress,
                                    const std::function<void(const QJsonObject &, const QString &)> &onFinished)
{
    const auto it = m_plugins.constFind(id);
    if (it == m_plugins.constEnd()) {
        if (onFinished) {
            onFinished({}, i18n("'%1' is not installed.", id));
        }
        return {};
    }
    const PluginManifest manifest = it.value();
    QString workDir;
    QString error;
    QProcess *process = startProcess(manifest, input, &workDir, &error);
    if (process == nullptr) {
        if (onFinished) {
            onFinished({}, error);
        }
        return {};
    }
    // A job the caller can call off. The chat needs it — an answer the user has
    // stopped waiting for should stop costing them the graphics card — and the
    // handle is dropped as soon as the process is gone, so nothing stale is
    // ever killed by a caller holding an old one.
    const QString handle = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    m_runningJobs.insert(handle, process);
    connect(process, &QObject::destroyed, this, [this, handle]() { m_runningJobs.remove(handle); });
    const QString name = manifest.name();
    auto state = QSharedPointer<PluginJobState>::create();
    QObject *owner = context != nullptr ? context : this;
    connect(process, &QProcess::readyReadStandardOutput, owner, [process, state, workDir, onProgress]() {
        const QString chunk = QString::fromUtf8(process->readAllStandardOutput());
        // Everything the plugin said, kept beside its job. The protocol only
        // carries progress and a final result, so a traceback — the one thing
        // worth reading when a run produces nothing — was parsed for a status
        // line and thrown away. Now the job folder explains itself.
        appendJobLog(workDir, chunk);
        state->buffer += chunk;
        int newline = state->buffer.indexOf(QLatin1Char('\n'));
        while (newline >= 0) {
            parsePluginLine(state->buffer.left(newline).trimmed(), *state, onProgress);
            state->buffer.remove(0, newline + 1);
            newline = state->buffer.indexOf(QLatin1Char('\n'));
        }
    });
    connect(process, &QProcess::finished, owner, [process, state, workDir, name, onProgress, onFinished](int exitCode, QProcess::ExitStatus) {
        // the last line may arrive without its newline
        const QString tail = QString::fromUtf8(process->readAllStandardOutput());
        appendJobLog(workDir, tail);
        state->buffer += tail;
        parsePluginLine(state->buffer.trimmed(), *state, onProgress);
        QString error;
        // A run that ends without a `result:` line did not do the work, whatever
        // its exit code says. Plugins decline for good reasons — "choose an
        // audio preset first", "a face preset is made from a photo" — and used
        // to do it by printing one `info:` line and returning 0. That reached
        // nobody: the caller saw a success with no outputs, the progress card in
        // the chat span forever, and the user was left watching a render that
        // had never started. Refusals are failures, and they carry the plugin's
        // own sentence as the reason.
        if (exitCode != 0 || state->failed || !state->reported) {
            // Prefer what the plugin actually said, wherever it said it: a
            // status line if it left one, otherwise the last thing it printed
            // before dying. "X failed." on its own helps nobody.
            error = state->message;
            if (error.isEmpty()) {
                error = lastMeaningfulLine(workDir);
            }
            error = error.isEmpty() ? i18n("%1 failed.", name) : i18n("%1: %2", name, error);
        }
        if (onFinished) {
            onFinished(resolveOutputs(state->result, workDir), error);
        }
        process->deleteLater();
    });
    connect(process, &QProcess::errorOccurred, owner, [process, name, onFinished](QProcess::ProcessError processError) {
        // A crash also emits finished(); only report here when the process never
        // started, so the callback runs exactly once.
        if (processError == QProcess::FailedToStart) {
            if (onFinished) {
                onFinished({}, i18n("Could not launch '%1'.", name));
            }
            process->deleteLater();
        }
    });
    return handle;
}

bool PluginManager::cancelPluginJob(const QString &handle)
{
    QProcess *process = m_runningJobs.value(handle);
    if (process == nullptr) {
        return false;
    }
    // terminate() first so the plugin can put its own tools down (the assistant
    // leaves a running model behind otherwise); kill() is the backstop.
    process->terminate();
    if (!process->waitForFinished(2000)) {
        process->kill();
    }
    return true;
}

namespace {
/** @brief One render, identified by what it renders — not by the widget that
 *  happens to show it at the moment. */
QString effectJobKey(const ObjectId &owner, int effectItemId)
{
    return QStringLiteral("%1/%2/%3/%4").arg(int(owner.type)).arg(owner.itemId).arg(owner.uuid.toString(), QString::number(effectItemId));
}
} // namespace

int PluginManager::effectJobProgress(const ObjectId &owner, int effectItemId) const
{
    return m_effectJobs.value(effectJobKey(owner, effectItemId), JobNone);
}

double PluginManager::driverCudaVersion()
{
    static double cached = -1;
    if (cached >= 0) {
        return cached;
    }
    cached = 0;
    // The lowest driver each CUDA runtime needs. A wheel built for a newer line
    // than this simply refuses to initialize and the plugin drops to the CPU.
    static const QList<QPair<int, double>> driverToCuda = {{560, 12.6}, {555, 12.5}, {550, 12.4}, {545, 12.3}, {535, 12.2},
                                                           {530, 12.1}, {525, 12.0}, {520, 11.8}, {515, 11.7}, {510, 11.6}};
    // Read the kernel module rather than asking nvidia-smi: inside a sandbox
    // that tool is not there (it belongs to the host), while /proc is.
    QFile nvrm(QStringLiteral("/proc/driver/nvidia/version"));
    if (nvrm.open(QIODevice::ReadOnly | QIODevice::Text)) {
        static const QRegularExpression moduleVersion(QStringLiteral("Kernel Module\\s+([0-9]+)\\."));
        const QRegularExpressionMatch match = moduleVersion.match(QString::fromUtf8(nvrm.readAll()));
        if (match.hasMatch()) {
            const int major = match.captured(1).toInt();
            for (const auto &entry : driverToCuda) {
                if (major >= entry.first) {
                    cached = entry.second;
                    break;
                }
            }
        }
    }
    if (cached == 0) {
        // outside a sandbox this is the more direct answer, and it also covers
        // setups where /proc is not what we expect
        QProcess smi;
        smi.start(QStringLiteral("nvidia-smi"), {});
        if (smi.waitForFinished(4000) && smi.exitCode() == 0) {
            // the banner carries it: "… Driver Version: 560.35.03  CUDA Version: 12.6 |"
            static const QRegularExpression cudaLine(QStringLiteral("CUDA Version:\\s*([0-9]+\\.[0-9]+)"));
            const QRegularExpressionMatch match = cudaLine.match(QString::fromUtf8(smi.readAllStandardOutput()));
            if (match.hasMatch()) {
                cached = match.captured(1).toDouble();
            }
        }
    }
    qDebug() << "::: GPU driver supports CUDA" << cached;
    return cached;
}

double PluginManager::gpuVramGb()
{
    static double cached = -1;
    if (cached >= 0) {
        return cached;
    }
    cached = 0;
    // NVML is the only way to ask NVIDIA from inside the sandbox: nvidia-smi
    // belongs to the host and is not mounted, but the driver's own library is
    // reachable under /run/host thanks to --filesystem=host. Loaded by name
    // first, so a normal (non-flatpak) install answers without the detour.
    static const QStringList nvmlCandidates = {QStringLiteral("libnvidia-ml.so.1"),
                                               QStringLiteral("/run/host/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
                                               QStringLiteral("/run/host/usr/lib64/libnvidia-ml.so.1"), QStringLiteral("nvml")};
    struct NvmlMemory {
        quint64 total;
        quint64 free;
        quint64 used;
    };
    for (const QString &candidate : nvmlCandidates) {
        QLibrary nvml(candidate);
        if (!nvml.load()) {
            continue;
        }
        auto init = reinterpret_cast<int (*)()>(nvml.resolve("nvmlInit_v2"));
        auto handleByIndex = reinterpret_cast<int (*)(unsigned int, void **)>(nvml.resolve("nvmlDeviceGetHandleByIndex_v2"));
        auto memoryInfo = reinterpret_cast<int (*)(void *, NvmlMemory *)>(nvml.resolve("nvmlDeviceGetMemoryInfo"));
        auto shutdown = reinterpret_cast<int (*)()>(nvml.resolve("nvmlShutdown"));
        if (!init || !handleByIndex || !memoryInfo || init() != 0) {
            nvml.unload();
            continue;
        }
        void *device = nullptr;
        NvmlMemory memory{0, 0, 0};
        // Card 0 only: a second GPU does not add up, the model runs on one of them.
        if (handleByIndex(0, &device) == 0 && memoryInfo(device, &memory) == 0 && memory.total > 0) {
            cached = double(memory.total) / (1024.0 * 1024.0 * 1024.0);
        }
        if (shutdown) {
            shutdown();
        }
        if (cached > 0) {
            break;
        }
    }
    if (cached == 0) {
        // AMD exposes it through sysfs, which the sandbox does see
        const QStringList cards = QDir(QStringLiteral("/sys/class/drm")).entryList({QStringLiteral("card[0-9]")}, QDir::Dirs);
        for (const QString &card : cards) {
            QFile total(QStringLiteral("/sys/class/drm/") + card + QStringLiteral("/device/mem_info_vram_total"));
            if (total.open(QIODevice::ReadOnly | QIODevice::Text)) {
                const double bytes = QString::fromUtf8(total.readAll()).trimmed().toDouble();
                cached = qMax(cached, bytes / (1024.0 * 1024.0 * 1024.0));
            }
        }
    }
    if (cached == 0) {
        // outside a sandbox this is the direct answer
        QProcess smi;
        smi.start(QStringLiteral("nvidia-smi"), {QStringLiteral("--query-gpu=memory.total"), QStringLiteral("--format=csv,noheader,nounits")});
        if (smi.waitForFinished(4000) && smi.exitCode() == 0) {
            const double mb = QString::fromUtf8(smi.readAllStandardOutput()).split(QLatin1Char('\n')).first().trimmed().toDouble();
            cached = mb / 1024.0;
        }
    }
    qDebug() << "::: GPU video memory (GB)" << cached;
    return cached;
}

QString PluginManager::gpuBackend()
{
    if (gpuVramGb() <= 0) {
        return QStringLiteral("cpu");
    }
    // CUDA builds are the fastest, but only where the driver can load them;
    // Vulkan covers the rest (AMD, Intel, and NVIDIA on an older driver).
    return driverCudaVersion() >= 11.8 ? QStringLiteral("cuda") : QStringLiteral("vulkan");
}

QString PluginManager::venvDir(const QString &venvName)
{
    if (venvName.isEmpty()) {
        return {};
    }
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QLatin1Char('/') + venvName;
}

QString PluginManager::venvPython(const QString &venvName)
{
    const QString dir = venvDir(venvName);
    if (dir.isEmpty()) {
        return {};
    }
    // Both layouts are tried on every platform rather than picked by #ifdef: an
    // environment can be carried between machines, and asking the filesystem
    // costs one stat.
    for (const QString &relative : {QStringLiteral("/bin/python3"), QStringLiteral("/Scripts/python.exe"), QStringLiteral("/bin/python"),
                                    QStringLiteral("/Scripts/python3.exe")}) {
        if (QFileInfo::exists(dir + relative)) {
            return dir + relative;
        }
    }
    return {};
}

qint64 PluginManager::pendingDownloadBytes(const PluginManifest &manifest)
{
    // A private venv with a GPU build of torch in it is the biggest thing a
    // plugin brings, and nothing in the manifest declares its size. This is a
    // deliberately blunt reserve: it exists so the check errs towards refusing
    // an install that would just fit, not towards filling the disk.
    static constexpr qint64 kVenvReserveBytes = 3LL * 1024 * 1024 * 1024;

    qint64 bytes = 0;
    const QList<PluginModel> models = applicableModels(manifest);
    for (const PluginModel &model : models) {
        if (model.url.isEmpty()) {
            continue;
        }
        if (instance().modelState(manifest.id(), model) != ModelReady) {
            // What an interrupted download already brought in stays on disk and
            // will be continued, not fetched again — asking for room for it a
            // second time is what turns "resume the last gigabyte" into "not
            // enough space".
            const qint64 fetched = FileDownloader::resumableBytes(instance().downloadTarget(manifest.id(), model));
            bytes += qMax<qint64>(0, model.sizeMb * 1024 * 1024 - fetched);
        }
    }
    if (manifest.hasDependencies() && venvPython(manifest.venvName()).isEmpty()) {
        // No interpreter means no usable environment, even if the folder is
        // there: a build that died half way leaves the directory behind.
        bytes += kVenvReserveBytes;
    }
    return bytes;
}

QString PluginManager::installBlocker(const PluginManifest &manifest) const
{
    if (!manifest.isValid()) {
        return i18n("The manifest of this plugin cannot be read.");
    }
    const QString versionBlocker = manifest.appVersionBlocker();
    if (!versionBlocker.isEmpty()) {
        return versionBlocker;
    }
    if (!manifest.osSupported()) {
        return i18n("%1 does not run on this operating system.", manifest.name());
    }

    return downloadBlocker(pendingDownloadBytes(manifest));
}

QString PluginManager::downloadBlocker(qint64 bytes)
{
    if (bytes <= 0) {
        // Nothing has to be fetched: neither the network nor the disk matters.
        return {};
    }

    // Nothing is asked about the network. Inside the sandbox the reachability
    // backend cannot see NetworkManager and answers "disconnected" on a machine
    // that is downloading happily, which refused perfectly good downloads. And
    // it no longer buys anything: a download that starts without a line waits
    // and continues by itself once there is one.
    const QString dataRoot = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    QDir().mkpath(dataRoot); // QStorageInfo needs a path that exists to resolve the device
    const QStorageInfo storage(dataRoot);
    if (storage.isValid()) {
        // Leave the disk room to breathe: a machine with nothing free is one
        // where the user's own project files stop saving.
        static constexpr qint64 kHeadroomBytes = 1LL * 1024 * 1024 * 1024;
        const qint64 available = storage.bytesAvailable();
        if (available < bytes + kHeadroomBytes) {
            return i18n("This needs about %1 free in %2, and there is %3.", KIO::convertSize(bytes), dataRoot,
                        KIO::convertSize(qMax<qint64>(available, 0)));
        }
    }
    return {};
}

QString PluginManager::runBlocker(const QString &id) const
{
    const PluginManifest manifest = m_plugins.value(id);
    if (!manifest.isValid()) {
        return i18n("'%1' is not installed.", id);
    }
    // The application it was built for comes before anything it is missing:
    // downloading weights for a plugin that cannot run here helps nobody.
    const QString versionBlocker = manifest.appVersionBlocker();
    if (!versionBlocker.isEmpty()) {
        return versionBlocker;
    }
    const QList<PluginModel> models = applicableModels(manifest);
    for (const PluginModel &model : models) {
        switch (modelState(id, model)) {
        case ModelMissing:
            return i18n("%1 needs the model %2 — download it in the plugin's settings.", manifest.name(), model.name);
        case ModelIncomplete:
            return i18n("The model %1 was not downloaded completely — get it again in the plugin's settings.", model.name);
        case ModelReady:
            break;
        }
    }
    return {};
}

QString PluginManager::installMessage(const QString &id) const
{
    return m_installing.value(id);
}

void PluginManager::fetchModels(const PluginManifest &manifest, const std::function<void()> &onDone)
{
    const QString id = manifest.id();
    const QList<PluginModel> models = applicableModels(manifest);
    for (const PluginModel &model : models) {
        const QString key = id + QLatin1Char('/') + model.name;
        if (modelState(id, model) == ModelReady || model.url.isEmpty() || m_failedDownloads.contains(key)) {
            continue;
        }
        const bool archived = !model.unpack.isEmpty();
        const QString dest = downloadTarget(id, model);
        // A model name may carry a folder — "whisper/tiny.pt" — because that is
        // where the engine looks for it.
        if (!QDir().mkpath(QFileInfo(dest).absolutePath())) {
            m_failedDownloads.insert(key);
            m_installing.insert(id, i18n("cannot create the folder for %1", model.name));
            Q_EMIT installProgress(id, m_installing.value(id));
            continue;
        }
        m_installing.insert(id, i18n("downloading %1…", model.name));
        Q_EMIT installProgress(id, m_installing.value(id));
        auto *download = new FileDownloader(this);
        // Every chunk that arrives is not news: at a percent a time this says
        // the same thing a hundred times over a download instead of thousands.
        connect(download, &FileDownloader::progress, this, [this, id, model, shown = -1](qint64 received, qint64 total) mutable {
            const int percent = total > 0 ? int(received * 100 / total) : -1;
            if (percent == shown) {
                return;
            }
            shown = percent;
            m_installing.insert(id, percent < 0 ? i18n("downloading %1…", model.name) : i18n("downloading %1 — %2%", model.name, percent));
            Q_EMIT installProgress(id, m_installing.value(id));
        });
        // A break in the line is not a failure here: the downloader is already
        // waiting to continue, and the message says so instead of looking stuck.
        connect(download, &FileDownloader::retrying, this, [this, id, model](const QString &reason, int seconds) {
            m_installing.insert(id, i18n("%1: %2 — retrying in %3 s", model.name, reason, seconds));
            Q_EMIT installProgress(id, m_installing.value(id));
        });
        connect(download, &FileDownloader::finished, this, [this, manifest, model, dest, key, archived, onDone, download](bool ok, const QString &error) {
            const QString id = manifest.id();
            download->deleteLater();
            if (!ok) {
                // The fragment stays where it is — the next install continues
                // from it rather than fetching those gigabytes again. The name
                // is remembered so that the pass over what is still missing
                // does not land on this one again a moment later.
                m_failedDownloads.insert(key);
                m_installing.insert(id, i18n("could not download %1: %2", model.name, error));
                Q_EMIT installProgress(id, m_installing.value(id));
            } else if (archived) {
                QString unpackError;
                unpackModel(id, model, dest, &unpackError);
                QFile::remove(dest);
            }
            // Whatever happened to this one, carry on with the rest: one weight
            // that will not come is no reason to abandon the others.
            fetchModels(manifest, onDone);
        });
        download->start(QUrl(model.url), dest);
        return; // one at a time — the rest follow from this download's result
    }
    // Nothing left to fetch. What failed this time is free to be tried again
    // the next time the plugin is installed or a weight is asked for by hand.
    for (const PluginModel &model : models) {
        m_failedDownloads.remove(id + QLatin1Char('/') + model.name);
    }
    m_installing.remove(id);
    Q_EMIT installProgress(id, QString());
    if (onDone) {
        onDone();
    }
}

void PluginManager::installPlugin(const QString &id)
{
    const PluginManifest manifest = m_plugins.value(id);
    if (!manifest.isValid() || m_installing.contains(id)) {
        return;
    }
    // Refuse before writing anything. The message is the whole point: it names
    // what is wrong and how much of it, which "the plugin failed" never did.
    // Nothing is put in m_installing — this is not an install in progress, and
    // the next attempt must be free to start once there is room or a
    // connection. Callers ask @ref installBlocker themselves when they need the
    // reason.
    const QString blocker = installBlocker(manifest);
    if (!blocker.isEmpty()) {
        Q_EMIT installProgress(id, blocker);
        return;
    }
    m_installing.insert(id, i18n("preparing %1…", manifest.name()));
    Q_EMIT installProgress(id, m_installing.value(id));

    // The environment first: the weights are useless without something to run
    // them, and building it is what takes the minutes.
    if (manifest.hasDependencies()) {
        auto *env = new PluginPythonEnv(manifest, this);
        connect(env, &AbstractPythonInterface::dependenciesAvailable, this, [this, manifest, env]() {
            env->deleteLater();
            fetchModels(manifest, nullptr);
        });
        connect(env, &AbstractPythonInterface::dependenciesMissing, this, [this, manifest, env](const QStringList &) {
            env->deleteLater();
            // Say so, but still fetch the weights: the user can repair an
            // environment from the plugin's own page, and a half-done install
            // that fetched nothing would only have to start again.
            m_installing.insert(manifest.id(), i18n("the environment of %1 could not be built", manifest.name()));
            Q_EMIT installProgress(manifest.id(), m_installing.value(manifest.id()));
            fetchModels(manifest, nullptr);
        });
        m_installing.insert(id, i18n("building the environment of %1…", manifest.name()));
        Q_EMIT installProgress(id, m_installing.value(id));
        env->checkVenvConcurrently(true);
        return;
    }
    fetchModels(manifest, nullptr);
}

QString PluginManager::runEffectJob(const QString &pluginId, const ObjectId &owner, int effectItemId, const QString &resultParam, const QJsonObject &input)
{
    const QString key = effectJobKey(owner, effectItemId);
    // The id a render answers to, here and in the chat: it names the effect, so
    // asking about it again while it runs asks about the same job.
    const QString jobId = QStringLiteral("render:") + key;
    if (m_effectJobs.contains(key)) {
        return jobId;
    }
    const QString blocker = runBlocker(pluginId);
    if (!blocker.isEmpty()) {
        pCore->displayMessage(blocker, ErrorMessage);
        // Refused before it began, and said why — the same answer a run that
        // fails half way gives, so a caller only has to read one thing.
        noteJobStarted(jobId, m_plugins.value(pluginId).name());
        noteJobEnded(jobId, true, blocker);
        return jobId;
    }
    const QueuedJob job{pluginId, owner, effectItemId, resultParam, input};
    if (m_busyPlugins.contains(pluginId)) {
        // These models hold gigabytes on the GPU; a second one started next to
        // the first does not go faster, it goes out of memory. Wait instead.
        m_effectJobQueue.append(job);
        m_effectJobs.insert(key, JobQueued);
        Q_EMIT effectJobProgressChanged(owner, effectItemId, JobQueued);
        noteJobQueued(jobId, i18n("%1 — waiting for the plugin to be free", m_plugins.value(pluginId).name()));
        return jobId;
    }
    startEffectJob(job);
    return jobId;
}

void PluginManager::startEffectJob(const QueuedJob &job)
{
    const QString key = effectJobKey(job.owner, job.effectItemId);
    const QString name = m_plugins.value(job.pluginId).name();
    // Results are filed under the effect that made them, not the plugin: one
    // plugin can carry three ways of working on a face, and their outputs have
    // no business sharing a folder.
    QString label = name;
    const QString effectId = job.input.value(QStringLiteral("effect_id")).toString();
    const QList<PluginEffect> effects = m_plugins.value(job.pluginId).effects();
    for (const PluginEffect &effect : effects) {
        if (effect.id == effectId && !effect.name.isEmpty()) {
            label = effect.name;
            break;
        }
    }
    m_busyPlugins.insert(job.pluginId);
    m_effectJobs.insert(key, 0);
    Q_EMIT effectJobProgressChanged(job.owner, job.effectItemId, 0);
    pCore->displayMessage(i18n("%1 is rendering…", name), InformationMessage);
    // A render is the longest thing a plugin does and the one an assistant asks
    // for and then walks away from: `generate_effect` returns at once and the
    // clip appears minutes later. The card is the editor's, so it ends when the
    // render does — with the reason when there is one.
    const QString card = QStringLiteral("render:") + key;
    noteJobStarted(card, label);
    runPluginJob(
        job.pluginId, job.input, this,
        [this, key, job, card](int progress) {
            m_effectJobs.insert(key, progress);
            noteJobProgress(card, progress);
            Q_EMIT effectJobProgressChanged(job.owner, job.effectItemId, progress);
        },
        [this, key, job, name, label, card](const QJsonObject &result, const QString &error) {
            m_effectJobs.remove(key);
            m_busyPlugins.remove(job.pluginId);
            Q_EMIT effectJobProgressChanged(job.owner, job.effectItemId, JobNone);
            // whatever happened here, the next one in line may go now
            for (int i = 0; i < m_effectJobQueue.size(); ++i) {
                if (m_effectJobQueue.at(i).pluginId == job.pluginId) {
                    const QueuedJob next = m_effectJobQueue.takeAt(i);
                    m_effectJobs.remove(effectJobKey(next.owner, next.effectItemId));
                    startEffectJob(next);
                    break;
                }
            }
            if (!error.isEmpty()) {
                pCore->displayMessage(error, ErrorMessage);
                noteJobEnded(card, true, error);
                return;
            }
            const QJsonArray outputs = result.value(QStringLiteral("outputs")).toArray();
            const QJsonObject first = outputs.isEmpty() ? QJsonObject() : outputs.first().toObject();
            const QString produced = first.value(QStringLiteral("path")).toString();
            if (produced.isEmpty() || !QFile::exists(produced)) {
                const QString message = i18n("%1 produced nothing.", name);
                pCore->displayMessage(message, ErrorMessage);
                noteJobEnded(card, true, message);
                return;
            }
            // Not every plugin answers with media. One that reports where a clip
            // should be cut hands back a list of timings, and a list of timings
            // is not a clip: importing it made a bin folder holding a file the
            // bin can neither play nor show. The effect still gets told it has a
            // result, so its button knows the analysis has been done.
            const QString place = result.value(QStringLiteral("place")).toString(QStringLiteral("bin"));
            const QString kind = first.value(QStringLiteral("type")).toString();
            if (place == QLatin1String("none") || kind == QLatin1String("data")) {
                if (!job.resultParam.isEmpty()) {
                    std::shared_ptr<EffectStackModel> stack = pCore->getItemEffectStack(job.owner.uuid, int(job.owner.type), job.owner.itemId);
                    if (stack) {
                        for (int i = 0; i < stack->rowCount(); ++i) {
                            auto item = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
                            if (item && item->getId() == job.effectItemId) {
                                item->setParameter(job.resultParam, produced, true);
                                break;
                            }
                        }
                    }
                }
                const QString note = result.value(QStringLiteral("message")).toString();
                const QString message = note.isEmpty() ? i18n("%1 finished.", name) : note;
                pCore->displayMessage(message, OperationCompletedMessage);
                noteJobEnded(card, false, message);
                return;
            }
            // Out of the job's scratch folder and into the project's own, under a
            // name of its own: rendering the same effect twice must not overwrite
            // what the bin already points at.
            QString destination = produced;
            if (auto *doc = pCore->currentDoc()) {
                const QString folder = doc->projectDataFolder() + QStringLiteral("/plugin-results");
                if (QDir().mkpath(folder)) {
                    // The name is what the user reads in the bin, so it says which
                    // effect made this and when: a row of identical UUIDs answers
                    // neither, and the newest one cannot be told from last week's.
                    const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH-mm-ss"));
                    QString base = QStringLiteral("%1 %2").arg(label, stamp);
                    base.replace(QRegularExpression(QStringLiteral("[/\\\\:*?\"<>|]")), QStringLiteral("-"));
                    destination = QStringLiteral("%1/%2.%3").arg(folder, base, QFileInfo(produced).suffix());
                    if (QFile::exists(destination)) {
                        destination = QStringLiteral("%1/%2 %3.%4")
                                          .arg(folder, base, QUuid::createUuid().toString(QUuid::WithoutBraces).left(8),
                                               QFileInfo(produced).suffix());
                    }
                    if (!QFile::rename(produced, destination) && !QFile::copy(produced, destination)) {
                        destination = produced;
                    }
                }
            }
            if (!job.resultParam.isEmpty()) {
                // tell the effect it has a result, wherever its widget is now
                std::shared_ptr<EffectStackModel> stack = pCore->getItemEffectStack(job.owner.uuid, int(job.owner.type), job.owner.itemId);
                if (stack) {
                    for (int i = 0; i < stack->rowCount(); ++i) {
                        auto item = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
                        if (item && item->getId() == job.effectItemId) {
                            item->setParameter(job.resultParam, destination, true);
                            break;
                        }
                    }
                }
            }
            // The bin clip is built here rather than through addProjectClip so
            // that its id comes back and so that there is something to wait on:
            // the producer loads in the background, and a clip with no duration
            // yet cannot be put on a track.
            const QString folderId = effectResultsFolder(label);
            const ObjectId owner = job.owner;
            QMetaObject::invokeMethod(
                pCore->window(),
                [destination, folderId, owner, name, label]() {
                    const QStringList existing = pCore->projectItemModel()->getClipByUrl(QFileInfo(destination));
                    if (!existing.isEmpty()) {
                        placeResultOnTimeline(owner, existing.constFirst());
                        return;
                    }
                    Fun undo = []() { return true; };
                    Fun redo = []() { return true; };
                    ClipCreator::createClipFromFile(
                        destination, folderId, pCore->projectItemModel(), undo, redo,
                        [owner](const QString &binId) { placeResultOnTimeline(owner, binId); });
                },
                Qt::QueuedConnection);
            const QString done = i18n("%1 finished — on a new track, and in the '%2' bin folder", name, label);
            pCore->displayMessage(done, OperationCompletedMessage);
            noteJobEnded(card, false, done);
        });
}

QString PluginManager::runPlugin(const QString &id, const QJsonObject &input, QWidget *messageParent)
{
    Q_UNUSED(messageParent)
    const auto it = m_plugins.constFind(id);
    if (it == m_plugins.constEnd()) {
        return {};
    }
    const QString name = it.value().name();
    pCore->displayMessage(i18n("Running %1…", name), InformationMessage);
    const QString card = QStringLiteral("plugin:") + QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    noteJobStarted(card, name);
    // Recording a set is not the same job as producing media, and what comes
    // back is a preset rather than a clip. Until now only the presets panel
    // filed one, so a set recorded any other way — by the assistant, say — was
    // written into a working directory and forgotten, and the effect that asked
    // for it still had nothing to choose.
    const QString action = input.value(QStringLiteral("action")).toString();
    const QString kind = input.value(QStringLiteral("kind")).toString();
    const QString source = input.value(QStringLiteral("source")).toString();
    runPluginJob(
        id, input, this, [this, card](int percent) { noteJobProgress(card, percent); },
        [this, id, name, action, kind, source, card](const QJsonObject &result, const QString &error) {
        if (!error.isEmpty()) {
            pCore->displayMessage(error, ErrorMessage);
            noteJobEnded(card, true, error);
            return;
        }
        if (action == QLatin1String("analyse")) {
            const QJsonArray outputs = result.value(QStringLiteral("outputs")).toArray();
            const QString file = outputs.isEmpty() ? QString() : outputs.first().toObject().value(QStringLiteral("path")).toString();
            if (file.isEmpty()) {
                const QString message = i18n("Nothing came back for %1.", QFileInfo(source).fileName());
                pCore->displayMessage(message, ErrorMessage);
                noteJobEnded(card, true, message);
                return;
            }
            QString storeError;
            const PluginSets::Set stored = PluginSets::store(id, QFileInfo(source).completeBaseName(), kind, file, &storeError);
            if (!stored.isValid()) {
                const QString message = storeError.isEmpty() ? i18n("The set could not be saved.") : storeError;
                pCore->displayMessage(message, ErrorMessage);
                noteJobEnded(card, true, message);
                return;
            }
            pCore->displayMessage(i18n("%1 recorded '%2'.", name, stored.name), InformationMessage);
            noteJobEnded(card, false, i18n("recorded '%1'", stored.name));
            return;
        }
        // Import produced media into the bin. "timeline"/"replace-zone" fall back
        // to the bin for now — the AI then places clips via the timeline tools.
        QStringList producedFiles;
        const QString place = result.value(QStringLiteral("place")).toString(QStringLiteral("bin"));
        if (place != QLatin1String("none")) {
            const QJsonArray outputs = result.value(QStringLiteral("outputs")).toArray();
            for (const QJsonValue &output : outputs) {
                const QString path = output.toObject().value(QStringLiteral("path")).toString();
                if (!path.isEmpty() && QFile::exists(path)) {
                    producedFiles << path;
                    QMetaObject::invokeMethod(pCore->window(), "addProjectClip", Qt::QueuedConnection, Q_ARG(QString, path),
                                              Q_ARG(QString, QStringLiteral("-1")));
                }
            }
        }
        QString message = result.value(QStringLiteral("message")).toString();
        if (message.isEmpty()) {
            message = producedFiles.isEmpty() ? i18n("%1 finished.", name)
                                              : i18np("%2: %1 file added to the bin.", "%2: %1 files added to the bin.", producedFiles.size(), name);
        }
        pCore->displayMessage(message, OperationCompletedMessage);
        noteJobEnded(card, false, message);
        });
    return card;
}
