/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "definitions.h"
#include "pluginmanifest.h"

#include <QHash>
#include <QJsonObject>
#include <QObject>
#include <QSet>
#include <QSharedPointer>
#include <QTemporaryDir>

#include <functional>

class QProcess;
class QWidget;

/** @class PluginManager
    @brief Registry and runner for user-loadable Python plugins.

    Plugins are folders under `<AppLocalData>/plugins/<id>/` (writable, user
    imports) plus an optional read-only set shipped in the app data dir. This
    singleton scans them, imports new ones from a folder or a `.wunjoplugin`
    archive after validation, uninstalls them (removing the private venv too),
    and launches a plugin's entry script for a given clip/face context.

    Menu code queries @ref pluginsForTarget to build the entries and calls
    @ref runPlugin on trigger; the settings page drives @ref inspect /
    @ref install / @ref uninstall.
 */
class PluginManager : public QObject
{
    Q_OBJECT
public:
    static PluginManager &instance();

    /** @brief A plugin previewed for import: its manifest plus the on-disk
     *  location of its files (a temp dir for archives, kept alive by @p temp). */
    struct ImportCandidate {
        PluginManifest manifest;
        QSharedPointer<QTemporaryDir> temp;
        QString sourceDir;
        bool valid() const { return !sourceDir.isEmpty() && manifest.isValid(); }
    };

    /** @brief All installed, valid plugins (user copies shadow bundled ones). */
    QList<PluginManifest> installedPlugins() const;
    /** @brief Installed plugins whose target matches @p target. */
    QList<PluginManifest> pluginsForTarget(const QString &target) const;
    /** @brief Installed plugin by id, or an invalid manifest if unknown. */
    PluginManifest plugin(const QString &id) const;
    bool isBundled(const QString &id) const;

    /** @brief Absolute models dir for a plugin (created on demand). */
    QString modelsDir(const QString &id) const;
    /** @brief Where @p model lives once installed: a file, or the folder an
     *  archived weight was unpacked into. */
    QString modelPath(const QString &id, const PluginModel &model) const;
    /** @brief Where the download of @p model lands. For a plain weight that is
     *  where it stays; an archive comes down under a working name beside the
     *  folder it is unpacked into and is deleted afterwards — what the plugin
     *  uses is the tree that came out of it. Both the settings page and the
     *  unattended install ask here, so they cannot disagree about the path and
     *  leave each other's leftovers on disk. */
    QString downloadTarget(const QString &id, const PluginModel &model) const;
    /** @brief The weights of @p manifest that this machine needs — the right
     *  quantisation for its video memory, the runtime build for its GPU. Every
     *  place that lists, checks or downloads weights goes through this, so the
     *  settings page and the run blocker never disagree about what is missing. */
    static QList<PluginModel> applicableModels(const PluginManifest &manifest);
    /** @brief Unpack a downloaded archive weight into its folder and restore the
     *  executable bit. Returns false with a user-readable reason in @p errorOut. */
    bool unpackModel(const QString &id, const PluginModel &model, const QString &archivePath, QString *errorOut = nullptr);

    /** @brief What a declared model weight looks like on disk. */
    enum ModelState {
        ModelMissing,    ///< never downloaded
        ModelIncomplete, ///< a download that was cut short — the file is there but unusable
        ModelReady
    };
    /** @brief State of @p model of plugin @p id. A file that exists is not a
     *  model that works: an interrupted download leaves a fraction of it
     *  behind, and the plugin then fails deep inside its own code with
     *  something unreadable. Judged by the declared size (they are estimates,
     *  so only a fraction of it counts as truncated) and, when the manifest
     *  gives one, by @p verifyChecksum — reading hundreds of megabytes is worth
     *  it right after a download, not on every glance at the settings. */
    ModelState modelState(const QString &id, const PluginModel &model, bool verifyChecksum = false) const;

    /** @brief The writable folder the effects repository reads custom effect
     *  XMLs from — where the effects of installed plugins are materialized. */
    static QString effectsDir();
    /** @brief Id of the plugin that brought effect @p effectId, or empty when
     *  no installed plugin claims it. */
    QString pluginForEffect(const QString &effectId) const;
    bool ownsEffect(const QString &effectId) const { return !pluginForEffect(effectId).isEmpty(); }
    /** @brief Write the effects of every installed plugin into @ref effectsDir
     *  and delete the ones left behind by plugins that are gone, announcing the
     *  difference with @ref pluginEffectsChanged. Runs once before the effects
     *  repository scans the folder, then after every import and uninstall. */
    void syncEffects();

    /** @brief Stored API key for @p provider, or empty. Kept in a private
     *  config file, never handed to the assistant — only booleans are. */
    QString apiKey(const QString &provider) const;
    void setApiKey(const QString &provider, const QString &key);

    /** @brief Read a folder or `.wunjoplugin` archive without installing it.
     *  The returned candidate carries validation errors for the UI. */
    ImportCandidate inspect(const QString &path) const;
    /** @brief Copy a previewed plugin into the user plugins dir. */
    bool install(const ImportCandidate &candidate, QString *errorOut = nullptr);
    /** @brief Remove a user plugin and its private venv. */
    bool uninstall(const QString &id, QString *errorOut = nullptr);

    /** @brief Run @p id for a job whose "input" object is @p input, reporting
     *  the outcome to the user. Non-blocking; returns the job's id, which
     *  @ref jobOutcome answers about, or empty when it could not be started. */
    QString runPlugin(const QString &id, const QJsonObject &input, QWidget *messageParent = nullptr);
    /** @brief What became of the job @p jobId: `{"state": "running"|"done"|
     *  "failed", "percent": n, "message": "…"}`, or an empty object when that
     *  id means nothing here.
     *
     *  A plugin run says everything worth knowing in one sentence — "choose an
     *  audio preset first", "3 files added to the bin" — and until now that
     *  sentence only ever reached a message banner. Whoever started the job,
     *  the assistant above all, could not tell a refusal from a success, and
     *  answered for work that had never happened. */
    QJsonObject jobOutcome(const QString &jobId) const;
    /** @brief Render what an effect describes, for the plugin that brought it.
     *
     *  The job belongs to the manager, never to the widget that started it: the
     *  effect stack is rebuilt on every seek and reselection, and a render that
     *  died with its button would be a render nobody can see, stop or trust.
     *  Whoever draws that button asks @ref effectJobProgress what is going on
     *  and follows @ref effectJobProgressChanged.
     *  The produced file is filed under the project, written into @p resultParam
     *  of the effect and added to the bin. */
    QString runEffectJob(const QString &pluginId, const ObjectId &owner, int effectItemId, const QString &resultParam, const QJsonObject &input);
    /** @brief Progress of that job in percent, @ref JobQueued while it waits for
     *  the plugin to be free, or @ref JobNone when there is nothing. */
    int effectJobProgress(const ObjectId &owner, int effectItemId) const;
    static constexpr int JobNone = -1;
    static constexpr int JobQueued = -2;
    /** @brief Why @p id cannot run right now (a model missing or half
     *  downloaded), or empty when nothing is in the way. */
    QString runBlocker(const QString &id) const;

    /** @brief Bytes still to be fetched before @p manifest can work: the
     *  weights this machine needs and does not have, plus a reserve for the
     *  environment when one still has to be built. */
    /** @brief The venv directory of @p venvName, built or not. */
    static QString venvDir(const QString &venvName);
    /** @brief The interpreter inside @p venvName, or empty when that
     *  environment has not been built.
     *
     *  A venv puts it in `bin/python3` on Unix and `Scripts/python.exe` on
     *  Windows. Four places used to spell that out, three of them Unix-only and
     *  one of them not — which is how "the plugin has no environment" would
     *  have read on Windows for environments that were there. Both layouts are
     *  checked here and nowhere else. */
    static QString venvPython(const QString &venvName);

    static qint64 pendingDownloadBytes(const PluginManifest &manifest);
    /** @brief Why @p bytes cannot be fetched right now — no connection, or not
     *  enough room where plugin data lives — or empty when they can. */
    static QString downloadBlocker(qint64 bytes);
    /** @brief Why an install of @p manifest must not start — the wrong
     *  application version, no network, or not enough room on the disk — or
     *  empty when it may go ahead.
     *
     *  All three fail the same way without this: minutes of downloading that
     *  end in a Python traceback, a truncated weight, or a full disk that takes
     *  the user's projects down with it. Checked once, before anything is
     *  written. */
    QString installBlocker(const PluginManifest &manifest) const;
    /** @brief Make @p id fit to run: build its environment and fetch the weights
     *  it declares but does not have.
     *
     *  A plugin arrives as a manifest and a script; what it needs to actually
     *  work is a few gigabytes that only make sense to fetch once somebody wants
     *  it. Until now that was the settings page's business, which left an
     *  assistant able to see a plugin, know it was not ready, and do nothing
     *  about it. Non-blocking: follow it with @ref installMessage, or simply ask
     *  the plugin's status again until nothing is missing.
     */
    void installPlugin(const QString &id);
    /** @brief What the install of @p id is doing, or empty when it is not
     *  running. Reads as a sentence, because it is shown to the user. */
    QString installMessage(const QString &id) const;

Q_SIGNALS:
    /** @brief An install moved on or ended; @p message is empty when it ended. */
    void installProgress(const QString &id, const QString &message);

public:
    /** @brief CUDA version the installed NVIDIA driver supports (12.6, 11.8…),
     *  or 0 when there is no usable GPU. Asked once per session: it decides
     *  which wheels a plugin's environment gets, and a wheel built for a newer
     *  CUDA than the driver simply refuses to start and falls back to the CPU
     *  without saying so. */
    static double driverCudaVersion();
    /** @brief Video memory of the first GPU in GB, or 0 when there is none.
     *  Decides which variant of a plugin's weights is offered: a model ships in
     *  two quantisations and only one of them fits this machine. */
    static double gpuVramGb();
    /** @brief Which llama.cpp-style runtime build this machine wants:
     *  "cuda" when the driver is new enough, "vulkan" for any other GPU,
     *  "cpu" when there is none. */
    static QString gpuBackend();

    /** @brief Same, but for callers that follow the job themselves: @p onProgress
     *  gets every `progress:` the plugin reports and @p onFinished its result
     *  object (paths made absolute) or a ready-to-show error. The callbacks
     *  belong to @p context and are dropped with it.
     *  @return A handle for @ref cancelPluginJob, or empty if it never started. */
    QString runPluginJob(const QString &id, const QJsonObject &input, QObject *context, const std::function<void(int)> &onProgress,
                         const std::function<void(const QJsonObject &result, const QString &error)> &onFinished);
    /** @brief Stop the job @p handle names. False when it has already ended. */
    bool cancelPluginJob(const QString &handle);
    /** @brief The MCP server shipped with the app (`share/wunjo/mcp`), or empty
     *  when this build has none. Both the built-in assistant and an outside
     *  agent are pointed at this one copy. */
    static QString mcpServerDir();
    /** @brief The interpreter plugin @p id runs under — its own environment
     *  when it has one, the system Python otherwise. Empty when there is none. */
    QString pythonFor(const QString &id) const;

    QString userPluginsDir() const;

Q_SIGNALS:
    void pluginsChanged();
    /** @brief Plugin effects appeared (@p addedFiles, XMLs in @ref effectsDir)
     *  or went away (@p removedIds) and the effect list should follow. */
    void pluginEffectsChanged(const QStringList &addedFiles, const QStringList &removedIds);
    /** @brief A render started by an effect moved on (percent) or ended (-1). */
    void effectJobProgressChanged(const ObjectId &owner, int effectItemId, int progress);

private:
    PluginManager();
    void rescan();
    /** @brief Interpreter to launch @p m with, or empty if none is available. */
    QString interpreterFor(const PluginManifest &m) const;
    /** @brief Write the job file and launch the plugin, or return nullptr with
     *  a user-readable reason in @p errorOut. */
    QProcess *startProcess(const PluginManifest &manifest, const QJsonObject &input, QString *workDirOut, QString *errorOut);
    /** @brief The plugin's effect XML with an ownership stamp added, or empty
     *  if it cannot be read. */
    static QByteArray stampedEffect(const PluginManifest &manifest, const PluginEffect &effect);
    /** @brief Read the ownership stamp of an installed effect file, filling
     *  @p effectId. Empty for an effect the user made themselves — those live
     *  in the same folder and must never be touched. */
    static QString effectStamp(const QString &path, QString *effectId);

    QHash<QString, PluginManifest> m_plugins;
    QSet<QString> m_bundledIds;
    /** @brief A render waiting for its plugin to be free. */
    struct QueuedJob {
        QString pluginId;
        ObjectId owner;
        int effectItemId;
        QString resultParam;
        QJsonObject input;
    };
    /** @brief Start @p job now and follow it through. */
    void startEffectJob(const QueuedJob &job);

    /** @brief Fetch the weights of @p manifest that are not here yet, one after
     *  another; @p onDone runs when the last of them has landed. */
    void fetchModels(const PluginManifest &manifest, const std::function<void()> &onDone);
    /** @brief What each running install is doing, by plugin id. */
    QHash<QString, QString> m_installing;
    /** @brief Open the chat's card for @p jobId and start remembering what
     *  becomes of it; @ref noteJobProgress and @ref noteJobEnded carry it on.
     *  The card and the outcome are the same story told to two audiences — the
     *  person watching the panel and whoever asked for the job — so they are
     *  written in one place and can never disagree. */
    void noteJobStarted(const QString &jobId, const QString &name);
    /** @brief A render that has to wait for the plugin to be free: known and
     *  running to anyone asking, but without a card until it truly starts. */
    void noteJobQueued(const QString &jobId, const QString &name);
    void noteJobProgress(const QString &jobId, int percent);
    void noteJobEnded(const QString &jobId, bool isError, const QString &message);
    void recordJob(const QString &jobId, const QString &state, int percent, const QString &message);
    /** @brief What each job the editor started is doing or ended as, by job id.
     *  Trimmed to the last @ref kRememberedJobs so a long session does not carry
     *  every run it ever made. */
    QHash<QString, QJsonObject> m_jobOutcomes;
    QStringList m_jobOrder;
    static constexpr int kRememberedJobs = 64;

    /** @brief "<plugin id>/<model name>" of the weights this install has given
     *  up on. @ref fetchModels walks the list of what is missing from the top
     *  every time one download ends, so without this the one that failed is
     *  simply the one it picks again — forever. */
    QSet<QString> m_failedDownloads;

    QHash<QString, QString> m_effectOwners;
    /** @brief Jobs a caller may still call off, by handle. */
    QHash<QString, QProcess *> m_runningJobs;
    /** @brief Renders in flight or waiting: "owner/effect" -> percent. */
    QHash<QString, int> m_effectJobs;
    /** @brief One heavy model at a time per plugin: two of them on the same GPU
     *  only take each other's memory. */
    QSet<QString> m_busyPlugins;
    QList<QueuedJob> m_effectJobQueue;
};
