/*
 *    SPDX-FileCopyrightText: 2024 Jean-Baptiste Mardelle <jb@kdenlive.org>
 *
 * SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
 */

#include "pluginssettings.h"
#include "core.h"
#include "wunjosettings.h"
#include "mainwindow.h"
#include "plugins/pluginmanager.h"
#include "plugins/pluginmanagerwidget.h"
#include "plugins/pluginsettingstab.h"
#include "pythoninterfaces/dialogs/modeldownloadwidget.h"
#include "pythoninterfaces/saminterface.h"
#include "pythoninterfaces/speechtotextvosk.h"
#include "pythoninterfaces/speechtotextwhisper.h"

#include <KArchive>
#include <KArchiveDirectory>
#include "plugins/filedownloadjob.h"

#include <KIO/JobTracker>
#include <KJobTrackerInterface>
#include <KLineEdit>
#include <KMessageBox>
#include <KTar>
#include <KUrlRequesterDialog>
#include <KZip>

#include <QButtonGroup>
#include <QScrollArea>
#include <QMimeData>
#include <QMimeDatabase>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

SpeechList::SpeechList(QWidget *parent)
    : QListWidget(parent)
{
    setAlternatingRowColors(true);
    setAcceptDrops(true);
    setDropIndicatorShown(true);
    viewport()->setAcceptDrops(true);
}

QStringList SpeechList::mimeTypes() const
{
    return QStringList() << QStringLiteral("text/uri-list");
}

void SpeechList::dropEvent(QDropEvent *event)
{
    const QMimeData *qMimeData = event->mimeData();
    if (qMimeData->hasUrls()) {
        QList<QUrl> urls = qMimeData->urls();
        if (!urls.isEmpty()) {
            Q_EMIT getDictionary(urls.takeFirst());
        }
    }
}

PluginsSettings::PluginsSettings(QWidget *parent)
    : QWidget(parent)
{
    setupUi(this);
    m_sttVosk = new SpeechToTextVosk(this);
    m_sttWhisper = new SpeechToTextWhisper(this);
    m_samInterface = new SamInterface(this);
    // Contextual help
    whisper_device_info->setContextualHelpText(i18n("CPU processing is very slow, GPU is recommended for faster processing"));
    seamless_device_info->setContextualHelpText(i18n("Text translation is performed by the SeamlessM4T model. This requires downloading "
                                                     "around 10Gb of data. Once installed, all processing will happen offline."));

    noModelMessage->hide();
    // What to recommend depends on what the machine can run: turbo is the one
    // worth having on a card, and on a processor it transcribes slower than the
    // recording plays — there "small" is the largest that stays usable.
    const bool gpuForSpeech = WunjoSettings::whisperDevice().startsWith(QLatin1String("cuda")) || PluginManager::driverCudaVersion() > 0;
    const QString recommendedModel = gpuForSpeech ? QStringLiteral("turbo") : QStringLiteral("small");
    m_downloadModelAction = new QAction(gpuForSpeech ? i18n("Download (1.4Gb)") : i18n("Download (0.5Gb)"), this);
    connect(m_downloadModelAction, &QAction::triggered, this, [this, recommendedModel]() {
        disconnect(m_sttWhisper, &SpeechToText::installFeedback, this, &PluginsSettings::showSpeechLog);
        if (m_sttWhisper->installNewModel(recommendedModel)) {
            reloadWhisperModels();
        }
        connect(m_sttWhisper, &SpeechToText::installFeedback, this, &PluginsSettings::showSpeechLog, Qt::QueuedConnection);
    });
    m_recommendedModel = recommendedModel;
    // Fill models list
    reloadWhisperModels();
    const QString seamlessModelScript = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("scripts/whisper/seamlessquery.py"));
    ModelDownloadWidget *modelDownload = new ModelDownloadWidget(m_sttWhisper, seamlessModelScript, {QStringLiteral("task=download")}, this);
    connect(modelDownload, &ModelDownloadWidget::installFeedback, this, &PluginsSettings::showSpeechLog, Qt::QueuedConnection);
    connect(modelDownload, &ModelDownloadWidget::jobDone, this, &PluginsSettings::downloadJobDone, Qt::QueuedConnection);
    seamless_layout->addWidget(modelDownload);
    modelDownload->setVisible(false);
    connect(install_seamless, &QPushButton::clicked, this, [this]() {
        const QString scriptPath = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("scripts/whisper/requirements-seamless.txt"));
        if (!scriptPath.isEmpty()) {
            m_sttWhisper->addDependency(scriptPath, QString());
        }
        // The prompt used to live inside installMissingDependencies; it was
        // pulled out so it could be shown before the slow part rather than
        // after it, which means every caller now asks for itself.
        if (!m_sttWhisper->confirmInstall()) {
            return;
        }
        install_seamless->setEnabled(false);
        install_seamless->setText(i18n("Installing multilingual data…"));
        m_sttWhisper->installMissingDependencies();
    });

    QString voskModelFolder = WunjoSettings::vosk_folder_path();
    if (voskModelFolder.isEmpty()) {
        voskModelFolder = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("speechmodels"), QStandardPaths::LocateDirectory);
    }
    if (!voskModelFolder.isEmpty()) {
        modelV_folder_label->setLink(voskModelFolder);
        modelV_folder_label->setVisible(true);
        QFuture<KIO::filesize_t> future = QtConcurrent::run(&MainWindow::fetchFolderSize, pCore->window(), voskModelFolder);
        QFutureWatcher<KIO::filesize_t> *watcher = new QFutureWatcher<KIO::filesize_t>(this);
        watcher->setFuture(future);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            KIO::filesize_t size = watcher->result();
            modelV_size->setText(KIO::convertSize(size));
            watcher->deleteLater();
        });
    } else {
        modelV_folder_label->setVisible(false);
    }
    QButtonGroup *speechEngineSelection = new QButtonGroup(this);
    speechEngineSelection->addButton(engine_vosk);
    speechEngineSelection->addButton(engine_whisper);
    connect(speechEngineSelection, &QButtonGroup::buttonClicked, this, [this](QAbstractButton *button) {
        if (button == engine_vosk) {
            speech_stack->setCurrentIndex(0);
            m_sttVosk->checkDependencies(false);
        } else if (button == engine_whisper) {
            speech_stack->setCurrentIndex(1);
            m_sttWhisper->checkDependencies(false);
        }
    });
    if (WunjoSettings::speechEngine() == QLatin1String("whisper")) {
        engine_whisper->setChecked(true);
        speech_stack->setCurrentIndex(1);
    } else {
        engine_vosk->setChecked(true);
        speech_stack->setCurrentIndex(0);
    }
    // Python setup
    connect(m_sttWhisper, &SpeechToText::gotPythonSize, this, [this](const QString &label) {
        wr_venv_size->setText(label);
        deleteWrVenv->setEnabled(!label.isEmpty());
    });
    m_sttWhisper->checkVenv(true);

    // Whisper
    if (WunjoSettings::whisperInstalledModels().isEmpty()) {
        combo_wr_model->setPlaceholderText(i18n("Probing..."));
    }

    combo_wr_device->setPlaceholderText(i18n("Probing..."));
    combo_wr_model->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    combo_wr_device->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    combo_sam_device->setPlaceholderText(i18n("Probing..."));
    combo_sam_device->setSizeAdjustPolicy(QComboBox::AdjustToContents);

    m_msgWhisper = new PythonDependencyMessage(this, m_sttWhisper);
    message_layout_wr->addWidget(m_msgWhisper);
    QMap<QString, QString> whisperLanguages = m_sttWhisper->speechLanguages();
    QMapIterator<QString, QString> j(whisperLanguages);
    while (j.hasNext()) {
        j.next();
        combo_wr_lang->addItem(j.key(), j.value());
    }
    int ix = combo_wr_lang->findData(WunjoSettings::whisperLanguage());
    if (ix > -1) {
        combo_wr_lang->setCurrentIndex(ix);
    }
    script_log->hide();
    script_log->setCenterOnScroll(true);
    connect(m_sttWhisper, &SpeechToText::scriptStarted, this, [this]() { QMetaObject::invokeMethod(script_log, "clear"); });
    connect(m_sttWhisper, &SpeechToText::installFeedback, this, &PluginsSettings::showSpeechLog, Qt::QueuedConnection);
    connect(m_sttWhisper, &SpeechToText::scriptFinished, this, [this, modelDownload](const QStringList &args) {
        if (args.join(QLatin1Char(' ')).contains("requirements-seamless.txt")) {
            install_seamless->setText(i18n("Downloading multilingual model…"));
            modelDownload->setVisible(true);
            modelDownload->startDownload();
        }
        QMetaObject::invokeMethod(m_msgWhisper, "checkAfterInstall", Qt::QueuedConnection);
    });
    connect(downloadButton, &QPushButton::clicked, this, [this]() {
        disconnect(m_sttWhisper, &SpeechToText::installFeedback, this, &PluginsSettings::showSpeechLog);
        if (m_sttWhisper->installNewModel()) {
            reloadWhisperModels();
        }
        connect(m_sttWhisper, &SpeechToText::installFeedback, this, &PluginsSettings::showSpeechLog, Qt::QueuedConnection);
    });

    connect(m_sttWhisper, &SpeechToText::scriptFeedback, this, &PluginsSettings::gotWhisperFeedback);
    connect(deleteWrVenv, &QPushButton::clicked, this, &PluginsSettings::doDeleteWrVenv);
    connect(m_sttWhisper, &SpeechToText::concurrentScriptFinished, this, &PluginsSettings::whisperFinished);
    connect(m_sttWhisper, &SpeechToText::dependenciesAvailable, this, &PluginsSettings::whisperAvailable);
    connect(m_sttWhisper, &SpeechToText::dependenciesMissing, this, &PluginsSettings::whisperMissing);

    // VOSK
    vosk_folder->setPlaceholderText(QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("speechmodels"), QStandardPaths::LocateDirectory));
    m_msgVosk = new PythonDependencyMessage(this, m_sttVosk);
    message_layout->addWidget(m_msgVosk);

    connect(m_sttVosk, &SpeechToText::dependenciesAvailable, this, [&]() {
        if (m_speechListWidget->count() == 0) {
            doShowSpeechMessage(i18n("Please add a speech model."), KMessageWidget::Information);
        }
    });
    connect(m_sttVosk, &SpeechToText::dependenciesMissing, this, [&](const QStringList &) { speech_info->animatedHide(); });
    connect(m_sttVosk, &SpeechToText::scriptStarted, this, [this]() { QMetaObject::invokeMethod(script_log, "clear"); });
    connect(m_sttVosk, &SpeechToText::installFeedback, this, &PluginsSettings::showSpeechLog, Qt::QueuedConnection);
    connect(m_sttVosk, &SpeechToText::scriptFinished, this, [this]() { QMetaObject::invokeMethod(m_msgVosk, "checkAfterInstall", Qt::QueuedConnection); });

    m_speechListWidget = new SpeechList(this);
    connect(m_speechListWidget, &SpeechList::getDictionary, this, &PluginsSettings::getDictionary);
    QVBoxLayout *l = new QVBoxLayout(list_frame);
    l->setContentsMargins(0, 0, 0, 0);
    l->addWidget(m_speechListWidget);
    speech_info->setWordWrap(true);
    connect(check_config, &QPushButton::clicked, this, &PluginsSettings::slotCheckSttConfig);

    connect(custom_vosk_folder, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState state) {
        vosk_folder->setEnabled(state != Qt::Unchecked);
        if (state == Qt::Unchecked) {
            // Clear custom folder
            vosk_folder->clear();
            WunjoSettings::setVosk_folder_path(QString());
            slotParseVoskDictionaries();
        }
    });
    connect(vosk_folder, &KUrlRequester::urlSelected, this, [this](const QUrl &url) {
        WunjoSettings::setVosk_folder_path(url.toLocalFile());
        slotParseVoskDictionaries();
    });
    models_url->setText(i18n("Download speech models from: <a href=\"https://alphacephei.com/vosk/models\">https://alphacephei.com/vosk/models</a>"));
    connect(models_url, &QLabel::linkActivated, this, &PluginsSettings::openBrowserUrl);
    connect(button_add, &QToolButton::clicked, this, [this]() { this->getDictionary(); });
    connect(button_delete, &QToolButton::clicked, this, &PluginsSettings::removeDictionary);
    connect(this, &PluginsSettings::parseDictionaries, this, &PluginsSettings::slotParseVoskDictionaries);
    slotParseVoskDictionaries();

    speech_system_python_message->setText(i18n(
        "<b>Using system packages. Only use this if you know what that means</b>. You need to install all required packages by yourself on your system:<br>%1",
        m_sttWhisper->listDependencies().join(QLatin1Char(','))));
    if (WunjoSettings::speech_system_python_path().isEmpty() || WunjoSettings::sam_system_python_path().isEmpty()) {
#ifdef Q_OS_WIN
        const QString pythonName = QStringLiteral("python");
#else
        const QString pythonName = QStringLiteral("python3");
#endif
        const QString pythonExe = QStandardPaths::findExecutable(pythonName);
        if (!pythonExe.isEmpty()) {
            if (WunjoSettings::speech_system_python_path().isEmpty()) {
                WunjoSettings::setSpeech_system_python_path(pythonExe);
            }
            if (WunjoSettings::sam_system_python_path().isEmpty()) {
                WunjoSettings::setSam_system_python_path(pythonExe);
            }
        }
    }
    speech_system_python_path->setText(WunjoSettings::speech_system_python_path());
    sam_system_python_path->setText(WunjoSettings::sam_system_python_path());
    if (WunjoSettings::speech_system_python()) {
        // Using system packages only, disable all dependency checks
        whisper_venv_params->setEnabled(false);
        reloadWhisperModels();
        script_log->setVisible(false);
        m_msgWhisper->setVisible(false);
    } else {
        speech_system_params->setVisible(false);
        checkSpeechDependencies();
        m_msgWhisper->setVisible(true);
    }
    connect(kcfg_speech_system_python, &QCheckBox::toggled, this, [this](bool systemPackages) {
        m_msgWhisper->setVisible(systemPackages == false);
        WunjoSettings::setSpeech_system_python(systemPackages);
        speech_system_params->setVisible(systemPackages);
        whisper_venv_params->setEnabled(systemPackages == false);
        script_log->setVisible(systemPackages == false);
        WunjoSettings::setSpeech_system_python_path(speech_system_python_path->text());
        if (systemPackages) {
            whispersettings->setEnabled(true);
            // Force device reload to ensure torch is found
            m_sttWhisper->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
            reloadWhisperModels();
        } else {
            m_sttWhisper->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
            checkSamEnvironement(false);
        }
    });

    // Sam
    m_pythonSamLabel = new PythonDependencyMessage(this, m_samInterface, false);
    message_layout_sam->addWidget(m_pythonSamLabel);
    connect(m_samInterface, &AbstractPythonInterface::gotPythonSize, this, [this](const QString &label) {
        sam_venv_size->setText(label);
        deleteSamVenv->setEnabled(!label.isEmpty());
        sam_rebuild->setEnabled(!label.isEmpty());
    });
    m_samInterface->checkVenv(true);
    connect(m_samInterface, &AbstractPythonInterface::installFeedback, this, &PluginsSettings::showSamLog, Qt::QueuedConnection);
    connect(m_samInterface, &AbstractPythonInterface::scriptFinished, this,
            [this]() { QMetaObject::invokeMethod(m_pythonSamLabel, "checkAfterInstall", Qt::QueuedConnection); });
    combo_sam_model->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    connect(downloadSamButton, &QPushButton::clicked, this, &PluginsSettings::downloadSamModels);
    connect(deleteSamVenv, &QPushButton::clicked, this, &PluginsSettings::doDeleteSamVenv);
    connect(deleteSamModels, &QPushButton::clicked, this, &PluginsSettings::doDeleteSamModels);
    connect(m_samInterface, &AbstractPythonInterface::venvSetupChanged, this,
            [this]() { QMetaObject::invokeMethod(this, "checkSamEnvironement", Qt::QueuedConnection); });
    connect(m_samInterface, &AbstractPythonInterface::dependenciesAvailable, this, &PluginsSettings::samDependenciesChecked);

    connect(m_samInterface, &AbstractPythonInterface::dependenciesMissing, this, &PluginsSettings::samMissing);
    connect(m_samInterface, &AbstractPythonInterface::scriptFeedback, this, &PluginsSettings::gotSamFeedback);
    connect(m_samInterface, &AbstractPythonInterface::concurrentScriptFinished, this, &PluginsSettings::samFinished);
    connect(sam_rebuild, &QToolButton::clicked, this, [this]() {
        if (KMessageBox::warningContinueCancel(this, i18n("This will attempt to rebuild the plugin's virtual environment. Only use if the plugin fails. If "
                                                          "this does not work, try deleting and reinstalling the plugin.")) != KMessageBox::Continue) {
            return;
        }
        sam_rebuild->setEnabled(false);
        setCursor(Qt::WaitCursor);
        m_samInterface->rebuildVenv();
        setCursor(Qt::ArrowCursor);
        sam_rebuild->setEnabled(true);
    });

    if (modelBox->isEnabled()) {
        m_samInterface->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
    }
    system_python_message->setText(i18n(
        "<b>Using system packages. Only use this if you know what that means</b>. You need to install all required packages by yourself on your system:<br>%1",
        m_samInterface->listDependencies().join(QLatin1Char(','))));

    if (WunjoSettings::sam_system_python()) {
        // Using system packages only, disable all dependency checks
        sam_venv_params->setEnabled(false);
        script_sam_log->setVisible(false);
        m_pythonSamLabel->setVisible(false);
        reloadSamModels();
    } else {
        sam_system_params->setVisible(false);
        m_pythonSamLabel->setVisible(true);
        checkSamEnvironement(false);
    }
    connect(install_nvidia_wr, &QPushButton::clicked, this, [this]() { checkCuda(false); });
    connect(install_nvidia_sam, &QPushButton::clicked, this, [this]() { checkCuda(true); });
    connect(kcfg_sam_system_python, &QCheckBox::toggled, this, [this](bool systemPackages) {
        m_pythonSamLabel->setVisible(systemPackages == false);
        WunjoSettings::setSam_system_python(systemPackages);
        sam_system_params->setVisible(systemPackages);
        sam_venv_params->setEnabled(systemPackages == false);
        script_sam_log->setVisible(systemPackages == false);
        WunjoSettings::setSam_system_python_path(sam_system_python_path->text());
        if (systemPackages) {
            modelBox->setEnabled(true);
            // Force device reload to ensure torch is found
            m_samInterface->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
            reloadSamModels();
        } else {
            checkSamEnvironement(false);
        }
    });

    // [Wunjo] Plugin loading comes first: import .wmplugin archives or plugin
    // folders. Each installed plugin then gets its own tab (built below and
    // kept in sync), exactly like the built-in Speech / Object Detection tabs.
    tabWidget->insertTab(0, new PluginManagerWidget(this), i18n("Load Plugins"));
    tabWidget->setCurrentIndex(0);
    rebuildPluginTabs();
    connect(&PluginManager::instance(), &PluginManager::pluginsChanged, this, &PluginsSettings::rebuildPluginTabs);
}

void PluginsSettings::rebuildPluginTabs()
{
    // What has to be remembered is the page the tab bar holds — the scroll area
    // — and not the settings widget inside it. Looking the inner widget up here
    // answers -1, nothing is removed, and every rebuild leaves the old tab
    // beside the new one with its contents deleted out from under it.
    for (QWidget *page : std::as_const(m_pluginTabs)) {
        const int index = tabWidget->indexOf(page);
        if (index >= 0) {
            tabWidget->removeTab(index);
        }
        page->deleteLater();
    }
    m_pluginTabs.clear();
    const QList<PluginManifest> plugins = PluginManager::instance().installedPlugins();
    for (const PluginManifest &manifest : plugins) {
        auto *tab = new PluginSettingsTab(manifest, this);
        // A plugin page grows with what the plugin declares — environment, a
        // list of model downloads, then its settings. Without a scroll area the
        // last of those simply falls off the bottom of the dialog.
        auto *scroll = new QScrollArea(this);
        scroll->setWidget(tab);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        tabWidget->addTab(scroll, manifest.name());
        m_pluginTabs.append(scroll);
    }
}

PluginsSettings::~PluginsSettings()
{
    delete m_msgWhisper;
    delete m_msgVosk;
    delete m_pythonSamLabel;
}

void PluginsSettings::whisperAvailable()
{
    // Check if a GPU is available
    whispersettings->setEnabled(true);
    combo_wr_device->setPlaceholderText(i18n("Probing…"));
    m_sttWhisper->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
}

void PluginsSettings::whisperMissing()
{
    // Check if a GPU is available
    whispersettings->setEnabled(false);
    // Nothing has been installed yet, so there is no torch to ask which devices
    // exist. Saying "Probing…" for ever reads as a hang; say what is true.
    combo_wr_device->setPlaceholderText(i18n("Install to detect"));
}

void PluginsSettings::whisperFinished(const QString &scriptName, const QStringList &args)
{
    qDebug() << "=========================\n\nCONCURRENT JOB FINISHED: " << scriptName << " / " << args << "\n\n================";
    if (scriptName.contains("checkgpu")) {
        if (!WunjoSettings::whisperDevice().isEmpty()) {
            int ix = combo_wr_device->findData(WunjoSettings::whisperDevice());
            if (ix > -1) {
                combo_wr_device->setCurrentIndex(ix);
            }
        } else if (combo_wr_device->count() > 0) {
            combo_wr_device->setCurrentIndex(0);
            WunjoSettings::setWhisperDevice(combo_wr_device->currentData().toString());
        }
    }
}

void PluginsSettings::gotWhisperFeedback(const QString &scriptName, const QStringList args, const QStringList jobData)
{
    Q_UNUSED(args);
    if (scriptName.contains("checkgpu")) {
        combo_wr_device->clear();
        // Empty means no --device argument, which is what the script already
        // treats as "use the card if there is one". Naming it makes the
        // default visible instead of leaving the box looking unset.
        combo_wr_device->addItem(i18n("Automatic"), QString());
        for (auto &s : jobData) {
            if (s.contains(QLatin1Char('#'))) {
                combo_wr_device->addItem(s.section(QLatin1Char('#'), 1).simplified(), s.section(QLatin1Char('#'), 0, 0).simplified());
            } else {
                combo_wr_device->addItem(s.simplified(), s.simplified());
            }
        }
    }
}

void PluginsSettings::samFinished(const QString &scriptName, const QStringList &args)
{
    qDebug() << "=========================\n\nCONCURRENT JOB FINISHED: " << scriptName << " / " << args << "\n\n================";
    if (scriptName.contains("checkgpu")) {
        if (!WunjoSettings::samDevice().isEmpty()) {
            int ix = combo_sam_device->findData(WunjoSettings::samDevice());
            if (ix > -1) {
                combo_sam_device->setCurrentIndex(ix);
            }
        } else if (combo_sam_device->count() > 0) {
            combo_sam_device->setCurrentIndex(0);
            WunjoSettings::setSamDevice(combo_sam_device->currentData().toString());
        }
    }
}

void PluginsSettings::samMissing(const QStringList &)
{
    modelBox->setEnabled(false);
    combo_sam_device->setPlaceholderText(i18n("Install to detect"));
}

void PluginsSettings::gotSamFeedback(const QString &scriptName, const QStringList args, const QStringList jobData)
{
    Q_UNUSED(args);
    if (scriptName.contains("checkgpu")) {
        combo_sam_device->clear();
        combo_sam_device->addItem(i18n("Automatic"), QString());
        for (auto &s : jobData) {
            if (s.contains(QLatin1Char('#'))) {
                combo_sam_device->addItem(s.section(QLatin1Char('#'), 1).simplified(), s.section(QLatin1Char('#'), 0, 0).simplified());
            } else {
                combo_sam_device->addItem(s.simplified(), s.simplified());
            }
        }
    }
}

void PluginsSettings::samDependenciesChecked()
{
    if (combo_sam_model->count() > 0) {
        Q_EMIT pCore->samConfigUpdated();
    }
}

void PluginsSettings::checkSamEnvironement(bool afterInstall)
{
    AbstractPythonInterface::PythonExec exes = m_samInterface->venvPythonExecs(true);
    if (exes.python.isEmpty() || exes.pip.isEmpty()) {
        // Venv not setup
        modelBox->setEnabled(false);
        // Update env folder size
        m_samInterface->checkVenv(true);
    } else {
        // Venv ready
        modelBox->setEnabled(true);
        // Fill models list
        if (afterInstall) {
            m_samInterface->checkVenv(true);
            installSamModelIfEmpty();
        } else {
            reloadSamModels();
        }
        m_samInterface->checkDependencies(false);
        QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
        if (pluginDir.cd(m_samInterface->getVenvPath())) {
            sam_venv_label->setLink(pluginDir.absolutePath());
        }
        if (combo_sam_device->count() == 0) {
            m_samInterface->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
        }
    }
}

void PluginsSettings::doDeleteSamModels()
{
    const QStringList models = m_samInterface->getInstalledModels();
    if (!models.isEmpty()) {
        if (KMessageBox::warningContinueCancel(this, i18n("This will delete the installed models\nYou can reinstall them later.")) != KMessageBox::Continue) {
            return;
        }
        for (auto &m : models) {
            if (!m.isEmpty() && m.endsWith(QLatin1String(".pt"))) {
                QFile::remove(m);
            }
        }
        reloadSamModels();
    }
}

void PluginsSettings::doDeleteSamVenv()
{
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    if (pluginDir.cd(m_samInterface->getVenvPath())) {
        if (KMessageBox::warningContinueCancel(this,
                                               i18n("This will delete the plugin python environment from:<br/><b>%1</b><br/>The environment will be recreated "
                                                    "and modules downloaded whenever you enable the plugin again.",
                                                    pluginDir.absolutePath())) != KMessageBox::Continue) {
            return;
        }
        m_samInterface->deleteVenv();
    }
}

void PluginsSettings::doDeleteWrVenv()
{
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    if (pluginDir.cd(m_sttWhisper->getVenvPath())) {
        if (KMessageBox::warningContinueCancel(this,
                                               i18n("This will delete the plugin python environment from:<br/><b>%1</b><br/>The environment will be recreated "
                                                    "and modules downloaded whenever you enable the plugin again.",
                                                    pluginDir.absolutePath())) != KMessageBox::Continue) {
            return;
        }
        m_sttWhisper->deleteVenv();
    }
}

void PluginsSettings::downloadJobDone(bool success)
{
    WunjoSettings::setEnableSeamless(success);
    if (!success) {
        install_seamless->setEnabled(true);
    }
    install_seamless->setText(i18n("Install multilingual translation"));
}

void PluginsSettings::reloadWhisperModels()
{
    combo_wr_model->clear();
    const QStringList models = m_sttWhisper->getInstalledModels();
    if (models.isEmpty()) {
        // Show install button
        noModelMessage->setText(i18n("Install a speech model - we recommend <b>%1</b>", m_recommendedModel));
        noModelMessage->addAction(m_downloadModelAction);
        noModelMessage->show();
        return;
    }
    noModelMessage->hide();
    for (auto &m : models) {
        if (m.isEmpty()) {
            continue;
        }
        QString modelName = m;
        modelName[0] = m.at(0).toUpper();
        combo_wr_model->addItem(modelName, m);
        qDebug() << ":::: LOADED MODEL: " << modelName << " = " << m;
    }
    int ix = combo_wr_model->findData(WunjoSettings::whisperModel());
    qDebug() << "LOOKING FOR MODEL: " << WunjoSettings::whisperModel() << " / IX: " << ix;

    if (ix == -1) {
        ix = 0;
    }
    combo_wr_model->setCurrentIndex(ix);
    checkWhisperFolderSize();
    Q_EMIT pCore->speechModelUpdate(SpeechToTextEngine::EngineWhisper, models);
}

void PluginsSettings::downloadSamModel(const QString &url)
{
    if (!url.startsWith(QLatin1String("https"))) {
        // TODO: error message
        return;
    }
    const QUrl srcUrl(url);
    QDir modelFolder(m_samInterface->modelFolder());
    // Not KIO: its http worker lives in kio-extras, which the Windows and macOS
    // builds do not ship, and a weight of this size needs a transfer that
    // survives a broken line rather than starting over.
    auto *getJob = new FileDownloadJob(srcUrl, modelFolder.absoluteFilePath(srcUrl.fileName()), this);
    connect(getJob, &KJob::result, this, &PluginsSettings::reloadSamModels);
    KIO::getJobTracker()->registerJob(getJob);
    getJob->start();
}

void PluginsSettings::downloadSamModels()
{
    KConfig conf(QStringLiteral("sammodelsinfo.rc"), KConfig::CascadeConfig, QStandardPaths::AppDataLocation);
    KConfigGroup group(&conf, QStringLiteral("models"));
    QStringList modelUrls = group.entryMap().values();
    const QStringList models = m_samInterface->getInstalledModels();
    QStringList installedModels;
    for (auto &m : models) {
        installedModels << QFileInfo(m).fileName();
    }
    QDialog d(this);
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    auto *l = new QVBoxLayout;
    d.setLayout(l);
    QList<QCheckBox *> buttons;
    l->addWidget(new QLabel(i18n("Select model to download"), &d));
    for (auto &u : modelUrls) {
        QString modelName = QUrl(u).fileName();
        if (installedModels.contains(modelName)) {
            // Already installed, drop
            continue;
        }
        modelName[0] = modelName.at(0).toUpper();
        auto *cb = new QCheckBox(modelName, this);
        cb->setProperty("url", u);
        buttons << cb;
        l->addWidget(cb);
    }
    if (buttons.isEmpty()) {
        auto km = new KMessageWidget(&d);
        km->setMessageType(KMessageWidget::Information);
        km->setText(i18n("All models are already installed"));
        km->setCloseButtonVisible(false);
        l->addWidget(km);
    }
    l->addStretch(10);
    l->addWidget(buttonBox);
    d.connect(buttonBox, &QDialogButtonBox::rejected, &d, &QDialog::reject);
    d.connect(buttonBox, &QDialogButtonBox::accepted, &d, &QDialog::accept);
    if (d.exec() != QDialog::Accepted) {
        return;
    }
    int downloadCount = 0;
    for (auto &b : buttons) {
        if (b->isChecked()) {
            const QString url = b->property("url").toString();
            downloadSamModel(url);
            downloadCount++;
        }
    }
    if (downloadCount > 0) {
        downloadSamButton->setText(i18n("Downloading…"));
        downloadSamButton->setEnabled(false);
    }
}

void PluginsSettings::installSamModelIfEmpty()
{
    const QStringList models = m_samInterface->getInstalledModels();
    if (models.isEmpty()) {
        // Download tiny model by default if none installed
        combo_sam_model->setPlaceholderText(i18n("Downloading smallest model…"));
        KConfig conf(QStringLiteral("sammodelsinfo.rc"), KConfig::CascadeConfig, QStandardPaths::AppDataLocation);
        KConfigGroup group(&conf, QStringLiteral("models"));
        QMap<QString, QString> values = group.entryMap();
        QMapIterator<QString, QString> i(values);
        while (i.hasNext()) {
            i.next();
            if (i.value().contains(QLatin1String("tiny"))) {
                downloadSamModel(i.value());
            }
        }
        return;
    }
    reloadSamModels();
}

void PluginsSettings::reloadSamModels()
{
    if (!downloadSamButton->isEnabled()) {
        downloadSamButton->setText(i18n("Download models"));
        downloadSamButton->setEnabled(true);
    }
    combo_sam_model->clear();
    const QStringList models = m_samInterface->getInstalledModels();
    for (auto &m : models) {
        if (m.isEmpty()) {
            continue;
        }
        QString modelName = QFileInfo(m).completeBaseName();
        modelName[0] = modelName.at(0).toUpper();
        combo_sam_model->addItem(modelName, m);
    }
    int ix = combo_sam_model->findData(WunjoSettings::samModelFile());
    qDebug() << "LOOKING FOR MODEL: " << WunjoSettings::samModelFile() << " / IX: " << ix;

    if (ix == -1) {
        ix = 0;
    }
    combo_sam_model->setCurrentIndex(ix);
    checkSamFolderSize();
    Q_EMIT pCore->samConfigUpdated();
}

void PluginsSettings::checkSamFolderSize()
{
    const QString folder = m_samInterface->modelFolder();
    QDir modelsFolder(folder);
    if (folder.isEmpty() || !modelsFolder.exists()) {
        sam_folder_label->setVisible(false);
    } else {
        const QString path = modelsFolder.absolutePath();
        sam_folder_label->setLink(path);
        sam_folder_label->setVisible(true);

        QFuture<KIO::filesize_t> future = QtConcurrent::run(&MainWindow::fetchFolderSize, pCore->window(), path);
        QFutureWatcher<KIO::filesize_t> *watcher = new QFutureWatcher<KIO::filesize_t>(this);
        watcher->setFuture(future);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            KIO::filesize_t size = watcher->result();
            sam_model_size->setText(KIO::convertSize(size));
            watcher->deleteLater();
        });
    }
}

void PluginsSettings::checkWhisperFolderSize()
{
    const QString folder = m_sttWhisper->modelFolder();
    QDir modelsFolder(folder);
    if (folder.isEmpty() || !modelsFolder.exists()) {
        whisper_folder_label->setVisible(false);
        downloadButton->setText(i18n("Install a model"));
    } else {
        const QString path = modelsFolder.absolutePath();
        whisper_folder_label->setLink(path);
        whisper_folder_label->setVisible(true);

        QFuture<KIO::filesize_t> future = QtConcurrent::run(&MainWindow::fetchFolderSize, pCore->window(), path);
        QFutureWatcher<KIO::filesize_t> *watcher = new QFutureWatcher<KIO::filesize_t>(this);
        watcher->setFuture(future);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            KIO::filesize_t size = watcher->result();
            whisper_model_size->setText(KIO::convertSize(size));
            if (size == 0) {
                downloadButton->setText(i18n("Install a model"));
            } else {
                downloadButton->setText(i18n("Manage models"));
            }
            watcher->deleteLater();
        });
    }
    const QString folder2 = m_sttWhisper->modelFolder(false);
    QDir seamlessFolder(folder2);
    if (folder2.isEmpty() || !seamlessFolder.exists()) {
        seamless_folder_label->setVisible(false);
    } else {
        seamless_folder_label->setLink(seamlessFolder.absolutePath());
        seamless_folder_label->setVisible(true);
        QFuture<KIO::filesize_t> future = QtConcurrent::run(&MainWindow::fetchFolderSize, pCore->window(), seamlessFolder.absolutePath());
        QFutureWatcher<KIO::filesize_t> *watcher = new QFutureWatcher<KIO::filesize_t>(this);
        watcher->setFuture(future);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            KIO::filesize_t size = watcher->result();
            seamless_folder_label->setText(KIO::convertSize(size));
            watcher->deleteLater();
        });
    }
}

void PluginsSettings::showSpeechLog(const QString &jobData)
{
    script_log->show();
    script_log->appendPlainText(jobData);
    script_log->ensureCursorVisible();
}

void PluginsSettings::showSamLog(const QString &jobData)
{
    script_sam_log->show();
    script_sam_log->appendPlainText(jobData);
    script_sam_log->ensureCursorVisible();
}

void PluginsSettings::slotCheckSttConfig()
{
    check_config->setEnabled(false);
    qApp->processEvents();
    if (engine_vosk->isChecked()) {
        m_sttVosk->checkDependencies(true);
    } else {
        m_sttWhisper->checkDependencies(true);
    }
    // Leave button disabled for 3 seconds so that the user doesn't trigger it again while it is processing
    QTimer::singleShot(3000, this, [&]() { check_config->setEnabled(true); });
}

void PluginsSettings::doShowSpeechMessage(const QString &message, int messageType)
{
    speech_info->setMessageType(static_cast<KMessageWidget::MessageType>(messageType));
    speech_info->setText(message);
    speech_info->animatedShow();
}

void PluginsSettings::getDictionary(const QUrl &sourceUrl)
{
    QUrl url = KUrlRequesterDialog::getUrl(sourceUrl, this, i18n("Enter url for the new dictionary"));
    if (url.isEmpty()) {
        return;
    }
    if (!url.isLocalFile()) {
        auto *copyjob = new FileDownloadJob(url, QDir::temp().absoluteFilePath(url.fileName()), this);
        doShowSpeechMessage(i18n("Downloading model…"), KMessageWidget::Information);
        connect(copyjob, &KJob::result, this, &PluginsSettings::downloadModelFinished);
        KIO::getJobTracker()->registerJob(copyjob);
        copyjob->start();
    } else {
        processArchive(url.toLocalFile());
    }
}

void PluginsSettings::removeDictionary()
{
    if (!WunjoSettings::vosk_folder_path().isEmpty()) {
        doShowSpeechMessage(i18n("We do not allow deleting custom folder models, please do it manually."), KMessageWidget::Warning);
        return;
    }
    QString modelDirectory = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir dir(modelDirectory);
    if (!dir.cd(QStringLiteral("speechmodels"))) {
        doShowSpeechMessage(i18n("Cannot access dictionary folder."), KMessageWidget::Warning);
        return;
    }

    if (!m_speechListWidget->currentItem()) {
        return;
    }
    QString currentModel = m_speechListWidget->currentItem()->text();
    if (!currentModel.isEmpty() && dir.cd(currentModel)) {
        if (KMessageBox::questionTwoActions(this, i18n("Delete folder:\n%1", dir.absolutePath()), {}, KStandardGuiItem::del(), KStandardGuiItem::cancel()) ==
            KMessageBox::PrimaryAction) {
            // Make sure we don't accidentally delete a folder that is not ours
            if (dir.absolutePath().contains(QLatin1String("speechmodels"))) {
                dir.removeRecursively();
                slotParseVoskDictionaries();
            }
        }
    }
}

void PluginsSettings::downloadModelFinished(KJob *job)
{
    if (job->error() == 0 || job->error() == 112) {
        auto *jb = qobject_cast<FileDownloadJob *>(job);
        if (jb) {
            processArchive(jb->destination());
        } else {
            speech_info->setMessageType(KMessageWidget::Warning);
            speech_info->setText(i18n("Download error"));
        }
    } else {
        speech_info->setMessageType(KMessageWidget::Warning);
        speech_info->setText(i18n("Download error %1", job->errorString()));
    }
}

void PluginsSettings::processArchive(const QString &archiveFile)
{
    QMimeDatabase db;
    QMimeType type = db.mimeTypeForFile(archiveFile);
    std::unique_ptr<KArchive> archive;
    if (type.inherits(QStringLiteral("application/zip"))) {
        archive = std::make_unique<KZip>(archiveFile);
    } else {
        archive = std::make_unique<KTar>(archiveFile);
    }
    QString modelDirectory = WunjoSettings::vosk_folder_path();
    QDir dir;
    if (modelDirectory.isEmpty()) {
        modelDirectory = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        dir = QDir(modelDirectory);
        dir.mkdir(QStringLiteral("speechmodels"));
        if (!dir.cd(QStringLiteral("speechmodels"))) {
            doShowSpeechMessage(i18n("Cannot access dictionary folder."), KMessageWidget::Warning);
            return;
        }
    } else {
        dir = QDir(modelDirectory);
    }
    if (archive->open(QIODevice::ReadOnly)) {
        doShowSpeechMessage(i18n("Extracting archive…"), KMessageWidget::Information);
        const KArchiveDirectory *archiveDir = archive->directory();
        if (!archiveDir->copyTo(dir.absolutePath())) {
            qDebug() << "=== Error extracting archive!!";
        } else {
            QFile::remove(archiveFile);
            Q_EMIT parseDictionaries();
            doShowSpeechMessage(i18n("New dictionary installed."), KMessageWidget::Positive);
        }
    } else {
        // Test if it is a folder
        QDir testDir(archiveFile);
        if (testDir.exists()) {
        }
        qDebug() << "=== CANNOT OPEN ARCHIVE!!";
    }
}

void PluginsSettings::slotParseVoskDictionaries()
{
    m_speechListWidget->clear();
    QStringList final = m_sttVosk->getInstalledModels();
    m_speechListWidget->addItems(final);
    QString voskModelFolder;
    if (!WunjoSettings::vosk_folder_path().isEmpty()) {
        custom_vosk_folder->setChecked(true);
        vosk_folder->setUrl(QUrl::fromLocalFile(WunjoSettings::vosk_folder_path()));
        voskModelFolder = WunjoSettings::vosk_folder_path();
    } else {
        voskModelFolder = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("speechmodels"), QStandardPaths::LocateDirectory);
    }
    if (!final.isEmpty() && m_sttVosk->missingDependencies().isEmpty()) {
        speech_info->animatedHide();
    } else if (final.isEmpty()) {
        doShowSpeechMessage(i18n("Please add a speech model."), KMessageWidget::Information);
    }
    if (!voskModelFolder.isEmpty()) {
        modelV_folder_label->setLink(voskModelFolder);
        modelV_folder_label->setVisible(true);
        QFuture<KIO::filesize_t> future = QtConcurrent::run(&MainWindow::fetchFolderSize, pCore->window(), voskModelFolder);
        QFutureWatcher<KIO::filesize_t> *watcher = new QFutureWatcher<KIO::filesize_t>(this);
        watcher->setFuture(future);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            KIO::filesize_t size = watcher->result();
            modelV_size->setText(KIO::convertSize(size));
            watcher->deleteLater();
        });
    } else {
        modelV_folder_label->setVisible(false);
    }
    Q_EMIT pCore->speechModelUpdate(SpeechToTextEngine::EngineVosk, final);
}

void PluginsSettings::checkSpeechDependencies()
{
    m_sttWhisper->checkDependenciesConcurrently();
    m_sttVosk->checkDependenciesConcurrently();
}

void PluginsSettings::setActiveTab(int index)
{
    // index is the historical option (0 = Speech To Text, 1 = Object Detection);
    // the Load Plugins tab now sits before them at position 0.
    tabWidget->setCurrentIndex(index + 1);
}

void PluginsSettings::showPluginTab(const QString &pluginId)
{
    const QString name = PluginManager::instance().plugin(pluginId).name();
    if (name.isEmpty()) {
        return;
    }
    for (QWidget *page : std::as_const(m_pluginTabs)) {
        const int index = tabWidget->indexOf(page);
        if (index >= 0 && tabWidget->tabText(index) == name) {
            tabWidget->setCurrentIndex(index);
            return;
        }
    }
}

void PluginsSettings::applySettings()
{
    // Speech
    switch (speech_stack->currentIndex()) {
    case 1:
        if (WunjoSettings::speechEngine() != QLatin1String("whisper")) {
            WunjoSettings::setSpeechEngine(QStringLiteral("whisper"));
            Q_EMIT pCore->speechEngineChanged();
        }
        break;
    default:
        if (!WunjoSettings::speechEngine().isEmpty()) {
            WunjoSettings::setSpeechEngine(QString());
            Q_EMIT pCore->speechEngineChanged();
        }
        break;
    }
    if (speech_system_python_path->text() != WunjoSettings::speech_system_python_path()) {
        WunjoSettings::setSpeech_system_python_path(speech_system_python_path->text());
    }
    if (sam_system_python_path->text() != WunjoSettings::sam_system_python_path()) {
        WunjoSettings::setSam_system_python_path(sam_system_python_path->text());
    }

    if (combo_wr_lang->currentData().toString() != WunjoSettings::whisperLanguage()) {
        WunjoSettings::setWhisperLanguage(combo_wr_lang->currentData().toString());
    }
    if (combo_wr_model->currentData().toString() != WunjoSettings::whisperModel()) {
        WunjoSettings::setWhisperModel(combo_wr_model->currentData().toString());
    }
    if (combo_sam_model->currentData().toString() != WunjoSettings::samModelFile()) {
        WunjoSettings::setSamModelFile(combo_sam_model->currentData().toString());
        Q_EMIT pCore->samConfigUpdated();
    }
    if (combo_sam_device->currentData().toString() != WunjoSettings::samDevice()) {
        WunjoSettings::setSamDevice(combo_sam_device->currentData().toString());
    }
    if (combo_wr_device->currentData().toString() != WunjoSettings::whisperDevice()) {
        WunjoSettings::setWhisperDevice(combo_wr_device->currentData().toString());
    }
}

void PluginsSettings::checkCuda(bool isSam)
{
    // What CUDA line this machine can run, asked of the driver — the same answer
    // the plugins use. It used to be read from `nvcc --version`, which is the
    // CUDA *toolkit*: a compiler almost nobody installs and which does not exist
    // inside the sandbox at all, so this dialog always opened saying it could
    // not tell, on machines whose card was working perfectly.
    const double driverCuda = PluginManager::driverCudaVersion();
    QString detectedCuda;
    if (driverCuda >= 12.8) {
        detectedCuda = QStringLiteral("cuda128");
    } else if (driverCuda >= 12.6) {
        detectedCuda = QStringLiteral("cuda126");
    } else if (driverCuda >= 12.4) {
        detectedCuda = QStringLiteral("cuda124");
    } else if (driverCuda >= 11.8) {
        detectedCuda = QStringLiteral("cuda118");
    }
    QDialog d(this);
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Apply | QDialogButtonBox::Cancel);
    connect(buttonBox->button(QDialogButtonBox::Apply), &QPushButton::clicked, &d, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, &d, &QDialog::reject);
    auto *l = new QVBoxLayout;
    d.setLayout(l);
    const QString featureName = isSam ? m_samInterface->featureName() : m_sttWhisper->featureName();
    l->addWidget(new QLabel(i18n("Nvidia GPU support for %1\nSelect the CUDA version to install.", featureName), &d));
    const QStringList versions = {QStringLiteral("11.8"), QStringLiteral("12.4"), QStringLiteral("12.6"), QStringLiteral("12.8")};
    QButtonGroup bg;
    for (auto &v : versions) {
        QRadioButton *button = new QRadioButton(i18n("CUDA %1", v), &d);
        QString versionName = QStringLiteral("cuda%1").arg(v);
        versionName.remove(QLatin1Char('.'));
        button->setObjectName(versionName);
        if (detectedCuda == versionName) {
            button->setChecked(true);
        }
        bg.addButton(button);
        l->addWidget(button);
    }
    KMessageWidget km;
    km.setCloseButtonVisible(false);
    if (!detectedCuda.isEmpty()) {
        km.setText(i18n("Detected version: %1", bg.checkedButton()->text()));
        km.setMessageType(KMessageWidget::Positive);
    } else {
        km.setText(i18n("Cannot determine CUDA version,\nplease select the version available on your system."));
        km.setMessageType(KMessageWidget::Information);
    }
    l->addWidget(&km);
    l->addWidget(buttonBox);

    if (detectedCuda.isEmpty()) {
        buttonBox->button(QDialogButtonBox::Apply)->setEnabled(false);
        connect(&bg, &QButtonGroup::buttonClicked, &d, [&buttonBox]() { buttonBox->button(QDialogButtonBox::Apply)->setEnabled(true); });
    }
    if (d.exec() != QDialog::Accepted) {
        return;
    }
    if (KMessageBox::warningContinueCancel(
            this, i18n("Only use this install option if your GPU\nis not correctly detected or not working with this plugin.")) != KMessageBox::Continue) {
        return;
    }
    // handle installation
    auto checkedB = bg.checkedButton();
    detectedCuda = checkedB->objectName();
    const QString requirementsFile = QStringLiteral("requirements-%1.txt").arg(detectedCuda);
    qDebug() << ":::: READY TO PROCESS REQ FILE: " << requirementsFile;
    if (isSam) {
        m_samInterface->installRequirements(requirementsFile);
    } else {
        m_sttWhisper->installRequirements(requirementsFile);
    }
}
