/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginsettingstab.h"

#include "filedownloader.h"
#include "filedownloadjob.h"
#include "pluginaboutdialog.h"
#include "pluginmanager.h"
#include "pluginpythonenv.h"
#include "pythoninterfaces/abstractpythoninterface.h"

#include <KConfig>
#include <KConfigGroup>
#include <KIO/Global>
#include <KIO/JobTracker>
#include <KJobTrackerInterface>
#include <KLocalizedString>
#include <KMessageWidget>

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QTextBlock>
#include <QTimer>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

PluginSettingsTab::PluginSettingsTab(const PluginManifest &manifest, QWidget *parent)
    : QWidget(parent)
    , m_manifest(manifest)
{
    auto *layout = new QVBoxLayout(this);

    // ---- what the plugin is, behind a button ----
    // The description used to sit on top of the page as a paragraph nobody
    // rereads after the first time. It moved into an About window of its own,
    // shaped like the application's, which also has room for how to use the
    // plugin and who wrote it.
    auto *aboutRow = new QHBoxLayout;
    aboutRow->addStretch();
    auto *aboutButton = new QToolButton(this);
    aboutButton->setIcon(QIcon::fromTheme(QStringLiteral("help-about")));
    aboutButton->setAutoRaise(true);
    aboutButton->setToolTip(i18n("About %1", manifest.name()));
    aboutRow->addWidget(aboutButton);
    layout->addLayout(aboutRow);
    connect(aboutButton, &QToolButton::clicked, this, [this]() { PluginAboutDialog::show(m_manifest, this); });

    // ---- device: which processor the plugin's model runs on, above the
    // environment and shaped like the Object Detection tab — "Device: […]" on
    // the left, the CUDA escape hatch on the right. Only for plugins that
    // install something; an API plugin runs on somebody else's hardware.
    // Only for a plugin that ships CUDA variants: those are the ones with a
    // torch to ask, and asking anywhere else ran checkgpu.py in an environment
    // that has no torch at all — which is what put a red "Error while running
    // python3 script" on the Assistant and Cut Finder pages.
    if (!manifest.requirementsVariants().isEmpty()) {
        auto *deviceRow = new QHBoxLayout;
        deviceRow->addWidget(new QLabel(i18n("Device:"), this));
        m_deviceCombo = new QComboBox(this);
        m_deviceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        // The first entry is the default and means "decide at run time": the
        // plugin takes the card when it is there and free, and the processor
        // otherwise. Naming a device is for overriding that.
        m_deviceCombo->addItem(i18n("Automatic"), QString());
        m_deviceCombo->setPlaceholderText(i18n("Install to detect"));
        deviceRow->addWidget(m_deviceCombo);
        deviceRow->addStretch();
        // The escape hatch for a card the automatic choice failed to use: name
        // the CUDA line by hand and reinstall against it.
        auto *gpuSupport = new QPushButton(i18n("GPU Support"), this);
        gpuSupport->setToolTip(i18n("Reinstall for a specific CUDA version — only if your GPU is not being used"));
        deviceRow->addWidget(gpuSupport);
        connect(gpuSupport, &QPushButton::clicked, this, &PluginSettingsTab::chooseCudaVariant);
        layout->addLayout(deviceRow);
        connect(m_deviceCombo, &QComboBox::currentIndexChanged, this, [this]() {
            KConfig config(QStringLiteral("wunjopluginsrc"), KConfig::SimpleConfig);
            KConfigGroup(&config, m_manifest.id()).writeEntry("device", m_deviceCombo->currentData().toString());
            config.sync();
        });
    }

    // ---- environment: create the plugin's private venv and install its
    // requirements, guided by the same colored banner the built-in Speech /
    // Object Detection tabs use (green = ready, orange = action, red = error).
    auto *envBox = new QGroupBox(i18n("Environment"), this);
    auto *envLayout = new QVBoxLayout(envBox);
    m_env = new PluginPythonEnv(manifest, this);
    m_installBanner = new PythonDependencyMessage(this, m_env);
    envLayout->addWidget(m_installBanner);
    // Installing a heavy stack runs for minutes behind one frozen sentence. The
    // banner itself carries the news instead: how long it has been running and
    // the last thing pip said, so it is visibly alive and on what.
    m_installTimer = new QTimer(this);
    m_installTimer->setInterval(1000);
    connect(m_installTimer, &QTimer::timeout, this, &PluginSettingsTab::updateInstallLine);
    // pip runs in a worker thread — hop back to this one before touching widgets
    connect(m_env, &AbstractPythonInterface::installFeedback, this, &PluginSettingsTab::showInstallFeedback, Qt::QueuedConnection);
    connect(m_env, &AbstractPythonInterface::installStatusChanged, this, [this]() {
        if (m_env->status() == AbstractPythonInterface::InProgress) {
            m_lastInstallLine.clear();
            m_installElapsed.start();
            m_installTimer->start();
            updateInstallLine();
        } else {
            // done or failed: the banner writes its own conclusion, leave it be
            m_installTimer->stop();
            m_installElapsed.invalidate();
        }
    });
    // "Plugin size … [Uninstall plugin] (refresh)" — same shape and left
    // alignment as the built-in Object Detection tab. Uninstall removes only
    // the environment and is disabled until something is installed.
    auto *sizeRow = new QHBoxLayout;
    sizeRow->addWidget(new QLabel(i18n("Plugin size"), envBox));
    m_venvSize = new QLabel(i18n("not installed"), envBox);
    m_venvSize->setStyleSheet(QStringLiteral("color:#696969"));
    sizeRow->addWidget(m_venvSize);
    m_uninstallEnv = new QPushButton(QIcon::fromTheme(QStringLiteral("edit-delete")), i18n("Uninstall plugin"), envBox);
    m_uninstallEnv->setEnabled(false);
    m_uninstallEnv->setToolTip(i18n("Remove the installed environment (keeps the plugin)"));
    sizeRow->addWidget(m_uninstallEnv);
    auto *refresh = new QToolButton(envBox);
    refresh->setIcon(QIcon::fromTheme(QStringLiteral("view-refresh")));
    refresh->setToolTip(i18n("Check configuration"));
    refresh->setAutoRaise(true);
    sizeRow->addWidget(refresh);
    sizeRow->addStretch();
    envLayout->addLayout(sizeRow);
    // The full pip transcript, exactly like the built-in Speech To Text and
    // Object Detection tabs keep one: the banner says how long it has been
    // running, this says what it has actually installed — and what it choked on
    // when it fails, which is otherwise only visible in the terminal.
    m_installLog = new QPlainTextEdit(envBox);
    m_installLog->setReadOnly(true);
    m_installLog->setUndoRedoEnabled(false);
    m_installLog->setFrameShape(QFrame::NoFrame);
    m_installLog->setCenterOnScroll(true);
    m_installLog->setLineWrapMode(QPlainTextEdit::NoWrap);
    // gigabytes of wheels print thousands of lines; keep the tail, not all of it
    m_installLog->setMaximumBlockCount(2000);
    m_installLog->setMinimumHeight(120);
    m_installLog->setMaximumHeight(220);
    m_installLog->hide();
    envLayout->addWidget(m_installLog);
    layout->addWidget(envBox);
    connect(
        m_env, &AbstractPythonInterface::scriptStarted, this,
        [this]() {
            m_installLog->clear();
            m_logEndsWithProgress = false;
        },
        Qt::QueuedConnection);
    // pip's own last words. The banner shows them too, one elided line at a
    // time; here they can be read to the end and copied into a bug report.
    connect(
        m_env, &AbstractPythonInterface::setupError, this, [this](const QString &message) { appendLogRow(message, false); }, Qt::QueuedConnection);
    connect(m_env, &AbstractPythonInterface::gotPythonSize, this, [this](const QString &label) {
        m_venvSize->setText(label.isEmpty() ? i18n("not installed") : label);
        m_uninstallEnv->setEnabled(!label.isEmpty());
    });
    connect(m_env, &AbstractPythonInterface::scriptFinished, this,
            [this]() { QMetaObject::invokeMethod(m_installBanner, "checkAfterInstall", Qt::QueuedConnection); });
    connect(refresh, &QToolButton::clicked, this, [this]() {
        m_env->checkVenv(true);
        QMetaObject::invokeMethod(m_installBanner, "checkAfterInstall", Qt::QueuedConnection);
    });
    connect(m_uninstallEnv, &QPushButton::clicked, this, &PluginSettingsTab::uninstallEnvironment);
    // What torch can actually see from inside this environment — the same
    // question the built-in tabs ask, answered by the same script. It needs the
    // environment to exist, so it is asked once it is known to be there and
    // again after an install has built one.
    connect(m_env, &AbstractPythonInterface::scriptFeedback, this, &PluginSettingsTab::gotDeviceList);
    connect(m_env, &AbstractPythonInterface::dependenciesAvailable, this, [this]() {
        // Only where there is a device list to fill: checkgpu.py imports torch,
        // and a plugin without it answers with a red error banner instead.
        if (m_deviceCombo != nullptr) {
            m_env->runConcurrentScript(QStringLiteral("checkgpu.py"), {});
        }
    });
    m_env->checkVenv(true);
    // An existing venv folder is not a working plugin: an install that was
    // aborted leaves python and pip behind with none of the packages, and the
    // size alone tells nobody that. Ask what is really importable — off this
    // thread, importing a heavy stack takes seconds — so an unfinished install
    // comes back as "install to use" instead of quietly looking ready.
    if (manifest.hasDependencies()) {
        m_env->checkDependenciesConcurrently();
    }

    // ---- models ----
    // A plugin that does several things declares which part each weight belongs
    // to; the page then reads as sections instead of one long heap of files.
    const QList<PluginModel> models = PluginManager::applicableModels(manifest);
    if (!models.isEmpty()) {
        QStringList groups;
        for (const PluginModel &model : models) {
            if (!groups.contains(model.group)) {
                groups << model.group;
            }
        }
        for (const QString &group : std::as_const(groups)) {
            auto *modelsBox = new QGroupBox(group.isEmpty() ? i18n("Models") : i18n("Models — %1", group), this);
            auto *modelsLayout = new QVBoxLayout(modelsBox);
            for (int i = 0; i < models.size(); ++i) {
                if (models.at(i).group != group) {
                    continue;
                }
                auto *row = new QHBoxLayout;
                const QString sizeText = models.at(i).sizeMb > 0 ? QStringLiteral(" (%1 MB)").arg(models.at(i).sizeMb) : QString();
                row->addWidget(new QLabel(models.at(i).name + sizeText, modelsBox));
                auto *status = new QLabel(modelsBox);
                status->setStyleSheet(QStringLiteral("color:#696969"));
                // The status takes the free space instead of a stretch, and its
                // own width counts for nothing: whatever it is given to say, the
                // row cannot grow past the dialog and push the button off the
                // screen. Long sentences go to @ref m_modelsMessage anyway.
                status->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
                status->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
                auto *button = new QPushButton(QIcon::fromTheme(QStringLiteral("download")), i18n("Download"), modelsBox);
                row->addWidget(status, 1);
                row->addWidget(button);
                modelsLayout->addLayout(row);
                while (m_modelRows.size() <= i) {
                    m_modelRows.append(ModelRow());
                }
                m_modelRows[i] = {status, button, nullptr};
                connect(button, &QPushButton::clicked, this, [this, i]() { downloadModel(i); });
            }
            layout->addWidget(modelsBox);
        }
        // Where a sentence goes: why a download will not start, what an unpack
        // choked on. It spans the page and wraps, so it can be as long as it
        // needs to be without touching the rows above it.
        m_modelsMessage = new KMessageWidget(this);
        m_modelsMessage->setWordWrap(true);
        m_modelsMessage->setCloseButtonVisible(false);
        m_modelsMessage->hide();
        layout->addWidget(m_modelsMessage);
        // Models folder + delete-all, left-aligned like the Object Detection tab.
        auto *folderRow = new QHBoxLayout;
        folderRow->addWidget(new QLabel(i18n("Models folder"), this));
        auto *openFolder = new QPushButton(QIcon::fromTheme(QStringLiteral("folder")), i18n("Open"), this);
        auto *deleteModels = new QPushButton(QIcon::fromTheme(QStringLiteral("edit-delete")), i18n("Delete all models"), this);
        folderRow->addWidget(openFolder);
        folderRow->addWidget(deleteModels);
        folderRow->addStretch();
        layout->addLayout(folderRow);
        connect(openFolder, &QPushButton::clicked, this, [this]() {
            QDesktopServices::openUrl(QUrl::fromLocalFile(PluginManager::instance().modelsDir(m_manifest.id())));
        });
        connect(deleteModels, &QPushButton::clicked, this, &PluginSettingsTab::deleteAllModels);
        refreshModels();
    }

    // ---- API key ----
    if (manifest.kind() == QLatin1String("api") && !manifest.providerName().isEmpty()) {
        auto *keyBox = new QGroupBox(i18n("API key"), this);
        auto *form = new QFormLayout(keyBox);
        m_keyEdit = new QLineEdit(keyBox);
        m_keyEdit->setEchoMode(QLineEdit::Password);
        m_keyEdit->setText(PluginManager::instance().apiKey(manifest.providerName()));
        m_keyEdit->setPlaceholderText(i18n("paste your %1 key", manifest.providerName()));
        form->addRow(i18n("%1 key:", manifest.providerName()), m_keyEdit);
        auto *actions = new QWidget(keyBox);
        auto *actionsLayout = new QHBoxLayout(actions);
        actionsLayout->setContentsMargins(0, 0, 0, 0);
        if (!manifest.providerSignupUrl().isEmpty()) {
            auto *link = new QLabel(QStringLiteral("<a href=\"%1\">%2</a>").arg(manifest.providerSignupUrl(), i18n("Get a key")), actions);
            link->setOpenExternalLinks(true);
            actionsLayout->addWidget(link);
        }
        actionsLayout->addStretch();
        auto *save = new QPushButton(i18n("Save key"), actions);
        actionsLayout->addWidget(save);
        form->addRow(actions);
        connect(save, &QPushButton::clicked, this, &PluginSettingsTab::saveKey);
        layout->addWidget(keyBox);
    }

    // ---- parameters ----
    const QList<PluginParam> params = manifest.params();
    if (!params.isEmpty()) {
        KConfig config(QStringLiteral("wunjopluginsrc"), KConfig::SimpleConfig);
        KConfigGroup group(&config, manifest.id());
        const QString pluginId = manifest.id();
        QStringList sections;
        for (const PluginParam &param : params) {
            if (!sections.contains(param.group)) {
                sections << param.group;
            }
        }
        QHash<QString, QFormLayout *> forms;
        for (const QString &section : std::as_const(sections)) {
            auto *box = new QGroupBox(section.isEmpty() ? i18n("Parameters") : section, this);
            forms.insert(section, new QFormLayout(box));
            layout->addWidget(box);
        }
        auto save = [pluginId](const QString &key, const QVariant &value) {
            KConfig config(QStringLiteral("wunjopluginsrc"), KConfig::SimpleConfig);
            KConfigGroup(&config, pluginId).writeEntry(key, value);
            config.sync();
        };
        for (const PluginParam &param : params) {
            QFormLayout *form = forms.value(param.group);
            QWidget *paramsBox = form->parentWidget();
            const QString label = param.label.isEmpty() ? param.key : param.label;
            const QVariant stored = group.readEntry(param.key, param.defaultValue);
            if (param.type == QLatin1String("bool")) {
                auto *box = new QCheckBox(paramsBox);
                box->setChecked(stored.toBool());
                connect(box, &QCheckBox::toggled, this, [save, key = param.key](bool on) { save(key, on); });
                form->addRow(label, box);
            } else if (param.type == QLatin1String("number")) {
                auto *spin = new QDoubleSpinBox(paramsBox);
                spin->setRange(param.min, param.max);
                spin->setSingleStep(param.step);
                spin->setValue(stored.toDouble());
                connect(spin, &QDoubleSpinBox::valueChanged, this, [save, key = param.key](double v) { save(key, v); });
                form->addRow(label, spin);
            } else if (param.type == QLatin1String("enum")) {
                auto *combo = new QComboBox(paramsBox);
                combo->addItems(param.options);
                combo->setCurrentText(stored.toString());
                connect(combo, &QComboBox::currentTextChanged, this, [save, key = param.key](const QString &v) { save(key, v); });
                form->addRow(label, combo);
            } else if (param.type == QLatin1String("file")) {
                auto *row = new QWidget(paramsBox);
                auto *h = new QHBoxLayout(row);
                h->setContentsMargins(0, 0, 0, 0);
                auto *edit = new QLineEdit(stored.toString(), row);
                auto *browse = new QPushButton(i18n("Browse…"), row);
                h->addWidget(edit);
                h->addWidget(browse);
                connect(edit, &QLineEdit::textChanged, this, [save, key = param.key](const QString &v) { save(key, v); });
                connect(browse, &QPushButton::clicked, this, [edit, filter = param.filter, this]() {
                    const QString path = QFileDialog::getOpenFileName(this, i18n("Select file"), QString(), filter);
                    if (!path.isEmpty()) {
                        edit->setText(path);
                    }
                });
                form->addRow(label, row);
            } else {
                auto *edit = new QLineEdit(stored.toString(), paramsBox);
                connect(edit, &QLineEdit::textChanged, this, [save, key = param.key](const QString &v) { save(key, v); });
                form->addRow(label, edit);
            }
        }
    }

    layout->addStretch();
}

void PluginSettingsTab::showInstallFeedback(const QString &feedback)
{
    appendInstallLog(feedback);
    // pip draws its progress with carriage returns, so one chunk can carry
    // several lines — only the last one still describes the present.
    const QStringList lines = feedback.split(QRegularExpression(QStringLiteral("[\r\n]")), Qt::SkipEmptyParts);
    if (lines.isEmpty()) {
        return;
    }
    m_lastInstallLine = lines.constLast().simplified();
    updateInstallLine();
}

void PluginSettingsTab::appendInstallLog(const QString &chunk)
{
    // What arrives here is not a line: installFeedback simplifies whatever pip
    // wrote since the last read, so one chunk is a wall in which a dozen redraws
    // of the same download bar sit between the two package lines that matter.
    // Take it apart the way a terminal does — the bar overwrites itself, the
    // rest scrolls — or the log is unreadable exactly when it is needed.
    // bars and spinners: pip draws with box-drawing glyphs, uv adds braille
    static const QRegularExpression barGlyphs(QStringLiteral("[\\x{2500}-\\x{257F}\\x{2588}\\x{2591}\\x{2800}-\\x{28FF}]+"));
    // "706.8/706.8 MB 17.7 MB/s eta 0:00:00" (pip) and "123.4MiB/766.3MiB" (uv)
    static const QRegularExpression reading(QStringLiteral("[\\d.]+ ?[kKMGT]?i?B/[\\d.]+ ?[kKMGT]?i?B(?: ?[\\d.?]+ ?[kKMGT]?i?B/s)?(?: eta)?(?: [\\d:?]+)?"));
    // neither prints a separator of its own once the newlines are gone; their
    // verbs are the only place a new step reliably begins
    static const QRegularExpression stepStart(
        QStringLiteral("\\s+(?=Collecting |Downloading |Using cached |Requirement already satisfied|Installing collected packages|Successfully installed|"
                       "Successfully built|Preparing metadata|Building wheel|Created wheel|Stored in directory|Attempting uninstall|Found existing|"
                       "Uninstalling |Resolved \\d|Prepared \\d|Installed \\d|Audited \\d|Built |warning: |error: |WARNING: |ERROR: )"));

    QString text = chunk;
    text.remove(barGlyphs);
    // of a run of readings only the last is still true; it becomes the live row
    QString progress;
    QRegularExpressionMatchIterator it = reading.globalMatch(text);
    while (it.hasNext()) {
        progress = it.next().captured().simplified();
    }
    text.remove(reading);
    text = text.simplified();
    if (!text.isEmpty()) {
        appendLogRow(text.replace(stepStart, QStringLiteral("\n")), false);
    }
    if (!progress.isEmpty()) {
        appendLogRow(progress, true);
    }
}

void PluginSettingsTab::appendLogRow(const QString &text, bool isProgress)
{
    if (!m_installLog->isVisible()) {
        m_installLog->show();
    }
    if (m_logEndsWithProgress) {
        // overwrite the stale reading in place — appending would stack hundreds
        // of near identical rows and bury the package lines between them
        QTextCursor cursor(m_installLog->document()->lastBlock());
        cursor.movePosition(QTextCursor::EndOfBlock);
        cursor.movePosition(QTextCursor::StartOfBlock, QTextCursor::KeepAnchor);
        cursor.insertText(text);
    } else {
        m_installLog->appendPlainText(text);
    }
    m_logEndsWithProgress = isProgress;
    m_installLog->verticalScrollBar()->setValue(m_installLog->verticalScrollBar()->maximum());
}

void PluginSettingsTab::updateInstallLine()
{
    if (!m_installElapsed.isValid()) {
        return;
    }
    const qint64 seconds = m_installElapsed.elapsed() / 1000;
    const QString elapsed = QStringLiteral("%1:%2").arg(seconds / 60).arg(seconds % 60, 2, 10, QLatin1Char('0'));
    QString text = m_lastInstallLine.isEmpty() ? i18n("Installing… %1", elapsed) : i18n("Installing… %1 — %2", elapsed, m_lastInstallLine);
    // A pip line is long enough to wrap the banner into three rows and make it
    // jump on every update, so it is cut to what one row can hold. The icon and
    // the two actions take their share of the width.
    const int available = qMax(200, m_installBanner->width() - 260);
    text = m_installBanner->fontMetrics().elidedText(text, Qt::ElideRight, available);
    m_installBanner->setText(text);
    m_installBanner->setToolTip(m_lastInstallLine);
}

void PluginSettingsTab::deleteAllModels()
{
    // Models that live in a folder of their own are models too: listing only the
    // top level would leave gigabytes behind and still report the plugin empty.
    const QString dir = PluginManager::instance().modelsDir(m_manifest.id());
    QDir modelsDir(dir);
    const QStringList files = modelsDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    for (const QString &file : files) {
        modelsDir.remove(file);
    }
    const QStringList folders = modelsDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QString &folder : folders) {
        QDir(modelsDir.absoluteFilePath(folder)).removeRecursively();
    }
    refreshModels();
}

void PluginSettingsTab::saveKey()
{
    PluginManager::instance().setApiKey(m_manifest.providerName(), m_keyEdit->text().trimmed());
    m_keyEdit->setPlaceholderText(i18n("saved"));
}

void PluginSettingsTab::refreshModels()
{
    const QList<PluginModel> models = PluginManager::applicableModels(m_manifest);
    for (int i = 0; i < m_modelRows.size() && i < models.size(); ++i) {
        if (m_modelRows.at(i).status == nullptr || m_modelRows.at(i).download != nullptr) {
            // A row that is downloading writes its own line — several times a
            // second, with the figures this one does not have.
            continue;
        }
        // A file that exists is not a model that works: a download cut short
        // leaves a fragment behind, and the plugin then dies deep inside torch
        // with "checkpoint file is corrupted". Say so, and let it be fetched
        // again instead of greying the button out on a broken file.
        const PluginManager::ModelState state = PluginManager::instance().modelState(m_manifest.id(), models.at(i));
        // What an interrupted download left behind. It is not lost work: the
        // next attempt asks the server for the rest of it, so the row offers to
        // continue and says how far it got rather than starting the count at
        // zero again.
        const qint64 partial =
            state == PluginManager::ModelReady ? 0 : FileDownloader::resumableBytes(PluginManager::instance().downloadTarget(m_manifest.id(), models.at(i)));
        switch (state) {
        case PluginManager::ModelReady:
            m_modelRows.at(i).status->setText(i18n("installed"));
            break;
        case PluginManager::ModelIncomplete:
            m_modelRows.at(i).status->setText(i18n("incomplete — download again"));
            break;
        case PluginManager::ModelMissing:
            m_modelRows.at(i).status->setText(partial > 0 ? i18n("%1 downloaded so far", KIO::convertSize(partial)) : QString());
            break;
        }
        m_modelRows.at(i).button->setEnabled(state != PluginManager::ModelReady);
        m_modelRows.at(i).button->setText(partial > 0 ? i18n("Continue") : i18n("Download"));
    }
}

void PluginSettingsTab::downloadModel(int index)
{
    const QList<PluginModel> models = PluginManager::applicableModels(m_manifest);
    if (index < 0 || index >= models.size() || models.at(index).url.isEmpty()) {
        return;
    }
    // The button is the only control this row has, so it stops what it started.
    // The window that came up with the download can do it too — this is the
    // same kill, from the other end.
    if (m_modelRows.at(index).download != nullptr) {
        m_modelRows.at(index).download->kill(KJob::EmitResult);
        return;
    }
    m_modelsMessage->hide();
    // Refuse before the first byte when the disk cannot hold what is coming:
    // filling the partition takes the user's projects down with it.
    const QString blocker = PluginManager::downloadBlocker(models.at(index).sizeMb * 1024 * 1024);
    if (!blocker.isEmpty()) {
        m_modelsMessage->setMessageType(KMessageWidget::Warning);
        m_modelsMessage->setText(blocker);
        m_modelsMessage->animatedShow();
        return;
    }
    const bool archived = !models.at(index).unpack.isEmpty();
    const QString dest = PluginManager::instance().downloadTarget(m_manifest.id(), models.at(index));
    // A model name may carry a folder — "whisper/tiny.pt", "buffalo_l/det_10g.onnx"
    // — because that is where the engine looks for it. Only the models folder
    // itself is made in advance, so without this the download has nowhere to
    // land and dies on its first write.
    if (!QDir().mkpath(QFileInfo(dest).absolutePath())) {
        m_modelRows.at(index).status->setText(i18n("cannot create the model folder"));
        return;
    }
    // The download shows itself in the application's own progress window —
    // source, destination, size, speed, Pause and Cancel — because that is the
    // window this has always opened and the one the user watches. The row only
    // keeps a short reading for when that window is closed.
    auto *download = new FileDownloadJob(QUrl(models.at(index).url), dest, this);
    m_modelRows[index].download = download;
    m_modelRows.at(index).button->setText(i18n("Cancel"));
    m_modelRows.at(index).status->setText(i18n("downloading…"));
    connect(download, &KJob::percentChanged, this, [this, index](KJob *, unsigned long percent) {
        m_modelRows.at(index).status->setText(i18n("%1%", percent));
    });
    // A dropped line is not the end of the download any more, and the row has
    // to say that — a silent pause is what used to look like a freeze.
    connect(download, &KJob::infoMessage, this, [this, index](KJob *, const QString &message) {
        m_modelRows.at(index).status->setText(i18n("retrying…"));
        m_modelRows.at(index).status->setToolTip(message);
    });
    connect(download, &KJob::result, this, [this, index, dest, archived](KJob *job) {
        m_modelRows[index].download = nullptr;
        // Whatever the outcome, the button offers what the disk allows: a
        // fragment is continued, everything else starts over.
        m_modelRows.at(index).button->setText(FileDownloader::resumableBytes(dest) > 0 ? i18n("Continue") : i18n("Download"));
        m_modelRows.at(index).status->setToolTip(QString());
        if (job->error() != 0) {
            // Whatever came down stays as a fragment for the next attempt. A
            // killed job is the user's own Cancel and needs no explanation.
            if (job->error() != KJob::KilledJobError) {
                m_modelsMessage->setMessageType(KMessageWidget::Error);
                m_modelsMessage->setText(i18n("Could not download %1: %2", QFileInfo(dest).fileName(), job->errorText()));
                m_modelsMessage->animatedShow();
            }
            refreshModels();
            return;
        }
        const QList<PluginModel> models = PluginManager::applicableModels(m_manifest);
        if (index >= models.size()) {
            refreshModels();
            return;
        }
        if (archived) {
            m_modelRows.at(index).status->setText(i18n("unpacking…"));
            QString unpackError;
            const bool unpacked = PluginManager::instance().unpackModel(m_manifest.id(), models.at(index), dest, &unpackError);
            QFile::remove(dest);
            if (!unpacked) {
                m_modelsMessage->setMessageType(KMessageWidget::Error);
                m_modelsMessage->setText(unpackError);
                m_modelsMessage->animatedShow();
                refreshModels();
                return;
            }
            refreshModels();
            return;
        }
        // the checksum is worth reading hundreds of megabytes for exactly once:
        // right after the download that produced them
        if (PluginManager::instance().modelState(m_manifest.id(), models.at(index), true) != PluginManager::ModelReady) {
            QFile::remove(dest);
            m_modelsMessage->setMessageType(KMessageWidget::Error);
            m_modelsMessage->setText(i18n("%1 arrived damaged and was removed — please download it again.", models.at(index).name));
            m_modelsMessage->animatedShow();
        }
        refreshModels();
    });
    // Hand it to the application's job tracker: that is what opens the window.
    KIO::getJobTracker()->registerJob(download);
    download->start();
}

void PluginSettingsTab::gotDeviceList(const QString &script, const QStringList &args, const QStringList &jobData)
{
    Q_UNUSED(args)
    if (m_deviceCombo == nullptr || !script.contains(QLatin1String("checkgpu"))) {
        return;
    }
    // "cuda:0#NVIDIA GeForce RTX 3070" per card. The bare "cpu" line the script
    // also prints is dropped: Automatic already means "the processor unless a
    // card is free", so offering it again only invites the slow path.
    const QString saved = m_deviceCombo->currentData().toString();
    m_deviceCombo->clear();
    m_deviceCombo->addItem(i18n("Automatic"), QString());
    for (const QString &line : jobData) {
        if (line.contains(QLatin1Char('#'))) {
            m_deviceCombo->addItem(line.section(QLatin1Char('#'), 1).simplified(), line.section(QLatin1Char('#'), 0, 0).simplified());
        }
    }
    const int index = m_deviceCombo->findData(saved);
    m_deviceCombo->setCurrentIndex(index > -1 ? index : 0);
}

void PluginSettingsTab::chooseCudaVariant()
{
    const QList<PluginRequirements> variants = m_manifest.requirementsVariants();
    if (variants.isEmpty()) {
        return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(i18n("GPU Support"));
    auto *layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel(i18n("Nvidia GPU support for %1\nSelect the CUDA version to install.", m_manifest.name()), &dialog));

    // What the driver says it can run — the same number the automatic choice is
    // made from, so the preselected entry is the one already installed.
    const double detected = PluginManager::driverCudaVersion();
    QButtonGroup group;
    for (const PluginRequirements &variant : variants) {
        auto *button = new QRadioButton(variant.minDriverCuda > 0 ? i18n("CUDA %1", variant.minDriverCuda) : i18n("Processor only"), &dialog);
        button->setProperty("file", variant.file);
        if (detected >= variant.minDriverCuda && group.checkedButton() == nullptr) {
            button->setChecked(true);
        }
        group.addButton(button);
        layout->addWidget(button);
    }
    auto *note = new KMessageWidget(&dialog);
    note->setCloseButtonVisible(false);
    note->setWordWrap(true);
    if (detected > 0) {
        note->setMessageType(KMessageWidget::Positive);
        note->setText(i18n("Detected driver: CUDA %1", detected));
    } else {
        note->setMessageType(KMessageWidget::Information);
        note->setText(i18n("Cannot determine the CUDA version,\nplease select the one available on your system."));
    }
    layout->addWidget(note);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Apply | QDialogButtonBox::Cancel, &dialog);
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    buttons->button(QDialogButtonBox::Apply)->setEnabled(group.checkedButton() != nullptr);
    connect(&group, &QButtonGroup::buttonClicked, &dialog, [buttons]() { buttons->button(QDialogButtonBox::Apply)->setEnabled(true); });
    layout->addWidget(buttons);

    if (dialog.exec() != QDialog::Accepted || group.checkedButton() == nullptr) {
        return;
    }
    if (QMessageBox::warning(this, i18n("GPU Support"),
                             i18n("Only use this if your GPU is not detected or the plugin does not use it.\n"
                                  "The packages will be reinstalled, which can take several minutes."),
                             QMessageBox::Apply | QMessageBox::Cancel) != QMessageBox::Apply) {
        return;
    }
    // An absolute path: the base implementation takes the file as given, and a
    // plugin's requirements live beside the plugin, not among the app's scripts.
    m_env->installRequirements(m_manifest.rootDir() + QLatin1Char('/') + group.checkedButton()->property("file").toString());
}

void PluginSettingsTab::uninstallEnvironment()
{
    if (!m_env) {
        return;
    }
    const auto answer = QMessageBox::question(this, i18n("Uninstall plugin"),
                                              i18n("Remove the installed environment for '%1'? The plugin stays and can be "
                                                   "reinstalled from here.",
                                                   m_manifest.name()));
    if (answer != QMessageBox::Yes) {
        return;
    }
    m_env->deleteVenv();
    m_env->checkVenv(true);
}
